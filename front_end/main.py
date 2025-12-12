import sys
import threading
import random
import json
from pathlib import Path

import numpy as np
import sounddevice as sd
import requests  # <-- NEW

from PySide6.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QHBoxLayout,
    QSlider,
    QFrame,
    QSpacerItem,
    QSizePolicy,
    QInputDialog,
    QMessageBox,
)
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QColor

import pyqtgraph as pg


SAMPLE_RATE = 44100
CHUNK = 1024  # frames per audio block

# Backend base URL
BACKEND_URL = "http://127.0.0.1:8000"


# ---------------------- COLOR BOX WIDGET ---------------------- #
class ColorBox(QFrame):
    """Clickable color box used to select palette colors."""

    def __init__(self, color: QColor, toggle_callback, parent=None):
        super().__init__(parent)
        self.color = color
        self.toggle_callback = toggle_callback
        self.selected = False

        self.setFixedSize(26, 26)
        self.setCursor(Qt.PointingHandCursor)
        self.update_style()

    def mousePressEvent(self, event):
        self.selected = not self.selected
        self.update_style()
        if self.toggle_callback:
            self.toggle_callback(self)

    def update_style(self):
        border = "#FFFFFF" if self.selected else "#444444"
        self.setStyleSheet(
            f"""
            QFrame {{
                background-color: {self.color.name()};
                border-radius: 4px;
                border: 2px solid {border};
            }}
        """
        )


# ---------------------- SPECTRUM WIDGET ---------------------- #
class SpectrogramWidget(QWidget):
    """
    PSP-style spectrum visual using pyqtgraph.

    Layout (per side, from center outward):
      low-mids → bass → mids → highs
    Both sides are mirrored, and layout is fixed (not chasing loudest peak).

    Controls:
      - amplitude_scale: vertical height of spikes
      - freq_spread: horizontal stretch + band detail
      - color_reactivity: 0..10 (5 = base brightness)
      - palette_colors: list of QColor chosen by user
      - fade_factor: 0.5..0.98 trail fading
      - organic_factor: 0..1 wobbliness of the line

    Extra:
      - Bottom plot shows frequency "buckets":
        each bin’s frequency range (lo..hi in kHz) vs its visual position.
    """

    def __init__(self, parent=None):
        super().__init__(parent)

        self.amplitude_scale = 5.0
        self.freq_spread = 1.0
        self.color_reactivity = 5.0  # 0..10, 5 = base
        self.palette_colors: list[QColor] = []

        # trail history
        self.trail_history: list[np.ndarray] = []
        self.max_history = 40
        self.fade_factor = 0.85  # 0..1, closer to 1 = slower fade

        # organic shape control
        self.organic_factor = 0.3  # 0..1

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        # main visualizer plot
        self.plot_widget = pg.PlotWidget()
        layout.addWidget(self.plot_widget, stretch=4)

        # bin-to-frequency mapping plot
        self.mapping_plot = pg.PlotWidget()
        layout.addWidget(self.mapping_plot, stretch=1)

        # style main plot
        self.plot_widget.setBackground("#050505")
        self.plot_widget.showGrid(x=False, y=False)
        self.plot_widget.setMenuEnabled(False)
        self.plot_widget.setMouseEnabled(x=False, y=False)
        self.plot_widget.hideButtons()
        self.plot_widget.getAxis("bottom").setPen(pg.mkPen("#555555"))
        self.plot_widget.getAxis("left").setPen(pg.mkPen("#555555"))
        self.plot_widget.getAxis("bottom").setTextPen("#888888")
        self.plot_widget.getAxis("left").setTextPen("#888888")

        # style mapping plot
        self.mapping_plot.setBackground("#050505")
        self.mapping_plot.showGrid(x=False, y=True)
        self.mapping_plot.setMenuEnabled(False)
        self.mapping_plot.setMouseEnabled(x=False, y=False)
        self.mapping_plot.hideButtons()
        self.mapping_plot.getAxis("bottom").setPen(pg.mkPen("#555555"))
        self.mapping_plot.getAxis("left").setPen(pg.mkPen("#555555"))
        self.mapping_plot.getAxis("bottom").setTextPen("#888888")
        self.mapping_plot.getAxis("left").setTextPen("#888888")
        self.mapping_plot.getAxis("left").setLabel("Freq (kHz)", color="#AAAAAA")
        self.mapping_plot.getAxis("bottom").setLabel("Visual position", color="#AAAAAA")

        # final rendered resolution (smooth curve)
        self.n_points = 255
        self.base_x = np.linspace(-1.0, 1.0, self.n_points)

    # ----- setters for controls -----
    def set_amplitude_scale(self, value: float):
        self.amplitude_scale = max(0.1, value)
        self.plot_widget.setYRange(0, self.amplitude_scale, padding=0.05)

    def set_freq_spread(self, value: float):
        # 0.2 .. 1.5 range feels nice
        self.freq_spread = max(0.2, min(1.5, value))

    def set_color_reactivity(self, value: float):
        # value: 0..10, 5 = base
        self.color_reactivity = max(0.0, min(10.0, value))

    def set_palette(self, colors: list[QColor]):
        self.palette_colors = colors

    def set_fade_factor(self, factor: float):
        # 0.5 .. 0.98 typically
        self.fade_factor = max(0.3, min(0.99, factor))

    def set_organic_factor(self, factor: float):
        # 0..1
        self.organic_factor = max(0.0, min(1.0, factor))

    def update_mapping_plot(self, x_positions: np.ndarray, y_lo_hz: np.ndarray, y_hi_hz: np.ndarray):
        """
        Draw frequency buckets: each visual bin’s frequency range (lo..hi) as a vertical bar.
        """
        self.mapping_plot.clear()

        if x_positions.size == 0 or y_lo_hz.size == 0 or y_hi_hz.size == 0:
            return

        # Convert to kHz
        y_lo_khz = y_lo_hz / 1000.0
        y_hi_khz = y_hi_hz / 1000.0
        heights = y_hi_khz - y_lo_khz

        n = len(x_positions)
        if n == 0:
            return

        # Width so all bars fill the -1..1 range
        bar_width = 2.0 / n

        # BarGraphItem API: x, height, width, y0, brush, pen
        bars = pg.BarGraphItem(
            x=x_positions,
            height=heights,
            width=bar_width,
            y0=y_lo_khz,
            brush=(80, 160, 255, 70),
            pen=(120, 200, 255, 160),
        )
        self.mapping_plot.addItem(bars)

        nyquist_khz = (SAMPLE_RATE / 2) / 1000.0
        self.mapping_plot.setYRange(0, nyquist_khz, padding=0.05)
        self.mapping_plot.setXRange(-1.0, 1.0, padding=0.02)

    # ----- main shaping + rendering -----
    def update_spectrum(self, magnitudes: np.ndarray):
        if magnitudes.size == 0:
            return

        mags_full = magnitudes.astype(float)

        # --- 0) PRE-PROCESSING: compress dynamic range & slightly boost mids/highs ---
        mags_full = np.log1p(mags_full)  # compress loud peaks

        freqs = np.linspace(0, SAMPLE_RATE / 2, mags_full.size)
        freqs_norm = freqs / (SAMPLE_RATE / 2 + 1e-9)

        # weight curve: bass still strong, but mids/highs more visible
        weights = 0.5 + 0.5 * (freqs_norm ** 0.7)
        mags_full *= weights

        # --- 1) FIXED, PSP-LIKE BINS ---
        min_half = 16
        max_half = 32
        norm_spread = (self.freq_spread - 0.2) / (1.5 - 0.2)
        norm_spread = max(0.0, min(1.0, norm_spread))
        n_half = int(min_half + norm_spread * (max_half - min_half))
        if n_half < 4:
            n_half = 4

        nyquist = SAMPLE_RATE / 2
        t = np.linspace(0.0, 1.0, n_half + 1)
        f_bounds = (t ** 2.0) * nyquist  # length n_half+1

        half_bands = np.zeros(n_half, dtype=float)
        for i in range(n_half):
            f_start = f_bounds[i]
            f_end = f_bounds[i + 1]

            idx_start = int(np.searchsorted(freqs, f_start, side="left"))
            idx_end = int(np.searchsorted(freqs, f_end, side="left"))

            if idx_end <= idx_start:
                idx_end = min(idx_start + 1, mags_full.size)

            segment = mags_full[idx_start:idx_end]
            if segment.size > 0:
                half_bands[i] = np.mean(segment)
            else:
                half_bands[i] = 0.0

        # smoothing across half_bands
        if n_half > 4:
            kernel = np.array([0.2, 0.6, 0.2])
            half_bands = np.convolve(half_bands, kernel, mode="same")

        # region partition along 0..n_half (freq-wise: bass->highs)
        N = n_half
        nbass = max(2, int(0.30 * N))
        nlowmid = max(2, int(0.25 * N))
        nmid = max(2, int(0.25 * N))
        nhigh = N - (nbass + nlowmid + nmid)
        if nhigh < 1:
            nhigh = 1
            if nmid > 2:
                nmid -= 1
            elif nbass > 2:
                nbass -= 1
            else:
                nlowmid = max(2, nlowmid - 1)

        bass_start = 0
        bass_end = bass_start + nbass
        lowmid_start = bass_end
        lowmid_end = lowmid_start + nlowmid
        mid_start = lowmid_end
        mid_end = mid_start + nmid
        high_start = mid_end
        high_end = N

        bass_idx = list(range(bass_start, bass_end))
        lowmid_idx = list(range(lowmid_start, lowmid_end))
        mid_idx = list(range(mid_start, min(mid_end, N)))
        high_idx = list(range(high_start, N))

        # tone down bass
        bass_attenuation = 0.6  # 0.0..1.0, lower = weaker bass
        if bass_idx:
            half_bands[bass_idx] *= bass_attenuation

        # visual order on one side (from center outward):
        # low-mids -> bass -> mids -> highs
        right_order = lowmid_idx + bass_idx + mid_idx + high_idx
        right_half = half_bands[right_order]

        # for frequency buckets, we also need lo/hi for each reordered band
        right_lo = np.array([f_bounds[i] for i in right_order])
        right_hi = np.array([f_bounds[i + 1] for i in right_order])

        # mirror to build full layout
        left_half = right_half[::-1]
        bands_full = np.concatenate([left_half, right_half[1:]])
        length = bands_full.size

        # mirrored lo/hi
        lo_full = np.concatenate([right_lo[::-1], right_lo[1:]])
        hi_full = np.concatenate([right_hi[::-1], right_hi[1:]])

        # --- 2) CENTER ENVELOPE ---
        indices = np.arange(length)
        mid = length // 2
        sigma = length / 3.5

        gauss = np.exp(-0.5 * ((indices - mid) / sigma) ** 2)
        bands_shaped = bands_full * gauss

        # --- 3) NORMALIZE, INTERPOLATE TO SMOOTH CURVE ---
        max_val = float(np.max(bands_shaped))
        if max_val <= 0:
            norm = np.zeros_like(bands_shaped)
        else:
            norm = bands_shaped / max_val

        x_bands = np.linspace(-1.0, 1.0, length)
        y_interp = np.interp(self.base_x, x_bands, norm)

        # --- bucket mapping plot ---
        self.update_mapping_plot(x_bands, lo_full, hi_full)

        # --- 4) ORGANIC NOISE (wiggle) ---
        if self.organic_factor > 0:
            noise = np.random.normal(scale=self.organic_factor * 0.25, size=y_interp.shape)
            kernel = np.array([0.25, 0.5, 0.25])
            noise = np.convolve(noise, kernel, mode="same")
            y_interp = np.clip(y_interp + noise, 0.0, 1.0)

        # apply amplitude
        y_vals = y_interp * self.amplitude_scale

        # --- 5) FREQUENCY SPREAD (horizontal scaling) ---
        x_vals = self.base_x * self.freq_spread

        # --- 6) COLOR LOGIC ---
        if self.palette_colors:
            base_color = random.choice(self.palette_colors)
        else:
            base_color = QColor(0, 255, 200)

        avg_energy = float(np.mean(y_interp))

        raw_len = magnitudes.size
        if raw_len > 1:
            dominant_idx_real = int(np.argmax(magnitudes))
            dominant_norm = dominant_idx_real / (raw_len - 1)
        else:
            dominant_norm = 0.0

        delta = self.color_reactivity - 5.0
        max_light_boost = 0.6
        max_dark_drop = 0.6

        if delta > 0:
            factor_from_slider = 1.0 - (delta / 5.0) * max_dark_drop
        elif delta < 0:
            factor_from_slider = 1.0 + (-delta / 5.0) * max_light_boost
        else:
            factor_from_slider = 1.0

        loudness_factor = 0.7 + 0.3 * avg_energy
        freq_factor = 1.0 - 0.4 * dominant_norm

        brightness = factor_from_slider * loudness_factor * freq_factor
        brightness = max(0.2, min(1.8, brightness))

        base_r = base_color.red()
        base_g = base_color.green()
        base_b = base_color.blue()

        r = int(base_r * brightness)
        g = int(base_g * brightness)
        b = int(base_b * brightness)

        r = max(0, min(255, r))
        g = max(0, min(255, g))
        b = max(0, min(255, b))

        # --- 7) TRAIL HISTORY + FADING ---
        self.trail_history.append(y_vals)
        if len(self.trail_history) > self.max_history:
            self.trail_history.pop(0)

        self.plot_widget.clear()

        # draw oldest first, newest last
        for age, y in enumerate(self.trail_history):
            alpha = self.fade_factor ** (len(self.trail_history) - age - 1)
            alpha = max(0.02, min(1.0, alpha))

            color = (r, g, b, int(255 * alpha))
            pen = pg.mkPen(color=color, width=2)
            self.plot_widget.plot(x_vals, y, pen=pen)


# ---------------------- MAIN WINDOW ---------------------- #
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("IoT Music Visualizer")
        self.setMinimumSize(1200, 600)

        self.audio_stream = None
        self.latest_audio_chunk = None
        self.audio_lock = threading.Lock()

        # ==== ROOT LAYOUT ====
        central = QWidget()
        root_layout = QHBoxLayout(central)
        root_layout.setContentsMargins(16, 16, 16, 16)
        root_layout.setSpacing(16)
        self.setCentralWidget(central)

        # ---------------------------------
        # LEFT SIDE: spectrogram + settings
        # ---------------------------------
        left_side = QVBoxLayout()
        left_side.setSpacing(12)

        # --- Spectrogram area ---
        self.spectrogram = SpectrogramWidget()
        self.spectrogram.setMinimumHeight(360)
        left_side.addWidget(self.spectrogram, stretch=3)

        # --- Settings block (under spectrogram) ---
        settings_frame = QFrame()
        settings_frame.setFrameShape(QFrame.StyledPanel)
        settings_frame.setStyleSheet(
            """
            QFrame {
                background-color: #181818;
                border: 1px solid #333333;
                border-radius: 8px;
            }
            QLabel {
                color: white;
            }
            QSlider::groove:horizontal {
                height: 6px;
                background: #333333;
                border-radius: 3px;
            }
            QSlider::handle:horizontal {
                background: #ffffff;
                width: 12px;
                margin: -4px 0;
                border-radius: 6px;
            }
            """
        )

        settings_layout = QVBoxLayout(settings_frame)
        settings_layout.setContentsMargins(12, 12, 12, 12)
        settings_layout.setSpacing(10)

        settings_title = QLabel("Settings")
        settings_title.setStyleSheet("font-size: 16px; font-weight: bold;")
        settings_layout.addWidget(settings_title)

        # --- Amplitude slider ---
        amp_row = QHBoxLayout()
        amp_label = QLabel("Amplitude:")
        self.amp_value_label = QLabel("5.0")
        self.amp_value_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)

        self.amp_slider = QSlider(Qt.Horizontal)
        self.amp_slider.setMinimum(10)   # 1.0
        self.amp_slider.setMaximum(100)  # 10.0
        self.amp_slider.setValue(50)     # 5.0

        self.amp_slider.valueChanged.connect(self.on_amp_changed)

        amp_row.addWidget(amp_label)
        amp_row.addWidget(self.amp_slider)
        amp_row.addWidget(self.amp_value_label)
        settings_layout.addLayout(amp_row)

        # --- Frequency spread slider ---
        spread_row = QHBoxLayout()
        spread_label = QLabel("Frequency Spread:")
        self.spread_value_label = QLabel("1.0")
        self.spread_value_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)

        self.spread_slider = QSlider(Qt.Horizontal)
        self.spread_slider.setMinimum(20)   # 0.2
        self.spread_slider.setMaximum(150)  # 1.5
        self.spread_slider.setValue(100)    # 1.0

        self.spread_slider.valueChanged.connect(self.on_spread_changed)

        spread_row.addWidget(spread_label)
        spread_row.addWidget(self.spread_slider)
        spread_row.addWidget(self.spread_value_label)
        settings_layout.addLayout(spread_row)

        # --- Color reactivity slider (0..10, 5 = base) ---
        color_row = QHBoxLayout()
        color_label = QLabel("Color Reactivity:")
        self.color_react_value_label = QLabel("5.0")
        self.color_react_value_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)

        self.color_react_slider = QSlider(Qt.Horizontal)
        self.color_react_slider.setMinimum(0)
        self.color_react_slider.setMaximum(100)
        self.color_react_slider.setValue(50)  # 5.0

        self.color_react_slider.valueChanged.connect(self.on_color_react_changed)

        color_row.addWidget(color_label)
        color_row.addWidget(self.color_react_slider)
        color_row.addWidget(self.color_react_value_label)
        settings_layout.addLayout(color_row)

        # --- Trail fade slider ---
        fade_row = QHBoxLayout()
        fade_label = QLabel("Trail Fade:")
        self.fade_value_label = QLabel("0.85")
        self.fade_value_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)

        self.fade_slider = QSlider(Qt.Horizontal)
        self.fade_slider.setMinimum(0)
        self.fade_slider.setMaximum(100)
        self.fade_slider.setValue(70)  # ~0.85

        self.fade_slider.valueChanged.connect(self.on_fade_changed)

        fade_row.addWidget(fade_label)
        fade_row.addWidget(self.fade_slider)
        fade_row.addWidget(self.fade_value_label)
        settings_layout.addLayout(fade_row)

        # --- Organic shape slider ---
        organic_row = QHBoxLayout()
        organic_label = QLabel("Organic Shape:")
        self.organic_value_label = QLabel("0.30")
        self.organic_value_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)

        self.organic_slider = QSlider(Qt.Horizontal)
        self.organic_slider.setMinimum(0)
        self.organic_slider.setMaximum(100)
        self.organic_slider.setValue(30)  # 0.30

        self.organic_slider.valueChanged.connect(self.on_organic_changed)

        organic_row.addWidget(organic_label)
        organic_row.addWidget(self.organic_slider)
        organic_row.addWidget(self.organic_value_label)
        settings_layout.addLayout(organic_row)

        # --- Color palette boxes ---
        palette_title = QLabel("Color Palette:")
        settings_layout.addWidget(palette_title)

        palette_row = QHBoxLayout()

        basic_colors = [
            QColor("#00ffd5"),  # teal
            QColor("#ff4b4b"),  # red
            QColor("#ffa500"),  # orange
            QColor("#ffd700"),  # yellow
            QColor("#4bff4b"),  # green
            QColor("#4b7bff"),  # blue
            QColor("#ee82ee"),  # violet
            QColor("#ffffff"),  # white
        ]

        self.color_boxes: list[ColorBox] = []

        for c in basic_colors:
            box = ColorBox(c, self.on_color_box_toggled)
            palette_row.addWidget(box)
            self.color_boxes.append(box)

        settings_layout.addLayout(palette_row)

        # start with teal & blue selected
        self.color_boxes[0].selected = True
        self.color_boxes[0].update_style()
        self.color_boxes[5].selected = True
        self.color_boxes[5].update_style()
        self.update_palette_from_boxes()

        left_side.addWidget(settings_frame, stretch=1)

        root_layout.addLayout(left_side, stretch=4)

        # ---------------------------------
        # RIGHT SIDE: Preset handler
        # ---------------------------------
        right_side = QVBoxLayout()
        right_side.setSpacing(12)

        preset_title = QLabel("Preset Handler")
        preset_title.setStyleSheet("color: white; font-size: 16px; font-weight: bold;")
        preset_title.setAlignment(Qt.AlignLeft | Qt.AlignTop)

        right_side.addWidget(preset_title)

        right_side.addSpacerItem(
            QSpacerItem(0, 20, QSizePolicy.Minimum, QSizePolicy.Expanding)
        )

        self.save_button = QPushButton("Save")
        self.save_button.setMinimumWidth(140)
        self.save_button.setMinimumHeight(40)

        self.load_button = QPushButton("Load")
        self.load_button.setMinimumWidth(140)
        self.load_button.setMinimumHeight(40)

        button_style = """
            QPushButton {
                background-color: #222222;
                color: white;
                border: 2px solid #888888;
                border-radius: 6px;
                font-size: 14px;
            }
            QPushButton:hover {
                background-color: #333333;
            }
            """
        self.save_button.setStyleSheet(button_style)
        self.load_button.setStyleSheet(button_style)

        right_side.addWidget(self.save_button)
        right_side.addWidget(self.load_button)

        right_side.addSpacerItem(
            QSpacerItem(0, 40, QSizePolicy.Minimum, QSizePolicy.Expanding)
        )

        root_layout.addLayout(right_side, stretch=1)

        self.setStyleSheet(
            """
            QMainWindow {
                background-color: #0c0c0c;
            }
            """
        )

        # wire buttons
        self.save_button.clicked.connect(self.on_save_clicked)
        self.load_button.clicked.connect(self.on_load_clicked)

        # ==== AUDIO + SPECTRUM UPDATE TIMER ====
        self.spectrum_timer = QTimer(self)
        self.spectrum_timer.timeout.connect(self.process_audio_chunk)
        self.spectrum_timer.start(30)  # ~33 FPS

        self.start_audio_stream()

    # ---------- sliders callbacks ----------

    def on_amp_changed(self, value: int):
        amp = value / 10.0  # 10..100 -> 1.0..10.0
        self.amp_value_label.setText(f"{amp:.1f}")
        self.spectrogram.set_amplitude_scale(amp)

    def on_spread_changed(self, value: int):
        spread = value / 100.0  # 20..150 -> 0.2..1.5
        self.spread_value_label.setText(f"{spread:.2f}")
        self.spectrogram.set_freq_spread(spread)

    def on_color_react_changed(self, value: int):
        react = value / 10.0  # 0..100 -> 0..10
        self.color_react_value_label.setText(f"{react:.1f}")
        self.spectrogram.set_color_reactivity(react)

    def on_fade_changed(self, value: int):
        # slider 0..100 -> fade factor 0.5..0.98
        factor = 0.5 + (value / 100.0) * 0.48
        self.fade_value_label.setText(f"{factor:.2f}")
        self.spectrogram.set_fade_factor(factor)

    def on_organic_changed(self, value: int):
        # 0..100 -> 0.0..1.0
        factor = value / 100.0
        self.organic_value_label.setText(f"{factor:.2f}")
        self.spectrogram.set_organic_factor(factor)

    def on_color_box_toggled(self, _box: ColorBox):
        self.update_palette_from_boxes()

    def update_palette_from_boxes(self):
        colors = [box.color for box in self.color_boxes if box.selected]
        self.spectrogram.set_palette(colors)

    # ---------- PRESET HANDLING (BACKEND) ----------

    def _current_preset_dict(self) -> dict:
        """Collect current UI + palette into a dict matching backend schema."""
        amp = self.amp_slider.value() / 10.0
        spread = self.spread_slider.value() / 100.0
        color_react = self.color_react_slider.value() / 10.0
        fade_factor = 0.5 + (self.fade_slider.value() / 100.0) * 0.48
        organic = self.organic_slider.value() / 100.0
        colors = [box.color.name() for box in self.color_boxes if box.selected]

        return {
            "amplitude": amp,
            "spread": spread,
            "color_reactivity": color_react,
            "fade_factor": fade_factor,
            "organic": organic,
            "colors": colors,
        }

    def _apply_preset_dict(self, preset: dict):
        """Apply preset dict back to UI and spectrogram."""
        amp = float(preset.get("amplitude", 5.0))
        spread = float(preset.get("spread", 1.0))
        color_react = float(preset.get("color_reactivity", 5.0))
        fade_factor = float(preset.get("fade_factor", 0.85))
        organic = float(preset.get("organic", 0.3))
        colors = preset.get("colors", [])

        # sliders
        self.amp_slider.setValue(int(max(10, min(100, amp * 10))))
        self.spread_slider.setValue(int(max(20, min(150, spread * 100))))
        self.color_react_slider.setValue(int(max(0, min(100, color_react * 10))))

        fade_slider_val = int(((fade_factor - 0.5) / 0.48) * 100)
        fade_slider_val = max(0, min(100, fade_slider_val))
        self.fade_slider.setValue(fade_slider_val)

        organic_slider_val = int(organic * 100)
        organic_slider_val = max(0, min(100, organic_slider_val))
        self.organic_slider.setValue(organic_slider_val)

        # palette
        for box in self.color_boxes:
            box.selected = (box.color.name() in colors)
            box.update_style()
        self.update_palette_from_boxes()

    def on_save_clicked(self):
        text, ok = QInputDialog.getText(self, "Save Preset", "Preset name:")
        if not ok or not text.strip():
            return
        name = text.strip()

        payload = self._current_preset_dict()
        payload["name"] = name

        try:
            resp = requests.post(f"{BACKEND_URL}/presets", json=payload, timeout=5)
            if resp.status_code == 200:
                QMessageBox.information(self, "Saved", f"Preset '{name}' saved to backend.")
            else:
                QMessageBox.critical(
                    self,
                    "Error",
                    f"Backend error while saving:\n{resp.status_code} {resp.text}",
                )
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to reach backend:\n{e}")

    def on_load_clicked(self):
        # get all presets from backend
        try:
            resp = requests.get(f"{BACKEND_URL}/presets", timeout=5)
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to reach backend:\n{e}")
            return

        if resp.status_code != 200:
            QMessageBox.critical(
                self,
                "Error",
                f"Backend error while listing presets:\n{resp.status_code} {resp.text}",
            )
            return

        presets = resp.json()  # list of PresetRead dicts
        if not presets:
            QMessageBox.information(self, "No presets", "No presets stored in backend yet.")
            return

        names = [p["name"] for p in presets]
        name, ok = QInputDialog.getItem(
            self, "Load Preset", "Choose preset:", names, 0, False
        )
        if not ok or not name:
            return

        # find preset dict by name
        chosen = None
        for p in presets:
            if p["name"] == name:
                chosen = p
                break

        if not chosen:
            QMessageBox.warning(self, "Not found", f"Preset '{name}' not found anymore.")
            return

        # apply directly
        self._apply_preset_dict(chosen)
        QMessageBox.information(self, "Loaded", f"Preset '{name}' loaded from backend.")

    # ---------- AUDIO HANDLING ----------

    def start_audio_stream(self):
        """Start microphone input stream."""

        def callback(indata, frames, time_info, status):
            if status:
                print(status)
            mono = indata[:, 0].copy()
            with self.audio_lock:
                self.latest_audio_chunk = mono

        self.audio_stream = sd.InputStream(
            channels=1,
            samplerate=SAMPLE_RATE,
            blocksize=CHUNK,
            callback=callback,
        )
        self.audio_stream.start()

    def stop_audio_stream(self):
        if self.audio_stream is not None:
            self.audio_stream.stop()
            self.audio_stream.close()
            self.audio_stream = None

    def process_audio_chunk(self):
        with self.audio_lock:
            data = self.latest_audio_chunk
            self.latest_audio_chunk = None

        if data is None:
            return

        if data.size < CHUNK:
            data = np.pad(data, (0, CHUNK - data.size), mode="constant")

        window = np.hanning(CHUNK)
        windowed = data[:CHUNK] * window

        fft = np.fft.rfft(windowed)
        mags = np.abs(fft)

        self.spectrogram.update_spectrum(mags)

    # ---------- MISC ----------

    def closeEvent(self, event):
        self.stop_audio_stream()
        super().closeEvent(event)


def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
