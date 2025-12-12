import json
from typing import List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from sqlalchemy import (
    create_engine,
    Column,
    Integer,
    String,
    Float,
    Text,
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session


# =========================
# DATABASE SETUP (SQLite)
# =========================

DATABASE_URL = "mysql+pymysql://user:password@db:3306/visualizer_db"

engine = create_engine(
    DATABASE_URL, connect_args={"check_same_thread": False}  # needed for SQLite + FastAPI
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()


class PresetModel(Base):
    __tablename__ = "presets"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(100), unique=True, index=True, nullable=False)

    amplitude = Column(Float, nullable=False)
    spread = Column(Float, nullable=False)
    color_reactivity = Column(Float, nullable=False)
    fade_factor = Column(Float, nullable=False)
    organic = Column(Float, nullable=False)

    # list of hex strings as JSON text
    colors_json = Column(Text, nullable=False, default="[]")

    def get_colors(self) -> List[str]:
        try:
            return json.loads(self.colors_json)
        except Exception:
            return []

    def set_colors(self, colors: List[str]):
        self.colors_json = json.dumps(colors)


Base.metadata.create_all(bind=engine)


# =========================
# Pydantic Schemas
# =========================

class PresetBase(BaseModel):
    amplitude: float = Field(..., description="Amplitude scale (1.0–10.0)")
    spread: float = Field(..., description="Frequency spread (0.2–1.5)")
    color_reactivity: float = Field(..., description="Color reactivity (0–10, 5 = base)")
    fade_factor: float = Field(..., description="Trail fade factor (0.5–0.98)")
    organic: float = Field(..., description="Organic shape factor (0–1)")
    colors: List[str] = Field(..., description="List of hex color strings")


class PresetCreate(PresetBase):
    name: str = Field(..., description="Human-readable preset name")


class PresetUpdate(PresetBase):
    name: Optional[str] = Field(None, description="New name (optional)")


class PresetRead(PresetBase):
    id: int
    name: str

    class Config:
        orm_mode = True


# =========================
# FASTAPI APP
# =========================

app = FastAPI(title="IoT Music Visualizer Backend", version="1.0.0")

# Allow local UI or future web UI to talk to this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # you can restrict later
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Dependency to get DB session per request
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# =========================
# HELPERS
# =========================

def model_to_read(preset: PresetModel) -> PresetRead:
    return PresetRead(
        id=preset.id,
        name=preset.name,
        amplitude=preset.amplitude,
        spread=preset.spread,
        color_reactivity=preset.color_reactivity,
        fade_factor=preset.fade_factor,
        organic=preset.organic,
        colors=preset.get_colors(),
    )


# =========================
# ROUTES
# =========================

@app.get("/presets", response_model=List[PresetRead])
def list_presets():
    db: Session = next(get_db())
    presets = db.query(PresetModel).order_by(PresetModel.name.asc()).all()
    return [model_to_read(p) for p in presets]


@app.get("/presets/{preset_id}", response_model=PresetRead)
def get_preset(preset_id: int):
    db: Session = next(get_db())
    preset = db.query(PresetModel).filter(PresetModel.id == preset_id).first()
    if not preset:
        raise HTTPException(status_code=404, detail="Preset not found")
    return model_to_read(preset)


@app.post("/presets", response_model=PresetRead)
def create_preset(preset_in: PresetCreate):
    db: Session = next(get_db())

    existing = db.query(PresetModel).filter(PresetModel.name == preset_in.name).first()
    if existing:
        raise HTTPException(status_code=400, detail="Preset with this name already exists")

    preset = PresetModel(
        name=preset_in.name,
        amplitude=preset_in.amplitude,
        spread=preset_in.spread,
        color_reactivity=preset_in.color_reactivity,
        fade_factor=preset_in.fade_factor,
        organic=preset_in.organic,
    )
    preset.set_colors(preset_in.colors)

    db.add(preset)
    db.commit()
    db.refresh(preset)

    return model_to_read(preset)


@app.put("/presets/{preset_id}", response_model=PresetRead)
def update_preset(preset_id: int, preset_in: PresetUpdate):
    db: Session = next(get_db())
    preset = db.query(PresetModel).filter(PresetModel.id == preset_id).first()
    if not preset:
        raise HTTPException(status_code=404, detail="Preset not found")

    if preset_in.name is not None:
        # check unique name
        other = (
            db.query(PresetModel)
            .filter(PresetModel.name == preset_in.name, PresetModel.id != preset_id)
            .first()
        )
        if other:
            raise HTTPException(status_code=400, detail="Another preset already uses that name")
        preset.name = preset_in.name

    preset.amplitude = preset_in.amplitude
    preset.spread = preset_in.spread
    preset.color_reactivity = preset_in.color_reactivity
    preset.fade_factor = preset_in.fade_factor
    preset.organic = preset_in.organic
    preset.set_colors(preset_in.colors)

    db.commit()
    db.refresh(preset)
    return model_to_read(preset)


@app.delete("/presets/{preset_id}")
def delete_preset(preset_id: int):
    db: Session = next(get_db())
    preset = db.query(PresetModel).filter(PresetModel.id == preset_id).first()
    if not preset:
        raise HTTPException(status_code=404, detail="Preset not found")
    db.delete(preset)
    db.commit()
    return {"detail": "Preset deleted"}


@app.get("/")
def root():
    return {"message": "IoT Music Visualizer backend is running."}
