# Use official Python image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy backend source
COPY . .

# Expose FastAPI port
EXPOSE 8000

# Run the backend
CMD ["uvicorn", "backend_server:app", "--host", "0.0.0.0", "--port", "8000"]
