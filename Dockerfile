# ============================================================
# Dockerfile — Run the complete Steel Defect Detection app
# ============================================================
# This Dockerfile packages everything so you can run the app
# with ONE COMMAND: docker compose up
#
# HOW IT WORKS:
# 1. Starts from a Python base image
# 2. Installs all dependencies
# 3. Copies our code
# 4. Runs both the FastAPI backend and Streamlit frontend
#
# BUILD & RUN:
#   docker compose up --build
#   Then open:
#     - Streamlit UI: http://localhost:8501
#     - API docs:     http://localhost:8000/docs
# ============================================================

FROM python:3.11-slim

# Set working directory inside the container
WORKDIR /app

# Install system dependencies needed for OpenCV
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (Docker caches this layer if requirements don't change)
COPY requirements.txt .

# Install Python dependencies
# Using CPU-only PyTorch to keep the image small (GPU needs nvidia-docker)
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r requirements.txt

# Copy all project files
COPY . .

# Expose ports for FastAPI (8000) and Streamlit (8501)
EXPOSE 8000 8501

# Default command: run both services
# We use a shell script to start both in the background
CMD ["sh", "-c", "uvicorn app.backend:app --host 0.0.0.0 --port 8000 & streamlit run app/frontend.py --server.port 8501 --server.address 0.0.0.0 --server.headless true"]
