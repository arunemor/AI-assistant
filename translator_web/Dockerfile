# Use Python 3.11 slim as base
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Disable buffering
ENV PYTHONUNBUFFERED=1

# Install system dependencies required for PyQt5 (minimal for headless)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libx11-6 libxext6 libxrender1 libglib2.0-0 \
    libsm6 libxkbcommon0 libxcb-xinerama0 libgl1-mesa-glx \
    wget curl && \
    rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy app code
COPY . .

# Expose Flask port
EXPOSE 5000

# Run the Flask app
CMD ["python3", "server.py"]
