# ---------- IRIS Offline AI Dockerfile ----------
FROM python:3.11-slim

ENV DEBIAN_FRONTEND=noninteractive

# --- Install system dependencies ---
RUN apt-get update && apt-get install -y \
    ffmpeg \
    espeak-ng \
    git \
    build-essential \
    pkg-config \                
    libavformat-dev \            
    libavdevice-dev \
    libavfilter-dev \
    libavcodec-dev \
    libswresample-dev \
    libswscale-dev \
    libavutil-dev \
 && rm -rf /var/lib/apt/lists/*

# --- Copy and install Python deps ---
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir -r requirements.txt

# --- Copy code and models ---
COPY app/ /app/
COPY models/ /models/

# --- Environment and entry ---
ENV WHISPER_DIR=/models/whisper/tiny.en
ENV YOLO_WEIGHTS=/models/yolo/yolov8n.pt
ENV PIPER_VOICE=/models/piper/en_US-amy-medium.onnx
ENV PYTHONUNBUFFERED=1

CMD ["python", "main.py"]
# -----------------------------------------------