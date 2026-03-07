FROM python:3.13-slim

# OpenCV runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install only what the service needs (no playwright, no tesseract)
RUN pip install --no-cache-dir \
    flask \
    opencv-python \
    numpy \
    psutil

COPY scorer.py app.py ./

RUN mkdir -p /data

EXPOSE 5000

CMD ["python", "app.py"]
