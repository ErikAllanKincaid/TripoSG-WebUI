# TripoSG-WebUI Docker Image
# GPU-accelerated Image-to-3D mesh generation
#
# Build: docker build -t triposg-webui .
# Run:   docker run --gpus all -p 5000:5000 -v triposg-weights:/app/pretrained_weights triposg-webui

FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime

WORKDIR /app

# System dependencies for OpenCV, pymeshlab, trimesh
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    && rm -rf /var/lib/apt/lists/*

# Clone TripoSG for model code (scripts/, triposg/)
RUN git clone --depth 1 https://github.com/VAST-AI-Research/TripoSG.git /app/triposg-src \
    && mv /app/triposg-src/scripts /app/scripts \
    && mv /app/triposg-src/triposg /app/triposg \
    && rm -rf /app/triposg-src

# Python dependencies
# torch already in base image, add the rest
COPY requirements.txt .
RUN pip install --no-cache-dir flask \
    && pip install --no-cache-dir -r requirements.txt

# Copy WebUI app and assets
COPY app.py .
COPY 3d-printing-logo.png eak.png ./
COPY TripoSG-WebUI_Parameters-explained.txt ./

# Outputs directory for generated meshes
RUN mkdir -p /app/outputs

# Model weights downloaded here on first run (~30GB)
# Mount a volume here to persist across container restarts
VOLUME /app/pretrained_weights

EXPOSE 5000

# Run Flask app (single worker, GPU workload is sequential anyway)
CMD ["python", "app.py"]
