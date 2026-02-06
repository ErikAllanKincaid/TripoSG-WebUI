# TripoSG-WebUI Docker Image
# GPU-accelerated Image-to-3D mesh generation
#
# Build: docker build -t triposg-webui .
# Run:   docker run --gpus all -p 5000:5000 -v triposg-weights:/app/pretrained_weights triposg-webui

# devel image required: diso compiles CUDA extensions and needs nvcc + cuda_runtime.h
FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel

# CUDA_HOME: tells torch.utils.cpp_extension where to find nvcc
# CPLUS_INCLUDE_PATH: tells g++ where to find cuda_runtime.h
# FORCE_CUDA: diso's setup.py skips .cu files if torch.cuda.is_available() is False
#   (no GPU at build time), FORCE_CUDA=1 overrides this check
# TORCH_CUDA_ARCH_LIST: which GPU architectures to compile for
#   7.0=Titan V, 8.0=A100, 8.6=RTX 3090, 8.9=RTX 4090, 9.0=H100
ENV CUDA_HOME=/usr/local/cuda
ENV CPLUS_INCLUDE_PATH=/usr/local/cuda/targets/x86_64-linux/include:${CPLUS_INCLUDE_PATH}
ENV FORCE_CUDA=1
ENV TORCH_CUDA_ARCH_LIST="7.0 8.0 8.6 8.9 9.0"

WORKDIR /app

# System dependencies for OpenCV, pymeshlab, trimesh
# g++ needed to compile diso (C++ CUDA extension for marching cubes)
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    g++ \
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
# diso requires --no-build-isolation so it can find torch's CUDA headers at build time
RUN pip install --no-cache-dir flask \
    && pip install --no-cache-dir --no-build-isolation diso \
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
