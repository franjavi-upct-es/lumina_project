# docker/Dockerfile.ml
# ML Service with GPU support for model training
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

# Set enviroment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    DEBIAN_FRONTEND=noninteractive \
    CUDA_HOME=/usr/local/cuda \
    PATH=/usr/local/cuda/bin:$PATH \
    LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.11 \
    python3-pip \
    python3-dev \
    build-essential \
    git \
    wget \
    curl \
    libgomp1 \
    libopenblas-dev \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN python3 -m pip install --upgrade pip setuptools wheel

# Install PyTorch with CUDA support
RUN pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu118

# Copy requirements
COPY requirements/base.txt requirements/base.txt
COPY requirements/ml.txt requirements/ml.txt

# Copy applicaction code
COPY . .

# Create necessary directories
RUN mkdir -p /app/models /app/data/parquet /app/logs

# Set Python path
ENV PYTHONPATH=/app

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python3 -c "import torch; print(torch.cuda.is_available())" || exit 1

# Default command (can be overridden)
CMD ["python3", "-m", "ml_engine.training.trainer"]
