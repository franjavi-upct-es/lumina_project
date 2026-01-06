# docker/Dockerfile.ml
# ML Service with CUDA 12 support and uv package manager
FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    DEBIAN_FRONTEND=noninteractive \
    CUDA_HOME=/usr/local/cuda \
    PATH=/usr/local/cuda/bin:/root/.cargo/bin:$PATH \
    LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH \
    UV_SYSTEM_PYTHON=1

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.11 \
    python3.11-dev \
    python3-pip \
    build-essential \
    git \
    wget \
    curl \
    libgomp1 \
    libopenblas-dev \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Create symlinks for python
RUN ln -sf /usr/bin/python3.11 /usr/bin/python3 && \
    ln -sf /usr/bin/python3.11 /usr/bin/python

# Install uv - fast Python package installer
RUN curl -LsSf https://astral.sh/uv/install.sh | sh

# Verify uv installation
RUN uv --version

# Copy requirements files
COPY requirements/base.txt requirements/base.txt
COPY requirements/ml.txt requirements/ml.txt

# Install PyTorch with CUDA 12.1 support using uv
RUN uv pip install --system \
    torch==2.1.2 \
    torchvision==0.16.2 \
    torchaudio==2.1.2 \
    --index-url https://download.pytorch.org/whl/cu121

# Install base requirements using uv
RUN uv pip install --system -r requirements/base.txt

# Install ML requirements using uv
RUN uv pip install --system -r requirements/ml.txt

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p /app/models /app/data/parquet /app/logs

# Set Python path
ENV PYTHONPATH=/app

# Verify CUDA and PyTorch installation
RUN python3 -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda}')"

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python3 -c "import torch; assert torch.cuda.is_available(), 'CUDA not available'" || exit 1

# Default command (can be overridden)
CMD ["python3", "-m", "ml_engine.training.trainer"]
