# docker/Dockerfile.ml
# ML Service with CUDA 13 support and uv package manager
FROM ghcr.io/astral-sh/uv:latest AS builder

WORKDIR /app

# Copiamos los archivos de configuración del proyecto
COPY pyproject.toml uv.lock ./

# Instalamos las dependencias con el grupo ml
# --frozen evita que uv intente actualizar el lockfile
RUN uv sync --frozen --no-cache --no-install-project --group ml

# Segunda etapa con CUDA
FROM nvidia/cuda:13.0.0-base-ubuntu24.04

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    DEBIAN_FRONTEND=noninteractive \
    CUDA_HOME=/usr/local/cuda \
    PATH=/usr/local/cuda/bin:/app/.venv/bin:$PATH \
    LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3 \
    python3-dev \
    python3-pip \
    python3-venv \
    build-essential \
    git \
    wget \
    curl \
    libgomp1 \
    libopenblas-dev \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Create symlink for python
RUN ln -sf /usr/bin/python3 /usr/bin/python

# Copiar uv desde builder
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Copiar el entorno virtual desde builder
COPY --from=builder /app/.venv /app/.venv

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
CMD ["uv", "run", "python", "-m", "backend.ml_engine.training.trainer"]
