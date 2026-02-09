# ============================================================================
# ML Worker Dockerfile
# ============================================================================

# ============================================================================
# Builder Stage - Install dependencies
# ============================================================================
FROM nvidia/cuda:12.6.0-cudnn-devel-ubuntu22.04 AS builder

# Build arguments
ARG UV_VERSION=0.5.11
ARG PYTHON_VERSION=3.11
ARG PYTORCH_VERSION=2.5.1
ARG CUDA_VERSION=cu124

# Environment variables
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    UV_SYSTEM_PYTHON=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_HTTP_TIMEOUT=600 \
    UV_CONCURRENT_DOWNLOADS=4 \
    PATH="/root/.local/bin:$PATH" \
    CUDA_HOME=/usr/local/cuda \
    TORCH_CUDA_ARCH_LIST="9.0" \
    CUDA_LAUNCH_BLOCKING=0

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python${PYTHON_VERSION} \
    python${PYTHON_VERSION}-dev \
    python3-pip \
    build-essential \
    git \
    curl \
    ca-certificates \
    libgomp1 \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Create symlink for python
RUN ln -sf /usr/bin/python${PYTHON_VERSION} /usr/bin/python && \
    ln -sf /usr/bin/python${PYTHON_VERSION} /usr/bin/python3

# Install UV package manager
RUN curl -LsSf https://astral.sh/uv/${UV_VERSION}/install.sh | sh

WORKDIR /app

# Copy dependency files
COPY pyproject.toml uv.lock* ./

# ============================================================================
# CRITICAL: Install PyTorch FIRST with correct CUDA version
# ============================================================================
RUN echo "Installing PyTorch ${PYTORCH_VERSION} with CUDA ${CUDA_VERSION}..." && \
    pip install --no-cache-dir --timeout 300 --retries 10 \
    torch==${PYTORCH_VERSION}+${CUDA_VERSION} \
    torchvision==0.20.1+${CUDA_VERSION} \
    torchaudio==${PYTORCH_VERSION}+${CUDA_VERSION} \
    --index-url https://download.pytorch.org/whl/${CUDA_VERSION}

# Verify PyTorch installation
RUN python -c "import torch; \
    print(f'PyTorch version: {torch.__version__}'); \
    print(f'CUDA available: {torch.cuda.is_available()}'); \
    print(f'CUDA version: {torch.version.cuda}'); \
    print(f'cuDNN version: {torch.backends.cudnn.version()}'); \
    assert '${CUDA_VERSION}' in torch.__version__, 'Wrong CUDA version!'"

# ============================================================================
# Install dependencies with retries and increased timeout
# ============================================================================
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-cache --no-install-project \
    --group ml --group db --group celery --group mlflow --group data || \
    (echo "First attempt failed, retrying with longer timeout..." && \
    UV_HTTP_TIMEOUT=900 uv sync --frozen --no-cache --no-install-project \
    --group ml --group db --group celery --group mlflow --group data)

# Verify installations
RUN .venv/bin/python -c "import torch, sklearn, pandas, numpy; \
    print('All core packages installed')"

# ============================================================================
# Runtime Stage - Production image
# ============================================================================
FROM nvidia/cuda:12.6.0-cudnn-runtime-ubuntu22.04 AS runtime

ARG PYTHON_VERSION=3.11

# Environment variables
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/app/.venv/bin:$PATH" \
    PYTHONPATH="/app:/app/backend" \
    CUDA_HOME=/usr/local/cuda \
    TORCH_CUDA_ARCH_LIST="9.0" \
    PYTORCH_ALLOC_CONF="expandable_segments:True" \
    OMP_NUM_THREADS=4 \
    MKL_NUM_THREADS=4 \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility

# Install runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python${PYTHON_VERSION} \
    python${PYTHON_VERSION}-distutils \
    libgomp1 \
    libopenblas0 \
    && rm -rf /var/lib/apt/lists/*

# Create symlink for python
RUN ln -sf /usr/bin/python${PYTHON_VERSION} /usr/bin/python && \
    ln -sf /usr/bin/python${PYTHON_VERSION} /usr/bin/python3

WORKDIR /app

# Copy virtual environment from builder
COPY --from=builder /app/.venv /app/.venv

# Copy application code
COPY backend/ backend/

# Create necessary directories
RUN mkdir -p /app/models /app/data/parquet /app/logs && \
    chmod -R 755 /app/models /app/data /app/logs

# Verify installation in runtime
RUN python -c "import torch; \
    print(f'✓ PyTorch {torch.__version__}'); \
    print(f'✓ CUDA: {torch.version.cuda}'); \
    print(f'✓ cuDNN: {torch.backends.cudnn.version()}'); \
    print(f'✓ GPU Available: {torch.cuda.is_available()}')" || \
    echo "⚠ Warning: CUDA not available (will work in CPU mode)"

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import torch; exit(0 if torch.cuda.is_available() else 1)" || exit 0

# Default command
CMD ["celery", "-A", "backend.workers.celery_app", "worker", \
    "--loglevel=info", \
    "--concurrency=2", \
    "--max-tasks-per-child=50", \
    "--task-events", \
    "--time-limit=3600", \
    "--soft-time-limit=3300", \
    "-Q", "ml"]
