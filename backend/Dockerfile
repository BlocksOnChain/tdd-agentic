# syntax=docker/dockerfile:1
# Default target: slim Python image (no PyTorch). Use for EMBEDDING_PROVIDER=openai.
#   docker compose build backend
#
# Local embeddings from official PyTorch images (pre-installed torch, no pip CUDA download):
#   docker compose -f docker-compose.yml -f docker-compose.pytorch.yml build backend
#
# Targets: slim | pytorch-local

# Must be before any FROM so build-args apply and slim builds don't get a blank image name.
# No plain *-runtime tag on Docker Hub; CUDA runtime images still run on CPU without a GPU.
ARG PYTORCH_BASE_IMAGE=pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

# ---------------------------------------------------------------------------
# slim — minimal image; optional pip-installed torch when INSTALL_LOCAL_EMBEDDINGS=true
# ---------------------------------------------------------------------------
FROM python:3.12-slim AS slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Node.js LTS is required at runtime: dev/devops agents execute `npm`, `node`
# and JS/TS test runners (jest/vitest) via shell_run/run_tests inside this
# container. Without it those commands fail with exit 127 ("node: not found").
ARG NODE_MAJOR=20

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    ca-certificates \
    gnupg \
    && curl -fsSL "https://deb.nodesource.com/setup_${NODE_MAJOR}.x" | bash - \
    && apt-get install -y --no-install-recommends nodejs \
    && npm install -g npm@latest \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml /app/pyproject.toml

ARG INSTALL_LOCAL_EMBEDDINGS=false
ARG TORCH_DEVICE=cpu

RUN pip install --upgrade pip \
    && if [ "$INSTALL_LOCAL_EMBEDDINGS" = "true" ]; then \
         case "$TORCH_DEVICE" in \
           cpu) pip install torch --index-url https://download.pytorch.org/whl/cpu ;; \
           cuda|cu124|gpu) pip install torch --index-url https://download.pytorch.org/whl/cu124 ;; \
           cu118) pip install torch --index-url https://download.pytorch.org/whl/cu118 ;; \
           auto|*) pip install torch --index-url https://download.pytorch.org/whl/cpu ;; \
         esac \
         && pip install -e ".[local-embeddings]"; \
       else \
         pip install -e .; \
       fi

COPY backend /app/backend
RUN mkdir -p /app/workspace
EXPOSE 8000
CMD ["uvicorn", "backend.api.main:app", "--host", "0.0.0.0", "--port", "8000"]

# ---------------------------------------------------------------------------
# pytorch-local — official CPU/runtime PyTorch image (torch pre-installed)
# ---------------------------------------------------------------------------
FROM ${PYTORCH_BASE_IMAGE} AS pytorch-local

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# See slim stage: Node.js LTS is required so dev/devops agents can run npm and
# JS/TS test runners via shell_run/run_tests.
ARG NODE_MAJOR=20

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    ca-certificates \
    gnupg \
    && curl -fsSL "https://deb.nodesource.com/setup_${NODE_MAJOR}.x" | bash - \
    && apt-get install -y --no-install-recommends nodejs \
    && npm install -g npm@latest \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml /app/pyproject.toml

RUN pip install --upgrade pip \
    && pip install -e ".[local-embeddings]"

COPY backend /app/backend
RUN mkdir -p /app/workspace
EXPOSE 8000
CMD ["uvicorn", "backend.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
