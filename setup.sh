#!/usr/bin/env bash


set -euo pipefail
IFS=$'\n\t'

ENV_NAME="${ENV_NAME:-core}"
PYTHON_VERSION="${PYTHON_VERSION:-3.9}"
INSTALL_FLASH_ATTN="${INSTALL_FLASH_ATTN:-1}"

log()  { printf "[%s] %s\n" "$(date '+%Y-%m-%d %H:%M:%S')" "$*"; }
warn() { printf "[%s] WARNING: %s\n" "$(date '+%Y-%m-%d %H:%M:%S')" "$*" >&2; }
die()  { printf "[%s] ERROR: %s\n" "$(date '+%Y-%m-%d %H:%M:%S')" "$*" >&2; exit 1; }

command -v conda >/dev/null 2>&1 || die "conda not found. Please install Miniconda/Anaconda and ensure 'conda' is on PATH."

# Enable `conda activate` in a non-interactive shell
if conda info --base >/dev/null 2>&1; then
  CONDA_BASE="$(conda info --base)"
  # shellcheck disable=SC1091
  source "${CONDA_BASE}/etc/profile.d/conda.sh"
else
  die "Failed to locate conda base directory."
fi

# Create environment if it doesn't exist
if conda env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
  log "Conda env '${ENV_NAME}' already exists. Skipping creation."
else
  log "Creating conda env '${ENV_NAME}' (python=${PYTHON_VERSION})..."
  conda create -y -n "${ENV_NAME}" "python=${PYTHON_VERSION}" ipython
fi

log "Activating conda env '${ENV_NAME}'..."
conda activate "${ENV_NAME}"

log "Upgrading pip/setuptools/wheel..."
python -m pip install --upgrade pip setuptools wheel

log "Installing PyTorch stack (torch/torchvision/torchaudio) via pip..."
python -m pip install --upgrade torch torchvision torchaudio

if [[ "${INSTALL_FLASH_ATTN}" == "1" ]]; then
  log "Installing flash-attn (CUDA build skipped by default)..."
  set +e
  FLASH_ATTENTION_SKIP_CUDA_BUILD=TRUE python -m pip install --no-build-isolation flash-attn
  status=$?
  set -e
  if [[ $status -ne 0 ]]; then
    warn "flash-attn install failed (exit=${status}). This is common on non-Linux or without compatible toolchain."
    warn "You can rerun with INSTALL_FLASH_ATTN=0 to skip, or install a compatible CUDA/toolchain and retry."
  fi
else
  log "Skipping flash-attn (INSTALL_FLASH_ATTN=0)."
fi

log "Installing core Python deps (transformers/accelerate/numpy/einops/protobuf/nltk/rouge/bert_score) via pip..."
python -m pip install --upgrade transformers accelerate numpy einops protobuf nltk rouge bert_score

log "Installing sentencepiece via conda-forge..."
conda install -y -c conda-forge sentencepiece

log "Done."
log "To use this env later:  conda activate ${ENV_NAME}"
