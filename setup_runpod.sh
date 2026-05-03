#!/usr/bin/env bash
# RunPod environment setup — installs only what is missing.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

log() { echo "[setup] $*"; }

pip_has() { python -c "import importlib.util; exit(0 if importlib.util.find_spec('$1') else 1)" 2>/dev/null; }

# ---------------------------------------------------------------------------
# 1. Upgrade pip quietly
# ---------------------------------------------------------------------------
log "Upgrading pip..."
pip install -q --upgrade pip

# ---------------------------------------------------------------------------
# 2. Core packages — check each before installing
# ---------------------------------------------------------------------------
declare -A PKGS=(
    ["torch"]="torch>=2.1"
    ["transformers"]="transformers>=4.40"
    ["peft"]="peft>=0.10"
    ["datasets"]="datasets>=2.18"
    ["numpy"]="numpy>=1.26"
    ["sklearn"]="scikit-learn>=1.4"
    ["scipy"]="scipy>=1.12"
    ["wandb"]="wandb>=0.17"
    ["pydantic"]="pydantic>=2.6"
    ["omegaconf"]="omegaconf>=2.3"
    ["yaml"]="PyYAML>=6.0"
    ["tqdm"]="tqdm>=4.66"
    ["dotenv"]="python-dotenv>=1.0"
    ["ftfy"]="ftfy>=6.1"
    ["bs4"]="beautifulsoup4>=4.12"
    ["bitsandbytes"]="bitsandbytes>=0.43"
    ["accelerate"]="accelerate>=0.30"
)

MISSING=()
for mod in "${!PKGS[@]}"; do
    if ! pip_has "$mod"; then
        MISSING+=("${PKGS[$mod]}")
    fi
done

if [ ${#MISSING[@]} -gt 0 ]; then
    log "Installing: ${MISSING[*]}"
    pip install -q "${MISSING[@]}"
else
    log "All core packages already present."
fi

# ---------------------------------------------------------------------------
# 3. faiss — prefer GPU build on RunPod, fall back to CPU
# ---------------------------------------------------------------------------
if ! pip_has "faiss"; then
    if python -c "import torch; exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
        log "GPU detected — installing faiss-gpu"
        pip install -q faiss-gpu>=1.8 || { log "faiss-gpu failed, falling back to faiss-cpu"; pip install -q "faiss-cpu>=1.8"; }
    else
        log "No GPU — installing faiss-cpu"
        pip install -q "faiss-cpu>=1.8"
    fi
else
    log "faiss already installed."
fi

# ---------------------------------------------------------------------------
# 4. Install the project package in editable mode
# ---------------------------------------------------------------------------
if ! pip show email-fraud &>/dev/null; then
    log "Installing email-fraud package (editable)..."
    pip install -q -e "$REPO_ROOT"
else
    log "email-fraud package already installed."
fi

# ---------------------------------------------------------------------------
# 5. Verify critical imports
# ---------------------------------------------------------------------------
log "Verifying imports..."
python - <<'EOF'
import sys
failures = []
checks = [
    "torch", "transformers", "peft", "datasets", "numpy",
    "sklearn", "scipy", "wandb", "pydantic", "omegaconf",
    "yaml", "tqdm", "dotenv", "ftfy", "bs4",
    "bitsandbytes", "accelerate", "faiss",
]
for mod in checks:
    try:
        __import__(mod)
    except ImportError:
        failures.append(mod)

if failures:
    print(f"[setup] FAILED imports: {failures}", file=sys.stderr)
    sys.exit(1)

import torch
print(f"[setup] torch {torch.__version__}  |  CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"[setup] GPU: {torch.cuda.get_device_name(0)}")
print("[setup] All imports OK.")
EOF

# ---------------------------------------------------------------------------
# 6. Node.js 20 + Claude Code CLI
# ---------------------------------------------------------------------------
if ! command -v node &>/dev/null || [[ "$(node --version 2>/dev/null)" != v20* ]]; then
    log "Installing Node.js 20..."
    curl -fsSL https://deb.nodesource.com/setup_20.x | bash -
    apt-get install -y nodejs
else
    log "Node.js $(node --version) already installed."
fi

if ! command -v claude &>/dev/null; then
    log "Installing Claude Code CLI..."
    npm install -g @anthropic-ai/claude-code
else
    log "Claude Code $(claude --version 2>/dev/null) already installed."
fi

log "Setup complete."
