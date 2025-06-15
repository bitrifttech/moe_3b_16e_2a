#!/bin/bash
set -e

# Remove old venv if it exists
if [ -d "venv" ]; then
  echo "Removing old venv..."
  rm -rf venv
fi

# Create new venv
python -m venv venv
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install torch (CPU by default, CUDA if on Linux/NVIDIA)
if [[ "$(uname -s)" == "Linux" ]] && python -c "import torch; print(torch.cuda.is_available())" 2>/dev/null | grep -q 'True'; then
  echo "Installing torch with CUDA support..."
  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
else
  echo "Installing torch (CPU only)..."
  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
fi

# Install core modules
pip install 'transformers>=4.42.0,<5.0.0' datasets accelerate deepspeed bitsandbytes
# Install tutel from GitHub
pip install -U --no-build-isolation git+https://github.com/microsoft/tutel@main

# Print installed versions
python -c "import torch; print('torch:', torch.__version__)
import transformers; print('transformers:', transformers.__version__)
import datasets; print('datasets:', datasets.__version__)
import accelerate; print('accelerate:', accelerate.__version__)
import deepspeed; print('deepspeed:', deepspeed.__version__)
import bitsandbytes as bnb; print('bitsandbytes:', bnb.__version__)
import tutel; print('tutel:', getattr(tutel, '__version__', 'installed'))"

echo "\nEnvironment setup complete! Activate with: source venv/bin/activate" 