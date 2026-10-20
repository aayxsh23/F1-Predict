#!/bin/bash
# The training stack on the rented GPU machine (AMI: Deep Learning Base OSS
# Nvidia Driver GPU, Ubuntu 24.04). Run as the ubuntu user; see training/README.md.
set -euxo pipefail
cd ~
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
uv venv --python 3.12 ~/venv
source ~/venv/bin/activate
uv pip install unsloth unsloth_zoo hf_transfer
# Qwen3.5's linear-attention layers: fast kernels (without them transformers falls back to slow PyTorch loops)
uv pip install flash-linear-attention
uv pip install causal-conv1d --no-build-isolation || echo "causal-conv1d not built: its slower fallback is used"

# llama.cpp, only for the GGUF export (CPU build is enough)
sudo apt-get update -qq && sudo apt-get install -y -qq cmake build-essential
git clone --depth 1 https://github.com/ggml-org/llama.cpp ~/llama.cpp
cmake -S ~/llama.cpp -B ~/llama.cpp/build -DGGML_CUDA=OFF -DLLAMA_CURL=OFF
cmake --build ~/llama.cpp/build --target llama-quantize -j
uv pip install -e ~/llama.cpp/gguf-py sentencepiece protobuf

# Ollama, the laptop's version, to check the exported file loads and answers
curl -fsSL https://ollama.com/install.sh | OLLAMA_VERSION=0.40.2 sh

HF_HUB_ENABLE_HF_TRANSFER=1 python -c "from huggingface_hub import snapshot_download; snapshot_download('Qwen/Qwen3.5-9B')"
touch ~/setup-done
