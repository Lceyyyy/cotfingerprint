#!/usr/bin/env bash
# 在已可见 GPU 的 shell 中执行（交互节点 / salloc / 单机 Docker 等）:
#   conda activate <你的环境>   # 或 source venv/bin/activate
#   bash scripts/gpu/train_gate_arith.sh
#
# 可选: export HF_HOME=/path/to/models  CUDA_VISIBLE_DEVICES=0

set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/run_env.sh"
llamafactory-cli train examples/train_lora/qwen35_4b_lora_sft_migration.yaml
