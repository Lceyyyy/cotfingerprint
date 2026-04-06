#!/usr/bin/env bash
# 在已可见 GPU 的 shell 中执行（需 jieba / nltk / rouge_chinese）:
#   conda activate <你的环境>
#   bash scripts/gpu/predict_arith.sh

set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/run_env.sh"
llamafactory-cli train examples/train_lora/qwen35_4b_predict_synthetic_ari.yaml
