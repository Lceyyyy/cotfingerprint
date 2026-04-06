#!/bin/bash
# 请在仓库根目录 cotfingerprint/ 下提交:
#   cd <path-to>/cotfingerprint && sbatch scripts/slurm/predict_arith.sbatch
# 依赖: pip install jieba nltk rouge-chinese

#SBATCH --job-name=lf-qwen4b-pred-arith
#SBATCH --gpus=rtx_4090:1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=16G
#SBATCH -t 08:00:00
#SBATCH -o logs/lf_qwen4b_predict_arith_%j.out
#SBATCH -e logs/lf_qwen4b_predict_arith_%j.err

set -euo pipefail

ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "$ROOT"
mkdir -p logs

set +u
source ~/.bashrc
set -u
conda activate cotfingerprint

cd "$ROOT/LlamaFactory"

export HF_HOME="${HF_HOME:-/cluster/home/yifhou/group_work/huggingface_models}"
export FORCE_TORCHRUN=1

llamafactory-cli train examples/train_lora/qwen35_4b_predict_synthetic_ari.yaml
