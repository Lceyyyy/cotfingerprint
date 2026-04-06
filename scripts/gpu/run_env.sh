# 由 train / eval / predict 脚本 source；不要单独执行。
# 使用方式：先登录到带 GPU 的节点（如 salloc、srun --pty），激活你的 Python 环境后再运行对应 .sh
#
# 多卡节点上只想用 1 卡训练时（否则会按可见 GPU 数起多个 torchrun 进程，全参微调极易 OOM）：
#   export CUDA_VISIBLE_DEVICES=0    # 或 1、2…
# 若仍可见多卡但强制单进程：
#   export NPROC_PER_NODE=1
# 单进程调试（不用 torchrun）：
#   export FORCE_TORCHRUN=0
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$ROOT/LlamaFactory" || exit 1
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export FORCE_TORCHRUN="${FORCE_TORCHRUN:-1}"

# DeepSpeed：import 时会跑 $CUDA_HOME/bin/nvcc -V；DS_SKIP_CUDA_CHECK 在部分版本仍不会跳过这一步。
# 常见坑：shell/作业里 CUDA_HOME 指到旧路径（如 .../scratch/softwares/.../envs/cot），与当前 conda 前缀不一致且无 nvcc。
export DS_SKIP_CUDA_CHECK="${DS_SKIP_CUDA_CHECK:-1}"
_fix_cuda_home_for_deepspeed() {
  if [ -n "${CUDA_HOME:-}" ] && [ ! -x "${CUDA_HOME}/bin/nvcc" ]; then
    unset CUDA_HOME
  fi
  if [ -n "${CONDA_PREFIX:-}" ] && [ -x "${CONDA_PREFIX}/bin/nvcc" ]; then
    export CUDA_HOME="${CONDA_PREFIX}"
    return 0
  fi
  if command -v nvcc >/dev/null 2>&1; then
    export CUDA_HOME="$(dirname "$(dirname "$(command -v nvcc)")")"
    return 0
  fi
  return 0
}
_fix_cuda_home_for_deepspeed
unset -f _fix_cuda_home_for_deepspeed
# 若上面仍无法得到 CUDA_HOME（which nvcc 为空），请在 cot 环境安装 nvcc 或先 module load cuda：
#   conda install -n cot -c nvidia cuda-nvcc
# Triton 缓存在 NFS 上可能慢或退出挂起，可改到节点本地盘：
# export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-/tmp/$USER/triton_cache}"
