# 由 train / eval / predict 脚本 source；不要单独执行。
# 使用方式：先登录到带 GPU 的节点（如 salloc、srun --pty），激活你的 Python 环境后再运行对应 .sh
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$ROOT/LlamaFactory" || exit 1
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export FORCE_TORCHRUN="${FORCE_TORCHRUN:-1}"
