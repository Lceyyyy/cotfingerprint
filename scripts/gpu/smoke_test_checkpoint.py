#!/usr/bin/env python3
"""不加载整模权重：检查 checkpoint 目录并加载 config + tokenizer（可在无 GPU 登录节点跑）。"""
from __future__ import annotations

import argparse
import json
import os
import sys


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument(
        "checkpoint_dir",
        nargs="?",
        default="saves/qwen35-4b/full/sft-migration/checkpoint-1563",
        help="相对 LlamaFactory 根目录，或绝对路径",
    )
    args = p.parse_args()
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "LlamaFactory"))
    ckpt = args.checkpoint_dir if os.path.isabs(args.checkpoint_dir) else os.path.join(root, args.checkpoint_dir)

    if not os.path.isdir(ckpt):
        print(f"ERROR: 目录不存在: {ckpt}", file=sys.stderr)
        return 1

    st = os.path.join(ckpt, "model.safetensors")
    if not os.path.isfile(st):
        print(f"ERROR: 缺少 model.safetensors: {st}", file=sys.stderr)
        return 1
    size_gb = os.path.getsize(st) / (1024**3)
    print(f"OK model.safetensors 存在，约 {size_gb:.2f} GiB")

    cfg_path = os.path.join(ckpt, "config.json")
    if not os.path.isfile(cfg_path):
        print(f"ERROR: 缺少 config.json", file=sys.stderr)
        return 1
    with open(cfg_path, encoding="utf-8") as f:
        cfg = json.load(f)
    print(f"OK config.json: model_type={cfg.get('model_type', '?')}, hidden_size={cfg.get('hidden_size', '?')}")

    try:
        from transformers import AutoTokenizer
    except ImportError:
        print("WARN: 未安装 transformers，跳过 tokenizer 测试")
        return 0

    tok = AutoTokenizer.from_pretrained(ckpt, trust_remote_code=True)
    print(f"OK tokenizer: vocab_size={getattr(tok, 'vocab_size', 'n/a')}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
