#!/usr/bin/env python3
"""由 ari/train.jsonl（expression + steps）生成「等号续写」格式。

prompt = "{expression} = "，response = steps 去掉该前缀后的剩余链式推导，
与仅给模型「算式 = 」再由模型补全后续的训练目标一致。
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def convert_line(obj: dict) -> dict:
    expr = obj["expression"]
    steps = obj["steps"]
    prefix = f"{expr} = "
    if not steps.startswith(prefix):
        raise ValueError(f"steps 不以 {prefix!r} 开头: {steps[:120]!r}")
    return {
        "prompt": prefix,
        "response": steps[len(prefix) :],
        "expression": expr,
        "result": obj.get("result"),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("input", type=Path)
    ap.add_argument("output", type=Path)
    args = ap.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with args.input.open(encoding="utf-8") as fin, args.output.open("w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            out = convert_line(json.loads(line))
            fout.write(json.dumps(out, ensure_ascii=False) + "\n")
            n += 1
    print(f"wrote {n} lines -> {args.output}")


if __name__ == "__main__":
    main()
