#!/usr/bin/env python3
"""Tabulate runs/<name>/metrics.json files as a Markdown table."""
from __future__ import annotations

import json
import sys
from pathlib import Path


def row(name: str, m: dict) -> str:
    a = m["args"]
    last = m["epochs"][-1] if m.get("epochs") else {}
    acc = f"{last['acc'] * 100:.1f}%" if "acc" in last else "-"
    ips = f"{m['train_images_per_s']:.1f}" if "train_images_per_s" in m else "-"
    fb = m.get("cpu_fallback_ops")
    fb = "-" if fb is None else (", ".join(fb) if fb else "none")
    return (f"| {name} | {a['device']} | {m['world_size']} | {a['mode']} | {'yes' if a['compile'] else 'no'} "
            f"| {'bf16' if a['bf16'] else 'fp32'} | {m['params_trainable'] / 1e6:.2f}M | {m.get('first_step_s', 0):.0f}s "
            f"| {ips} | {acc} | {m['train_s']:.0f}s | {fb} |")


def main(root: str = "runs") -> None:
    print("| run | device | workers | mode | compile | precision | trainable | first step | train img/s "
          "| test acc | wall | CPU fallback ops |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for f in sorted(Path(root).glob("*/metrics.json")):
        print(row(f.parent.name, json.loads(f.read_text())))


if __name__ == "__main__":
    main(*sys.argv[1:])
