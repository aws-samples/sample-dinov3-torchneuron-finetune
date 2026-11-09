#!/usr/bin/env python3
"""Inference with a fine-tuned checkpoint (``model.pt`` written by ``dinoft.train``).

Classify your own images (top-k breeds per image):
    python -m dinoft.predict --checkpoint runs/latest/model.pt --images photo1.jpg photos/

Re-measure test-set accuracy and inference throughput:
    python -m dinoft.predict --checkpoint runs/latest/model.pt --eval

Inference uses the same static-shape rule as training: images are resized to the
training size and every batch is padded to ``--batch-size``, so Neuron compiles
one graph and reuses it. ``--device cpu`` runs the same code off Trainium.
"""
from __future__ import annotations

import argparse
import json
import time
from contextlib import nullcontext
from pathlib import Path

import torch

from .data import build_datasets, pad_collate, transforms
from .model import build_model

EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--device", choices=["neuron", "cpu"], default="neuron")
    p.add_argument("--images", nargs="*", default=[], help="image files and/or folders to classify")
    p.add_argument("--eval", action="store_true", help="evaluate on the dataset's test split")
    p.add_argument("--eval-limit", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--top-k", type=int, default=3)
    p.add_argument("--compile", action="store_true", help='torch.compile(backend="neuron", dynamic=False)')
    p.add_argument("--bf16", action="store_true", help="bfloat16 autocast")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--out", default=None, help="write predictions / metrics as JSON here")
    return p.parse_args(argv)


def load(checkpoint: str, device: str, compile_: bool):
    # weights_only=True: a checkpoint is untrusted input, and unpickling arbitrary objects can
    # execute code. dinoft.train only stores tensors, dicts, lists, str/int/float/bool/None.
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=True)
    targs = ckpt["args"]
    model = build_model(targs["model"], ckpt["num_classes"], mode="full", pretrained=False)
    model.load_state_dict(ckpt["model"])
    model.eval().to(device)
    if compile_:
        backend = "neuron" if device == "neuron" else "aot_eager"
        model = torch.compile(model, backend=backend, dynamic=False)
    names = ckpt.get("class_names") or [str(i) for i in range(ckpt["num_classes"])]
    return model, names, targs


def batches(items, n):
    for i in range(0, len(items), n):
        yield items[i : i + n]


def expand(paths):
    out = []
    for p in map(Path, paths):
        out += sorted(f for f in p.iterdir() if f.suffix.lower() in EXTS) if p.is_dir() else [p]
    return out


@torch.no_grad()
def classify(model, names, files, image_size, device, batch_size, top_k, amp):
    from PIL import Image

    tf, collate, results = transforms(image_size, train=False), pad_collate(batch_size), []
    for chunk in batches(files, batch_size):
        x, _, valid = collate([(tf(Image.open(f).convert("RGB")), 0) for f in chunk])
        with amp:
            probs = model(x.to(device)).float().softmax(-1).cpu()[valid]
        for f, pr in zip(chunk, probs):
            conf, idx = pr.topk(top_k)
            results.append({"image": str(f), "top": [[names[i], round(c.item(), 4)] for c, i in zip(conf, idx)]})
    return results


@torch.no_grad()
def evaluate(model, loader, device, amp, warmup_batches=2):
    correct = seen = timed = 0
    t0 = None
    for b, (x, y, valid) in enumerate(loader):
        if b == warmup_batches:
            t0 = time.perf_counter()
        with amp:
            pred = model(x.to(device)).argmax(-1).cpu()  # .cpu() waits for the device
        correct += int(((pred == y) & valid).sum())
        seen += int(valid.sum())
        if t0 is not None:
            timed += int(valid.sum())
    dt = time.perf_counter() - t0 if t0 else float("nan")
    return {"acc": correct / seen, "n": seen, "images_per_s": timed / dt if t0 else None}


def main(argv=None):
    args = parse_args(argv)
    if args.device == "neuron":
        import torch_neuronx  # noqa: F401
    if not args.images and not args.eval:
        raise SystemExit("pass --images and/or --eval")
    model, names, targs = load(args.checkpoint, args.device, args.compile)
    amp = torch.autocast(device_type=args.device, dtype=torch.bfloat16) if args.bf16 else nullcontext()
    size = targs["image_size"]
    report = {"checkpoint": args.checkpoint, "model": targs["model"], "device": args.device}

    if args.images:
        t = time.perf_counter()
        report["predictions"] = classify(model, names, expand(args.images), size, args.device,
                                         args.batch_size, args.top_k, amp)
        print(f"[time] classified {len(report['predictions'])} images in {time.perf_counter() - t:.1f}s "
              "(the first call includes compile)")
        for r in report["predictions"]:
            print(f"{r['image']}: " + ", ".join(f"{n} {c:.3f}" for n, c in r["top"]))

    if args.eval:
        from torch.utils.data import DataLoader

        _, test_ds, _ = build_datasets(targs["dataset"], size, eval_limit=args.eval_limit)
        loader = DataLoader(test_ds, batch_size=args.batch_size, num_workers=args.workers,
                            collate_fn=pad_collate(args.batch_size))
        t = time.perf_counter()
        report["eval"] = evaluate(model, loader, args.device, amp)
        e = report["eval"]
        ips = f"{e['images_per_s']:.1f}" if e["images_per_s"] else "n/a"
        print(f"[eval] acc {e['acc']:.4f} on {e['n']} images | {ips} images/s after warm-up "
              f"| total {time.perf_counter() - t:.0f}s")

    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    main()
