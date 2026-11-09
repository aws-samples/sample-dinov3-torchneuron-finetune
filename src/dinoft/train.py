#!/usr/bin/env python3
"""Fine-tune DINOv3 on AWS Trainium with plain PyTorch + TorchNeuron.

The only Neuron-specific choice is the device string: the loop is the same
forward -> loss -> backward -> clip -> step you would write for CUDA.

Single NeuronCore, eager mode (start here):
    python -m dinoft.train --device neuron --mode lastn --epochs 3

Compiled (one line: torch.compile(model, backend="neuron", dynamic=False)):
    python -m dinoft.train --device neuron --compile

Data parallel over the 4 logical NeuronCores of a trn2.3xlarge:
    torchrun --standalone --nproc-per-node=4 -m dinoft.train --device neuron --compile

CPU smoke test (no Trainium, no download):
    python -m dinoft.train --device cpu --dataset fake --max-steps 5
"""
from __future__ import annotations

import argparse
import json
import math
import os
import platform
import time
from contextlib import nullcontext
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn.functional as F

from .data import build_datasets, build_loaders
from .model import MODES, build_model, count_params, param_groups


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--device", choices=["neuron", "cpu"], default="neuron")
    p.add_argument("--model", default="vit_small_patch16_dinov3.lvd1689m")
    p.add_argument("--no-pretrained", action="store_true", help="random init (tests only)")
    p.add_argument("--dataset", default="pets", help="pets | fake")
    p.add_argument("--train-limit", type=int, default=None, help="use only N training images")
    p.add_argument("--eval-limit", type=int, default=None, help="use only N test images")
    p.add_argument("--mode", choices=MODES, default="lastn")
    p.add_argument("--last-n", type=int, default=2, help="blocks to unfreeze in --mode lastn")
    p.add_argument("--image-size", type=int, default=224)
    p.add_argument("--batch-size", type=int, default=32, help="per worker")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--max-steps", type=int, default=None, help="stop after N optimizer steps")
    p.add_argument("--lr", type=float, default=1e-3, help="head learning rate")
    p.add_argument("--backbone-lr-scale", type=float, default=0.1)
    p.add_argument("--weight-decay", type=float, default=0.05)
    p.add_argument("--warmup-steps", type=int, default=20)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--label-smoothing", type=float, default=0.1)
    p.add_argument("--bf16", action="store_true", help="autocast forward to bfloat16 (fp32 master weights)")
    p.add_argument("--compile", action="store_true", help='torch.compile(backend="neuron", dynamic=False)')
    p.add_argument("--fullgraph", action="store_true", help="make graph breaks an error under --compile")
    p.add_argument("--skip-eval", action="store_true")
    p.add_argument("--log-every", type=int, default=10)
    p.add_argument("--warmup-timing-steps", type=int, default=3, help="steps excluded from throughput (compile)")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="runs/latest")
    p.add_argument("--no-checkpoint", action="store_true")
    p.add_argument("--cache-dir", default=None, help="HF datasets cache dir")
    return p.parse_args(argv)


# --------------------------------------------------------------------------- setup


def setup_distributed(device_type: str):
    """Return (rank, world_size, local_rank, device). Works with or without torchrun."""
    world = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local = int(os.environ.get("LOCAL_RANK", "0"))
    if device_type == "neuron":
        import torch_neuronx  # noqa: F401  registers the "neuron" process-group backend
    if world > 1:
        # Initialise BEFORE allocating any Neuron tensor: the neuron backend uses
        # LOCAL_RANK / LOCAL_WORLD_SIZE to give each worker its own visible NeuronCore,
        # after which plain "neuron" means "this worker's core".
        dist.init_process_group(backend="neuron" if device_type == "neuron" else "gloo")
    device = torch.device("neuron" if device_type == "neuron" else "cpu")
    return rank, world, local, device


def versions() -> dict:
    out = {"python": platform.python_version(), "torch": torch.__version__}
    for mod in ("torch_neuronx", "timm", "torchvision", "datasets"):
        try:
            out[mod] = __import__(mod).__version__
        except Exception:
            pass
    return out


def lr_lambda(warmup: int, total: int):
    def f(step: int) -> float:
        if step < warmup:
            return (step + 1) / warmup
        progress = (step - warmup) / max(1, total - warmup)
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))

    return f


def unwrap(model):
    model = getattr(model, "_orig_mod", model)  # torch.compile wrapper
    return getattr(model, "module", model)  # DDP wrapper


def sync(device) -> None:
    if device.type == "neuron":
        torch.neuron.synchronize()


# --------------------------------------------------------------------------- eval


@torch.no_grad()
def evaluate(model, loader, device, amp_ctx, world: int) -> dict:
    model.eval()
    correct = torch.zeros((), dtype=torch.float32)
    seen = torch.zeros((), dtype=torch.float32)
    loss_sum = torch.zeros((), dtype=torch.float32)
    for x, y, valid in loader:
        x, y = x.to(device), y.to(device)
        with amp_ctx:
            logits = model(x)
        logits, y = logits.float().cpu(), y.cpu()  # pad mask is applied on the host
        loss = F.cross_entropy(logits, y, reduction="none")
        correct += ((logits.argmax(-1) == y) & valid).sum()
        loss_sum += (loss * valid).sum()
        seen += valid.sum()
    if world > 1:
        # The collective runs on the process group's device (neuron or cpu/gloo).
        stats = torch.stack([correct, seen, loss_sum]).to(device)
        dist.all_reduce(stats)
        correct, seen, loss_sum = stats.cpu()
    model.train()
    return {"acc": (correct / seen).item(), "loss": (loss_sum / seen).item(), "n": int(seen.item())}


# --------------------------------------------------------------------------- main


def main(argv=None) -> dict:
    args = parse_args(argv)
    torch.manual_seed(args.seed)
    rank, world, local, device = setup_distributed(args.device)
    is_main = rank == 0
    log = print if is_main else (lambda *a, **k: None)
    out = Path(args.out)
    if is_main:
        out.mkdir(parents=True, exist_ok=True)

    # ---- data
    train_ds, eval_ds, num_classes = build_datasets(
        args.dataset, args.image_size, train_limit=args.train_limit, eval_limit=args.eval_limit,
        cache_dir=args.cache_dir)
    train_loader, eval_loader = build_loaders(train_ds, eval_ds, args.batch_size, args.workers, world, rank, args.seed)

    # ---- model: build on CPU (same init on every rank), then move to the device
    model = build_model(args.model, num_classes, args.mode, args.last_n, pretrained=not args.no_pretrained)
    total, trainable = count_params(model)
    model.to(device)
    model.train()
    if world > 1:
        model = torch.nn.parallel.DistributedDataParallel(model)
    if args.compile:
        backend = "neuron" if args.device == "neuron" else "aot_eager"
        model = torch.compile(model, backend=backend, dynamic=False, fullgraph=args.fullgraph)

    optimizer = torch.optim.AdamW(param_groups(unwrap(model), args.lr, args.backbone_lr_scale, args.weight_decay))
    steps_per_epoch = len(train_loader)
    total_steps = min(args.max_steps or 10**12, steps_per_epoch * args.epochs)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda(args.warmup_steps, total_steps))
    amp_ctx = torch.autocast(device_type=device.type, dtype=torch.bfloat16) if args.bf16 else nullcontext()

    log(f"[setup] {args.model} mode={args.mode} params={total / 1e6:.2f}M trainable={trainable / 1e6:.2f}M "
        f"device={device} world={world} compile={args.compile} bf16={args.bf16}")
    log(f"[setup] train={len(train_ds)} eval={len(eval_ds)} classes={num_classes} "
        f"batch={args.batch_size}x{world} steps/epoch={steps_per_epoch} total_steps={total_steps}")

    track_fallbacks = args.device == "neuron"
    if track_fallbacks:
        import torch_neuronx

        torch_neuronx.clear_op_tracking()

    metrics = {"args": vars(args), "versions": versions(), "world_size": world, "params_total": total,
               "params_trainable": trainable, "steps": [], "epochs": []}
    step, t_start, timed_imgs, t_timed = 0, time.perf_counter(), 0, None
    done = False
    for epoch in range(args.epochs):
        if world > 1:
            train_loader.sampler.set_epoch(epoch)
        for x, y, _ in train_loader:
            x, y = x.to(device), y.to(device)
            t0 = time.perf_counter()
            with amp_ctx:
                logits = model(x)
            loss = F.cross_entropy(logits.float(), y, label_smoothing=args.label_smoothing)
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], args.grad_clip)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)

            if step == 0:
                sync(device)
                metrics["first_step_s"] = time.perf_counter() - t0
                log(f"[time] first step (includes compile / cache warm-up): {metrics['first_step_s']:.1f}s")
                if track_fallbacks:
                    fb = sorted(map(str, torch_neuronx.get_fallback_ops()))
                    metrics["cpu_fallback_ops"] = fb
                    log(f"[neuron] CPU fallback ops after step 1: {fb or 'none'}")
            if step == args.warmup_timing_steps:
                sync(device)
                t_timed, timed_imgs = time.perf_counter(), 0
            if t_timed is not None:
                timed_imgs += x.shape[0] * world

            if step % args.log_every == 0 or step == total_steps - 1:
                rec = {"step": step, "epoch": epoch, "loss": loss.item(), "grad_norm": grad_norm.item(),
                       "lr": scheduler.get_last_lr()[-1], "elapsed_s": time.perf_counter() - t_start}
                metrics["steps"].append(rec)
                log(f"step {step:5d} | ep {epoch} | loss {rec['loss']:.4f} | grad_norm {rec['grad_norm']:.3f} "
                    f"| lr {rec['lr']:.2e} | {rec['elapsed_s']:.0f}s")
            step += 1
            if step >= total_steps:
                done = True
                break

        ep = {"epoch": epoch, "step": step}
        if not args.skip_eval:
            t_eval = time.perf_counter()
            ep.update(evaluate(model, eval_loader, device, amp_ctx, world))
            ep["eval_s"] = time.perf_counter() - t_eval
            log(f"[eval] epoch {epoch}: acc {ep['acc']:.4f} loss {ep['loss']:.4f} on {ep['n']} images "
                f"({ep['eval_s']:.0f}s)")
        metrics["epochs"].append(ep)
        if done:
            break

    sync(device)
    t_end = time.perf_counter()
    metrics["train_s"] = t_end - t_start
    if args.compile:
        # Evidence that Dynamo really captured graphs (and how many breaks it hit).
        from torch._dynamo.utils import counters

        metrics["dynamo"] = {
            "stats": dict(counters["stats"]),
            "graph_breaks": {str(k): v for k, v in counters["graph_break"].items()},
        }
        log(f"[compile] dynamo stats {metrics['dynamo']['stats']} graph breaks "
            f"{sum(metrics['dynamo']['graph_breaks'].values())}")
    if t_timed is not None and t_end > t_timed:
        metrics["train_images_per_s"] = timed_imgs / (t_end - t_timed)
        log(f"[perf] {metrics['train_images_per_s']:.1f} train images/s (steps {args.warmup_timing_steps}+, "
            f"eval excluded only if --skip-eval)")

    if is_main:
        if not args.no_checkpoint:
            state = {k: v.detach().cpu() for k, v in unwrap(model).state_dict().items()}
            torch.save({"model": state, "args": vars(args), "num_classes": num_classes,
                        "class_names": getattr(train_ds, "classes", None)}, out / "model.pt")
            log(f"[save] {out / 'model.pt'}")
        (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
        log(f"[save] {out / 'metrics.json'}")
    if world > 1:
        dist.destroy_process_group()
    return metrics


if __name__ == "__main__":
    main()
