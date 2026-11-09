"""DINOv3 backbone + classification head, with the three fine-tuning modes.

linear : backbone frozen, only the head trains (a "linear probe").
lastn  : the last N transformer blocks, the final norms and the head train.
full   : every parameter trains, the backbone at a lower learning rate.

The weights come from timm's ungated copy of Meta's DINOv3 checkpoints
(e.g. ``vit_small_patch16_dinov3.lvd1689m``). The DINOv3 License still applies.
"""
from __future__ import annotations

import torch
from torch import nn

MODES = ("linear", "lastn", "full")


def build_model(
    name: str = "vit_small_patch16_dinov3.lvd1689m",
    num_classes: int = 37,
    mode: str = "lastn",
    last_n: int = 2,
    pretrained: bool = True,
) -> nn.Module:
    import timm

    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
    model = timm.create_model(name, pretrained=pretrained, num_classes=num_classes)
    set_trainable(model, mode, last_n)
    return model


def set_trainable(model: nn.Module, mode: str, last_n: int = 2) -> None:
    """Freeze/unfreeze parameters in place for the chosen mode."""
    for p in model.parameters():
        p.requires_grad = mode == "full"
    if mode == "full":
        return
    trainable: list[nn.Module] = [model.get_classifier()]
    if mode == "lastn":
        if not 1 <= last_n <= len(model.blocks):
            raise ValueError(f"last_n must be in 1..{len(model.blocks)}")
        trainable += list(model.blocks[-last_n:])
        trainable += [m for m in (getattr(model, "norm", None), getattr(model, "fc_norm", None)) if m is not None]
    for module in trainable:
        for p in module.parameters():
            p.requires_grad = True


def param_groups(model: nn.Module, lr: float, backbone_lr_scale: float = 0.1, weight_decay: float = 0.05):
    """Head at ``lr``, unfrozen backbone params at ``lr * backbone_lr_scale``.

    Biases and norm weights (1-D tensors) get no weight decay.
    """
    head_ids = {id(p) for p in model.get_classifier().parameters()}
    groups: dict[tuple[bool, bool], list[torch.Tensor]] = {}
    for p in model.parameters():
        if not p.requires_grad:
            continue
        key = (id(p) in head_ids, p.ndim > 1)
        groups.setdefault(key, []).append(p)
    out = []
    for (is_head, decays), params in groups.items():
        out.append(
            {
                "params": params,
                "lr": lr if is_head else lr * backbone_lr_scale,
                "weight_decay": weight_decay if decays else 0.0,
            }
        )
    return out


def count_params(model: nn.Module) -> tuple[int, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable
