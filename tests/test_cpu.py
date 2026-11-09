"""CPU tests: run anywhere, no Trainium and no dataset download.

They use a randomly initialised DINOv3 ViT-S/16 (``--no-pretrained``) on the
synthetic ``fake`` dataset at 64x64 so the whole suite takes about a minute.
"""
from __future__ import annotations

import json
import os
import subprocess  # nosec B404 - used only to launch torchrun with a fixed argv in a test
import sys
from pathlib import Path

import pytest
import torch

from dinoft.data import FakeDataset, build_loaders, pad_collate
from dinoft.model import build_model, count_params, param_groups, set_trainable
from dinoft.train import main

SRC = Path(__file__).resolve().parents[1] / "src"
BASE = ["--device", "cpu", "--dataset", "fake", "--no-pretrained", "--image-size", "64",
        "--batch-size", "8", "--workers", "0", "--warmup-steps", "2", "--log-every", "1"]


@pytest.fixture(scope="module")
def tiny():
    return build_model(num_classes=4, mode="full", pretrained=False)


def test_modes_freeze_the_right_params(tiny):
    total, _ = count_params(tiny)
    set_trainable(tiny, "linear")
    head = sum(p.numel() for p in tiny.get_classifier().parameters())
    assert count_params(tiny)[1] == head
    set_trainable(tiny, "lastn", last_n=2)
    lastn = count_params(tiny)[1]
    assert head < lastn < total
    assert all(p.requires_grad for p in tiny.blocks[-1].parameters())
    assert not any(p.requires_grad for p in tiny.blocks[0].parameters())
    set_trainable(tiny, "full")
    assert count_params(tiny)[1] == total


def test_param_groups_split_lr_and_decay(tiny):
    set_trainable(tiny, "lastn", last_n=1)
    groups = param_groups(tiny, lr=1e-3, backbone_lr_scale=0.1, weight_decay=0.05)
    n = sum(len(g["params"]) for g in groups)
    assert n == sum(1 for p in tiny.parameters() if p.requires_grad)
    assert {g["lr"] for g in groups} == {1e-3, 1e-4}
    for g in groups:
        assert g["weight_decay"] == (0.05 if all(p.ndim > 1 for p in g["params"]) else 0.0)


def test_pad_collate_keeps_shape_static():
    collate = pad_collate(8)
    xs, ys, valid = collate([(torch.zeros(3, 4, 4), i) for i in range(3)])
    assert xs.shape == (8, 3, 4, 4) and ys.shape == (8,)
    assert valid.tolist() == [True] * 3 + [False] * 5


def test_loaders_yield_one_shape_only():
    ds = FakeDataset(21, 4, 32)
    train, evl = build_loaders(ds, ds, batch_size=8, workers=0)
    assert {tuple(x.shape) for x, _, _ in train} == {(8, 3, 32, 32)}  # 21 -> 2 full batches
    assert len(train) == 2
    batches = list(evl)
    assert {tuple(x.shape) for x, _, _ in batches} == {(8, 3, 32, 32)}
    assert sum(int(v.sum()) for _, _, v in batches) == 21  # every real sample counted once


def test_train_loss_drops_and_checkpoint_reloads(tmp_path):
    m = main(BASE + ["--mode", "full", "--epochs", "4", "--lr", "2e-3", "--backbone-lr-scale", "0.5",
                     "--out", str(tmp_path)])
    losses = [s["loss"] for s in m["steps"]]
    assert all(torch.isfinite(torch.tensor(losses)))
    assert min(losses[-3:]) < losses[0] - 0.2, losses
    assert json.loads((tmp_path / "metrics.json").read_text())["params_trainable"] == m["params_trainable"]
    ckpt = torch.load(tmp_path / "model.pt", map_location="cpu", weights_only=True)
    fresh = build_model(num_classes=ckpt["num_classes"], pretrained=False)
    fresh.load_state_dict(ckpt["model"])  # keys match: no _orig_mod./module. prefixes


def test_linear_mode_leaves_backbone_untouched(tmp_path):
    torch.manual_seed(0)
    before = build_model(num_classes=4, mode="linear", pretrained=False).state_dict()
    main(BASE + ["--mode", "linear", "--max-steps", "3", "--skip-eval", "--out", str(tmp_path)])
    after = torch.load(tmp_path / "model.pt", weights_only=True)["model"]
    changed = {k for k in before if not torch.equal(before[k], after[k])}
    assert changed and all(k.startswith("head.") for k in changed), changed


def test_compiled_step_runs(tmp_path):
    # Same code path as --device neuron --compile, with PyTorch's aot_eager backend on CPU.
    m = main(BASE + ["--mode", "lastn", "--last-n", "1", "--max-steps", "3", "--compile", "--fullgraph",
                     "--out", str(tmp_path)])
    assert len(m["steps"]) == 3 and "acc" in m["epochs"][-1]
    ckpt = torch.load(tmp_path / "model.pt", weights_only=True)["model"]
    assert not any(k.startswith("_orig_mod.") for k in ckpt)


@pytest.mark.slow
def test_ddp_two_ranks_gloo(tmp_path):
    env = dict(os.environ, PYTHONPATH=str(SRC), OMP_NUM_THREADS="1")
    cmd = [sys.executable, "-m", "torch.distributed.run", "--standalone", "--nproc-per-node=2", "-m",
           "dinoft.train", *BASE, "--mode", "lastn", "--max-steps", "2", "--out", str(tmp_path)]
    # Static argv (sys.executable + fixed flags), no shell, no external input: test-only.
    r = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=600)  # nosec B603  # nosemgrep
    assert r.returncode == 0, r.stderr[-3000:]
    m = json.loads((tmp_path / "metrics.json").read_text())
    assert m["world_size"] == 2 and m["epochs"][-1]["n"] == 32  # eval counted across both ranks


def test_eval_shards_count_every_sample_once():
    # 3669 Pets test images over 4 ranks: DistributedSampler would count 3672.
    from dinoft.data import EvalShard

    ds = [(torch.zeros(1), i) for i in range(3669)]
    shards = [EvalShard(ds, r, 4) for r in range(4)]
    assert len({len(s) for s in shards}) == 1  # equal batch counts on every rank
    seen = [s[i][1] for s in shards for i in range(len(s)) if s[i][2]]
    assert sorted(seen) == list(range(3669))



@pytest.mark.parametrize("compiled", [False, True])
def test_predict_classifies_files_and_evaluates(tmp_path, compiled):
    from PIL import Image

    from dinoft.predict import main as predict

    main(BASE + ["--mode", "lastn", "--last-n", "1", "--max-steps", "2", "--skip-eval", "--out", str(tmp_path)])
    imgs = tmp_path / "imgs"
    imgs.mkdir()
    for i in range(3):  # 3 files with batch 8 -> exercises the padded last batch
        Image.new("RGB", (80, 60), (40 * i, 90, 200)).save(imgs / f"{i}.png")
    args = ["--checkpoint", str(tmp_path / "model.pt"), "--device", "cpu", "--batch-size", "8",
            "--workers", "0", "--images", str(imgs), "--eval", "--top-k", "2", "--out", str(tmp_path / "p.json")]
    rep = predict(args + (["--compile"] if compiled else []))
    assert [Path(p["image"]).name for p in rep["predictions"]] == ["0.png", "1.png", "2.png"]
    assert all(len(p["top"]) == 2 and 0 <= p["top"][0][1] <= 1 for p in rep["predictions"])
    assert rep["eval"]["n"] == 32 and 0 <= rep["eval"]["acc"] <= 1  # fake eval split, padding not counted
    assert json.loads((tmp_path / "p.json").read_text())["eval"]["n"] == 32



class _NotATensor:
    """Stand-in for an arbitrary pickled object an untrusted checkpoint could carry."""


def test_predict_refuses_checkpoints_with_arbitrary_objects(tmp_path):
    from dinoft.predict import load

    bad = tmp_path / "bad.pt"
    torch.save({"model": {}, "args": {}, "num_classes": 1, "payload": _NotATensor()}, bad)
    with pytest.raises(Exception, match="(?i)weights_only|unsupported global"):
        load(str(bad), "cpu", compile_=False)
