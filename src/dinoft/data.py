"""Oxford-IIIT Pet (37 breeds) loaders with STATIC shapes.

Neuron compiles one graph per input shape, so every batch this module yields has
the same shape:

* images are resized/cropped to a fixed ``image_size`` on the CPU;
* training drops the last partial batch (``drop_last=True``);
* evaluation pads the last batch up to ``batch_size`` and returns a ``valid``
  mask, so accuracy only counts the real samples.

``--dataset fake`` gives a tiny synthetic dataset for smoke tests with no download.
"""
from __future__ import annotations

import torch
from torch.utils.data import DataLoader, Dataset, DistributedSampler

MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)
NUM_CLASSES = {"pets": 37}
HF_DATASETS = {"pets": "timm/oxford-iiit-pet"}


def transforms(image_size: int, train: bool):
    from torchvision.transforms import v2 as T

    if train:
        aug = [T.RandomResizedCrop(image_size, scale=(0.35, 1.0), antialias=True), T.RandomHorizontalFlip()]
    else:
        aug = [T.Resize(int(image_size * 256 / 224), antialias=True), T.CenterCrop(image_size)]
    return T.Compose([T.ToImage(), *aug, T.ToDtype(torch.float32, scale=True), T.Normalize(MEAN, STD)])


class HFImageDataset(Dataset):
    def __init__(self, hf_split, tf):
        self.ds, self.tf = hf_split, tf
        self.classes = list(hf_split.features["label"].names)

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, i):
        row = self.ds[i]
        return self.tf(row["image"].convert("RGB")), int(row["label"])


class FakeDataset(Dataset):
    """Class-dependent noise: learnable, so a working loop drives the loss down."""

    def __init__(self, n: int, num_classes: int, image_size: int, seed: int = 0):
        # Class prototypes are shared by every split (fixed seed); only the labels and
        # noise depend on ``seed``, so a model trained on split 0 generalises to split 1.
        self.protos = torch.randn(num_classes, 3, image_size, image_size,
                                  generator=torch.Generator().manual_seed(1234))
        g = torch.Generator().manual_seed(seed)
        self.labels = torch.randint(0, num_classes, (n,), generator=g)
        self.noise = 0.5
        self.seed = seed

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        g = torch.Generator().manual_seed(self.seed * 100003 + i)
        y = int(self.labels[i])
        x = self.protos[y] + self.noise * torch.randn(self.protos[y].shape, generator=g)
        return x, y


class EvalShard(Dataset):
    """Every ``world``-th sample starting at ``rank``, padded so all ranks have equal length.

    Unlike DistributedSampler (which pads with duplicates that would be counted twice),
    padding entries are flagged invalid, so the all-reduced accuracy counts each test
    image exactly once while every rank still runs the same number of batches.
    """

    def __init__(self, ds, rank: int, world: int):
        self.ds, self.idx = ds, list(range(rank, len(ds), world))
        self.length = -(-len(ds) // world)  # ceil

    def __len__(self):
        return self.length

    def __getitem__(self, i):
        if i < len(self.idx):
            return (*self.ds[self.idx[i]], True)
        return (*self.ds[self.idx[-1] if self.idx else 0], False)


def pad_collate(batch_size: int):
    """Stack a batch and pad it to ``batch_size`` (repeat last sample, valid=False).

    Samples are ``(x, y)`` or ``(x, y, valid)``.
    """

    def collate(samples):
        xs = torch.stack([s[0] for s in samples])
        ys = torch.tensor([s[1] for s in samples], dtype=torch.long)
        valid = torch.tensor([bool(s[2]) if len(s) > 2 else True for s in samples], dtype=torch.bool)
        pad = batch_size - len(samples)
        if pad > 0:
            xs = torch.cat([xs, xs[-1:].expand(pad, *xs.shape[1:])])
            ys = torch.cat([ys, ys[-1:].expand(pad)])
            valid = torch.cat([valid, torch.zeros(pad, dtype=torch.bool)])
        return xs, ys, valid

    return collate


def build_datasets(name: str, image_size: int, fake_size: int = 64, train_limit: int | None = None,
                   eval_limit: int | None = None, cache_dir: str | None = None):
    """Return (train_ds, eval_ds, num_classes)."""
    if name == "fake":
        return (FakeDataset(fake_size, 4, image_size, seed=0), FakeDataset(fake_size // 2, 4, image_size, seed=1), 4)
    if name not in HF_DATASETS:
        raise ValueError(f"unknown dataset {name!r}; choose from {['fake', *HF_DATASETS]}")
    from datasets import load_dataset

    ds = load_dataset(HF_DATASETS[name], cache_dir=cache_dir)
    train, test = ds["train"], ds["test"]
    if train_limit:
        train = train.shuffle(seed=0).select(range(min(train_limit, len(train))))
    if eval_limit:
        test = test.shuffle(seed=0).select(range(min(eval_limit, len(test))))
    return (HFImageDataset(train, transforms(image_size, True)),
            HFImageDataset(test, transforms(image_size, False)), NUM_CLASSES[name])


def build_loaders(train_ds, eval_ds, batch_size: int, workers: int = 4, world_size: int = 1, rank: int = 0,
                  seed: int = 0):
    train_sampler = eval_sampler = None
    if world_size > 1:
        train_sampler = DistributedSampler(train_ds, world_size, rank, shuffle=True, seed=seed, drop_last=True)
        eval_ds = EvalShard(eval_ds, rank, world_size)
    common = dict(batch_size=batch_size, num_workers=workers, persistent_workers=workers > 0,
                  collate_fn=pad_collate(batch_size))
    train = DataLoader(train_ds, sampler=train_sampler, shuffle=train_sampler is None, drop_last=True,
                       generator=torch.Generator().manual_seed(seed), **common)
    evl = DataLoader(eval_ds, sampler=eval_sampler, shuffle=False, drop_last=False, **common)
    if len(train) == 0:
        raise ValueError(f"train set ({len(train_ds)} samples) is smaller than one batch ({batch_size} x {world_size})")
    return train, evl
