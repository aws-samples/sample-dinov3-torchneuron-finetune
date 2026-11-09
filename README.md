# Fine-tune DINOv3 on AWS Trainium with plain PyTorch + TorchNeuron

A small, readable reference for fine-tuning Meta's [DINOv3](https://github.com/facebookresearch/dinov3)
vision backbone on AWS Trainium2 with **TorchNeuron**, the native PyTorch backend for Trainium
(public beta in Neuron SDK 2.33). The training loop is the one you would write for a GPU. The
device string is `"neuron"` instead of `"cuda"`, and that is the only Neuron-specific line.

Measured on one `trn2.3xlarge` (Neuron SDK 2.33). Each training run is 3 epochs on Oxford-IIIT Pet
(37 cat and dog breeds):

| | result |
|---|---|
| ViT-B/16, last 2 blocks fine-tuned, 1 NeuronCore | **96.0 %** test accuracy |
| ViT-S/16, last 2 blocks fine-tuned, 1 NeuronCore | **92.9 %** test accuracy |
| ViT-S/16 training throughput, eager vs `torch.compile` | 205-225 vs **408-438** images/s |
| ViT-S/16, compiled + bf16 autocast, 1 NeuronCore | **640** images/s |
| ViT-S/16, compiled, DDP over 4 NeuronCores | **1,163** images/s |
| ops that fell back to the CPU, any run | **none** |

The raw `metrics.json` for every run is in [`results/`](results/).

What the code shows:

- three fine-tuning strategies: **linear probe**, **last-N blocks**, **full fine-tune**
- **eager mode** first, then **`torch.compile(backend="neuron")`** as a one-line change
- **DDP** across the NeuronCores of one instance with `torchrun`
- how to keep shapes **static** (Neuron compiles one graph per shape) and how to check which ops
  **fall back to the CPU**
- **inference** with the fine-tuned checkpoint: classify your own images, re-measure accuracy
- a CPU test suite that runs anywhere, plus an optional unattended, self-terminating Trn2 runner

## Contents

1. [Prerequisites](#1-prerequisites) and [Cost estimate](#cost-estimate)
2. [Get a Trn2 instance](#2-get-a-trn2-instance)
3. [Connect and set up the environment](#3-connect-and-set-up-the-environment)
4. [Clone the code and run the tests](#4-clone-the-code-and-run-the-tests)
5. [Run fine-tuning](#5-run-fine-tuning)
6. [Run inference tests](#6-run-inference-tests)
7. [Reproduce all the numbers](#7-reproduce-all-the-numbers-optional)
8. [Clean up](#8-clean-up)
9. [Develop off Trainium](#9-develop-off-trainium) / [Layout](#layout) / [Flags](#flags) /
   [Weights and licenses](#weights-and-licenses) / [Citation](#citation) / [Known limits](#known-limits-torchneuron-public-beta)

## 1. Prerequisites

- An AWS account and the [AWS CLI v2](https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html)
  configured with credentials that can launch EC2 instances.
- EC2 quota for Trainium: **Running On-Demand Trn instances** must be at least the instance's vCPU
  count (12 for `trn2.3xlarge`). Check and request it in Service Quotas, see
  [Amazon EC2 service quotas](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/ec2-resource-limits.html).
- A network path from the instance to the internet. The scripts download the DINOv3 weights and the
  dataset from the Hugging Face Hub, plus a few pip packages. A public subnet works. In a private
  subnet you need a [NAT gateway](https://docs.aws.amazon.com/vpc/latest/userguide/vpc-nat-gateway.html).

See [Cost estimate](#cost-estimate) below before you launch anything.

### Cost estimate

All prices are for one `trn2.3xlarge` (Linux) in **South America (São Paulo), `sa-east-1`**, as
listed on 2026-11-06. They change over time and differ by Region, so check the links before you
rely on them.

**Unit prices**

| item | price | source |
|---|---|---|
| `trn2.3xlarge` via EC2 Capacity Block | **$2.235 / instance-hour**, paid up front for the whole block | [Capacity Blocks pricing](https://aws.amazon.com/ec2/capacityblocks/pricing/) |
| Capacity Block operating-system fee (Linux / Ubuntu DLAMI) | $0.00 / hour | same page |
| `trn2.3xlarge` On-Demand | not in the public Price List on 2026-11-06; check the [EC2 console](https://console.aws.amazon.com/ec2/) or [AWS Pricing Calculator](https://calculator.aws/) | |
| 200 GB gp3 root volume | $0.152 / GB-month, about **$0.04 / hour** | [EBS pricing](https://aws.amazon.com/ebs/pricing/) |
| NAT gateway, only if you use a private subnet | $0.093 / hour + $0.093 / GB processed | [VPC pricing](https://aws.amazon.com/vpc/pricing/) |

**How long each step takes.** These are measured wall-clock times from our runs, with a cold
compile cache unless noted:

| step | time on the instance | instance-hours | compute at $2.235/h |
|---|---|---|---|
| Boot, connect, `pip install`, dataset download (sections 2-3) | ~8 min | 0.13 | $0.30 |
| CPU test suite + 40-step NeuronCore smoke test (section 4) | ~5 min | 0.08 | $0.19 |
| One 3-epoch ViT-S/16 fine-tune, cold cache (section 5) | ~5-6 min | 0.10 | $0.22 |
| The same run with a warm cache | ~1 min | 0.02 | $0.04 |
| One 3-epoch ViT-B/16 fine-tune, cold cache | ~5 min | 0.08 | $0.19 |
| Inference: test-set eval + a few images (section 6) | ~2-3 min | 0.05 | $0.11 |
| Full experiment matrix, `MATRIX=full` (section 7) | 22 min | 0.37 | $0.82 |
| Verification matrix, `MATRIX=verify` | 24 min | 0.40 | $0.89 |

So the **compute** for the whole walkthrough (sections 2-6, one ViT-S and one ViT-B fine-tune) is
about **30-45 instance-minutes, $1.10-$1.70**. Running both experiment matrices as well adds about
$1.70.

**What you actually pay depends on how you get the instance:**

- **Capacity Block** (how the numbers in this repo were measured): you pay for the **whole block up
  front**, however little of it you use, and it cannot be cancelled. Blocks end at 11:30 UTC, and
  the first day can be shorter than 24 hours. Our 17.7-hour block cost **$39.52** (17.7 h x
  $2.235). A 24-hour block was quoted at **$53.64**. One block covers the whole walkthrough many
  times over, so batch your experiments into it.
- **On-Demand**, if your account gets capacity: you pay per second from launch to termination, so
  the cost is close to the instance-hours above times the On-Demand rate. Terminate the instance
  as soon as you finish.

**Small extras:**

- **Storage:** 200 GB gp3 for 1 hour is about $0.04, deleted with the instance.
- **NAT gateway:** only if you use one. It costs $0.093/hour while it exists, plus about $0.15 for
  the ~1.5 GB this project downloads. **Delete it when you are done**, because it keeps billing
  after the instance is gone.
- **S3:** negligible. `scripts/aws/launch.sh` stores less than 1 MB of code and results.

**Example, our own bill for this repo:** one Capacity Block ($39.52) covered both experiment matrices
on two instances (about 1 instance-hour in total). EBS added about $0.05. A NAT gateway that already
existed added its hourly rate.

## 2. Get a Trn2 instance

Any Trn2 size works. Everything here was measured on **`trn2.3xlarge`**: one Trainium2 device,
4 logical NeuronCores, 96 GB device memory. TorchNeuron requires Trn2 or Trn3. It does not run on
Inf2 or Trn1.

**2a. Find a Region and Availability Zone that offer it.** Availability differs by Region and
account:

```bash
REGION=sa-east-1      # pick a Region from the Trn2 page and change as needed
aws ec2 describe-instance-type-offerings --region $REGION --location-type availability-zone \
  --filters Name=instance-type,Values=trn2.3xlarge --query 'InstanceTypeOfferings[].Location'
```

**2b. Pick the AMI.** Use the **Neuron multi-framework Deep Learning AMI (Ubuntu 24.04)**. It ships
the Neuron driver, runtime and compiler, plus a TorchNeuron virtual environment at
`/opt/aws_neuronx_venv_torch_neuronx_2_14_0`. Look up the latest AMI ID through its public SSM
parameter (see [Neuron DLAMI](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/deploy/environments/dlami.html)
and [Finding an AMI with Parameter Store](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/finding-an-ami-parameter-store.html)):

```bash
AMI=$(aws ssm get-parameter --region $REGION \
  --name /aws/service/neuron/dlami/multi-framework/ubuntu-24.04/latest/image_id \
  --query Parameter.Value --output text)
echo $AMI
```

**2c. Launch it.** You can use the console, following
[Launch an instance using the launch instance wizard](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/ec2-launch-instance-wizard.html):
choose the AMI above, instance type `trn2.3xlarge`, a key pair or an instance profile for Session
Manager, and a **200 GB gp3** root volume. You can also use the CLI:

```bash
aws ec2 run-instances --region $REGION --image-id $AMI --instance-type trn2.3xlarge \
  --subnet-id <subnet-id> --security-group-ids <sg-id> \
  --key-name <key-pair-name> \
  --block-device-mappings 'DeviceName=/dev/sda1,Ebs={VolumeSize=200,VolumeType=gp3,DeleteOnTermination=true}' \
  --metadata-options HttpTokens=required \
  --tag-specifications 'ResourceType=instance,Tags=[{Key=Name,Value=dinov3-torchneuron}]' \
  --query 'Instances[0].InstanceId' --output text
```

Use `--iam-instance-profile Name=<profile>` instead of `--key-name` if you connect with Session
Manager. See [Create a key pair](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/ec2-key-pairs.html).

**If you get `InsufficientInstanceCapacity`,** try another Availability Zone (another subnet).
You can also reserve capacity with an
[EC2 Capacity Block for ML](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/ec2-capacity-blocks.html):
find and buy one following
[Find and purchase Capacity Blocks](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/capacity-blocks-purchase.html),
then add these flags to the `run-instances` command above and use a subnet in the block's
Availability Zone:

```bash
  --instance-market-options MarketType=capacity-block \
  --capacity-reservation-specification 'CapacityReservationTarget={CapacityReservationId=<cr-id>}'
```

A Capacity Block is paid up front and cannot be cancelled. It is a good fit when you want a fixed
window. The numbers in this repo were measured on one.

## 3. Connect and set up the environment

Connect over SSH with your key pair, following
[Connect to your Linux instance](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/connect-to-linux-instance.html).
The user is `ubuntu`. You can also use
[Session Manager](https://docs.aws.amazon.com/systems-manager/latest/userguide/session-manager.html):

```bash
ssh -i <key>.pem ubuntu@<instance-public-dns>
# or: aws ssm start-session --region $REGION --target <instance-id>   (then: sudo -iu ubuntu)
```

On the instance, activate the TorchNeuron environment and check the hardware:

```bash
source /opt/aws_neuronx_venv_torch_neuronx_2_14_0/bin/activate
neuron-ls                         # expect 1 device, 4 cores (trn2.3xlarge, LNC=2)
python -c "import torch, torch_neuronx; print(torch.__version__, torch_neuronx.__version__)"
```

Keep this venv active for every command below. You install the project's own dependencies into it
in the next section, right after cloning.

## 4. Clone the code and run the tests

```bash
git clone https://github.com/aws-samples/sample-dinov3-torchneuron-finetune.git
cd sample-dinov3-torchneuron-finetune
bash scripts/install_neuron_deps.sh --dev     # --dev adds pytest
export PYTHONPATH=$PWD/src
```

`scripts/install_neuron_deps.sh` installs exact, tested versions (timm 1.0.31, datasets 5.1.0,
torchvision 0.29.2, pillow 13.0.0, and pytest 9.2.0 with `--dev`) **into the existing venv
without touching its PyTorch**. Every package the DLAMI already provides, including torch 2.14.0,
torch-neuronx 2.14.3 and neuronx-cc 2.28, is constrained to its installed version.

> Never `pip install torch` into this venv. It would replace the Neuron build of PyTorch.

Optional, but it makes re-runs start in seconds: keep the Neuron compile caches on disk instead of
the default `/tmp`:

```bash
export TORCH_NEURONX_NEFF_CACHE_DIR=$PWD/.neuron-cache/neff
export TORCH_NEURONX_HLO_CACHE_DIR=$PWD/.neuron-cache/hlo
```

**Run the test suite.** The tests run on the CPU (`--device cpu`, a randomly initialised model and
a synthetic dataset), so they need no download and no NeuronCore. They check freezing, static
shapes, eval counting, checkpoint round-trips, a compiled training step (PyTorch's `aot_eager`
backend), 2-rank DDP over `gloo`, and inference:

```bash
python -m pytest            # 12 tests, about 1-2 minutes
```

**Smoke-test on the NeuronCore** (about 3 minutes, most of it a one-time compile):

```bash
python -m dinoft.train --device neuron --mode lastn --max-steps 40 --skip-eval --out runs/smoke
```

Look for these lines:

```text
[neuron] CPU fallback ops after step 1: none
step    39 | ep 0 | loss 2.8655 | ...
[perf] ~205-225 train images/s
```

## 5. Run fine-tuning

The default run unfreezes the last 2 blocks of DINOv3 ViT-S/16 and trains for 3 epochs on Oxford-IIIT
Pet. The first run downloads the weights (about 90 MB for ViT-S, 340 MB for ViT-B) and the dataset
(about 800 MB).

```bash
# 1 NeuronCore, compiled. About 2-4 minutes on a cold cache. Expect ~92.9 % test accuracy.
NEURON_RT_VISIBLE_CORES=0 python -m dinoft.train --device neuron --mode lastn --epochs 3 \
  --compile --out runs/vits_lastn

# Larger backbone (ViT-B/16, 86M params). Expect ~96.0 %.
NEURON_RT_VISIBLE_CORES=0 python -m dinoft.train --device neuron --mode lastn --epochs 3 \
  --compile --model vit_base_patch16_dinov3.lvd1689m --out runs/vitb_lastn

# Other strategies
python -m dinoft.train --device neuron --mode linear --lr 1e-2 --epochs 3 --compile --out runs/vits_linear  # ~87.4 %
python -m dinoft.train --device neuron --mode full --lr 5e-4 --epochs 3 --compile --out runs/vits_full      # ~89.4 %

# Eager mode (no --compile): same results, about half the throughput
python -m dinoft.train --device neuron --mode lastn --epochs 3 --out runs/vits_lastn_eager

# bf16 autocast for the forward pass (fp32 master weights)
python -m dinoft.train --device neuron --mode lastn --epochs 3 --compile --bf16 --out runs/vits_lastn_bf16

# Data parallel over all 4 logical NeuronCores (global batch 4 x 32 = 128)
torchrun --standalone --nproc-per-node=4 -m dinoft.train --device neuron --mode lastn --epochs 3 \
  --compile --out runs/vits_lastn_ddp4
```

Each run prints the loss every 10 steps, test accuracy after every epoch, the CPU-fallback report
and throughput. It writes two files:

- `runs/<name>/metrics.json`: config, versions, loss curve, accuracy, images/s, first-step time,
  CPU-fallback ops, and for `--compile` the Dynamo capture stats and graph breaks
- `runs/<name>/model.pt`: a plain `state_dict` with no `_orig_mod.` / `module.` prefixes, plus the
  class names

Tabulate any set of runs with `python scripts/summarize.py runs`.

Notes:

- **The first step of every new shape compiles** (1-3 minutes). That covers the first training
  step and the first evaluation pass. Later steps and re-runs reuse the cache.
- With DDP the global batch is 4x larger, so the same 3 epochs are 4x fewer optimizer steps.
  Scale `--lr` or `--epochs` accordingly.
- The training throughput printed by a run **with** evaluation includes evaluation time. Add
  `--skip-eval --max-steps 40` for clean speed measurements.

## 6. Run inference tests

`dinoft.predict` loads a `model.pt` and runs inference on the NeuronCore with the same static-shape
rules as training. Images are resized to the training size and batches are padded to
`--batch-size`.

**Re-measure accuracy and inference throughput on the test set:**

```bash
python -m dinoft.predict --checkpoint runs/vits_lastn/model.pt --eval --compile
# [eval] acc <accuracy> on 3669 images | <N> images/s after warm-up | total <T>s
```

**Classify your own images** (files or folders; prints the top-k breeds per image). To try it
without your own photos, save a few test images first:

```bash
python - <<'EOF'
from datasets import load_dataset
ds = load_dataset("timm/oxford-iiit-pet", split="test")
names = ds.features["label"].names
import os; os.makedirs("samples", exist_ok=True)
for i in (0, 500, 1500, 2500):
    r = ds[i]; r["image"].convert("RGB").save(f"samples/{names[r['label']]}_{i}.jpg")
EOF

python -m dinoft.predict --checkpoint runs/vits_lastn/model.pt --images samples/ --top-k 3 \
  --out runs/vits_lastn/predictions.json
# samples/newfoundland_0.jpg: <breed> <probability>, <breed> <probability>, ...
```

The file name holds the true label, so you can check each prediction by eye. Add `--bf16` for
bfloat16 inference, or `--device cpu` to run the same checkpoint without Trainium.

## 7. Reproduce all the numbers (optional)

`scripts/run_experiments.sh` runs the whole matrix behind `results/` in about 25 minutes and writes
`runs/summary.md`:

```bash
bash scripts/run_experiments.sh runs                 # MATRIX=full (default)
MATRIX=verify bash scripts/run_experiments.sh runs   # eager vs compile, ViT-B, DDP speed
```

To run it **unattended** instead, `scripts/aws/launch.sh` starts an instance with
`scripts/aws/userdata.sh` as user data. The instance pulls the code from S3, runs the matrix,
streams the results back to S3, and **terminates itself** when done (with a hard timeout as a cost
backstop). The script header lists the required variables (`S3`, `SUBNET`, `PROFILE`, and
optionally `CAPACITY_RESERVATION`). The instance profile needs read access to `$S3/code/*` and write
access to `$S3/results/*`.

## 8. Clean up

On the instance, if you plan to keep using it:

```bash
rm -rf runs/ samples/ .neuron-cache/                  # checkpoints, results, compile caches
rm -rf ~/.cache/huggingface                           # downloaded weights and dataset (~1 GB)
rm -rf /tmp/neff_cache /tmp/hlo_cache /tmp/local_cache  # default compile caches, if you did not move them
```

When you are done, **terminate the instance** to stop billing. Copy anything you want to keep first,
for example `scp -r ubuntu@<host>:sample-dinov3-torchneuron-finetune/runs .`:

```bash
aws ec2 terminate-instances --region $REGION --instance-ids <instance-id>
aws ec2 describe-instances --region $REGION --instance-ids <instance-id> \
  --query 'Reservations[0].Instances[0].State.Name'          # expect "terminated"
```

See [Terminate Amazon EC2 instances](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/terminating-instances.html).
The root volume is deleted with the instance when `DeleteOnTermination=true`, as in the launch
command above. Also remove anything you created only for this walkthrough:

- **Capacity Block**: it cannot be cancelled and ends on its own. AWS starts terminating its
  instances at 11:00 UTC on the block's final day, see
  [Use Capacity Blocks](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/capacity-blocks-using.html).
- **NAT gateway / Elastic IP**: if you created them for a private subnet, delete the NAT gateway and
  then release its Elastic IP. Both are billed by the hour.
- **Key pair, security group, instance profile / IAM role**: delete them if they were created only
  for this.
- **S3**: if you used `scripts/aws/launch.sh`, delete the `code/` and `results/` prefixes you
  uploaded to.

## 9. Develop off Trainium

On a laptop or in CI, with [uv](https://docs.astral.sh/uv/) (installs CPU-only torch 2.14, the same
minor version as Neuron 2.33):

```bash
uv sync --extra cpu --extra dev
uv run pytest
uv run python -m dinoft.train --device cpu --dataset fake --no-pretrained --max-steps 5
```

## Layout

| path | what |
|---|---|
| `src/dinoft/model.py` | DINOv3 from timm + head; `linear` / `lastn` / `full` freezing; param groups |
| `src/dinoft/data.py` | Oxford-IIIT Pet loaders with static shapes (drop-last train, padded + masked eval, `EvalShard` for DDP) |
| `src/dinoft/train.py` | the training loop: eager / compiled, DDP, CPU-fallback report, metrics, checkpoint |
| `src/dinoft/predict.py` | inference: classify image files, re-measure test accuracy and images/s |
| `tests/test_cpu.py` | CPU tests (compiled step via `aot_eager`, 2-rank DDP via `gloo`, inference) |
| `scripts/run_experiments.sh` | the experiment matrix behind `results/` |
| `scripts/summarize.py` | turns `runs/*/metrics.json` into a Markdown table |
| `scripts/aws/launch.sh`, `userdata.sh` | launch one self-terminating Trn2 instance that runs the matrix |
| `results/` | measured `metrics.json` for every run, with notes |

## Flags

`python -m dinoft.train --help` lists them all. The main ones:

| flag | default | notes |
|---|---|---|
| `--mode {linear,lastn,full}` | `lastn` | `--last-n 2` blocks unfrozen in `lastn` |
| `--model` | `vit_small_patch16_dinov3.lvd1689m` | any timm DINOv3 ViT, e.g. `vit_base_patch16_dinov3.lvd1689m` |
| `--compile` / `--fullgraph` | off | `torch.compile(backend="neuron", dynamic=False)`; `--fullgraph` turns graph breaks into errors (we measured 0 breaks on one core and 7 under DDP) |
| `--bf16` | off | bfloat16 autocast for the forward pass, fp32 master weights |
| `--batch-size` | 32 | per worker; the global batch is `batch-size x workers` |
| `--lr`, `--backbone-lr-scale` | 1e-3, 0.1 | head LR; unfrozen backbone params use `lr x scale` |
| `--image-size` | 224 | fixed; 224 / 16 = 14 x 14 patches |
| `--max-steps`, `--skip-eval` | | for quick timing runs |
| `--train-limit`, `--eval-limit` | | use a subset of the data |

## Weights and licenses

The code in this repository is MIT-0 (MIT No Attribution), see [`LICENSE`](LICENSE). The model
weights and the dataset are **not** part of this repository and are **not** covered by that
license. Your machine downloads them from the Hugging Face Hub the first time you run the code,
under their own terms.

**DINOv3 weights: [DINOv3 License](https://ai.meta.com/resources/models-and-libraries/dinov3-license)**
(Meta, last updated 2025-08-19).

- Meta's own `facebook/dinov3-*` repositories on Hugging Face are gated. This project loads timm's
  ungated conversions, `timm/vit_*_patch16_dinov3.lvd1689m`, which ship the same DINOv3 License.
  They are converted, not byte-identical: timm drops the all-zero QKV biases and keeps the RoPE
  periods in float32, see the [model card](https://huggingface.co/timm/vit_small_patch16_dinov3.lvd1689m).
- Downloading or using the weights means you accept the DINOv3 License, so read it first. In short:
  - it grants a royalty-free license to use, modify and redistribute the weights, including for
    commercial use;
  - a model you fine-tune from them (for example the `model.pt` files this code writes) is a
    derivative work. If you share it with anyone, you must share it under the DINOv3 License and
    include a copy of that license;
  - if you publish results obtained with DINOv3, you must acknowledge its use (see [Citation](#citation));
  - use must comply with trade-control laws, and use for military or warfare, nuclear, espionage or
    weapons purposes is prohibited.
- This repository does not contain or redistribute any DINOv3 weights. `*.pt` checkpoints are
  git-ignored.

**Dataset: [Oxford-IIIT Pet](https://www.robots.ox.ac.uk/~vgg/data/pets/)**, by Parkhi, Vedaldi,
Zisserman and Jawahar, licensed under
[CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/). The copyright of each image stays
with its original owner. It is loaded from the
[`timm/oxford-iiit-pet`](https://huggingface.co/datasets/timm/oxford-iiit-pet) copy. This repository
contains no images from it. The sample images in section 6 are saved locally when you run the
commands.

**Libraries:** [PyTorch](https://pytorch.org/) (BSD-3-Clause),
[timm](https://github.com/huggingface/pytorch-image-models) (Apache-2.0), Hugging Face
[datasets](https://github.com/huggingface/datasets) (Apache-2.0), and the AWS Neuron SDK
([torch-neuronx](https://github.com/aws-neuron/torch-neuronx), Apache-2.0) preinstalled in the Neuron DLAMI.

## Citation

If you publish results obtained with this code, cite DINOv3. The DINOv3 License requires you to
acknowledge its use. Also cite the dataset and timm:

```bibtex
@misc{simeoni2025dinov3,
  title={{DINOv3}},
  author={Sim{\'e}oni, Oriane and Vo, Huy V. and Seitzer, Maximilian and Baldassarre, Federico and Oquab, Maxime and Jose, Cijo and Khalidov, Vasil and Szafraniec, Marc and Yi, Seungeun and Ramamonjisoa, Micha{\"e}l and Massa, Francisco and Haziza, Daniel and Wehrstedt, Luca and Wang, Jianyuan and Darcet, Timoth{\'e}e and Moutakanni, Th{\'e}o and Sentana, Leonel and Roberts, Claire and Vedaldi, Andrea and Tolan, Jamie and Brandt, John and Couprie, Camille and Mairal, Julien and J{\'e}gou, Herv{\'e} and Labatut, Patrick and Bojanowski, Piotr},
  year={2025},
  eprint={2508.10104},
  archivePrefix={arXiv},
  primaryClass={cs.CV},
  url={https://arxiv.org/abs/2508.10104},
}

@inproceedings{parkhi12a,
  author={Omkar M. Parkhi and Andrea Vedaldi and Andrew Zisserman and C. V. Jawahar},
  title={Cats and Dogs},
  booktitle={IEEE Conference on Computer Vision and Pattern Recognition},
  year={2012},
}

@misc{rw2019timm,
  author={Ross Wightman},
  title={PyTorch Image Models},
  year={2019},
  publisher={GitHub},
  journal={GitHub repository},
  doi={10.5281/zenodo.4414861},
  howpublished={\url{https://github.com/rwightman/pytorch-image-models}},
}
```

The DINOv3 entry is copied from the
[official repository](https://github.com/facebookresearch/dinov3#citing-dinov3), the dataset entry
from its [Hugging Face card](https://huggingface.co/datasets/timm/oxford-iiit-pet), and the timm
entry from the [timm README](https://github.com/huggingface/pytorch-image-models#citing).

## Security

- **Checkpoints**: `dinoft.predict` loads `model.pt` with `torch.load(..., weights_only=True)`,
  so a checkpoint cannot execute code while it is unpickled. `dinoft.train` saves only tensors and
  plain Python values, and a test makes sure such a checkpoint loads while one carrying an arbitrary
  object is refused. Only load checkpoints you trust anyway.
- **Dependencies** are pinned to exact versions (`pyproject.toml`, `uv.lock`,
  `scripts/install_neuron_deps.sh`). Re-check them for advisories before you reuse this code.
- **Launch scripts** require IMDSv2 (`HttpTokens=required`), take every account-specific identifier
  from environment variables, and make the instance terminate itself with a hard timeout.
- If you discover a potential security issue in this project, notify AWS Security through the
  [vulnerability reporting page](https://aws.amazon.com/security/vulnerability-reporting/). Do not
  open a public GitHub issue. See [CONTRIBUTING](CONTRIBUTING.md#security-issue-notifications).

## Contributing

See [CONTRIBUTING](CONTRIBUTING.md) and the [Code of Conduct](CODE_OF_CONDUCT.md).

## Known limits (TorchNeuron public beta)

- Trn2 / Trn3 only. Inf2 and Trn1 use the older XLA-based path.
- Static shapes only (`dynamic=False`); `mode="default"` only.
- Unsupported ops fall back to the CPU silently. `train.py` prints the fallback list after the
  first step, and `TORCH_NEURONX_FALLBACK_ONLY_FOR_UNIMPLEMENTED_OPS=1` makes the rules stricter.
- Under DDP, Dynamo splits the compiled graph: 7 graph breaks from DDP's hooks, against 0 on a
  single core.
- Further reading: [TorchNeuron documentation](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/torch/index.html),
  [quickstart](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/torch/get-started/quickstart-torch-neuron.html).
