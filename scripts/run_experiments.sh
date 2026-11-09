#!/usr/bin/env bash
# Run the experiment matrix behind results/ on ONE Trn2 host (trn2.3xlarge = 4 logical NeuronCores).
#
# Run inside the Neuron DLAMI's TorchNeuron venv (torch 2.14 + torch-neuronx), from the
# repo root:   bash scripts/run_experiments.sh  [results_dir]
# Every run writes runs/<name>/metrics.json + a log; scripts/summarize.py tabulates them.
set -uo pipefail
OUT=${1:-runs}
EPOCHS=${EPOCHS:-3}
MODEL=${MODEL:-vit_small_patch16_dinov3.lvd1689m}
mkdir -p "$OUT"
export PYTHONPATH=$PWD/src NEURON_LOGICAL_NC_CONFIG=2 OMP_NUM_THREADS=1
# Keep compiled graphs across runs (default is /tmp, which a reboot clears).
export TORCH_NEURONX_NEFF_CACHE_DIR=${TORCH_NEURONX_NEFF_CACHE_DIR:-$PWD/.neuron-cache/neff}
export TORCH_NEURONX_HLO_CACHE_DIR=${TORCH_NEURONX_HLO_CACHE_DIR:-$PWD/.neuron-cache/hlo}

run() {  # run <name> <args...>
  local name=$1; shift
  echo "=== $(date -u +%T) $name: $*"
  "$@" --model "$MODEL" --out "$OUT/$name" > "$OUT/$name.log" 2>&1
  local rc=$?; echo "rc=$rc" >> "$OUT/$name.log"; echo "    rc=$rc"; tail -3 "$OUT/$name.log"
}
one() { run "$1" python3 -m dinoft.train "${@:2}"; }
ddp() { run "$1" torchrun --standalone --nproc-per-node="$2" -m dinoft.train "${@:3}"; }

# Fetch the dataset once so the timed runs do not include the download.
python3 -c "from datasets import load_dataset; load_dataset('timm/oxford-iiit-pet')"

if [ "${MATRIX:-full}" = "check" ]; then
  # Release check: CPU test suite in the Neuron venv, one fine-tune, then inference with it.
  python3 -m pytest -q -p no:cacheprovider > "$OUT/pytest.log" 2>&1; echo "rc=$?" >> "$OUT/pytest.log"
  tail -3 "$OUT/pytest.log"
  NEURON_RT_VISIBLE_CORES=0 one check_train --device neuron --mode lastn --epochs 3 --compile
  python3 - "$OUT/samples" <<'EOF'
import os, sys
from datasets import load_dataset
ds = load_dataset("timm/oxford-iiit-pet", split="test"); names = ds.features["label"].names
os.makedirs(sys.argv[1], exist_ok=True)
for i in (0, 500, 1500, 2500):
    r = ds[i]; r["image"].convert("RGB").save(f"{sys.argv[1]}/{names[r['label']]}_{i}.jpg")
EOF
  for c in "" "--compile"; do
    tag=predict${c:+_compiled}
    echo "=== $(date -u +%T) $tag"
    NEURON_RT_VISIBLE_CORES=0 python3 -m dinoft.predict --checkpoint "$OUT/check_train/model.pt" --device neuron \
      --eval --images "$OUT/samples" $c --out "$OUT/$tag.json" > "$OUT/$tag.log" 2>&1
    echo "rc=$?" >> "$OUT/$tag.log"; tail -7 "$OUT/$tag.log"
  done
  exit 0
fi

if [ "${MATRIX:-full}" = "verify" ]; then
  # Short check: eager vs compiled (with Dynamo capture stats) for ViT-S and ViT-B.
  NEURON_RT_VISIBLE_CORES=0 one verify_eager_s --device neuron --mode lastn --max-steps 40 --skip-eval
  NEURON_RT_VISIBLE_CORES=0 one verify_compile_s --device neuron --mode lastn --max-steps 40 --skip-eval --compile
  MODEL=vit_base_patch16_dinov3.lvd1689m NEURON_RT_VISIBLE_CORES=0 \
    one verify_compile_b --device neuron --mode lastn --max-steps 40 --skip-eval --compile
  MODEL=vit_base_patch16_dinov3.lvd1689m NEURON_RT_VISIBLE_CORES=0 \
    one verify_compile_b_acc --device neuron --mode lastn --epochs "$EPOCHS" --compile
  # DDP over 4 cores: throughput without eval, then accuracy with the fixed eval sharding.
  ddp verify_ddp4_speed 4 --device neuron --mode lastn --max-steps 40 --skip-eval --compile
  ddp verify_ddp4_acc 4 --device neuron --mode lastn --epochs "$EPOCHS" --compile
  # Linear probe with a 10x higher head LR (a frozen backbone tolerates it).
  NEURON_RT_VISIBLE_CORES=0 one verify_linear_lr1e-2 --device neuron --mode linear --epochs "$EPOCHS" --compile --lr 1e-2
  python3 scripts/summarize.py "$OUT" | tee "$OUT/summary.md"
  exit 0
fi

# 1. Eager on one NeuronCore: correctness + which ops fall back to CPU.
NEURON_RT_VISIBLE_CORES=0 one eager_lastn_1core --device neuron --mode lastn --max-steps 40 --skip-eval
# 2. Same, compiled: the one-line torch.compile change.
NEURON_RT_VISIBLE_CORES=0 one compile_lastn_1core --device neuron --mode lastn --max-steps 40 --skip-eval --compile
# 3. Accuracy runs (compiled, one core): linear probe vs last-2-blocks vs full fine-tune.
NEURON_RT_VISIBLE_CORES=0 one compile_linear_1core --device neuron --mode linear --epochs "$EPOCHS" --compile
NEURON_RT_VISIBLE_CORES=0 one compile_lastn_1core_acc --device neuron --mode lastn --epochs "$EPOCHS" --compile
NEURON_RT_VISIBLE_CORES=0 one compile_full_1core --device neuron --mode full --epochs "$EPOCHS" --compile --lr 5e-4
# 4. Data parallel over all 4 logical cores (global batch 4 x 32 = 128).
ddp compile_lastn_ddp4 4 --device neuron --mode lastn --epochs "$EPOCHS" --compile
# 5. bf16 autocast, compiled, one core.
NEURON_RT_VISIBLE_CORES=0 one compile_lastn_bf16_1core --device neuron --mode lastn --max-steps 40 --skip-eval --compile --bf16
# 6. Host-CPU baseline for the same step (few steps; it is slow).
one cpu_lastn --device cpu --mode lastn --max-steps 8 --log-every 2 --skip-eval --workers 2

PATH=$PATH:/opt/aws/neuron/bin neuron-ls > "$OUT/neuron-ls.txt" 2>&1 || true
python3 -c 'import torch, torch_neuronx, timm; print(torch.__version__, torch_neuronx.__version__, timm.__version__)' \
  > "$OUT/versions.txt" 2>&1 || true
python3 scripts/summarize.py "$OUT" | tee "$OUT/summary.md"
