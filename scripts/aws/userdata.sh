#!/usr/bin/env bash
# EC2 user data for an unattended Trn2 run of scripts/run_experiments.sh.
#
# Pulls the repo from S3, runs the experiment matrix, streams results back to S3
# and powers off. Launch with --instance-initiated-shutdown-behavior terminate so
# power-off ends billing; the `shutdown -h +MAX_MIN` line is a hard cost backstop.
# launch.sh replaces __S3__ and __MAX_MIN__ before passing this file as user data.
set -uo pipefail
S3=__S3__
MAX_MIN=__MAX_MIN__
S3_REGION=${S3_REGION:-us-east-1}
RUN=/opt/dinoft; LOG=$RUN/results; mkdir -p "$LOG"
exec > >(tee -a "$LOG/runner.log") 2>&1

shutdown -h "+$MAX_MIN" "dinoft safety timeout"
push() { aws s3 sync "$LOG" "$S3/results/" --quiet --region "$S3_REGION" || true; }
trap 'echo "[runner] exiting"; push; shutdown -h now' EXIT
date -u; echo "[runner] start"

aws s3 sync "$S3/code/" "$RUN/" --region "$S3_REGION" --quiet

# Use the DLAMI venv that has native TorchNeuron (torch 2.14 + torch-neuronx).
VENV=""
for v in /opt/aws_neuronx_venv_*; do
  if "$v/bin/python3" -c 'import torch_neuronx, torch; assert torch.__version__.startswith("2.14")' 2>/dev/null; then
    VENV=$v; break
  fi
done
[ -n "$VENV" ] || { echo "[runner] no TorchNeuron venv"; ls -d /opt/aws_neuronx_venv_*; exit 1; }
echo "[runner] venv $VENV"; source "$VENV/bin/activate"

# Exact, tested pins; the venv's own torch/torch-neuronx stay as installed (see the script).
bash "$RUN/scripts/install_neuron_deps.sh" --dev > "$LOG/pip.log" 2>&1 \
  || { echo "[runner] pip failed"; tail -20 "$LOG/pip.log"; exit 1; }
tail -8 "$LOG/pip.log" | tee "$LOG/pip-versions.txt"
pip list --format=freeze > "$LOG/pip-freeze.txt"
push

export HF_HOME=$RUN/hf
cd "$RUN"
( while sleep 60; do push; done ) & PUSHER=$!
MATRIX=__MATRIX__ bash scripts/run_experiments.sh "$LOG/runs"
kill $PUSHER
date -u; echo "[runner] done"
