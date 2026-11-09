#!/usr/bin/env bash
# Install this project's dependencies into an ACTIVATED Neuron DLAMI TorchNeuron venv
# (e.g. /opt/aws_neuronx_venv_torch_neuronx_2_14_0) without replacing its PyTorch.
#
#   source /opt/aws_neuronx_venv_torch_neuronx_2_14_0/bin/activate
#   bash scripts/install_neuron_deps.sh            # add --dev to also install pytest
#
# Every package we add is pinned to the exact version this repo was tested with. Every other
# package already in the venv (torch, torch-neuronx, neuronx-cc, numpy, ...) is constrained to
# the version that is installed, so pip can never swap the Neuron build of torch for a stock one.
set -euo pipefail

PINS=(timm==1.0.31 datasets==5.1.0 torchvision==0.29.2 pillow==13.0.0)
[ "${1:-}" = "--dev" ] && PINS+=(pytest==9.2.0)

python3 -c 'import torch_neuronx' 2>/dev/null || {
  echo "error: activate the Neuron TorchNeuron venv first (torch_neuronx not importable)" >&2; exit 1; }

names_re=$(printf '%s\n' "${PINS[@]}" | sed 's/==.*//' | paste -sd'|')
constraints=$(mktemp)
trap 'rm -f "$constraints"' EXIT
# Pin what is already installed, except the packages we pin explicitly above.
pip list --format=freeze | grep -viE "^(${names_re})==" > "$constraints"

pip install -c "$constraints" "${PINS[@]}"
pip list 2>/dev/null | grep -iE '^(torch|torch-neuronx|neuronx-cc|torchvision|timm|datasets|pillow|pytest) '
