# Measured results (trn2.3xlarge, Neuron SDK 2.33, 2026-11-06)

Raw `metrics.json` for every run behind the numbers in the top-level README, from two separate instances on the same day.

- `matrix1/`: `MATRIX=full` from `scripts/run_experiments.sh`, run on the first instance.
- `verify/`: `MATRIX=verify`, run on a second, fresh instance. It added Dynamo capture stats to
  `metrics.json`, ViT-B/16, a DDP throughput run without eval, and the fixed DDP eval sharding.
- `check/`: `MATRIX=check` on a third instance, after the security fixes. It used the
  `scripts/install_neuron_deps.sh` pins (`pip-freeze.txt`), ran the CPU test suite inside the
  Neuron venv (`pytest.log`: 12 passed), ran a 3-epoch ViT-S/16 fine-tune (92.86 %), then ran
  `dinoft.predict` on that checkpoint with `weights_only=True`: 92.86 % on all 3,669 test images
  and the correct breed for all 4 sample images. Inference throughput was 302 images/s eager and
  737 images/s compiled, on 1 NeuronCore at batch 32.
- `pip-versions.txt`: the package versions in the Neuron DLAMI venv after `pip install`
  (first two instances).

Notes for reading the tables:

- `train img/s` covers the whole run after the first 3 steps. For runs **with** evaluation it
  includes eval time and the one-time eval-graph compile. Use the `--skip-eval` runs
  (`*_1core`, `verify_*_s`, `verify_compile_b`, `verify_ddp4_speed`, `*_bf16_*`) for throughput.
- `matrix1/compile_lastn_ddp4.json` was measured before the eval-sharding fix and counted 3,672
  test images, 3 of them duplicates. `verify/verify_ddp4_acc.json` is the corrected run (3,669).
- `cpu_lastn` ran on the host CPU of the same trn2.3xlarge, for reference only.
