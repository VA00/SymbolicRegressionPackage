# EML Tree Training (PyTorch v16)

This directory contains the PyTorch `v16` EML tree trainer used for the depth-2 to depth-6 experiments reported in the PNAS Supplementary Information.

Core files:
- `tree_prototype_torch_v16_final.py`: single-run trainer and exporter
- `depth_2_to_6_headless.sh`: headless batch script used for the full sweep
- `requirements.txt`: Python dependencies

## Setup

Run from this directory:

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Running `tree_prototype_torch_v16_final.py`

Most important arguments:
- `--target-fn`: target function (`eml_depth2`, `eml_depth3`, `eml_depth4`, `eml_depth5`, `eml_depth6`)
- `--depth`: tree depth
- `--init-strategy`: `all`, `biased`, `uniform`, `xy_biased`, `random_hot`, or `manual`
- `--init-expr` and `--init-noise`: manual initialization and perturbation
- `--seed0` and `--seeds`: contiguous seed range starting at `seed0`
- `--save-prefix`: output directory and filename stem
- `--skip-plot`: disable live plotting for headless runs

Example: random-start depth-2 benchmark

```bash
python3 tree_prototype_torch_v16_final.py \
  --target-fn eml_depth2 \
  --depth 2 \
  --init-strategy all \
  --seed0 137 \
  --seeds 8 \
  --save-prefix example_d2_random \
  --skip-plot
```

Example: manual depth-6 initialization with noise

```bash
python3 tree_prototype_torch_v16_final.py \
  --target-fn eml_depth6 \
  --depth 6 \
  --init-strategy manual \
  --init-expr "EML[1, EML[EML[EML[1, EML[EML[x, y], 1]], 1], 1]]" \
  --init-noise 12 \
  --seed0 2048 \
  --seeds 4 \
  --search-iters 3000 \
  --hardening-iters 1200 \
  --patience 2000 \
  --eval-every 100 \
  --tail-eval-every 25 \
  --save-prefix example_d6_manual \
  --skip-plot
```

Each run writes:
- `*_stdout_*.log`: human-readable log
- `*_metrics_*.json`: structured summary
- `*.m` and `*_continuous.m`: exported symbolic model
- `*.pt`: PyTorch state dict
- `png/*_loss.png`: loss plot

`--save-prefix NAME` creates a directory `NAME/` and uses `NAME` as the filename stem.

## Full Headless Sweep

The full batch used for the PNAS supplementary package is:

```bash
bash depth_2_to_6_headless.sh
```

Current `pnas_*` jobs in `depth_2_to_6_headless.sh`:
- `pnas_d2_random`: depth 2, `init-strategy all`, `seed0=137`, `seeds=8`
- `pnas_d3_random`: depth 3, `init-strategy all`, `seed0=137`, `seeds=16`
- `pnas_d4_random`: depth 4, `init-strategy all`, `seed0=137`, `seeds=16`
- `pnas_d5_random64_random_hot`: depth 5, `init-strategy random_hot`, `seed0=137`, `seeds=64`
- `pnas_d6_random64_random_hot`: depth 6, `init-strategy random_hot`, `seed0=137`, `seeds=64`
- `pnas_d5_random64_uniform`: depth 5, `init-strategy uniform`, `seed0=1234`, `seeds=64`
- `pnas_d6_random64_uniform`: depth 6, `init-strategy uniform`, `seed0=1234`, `seeds=64`
- `pnas_d5_manual_noise12`: depth 5, manual exact tree + noise `12`, `seed0=2048`, `seeds=4`
- `pnas_d6_manual_noise12`: depth 6, manual exact tree + noise `12`, `seed0=2048`, `seeds=4`

`--seeds N` means the contiguous seed range `seed0, seed0+1, ..., seed0+N-1`.

## Estimated Runtime

The completed March 20, 2026 run in this workspace took `2:25:17` wall-clock time from start to finish.

Per-stage wall times:
- `pnas_d2_random`: about `4m 49s`
- `pnas_d3_random`: about `14m 01s`
- `pnas_d4_random`: about `20m 25s`
- `pnas_d5_random64_random_hot`: about `19m 22s`
- `pnas_d6_random64_random_hot`: about `11m 35s`
- `pnas_d5_random64_uniform`: about `31m 32s`
- `pnas_d6_random64_uniform`: about `43m 14s`
- `pnas_d5_manual_noise12`: about `12s`
- `pnas_d6_manual_noise12`: about `16s`

On comparable hardware, the full script should be expected to take about `2.5 hours`.

## Latest Headless Results

Results below are from the completed March 20, 2026 run with the current seeds listed above.

Random-start runs:
- `pnas_d2_random`: `32/32` successes
- `pnas_d3_random`: `17/64` successes
  `biased 4/16`, `uniform 4/16`, `xy_biased 4/16`, `random_hot 5/16`
- `pnas_d4_random`: `15/64` successes
  `biased 2/16`, `uniform 3/16`, `xy_biased 4/16`, `random_hot 6/16`
  `1` run was abandoned after reaching the maximum number of NaN restarts
- `pnas_d5_random64_random_hot`: `1/64` successes
  `30` runs were abandoned after reaching the maximum number of NaN restarts
- `pnas_d6_random64_random_hot`: `0/64` successes
  `56` runs were abandoned after reaching the maximum number of NaN restarts
- `pnas_d5_random64_uniform`: `1/64` successes
- `pnas_d6_random64_uniform`: `0/64` successes

Manual-initialization runs:
- `pnas_d5_manual_noise12`: `4/4` successes and `4/4` stable symbolic successes
- `pnas_d6_manual_noise12`: `4/4` exact symbolic fits, but `0/4` stable symbolic successes
  each run still had `1-2` uncertain weights at `snap_threshold=0.01`

Short interpretation:
- Depth 2 is easy from random starts.
- Depth 3 and depth 4 are partly recoverable from random starts under this budget.
- Depth 5 random discovery is rare but nonzero for both `random_hot` and `uniform`.
- Depth 6 random discovery was not observed in either 64-seed family under this budget.
- Known-tree initialization remains robust at depths 5 and 6.
