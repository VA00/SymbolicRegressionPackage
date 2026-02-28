# EML Training Experiments (PyTorch_v16_final)

## Purpose
`PyTorch_v16_final` is a standalone review package (4 files only):
- `tree_prototype_torch_v16_final.py`
- `depth_2_to_6_headless.sh`
- `README.md`
- `requirements.txt`

It keeps the reproducibility style of `v15_final` (single Python command, saved `.m/.pt/.json/.png`) but uses a simpler training flow:

1. Search at high tau (`tau_search`) with Adam.
2. Harden by annealing tau to `tau_hard` with entropy/binarity penalties.
3. Optional short LBFGS polish.
4. Hard projection (snap) and exactness check.

Main improvement: plots show both
- `soft RMSE` (optimized objective), and
- `hard RMSE` at fixed `tau_hard` (discrete diagnostic),
so convergence is easier to interpret.

Default behavior in `v16_final`:
- LBFGS polish is off by default (`--lbfgs-steps 0`).
- If hard diagnostic is already exact during search, hardening starts early
  (`--hard-trigger-mse`, `--hard-trigger-count`).
- `--skip-plot` disables live plotting windows, but still saves final `png/*_loss.png`.

## Environment
From repository root:

On Windows:

```powershell
pip install -r requirements.txt
```

On Linux:

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```


## Reproduce Main Result Style
Run from `PyTorch_v16_final`.

### 1) Random initialization benchmark (depth 2 example)

```powershell
python tree_prototype_torch_v16_final.py --target-fn eml_depth2 --depth 2 --save-prefix test_depth2_v16 --init-strategy all --seed0 137 --seeds 8 --data-lo 1.0 --data-hi 3.0 --data-step 0.1 --gen-lo 0.5 --gen-hi 5.0 --search-iters 6000 --hardening-iters 2000 --patience 4200 --eval-every 200 --tail-eval-every 50
```

### 2) Known-tree init + noise (depth 3 example)

```powershell
python tree_prototype_torch_v16_final.py --target-fn eml_depth3 --depth 3 --save-prefix exact_noise_v16 --init-strategy manual --init-expr "EML[EML[1, EML[y, x]], 1]" --init-noise 12 --seed0 137 --seeds 2 --data-lo 1.0 --data-hi 3.0 --data-step 0.1 --gen-lo 0.5 --gen-hi 5.0 --search-iters 3000 --hardening-iters 1200 --patience 2000 --eval-every 100 --tail-eval-every 25
```

## Output Files
- `*_stdout_*.log`: human-readable run log
- `*_metrics_*.json`: structured metrics
- `*.m`: snapped symbolic model (plus `_continuous.m`)
- `*.pt`: PyTorch state dict
- `png/*_loss.png`: per-run loss chart

# Convergence Summary (v16_final)

Summary below is from runs executed on February 28, 2026 in this workspace.

## Random Initialization

### Depth 2 (`target=eml_depth2`, depth=2, 32 runs total)
- total success: `32/32`
- by strategy:
  - `biased`: `8/8`
  - `uniform`: `8/8`
  - `xy_biased`: `8/8`
  - `random_hot`: `8/8`
- source metrics:
  - `v16_final/bench_depth2_v16/bench_depth2_v16_metrics_20260228-075403.json`

### Depth 3 (`target=eml_depth3`, depth=3, 32 runs total)
- total success: `8/32`
- by strategy:
  - `biased`: `2/8`
  - `uniform`: `2/8`
  - `xy_biased`: `2/8`
  - `random_hot`: `2/8`
- source metrics:
  - `v16_final/bench_depth3_random_v16/bench_depth3_random_v16_metrics_20260228-080459.json`

### Depth 4 (`target=eml_depth4`, depth=4, 16 runs total)
- total success: `8/16`
- by strategy:
  - `biased`: `1/4`
  - `uniform`: `2/4`
  - `xy_biased`: `2/4`
  - `random_hot`: `3/4`
- source metrics:
  - `v16_final/bench_depth4_random_v16/bench_depth4_random_v16_metrics_20260228-085029.json`

Interpretation:
- Depth 2 is reliably solved from random starts.
- Depth 3 remains difficult from random starts under this budget.
- Depth 4 shows mixed outcomes from random starts (about 50% overall in this sample).

## Known-Tree Initialization + Noise

### Depth 3 (manual exact tree + noise `12`, 2 runs)
- success: `2/2`
- stable symbolic success: `2/2`
- source metrics:
  - `v16_final/bench_depth3_manual_noise12_v16/bench_depth3_manual_noise12_v16_metrics_20260228-082216.json`

### Depth 4 (manual exact tree + noise `12`, 2 runs)
- success: `2/2`
- stable symbolic success: `2/2`
- source metrics:
  - `v16_final/bench_depth4_manual_noise12_v16/bench_depth4_manual_noise12_v16_metrics_20260228-090000.json`

Interpretation:
- With known-tree initialization, both depth 3 and 4 converge robustly at this noise level.
- The main bottleneck remains discovery from random initialization, not representational capacity.
