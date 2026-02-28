# EML Training Experiments (v15_final)

## Purpose
This package contains the PyTorch implementation and benchmark scripts used to fit symbolic trees built from the operator

`EML[a,b] = exp(a) - log(b)`

to predefined target functions (`eml_depth2..5`, `multiply`) on sampled `(x,y)` grids.

The main research question is practical identifiability:
- can optimization recover exact discrete trees from random initialization?
- how does convergence change with tree depth?
- when exact tree structure is known, how much logit noise can be tolerated?

## What Is Included
- `tree_prototype_torch_v15_final.py`
  - Main trainer (two-phase search + hardening + snap/eval).


## Environment
Tested with local Python and PyTorch in this workspace.

Install dependencies:

```powershell
pip install -r requirements.txt
```

## Reproducing Main Results
Run from this directory (`v15_final`)

### 1) Random initialization benchmark (depth 2/3/4/5/6)

```powershell
python tree_prototype_torch_v15_final.py --save-prefix "depth2_test" --target-fn eml_depth2 --depth $D --init-strategy all --seed0 137 --seeds 8 --data-lo 1.0 --data-hi 3.0 --data-step 0.1 --gen-lo 0.5 --gen-hi 5.0 --search-iters 6000 --hardening-iters 2000 --patience 4200 --hardening-trigger-min-iter 2600 --lbfgs-steps 160 --hardening-tau-power 2.0 --log-every 800 --tail-log-every 200 
```


### 2) Known-tree initialization + increasing logit noise (depth 3/4/5)

```powershell
python tree_prototype_torch_v15_final.py --save-prefix "exact_noide_test" --target-fn eml_depth3 --depth 3 --init-strategy manual --init-expr "EML[EML[1, EML[y, x]], 1]" --init-noise 12 --seed0 137 --seeds 2 --data-lo 1.0 --data-hi 3.0 --data-step 0.1 --gen-lo 0.5 --gen-hi 5.0 --search-iters 3000 --hardening-iters 1200 --patience 2000 --hardening-trigger-min-iter 1200 --lbfgs-steps 120 --hardening-tau-power 2.0 --log-every 600 --tail-log-every 200
```



# Convergence Summary (v15_final)

This is a compact summary of benchmark outcomes included in this package.

## Random Initialization

### Depth 2 (`target=eml_depth2`, depth=2, 16 runs/strategy)
- `biased`: 16/16
- `uniform`: 16/16
- `xy_biased`: 16/16
- `random_hot`: 15/16

### Depth 3 (`target=eml_depth3`, depth=3, 16 runs/strategy)
- `biased`: 1/16
- `uniform`: 1/16
- `xy_biased`: 1/16
- `random_hot`: 8/16

### Depth 4 (`target=eml_depth4`, depth=4, 16 runs, mixed `init-strategy=all`)
- total success: 13/16

### Depth 5 (`target=eml_depth5`, depth=5, 16 runs/strategy)
- all strategies: 0/16

### Depth 6 with Multiply target (`target=multiply`, depth=6, 16 runs/strategy)
- all strategies: 0/16

Interpretation:
- Depth 2 is solved reliably from random starts.
- Depth 3 is strongly strategy-sensitive (`random_hot` best in this test).
- Depth 4 can succeed with the tested mixed-strategy schedule.
- Depth 5 fails from random starts in tested budgets.


## Known-Tree + Noise (Logit Noise)

### Depth 3-8 manual known-tree initialization 
- Depth 3:
  - full success up to noise 24
  - partial at 32
  - first zero-success at 48
- Depth 4:
  - full success up to noise 24
  - partial at 32
  - first zero-success at 48
- Depth 5:
  - full success up to noise 32
  - partial at 48
  - first zero-success at 64


Interpretation:
- The model can represent and converge to exact symbolic solutions when initialized in the correct basin.
- Main bottleneck at higher depth is discovery from random starts (global optimization), not representational capacity.
