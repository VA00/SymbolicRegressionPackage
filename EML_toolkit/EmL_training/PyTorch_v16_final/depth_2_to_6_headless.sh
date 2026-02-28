#!/usr/bin/env bash
set -euo pipefail

# Lighter repeats for easy depths:
python3 tree_prototype_torch_v16_final.py --target-fn eml_depth2 --depth 2 --save-prefix overnight_d2_random --init-strategy all --seed0 137 --seeds 8 --data-lo 1.0 --data-hi 3.0 --data-step 0.1 --gen-lo 0.5 --gen-hi 5.0 --search-iters 6000 --hardening-iters 2000 --patience 4200 --eval-every 200 --tail-eval-every 50 --skip-plot
python3 tree_prototype_torch_v16_final.py --target-fn eml_depth3 --depth 3 --save-prefix overnight_d3_random --init-strategy all --seed0 137 --seeds 16 --data-lo 1.0 --data-hi 3.0 --data-step 0.1 --gen-lo 0.5 --gen-hi 5.0 --search-iters 6000 --hardening-iters 2000 --patience 4200 --eval-every 200 --tail-eval-every 50 --skip-plot
python3 tree_prototype_torch_v16_final.py --target-fn eml_depth4 --depth 4 --save-prefix overnight_d4_random --init-strategy all --seed0 137 --seeds 16 --data-lo 1.0 --data-hi 3.0 --data-step 0.1 --gen-lo 0.5 --gen-hi 5.0 --search-iters 6000 --hardening-iters 2000 --patience 4200 --eval-every 200 --tail-eval-every 50 --skip-plot

# Probability-focused runs for difficult depths (>=64 seeds each):
python3 tree_prototype_torch_v16_final.py --target-fn eml_depth5 --depth 5 --save-prefix overnight_d5_random64_random_hot --init-strategy random_hot --seed0 137 --seeds 64 --data-lo 1.0 --data-hi 3.0 --data-step 0.1 --gen-lo 0.5 --gen-hi 5.0 --search-iters 9000 --hardening-iters 3000 --patience 6500 --eval-every 200 --tail-eval-every 50 --skip-plot
python3 tree_prototype_torch_v16_final.py --target-fn eml_depth6 --depth 6 --save-prefix overnight_d6_random64_random_hot --init-strategy random_hot --seed0 137 --seeds 64 --data-lo 1.0 --data-hi 3.0 --data-step 0.1 --gen-lo 0.5 --gen-hi 5.0 --search-iters 9000 --hardening-iters 3000 --patience 6500 --eval-every 200 --tail-eval-every 50 --skip-plot

# Same 64-seed check for another init family (coverage against unlucky strategy choice):
python3 tree_prototype_torch_v16_final.py --target-fn eml_depth5 --depth 5 --save-prefix overnight_d5_random64_uniform --init-strategy uniform --seed0 1137 --seeds 64 --data-lo 1.0 --data-hi 3.0 --data-step 0.1 --gen-lo 0.5 --gen-hi 5.0 --search-iters 9000 --hardening-iters 3000 --patience 6500 --eval-every 200 --tail-eval-every 50 --skip-plot
python3 tree_prototype_torch_v16_final.py --target-fn eml_depth6 --depth 6 --save-prefix overnight_d6_random64_uniform --init-strategy uniform --seed0 1137 --seeds 64 --data-lo 1.0 --data-hi 3.0 --data-step 0.1 --gen-lo 0.5 --gen-hi 5.0 --search-iters 9000 --hardening-iters 3000 --patience 6500 --eval-every 200 --tail-eval-every 50 --skip-plot

# Known-tree + noise checks (small, sanity/reference):
python3 tree_prototype_torch_v16_final.py --target-fn eml_depth5 --depth 5 --save-prefix overnight_d5_manual_noise12 --init-strategy manual --init-expr "EML[1, EML[EML[1, EML[1, EML[x, y]]], 1]]" --init-noise 12 --seed0 2137 --seeds 4 --data-lo 1.0 --data-hi 3.0 --data-step 0.1 --gen-lo 0.5 --gen-hi 5.0 --search-iters 3000 --hardening-iters 1200 --patience 2000 --eval-every 100 --tail-eval-every 25 --skip-plot
python3 tree_prototype_torch_v16_final.py --target-fn eml_depth6 --depth 6 --save-prefix overnight_d6_manual_noise12 --init-strategy manual --init-expr "EML[1, EML[EML[EML[1, EML[EML[x, y], 1]], 1], 1]]" --init-noise 12 --seed0 2137 --seeds 4 --data-lo 1.0 --data-hi 3.0 --data-step 0.1 --gen-lo 0.5 --gen-hi 5.0 --search-iters 3000 --hardening-iters 1200 --patience 2000 --eval-every 100 --tail-eval-every 25 --skip-plot
