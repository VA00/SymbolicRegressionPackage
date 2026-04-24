## EML Toolkit Reproducibility Quick Start

The reproducibility materials are in `EML_toolkit/`:
- `EmL_compiler/` (compiler + numeric backends)
- `EmL_verification/` (Wolfram verification scripts/notebooks)
- `EmL_training/` (PyTorch + Mathematica training artifacts)
- `EmL_figures/` (figure generators)
- `EmL_recognizer/` (CUDA recognizer tools)

For the long PyTorch sweep used in the Supplementary Information:

```bash
cd EML_toolkit/EmL_training/PyTorch_v16_final
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
bash depth_2_to_6_headless.sh
```

Outputs are written under `eml_*` directories as:
- `*_stdout_*.log` (run logs)
- `*_metrics_*.json` (structured summaries)
- `.m`, `_continuous.m`, `.pt`, and `png/*_loss.png`

Linux validation run (March 22, 2026):
- Host: Intel Ultra 285, 128 GB RAM, NVIDIA GeForce RTX 5080 (16 GB)
- Driver/CUDA: 580.126.20 / 13.0
- Python/PyTorch: 3.12.3 / 2.10.0+cu128
- Full sweep wall time: `2:22:25` (2.5 hours)

Zenodo note: source code and scripts are sufficient to reproduce results; bulky generated outputs (model checkpoints and logs, 350 MB) were omitted as they are regenerable from archived sources.


