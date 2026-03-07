# PyTorch backend test

This backend checks compiled EML expressions with `torch.complex128`. The generated test evaluates the full input grid in parallel.

Platform:

- Linux: yes
- macOS: yes
- Windows 11 PowerShell 7: yes

This directory uses only Python scripts, so it does not require `bash` or `sh`.

Set up the shared Python environment first from the parent directory:

```sh
cd ..
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-test.txt
cd Test_torch
```

Single function:

```sh
python make_eml_torch.py 'ArcCos[x]' eml_arccos -1 1 0.01
python test_eml_torch.py
```

Full unary suite:

```sh
python run_unary_suite_torch.py
```

Input required from parent directory:

- `../eml_compiler_v4.py`

Generated here:

- `test_eml_torch.py`
- `unary_suite_report_torch.txt`
