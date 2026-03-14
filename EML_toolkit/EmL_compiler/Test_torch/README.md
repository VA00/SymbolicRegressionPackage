# PyTorch test

This test evaluates compiled EML expressions with `torch.complex128`. The generated program checks the full input grid in parallel.

Platform:

- Linux, macOS: yes
- Windows 11 PowerShell 7: yes

Set up the shared Python environment first, see README from the parent directory.

Single function:

```sh
python make_eml_torch.py ArcCos[x] -1 1 0.01
python test_eml_torch.py
```

Full unary suite:

```sh
python run_unary_suite_torch.py
```

Single binary operation:

```sh
python make_eml_binary_torch.py Hypot[x,y] 0.25 4 0.25 0.25 4 0.25
python test_eml_binary_torch.py
```

Full binary suite:

```sh
python run_binary_suite_torch.py
```

Input required from parent directory:

- `../eml_compiler_v4.py`

Generated here:

- `test_eml_torch.py`
- `test_eml_binary_torch.py`
- `unary_suite_report_torch.txt`
- `binary_suite_report_torch.txt`
