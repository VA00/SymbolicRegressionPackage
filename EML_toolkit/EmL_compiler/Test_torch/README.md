# PyTorch test

This test evaluates compiled EML expressions with `torch.complex128`. The generated program checks the full input grid in parallel.

Platform:

- Linux, macOS: yes
- Windows 11 PowerShell 7: yes

Set up the shared Python environment first, see README from the parent directory.

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
