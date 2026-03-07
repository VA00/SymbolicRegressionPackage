# PyTorch backend test

This test checks compiled EML expressions evaluation with tensor `torch.complex128` format. The generated test evaluates the full input grid in parallel. PyTorch (https://github.com/pytorch/pytorch) is modern standard for massively-parallel numerical computations using GPUs. 

Platform:

    Linux,macOS: yes; Windows 11 PowerShell 7: yes.

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
