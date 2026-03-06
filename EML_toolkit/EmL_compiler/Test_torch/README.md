# `torch` test

The generated test evaluates the whole input grid in parallel as a `torch.complex128` tensor.

One function:

```sh
python3 make_eml_torch.py 'ArcCos[x]' eml_arccos -1 1 0.01
python3 test_eml_torch.py
```

Full unary suite:

```sh
python3 run_unary_suite_torch.py
```

Input required from parent directory:

- `../eml_compiler_v4.py`

Generated here:

- `test_eml_torch.py`
- `unary_suite_report_torch.txt`
