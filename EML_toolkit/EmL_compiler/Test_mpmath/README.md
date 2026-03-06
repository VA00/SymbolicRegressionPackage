# `mpmath` test

One function:

```sh
./make_eml_mpmath.sh 'ArcCos[x]' eml_arccos -1 1 0.01 64
python3 test_eml_mpmath.py
```

Full unary suite:

```sh
python3 run_unary_suite_mpmath.py
```

Optional higher precision:

```sh
python3 run_unary_suite_mpmath.py --dps 128
```

Inputs required from parent directory:

- `../eml_compiler_v4.py`

Generated here:

- `test_eml_mpmath.py`
- `unary_suite_report_mpmath.txt`
