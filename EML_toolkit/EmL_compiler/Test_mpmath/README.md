# Arbitrary-precision mpmath test

This test evaluates compiled EML expressions with arbitrary-precision complex arithmetic from `mpmath`.

Platform:

- Linux, macOS: yes
- Windows 11 PowerShell 7: yes

Set up the shared Python environment first, see README from the parent directory.

To test single function, e.g, ArcCos[x] in the range -1<=x<=1 with step 0.01 using 64-digit arithmetic (i.e., 4x machine precision):

```sh
python make_eml_mpmath.py ArcCos[x] -1 1 0.01 64
python test_eml_mpmath.py
```

Full unary suite:

```sh
python run_unary_suite_mpmath.py
```

Single binary operation:

```sh
python make_eml_binary_mpmath.py Hypot[x,y] 0.25 4 0.25 0.25 4 0.25 64
python test_eml_binary_mpmath.py
```

Full binary suite:

```sh
python run_binary_suite_mpmath.py
```


Optional test with higher precision, 128 digits:

```sh
python run_unary_suite_mpmath.py --dps 128
```

Input required from parent directory:

- `../eml_compiler_v4.py`

Generated files:

- `test_eml_mpmath.py`
- `test_eml_binary_mpmath.py`
- `unary_suite_report_mpmath.txt`
- `binary_suite_report_mpmath.txt`
