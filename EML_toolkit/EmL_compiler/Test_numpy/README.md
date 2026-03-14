# NumPy test

This test evaluates compiled EML expressions using `numpy` complex arithmetic.

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
cd Test_numpy
```

Single function:

```sh
python make_eml_numpy.py ArcCos[x] -1 1 0.01
python test_eml_numpy.py
```

Full unary suite:

```sh
python run_unary_suite_numpy.py
```

Single binary operation:

```sh
python make_eml_binary_numpy.py Hypot[x,y] -4 4 0.25 -4 4 0.25
python test_eml_binary_numpy.py
```

Full binary suite:

```sh
python run_binary_suite_numpy.py
```

The binary suite uses the same `[-8, 8] x [-8, 8]` grid with step `0.5` for all operations. Points outside the real domain are counted as `out_of_domain`.

Input required from parent directory:

- `../eml_compiler_v4.py`

Generated here:

- `test_eml_numpy.py`
- `test_eml_binary_numpy.py`
- `unary_suite_report_numpy.txt`
- `binary_suite_report_numpy.txt`
