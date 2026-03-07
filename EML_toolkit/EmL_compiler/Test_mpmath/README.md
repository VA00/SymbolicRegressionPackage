# mpmath backend test

This backend checks compiled EML expressions with arbitrary-precision complex arithmetic from `mpmath`.

Platform:

- Linux: yes
- macOS: yes
- Windows 11 PowerShell 7: not with the provided helper workflow, because both `./make_eml_mpmath.sh` and `run_unary_suite_mpmath.py` rely on a POSIX `sh` script. Use Git Bash, MSYS2, or WSL if you want to run the supplied helpers on Windows.

Set up the shared Python environment first from the parent directory:

```sh
cd ..
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-test.txt
cd Test_mpmath
```

Single function:

```sh
./make_eml_mpmath.sh 'ArcCos[x]' eml_arccos -1 1 0.01 64
python test_eml_mpmath.py
```

Full unary suite:

```sh
python run_unary_suite_mpmath.py
```

The suite itself is launched with Python, but internally it calls `./make_eml_mpmath.sh`, so the same shell requirement applies.

Optional higher precision:

```sh
python run_unary_suite_mpmath.py --dps 128
```

Input required from parent directory:

- `../eml_compiler_v4.py`

Generated here:

- `test_eml_mpmath.py`
- `unary_suite_report_mpmath.txt`
