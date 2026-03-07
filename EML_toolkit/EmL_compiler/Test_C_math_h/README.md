# C backend test

This backend checks compiled EML expressions in C using `<math.h>`, `<complex.h>`, and a compiler available as `cc`.

Platform:

- Linux: yes
- macOS: yes
- Windows 11 PowerShell 7: not with the provided helper workflow, because both `./make_eml_c.sh` and `run_unary_suite_c.py` rely on a POSIX `sh` script. Use Git Bash, MSYS2, or WSL if you want to run the supplied helpers on Windows.

Set up the shared Python environment first from the parent directory:

```sh
cd ..
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-test.txt
cd Test_C_math_h
```

Single function:

```sh
./make_eml_c.sh 'ArcCos[x]' eml_arccos -1 1 0.01
./test_eml
```

Full unary suite:

```sh
python run_unary_suite_c.py
```

The suite itself is launched with Python, but internally it calls `./make_eml_c.sh`, so the same shell requirement applies.

Input required from parent directory:

- `../eml_compiler_v4.py`

Generated here:

- `eml_math.h`
- `test_eml.c`
- `test_eml`
- `unary_suite_report_c.txt`
