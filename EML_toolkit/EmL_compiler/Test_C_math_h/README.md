# C standard math test

This script purpose it to check if compiled EML expressions can be properly evaluated in C using `<math.h>`, `<complex.h>`, and a compiler available as `cc`. Script generates C code files `eml_math.h` and `test_eml.c`, and compiles them to executable file `test_eml`.

Platform:

- Linux,macOS: yes; Windows 11 PowerShell 7: no, use WSL.

Set up the shared Python environment first, see README from the parent directory.

To test single function, e.g, ArcCos[x] in the range -1<=x<=1 with step 0.01:

```sh
./make_eml_c.sh 'ArcCos[x]' eml_arccos -1 1 0.01
./test_eml
```

Run full unary suite:

```sh
python run_unary_suite_c.py
```

The suite itself is launched with Python, but internally it calls `./make_eml_c.sh`, so the same shell requirement applies.

Input (EML compiler) required from parent directory:

- `../eml_compiler_v4.py`

Generated files:

- `eml_math.h`
- `test_eml.c`
- `test_eml`
- `unary_suite_report_c.txt`
