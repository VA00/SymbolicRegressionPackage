# C `<math.h>` test


Tests works in Linux and MacOSX terminals. 

To check EML code/formula for one function, e.g. ArcCos[x] in the range of -1 to 1 with step 0.01, run:

```sh
./make_eml_c.sh 'ArcCos[x]' eml_arccos -1 1 0.01
./test_eml
```
Generated files:

- `eml_math.h`
- `test_eml.c`
- `test_eml`


Full unary function suite (Sqrt, Sin, Cos, Tanh, etc.) run:

```sh
python3 run_unary_suite_c.py
```

Report saved at `unary_suite_report_c.txt`.

Test require EML compiler from parent directory:

- `../eml_compiler_v4.py`

Tests works in Linux and MacOSX terminals. 
