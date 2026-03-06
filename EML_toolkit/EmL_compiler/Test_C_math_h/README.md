# C `<math.h>` test

One function:

```sh
./make_eml_c.sh 'ArcCos[x]' eml_arccos -1 1 0.01
./test_eml
```

Full unary suite:

```sh
python3 run_unary_suite_c.py
```

Inputs required from parent directory:

- `../eml_compiler_v4.py`

Generated here:

- `eml_math.h`
- `test_eml.c`
- `test_eml`
- `unary_suite_report_c.txt`
