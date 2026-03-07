# EML compiler backend tests

This directory contains a standalone EML compiler plus four backend-specific test suites that check whether compiled expressions can be evaluated with:

- C `<math.h>` and `<complex.h>`
- `numpy`
- `torch`
- `mpmath`

The compiler itself is [eml_compiler_v4.py](/Users/misiek/Documents/Articles/EML%20Sheffer/EML_Sheffer_PNAS_2026-02-28/SymbolicRegressionPackage/EML_toolkit/EmL_compiler/eml_compiler_v4.py). It translates a Wolfram-style expression such as `ArcCos[x]` into a pure EML expression.

## What the backend tests do

Each backend suite follows the same pattern:

1. Compile a Wolfram-style expression to EML.
2. Generate backend-specific source code from a template.
3. Evaluate the generated function on a real input grid.
4. Compare the real part against a native backend reference function.
5. Report the largest real and imaginary errors on that grid.

The per-backend subdirectories are:

- `Test_C_math_h`
- `Test_numpy`
- `Test_torch`
- `Test_mpmath`

The current suites cover unary expressions on real grids. They are intended as explicit demonstration runs for the manuscript claim that compiled EML expressions can be evaluated in these environments.

## Requirements

Minimal tools:

- Python 3.12 or newer
- a C compiler available as `cc`

Python packages:

- `sympy`
- `numpy`
- `mpmath`
- `torch`

The compiler depends on `sympy`. The backend suites additionally depend on their own libraries.

## Reproducible setup

Run everything from this directory:

```sh
cd SymbolicRegressionPackage/EML_toolkit/EmL_compiler
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-test.txt
```

If `python3` on your machine is not Python 3.12+, replace it with an explicit interpreter such as `python3.12`.

## Quick sanity check

Compile one expression:

```sh
python eml_compiler_v4.py 'ArcCos[x]'
```

Generate Wolfram lists used elsewhere in the toolkit:

```sh
python eml_compiler_v4.py --emit-test
```

## Run all four backend suites

From this directory, with the virtual environment activated:

```sh
cd Test_C_math_h
python run_unary_suite_c.py

cd ../Test_numpy
python run_unary_suite_numpy.py

cd ../Test_torch
python run_unary_suite_torch.py

cd ../Test_mpmath
python run_unary_suite_mpmath.py
```

Each suite writes a report in its own directory:

- `Test_C_math_h/unary_suite_report_c.txt`
- `Test_numpy/unary_suite_report_numpy.txt`
- `Test_torch/unary_suite_report_torch.txt`
- `Test_mpmath/unary_suite_report_mpmath.txt`

## Single-function examples

C:

```sh
cd Test_C_math_h
./make_eml_c.sh 'ArcCos[x]' eml_arccos -1 1 0.01
./test_eml
```

NumPy:

```sh
cd Test_numpy
python make_eml_numpy.py 'ArcCos[x]' eml_arccos -1 1 0.01
python test_eml_numpy.py
```

PyTorch:

```sh
cd Test_torch
python make_eml_torch.py 'ArcCos[x]' eml_arccos -1 1 0.01
python test_eml_torch.py
```

mpmath:

```sh
cd Test_mpmath
./make_eml_mpmath.sh 'ArcCos[x]' eml_arccos -1 1 0.01 64
python test_eml_mpmath.py
```

## Notes

- The C backend uses C complex arithmetic, so the generated code includes both `<math.h>` and `<complex.h>`.
- `torch` runs the full grid in parallel as a `torch.complex128` tensor.
- `mpmath` supports higher precision; rerun its suite with `python run_unary_suite_mpmath.py --dps 128` to increase `mp.dps`.
- Generated files such as `test_eml_numpy.py`, `test_eml_torch.py`, `test_eml_mpmath.py`, `test_eml.c`, `test_eml`, and `eml_math.h` are disposable build artifacts.
