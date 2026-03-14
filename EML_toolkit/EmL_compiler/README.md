# EML compiler with tests

This directory contains a standalone EML compiler plus four test suites that check whether compiled expressions can be evaluated with:

- C `<math.h>` and `<complex.h>`
- `numpy`
- `torch`
- `mpmath`

Additionaly, Wolfram Mathematica Notebook 

- `EmL_symbolic_simplification_test.nb`

is provided for check, if EML-compiled expressions do simplify symbolically to their originals. 
  

The compiler itself is eml_compiler_v4.py. It translates a Wolfram-style expression such as `ArcCos[x]`, or `x+y/z-2` into a pure EML expression. 

## Usage example

Compiler is able to convert any elementary formula (''function '') to pure EML form. Input should be proper Wolfram Mathematica expression with correct syntax. For those who do not use Mathematica, this usually means square brackets, and function name starting uppercase, e.g. Sin[x] vs sin(x).

For example, to obtain EML form for natural logarithm, use

python3 .\eml_compiler_v4.py Log[x]

Expected output:

EML[1,EML[EML[1,x],1]]

Verify in Mathematica by back-substitution of the EML:

EML[1, EML[EML[1, x], 1]] /. EML -> Function[{x, y}, Exp[x] - Log[y]]

Resulting expressions usually require heavy simplification. Try FullSimplify and specify real domain or range for x; use PowerExpand for quick check.

## What the symbolic simplification test do

First, you must generate EML-compiled expressions for all considered constants, unary functions, and binary operations. 

```sh
python eml_compiler_v4.py --emit-test
```

It will create `eml_tests_out/` directory with Wolfram Language `*.wl` files for originals and EML-compiled equivalents. After that, notebook `EmL_symbolic_simplification_test.nb` can be evaluated to provide symbolic check if all EML-compiled expressions are equivalent to originals. 

## What the numerical tests do

Each test suite follows the same pattern:

1. Compile a Wolfram-style expression to EML.
2. Generate source code for the chosen language or library from a template.
3. Evaluate the generated function on a real input grid.
4. Compare the real part against a reference function from the same language or library.
5. Report the largest real and imaginary errors on that grid.

The test subdirectories are:

- `Test_C_math_h`
- `Test_numpy`
- `Test_torch`
- `Test_mpmath`

The current suites cover unary expressions on real grids. They are intended as explicit demonstration runs for the claim that compiled EML expressions can be evaluated in these environments.

## Requirements

Minimal tools:

- Python 3.12 or newer
- for the C tests: either a compiler available as `cc`, or on Windows 11 an activated Intel oneAPI environment providing `icx`

Python packages:

- `sympy`
- `numpy`
- `mpmath`
- `torch`

The compiler depends on `sympy`. The test suites additionally depend on their own libraries.

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

## Run all four test suites

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

On Windows 11 for the C tests, first activate Intel oneAPI for `intel64` and re-enter PowerShell 7 so `icx` is on `PATH`:

```powershell
cmd.exe /K '"C:\Program Files (x86)\Intel\oneAPI\setvars.bat" intel64 && pwsh'
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
python make_eml_c.py 'ArcCos[x]' eml_arccos -1 1 0.01
```

Then run the generated binary:

```sh
./test_eml
```

On Windows 11 PowerShell 7, after activating oneAPI, use:

```powershell
cd Test_C_math_h
python .\make_eml_c.py 'ArcCos[x]' eml_arccos -1 1 0.01
.\test_eml.exe
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
python make_eml_mpmath.py 'ArcCos[x]' eml_arccos -1 1 0.01 64
python test_eml_mpmath.py
```

## Notes

- The C tests use C complex arithmetic, so the generated code includes both `<math.h>` and `<complex.h>`.
- `torch` runs the full grid in parallel as a `torch.complex128` tensor.
- `mpmath` supports higher precision; rerun its suite with `python run_unary_suite_mpmath.py --dps 256` to increase `mp.dps`.
- Generated files such as `test_eml_numpy.py`, `test_eml_torch.py`, `test_eml_mpmath.py`, `test_eml.c`, `test_eml`, `test_eml.exe`, and `eml_math.h` are disposable build artifacts.
