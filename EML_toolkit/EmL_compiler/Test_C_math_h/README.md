# C standard math test

This test checks whether compiled EML expressions can be evaluated in C using `<math.h>`, `<complex.h>`, and a working C compiler. It generates `eml_math.h` and `test_eml.c` from templates `eml_math.h.in` and `test_eml.c.in` for the requested function. Then it compiles them to `test_eml` on Linux and macOS, or `test_eml.exe` on Windows.


You can also edit the templates manually and compile them with a standard C compiler. Working C files `eml_math.h` and `test_eml.c` for the `Sqrt[x]` test are included for convenience, in case Python is unavailable.

Platform:

- Linux, macOS: yes
- Windows 11 PowerShell 7: yes, after activating Intel oneAPI so `icx` is on `PATH`

Set up the shared Python environment first, see README from the parent directory.

Windows 11 PowerShell 7 setup:

```powershell
cmd.exe /K '"C:\Program Files (x86)\Intel\oneAPI\setvars.bat" intel64 && pwsh'
```

The `Program Files (x86)` part is only where Intel installs `setvars.bat`; it does not imply a 32-bit target. The requested target is `intel64`, and the initialization banner should confirm that the Visual Studio environment is configured for `x64`.

To test a single unary function, e.g, ArcCos[x] in the range -1<=x<=1 with step 0.01:

```sh
python make_eml_c.py ArcCos[x] -1 1 0.01
```

Run the generated binary:

```sh
./test_eml
```

On Windows 11 PowerShell 7, run:

```powershell
python .\make_eml_c.py ArcCos[x] -1 1 0.01
.\test_eml.exe
```

Run full unary suite:

```sh
python run_unary_suite_c.py
```

To test a single binary operation, e.g, Hypot[x,y] on a positive grid:

```sh
python make_eml_binary_c.py Hypot[x,y] 0.25 4 0.25 0.25 4 0.25
./test_eml_binary
```

On Windows 11 PowerShell 7:

```powershell
python .\make_eml_binary_c.py Hypot[x,y] 0.25 4 0.25 0.25 4 0.25
.\test_eml_binary.exe
```

Run full binary suite:

```sh
python run_binary_suite_c.py
```

Use `make_eml_c.py` for unary tests and `make_eml_binary_c.py` for binary tests.

Input (EML compiler) required from parent directory:

- `../eml_compiler_v4.py`

Generated files:

- `eml_math.h`
- `eml_math_binary.h`
- `test_eml.c`
- `test_eml_binary.c`
- `test_eml`
- `test_eml.exe`
- `test_eml_binary`
- `test_eml_binary.exe`
- `unary_suite_report_c.txt`
- `binary_suite_report_c.txt`
