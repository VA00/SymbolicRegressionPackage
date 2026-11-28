# EmL CUDA recognizer

Six standalone CUDA tools live here, for constant, unary, and binary search.
Three with `fp32` in the name search the real domain, and three with `complex32` search the complex domain.
The algorithm uses massive single-precision throughput to sieve candidates, then verifies them in double precision.
As of March 2026, this is the fastest method known to me.

## Requirements

- 64-bit Windows 11 or 64-bit Linux.
- NVIDIA CUDA-capable GPU. There is no CPU-only path.
- NVIDIA driver, CUDA Toolkit, and `nvcc`.
- A host compiler visible to `nvcc`.
- Tested GPUs: Tesla T4, RTX 5080 (local), RTX PRO 6000 Blackwell Server Edition (Google Cloud, 120 TFLOPS FP32 theoretical).
- Tested using CUDA 13.1, Driver Version: 591.74

## Compile on Windows 11

Open a Microsoft Visual Studio developer shell where the x64 MSVC host compiler is on `PATH`. 
If you have an Intel oneAPI, use:

```powershell
cmd.exe /K '"C:\Program Files (x86)\Intel\oneAPI\setvars.bat" intel64 && pwsh'
```

Then run in this directory:

```powershell
nvcc -std=c++17 -O3 eml_gpu_fp32_hybrid_const.cu -o eml_gpu_fp32_hybrid_const.exe
```

Use the same pattern for the other files. `-std=c++17` is required by the `complex32` tools, which depend on modern Thrust/CCCL headers.

## Run

Each tool also supports `--help`. For real search, the important switch is `--disable-inf`, which disables the extended-real conventions enabled by default, such as `log(0) = -inf` and `exp(-inf) = 0`. If a candidate is found, it is returned in RPN form, e.g.
```
rpnRule[{1, 1, 1, 1, 1, EML, EML, EML, 1, EML, EML, 1, 1, EML, 1, EML, EML}]
```
This can be pasted into `EML_symbolic_check.nb` in Wolfram Mathematica for symbolic verification.

Minimal sanity check on Windows 11 after compilation:

```powershell
.\eml_gpu_fp32_hybrid_const.exe --target 2 --max-tokens 19
```

This should report an `exact-hit` witness for the constant `2`, together with the
RPN form and a GPU timing summary.
