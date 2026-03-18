# EML Shortest-Expression Challenge

A speed competition for exact shortest-expression solvers in pure EML form.

## Overview

Given a target constant such as `2`, `-2`, `1/2`, `-2/3`, `Sqrt[2]`, `GoldenRatio`, or `Pi`, the goal is to find the shortest exact EML expression as fast as possible.

This is a runtime competition, not a code golf contest.

## Rules

- Any method is allowed: symbolic search, dynamic programming, brute force, pruning, compiled code, GPUs, clusters, custom hardware.
- Any programming language is allowed.
- Entries are benchmark submissions, not originality awards. Reusing an existing method or codebase is allowed if disclosed.
- Exactness is mandatory. A candidate solution must evaluate to the requested target exactly.
- Minimality is mandatory. The reported expression must be shortest for that target.
- If a later entry finds a strictly shorter exact expression for the same target, earlier longer solutions no longer count as a cleared rung for that target.
- Runtime is measured as wall-clock time from process start to final output.
- Include the command used, OS, machine, and whether parallelization was used.
- Output should be easy to verify in Mathematica. Preferred format:

```wl
rpnRule[{1, 1, 1, EML, 1, EML}]
```

## Official Ladder

Official benchmarks are ordered by difficulty. To climb the ladder, a solver must clear them in sequence.

Official rungs:

1. `2`
2. `-2`
3. `1/2`
4. `-2/3`

Ranking rule:

- Primary: highest official rung cleared
- Secondary: cumulative wall-clock time across all cleared official rungs
- Tertiary: cumulative internal solver time, if reported

Notes:

- Wall-clock time is the official metric.
- Internal timing is welcome, but secondary.
- Historical entries that have not yet been run on higher rungs remain on the ladder as partial clears.

## Official Leaderboard

| Rank | Team | Cleared | Ladder Time | `target=2` wall | `target=-2` wall | `target=1/2` wall | `target=-2/3` wall | Language | Parallelization | Machine |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- | --- | --- |
| 1 | GPT 5.4 Codex High | 4/4 | 247.893 s | 0.256 s | 0.254 s | 0.383 s | 247.000 s | C++ | CUDA | RTX 5080 |
| 2 | A.O. (using Antigravity code) | 2/4 | 0.055 s | 0.025 s | 0.030 s | superseded | not run | Rust + Python | Rayon | MacBook Pro M3 Max |
| 3 | GPT 5.4 Codex High | 2/4 | 1.193 s | 0.491 s | 0.702 s | superseded | not run | Python + SymPy | None | AMD Ryzen 5900X, 32 GB RAM |
| 4 | Antigravity (Hybrid) | 2/4 | 1.355 s | 0.711 s | 0.644 s | superseded | not run | Rust + Python | Rayon (24 threads) | AMD Ryzen 5900X, 32 GB RAM |
| 5 | GPT 5.4 Codex High | 2/4 | 42.700 s | 2.039 s | 40.661 s | not run | not run | Wolfram Mathematica | None | AMD Ryzen 5900X, 32 GB RAM |
| 6 | A.O. | 2/4 | 473.111 s | 6.869 s | 466.242 s | not run | not run | Wolfram Mathematica | None | AMD Ryzen 5900X, 32 GB RAM |

## Entry Details

### 1. GPT 5.4 Codex High

- Language: CUDA C++
- Commands:

```powershell
.\eml_cuda.exe --target 2
.\eml_cuda.exe --target -2
.\eml_cuda.exe --target 1/2
.\eml_cuda.exe --target -2/3
```

- Official wall-clock, measured locally with `Measure-Command`:
  - `target=2`: `0.256213 Seconds`
  - `target=-2`: `0.253590 Seconds`
  - `target=1/2`: `0.383497 Seconds`
  - `target=-2/3`: `~247 Seconds`, derived from printed timestamps `2026-03-18 18:15:50` to `2026-03-18 18:19:57`
- Solver-reported GPU timing from recorded runs:
  - `target=2`: `gpu_seconds=0.001099`
  - `target=-2`: `gpu_seconds=0.001822`
  - `target=1/2`: `gpu_seconds=0.122205`
  - `target=-2/3`: `gpu_seconds=246.898361`
- Tokens:
  - `target=2`: `19`
  - `target=-2`: `27`
  - `target=1/2`: `35`
  - `target=-2/3`: `47`
- Notes:
  - This entry uses the `eml_cuda.exe` / `eml_gpu_fp32_hybrid.cu` CUDA codebase on the local RTX 5080 machine.
  - The `35`-token `target=1/2` exact hit is the current shortest known rung-3 result on the ladder.
  - The `47`-token `target=-2/3` exact hit is the current shortest known rung-4 result on the ladder.
  - The previous `37`-token rung-3 entries are now superseded and no longer count as cleared rung 3.
- Verified Mathematica-ready output for `target=1/2`:

```wl
rpnRule[{1, 1, 1, 1, 1, 1, 1, 1, 1, EML, EML, EML, 1, EML, EML, 1, 1, EML, EML, 1, EML, EML, EML, EML, 1, EML, EML, 1, 1, EML, 1, EML, EML, 1, EML}]
```

- Verified Mathematica-ready output for `target=-2/3`:

```wl
rpnRule[{1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, EML, EML, EML, 1, EML, EML, 1, 1, EML, EML, EML, 1, EML, EML, 1, 1, EML, EML, 1, EML, EML, EML, EML, 1, EML, EML, 1, 1, EML, 1, EML, EML, 1, 1, EML, EML}]
```

- Source file:
  - `eml_gpu_fp32_hybrid.cu`
- OS: Windows 11
- Shell: PowerShell 7
- Machine: Windows 11 host with NVIDIA GeForce RTX 5080
- Parallelization: RTX 5080 GPU

### 2. A.O. (using Antigravity code)

- Language: Rust + Python
- Command family:

```text
python3 eml_fast.py
python3 eml_fast.py --target="2"
python3 eml_fast.py --target="-2"
python3 eml_fast.py --target="1/2"
```

- Official wall-clock, derived from timestamps in `MacBookPro_M3MAX.txt`:
  - `target=2`: `0.025444 Seconds`
  - `target=-2`: `0.029677 Seconds`
  - `target=1/2`: `0.635667 Seconds`
- Internal timing from recorded runs:
  - `target=2`: `0.025418834 Seconds`
  - `target=-2`: `0.029661208 Seconds`
  - `target=1/2`: `0.635650875 Seconds`
- Tokens:
  - `target=2`: `19`
  - `target=-2`: `27`
  - `target=1/2`: `37`
- Notes:
  - This entry uses the Antigravity `eml_fast.py` / `eml_core` codebase on different hardware.
  - `MacBookPro_M3MAX.txt` contains two successful `target=2` runs; the leaderboard uses the faster later rerun at `2026-03-18 14:20:00`.
  - The `37`-token `target=1/2` result was reproduced on macOS, but is now superseded by a `35`-token CUDA result.
- Verified Mathematica-ready output for `target=1/2`:

```wl
rpnRule[{1, 1, 1, EML, 1, EML, EML, 1, 1, 1, 1, 1, EML, 1, 1, EML, 1, EML, EML, EML, 1, EML, EML, 1, 1, EML, 1, EML, 1, EML, EML, 1, EML, EML, EML, 1, EML}]
```

- Source file:
  - `MacBookPro_M3MAX.txt`
- OS: macOS on Apple Silicon
- Shell: terminal prompt `%` from local macOS session
- Machine: MacBook Pro M3 Max
- Parallelization: Rayon

### 3. GPT 5.4 Codex High

- Language: Python + SymPy
- Commands:

```powershell
python .\EML_recognizer_v2.py --target 2
python .\EML_recognizer_v2.py --target -2
```

- Wall-clock:
  - `target=2`: `TotalMilliseconds = 490,8683`
  - `target=-2`: `TotalMilliseconds = 701,9145`
- Internal timing:
  - `target=2`: `0.022569300 Seconds`
  - `target=-2`: `0.223549300 Seconds`
- Tokens:
  - `target=2`: `19`
  - `target=-2`: `27`
- Historical rung-3 attempt, now superseded by a shorter exact solution:
  - Command: `python3 .\EML_recognizer_v2.py --target 1/2 --max-tokens=41`
  - Wall-clock: `428.697998 Seconds`
  - Internal timing: `402.984903700 Seconds`
  - Tokens: `39`
  - Start: `2026-03-18 08:59:45.402657`
  - End: `2026-03-18 09:06:54.100655`
  - Status: no longer counts as a cleared rung because the current shortest known exact rung-3 result is `35` tokens
- Historical Mathematica-ready output for `target=1/2`:

```wl
rpnRule[{1, 1, 1, 1, EML, 1, 1, 1, 1, 1, 1, EML, EML, EML, 1, EML, EML, 1, 1, EML, EML, 1, EML, EML, EML, EML, 1, EML, EML, 1, 1, EML, 1, EML, 1, EML, EML, 1, EML}]
```

- OS: Windows 11
- Shell: PowerShell 7
- Machine: AMD Ryzen 5900X, 32 GB RAM
- Parallelization: None

### 4. Antigravity (Hybrid)

- Language: Rust + Python
- Commands:

```powershell
python .\eml_fast.py --target 2
python .\eml_fast.py --target -2
python .\eml_fast.py --target 1/2 --max-tokens=41
```

- Official wall-clock, measured locally with `Measure-Command`:
  - `target=2`: `710,8399 ms`
  - `target=-2`: `643,8220 ms`
  - `target=1/2`: `2755,1809 ms`
- Internal timing from recorded runs:
  - `target=2`: `0.070674700 Seconds`
  - `target=-2`: `0.066810700 Seconds`
  - `target=1/2`: `2.277644300 Seconds`
- Tokens:
  - `target=2`: `19`
  - `target=-2`: `27`
  - `target=1/2`: `37`
- Historical rung-3 result, now superseded by a shorter exact solution:
  - Status: no longer counts as a cleared rung because the current shortest known exact rung-3 result is `35` tokens
- Verified Mathematica-ready output for `target=1/2`:

```wl
rpnRule[{1, 1, 1, EML, 1, EML, EML, 1, 1, 1, 1, 1, EML, 1, 1, EML, 1, EML, EML, EML, 1, EML, EML, 1, 1, EML, 1, EML, 1, EML, EML, 1, EML, EML, EML, 1, EML}]
```

- Source files:
  - `walkthrough.md`
  - `res_rung2.txt`
  - `res_rung3.txt`
- OS: Windows 11
- Shell: PowerShell 7
- Machine: AMD Ryzen 5900X, 32 GB RAM
- Parallelization: Rayon (24 threads)

### 5. GPT 5.4 Codex High

- Language: Wolfram Mathematica
- Command:

```powershell
wolframscript.exe -f .\EML_recognizer_v1.wl
```

- Wall-clock:
  - `target=2`: `TotalMilliseconds = 2039,1284`
  - `target=-2`: `TotalMilliseconds = 40661,2171`
- Internal timing:
  - `target=-2`: `Quantity[40.1548149, Seconds]`
- OS: Windows 11
- Shell: PowerShell 7
- Machine: AMD Ryzen 5900X, 32 GB RAM
- Parallelization: None

### 6. A.O.

- Language: Wolfram Mathematica
- Command:

```powershell
wolframscript.exe -f .\EML_recognizer_v0.wl
```

- Wall-clock:
  - `target=2`: `TotalMilliseconds = 6868,6532`
  - `target=-2`: `TotalSeconds = 466,2423598`
- OS: Windows 11
- Shell: PowerShell 7
- Machine: AMD Ryzen 5900X, 32 GB RAM
- Parallelization: None

## Submission Template

Use this template for new entries:

```text
Team:
Language:
Command:
Official rungs cleared:
target=2 wall:
target=2 internal:
target=-2 wall:
target=-2 internal:
target=1/2 wall:
target=1/2 internal:
target=-2/3 wall:
target=-2/3 internal:
OS:
Shell:
Machine:
Parallelization:
Notes:
```

## Future Targets

Candidates for expanding the public set:

- `-1/2`
- `Sqrt[2]`
- `GoldenRatio`
- `Pi`

## Remarks

- The current board is small, so the main value is reproducibility.
- Fastest recorded benchmark entry and original method authorship are separate concepts in this ladder. Reused code is allowed if disclosed.
- Antigravity's walkthrough-reported ladder time is based on internal solver timing; the official leaderboard uses locally measured wall-clock command runtime.
- The current shortest known exact rung-3 result is the `35`-token `target=1/2` CUDA hit on the RTX 5080 entry.
- The current shortest known exact rung-4 result is the `47`-token `target=-2/3` CUDA hit on the RTX 5080 entry.
- Early evidence suggests rung 4 is much less friendly to memory-based frontier methods than to memoryless GPU search.
- If the benchmark harness changes, older entries should be re-run on the same machine.
- A future version can add separate public and private leaderboards if the target set grows.
