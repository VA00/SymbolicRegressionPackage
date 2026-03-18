# EML Shortest-Expression Challenge

A speed competition for exact shortest-expression solvers in pure EML form.

## Overview

Given a target constant such as `2`, `-2`, `-1/2`, `1/2`, `Sqrt[2]`, `GoldenRatio`, or `Pi`, the goal is to find the shortest exact EML expression as fast as possible.

This is a runtime competition, not a code golf contest.

## Rules

- Any method is allowed: symbolic search, dynamic programming, brute force, pruning, compiled code, GPUs, clusters, custom hardware.
- Any programming language is allowed.
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

Ranking rule:

- Primary: highest official rung cleared
- Secondary: cumulative wall-clock time across all cleared official rungs
- Tertiary: cumulative internal solver time, if reported

Notes:

- Wall-clock time is the official metric.
- Internal timing is welcome, but secondary.
- Historical entries that have not yet been run on higher rungs remain on the ladder as partial clears.

## Official Leaderboard

| Rank | Team | Cleared | Ladder Time | `target=2` wall | `target=-2` wall | `target=1/2` wall | Language | Parallelization |
| --- | --- | --- | ---: | ---: | ---: | ---: | --- | --- |
| 1 | Antigravity (Hybrid) | 3/3 | 4.110 s | 0.711 s | 0.644 s | 2.755 s | Rust + Python | Rayon (24 threads) |
| 2 | GPT 5.4 Codex High | 2/3 | 1.193 s | 0.491 s | 0.702 s | superseded | Python + SymPy | None |
| 3 | GPT 5.4 Codex High | 2/3 | 42.700 s | 2.039 s | 40.661 s | not run | Wolfram Mathematica | None |
| 4 | A.O. | 2/3 | 473.111 s | 6.869 s | 466.242 s | not run | Wolfram Mathematica | None |

## Entry Details

### 1. Antigravity (Hybrid)

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
- `target=1/2` is currently the shortest verified exact result on the ladder.
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

### 2. GPT 5.4 Codex High

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
  - Status: no longer counts as a cleared rung because Antigravity found an exact `37`-token solution
- Historical Mathematica-ready output for `target=1/2`:

```wl
rpnRule[{1, 1, 1, 1, EML, 1, 1, 1, 1, 1, 1, EML, EML, EML, 1, EML, EML, 1, 1, EML, EML, 1, EML, EML, EML, EML, 1, EML, EML, 1, 1, EML, 1, EML, 1, EML, EML, 1, EML}]
```

- OS: Windows 11
- Shell: PowerShell 7
- Machine: AMD Ryzen 5900X, 32 GB RAM
- Parallelization: None

### 3. GPT 5.4 Codex High

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

### 4. A.O.

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
- Antigravity's walkthrough-reported ladder time is based on internal solver timing; the official leaderboard uses locally measured wall-clock command runtime.
- If the benchmark harness changes, older entries should be re-run on the same machine.
- A future version can add separate public and private leaderboards if the target set grows.
