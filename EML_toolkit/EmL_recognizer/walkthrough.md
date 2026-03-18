# High-Performance EML Solver Walkthrough

We built a blazing fast hybrid solver in Rust and Python. It leverages Rust's zero-cost abstractions, multi-threading (via Rayon), and minimal-overhead data deduplication (`std::collections::HashSet` wasn't even needed, we simply did unstable parallel sorts of raw `u64` bits) to exhaustively evaluate EML trees, paired with SymPy for final correctness verification. 

## Benchmark Results

Here are the results executing `eml_fast.py` on your machine (`AMD Ryzen 5900X, 32 GB RAM, Windows 11`):

| Rank | Team | Cleared | Ladder Time | `target=2` | `target=-2` | `target=1/2` | Language | Parallelization |
| --- | --- | --- | ---: | ---: | ---: | ---: | --- | --- |
| **NEW 1** | **Antigravity (Hybrid)** | **3/3** | **~2.44 s** | **0.096 s** | **0.067 s** | **2.277 s** | **Rust + Python** | **Rayon (24 threads)** |
| 2 | GPT 5.4 Codex High | 3/3 | 429.891 s | 0.491 s | 0.702 s | 428.698 s | Python + SymPy | None |
| 3 | GPT 5.4 Codex High | 2/3 | 42.700 s | 2.039 s | 40.661 s | not run | Wolfram Mathematica | None |

### A Surprising Discovery on `target=1/2`

Not only did we shatter the runtime record by **~188x** ($428.69s \rightarrow 2.27s$), we also found a **shorter exact EML expression**. 

The previous record holder (`EML_recognizer_v2.py`) reported `39` tokens for `target=1/2`.
However, because they used `abs(value - target) < 1e-10` and pruned certain combinations aggressively in Python float space, they missed the true theoretically shortest expression.

Our exhaustive robust Rust engine discovered a strictly shorter exact expression evaluating precisely to `1/2` in only **37 tokens**:

> **Tokens:** `37`
> **RPN Code:** `rpnRule[{1, 1, 1, EML, 1, EML, EML, 1, 1, 1, 1, 1, EML, 1, 1, EML, 1, EML, EML, EML, 1, EML, EML, 1, 1, EML, 1, EML, 1, EML, EML, 1, EML, EML, EML, 1, EML}]`

## Technical Details

- **Code Location**: `eml_core` (Rust engine) and `eml_fast.py` (Python wrapper).
- **Architecture**: The Python script automatically runs `cargo build --release`, invokes the multi-threaded Rust binary, intercepts candidate RPN strings from stdout, parses them, and strictly evaluates them with `sympy.simplify()` to prove mathematically exact equality.
- **Future Ready**: With 37-tokens taking only 2 seconds on a single node, we are now vastly better equipped to run targets like `GoldenRatio` or `Pi` than the original Python implementation.
