# SymbolicRegressionPackage

Wolfram Mathematica package for brute-force symbolic regression: exhaustive search over expression trees built from user-specified constants, unary functions, and binary operations.


## Core functionality


- `RecognizeConstant[x]` — identify an analytic formula for a numeric value, searching in order of growing Kolmogorov complexity.
- `RecognizeFunction`, `RecognizeSequence` — same idea for univariate data and integer sequences.
- `RandomExpression[]`, `EnumerateExpressions[]` — generate random or exhaustive expression trees from given building blocks.
- `VerifyBaseSet[constants, functions, operations]` — bootstrapping completeness check: given a set of primitives, verify whether elementary functions can be reconstructed from given primitives.

## Quick start

Check out examples from `SymbolicRegressionPackage_Examples.nb`.


## Repository structure

- `SymbolicRegression.m` — main Mathematica package
- `EML_toolkit/` — EML compiler (Python), numerical test harnesses (C, NumPy, PyTorch, mpmath), symbolic verification notebooks, PyTorch tree trainer, figure scripts, and CUDA shortest-expression search tools
- `rust_*/` — Rust reimplementation of the bootstrapping procedure and search tools (~35 s vs ~40 min in Mathematica)


## Requirements

- Wolfram Mathematica (≥13.0)
- Python ≥3.9 with NumPy, PyTorch, mpmath (for EML toolkit)
- Rust ≥1.70 (for rust tools)
- C compiler (for Eml_verify)
- CUDA toolkit (for EmL_recognizer)


## License

MIT
