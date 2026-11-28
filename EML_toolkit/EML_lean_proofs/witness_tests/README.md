# Tests if the witnesses of the Lean proofs work with Mma and mpmath

The scripts builds the witnesses of Lean proofs from `eml` and the constant 1 in Wolfram Mathematica or in Python (mpmath) 
and compares them with the built-in functions.

| File | What it tests | Result |
|---|---|---|
| `Theorem1/verify_theorem1.wl` | Theorem 1 (Lemmas 1-24) and Corollary 25, exact points | 411 checks, all exact |
| `Theorem1/verify_theorem1.py` | Theorem 1, mpmath, 150 digits | 445 checks, no failure |
| `Theorem1a/verify_theorem1a.wl` | Theorem 1a (Table A.3) and Section A.4, exact points | 5101 checks, all exact |
| `Theorem1a/verify_theorem1a.py` | Theorem 1a, mpmath, 120 digits | 18726 checks, no failure |
| `Theorem1b_1c/verify_theorem1b_1c.wl` | Theorems 1b and 1c (Tables B.1, B.2, Appendix C), exact points | 509 checks, all exact |
| `Theorem1b_1c/verify_theorem1b_1c.py` | Theorems 1b and 1c, mpmath, 2000 digits, arguments of size 2 at most | 576 checks, no failure |

Run a script from its own folder:

```sh
wolframscript -file verify_theorem1.wl
python verify_theorem1.py
```

Next to each script is its output, in the file with `.out` added to the name of the script
(Mathematica 15.0, Python 3.12, mpmath 1.3, macOS). A run takes from a quarter of a minute to
five minutes.

No message of the Mathematica kernel is suppressed. The outputs of the `.wl` scripts contain a
few; the comment at the head of each script says which ones and why.

The `.wl` scripts work in exact arithmetic: a check counts as exact only if the difference to the
built-in function simplifies to the integer 0. The points include every kind of special point:
branch points, points on the branch cuts, and the ends of the real domains.

## The terms are the same in every source

`tools/terms.py` reads term definitions from Lean files and from Wolfram Language and Python
scripts, and expands each definition to its tree of `eml` nodes. Every tree is stored once, so
two terms are the same tree or they are not. The `cmp_*.py` programs compare a script with the
Lean files:

```sh
python tools/cmp_theorem1_wl_lean.py    ../EML Theorem1/verify_theorem1.wl
python tools/cmp_theorem1a_wl_lean.py   ../EML Theorem1a/verify_theorem1a.wl
python tools/cmp_theorem1b1c_wl_lean.py ../EML Theorem1b_1c/verify_theorem1b_1c.wl
python tools/cmp_py_lean.py             ../EML Theorem1/verify_theorem1.py Theorem1a/verify_theorem1a.py
python tools/cmp_py1b1c_lean.py         ../EML Theorem1b_1c/verify_theorem1b_1c.py
```

All of them report 0 different terms (51, 53, 38, 51 + 52, 38 terms).

