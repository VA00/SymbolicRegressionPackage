## EmL_proof: Lean 4 proof of Theorem 1

A machine-checked proof of Theorem 1 of the paper: each of the 33 real
primitives of Table 1 (5 constants, 20 unary functions, 8 binary operations)
has one EML term, built from `eml` and `1`, that is defined and equal to the
primitive at every point of its natural real domain. The proof follows Lemmas 1-24
of the paper, one Lean theorem per lemma.

Lean 4.35.0 with Mathlib 4.35.0.

### Files

| File | Content |
|---|---|
| `EML/Basic.lean` | EML terms (`Term`), their value (`eval`), the identities (F1), (F2) |
| `EML/Core.lean` | Lemmas 1-8: E, e, L, S, N, the constants, M, D, i, pi |
| `EML/Arith.lean` | Lemmas 9-16: U, +, -, sqr, half, x, inv, /, avg |
| `EML/UnitStep.lean` | Lemmas 17-20: X, T, the step function H, sqrt, hypot |
| `EML/ExpLog.lean` | Lemma 21: sigma, pow, log, sinh, cosh, tanh, arsinh, arcosh, artanh |
| `EML/Trig.lean` | Lemmas 22-24: A, cos, sin, tan, arcsin, arccos, arctan |
| `EML/Theorem1.lean` | Table 1, Table 2 and the statement `theorem1` |
| `EML/CutPoints.lean` | nine single instances at points on the branch cut, e.g. H(0) = 1, arcsin(1) = pi/2; nothing depends on this file |
| `EML.lean` | imports the eight files above |
| `lean-toolchain`, `lakefile.toml`, `lake-manifest.json` | Lean version and the Mathlib version |

### What has to be read

To know what is proved, two places are enough.

1. `EML/Basic.lean`: the definitions `Term` and `eval`. A term is `1`, a
   variable, or `eml s t`. Its value is `exp u - log v` with the principal
   logarithm, and it is undefined (`none`) when `v = 0`.
2. `EML/Theorem1.lean`: for every primitive `P` the real function `P.val`,
   its domain `P.dom` and its term `P.witness`, and the statement

```lean
theorem theorem1 :
    (∀ P : Prim0, ∃ W : Term 0, ∀ a, eval W a = some ((P.val : ℝ) : ℂ)) ∧
    (∀ P : Prim1, ∃ W : Term 1, ∀ x ∈ P.dom, eval W ![(x : ℂ)] = some ((P.val x : ℝ) : ℂ)) ∧
    (∀ P : Prim2, ∃ W : Term 2, ∀ p ∈ P.dom,
      eval W ![(p.1 : ℂ), (p.2 : ℂ)] = some ((P.val p.1 p.2 : ℝ) : ℂ))
```

Everything else is checked by Lean.

### How to run

Install Lean 4 once, through `elan`:
https://leanprover-community.github.io/get_started.html

Then, in this folder:

```bash
lake exe cache get
lake build
```

The first command downloads the compiled Mathlib into `.lake/` (about 7 GB,
needed once). The second checks the eight files. The output must contain the
line

```
info: EML/Theorem1.lean:230:0: 'EML.theorem1' depends on axioms: [propext, Classical.choice, Quot.sound]
```

and end with

```
Build completed successfully (9056 jobs).
```

"Build completed successfully" means that Lean accepted every proof. The
axiom line lists what the proof of `theorem1` rests on: the three standard
axioms of Lean and Mathlib. An unfinished proof would show `sorryAx` there.

`lake build` remembers what it has checked. To check everything again
(about 45 seconds):

```bash
rm -r .lake/build
lake build
```

To check one file (about 3 seconds):

```bash
lake env lean EML/Theorem1.lean
```

### Reading a proof step by step

Open this folder in VS Code with the "Lean 4" extension and click inside a
proof. The panel "Lean Infoview" shows what is known and what remains to be
proved at that line.

### A wrong statement is rejected

Example: in `EML/Theorem1.lean` change the witness of the constant 2 from
`const_2` to `const_3` (line 61). Then `lake build` stops with

```
error: EML/Theorem1.lean:168:4: Type mismatch
  eval_const_2_real
has type
  eval const_2 ?m.65 = some ↑2
but is expected to have type
  eval Prim0.two.witness a = some ↑Prim0.two.val
```
