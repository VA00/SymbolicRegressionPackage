## EmL_proof: Lean 4 proofs of Theorems 1 and 1a

A machine-checked proof of Theorem 1 of the paper: each of the 33 real
primitives of Table 1 (5 constants, 20 unary functions, 8 binary operations)
has one EML term, built from `eml` and `1`, that is defined and equal to the
primitive at every point of its natural real domain. The proof follows Lemmas 1-24
of the paper, one Lean theorem per lemma.

The same for Theorem 1a of Appendix A: each of the 34 complex primitives of
Table A.2 (6 constants, 20 unary functions, 8 binary operations) has one EML term
that is defined and equal to the primitive at every point of its complex domain,
branch cuts included, in the semantics of Appendix A (values in the complex
numbers and the two infinities, Table A.1). The proof follows Lemmas A.1-A.11.

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
| `EML/Complex/Semantics.lean` | Appendix A: values with the two infinities (`XC`), the operator of Table A.1 (`EMLop`), the value of a term (`evalX`) |
| `EML/Complex/Arith.lean` | Lemmas A.1-A.5, Corollary A.6: the two infinities, E, L, S, inv, x, -, minus, +, sqr, half, avg, / |
| `EML/Complex/PowLog.lean` | Lemmas A.7, A.9, A.10, Corollary A.8: pow, sqrt, hypot, ln, log |
| `EML/Complex/Functions.lean` | Lemma A.11: sigma, sinh, cosh, cos, sin, tan, tanh and the six inverse functions |
| `EML/Theorem1a.lean` | Table A.2, Table A.3 and the statement `theorem1a` |
| `EML/Complex/ZeroTest.lean` | Section A.4: Lemma A.12 (test for zero J), Corollary A.13 (a prescribed value at one point) and the three examples (1/0 = 0 and Log 0 = 0 as in Lean, Log 0 = infinity_- as in the Wolfram Language), and d(z), equal to 1 on the whole negative real axis and 0 elsewhere, which replaces a term on that entire branch cut; nothing depends on this file |
| `EML.lean` | imports the fourteen files above |
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

For Theorem 1a, the same two places:

3. `EML/Complex/Semantics.lean`: the values `XC` (a complex number `num z`,
   `ninf` or `pinf`), Table A.1 as `EMLop`, and `evalX`, the value of a term.
   A node is undefined (`none`) only where Table A.1 says so.
4. `EML/Theorem1a.lean`: for every primitive `P` its value `P.val`, its domain
   `P.dom` and its term `P.witness`, and the statement

```lean
theorem theorem1a :
    (∀ P : Prim0a, ∃ W : Term 0, ∀ a, evalX W a = some (num P.val)) ∧
    (∀ P : Prim1a, ∃ W : Term 1, ∀ z ∈ P.dom, evalX W ![z] = some (num (P.val z))) ∧
    (∀ P : Prim2a, ∃ W : Term 2, ∀ p ∈ P.dom,
      evalX W ![p.1, p.2] = some (num (P.val p.1 p.2)))
```

The values are Mathlib's complex functions, except for arcsin, arccos, arsinh,
arcosh, arctan and artanh, whose value is the formula of Lemma A.11 (Mathlib has
no complex arcsin, arccos, arsinh, arcosh, artanh, and its `Complex.arctan` takes
the other value on the lower branch cut).

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
needed once). The second checks the fourteen files. The output must contain the
lines

```
info: EML/Theorem1a.lean:241:0: 'EML.theorem1a' depends on axioms: [propext, Classical.choice, Quot.sound]
info: EML/Theorem1.lean:229:0: 'EML.theorem1' depends on axioms: [propext, Classical.choice, Quot.sound]
```

and end with

```
Build completed successfully (9062 jobs).
```

"Build completed successfully" means that Lean accepted every proof. The
axiom lines list what the proofs of `theorem1` and `theorem1a` rest on: the
three standard axioms of Lean and Mathlib. An unfinished proof would show
`sorryAx` there.

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
`const_2` to `const_3` (line 60). Then `lake build` stops with

```
error: EML/Theorem1.lean:167:4: Type mismatch
  eval_const_2_real
has type
  eval const_2 ?m.65 = some ↑2
but is expected to have type
  eval Prim0.two.witness a = some ↑Prim0.two.val
```
