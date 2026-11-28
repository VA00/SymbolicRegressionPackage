## EmL_proof: Lean 4 proofs of Theorems 1, 1a, 1b and 1c

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

Theorem 1b of Appendix B: when every node is evaluated in the real numbers
(`exp u - ln v` with `v > 0`), 23 of the 33 primitives of Table 1 have one term
that is defined and equal to the primitive on the whole natural domain
(Lemmas B.1-B.4, Table B.1). `sqrt`, `hypot` and `arcosh` have such a term away
from the endpoints 0, (0, 0) and 1, and none on the whole domain (Lemmas B.6,
B.7). `sin`, `cos` and `tan` have none.

Theorem 1c of Appendix C: with `ln 0 = -infinity`, and the values `-infinity`
and `+infinity` allowed inside a term (Table C.1), 26 primitives have one term on
the whole natural domain: the 23 of Theorem 1b, and `sqrt` on [0, infinity),
`hypot` on R^2 and `arcosh` on [1, infinity) (Lemmas C.1, C.2, Corollary C.3).
`sin`, `cos` and `tan` have none.

Theorem 2: every calculator expression (Definition 3) has a term that is defined
and equal to it at every point of its natural domain; the term is the expression
with every primitive replaced by its term of Theorem 1. The same holds for
expressions over the complex primitives of Table A.2 whose intermediate values are
complex numbers, with the terms of Theorem 1a (the remark after Theorem 1a).

Corollary 25 (All rationals): every rational number is the value of a closed term
in which every node is real and every logarithm is applied to a positive number;
one term gives the floor of every real number, one the ceiling, and one the map
`R: x -> 1/(2 floor(x) - x + 1)` for `x != -1`; every positive rational number is the
value of exactly one of the closed terms `1, R(1), R(R(1)), ...` (the theorem of
Calkin, Wilf and Newman, proved here because Mathlib does not have it).

Not checked in Lean: that `arctan`, `arcsin` and `arccos` have no real term
(Appendix B uses Liouville's theorem, of which Mathlib has only the case of
algebraic extensions), and the remark on Schanuel's conjecture.

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
| `EML/Real/Strict.lean` | Appendix B: the value of a term in strict real arithmetic (`evalR`). Lemmas B.1-B.4 and the first half of B.7: E, L, F, B, +, -, the constants, G, x, sqr, inv, /, half, avg, sigma, pow, log, sinh, cosh, tanh, Q, arsinh, artanh; sqrt on x > 0, hypot off the origin, arcosh on x > 1 |
| `EML/Real/Analytic.lean` | Lemma B.6 (a term is analytic near every point of its domain); no strict real term for sqrt on [0, infinity), hypot on R^2, arcosh on [1, infinity) |
| `EML/Real/Class.lean` | Corollary B.5: strict real terms = real exp-log expressions (1, variables, +, -, x, /, exp, ln), with the same domains |
| `EML/Real/Hardy.lean` | Paragraph "Sine, cosine and tangent": for large x a strict real term in one variable is undefined, or 0, or positive, or negative; hence no strict real term for sin, cos, tan |
| `EML/Theorem1b.lean` | Table B.1 and the statements `theorem1b`, `theorem1b_restricted`, `theorem1b_endpoints`, `theorem1b_circular` |
| `EML/Real/Extended.lean` | Appendix C: the operator of Table C.1 on the real numbers with the two infinities (`emlE`), the value of a term (`evalE`). Lemma C.1 (E, L, F at the two ends), Lemma C.2 (sqrt on [0, infinity)) and the sentence after it, Corollary C.3 (hypot on R^2, arcosh on [1, infinity)) |
| `EML/Real/Reduction.lean` | Table C.1 read backwards; Lemma C.4; "Finite constants": the closed terms have the same finite values in both real arithmetics, in particular pi |
| `EML/Real/Continuity.lean` | "Absolute value, ReLU and the unit step": every extended real term is continuous on its domain; no unit step; abs, ReLU, max |
| `EML/Real/Circular.lean` | Paragraph "Direct circular functions": for large x an extended real term in one variable is undefined, +infinity, -infinity, or finite and equal to a strict real term; hence no extended real term for sin, cos, tan |
| `EML/Theorem1c.lean` | the witnesses of Theorem 1c and the statements `theorem1c`, `theorem1c_circular` |
| `EML/Rationals.lean` | Corollary 25 (All rationals): every rational number from a closed term with real nodes only (`corollary25_a`); floor and ceiling (`eval_Wfloor`, `eval_Wceil`); `R(x) = 1/(2 floor(x) - x + 1)` (`eval_R`) |
| `EML/CalkinWilf.lean` | Corollary 25 (c), second half: Stern's sequence `fusc`, the Calkin-Wilf sequence `cw`, Newman's formula (`cw_succ`), every positive rational exactly once (`exists_unique_cw`), and the terms `1, R(1), R(R(1)), ...` (`corollary25_c`) |
| `EML/Theorem2.lean` | Theorem 2: calculator expressions (`CalcExpr`, Definition 3), substitution of terms (`eval_subst`), and the statement `theorem2` |
| `EML/Theorem2a.lean` | Theorem 2 for the complex primitives of Table A.2 (the remark after Theorem 1a): `CalcExprC`, `evalX_subst`, `theorem2a` |
| `EML.lean` | imports the twenty-eight files above |
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

For Theorems 1b and 1c, four places. The primitives, `P.val` and `P.dom` are
those of Theorem 1.

5. `EML/Real/Strict.lean`: `evalR`, the value of a term in strict real
   arithmetic. It is `exp u - log v` with the real logarithm, and it is
   undefined (`none`) when `v` is not positive.
6. `EML/Theorem1b.lean`: the term `P.witness1b` of each of the 23 primitives
   (`none` for the others), and the statements

```lean
theorem theorem1b :
    (∀ P : Prim0, P ≠ .pi → ∃ W : Term 0, ∀ a, evalR W a = some P.val) ∧
    (∀ P : Prim1, P ≠ .sin → P ≠ .cos → P ≠ .tan → P ≠ .arcsin → P ≠ .arccos → P ≠ .arctan →
      P ≠ .sqrt → P ≠ .arcosh → ∃ W : Term 1, ∀ x ∈ P.dom, evalR W ![x] = some (P.val x)) ∧
    (∀ P : Prim2, P ≠ .hypot →
      ∃ W : Term 2, ∀ p ∈ P.dom, evalR W ![p.1, p.2] = some (P.val p.1 p.2))

theorem theorem1b_circular :
    (¬ ∃ W : Term 1, ∀ x ∈ Prim1.sin.dom, evalR W ![x] = some (Prim1.sin.val x)) ∧
    (¬ ∃ W : Term 1, ∀ x ∈ Prim1.cos.dom, evalR W ![x] = some (Prim1.cos.val x)) ∧
    (¬ ∃ W : Term 1, ∀ x ∈ Prim1.tan.dom, evalR W ![x] = some (Prim1.tan.val x))
```

   and `theorem1b_endpoints`, the same for `sqrt`, `hypot` and `arcosh`.
7. `EML/Real/Extended.lean`: Table C.1 as `emlE`, on Mathlib's extended real
   numbers `EReal` (`⊥` is -infinity, `⊤` is +infinity), and `evalE`, the value
   of a term. A node is undefined (`none`) only where Table C.1 says so.
8. `EML/Theorem1c.lean`: the terms of the 26 primitives (`P.witness1c`; for
   the constants those of Theorem 1b), the statement

```lean
theorem theorem1c :
    (∀ P : Prim0, P ≠ .pi → ∃ W : Term 0, ∀ a, evalE W a = some ((P.val : ℝ) : EReal)) ∧
    (∀ P : Prim1, P ≠ .sin → P ≠ .cos → P ≠ .tan → P ≠ .arcsin → P ≠ .arccos → P ≠ .arctan →
      ∃ W : Term 1, ∀ x ∈ P.dom, evalE W ![x] = some ((P.val x : ℝ) : EReal)) ∧
    (∀ P : Prim2,
      ∃ W : Term 2, ∀ p ∈ P.dom, evalE W ![p.1, p.2] = some ((P.val p.1 p.2 : ℝ) : EReal))
```

   and `theorem1c_circular`, the statement of `theorem1b_circular` with `evalE`.

Two remarks on these files. The lemmas on the 23 terms of Table B.1 are proved
once, for every evaluation in which the leaf 1 has the value 1 and a node with
real children `u` and `v > 0` has the value `exp u - ln v` (`StrictVal`); this
covers `evalR`, `evalE` and `eval`, and such lemmas are named `val_...`. For
`sin`, `cos` and `tan` the paper gives two arguments; the one checked here is
the second, Hardy's theorem (G. H. Hardy, Properties of logarithmico-exponential
functions, Proc. London Math. Soc. (2) 10 (1912), 54-90): beyond some point a
strict real term in one variable is undefined everywhere, or 0 everywhere, or of
one sign everywhere. Wilkie's theorem is not used.

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
needed once). The second checks the twenty-eight files. The output must contain
the lines

```
info: EML/Theorem1a.lean:241:0: 'EML.theorem1a' depends on axioms: [propext, Classical.choice, Quot.sound]
info: EML/Theorem1.lean:229:0: 'EML.theorem1' depends on axioms: [propext, Classical.choice, Quot.sound]
info: EML/Theorem1b.lean:194:0: 'EML.theorem1b' depends on axioms: [propext, Classical.choice, Quot.sound]
info: EML/Theorem1c.lean:158:0: 'EML.theorem1c' depends on axioms: [propext, Classical.choice, Quot.sound]
```

(and eighteen more lines of this form, for the other statements of Theorems 1b
and 1c, for Corollary 25 and for Theorems 2 and 2a) and end with

```
Build completed successfully (9076 jobs).
```

"Build completed successfully" means that Lean accepted every proof. The
axiom lines list what the proofs of `theorem1`, `theorem1a`, `theorem1b` and
`theorem1c` rest on: the three standard axioms of Lean and Mathlib. An
unfinished proof would show `sorryAx` there.

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
