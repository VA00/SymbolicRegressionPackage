module

public import EML.Theorem1b
public import EML.Real.Continuity
public import EML.Real.Circular

/-!
# Theorem 1c (Appendix C): single-term witnesses in extended real arithmetic

The primitives, their values (`P.val`) and their natural domains (`P.dom`) are those of
Theorem 1 (`EML.Theorem1`, Table 1): finite real inputs and finite real outputs. The value of a
node inside a term may be `-∞` or `+∞` (Definition C.1, `evalE`).

* `P.witness1c` — the term `w_P` of Theorem 1c (written with an overline in the paper), for 26
  of the 33 primitives: the 23 terms of Theorem 1b (`P.witness1b`), and three new ones for `√x`
  on `[0, ∞)` (Lemma C.2), `hypot(x, y)` on `ℝ²` and `arcosh x` on `[1, ∞)` (Corollary C.3).
* `theorem1c` — for each of the 26 primitives the term `w_P` is defined in extended real
  arithmetic at every point of the natural domain, and its value is the real number `P`.
* `theorem1c_circular` — no extended real term computes `sin x` or `cos x` on `ℝ`, or `tan x`
  on its natural domain. Of the two arguments of the paper, the one checked here
  (`EML.Real.Circular`) is Hardy's, for large `x`; Wilkie's o-minimality theorem is not used.

Proved in the files imported here, and listed at the end with the axioms they depend on:

* `closed_pi_iff` — a closed term gives `π` in extended real arithmetic if and only if a closed
  term gives `π` in strict real arithmetic;
* `exists_strict_on_subinterval` — Lemma C.4;
* `no_extended_of_no_strict` — a function with no strict real term on any open interval has no
  extended real term on any open interval;
* `continuousOn_valE`, `no_unit_step1c` — every term is continuous on its domain; no unit step.

Not formalized: that `arctan`, `arcsin`, `arccos` have no strict real term on any interval (the
paper uses Liouville's theorem in Rosenlicht's form).
-/

@[expose] public section

namespace EML

open Term

namespace Prim1

/-- The extended real witness of a unary primitive (Theorem 1c): the 12 terms of Table B.1 and
the new terms for `√` and `arcosh`; `none` for the six circular functions. -/
def witness1c : Prim1 → Option (Term 1)
  | exp => some (E x₁)
  | half => some (Whalf1b x₁)
  | minus => some (Wminus1b x₁)
  | sqr => some (Wsqr1b x₁)
  | sigmoid => some (Wsigma1b x₁)
  | sinh => some (Wsinh1b x₁)
  | cosh => some (Wcosh1b x₁)
  | tanh => some (Wtanh1b x₁)
  | arsinh => some (Warsinh1b x₁)
  | ln => some (L x₁)
  | inv => some (Winv1b x₁)
  | artanh => some (Wartanh1b x₁)
  | sqrt => some (Wsqrt1c x₁)
  | arcosh => some (Warcosh1c x₁)
  | sin | cos | arctan | tan | arcsin | arccos => none

end Prim1

namespace Prim2

/-- The extended real witness of a binary primitive (Theorem 1c): the 7 terms of Table B.1 and
the new term for `hypot`. For `logarithm` the first variable is the base. -/
def witness1c : Prim2 → Term 2
  | plus => Wadd1b x₂ y₂
  | subtract => Wsub1b x₂ y₂
  | times => Wmul1b x₂ y₂
  | avg => Wavg1b x₂ y₂
  | hypot => Whypot1c x₂ y₂
  | divide => Wdiv1b x₂ y₂
  | power => Wpow1b x₂ y₂
  | logarithm => Wlog1b x₂ y₂

end Prim2

/-- Theorem 1c for the constants `1`, `2`, `-1`, `e`: the closed terms of Theorem 1b. -/
theorem theorem1c_const (P : Prim0) (W : Term 0) (hW : P.witness1b = some W) (a : Fin 0 → ℝ) :
    evalE W a = some ((P.val : ℝ) : EReal) :=
  evalE_of_evalR (theorem1b_const P W hW a)

/-- Theorem 1c for its 14 unary primitives. -/
theorem theorem1c_unary (P : Prim1) (W : Term 1) (hW : P.witness1c = some W) (x : ℝ)
    (hx : x ∈ P.dom) : evalE W ![x] = some ((P.val x : ℝ) : EReal) := by
  have hV := strictVal_evalE ![x]
  have hv : ValE ![x] x₁ x := by simp [ValE, evalE]
  cases P <;> simp only [Prim1.witness1c, Option.some.injEq, reduceCtorEq] at hW <;> subst hW
  · exact val_E hV hv
  · exact val_Whalf1b hV hv
  · exact val_Wminus1b hV hv
  · exact val_Wsqr1b hV hv
  · exact val_Wsigma1b hV hv
  · exact val_Wsinh1b hV hv
  · exact val_Wcosh1b hV hv
  · exact val_Wtanh1b hV hv
  · exact val_Warsinh1b hV hv
  · exact val_L hV hv hx
  · exact val_Winv1b hV hv hx
  · exact evalE_Wsqrt1c hv hx
  · exact evalE_Warcosh1c hv hx
  · exact val_Wartanh1b hV hv hx.1 hx.2

/-- Theorem 1c for the 8 binary primitives. -/
theorem theorem1c_binary (P : Prim2) (x y : ℝ) (h : (x, y) ∈ P.dom) :
    evalE P.witness1c ![x, y] = some ((P.val x y : ℝ) : EReal) := by
  have hV := strictVal_evalE ![x, y]
  have hx : ValE ![x, y] x₂ x := by simp [ValE, evalE]
  have hy : ValE ![x, y] y₂ y := by simp [ValE, evalE]
  cases P
  · exact val_Wadd1b hV hx hy
  · exact val_Wsub1b hV hx hy
  · exact val_Wmul1b hV hx hy
  · exact val_Wavg1b hV hx hy
  · exact evalE_Whypot1c hx hy
  · exact val_Wdiv1b hV hx hy h
  · exact val_Wpow1b hV hx hy h
  · exact val_Wlog1b hV hx hy h.1 h.2.1 h.2.2

/-- **Theorem 1c**, first sentence (single-term witnesses with `ln 0 = -∞`). For each of the 26
primitives `P` of Theorem 1c — every primitive of Table 1 except `π`, `sin`, `cos`, `tan`,
`arcsin`, `arccos`, `arctan` — there is a term `W` such that, at every point of the natural
domain of `P`, `W` is defined in extended real arithmetic and its value is the real number `P`.
The domains are closed where Table 1 says so: `√0 = 0`, `hypot(0, 0) = 0`, `arcosh 1 = 0`. -/
theorem theorem1c :
    (∀ P : Prim0, P ≠ .pi → ∃ W : Term 0, ∀ a, evalE W a = some ((P.val : ℝ) : EReal)) ∧
    (∀ P : Prim1, P ≠ .sin → P ≠ .cos → P ≠ .tan → P ≠ .arcsin → P ≠ .arccos → P ≠ .arctan →
      ∃ W : Term 1, ∀ x ∈ P.dom, evalE W ![x] = some ((P.val x : ℝ) : EReal)) ∧
    (∀ P : Prim2,
      ∃ W : Term 2, ∀ p ∈ P.dom, evalE W ![p.1, p.2] = some ((P.val p.1 p.2 : ℝ) : EReal)) := by
  refine ⟨fun P h => ?_, fun P h1 h2 h3 h4 h5 h6 => ?_,
    fun P => ⟨P.witness1c, fun p hp => theorem1c_binary P p.1 p.2 hp⟩⟩
  · cases P <;> first
      | exact absurd rfl h
      | exact ⟨_, fun a => theorem1c_const _ _ rfl a⟩
  · cases P <;> first
      | exact absurd rfl h1 | exact absurd rfl h2 | exact absurd rfl h3 | exact absurd rfl h4
      | exact absurd rfl h5 | exact absurd rfl h6
      | exact ⟨_, fun x hx => theorem1c_unary _ _ rfl x hx⟩

/-- **Theorem 1c**, second sentence, for `sin`, `cos`, `tan`. No extended real term is equal to
`sin x`, to `cos x` or to `tan x` on the whole natural domain of Table 1. -/
theorem theorem1c_circular :
    (¬ ∃ W : Term 1, ∀ x ∈ Prim1.sin.dom, evalE W ![x] = some ((Prim1.sin.val x : ℝ) : EReal)) ∧
    (¬ ∃ W : Term 1, ∀ x ∈ Prim1.cos.dom, evalE W ![x] = some ((Prim1.cos.val x : ℝ) : EReal)) ∧
    (¬ ∃ W : Term 1, ∀ x ∈ Prim1.tan.dom, evalE W ![x] = some ((Prim1.tan.val x : ℝ) : EReal)) :=
  ⟨fun ⟨W, hW⟩ => no_sin1c ⟨W, fun x => hW x trivial⟩,
    fun ⟨W, hW⟩ => no_cos1c ⟨W, fun x => hW x trivial⟩,
    fun ⟨W, hW⟩ => no_tan1c ⟨W, fun x hx => hW x hx⟩⟩

end EML

-- The axioms the proofs depend on. Expected: only Lean's three standard axioms
-- `propext`, `Classical.choice`, `Quot.sound`; in particular no `sorryAx`.
#print axioms EML.theorem1c
#print axioms EML.theorem1c_circular
#print axioms EML.closed_pi_iff
#print axioms EML.exists_strict_on_subinterval
#print axioms EML.no_extended_of_no_strict
#print axioms EML.continuousOn_valE
#print axioms EML.no_unit_step1c
