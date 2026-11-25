module

public import EML.Theorem1
public import EML.Real.Analytic
public import EML.Real.Class
public import EML.Real.Hardy

/-!
# Theorem 1b (Appendix B): single-term witnesses in strict real arithmetic

The primitives, their values (`P.val`) and their natural domains (`P.dom`) are those of
Theorem 1 (`EML.Theorem1`, Table 1). New here:

* `P.witness1b` — the strict real `eml` term `w_P` of Table B.1, for 23 of the 33 primitives:
  4 constants, 12 unary functions, 7 binary operations. It is `none` for the constant `π` and
  for the nine functions of Table B.2.
* `theorem1b` — for each of the 23 primitives the term `w_P` is defined in strict real
  arithmetic (`evalR`) at every point of the natural domain, and its value is `P`.
* `theorem1b_restricted` — strict real witnesses for `√x` on `x > 0`, for `hypot(x, y)` on
  `(x, y) ≠ (0, 0)` and for `arcosh x` on `x > 1`.
* `theorem1b_endpoints` — no strict real term computes `√x`, `hypot(x, y)` or `arcosh x` on the
  full natural domain `[0, ∞)`, `ℝ²`, `[1, ∞)`.
* `theorem1b_circular` — no strict real term computes `sin x` or `cos x` on `ℝ`, or `tan x` on
  its natural domain. Of the two arguments of the paper, the one checked here
  (`EML.Real.Hardy`) is Hardy's, for large `x`; Wilkie's o-minimality theorem is not used.

Not formalized: that `arctan`, `arcsin`, `arccos` have no strict real term on any interval (the
paper uses Liouville's theorem in Rosenlicht's form; of that theorem Mathlib 4.35 has only the
case of algebraic extensions).
-/

@[expose] public section

namespace EML

open Term

namespace Prim0

/-- The strict real witness of a constant (Table B.1). There is none for `π`: whether a closed
strict real term has the value `π` is left open in the paper. -/
def witness1b : Prim0 → Option (Term 0)
  | one => some Term.one
  | two => some const_2_1b
  | neg => some const_neg1b
  | e => some const_e
  | pi => none

end Prim0

namespace Prim1

/-- The strict real witness of a unary primitive (Table B.1); `none` for the eight unary
functions of Table B.2. -/
def witness1b : Prim1 → Option (Term 1)
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
  | sin | cos | arctan | sqrt | tan | arcsin | arccos | arcosh => none

end Prim1

namespace Prim2

/-- The strict real witness of a binary primitive (Table B.1); `none` for `hypot`
(Table B.2). For `logarithm` the first variable is the base. -/
def witness1b : Prim2 → Option (Term 2)
  | plus => some (Wadd1b x₂ y₂)
  | subtract => some (Wsub1b x₂ y₂)
  | times => some (Wmul1b x₂ y₂)
  | avg => some (Wavg1b x₂ y₂)
  | divide => some (Wdiv1b x₂ y₂)
  | power => some (Wpow1b x₂ y₂)
  | logarithm => some (Wlog1b x₂ y₂)
  | hypot => none

end Prim2

/-- Theorem 1b for the constants `1`, `2`, `-1`, `e`. -/
theorem theorem1b_const (P : Prim0) (W : Term 0) (hW : P.witness1b = some W) (a : Fin 0 → ℝ) :
    evalR W a = some P.val := by
  have hV := strictVal_evalR a
  cases P <;> simp only [Prim0.witness1b, Option.some.injEq, reduceCtorEq] at hW <;> subst hW
  · exact evalR_one
  · exact val_const_2_1b hV
  · exact val_const_neg1b hV
  · exact val_const_e hV

/-- Theorem 1b for the 12 unary primitives of Table B.1. -/
theorem theorem1b_unary (P : Prim1) (W : Term 1) (hW : P.witness1b = some W) (x : ℝ)
    (hx : x ∈ P.dom) : evalR W ![x] = some (P.val x) := by
  have hV := strictVal_evalR ![x]
  have hv : ValR ![x] x₁ x := by simp [ValR]
  cases P <;> simp only [Prim1.witness1b, Option.some.injEq, reduceCtorEq] at hW <;> subst hW
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
  · exact val_Wartanh1b hV hv hx.1 hx.2

/-- Theorem 1b for the 7 binary primitives of Table B.1. -/
theorem theorem1b_binary (P : Prim2) (W : Term 2) (hW : P.witness1b = some W) (x y : ℝ)
    (h : (x, y) ∈ P.dom) : evalR W ![x, y] = some (P.val x y) := by
  have hV := strictVal_evalR ![x, y]
  have hx : ValR ![x, y] x₂ x := by simp [ValR]
  have hy : ValR ![x, y] y₂ y := by simp [ValR]
  cases P <;> simp only [Prim2.witness1b, Option.some.injEq, reduceCtorEq] at hW <;> subst hW
  · exact val_Wadd1b hV hx hy
  · exact val_Wsub1b hV hx hy
  · exact val_Wmul1b hV hx hy
  · exact val_Wavg1b hV hx hy
  · exact val_Wdiv1b hV hx hy h
  · exact val_Wpow1b hV hx hy h
  · exact val_Wlog1b hV hx hy h.1 h.2.1 h.2.2

/-- **Theorem 1b**, first sentence (single-term witnesses in strict real arithmetic). For each
of the 23 primitives `P` of Table B.1 — every primitive of Table 1 except `π`, `sin`, `cos`,
`tan`, `arcsin`, `arccos`, `arctan`, `√`, `arcosh` and `hypot` — there is a term `W` such that,
at every point of the natural domain of `P`, `W` is defined in strict real arithmetic and its
value is `P`. -/
theorem theorem1b :
    (∀ P : Prim0, P ≠ .pi → ∃ W : Term 0, ∀ a, evalR W a = some P.val) ∧
    (∀ P : Prim1, P ≠ .sin → P ≠ .cos → P ≠ .tan → P ≠ .arcsin → P ≠ .arccos → P ≠ .arctan →
      P ≠ .sqrt → P ≠ .arcosh → ∃ W : Term 1, ∀ x ∈ P.dom, evalR W ![x] = some (P.val x)) ∧
    (∀ P : Prim2, P ≠ .hypot →
      ∃ W : Term 2, ∀ p ∈ P.dom, evalR W ![p.1, p.2] = some (P.val p.1 p.2)) := by
  refine ⟨fun P h => ?_, fun P h1 h2 h3 h4 h5 h6 h7 h8 => ?_, fun P h => ?_⟩
  · cases P <;> first
      | exact absurd rfl h
      | exact ⟨_, fun a => theorem1b_const _ _ rfl a⟩
  · cases P <;> first
      | exact absurd rfl h1 | exact absurd rfl h2 | exact absurd rfl h3 | exact absurd rfl h4
      | exact absurd rfl h5 | exact absurd rfl h6 | exact absurd rfl h7 | exact absurd rfl h8
      | exact ⟨_, fun x hx => theorem1b_unary _ _ rfl x hx⟩
  · cases P <;> first
      | exact absurd rfl h
      | exact ⟨_, fun p hp => theorem1b_binary _ _ rfl p.1 p.2 hp⟩

/-- **Theorem 1b**, last sentence. There are strict real witnesses for `√x` on `x > 0`, for
`hypot(x, y)` on `(x, y) ≠ (0, 0)`, and for `arcosh x` on `x > 1`. -/
theorem theorem1b_restricted :
    (∃ W : Term 1, ∀ x : ℝ, 0 < x → evalR W ![x] = some (Real.sqrt x)) ∧
    (∃ W : Term 2, ∀ x y : ℝ, (x ≠ 0 ∨ y ≠ 0) →
      evalR W ![x, y] = some (Real.sqrt (x ^ 2 + y ^ 2))) ∧
    (∃ W : Term 1, ∀ x : ℝ, 1 < x → evalR W ![x] = some (Real.arcosh x)) := by
  refine ⟨⟨Wsqrt1b x₁, fun x hx => ?_⟩, ⟨Whypot1b x₂ y₂, fun x y h => ?_⟩,
    ⟨Warcosh1b x₁, fun x hx => ?_⟩⟩
  · exact val_Wsqrt1b (strictVal_evalR ![x]) (by simp [ValR]) hx
  · exact val_Whypot1b (strictVal_evalR ![x, y]) (by simp [ValR]) (by simp [ValR]) h
  · exact val_Warcosh1b (strictVal_evalR ![x]) (by simp [ValR]) hx

/-- **Theorem 1b**, second sentence, for the three functions with an endpoint obstruction.
No strict real term is equal to `√x`, to `hypot(x, y)` or to `arcosh x` on the whole natural
domain of Table 1. -/
theorem theorem1b_endpoints :
    (¬ ∃ W : Term 1, ∀ x ∈ Prim1.sqrt.dom, evalR W ![x] = some (Prim1.sqrt.val x)) ∧
    (¬ ∃ W : Term 2, ∀ p ∈ Prim2.hypot.dom,
      evalR W ![p.1, p.2] = some (Prim2.hypot.val p.1 p.2)) ∧
    (¬ ∃ W : Term 1, ∀ x ∈ Prim1.arcosh.dom, evalR W ![x] = some (Prim1.arcosh.val x)) := by
  refine ⟨fun ⟨W, hW⟩ => no_sqrt1b ⟨W, fun x hx => hW x hx⟩,
    fun ⟨W, hW⟩ => no_hypot1b ⟨W, fun x y => hW (x, y) trivial⟩,
    fun ⟨W, hW⟩ => no_arcosh1b ⟨W, fun x hx => hW x hx⟩⟩

/-- **Theorem 1b**, second sentence, for `sin`, `cos`, `tan`. No strict real term is equal to
`sin x`, to `cos x` or to `tan x` on the whole natural domain of Table 1. -/
theorem theorem1b_circular :
    (¬ ∃ W : Term 1, ∀ x ∈ Prim1.sin.dom, evalR W ![x] = some (Prim1.sin.val x)) ∧
    (¬ ∃ W : Term 1, ∀ x ∈ Prim1.cos.dom, evalR W ![x] = some (Prim1.cos.val x)) ∧
    (¬ ∃ W : Term 1, ∀ x ∈ Prim1.tan.dom, evalR W ![x] = some (Prim1.tan.val x)) :=
  ⟨fun ⟨W, hW⟩ => no_sin1b ⟨W, fun x => hW x trivial⟩,
    fun ⟨W, hW⟩ => no_cos1b ⟨W, fun x => hW x trivial⟩,
    fun ⟨W, hW⟩ => no_tan1b ⟨W, fun x hx => hW x hx⟩⟩

end EML

-- The axioms the proofs depend on. Expected: only Lean's three standard axioms
-- `propext`, `Classical.choice`, `Quot.sound`; in particular no `sorryAx`.
#print axioms EML.theorem1b
#print axioms EML.theorem1b_restricted
#print axioms EML.theorem1b_endpoints
#print axioms EML.theorem1b_circular
#print axioms EML.corollary_B5
