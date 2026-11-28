module

public import EML.Trig

/-!
# Theorem 1: single-term witnesses for all 33 primitives of Table 1

`Prim0`, `Prim1`, `Prim2` list the primitives of Table 1 by arity
(5 constants, 20 unary functions, 8 binary operations).  For each primitive `P`:

* `P.val`     — the real function (Mathlib's),
* `P.dom`     — its natural domain (Table 1),
* `P.witness` — the EML term `W_P` (Table 2).

`theorem1` says: for every `P` and every point of `P.dom`, the term `P.witness`
is defined and its value is the real number `P.val`.  The witness depends on
`P` only, not on the point: one term serves the whole domain.

Mathlib's real functions are total (for instance `Real.log 0 = 0`, `x / 0 = 0`).
These conventions play no role: every statement is restricted to `P.dom`.
-/

@[expose] public section

namespace EML

open Complex Term

/-- The 5 constants of Table 1. -/
inductive Prim0
  | one | two | neg | e | pi

/-- The 20 unary primitives of Table 1. -/
inductive Prim1
  | exp | half | minus | sqr | sigmoid | sin | cos | arctan | sinh | cosh | tanh | arsinh
  | ln | inv | sqrt | tan | arcsin | arccos | arcosh | artanh

/-- The 8 binary primitives of Table 1. -/
inductive Prim2
  | plus | subtract | times | avg | hypot | divide | power | logarithm

/-- The variable `x` of a unary term, and the variables `x`, `y` of a binary term. -/
abbrev x₁ : Term 1 := var 0
abbrev x₂ : Term 2 := var 0
abbrev y₂ : Term 2 := var 1

namespace Prim0

/-- The value of a constant. -/
noncomputable def val : Prim0 → ℝ
  | one => 1
  | two => 2
  | neg => -1
  | e => Real.exp 1
  | pi => Real.pi

/-- The witness of a constant: a closed term. -/
def witness : Prim0 → Term 0
  | one => Term.one
  | two => const_2
  | neg => const_neg
  | e => const_e
  | pi => const_pi

end Prim0

namespace Prim1

/-- The real function computed by a unary primitive. -/
noncomputable def val : Prim1 → ℝ → ℝ
  | exp => Real.exp
  | half => fun x => x / 2
  | minus => fun x => -x
  | sqr => fun x => x ^ 2
  | sigmoid => Real.sigmoid
  | sin => Real.sin
  | cos => Real.cos
  | arctan => Real.arctan
  | sinh => Real.sinh
  | cosh => Real.cosh
  | tanh => Real.tanh
  | arsinh => Real.arsinh
  | ln => Real.log
  | inv => fun x => x⁻¹
  | sqrt => Real.sqrt
  | tan => Real.tan
  | arcsin => Real.arcsin
  | arccos => Real.arccos
  | arcosh => Real.arcosh
  | artanh => Real.artanh

/-- The natural domain of a unary primitive (Table 1). -/
def dom : Prim1 → Set ℝ
  | ln => {x | 0 < x}
  | inv => {x | x ≠ 0}
  | sqrt => {x | 0 ≤ x}
  | tan => {x | Real.cos x ≠ 0}
  | arcsin => {x | -1 ≤ x ∧ x ≤ 1}
  | arccos => {x | -1 ≤ x ∧ x ≤ 1}
  | arcosh => {x | 1 ≤ x}
  | artanh => {x | -1 < x ∧ x < 1}
  | exp | half | minus | sqr | sigmoid | sin | cos | arctan | sinh | cosh | tanh | arsinh => Set.univ

/-- The witness of a unary primitive (Table 2). -/
def witness : Prim1 → Term 1
  | exp => E x₁
  | half => Whalf x₁
  | minus => N x₁
  | sqr => Wsqr x₁
  | sigmoid => Wsigma x₁
  | sin => Wsin x₁
  | cos => Wcos x₁
  | arctan => Warctan x₁
  | sinh => Wsinh x₁
  | cosh => Wcosh x₁
  | tanh => Wtanh x₁
  | arsinh => Warsinh x₁
  | ln => L x₁
  | inv => Winv x₁
  | sqrt => Wsqrt x₁
  | tan => Wtan x₁
  | arcsin => Warcsin x₁
  | arccos => Warccos x₁
  | arcosh => Warcosh x₁
  | artanh => Wartanh x₁

end Prim1

namespace Prim2

/-- The real function computed by a binary primitive. -/
noncomputable def val : Prim2 → ℝ → ℝ → ℝ
  | plus => fun x y => x + y
  | subtract => fun x y => x - y
  | times => fun x y => x * y
  | avg => fun x y => (x + y) / 2
  | hypot => fun x y => Real.sqrt (x ^ 2 + y ^ 2)
  | divide => fun x y => x / y
  | power => fun x y => x ^ y
  | logarithm => fun x y => Real.logb x y

/-- The natural domain of a binary primitive (Table 1). -/
def dom : Prim2 → Set (ℝ × ℝ)
  | divide => {p | p.2 ≠ 0}
  | power => {p | 0 < p.1}
  | logarithm => {p | 0 < p.1 ∧ p.1 ≠ 1 ∧ 0 < p.2}
  | plus | subtract | times | avg | hypot => Set.univ

/-- The witness of a binary primitive (Table 2). -/
def witness : Prim2 → Term 2
  | plus => Wadd x₂ y₂
  | subtract => Wsub x₂ y₂
  | times => Wmul x₂ y₂
  | avg => Wavg x₂ y₂
  | hypot => Whypot x₂ y₂
  | divide => Wdiv x₂ y₂
  | power => Wpow x₂ y₂
  | logarithm => Wlog x₂ y₂

end Prim2

/-- Theorem 1 for the constants. -/
theorem theorem1_const (P : Prim0) (a : Fin 0 → ℂ) :
    eval P.witness a = some ((P.val : ℝ) : ℂ) := by
  cases P
  · exact eval_one_real
  · exact eval_const_2_real
  · exact eval_congr eval_const_neg (by norm_num [Prim0.val])
  · exact eval_const_e
  · exact eval_const_pi

/-- Theorem 1 for the unary primitives. -/
theorem theorem1_unary (P : Prim1) (x : ℝ) (hx : x ∈ P.dom) :
    eval P.witness ![(x : ℂ)] = some ((P.val x : ℝ) : ℂ) := by
  have hv : eval x₁ ![(x : ℂ)] = some (x : ℂ) := by simp
  cases P
  · exact eval_E_real hv
  · exact eval_Whalf hv
  · exact eval_N_real hv
  · exact eval_Wsqr hv
  · exact eval_Wsigma hv
  · exact eval_Wsin hv
  · exact eval_Wcos hv
  · exact eval_Warctan hv
  · exact eval_Wsinh hv
  · exact eval_Wcosh hv
  · exact eval_Wtanh hv
  · exact eval_Warsinh hv
  · exact eval_L_pos hv hx
  · exact eval_Winv hv hx
  · exact eval_Wsqrt hv hx
  · exact eval_Wtan hv hx
  · exact eval_Warcsin hv hx.1 hx.2
  · exact eval_Warccos hv hx.1 hx.2
  · exact eval_Warcosh hv hx
  · exact eval_Wartanh hv hx.1 hx.2

/-- Theorem 1 for the binary primitives. -/
theorem theorem1_binary (P : Prim2) (x y : ℝ) (h : (x, y) ∈ P.dom) :
    eval P.witness ![(x : ℂ), (y : ℂ)] = some ((P.val x y : ℝ) : ℂ) := by
  have hx : eval x₂ ![(x : ℂ), (y : ℂ)] = some (x : ℂ) := by simp
  have hy : eval y₂ ![(x : ℂ), (y : ℂ)] = some (y : ℂ) := by simp
  cases P
  · exact eval_Wadd hx hy
  · exact eval_Wsub hx hy
  · exact eval_Wmul hx hy
  · exact eval_Wavg hx hy
  · exact eval_Whypot hx hy
  · exact eval_Wdiv hx hy h
  · exact eval_Wpow hx hy h
  · exact eval_Wlog hx hy h.1 h.2.1 h.2.2

/-- **Theorem 1** (single-term witnesses for all primitives). For every real primitive `P`
of Table 1 there is an EML term `W` such that, at every point of the natural domain of `P`,
`W` is defined and its value is the real number `P`. -/
theorem theorem1 :
    (∀ P : Prim0, ∃ W : Term 0, ∀ a, eval W a = some ((P.val : ℝ) : ℂ)) ∧
    (∀ P : Prim1, ∃ W : Term 1, ∀ x ∈ P.dom, eval W ![(x : ℂ)] = some ((P.val x : ℝ) : ℂ)) ∧
    (∀ P : Prim2, ∃ W : Term 2, ∀ p ∈ P.dom,
      eval W ![(p.1 : ℂ), (p.2 : ℂ)] = some ((P.val p.1 p.2 : ℝ) : ℂ)) :=
  ⟨fun P => ⟨P.witness, theorem1_const P⟩,
   fun P => ⟨P.witness, theorem1_unary P⟩,
   fun P => ⟨P.witness, fun p hp => theorem1_binary P p.1 p.2 hp⟩⟩

end EML

-- The axioms the proof depends on. Expected: only Lean's three standard axioms
-- `propext`, `Classical.choice`, `Quot.sound`; in particular no `sorryAx`.
#print axioms EML.theorem1
