module

public import EML.Complex.Functions

/-!
# Theorem 1a: single-term witnesses for all 34 complex primitives of Table A.2

`Prim0a`, `Prim1a`, `Prim2a` list the primitives of Table A.2 by arity
(6 constants, 20 unary functions, 8 binary operations), in the order of the table.
For each primitive `P`:

* `P.val`     — its value (Section A.2),
* `P.dom`     — its domain (Table A.2): the arguments at which `P` has a value in `ℂ`,
* `P.witness` — the EML term `𝒲_P` (Table A.3).

`theorem1a` says: for every `P` and every point of `P.dom`, the term `P.witness` is defined
in the semantics of Appendix A (`evalX`) and its value is the complex number `P.val`.
The witness depends on `P` only, not on the point: one term serves the whole domain,
branch cuts included.

Values: Mathlib's `Complex.exp`, `Complex.log` (principal branch), `Complex.sin`, `cos`, `tan`,
`sinh`, `cosh`, `tanh` and the complex power `z ^ w` (`exp (w Log z)` for `z ≠ 0`; the square
root is `z ^ (1/2)`). For `arcsin`, `arccos`, `arsinh`, `arcosh`, `arctan`, `artanh` the value
is the formula of Lemma A.11 (Section A.2). Mathlib's conventions outside the domains
(for instance `Complex.log 0 = 0`) play no role: every statement is restricted to `P.dom`.
-/

@[expose] public section

namespace EML

open Complex Term XC

/-- The 6 constants of Table A.2. -/
inductive Prim0a
  | one | two | neg | e | pi | i

/-- The 20 unary primitives of Table A.2. -/
inductive Prim1a
  | exp | half | minus | sqr | sin | cos | sinh | cosh | sqrt | arcsin | arccos | arsinh
  | arcosh | inv | ln | sigmoid | tan | tanh | arctan | artanh

/-- The 8 binary primitives of Table A.2. -/
inductive Prim2a
  | plus | subtract | times | avg | hypot | divide | power | logarithm

/-- The variable `z` of a unary term, and the variables `z`, `w` of a binary term
(for `log` the first variable is the base `b`). -/
abbrev z₁ : Term 1 := var 0
abbrev z₂ : Term 2 := var 0
abbrev w₂ : Term 2 := var 1

namespace Prim0a

/-- The value of a constant. -/
noncomputable def val : Prim0a → ℂ
  | one => 1
  | two => 2
  | neg => -1
  | e => ((Real.exp 1 : ℝ) : ℂ)
  | pi => ((Real.pi : ℝ) : ℂ)
  | i => Complex.I

/-- The witness of a constant: the closed term of Theorem 1. -/
def witness : Prim0a → Term 0
  | one => Term.one
  | two => const_2
  | neg => const_neg
  | e => const_e
  | pi => const_pi
  | i => const_i

end Prim0a

namespace Prim1a

/-- The value of a unary primitive. -/
noncomputable def val : Prim1a → ℂ → ℂ
  | exp => Complex.exp
  | half => fun z => z / 2
  | minus => fun z => -z
  | sqr => fun z => z ^ 2
  | sin => Complex.sin
  | cos => Complex.cos
  | sinh => Complex.sinh
  | cosh => Complex.cosh
  | sqrt => fun z => z ^ (1 / 2 : ℂ)
  | arcsin => fun z => -Complex.I * Complex.log (Complex.I * z + (1 - z ^ 2) ^ (1 / 2 : ℂ))
  | arccos => fun z =>
      (Real.pi : ℂ) / 2 - (-Complex.I * Complex.log (Complex.I * z + (1 - z ^ 2) ^ (1 / 2 : ℂ)))
  | arsinh => fun z => Complex.log (z + (1 + z ^ 2) ^ (1 / 2 : ℂ))
  | arcosh => fun z => Complex.log (z + (z - 1) ^ (1 / 2 : ℂ) * (z + 1) ^ (1 / 2 : ℂ))
  | inv => fun z => z⁻¹
  | ln => Complex.log
  | sigmoid => fun z => (1 + Complex.exp (-z))⁻¹
  | tan => Complex.tan
  | tanh => Complex.tanh
  | arctan => fun z =>
      (Complex.log (1 + Complex.I * z) - Complex.log (1 - Complex.I * z)) / (2 * Complex.I)
  | artanh => fun z => (Complex.log (1 + z) - Complex.log (1 - z)) / 2

/-- The domain of a unary primitive (Table A.2). -/
def dom : Prim1a → Set ℂ
  | inv => {z | z ≠ 0}
  | ln => {z | z ≠ 0}
  | sigmoid => {z | Complex.exp z ≠ -1}
  | tan => {z | Complex.cos z ≠ 0}
  | tanh => {z | Complex.cosh z ≠ 0}
  | arctan => {z | z ≠ Complex.I ∧ z ≠ -Complex.I}
  | artanh => {z | z ≠ 1 ∧ z ≠ -1}
  | exp | half | minus | sqr | sin | cos | sinh | cosh | sqrt | arcsin | arccos | arsinh
  | arcosh => Set.univ

/-- The witness of a unary primitive (Table A.3). -/
def witness : Prim1a → Term 1
  | exp => E z₁
  | half => Whalf1a z₁
  | minus => Wminus1a z₁
  | sqr => Wsqr1a z₁
  | sin => Wsin1a z₁
  | cos => Wcos1a z₁
  | sinh => Wsinh1a z₁
  | cosh => Wcosh1a z₁
  | sqrt => Wsqrt1a z₁
  | arcsin => Warcsin1a z₁
  | arccos => Warccos1a z₁
  | arsinh => Warsinh1a z₁
  | arcosh => Warcosh1a z₁
  | inv => Winv1a z₁
  | ln => Wln1a z₁
  | sigmoid => Wsigma1a z₁
  | tan => Wtan1a z₁
  | tanh => Wtanh1a z₁
  | arctan => Warctan1a z₁
  | artanh => Wartanh1a z₁

end Prim1a

namespace Prim2a

/-- The value of a binary primitive; for `logarithm` the first argument is the base. -/
noncomputable def val : Prim2a → ℂ → ℂ → ℂ
  | plus => fun z w => z + w
  | subtract => fun z w => z - w
  | times => fun z w => z * w
  | avg => fun z w => (z + w) / 2
  | hypot => fun z w => (z ^ 2 + w ^ 2) ^ (1 / 2 : ℂ)
  | divide => fun z w => z / w
  | power => fun z w => z ^ w
  | logarithm => fun b z => Complex.log z / Complex.log b

/-- The domain of a binary primitive (Table A.2). For `power`: `z ≠ 0`, or `z = 0` and
`Re w > 0`. For `logarithm`: the base `b ∉ {0, 1}` and `z ≠ 0`. -/
def dom : Prim2a → Set (ℂ × ℂ)
  | divide => {p | p.2 ≠ 0}
  | power => {p | p.1 ≠ 0 ∨ 0 < p.2.re}
  | logarithm => {p | p.1 ≠ 0 ∧ p.1 ≠ 1 ∧ p.2 ≠ 0}
  | plus | subtract | times | avg | hypot => Set.univ

/-- The witness of a binary primitive (Table A.3). -/
def witness : Prim2a → Term 2
  | plus => Wadd1a z₂ w₂
  | subtract => Wsub1a z₂ w₂
  | times => Wmul1a z₂ w₂
  | avg => Wavg1a z₂ w₂
  | hypot => Whypot1a z₂ w₂
  | divide => Wdiv1a z₂ w₂
  | power => Wpow1a z₂ w₂
  | logarithm => Wlog1a z₂ w₂

end Prim2a

/-- Theorem 1a for the constants. -/
theorem theorem1a_const (P : Prim0a) (a : Fin 0 → ℂ) :
    evalX P.witness a = some (num P.val) := by
  cases P with
  | one => exact evalX_one
  | two => exact evalX_const_2
  | neg => exact evalX_const_neg
  | e => exact evalX_const_e
  | pi => exact evalX_const_pi
  | i => exact evalX_const_i

/-- Theorem 1a for the unary primitives. -/
theorem theorem1a_unary (P : Prim1a) (z : ℂ) (hz : z ∈ P.dom) :
    evalX P.witness ![z] = some (num (P.val z)) := by
  have hv : evalX z₁ ![z] = some (num z) := by simp
  cases P with
  | exp => exact evalX_E hv
  | half => exact evalX_Whalf1a hv
  | minus => exact evalX_Wminus1a hv
  | sqr => exact evalX_Wsqr1a hv
  | sin => exact evalX_Wsin1a hv
  | cos => exact evalX_Wcos1a hv
  | sinh => exact evalX_Wsinh1a hv
  | cosh => exact evalX_Wcosh1a hv
  | sqrt => exact evalX_Wsqrt1a hv
  | arcsin => exact evalX_Warcsin1a hv
  | arccos => exact evalX_Warccos1a hv
  | arsinh => exact evalX_Warsinh1a hv
  | arcosh => exact evalX_Warcosh1a hv
  | inv => exact evalX_Winv1a hv hz
  | ln => exact evalX_Wln1a hv hz
  | sigmoid => exact evalX_Wsigma1a hv hz
  | tan => exact evalX_Wtan1a hv hz
  | tanh => exact evalX_Wtanh1a hv hz
  | arctan => exact evalX_Warctan1a hv hz.1 hz.2
  | artanh => exact evalX_Wartanh1a hv hz.1 hz.2

/-- Theorem 1a for the binary primitives. -/
theorem theorem1a_binary (P : Prim2a) (z w : ℂ) (h : (z, w) ∈ P.dom) :
    evalX P.witness ![z, w] = some (num (P.val z w)) := by
  have hz : evalX z₂ ![z, w] = some (num z) := by simp
  have hw : evalX w₂ ![z, w] = some (num w) := by simp
  cases P with
  | plus => exact evalX_Wadd1a hz hw
  | subtract => exact evalX_Wsub1a hz hw
  | times => exact evalX_Wmul1a hz hw
  | avg => exact evalX_Wavg1a hz hw
  | hypot => exact evalX_Whypot1a hz hw
  | divide => exact evalX_Wdiv1a hz hw h
  | power => exact evalX_Wpow1a_of_mem hz hw h
  | logarithm => exact evalX_Wlog1a hz hw h.1 h.2.1 h.2.2

/-- **Theorem 1a** (single-term witnesses for all complex primitives). For every primitive `P`
of Table A.2 there is an EML term `W` such that, at every point of the domain of `P`,
`W` is defined in the semantics of Appendix A and its value is the complex number `P`. -/
theorem theorem1a :
    (∀ P : Prim0a, ∃ W : Term 0, ∀ a, evalX W a = some (num P.val)) ∧
    (∀ P : Prim1a, ∃ W : Term 1, ∀ z ∈ P.dom, evalX W ![z] = some (num (P.val z))) ∧
    (∀ P : Prim2a, ∃ W : Term 2, ∀ p ∈ P.dom,
      evalX W ![p.1, p.2] = some (num (P.val p.1 p.2))) :=
  ⟨fun P => ⟨P.witness, theorem1a_const P⟩,
   fun P => ⟨P.witness, theorem1a_unary P⟩,
   fun P => ⟨P.witness, fun p hp => theorem1a_binary P p.1 p.2 hp⟩⟩

end EML

-- The axioms the proof depends on. Expected: only Lean's three standard axioms
-- `propext`, `Classical.choice`, `Quot.sound`; in particular no `sorryAx`.
#print axioms EML.theorem1a
