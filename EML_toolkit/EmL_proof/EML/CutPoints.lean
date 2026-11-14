module

public import EML.Theorem1

/-!
# The points on the branch cut, as explicit instances

These follow from the lemmas of the proof of Theorem 1; they are listed because
they are the points where a numerical check cannot decide anything.
Nothing depends on this file.
-/

@[expose] public section

namespace EML

open Complex Term
open scoped Real

/-- `H(0) = 1`. -/
example : eval (H x₁) ![(0 : ℂ)] = some 1 := by
  simpa using eval_H (t := x₁) (a := ![(0 : ℂ)]) (x := 0) (by simp)

/-- `√0 = 0`. -/
example : eval (Wsqrt x₁) ![(0 : ℂ)] = some 0 := by
  simpa using eval_Wsqrt (t := x₁) (a := ![(0 : ℂ)]) (x := 0) (by simp) le_rfl

/-- `hypot(0, 0) = 0`. -/
example : eval (Whypot x₂ y₂) ![(0 : ℂ), 0] = some 0 := by
  simpa using eval_Whypot (s := x₂) (t := y₂) (a := ![(0 : ℂ), 0]) (x := 0) (y := 0)
    (by simp) (by simp)

/-- `arcosh 1 = 0`. -/
example : eval (Warcosh x₁) ![(1 : ℂ)] = some 0 := by
  simpa using eval_Warcosh (t := x₁) (a := ![(1 : ℂ)]) (x := 1) (by simp) le_rfl

/-- `arcsin 1 = π/2`. -/
example : eval (Warcsin x₁) ![(1 : ℂ)] = some ((π : ℂ) / 2) := by
  simpa using eval_Warcsin (t := x₁) (a := ![(1 : ℂ)]) (x := 1) (by simp) (by norm_num) le_rfl

/-- `arcsin (-1) = -π/2`. -/
example : eval (Warcsin x₁) ![(-1 : ℂ)] = some (-((π : ℂ) / 2)) := by
  simpa using eval_Warcsin (t := x₁) (a := ![(-1 : ℂ)]) (x := -1) (by simp) le_rfl (by norm_num)

/-- `arccos (-1) = π`. -/
example : eval (Warccos x₁) ![(-1 : ℂ)] = some (π : ℂ) := by
  simpa using eval_Warccos (t := x₁) (a := ![(-1 : ℂ)]) (x := -1) (by simp) le_rfl (by norm_num)

/-- `arccos 1 = 0`. -/
example : eval (Warccos x₁) ![(1 : ℂ)] = some 0 := by
  simpa using eval_Warccos (t := x₁) (a := ![(1 : ℂ)]) (x := 1) (by simp) (by norm_num) le_rfl

/-- The term `S(x, y)` is undefined at `x = 0` (remark after Lemma 4). -/
example (y : ℂ) : eval (S x₂ y₂) ![(0 : ℂ), y] = none := eval_S_zero (by simp)

end EML
