module

public import EML.Core

/-!
# Theorem 1a, Section A.1: the EML operator on ℂ ∪ {∞₋, ∞₊}

* `XC`        — the values of Appendix A: a complex number `num z`, or one of the two
  directed infinities `ninf` (∞₋) and `pinf` (∞₊).
* `EMLop z w` — Definition A.1: `EML(z, w) = exp z - Log w`, with the values of Table A.1.
  `none` means "undefined".
* `evalX t a` — the value of the term `t` at the point `a ∈ ℂⁿ`: Definition 2 with `EML`
  in place of `eml`. A node is defined if and only if both of its arguments are defined
  and Table A.1 gives a value.
* `evalX_of_eval` — a term that is defined in the sense of Definition 2 has the same value
  here, so the lemmas of Section 2 remain valid.

The terms (`Term n`) are those of Theorem 1; only their evaluation is new.
-/

@[expose] public section

namespace EML

open Complex Term

/-- The values of Appendix A: the complex numbers and the two directed infinities. -/
inductive XC
  /-- a complex number -/
  | num : ℂ → XC
  /-- `∞₋`, `DirectedInfinity[-1]` of the Wolfram Language -/
  | ninf : XC
  /-- `∞₊`, `DirectedInfinity[1]` of the Wolfram Language -/
  | pinf : XC

open XC

/-- **Definition A.1** and **Table A.1**: `EML(z, w) = exp z - Log w`, where
`exp ∞₋ = 0`, `exp ∞₊ = ∞₊`, `Log 0 = ∞₋` and `Log ∞₋ = Log ∞₊ = ∞₊`.
Rows: `z` a number, `∞₋`, `∞₊`. In each row: `w` a number (`0` or not), `∞₋`, `∞₊`.
`none` means undefined (`∞₊ - ∞₊`). -/
noncomputable def EMLop : XC → XC → Option XC
  | num z, num w => if w = 0 then some pinf else some (num (exp z - log w))
  | num _, ninf => some ninf
  | num _, pinf => some ninf
  | ninf, num w => if w = 0 then some pinf else some (num (-log w))
  | ninf, ninf => some ninf
  | ninf, pinf => some ninf
  | pinf, num _ => some pinf
  | pinf, ninf => none
  | pinf, pinf => none

variable {n : ℕ} {a : Fin n → ℂ} {s t : Term n} {u v : XC} {z w : ℂ}

/-- Definition 2 with `EML` in place of `eml`: the node `EML(S, T)` is defined if and only if
`S` and `T` are defined and Table A.1 gives a value at their values. -/
noncomputable def evalX : Term n → (Fin n → ℂ) → Option XC
  | one, _ => some (num 1)
  | var j, a => some (num (a j))
  | eml s t, a =>
    match evalX s a, evalX t a with
    | some u, some v => EMLop u v
    | _, _ => none

/-- The domain of a term in Appendix A: the points where it is defined. -/
def domX (t : Term n) : Set (Fin n → ℂ) := {a | (evalX t a).isSome}

@[simp] theorem evalX_one : evalX (one : Term n) a = some (num 1) := rfl

@[simp] theorem evalX_var (j : Fin n) : evalX (var j) a = some (num (a j)) := rfl

/-- The rule for a node: the value is read off Table A.1. -/
theorem evalX_eml (hs : evalX s a = some u) (ht : evalX t a = some v) :
    evalX (eml s t) a = EMLop u v := by
  simp [evalX, hs, ht]

/-- If a term has the value `u` and `u = u'`, it has the value `u'`. -/
theorem evalX_congr (h : evalX t a = some u) {u' : XC} (hu : u = u') : evalX t a = some u' :=
  hu ▸ h

/-- Section A.1: the operators `eml` and `EML` agree on `ℂ × ℂ*`. -/
theorem EMLop_num (hw : w ≠ 0) : EMLop (num z) (num w) = some (num (exp z - log w)) := by
  simp [EMLop, hw]

/-- Section A.1: a term that is defined at a point in the sense of Definition 2 has the same
value here. Hence the lemmas of Section 2 remain valid. -/
theorem evalX_of_eval {t : Term n} {z : ℂ} (h : eval t a = some z) :
    evalX t a = some (num z) := by
  induction t generalizing z with
  | one =>
    simp only [eval_one, Option.some.injEq] at h
    simp [h]
  | var j =>
    simp only [eval_var, Option.some.injEq] at h
    simp [h]
  | eml s t ihs iht =>
    cases hs : eval s a with
    | none => simp [eval, hs] at h
    | some u =>
      cases ht : eval t a with
      | none => simp [eval, hs, ht] at h
      | some v =>
        by_cases hv : v = 0
        · simp [eval, hs, ht, hv] at h
        · have hz : exp u - log v = z := by simpa [eval, hs, ht, hv] using h
          rw [evalX_eml (ihs hs) (iht ht), EMLop_num hv, hz]

end EML
