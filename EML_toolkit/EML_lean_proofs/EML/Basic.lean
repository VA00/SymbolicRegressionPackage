module

public import Mathlib

/-!
# EML: definitions and the two classical facts (Section 2.1 of the paper)

* `Term n`   — Definition 2: EML terms in `n` variables, `T ::= 1 | x_j | eml(T,T)`.
* `eval t a` — Definition 2: the value of the term `t` at the point `a`.
  It is an `Option ℂ`: `some z` means "defined, with value `z`",
  `none` means "undefined" (some logarithm inside `t` is applied to `0`).
* `F1`, `F2` — the facts (F1), (F2) about `exp` and the principal `Log`.

Mathlib's `Complex.log` is the principal branch, `Arg ∈ (-π, π]`, as in the paper.
Mathlib also sets `Complex.log 0 = 0`. That convention is never used here:
`eval` returns `none` as soon as the second argument of an `eml` is `0`.
-/

@[expose] public section

namespace EML

open Complex

/-- Definition 2. EML terms in the variables `x_0, …, x_{n-1}`. -/
inductive Term (n : ℕ) : Type
  /-- the constant leaf `1` -/
  | one : Term n
  /-- the variable leaf `x_j` -/
  | var : Fin n → Term n
  /-- the node `eml(S, T)` -/
  | eml : Term n → Term n → Term n

open Term

variable {n : ℕ} {a : Fin n → ℂ} {s t : Term n} {u v z : ℂ}

/-- Definition 2. Value of a term at the point `a`.
`eml(S,T)` is defined iff `S` and `T` are defined and the value of `T` is not `0`;
its value is then `exp (value of S) - Log (value of T)` (Definition 1). -/
noncomputable def eval : Term n → (Fin n → ℂ) → Option ℂ
  | one, _ => some 1
  | var j, a => some (a j)
  | eml s t, a =>
    match eval s a, eval t a with
    | some u, some v => if v = 0 then none else some (exp u - log v)
    | _, _ => none

/-- The domain of a term: the points where it is defined. -/
def dom (t : Term n) : Set (Fin n → ℂ) := {a | (eval t a).isSome}

@[simp] theorem eval_one : eval (one : Term n) a = some 1 := rfl

@[simp] theorem eval_var (j : Fin n) : eval (var j) a = some (a j) := rfl

/-- The rule for a node whose second argument is not `0`. -/
theorem eval_eml (hs : eval s a = some u) (ht : eval t a = some v) (hv : v ≠ 0) :
    eval (eml s t) a = some (exp u - log v) := by
  simp [eval, hs, ht, hv]

/-- A node whose second argument has the value `0` is undefined. -/
theorem eval_eml_log_zero (ht : eval t a = some 0) : eval (eml s t) a = none := by
  cases h : eval s a <;> simp [eval, h, ht]

/-- A node with an undefined first argument is undefined. -/
theorem eval_eml_undef_left (hs : eval s a = none) : eval (eml s t) a = none := by
  simp [eval, hs]

/-- A node with an undefined second argument is undefined. -/
theorem eval_eml_undef_right (ht : eval t a = none) : eval (eml s t) a = none := by
  cases h : eval s a <;> simp [eval, h, ht]

/-- If a term has the value `z` and `z = z'`, it has the value `z'`. -/
theorem eval_congr (h : eval t a = some z) {z' : ℂ} (hz : z = z') : eval t a = some z' :=
  hz ▸ h

/-! ## The facts (F1) and (F2) -/

/-- (F1): `exp (Log z) = z` for every `z ≠ 0`. -/
theorem F1 (hz : z ≠ 0) : exp (log z) = z := exp_log hz

/-- If `w - 2πik` lies in the strip `-π < Im ≤ π`, then `Log (exp w) = w - 2πik`. -/
theorem log_exp_shift (w : ℂ) (k : ℤ) (h₁ : -Real.pi < w.im - 2 * Real.pi * k) (h₂ : w.im - 2 * Real.pi * k ≤ Real.pi) :
    log (exp w) = w - k * (2 * Real.pi * I) := by
  have hexp : exp w = exp (w - k * (2 * Real.pi * I)) := by
    rw [exp_sub, exp_int_mul_two_pi_mul_I, div_one]
  have him : (w - k * (2 * Real.pi * I)).im = w.im - 2 * Real.pi * k := by
    simp; ring
  rw [hexp]
  exact log_exp (by rw [him]; exact h₁) (by rw [him]; exact h₂)

/-- (F2): `Log (exp w) = w - 2πi·K(w)`, where `K(w) = ⌈(Im w - π)/(2π)⌉` is the unwinding number. -/
theorem F2 (w : ℂ) : log (exp w) = w - (⌈(w.im - Real.pi) / (2 * Real.pi)⌉ : ℤ) * (2 * Real.pi * I) := by
  have hpos : (0 : ℝ) < 2 * Real.pi := by positivity
  apply log_exp_shift
  · have h := Int.ceil_lt_add_one ((w.im - Real.pi) / (2 * Real.pi))
    have h' : (⌈(w.im - Real.pi) / (2 * Real.pi)⌉ : ℝ) - 1 < (w.im - Real.pi) / (2 * Real.pi) := by linarith
    rw [lt_div_iff₀ hpos] at h'
    linarith
  · have h := Int.le_ceil ((w.im - Real.pi) / (2 * Real.pi))
    rw [div_le_iff₀ hpos] at h
    linarith

/-- (F2), exact case: `Log (exp w) = w` inside the strip `-π < Im w ≤ π`. -/
theorem F2_strip {w : ℂ} (h₁ : -Real.pi < w.im) (h₂ : w.im ≤ Real.pi) : log (exp w) = w := log_exp h₁ h₂

/-- (F2), converse: `Log (exp w) = w` only inside the strip. -/
theorem F2_strip_iff (w : ℂ) : log (exp w) = w ↔ -Real.pi < w.im ∧ w.im ≤ Real.pi := by
  constructor
  · intro h
    rw [← h, log_im]
    exact ⟨neg_pi_lt_arg _, arg_le_pi _⟩
  · exact fun h => log_exp h.1 h.2

/-! ## Small helpers about real and imaginary parts -/

theorem ne_zero_of_im_ne_zero (h : z.im ≠ 0) : z ≠ 0 := by
  rintro rfl; simp at h

theorem ne_one_of_im_ne_zero (h : z.im ≠ 0) : z ≠ 1 := by
  rintro rfl; simp at h

theorem ne_zero_of_re_ne_zero (h : z.re ≠ 0) : z ≠ 0 := by
  rintro rfl; simp at h

theorem mem_slitPlane_of_im_ne_zero (h : z.im ≠ 0) : z ∈ slitPlane :=
  mem_slitPlane_iff.mpr (Or.inr h)

theorem I_mem_slitPlane : I ∈ slitPlane := mem_slitPlane_of_im_ne_zero (by simp)

/-- Real numbers lie in the strip of (F2). -/
theorem neg_pi_lt_ofReal_im (x : ℝ) : -Real.pi < (x : ℂ).im := by simp [Real.pi_pos]

theorem ofReal_im_le_pi (x : ℝ) : (x : ℂ).im ≤ Real.pi := by simp [Real.pi_pos.le]

theorem ofReal_im_lt_pi (x : ℝ) : (x : ℂ).im < Real.pi := by simp [Real.pi_pos]

/-- The number `i` lies in the strip of (F2): `Im i = 1`. -/
theorem neg_pi_lt_I_im : -Real.pi < I.im := by rw [I_im]; linarith [Real.pi_pos]

theorem I_im_le_pi : I.im ≤ Real.pi := by rw [I_im]; linarith [Real.pi_gt_three]

/-- Two complex numbers with different real parts are different. -/
theorem ne_of_re_lt {w : ℂ} (h : w.re < z.re) : z ≠ w := by
  rintro rfl; exact lt_irrefl _ h

/-- `x + i ≠ 0` and `x + i ≠ 1` for real `x`. -/
theorem ofReal_add_I_ne_zero (x : ℝ) : (x : ℂ) + I ≠ 0 := ne_zero_of_im_ne_zero (by simp)

theorem ofReal_add_I_ne_one (x : ℝ) : (x : ℂ) + I ≠ 1 := ne_one_of_im_ne_zero (by simp)

end EML
