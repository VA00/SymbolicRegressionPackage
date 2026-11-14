module

public import EML.Arith

/-!
# Proof of Theorem 1, Step 5 (Lemmas 17–20)

Multiplication by `i` (`X`), the second branch of the logarithm (`T`),
the exact unit step (`H`), the square root on `[0, ∞)` and the hypotenuse.
This is where the closed endpoints (`√0`, `hypot(0,0)`) are obtained.
-/

@[expose] public section

namespace EML

open Complex Term
open scoped Real

variable {n : ℕ} {a : Fin n → ℂ} {s t : Term n} {u v z : ℂ} {x y : ℝ}

/-- `X(x) := S(M(U(x), i), -1)`. -/
def X (t : Term n) : Term n := S (M (U t) const_i) const_neg

/-- **Lemma 17.** For every real `x`: `X(x) = i x`. -/
theorem eval_X (ht : eval t a = some (x : ℂ)) : eval (X t) a = some (x * I) := by
  -- `M(x+i, i) = (x+i) i = ix - 1`
  have hM := eval_M (eval_U ht) eval_const_i (ofReal_add_I_ne_zero x) (ofReal_add_I_ne_one x)
    I_mem_slitPlane
  -- `S(ix-1, -1) = ix`
  have hS := eval_S hM eval_const_neg (mul_ne_zero (ofReal_add_I_ne_zero x) I_ne_zero)
    (by simp [Real.pi_pos]) (by simp [Real.pi_pos.le])
  exact eval_congr hS (by linear_combination I_sq)

/-- `T(z) := eml(L(iπ), E(eml(L(iπ), z)))`, the second branch of the logarithm. -/
def T (t : Term n) : Term n := eml (L const_ipi) (E (eml (L const_ipi) t))

/-- The value of `T` in Lemma 18: `Log z + 2πi` if `Im z < 0`, `Log z` otherwise. -/
noncomputable def Tval (z : ℂ) : ℂ := if z.im < 0 then log z + 2 * π * I else log z

/-- The computation in the proof of Lemma 18. -/
theorem Tval_eq (z : ℂ) : I * π - log (exp (I * π - log z)) = Tval z := by
  have him : (I * π - log z).im = π - arg z := by simp [log_im]
  unfold Tval
  split_ifs with h
  · -- `Im z < 0`: `Arg z ∈ (-π, 0)`, so `Im q ∈ (π, 2π)`, unwinding number `1`
    have h1 : arg z < 0 := arg_neg_iff.mpr h
    rw [log_exp_shift _ 1 (by rw [him]; push_cast; linarith [neg_pi_lt_arg z, Real.pi_pos])
      (by rw [him]; push_cast; linarith [neg_pi_lt_arg z, Real.pi_pos])]
    push_cast; ring
  · -- `Im z ≥ 0`: `Arg z ∈ [0, π]`, so `Im q ∈ [0, π]`, inside the strip
    have h1 : 0 ≤ arg z := arg_nonneg_iff.mpr (not_lt.mp h)
    rw [log_exp (by rw [him]; linarith [arg_le_pi z, Real.pi_pos])
      (by rw [him]; linarith [arg_le_pi z, Real.pi_pos])]
    ring

/-- Lemma 18: `exp (T(z)) = z` for `z ≠ 0`. -/
theorem exp_Tval (hz : z ≠ 0) : exp (Tval z) = z := by
  unfold Tval
  split_ifs
  · rw [exp_add, exp_two_pi_mul_I, mul_one, exp_log hz]
  · exact exp_log hz

/-- **Lemma 18.** For every `z ≠ 0`, `T(z)` is defined and has the value `Tval z`. -/
theorem eval_T (ht : eval t a = some z) (hz : z ≠ 0) : eval (T t) a = some (Tval z) := by
  have hc : (I * π : ℂ) ≠ 0 := mul_ne_zero I_ne_zero (ofReal_ne_zero.mpr Real.pi_ne_zero)
  have hL := eval_L (eval_const_ipi (a := a)) hc
  have h1 := eval_E (eval_eml hL ht hz)
  have h2 := eval_eml hL h1 (exp_ne_zero _)
  exact eval_congr h2 (by rw [exp_Lval hc, Tval_eq])

/-- **Lemma 18**, second formula: `T(z) - L(z)` is `2πi` if `Im z < 0` or `z` is a
negative real, and `0` otherwise. -/
theorem Tval_sub_Lval (z : ℂ) :
    Tval z - Lval z = if z.im < 0 ∨ (z.re < 0 ∧ z.im = 0) then 2 * π * I else 0 := by
  unfold Tval Lval
  by_cases h1 : z.im < 0
  · have h2 : ¬(z.re < 0 ∧ z.im = 0) := fun h => by linarith [h.2]
    rw [ite_eq_left h1, ite_eq_right h2, ite_eq_left (Or.inl h1)]
    ring
  · by_cases h2 : z.re < 0 ∧ z.im = 0
    · rw [ite_eq_right h1, ite_eq_left h2, ite_eq_left (Or.inr h2)]
      ring
    · rw [ite_eq_right h1, ite_eq_right h2, ite_eq_right (by tauto)]
      ring

/-- If `exp c = w` and `Re w < 0`, then `c ∉ {0, 1}`. -/
theorem ne_zero_one_of_exp_eq {w c : ℂ} (hexp : exp c = w) (hre : w.re < 0) : c ≠ 0 ∧ c ≠ 1 := by
  constructor
  · rintro rfl
    rw [exp_zero] at hexp
    rw [← hexp] at hre
    norm_num at hre
  · rintro rfl
    have h1 : (exp (1 : ℂ)).re = Real.exp 1 := by simp [exp_re]
    rw [← hexp, h1] at hre
    linarith [Real.exp_pos 1]

/-- The imaginary part of `z / (2πi)` is `-Re z / (2π)`. -/
theorem div_two_I_pi_im (z : ℂ) : (z / (2 * (I * π))).im = -z.re / (2 * π) := by
  have h : z / (2 * (I * (π : ℂ))) = -(z * I) / ((2 * π : ℝ) : ℂ) := by
    rw [div_eq_div_iff (by simp [Real.pi_ne_zero]) (by simp [Real.pi_ne_zero])]
    push_cast
    linear_combination (2 * (π : ℂ) * z) * I_sq
  rw [h, div_ofReal_im]
  simp

/-- `τ(x) := W/(N(x), W₊(1, W_sqr(x)))`. -/
def tau (t : Term n) : Term n := Wdiv (N t) (Wadd one (Wsqr t))
/-- `w(x) := M(U(τ(x)), i)`. -/
def wH (t : Term n) : Term n := M (U (tau t)) const_i
/-- `H(x) := S(D(T(w(x)), 2πi), D(L(w(x)), 2πi))`, the unit step. -/
def H (t : Term n) : Term n := S (D (T (wH t)) const_2ipi) (D (L (wH t)) const_2ipi)

/-- Lemma 19: `τ(x) = -x/(1+x²)` for every real `x`. -/
theorem eval_tau (ht : eval t a = some (x : ℂ)) :
    eval (tau t) a = some ((-x / (1 + x ^ 2) : ℝ) : ℂ) :=
  eval_Wdiv (eval_N_real ht) (eval_Wadd eval_one_real (eval_Wsqr ht)) (by positivity)

/-- Lemma 19: `w(x) = -1 + iτ(x)` for every real `x`. -/
theorem eval_wH (ht : eval t a = some (x : ℂ)) :
    eval (wH t) a = some (-1 + ((-x / (1 + x ^ 2) : ℝ) : ℂ) * I) :=
  eval_congr (eval_M (eval_U (eval_tau ht)) eval_const_i (ofReal_add_I_ne_zero _)
    (ofReal_add_I_ne_one _) I_mem_slitPlane) (by linear_combination I_sq)

/-- **Lemma 19.** For every real `x`, `H(x)` is defined, `H(x) = 0` for `x < 0`
and `H(x) = 1` for `x ≥ 0`. -/
theorem eval_H (ht : eval t a = some (x : ℂ)) :
    eval (H t) a = some (((if 0 ≤ x then 1 else 0 : ℝ)) : ℂ) := by
  have hpos : (0 : ℝ) < 1 + x ^ 2 := by positivity
  set τ : ℝ := -x / (1 + x ^ 2) with hτ
  have hw : eval (wH t) a = some (-1 + (τ : ℂ) * I) := eval_wH ht
  set w : ℂ := -1 + (τ : ℂ) * I with hwdef
  have hre : w.re = -1 := by simp [hwdef]
  have him : w.im = τ := by simp [hwdef]
  have hw0 : w ≠ 0 := ne_zero_of_re_ne_zero (by rw [hre]; norm_num)
  -- `|τ| ≤ 1`, hence `1 ≤ |w| ≤ 2`
  have hτ1 : |τ| ≤ 1 := by
    rw [hτ, abs_div, abs_neg, abs_of_pos hpos, div_le_one hpos]
    nlinarith [sq_abs x, sq_nonneg (|x| - 1), abs_nonneg x]
  have hn1 : 1 ≤ ‖w‖ := by
    have := abs_re_le_norm w
    rw [hre] at this
    simpa using this
  have hn2 : ‖w‖ ≤ 2 := by
    have := norm_le_abs_re_add_abs_im w
    rw [hre, him] at this
    norm_num at this
    linarith
  have hlog0 : 0 ≤ Real.log ‖w‖ := Real.log_nonneg hn1
  have hlog1 : Real.log ‖w‖ ≤ 1 := by
    have := Real.log_le_sub_one_of_pos (by linarith : 0 < ‖w‖)
    linarith
  -- `T(w)` and `L(w)` are defined and are not `0` or `1`
  have hT := eval_T hw hw0
  have hL := eval_L hw hw0
  obtain ⟨hT0, hT1⟩ := ne_zero_one_of_exp_eq (exp_Tval hw0) (by rw [hre]; norm_num)
  obtain ⟨hL0, hL1⟩ := ne_zero_one_of_exp_eq (exp_Lval hw0) (by rw [hre]; norm_num)
  -- the two divisions by `2πi`
  have hc : (2 * (I * π) : ℂ) ∈ slitPlane :=
    mem_slitPlane_of_im_ne_zero (by simp [Real.pi_ne_zero])
  have hc0 : (2 * (I * π) : ℂ) ≠ 0 := slitPlane_ne_zero hc
  have hDT := eval_D hT eval_const_2ipi hT0 hT1 hc
  have hDL := eval_D hL eval_const_2ipi hL0 hL1 hc
  -- the strip condition for the final subtraction: `Im b = -ln|w| / (2π)`
  have h2π : (0 : ℝ) < 2 * π := by positivity
  have hb : (Lval w / (2 * (I * π))).im = -Real.log ‖w‖ / (2 * π) := by
    rw [div_two_I_pi_im, Lval_re]
  have hb1 : -π < (Lval w / (2 * (I * π))).im := by
    rw [hb, neg_div, neg_lt_neg_iff, div_lt_iff₀ h2π]
    nlinarith [Real.pi_gt_three, mul_self_nonneg (π - 3)]
  have hb2 : (Lval w / (2 * (I * π))).im ≤ π := by
    have : -Real.log ‖w‖ / (2 * π) ≤ 0 :=
      div_nonpos_of_nonpos_of_nonneg (by linarith) h2π.le
    rw [hb]
    linarith [Real.pi_pos]
  have hS := eval_S hDT hDL (div_ne_zero hT0 hc0) hb1 hb2
  -- the value: `(T(w) - L(w)) / (2πi)`
  refine eval_congr hS ?_
  rw [← sub_div, Tval_sub_Lval]
  by_cases hx : 0 ≤ x
  · -- `x ≥ 0`: `τ ≤ 0`, so `Im w < 0`, or `w = -1`
    have hτ0 : τ ≤ 0 := div_nonpos_of_nonpos_of_nonneg (by linarith) hpos.le
    have hcond : w.im < 0 ∨ (w.re < 0 ∧ w.im = 0) := by
      rw [him, hre]
      rcases hτ0.lt_or_eq with h | h
      · exact Or.inl h
      · exact Or.inr ⟨by norm_num, h⟩
    rw [ite_eq_left hcond, ite_eq_left hx, ofReal_one, div_eq_one_iff_eq hc0]
    ring
  · -- `x < 0`: `τ > 0`, so `Im w > 0`
    have hτ0 : 0 < τ := div_pos (by linarith) hpos
    have hcond : ¬(w.im < 0 ∨ (w.re < 0 ∧ w.im = 0)) := by
      rw [him]
      rintro (h | ⟨_, h⟩) <;> linarith
    rw [ite_eq_right hcond, ite_eq_right hx]
    simp

/-- `W_√(x) := W₋(E(W_half[L(W₊(x, H(N(x))))]), H(N(x)))`. -/
def Wsqrt (t : Term n) : Term n := Wsub (E (Whalf (L (Wadd t (H (N t)))))) (H (N t))
/-- `W_hypot(x, y) := W_√(W₊(W_sqr(x), W_sqr(y)))`. -/
def Whypot (s t : Term n) : Term n := Wsqrt (Wadd (Wsqr s) (Wsqr t))

/-- **Lemma 20** (square root). For every real `x ≥ 0`, `W_√(x)` is defined and equals `√x`,
including `W_√(0) = 0`. -/
theorem eval_Wsqrt (ht : eval t a = some (x : ℂ)) (hx : 0 ≤ x) :
    eval (Wsqrt t) a = some ((Real.sqrt x : ℝ) : ℂ) := by
  have hH := eval_H (eval_N_real ht)
  rcases hx.eq_or_lt with h0 | hpos
  · -- `x = 0`: the reflected step is `1`, `g = 1`, and `√1 - 1 = 0`
    subst h0
    have hH' : eval (H (N t)) a = some ((1 : ℝ) : ℂ) := eval_congr hH (by simp)
    have hL := eval_L_pos (eval_Wadd ht hH') (by norm_num)
    have hE := eval_E_real (eval_Whalf hL)
    exact eval_congr (eval_Wsub hE hH') (by simp)
  · -- `x > 0`: the reflected step is `0`, `g = x`
    have hneg : ¬(0 ≤ -x) := by intro h; linarith
    have hH' : eval (H (N t)) a = some ((0 : ℝ) : ℂ) := eval_congr hH (by simp [hneg])
    have hL := eval_L_pos (eval_Wadd ht hH') (by linarith)
    have hE := eval_E_real (eval_Whalf hL)
    exact eval_congr (eval_Wsub hE hH') (by
      congr 1; rw [add_zero, sub_zero, Real.exp_half, Real.exp_log hpos])

/-- **Lemma 20** (hypotenuse). For all real `x, y`, `W_hypot(x, y)` is defined and equals
`√(x² + y²)`, including `W_hypot(0, 0) = 0`. -/
theorem eval_Whypot (hs : eval s a = some (x : ℂ)) (ht : eval t a = some (y : ℂ)) :
    eval (Whypot s t) a = some ((Real.sqrt (x ^ 2 + y ^ 2) : ℝ) : ℂ) :=
  eval_Wsqrt (eval_Wadd (eval_Wsqr hs) (eval_Wsqr ht)) (by positivity)

end EML
