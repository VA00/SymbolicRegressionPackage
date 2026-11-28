module

public import EML.ExpLog

/-!
# Proof of Theorem 1, Step 7 (Lemmas 22–24): trigonometric functions and their inverses

`A` adds two complex numbers (shift by the real constant `3`);
then `cos`, `sin`, `tan`, `arcsin`, `arccos`, `arctan`,
including the endpoints `arcsin(±1)`, `arccos(±1)`.
-/

@[expose] public section

namespace EML

open Complex Term

variable {n : ℕ} {a : Fin n → ℂ} {s t : Term n} {u v : ℂ} {x y : ℝ}

/-- `A(u, v) := S(S(S(3, N(u)), N(v)), 3)`. -/
def A (s t : Term n) : Term n := S (S (S const_3 (N s)) (N t)) const_3

/-- **Lemma 22.** If `|Im u| < π`, `|Im v| < π`, `u ≠ -3` and `u + v ≠ -3`,
then `A(u, v)` is defined and equals `u + v`. -/
theorem eval_A (hs : eval s a = some u) (ht : eval t a = some v) (hu : |u.im| < Real.pi)
    (hv : |v.im| < Real.pi) (h₁ : u ≠ -3) (h₂ : u + v ≠ -3) : eval (A s t) a = some (u + v) := by
  rw [abs_lt] at hu hv
  have hNu := eval_N hs hu.1 hu.2
  have hNv := eval_N ht hv.1 hv.2
  -- `S(3, -u) = 3 + u`
  have h1 := eval_S eval_const_3 hNu three_ne_zero (by simp; linarith) (by simp; linarith)
  -- `S(3+u, -v) = 3 + u + v`
  have h2 := eval_S h1 hNv (by intro h; apply h₁; linear_combination h)
    (by simp; linarith) (by simp; linarith)
  -- `S(3+u+v, 3) = u + v`
  have h3 := eval_S h2 eval_const_3 (by intro h; apply h₂; linear_combination h)
    (by simp [Real.pi_pos]) (by simp [Real.pi_pos.le])
  exact eval_congr h3 (by ring)

/-- **Lemma 22**, last sentence: the hypotheses hold whenever `|u| ≤ 1` and `|v| ≤ 1`. -/
theorem eval_A_of_norm_le_one (hs : eval s a = some u) (ht : eval t a = some v)
    (hu : ‖u‖ ≤ 1) (hv : ‖v‖ ≤ 1) : eval (A s t) a = some (u + v) := by
  have h3 : ‖(-3 : ℂ)‖ = 3 := by simp
  refine eval_A hs ht ?_ ?_ ?_ ?_
  · linarith [abs_im_le_norm u, Real.pi_gt_three]
  · linarith [abs_im_le_norm v, Real.pi_gt_three]
  · intro h
    rw [h, h3] at hu
    norm_num at hu
  · intro h
    have := norm_add_le u v
    rw [h, h3] at this
    linarith

/-- `W_cos(x) := W_half(A(E(X(x)), E(X(N(x)))))`. -/
def Wcos (t : Term n) : Term n := Whalf (A (E (X t)) (E (X (N t))))

/-- **Lemma 23** (cosine). `W_cos(x) = cos x` for every real `x`. -/
theorem eval_Wcos (ht : eval t a = some (x : ℂ)) :
    eval (Wcos t) a = some ((Real.cos x : ℝ) : ℂ) := by
  -- `u = e^{ix}`, `v = e^{-ix}`, both of modulus `1`
  have hu := eval_E (eval_X ht)
  have hv := eval_E (eval_X (eval_N_real ht))
  have hA := eval_A_of_norm_le_one hu hv (norm_exp_ofReal_mul_I x).le
    (norm_exp_ofReal_mul_I (-x)).le
  -- `A(u, v) = 2 cos x`, a real number
  have hA' : eval (A (E (X t)) (E (X (N t)))) a = some ((2 * Real.cos x : ℝ) : ℂ) :=
    eval_congr hA (by push_cast; rw [two_cos])
  exact eval_congr (eval_Whalf hA') (by congr 1; ring)

/-- `W_sin(x) := W_cos(W₋(x, π/2))`. -/
def Wsin (t : Term n) : Term n := Wcos (Wsub t const_pi_half)

/-- **Lemma 23** (sine). `W_sin(x) = sin x` for every real `x`. -/
theorem eval_Wsin (ht : eval t a = some (x : ℂ)) :
    eval (Wsin t) a = some ((Real.sin x : ℝ) : ℂ) :=
  eval_congr (eval_Wcos (eval_Wsub ht eval_const_pi_half)) (by rw [Real.cos_sub_pi_div_two])

/-- `W_tan(x) := W/(W_sin(x), W_cos(x))`. -/
def Wtan (t : Term n) : Term n := Wdiv (Wsin t) (Wcos t)

/-- **Lemma 23** (tangent). `W_tan(x) = tan x` for every real `x` with `cos x ≠ 0`. -/
theorem eval_Wtan (ht : eval t a = some (x : ℂ)) (hx : Real.cos x ≠ 0) :
    eval (Wtan t) a = some ((Real.tan x : ℝ) : ℂ) :=
  eval_congr (eval_Wdiv (eval_Wsin ht) (eval_Wcos ht) hx) (by rw [Real.tan_eq_sin_div_cos])

/-- `W_arcsin(x) := W₋(D{A[L(A{W_√[W₋(1, W_sqr(x))], X(x)}), 2i], i}, 2)`. -/
def Warcsin (t : Term n) : Term n :=
  Wsub (D (A (L (A (Wsqrt (Wsub one (Wsqr t))) (X t))) const_2i) const_i) const_2

/-- **Lemma 24** (arcsine). `W_arcsin(x) = arcsin x` for every real `x ∈ [-1, 1]`,
including the endpoints `x = ±1`. -/
theorem eval_Warcsin (ht : eval t a = some (x : ℂ)) (h₁ : -1 ≤ x) (h₂ : x ≤ 1) :
    eval (Warcsin t) a = some ((Real.arcsin x : ℝ) : ℂ) := by
  set θ : ℝ := Real.arcsin x with hθ
  have hθ1 : -(Real.pi / 2) ≤ θ := Real.neg_pi_div_two_le_arcsin x
  have hθ2 : θ ≤ Real.pi / 2 := Real.arcsin_le_pi_div_two x
  have hsin : Real.sin θ = x := Real.sin_arcsin h₁ h₂
  have hcos : Real.cos θ = Real.sqrt (1 - x ^ 2) := Real.cos_arcsin x
  have hcos0 : 0 ≤ Real.cos θ := Real.cos_arcsin_nonneg x
  have h0 : 0 ≤ 1 - x ^ 2 := by nlinarith
  -- `W_√(1 - x²) = cos θ`
  have hr : eval (Wsqrt (Wsub one (Wsqr t))) a = some ((Real.cos θ : ℝ) : ℂ) :=
    eval_congr (eval_Wsqrt (eval_Wsub eval_one_real (eval_Wsqr ht)) h0) (by rw [hcos])
  -- `X(x) = i x = i sin θ`
  have hX : eval (X t) a = some ((Real.sin θ : ℝ) * I) := eval_congr (eval_X ht) (by rw [hsin])
  -- `A(cos θ, i sin θ) = e^{iθ}`
  have hre2 : (((Real.cos θ : ℝ) : ℂ) + ((Real.sin θ : ℝ) : ℂ) * I).re = Real.cos θ := by
    rw [add_re, ofReal_re, mul_I_re, ofReal_im, neg_zero, add_zero]
  have him1 : (((Real.sin θ : ℝ) : ℂ) * I).im = Real.sin θ := by rw [mul_I_im, ofReal_re]
  have hm3 : (-3 : ℂ).re = -3 := by norm_num
  have hA1 := eval_A hr hX (by rw [ofReal_im, abs_zero]; exact Real.pi_pos)
    (by rw [him1]; linarith [Real.abs_sin_le_one θ, Real.pi_gt_three])
    (ne_of_re_lt (by rw [ofReal_re, hm3]; linarith))
    (ne_of_re_lt (by rw [hre2, hm3]; linarith))
  have hA1' : eval (A (Wsqrt (Wsub one (Wsqr t))) (X t)) a = some (exp (θ * I)) :=
    eval_congr hA1 (by rw [exp_mul_I]; push_cast; ring)
  -- `L(e^{iθ}) = iθ`, because `e^{iθ}` is not a negative real and `θ ∈ (-π, π]`
  have hLv : Lval (exp ((θ : ℂ) * I)) = θ * I := by
    have hre : (exp ((θ : ℂ) * I)).re = Real.cos θ := exp_ofReal_mul_I_re θ
    unfold Lval
    rw [ite_eq_right (by rw [hre]; intro h; linarith [h.1]),
      log_exp (by simp; linarith [Real.pi_pos]) (by simp; linarith [Real.pi_pos])]
  have hL := eval_congr (eval_L hA1' (exp_ne_zero _)) hLv
  -- `A(iθ, 2i) = i(θ + 2)`
  have hA2 := eval_A hL eval_const_2i
    (by rw [show ((θ : ℂ) * I).im = θ by simp, abs_lt]; constructor <;> linarith [Real.pi_pos])
    (by simp; linarith [Real.pi_gt_three])
    (ne_of_re_lt (by norm_num)) (ne_of_re_lt (by norm_num))
  -- `D(i(θ+2), i) = θ + 2`
  have hpos2 : 0 < θ + 2 := by linarith [Real.pi_lt_four]
  have him2 : ((θ : ℂ) * I + 2 * I).im = θ + 2 := by simp
  have hD := eval_D hA2 eval_const_i (ne_zero_of_im_ne_zero (by rw [him2]; exact hpos2.ne'))
    (ne_one_of_im_ne_zero (by rw [him2]; exact hpos2.ne')) I_mem_slitPlane
  have hD' : eval (D (A (L (A (Wsqrt (Wsub one (Wsqr t))) (X t))) const_2i) const_i) a
      = some ((θ + 2 : ℝ) : ℂ) :=
    eval_congr hD (by push_cast; field_simp)
  -- `W₋(θ + 2, 2) = θ`
  exact eval_congr (eval_Wsub hD' eval_const_2_real) (by congr 1; ring)

/-- `W_arccos(x) := W₋(π/2, W_arcsin(x))`. -/
def Warccos (t : Term n) : Term n := Wsub const_pi_half (Warcsin t)

/-- **Lemma 24** (arccosine). `W_arccos(x) = arccos x` for every real `x ∈ [-1, 1]`. -/
theorem eval_Warccos (ht : eval t a = some (x : ℂ)) (h₁ : -1 ≤ x) (h₂ : x ≤ 1) :
    eval (Warccos t) a = some ((Real.arccos x : ℝ) : ℂ) :=
  eval_congr (eval_Wsub eval_const_pi_half (eval_Warcsin ht h₁ h₂))
    (by rw [Real.arccos_eq_pi_div_two_sub_arcsin])

/-- `W_arctan(x) := W_arcsin{W/[x, W_√(W₊(W_sqr(x), 1))]}`. -/
def Warctan (t : Term n) : Term n := Warcsin (Wdiv t (Wsqrt (Wadd (Wsqr t) one)))

/-- **Lemma 24** (arctangent). `W_arctan(x) = arctan x` for every real `x`. -/
theorem eval_Warctan (ht : eval t a = some (x : ℂ)) :
    eval (Warctan t) a = some ((Real.arctan x : ℝ) : ℂ) := by
  have hr := eval_Wsqrt (eval_Wadd (eval_Wsqr ht) eval_one_real) (by positivity)
  have hpos : 0 < Real.sqrt (x ^ 2 + 1) := Real.sqrt_pos.mpr (by positivity)
  -- `|x| ≤ √(x² + 1)`, so the quotient lies in `[-1, 1]`
  have habs : |x| ≤ Real.sqrt (x ^ 2 + 1) := Real.abs_le_sqrt (by linarith)
  obtain ⟨hl, hu⟩ := abs_le.mp habs
  have hq := eval_Wdiv ht hr hpos.ne'
  have h := eval_Warcsin hq ((le_div_iff₀ hpos).mpr (by linarith)) ((div_le_one hpos).mpr hu)
  exact eval_congr h (by rw [Real.arctan_eq_arcsin, add_comm (x ^ 2) 1])

end EML
