module

public import EML.Core

/-!
# Proof of Theorem 1, Step 4 (Lemmas 9–16): arithmetic for all real arguments

Terms: `U` (shift by `i`), and the witnesses `Wsub` (W₋), `Wadd` (W₊), `Wsqr`,
`Whalf`, `Wmul` (W×), `Winv`, `Wdiv` (W/), `Wavg`.
Real numbers appear as complex numbers `(x : ℂ)` with `x : ℝ`.
-/

@[expose] public section

namespace EML

open Complex Term
open scoped Real

variable {n : ℕ} {a : Fin n → ℂ} {s t : Term n} {u v z : ℂ} {x y : ℝ}

/-- The leaf `1`, read as a real number. -/
theorem eval_one_real : eval (one : Term n) a = some ((1 : ℝ) : ℂ) := by simp

/-- The constant `2`, read as a real number. -/
theorem eval_const_2_real : eval (const_2 : Term n) a = some ((2 : ℝ) : ℂ) :=
  eval_congr eval_const_2 (by norm_num)

/-- `U(x) := eml(L(i), E(N(x)))`. -/
def U (t : Term n) : Term n := eml (L const_i) (E (N t))

/-- **Lemma 9.** For every real `x`, `U(x)` is defined and `U(x) = x + i`. -/
theorem eval_U (ht : eval t a = some (x : ℂ)) : eval (U t) a = some (x + I) := by
  have hE := eval_E (eval_N_real ht)
  have h := eval_eml (eval_L (eval_const_i (a := a)) I_ne_zero) hE (exp_ne_zero _)
  exact eval_congr h (by
    rw [exp_Lval I_ne_zero, log_exp (neg_pi_lt_ofReal_im _) (ofReal_im_le_pi _)]
    push_cast; ring)

/-- `W₋(a, b) := S(S(U(a), b), i)`. -/
def Wsub (s t : Term n) : Term n := S (S (U s) t) const_i
/-- `W₊(a, b) := S(S(U(a), N(b)), i)`. -/
def Wadd (s t : Term n) : Term n := S (S (U s) (N t)) const_i

/-- **Lemma 10** (subtraction). For all real `x, y`: `W₋(x, y) = x - y`. -/
theorem eval_Wsub (hs : eval s a = some (x : ℂ)) (ht : eval t a = some (y : ℂ)) :
    eval (Wsub s t) a = some ((x - y : ℝ) : ℂ) := by
  -- `S(x+i, y) = x + i - y`
  have h1 := eval_S_real (eval_U hs) ht (ofReal_add_I_ne_zero x)
  -- `S(x-y+i, i) = x - y`
  have h2 := eval_S h1 eval_const_i (ne_zero_of_im_ne_zero (by simp)) neg_pi_lt_I_im I_im_le_pi
  exact eval_congr h2 (by push_cast; ring)

/-- **Lemma 10** (addition). For all real `x, y`: `W₊(x, y) = x + y`. -/
theorem eval_Wadd (hs : eval s a = some (x : ℂ)) (ht : eval t a = some (y : ℂ)) :
    eval (Wadd s t) a = some ((x + y : ℝ) : ℂ) :=
  eval_congr (eval_Wsub hs (eval_N_real ht)) (by push_cast; ring)

/-- `W_sqr(x) := W₋(M(U(x), S(U(x), 2i)), 1)`. -/
def Wsqr (t : Term n) : Term n := Wsub (M (U t) (S (U t) const_2i)) one

/-- **Lemma 11.** For every real `x`: `W_sqr(x) = x²`. -/
theorem eval_Wsqr (ht : eval t a = some (x : ℂ)) :
    eval (Wsqr t) a = some ((x ^ 2 : ℝ) : ℂ) := by
  have hU := eval_U ht
  -- `S(x+i, 2i) = x - i`, because `Im(2i) = 2 ≤ π`
  have hS := eval_S hU eval_const_2i (ofReal_add_I_ne_zero x)
    (by simp; linarith [Real.pi_pos]) (by simp; linarith [Real.pi_gt_three])
  -- `M(x+i, x-i) = x² + 1`
  have hM := eval_M hU hS (ofReal_add_I_ne_zero x) (ofReal_add_I_ne_one x)
    (mem_slitPlane_of_im_ne_zero (by norm_num))
  have hM' : eval (M (U t) (S (U t) const_2i)) a = some ((x ^ 2 + 1 : ℝ) : ℂ) :=
    eval_congr hM (by push_cast; linear_combination (-1 : ℂ) * I_sq)
  exact eval_congr (eval_Wsub hM' eval_one_real) (by push_cast; ring)

/-- `W_half(x) := S(D(U(x), 2), D(i, 2))`. -/
def Whalf (t : Term n) : Term n := S (D (U t) const_2) (D const_i const_2)

/-- **Lemma 12.** For every real `x`: `W_half(x) = x/2`. -/
theorem eval_Whalf (ht : eval t a = some (x : ℂ)) :
    eval (Whalf t) a = some ((x / 2 : ℝ) : ℂ) := by
  have h1 := eval_D (eval_U ht) eval_const_2 (ofReal_add_I_ne_zero x) (ofReal_add_I_ne_one x)
    (ofNat_mem_slitPlane 2)
  have h2 := eval_D (eval_const_i (a := a)) eval_const_2 I_ne_zero (ne_one_of_im_ne_zero (by simp))
    (ofNat_mem_slitPlane 2)
  have h3 := eval_S h1 h2 (div_ne_zero (ofReal_add_I_ne_zero x) two_ne_zero)
    (by simp; linarith [Real.pi_pos]) (by simp; linarith [Real.pi_gt_three])
  exact eval_congr h3 (by push_cast; ring)

/-- `W×(x, y) := W_half{W_half[W₋(W_sqr[W₊(x,y)], W_sqr[W₋(x,y)])]}`. -/
def Wmul (s t : Term n) : Term n :=
  Whalf (Whalf (Wsub (Wsqr (Wadd s t)) (Wsqr (Wsub s t))))

/-- **Lemma 13.** For all real `x, y`: `W×(x, y) = x y`. -/
theorem eval_Wmul (hs : eval s a = some (x : ℂ)) (ht : eval t a = some (y : ℂ)) :
    eval (Wmul s t) a = some ((x * y : ℝ) : ℂ) := by
  have h := eval_Whalf (eval_Whalf
    (eval_Wsub (eval_Wsqr (eval_Wadd hs ht)) (eval_Wsqr (eval_Wsub hs ht))))
  exact eval_congr h (by congr 1; ring)

/-- `W_inv(x) := W×(x, E(N(L(W_sqr(x)))))`. -/
def Winv (t : Term n) : Term n := Wmul t (E (N (L (Wsqr t))))

/-- **Lemma 14.** For every real `x ≠ 0`: `W_inv(x) = 1/x`. -/
theorem eval_Winv (ht : eval t a = some (x : ℂ)) (hx : x ≠ 0) :
    eval (Winv t) a = some ((x⁻¹ : ℝ) : ℂ) := by
  have hpos : 0 < x ^ 2 := by positivity
  have hE := eval_E_real (eval_N_real (eval_L_pos (eval_Wsqr ht) hpos))
  exact eval_congr (eval_Wmul ht hE) (by
    congr 1; rw [Real.exp_neg, Real.exp_log hpos]; field_simp)

/-- `W/(x, y) := W×(x, W_inv(y))`. -/
def Wdiv (s t : Term n) : Term n := Wmul s (Winv t)

/-- **Lemma 15.** For every real `x` and every real `y ≠ 0`: `W/(x, y) = x/y`. -/
theorem eval_Wdiv (hs : eval s a = some (x : ℂ)) (ht : eval t a = some (y : ℂ)) (hy : y ≠ 0) :
    eval (Wdiv s t) a = some ((x / y : ℝ) : ℂ) :=
  eval_congr (eval_Wmul hs (eval_Winv ht hy)) (by rw [div_eq_mul_inv])

/-- `W_avg(x, y) := W_half(W₊(x, y))`. -/
def Wavg (s t : Term n) : Term n := Whalf (Wadd s t)

/-- **Lemma 16.** For all real `x, y`: `W_avg(x, y) = (x + y)/2`. -/
theorem eval_Wavg (hs : eval s a = some (x : ℂ)) (ht : eval t a = some (y : ℂ)) :
    eval (Wavg s t) a = some (((x + y) / 2 : ℝ) : ℂ) :=
  eval_Whalf (eval_Wadd hs ht)

end EML
