module

public import EML.UnitStep

/-!
# Proof of Theorem 1, Step 6 (Lemma 21): exponential–logarithmic primitives

Items (a)–(i) of Lemma 21: `σ`, `pow`, `log`, `sinh`, `cosh`, `tanh`,
`arsinh`, `arcosh` (including `arcosh 1 = 0`), `artanh`.
The real functions on the right-hand sides are those of Mathlib.
-/

@[expose] public section

namespace EML

open Complex Term

variable {n : ℕ} {a : Fin n → ℂ} {s t : Term n} {x y : ℝ}

/-- `W_σ(x) := W_inv(W₊(1, E(N(x))))`. -/
def Wsigma (t : Term n) : Term n := Winv (Wadd one (E (N t)))

/-- **Lemma 21 (a).** `W_σ(x) = 1/(1 + e^{-x})` for every real `x`. -/
theorem eval_Wsigma (ht : eval t a = some (x : ℂ)) :
    eval (Wsigma t) a = some ((Real.sigmoid x : ℝ) : ℂ) :=
  eval_congr (eval_Winv (eval_Wadd eval_one_real (eval_E_real (eval_N_real ht)))
    (by positivity)) (by rw [Real.sigmoid_def])

/-- `W_pow(x, y) := E(W×(y, L(x)))`. -/
def Wpow (s t : Term n) : Term n := E (Wmul t (L s))

/-- **Lemma 21 (b).** `W_pow(x, y) = x^y` for real `x > 0` and real `y`. -/
theorem eval_Wpow (hs : eval s a = some (x : ℂ)) (ht : eval t a = some (y : ℂ)) (hx : 0 < x) :
    eval (Wpow s t) a = some ((x ^ y : ℝ) : ℂ) :=
  eval_congr (eval_E_real (eval_Wmul ht (eval_L_pos hs hx)))
    (by rw [Real.rpow_def_of_pos hx, mul_comm])

/-- `W_log(x, y) := W/(L(y), L(x))`. -/
def Wlog (s t : Term n) : Term n := Wdiv (L t) (L s)

/-- **Lemma 21 (c).** `W_log(x, y) = log_x y` for real `x > 0`, `x ≠ 1`, and real `y > 0`. -/
theorem eval_Wlog (hs : eval s a = some (x : ℂ)) (ht : eval t a = some (y : ℂ))
    (hx : 0 < x) (hx1 : x ≠ 1) (hy : 0 < y) :
    eval (Wlog s t) a = some ((Real.logb x y : ℝ) : ℂ) :=
  eval_congr (eval_Wdiv (eval_L_pos ht hy) (eval_L_pos hs hx)
    (Real.log_ne_zero_of_pos_of_ne_one hx hx1)) (by rw [Real.log_div_log])

/-- `W_sinh(x) := W_half(W₋(E(x), E(N(x))))`. -/
def Wsinh (t : Term n) : Term n := Whalf (Wsub (E t) (E (N t)))

/-- **Lemma 21 (d).** `W_sinh(x) = sinh x` for every real `x`. -/
theorem eval_Wsinh (ht : eval t a = some (x : ℂ)) :
    eval (Wsinh t) a = some ((Real.sinh x : ℝ) : ℂ) :=
  eval_congr (eval_Whalf (eval_Wsub (eval_E_real ht) (eval_E_real (eval_N_real ht))))
    (by rw [Real.sinh_eq])

/-- `W_cosh(x) := W_avg(E(x), E(N(x)))`. -/
def Wcosh (t : Term n) : Term n := Wavg (E t) (E (N t))

/-- **Lemma 21 (e).** `W_cosh(x) = cosh x` for every real `x`. -/
theorem eval_Wcosh (ht : eval t a = some (x : ℂ)) :
    eval (Wcosh t) a = some ((Real.cosh x : ℝ) : ℂ) :=
  eval_congr (eval_Wavg (eval_E_real ht) (eval_E_real (eval_N_real ht)))
    (by rw [Real.cosh_eq])

/-- `W_tanh(x) := W/(W_sinh(x), W_cosh(x))`. -/
def Wtanh (t : Term n) : Term n := Wdiv (Wsinh t) (Wcosh t)

/-- **Lemma 21 (f).** `W_tanh(x) = tanh x` for every real `x`. -/
theorem eval_Wtanh (ht : eval t a = some (x : ℂ)) :
    eval (Wtanh t) a = some ((Real.tanh x : ℝ) : ℂ) :=
  eval_congr (eval_Wdiv (eval_Wsinh ht) (eval_Wcosh ht) (Real.cosh_pos x).ne')
    (by rw [Real.tanh_eq_sinh_div_cosh])

/-- `W_arsinh(x) := L(W₊(x, W_√(W₊(W_sqr(x), 1))))`. -/
def Warsinh (t : Term n) : Term n := L (Wadd t (Wsqrt (Wadd (Wsqr t) one)))

/-- **Lemma 21 (g).** `W_arsinh(x) = arsinh x` for every real `x`. -/
theorem eval_Warsinh (ht : eval t a = some (x : ℂ)) :
    eval (Warsinh t) a = some ((Real.arsinh x : ℝ) : ℂ) := by
  have h1 := eval_Wadd ht (eval_Wsqrt (eval_Wadd (eval_Wsqr ht) eval_one_real) (by positivity))
  have hpos : 0 < x + Real.sqrt (x ^ 2 + 1) := by
    have h := Real.exp_pos (Real.arsinh x)
    rw [Real.exp_arsinh, add_comm 1 (x ^ 2)] at h
    exact h
  exact eval_congr (eval_L_pos h1 hpos) (by rw [Real.arsinh, add_comm (x ^ 2) 1])

/-- `W_arcosh(x) := L(W₊(x, W_√(W₋(W_sqr(x), 1))))`. -/
def Warcosh (t : Term n) : Term n := L (Wadd t (Wsqrt (Wsub (Wsqr t) one)))

/-- **Lemma 21 (h).** `W_arcosh(x) = arcosh x` for real `x ≥ 1`, including `x = 1`. -/
theorem eval_Warcosh (ht : eval t a = some (x : ℂ)) (hx : 1 ≤ x) :
    eval (Warcosh t) a = some ((Real.arcosh x : ℝ) : ℂ) := by
  have h0 : 0 ≤ x ^ 2 - 1 := by nlinarith
  have h1 := eval_Wadd ht (eval_Wsqrt (eval_Wsub (eval_Wsqr ht) eval_one_real) h0)
  have hpos : 0 < x + Real.sqrt (x ^ 2 - 1) := by
    have := Real.sqrt_nonneg (x ^ 2 - 1)
    linarith
  exact eval_congr (eval_L_pos h1 hpos) (by rw [Real.arcosh])

/-- `W_artanh(x) := W_half(W₋(L(W₊(1, x)), L(W₋(1, x))))`. -/
def Wartanh (t : Term n) : Term n := Whalf (Wsub (L (Wadd one t)) (L (Wsub one t)))

/-- **Lemma 21 (i).** `W_artanh(x) = artanh x` for real `-1 < x < 1`. -/
theorem eval_Wartanh (ht : eval t a = some (x : ℂ)) (h₁ : -1 < x) (h₂ : x < 1) :
    eval (Wartanh t) a = some ((Real.artanh x : ℝ) : ℂ) := by
  have hp : 0 < 1 + x := by linarith
  have hm : 0 < 1 - x := by linarith
  have h := eval_Whalf (eval_Wsub (eval_L_pos (eval_Wadd eval_one_real ht) hp)
    (eval_L_pos (eval_Wsub eval_one_real ht) hm))
  exact eval_congr h (by
    rw [Real.artanh_eq_half_log ⟨h₁.le, h₂.le⟩, Real.log_div hp.ne' hm.ne']
    congr 1; ring)

end EML
