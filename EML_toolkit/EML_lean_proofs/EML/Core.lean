module

public import EML.Basic

/-!
# Proof of Theorem 1, Steps 1–3 (Lemma 1, Corollary 2, Lemmas 3–8)

Each lemma of the paper is a theorem `eval_…` here. It reads:
"if the argument terms have these values, then the new term is defined
and has that value".  The names of the terms are those of the paper:
`E, L, S, N, M, D`; the constants are `const_e` (e), `const_0`, `const_neg` (-1),
`const_2`, `const_3`, `const_half`, `const_minus_ipi` (-iπ), `const_i` (i), `const_pi`,
`const_pi_half`, `const_2i`, `const_ipi`, `const_2ipi`.
-/

@[expose] public section

namespace EML

open Complex Term

variable {n : ℕ} {a : Fin n → ℂ} {s t : Term n} {u v z : ℂ} {x y : ℝ}

/-! ## Step 1: exponential, logarithm, subtraction -/

/-- `E(z) := eml(z, 1)`. -/
def E (t : Term n) : Term n := eml t one

/-- **Lemma 1.** For every `z ∈ ℂ`, `E(z)` is defined and `E(z) = exp z`. -/
theorem eval_E (ht : eval t a = some z) : eval (E t) a = some (exp z) := by
  have h := eval_eml ht (eval_one (a := a)) one_ne_zero
  simpa [E] using h

/-- Lemma 1 for a real argument. -/
theorem eval_E_real (ht : eval t a = some (x : ℂ)) :
    eval (E t) a = some ((Real.exp x : ℝ) : ℂ) :=
  eval_congr (eval_E ht) (ofReal_exp x).symm

/-- The closed term `e := eml(1, 1)`. -/
def const_e : Term n := eml one one

/-- **Corollary 2.** `eml(1,1)` is defined and has the value `e`. -/
theorem eval_const_e : eval (const_e : Term n) a = some ((Real.exp 1 : ℝ) : ℂ) := by
  have h : eval (E (one : Term n)) a = some (exp 1) := eval_E eval_one
  exact eval_congr h (by simp)

/-- `L(z) := eml(1, E(eml(1, z)))`, the auxiliary logarithm. -/
def L (t : Term n) : Term n := eml one (E (eml one t))

/-- The value of `L` in Lemma 3: `Log z - 2πi` on the negative real axis, `Log z` elsewhere. -/
noncomputable def Lval (z : ℂ) : ℂ :=
  if z.re < 0 ∧ z.im = 0 then log z - 2 * Real.pi * I else log z

/-- The computation in the proof of Lemma 3. -/
theorem Lval_eq (z : ℂ) : exp 1 - log (exp (exp 1 - log z)) = Lval z := by
  have him : (exp 1 - log z).im = -arg z := by simp [log_im, exp_im]
  unfold Lval
  split_ifs with h
  · -- `z` is a negative real: `Arg z = π`, so `Im w = -π`, outside the strip
    have harg : arg z = Real.pi := arg_eq_pi_iff.mpr h
    rw [log_exp_shift _ (-1) (by rw [him, harg]; push_cast; linarith [Real.pi_pos])
      (by rw [him, harg]; push_cast; linarith [Real.pi_pos])]
    push_cast; ring
  · -- otherwise `Arg z < π`, so `-π < Im w < π`, inside the strip
    have harg : arg z < Real.pi := arg_lt_pi_iff.mpr (by
      by_contra hc; push Not at hc; exact h ⟨hc.1, hc.2⟩)
    rw [log_exp (by rw [him]; linarith) (by rw [him]; linarith [neg_pi_lt_arg z])]
    ring

/-- Lemma 3: `exp (L(z)) = z` for `z ≠ 0`, in both cases. -/
theorem exp_Lval (hz : z ≠ 0) : exp (Lval z) = z := by
  unfold Lval
  split_ifs
  · rw [exp_sub, exp_two_pi_mul_I, div_one, exp_log hz]
  · exact exp_log hz

/-- Lemma 3: `L(z) = Log z` if `z` is not a negative real (and not `0`). -/
theorem Lval_of_mem_slitPlane (hz : z ∈ slitPlane) : Lval z = log z := by
  unfold Lval
  split_ifs with hc
  · rcases mem_slitPlane_iff.mp hz with h | h
    · linarith [hc.1]
    · exact absurd hc.2 h
  · rfl

/-- Lemma 3: `L(x) = ln x` for real `x > 0`. -/
theorem Lval_ofReal_pos (hx : 0 < x) : Lval (x : ℂ) = ((Real.log x : ℝ) : ℂ) := by
  rw [Lval_of_mem_slitPlane (ofReal_mem_slitPlane.mpr hx), ofReal_log hx.le]

/-- Lemma 3: `L(z) = Log z - 2πi` if `z` is a negative real. -/
theorem Lval_of_negReal (h₁ : z.re < 0) (h₂ : z.im = 0) : Lval z = log z - 2 * Real.pi * I := by
  simp [Lval, h₁, h₂]

/-- The imaginary part of `L(z)` lies in `(-π, π)` if `z` is not a negative real. -/
theorem Lval_im_of_mem_slitPlane (hz : z ∈ slitPlane) :
    -Real.pi < (Lval z).im ∧ (Lval z).im < Real.pi := by
  rw [Lval_of_mem_slitPlane hz, log_im]
  exact ⟨neg_pi_lt_arg z, lt_of_le_of_ne (arg_le_pi z) (mem_slitPlane_iff_arg.mp hz).1⟩

/-- The real part of `L(z)` is `ln |z|`. -/
theorem Lval_re (z : ℂ) : (Lval z).re = Real.log ‖z‖ := by
  unfold Lval
  split_ifs <;> simp [log_re]

/-- `L(u) ≠ 0` if `u ∉ {0, 1}`. -/
theorem Lval_ne_zero (hu0 : u ≠ 0) (hu1 : u ≠ 1) : Lval u ≠ 0 := by
  intro h
  have h' := exp_Lval hu0
  rw [h, exp_zero] at h'
  exact hu1 h'.symm

/-- **Lemma 3.** For every `z ≠ 0`, `L(z)` is defined and has the value `Lval z`. -/
theorem eval_L (ht : eval t a = some z) (hz : z ≠ 0) : eval (L t) a = some (Lval z) := by
  have h1 := eval_E (eval_eml (eval_one (a := a)) ht hz)
  have h2 := eval_eml (eval_one (a := a)) h1 (exp_ne_zero _)
  exact eval_congr h2 (Lval_eq z)

/-- Lemma 3, real case: `L(x) = ln x` for real `x > 0`. -/
theorem eval_L_pos (ht : eval t a = some (x : ℂ)) (hx : 0 < x) :
    eval (L t) a = some ((Real.log x : ℝ) : ℂ) :=
  eval_congr (eval_L ht (ofReal_ne_zero.mpr hx.ne')) (Lval_ofReal_pos hx)

/-- `S(u, v) := eml(L(u), E(v))`, the auxiliary subtraction. -/
def S (s t : Term n) : Term n := eml (L s) (E t)

/-- **Lemma 4**, first part: `S(u,v)` is defined for every `u ≠ 0` and every `v`. -/
theorem eval_S_defined (hs : eval s a = some u) (ht : eval t a = some v) (hu : u ≠ 0) :
    eval (S s t) a = some (u - log (exp v)) := by
  have h := eval_eml (eval_L hs hu) (eval_E ht) (exp_ne_zero _)
  exact eval_congr h (by rw [exp_Lval hu])

/-- **Lemma 4.** If `u ≠ 0` and `-π < Im v ≤ π`, then `S(u,v) = u - v`. -/
theorem eval_S (hs : eval s a = some u) (ht : eval t a = some v) (hu : u ≠ 0)
    (h₁ : -Real.pi < v.im) (h₂ : v.im ≤ Real.pi) : eval (S s t) a = some (u - v) :=
  eval_congr (eval_S_defined hs ht hu) (by rw [log_exp h₁ h₂])

/-- Lemma 4 for a real second argument. -/
theorem eval_S_real (hs : eval s a = some u) (ht : eval t a = some (y : ℂ)) (hu : u ≠ 0) :
    eval (S s t) a = some (u - y) :=
  eval_S hs ht hu (neg_pi_lt_ofReal_im y) (ofReal_im_le_pi y)

/-- The remark after Lemma 4: `S(u,v)` is undefined at `u = 0`. -/
theorem eval_S_zero (hs : eval s a = some 0) : eval (S s t) a = none :=
  eval_eml_undef_left (eval_eml_undef_right (eval_eml_undef_left (eval_eml_log_zero hs)))

/-! ## Step 2: negation -/

/-- `N(z) := S(1, S(e, S(S(e,1), z)))`. -/
def N (t : Term n) : Term n := S one (S const_e (S (S const_e one) t))

/-- **Lemma 5.** For `-π < Im z < π`, `N(z)` is defined and `N(z) = -z`. -/
theorem eval_N (ht : eval t a = some z) (h₁ : -Real.pi < z.im) (h₂ : z.im < Real.pi) :
    eval (N t) a = some (-z) := by
  have he : ((Real.exp 1 : ℝ) : ℂ) ≠ 0 := ofReal_ne_zero.mpr (Real.exp_pos 1).ne'
  have he1 : ((Real.exp 1 : ℝ) : ℂ) - 1 ≠ 0 :=
    sub_ne_zero.mpr (ofReal_ne_one.mpr (by
      have := Real.add_one_lt_exp (x := 1) one_ne_zero; linarith))
  -- `S(e,1) = e - 1`
  have h0 : eval (S const_e one) a = some (((Real.exp 1 : ℝ) : ℂ) - 1) :=
    eval_S eval_const_e eval_one he (by simp [Real.pi_pos]) (by simp [Real.pi_pos.le])
  -- `S(e-1, z) = e - 1 - z`
  have h1 := eval_S h0 ht he1 h₁ h₂.le
  -- `S(e, e-1-z) = 1 + z`
  have h2 := eval_S eval_const_e h1 he (by simp [exp_im]; linarith) (by simp [exp_im]; linarith)
  -- `S(1, 1+z) = -z`
  have h3 := eval_S eval_one h2 one_ne_zero (by simp; linarith) (by simp; linarith)
  exact eval_congr h3 (by ring)

/-- Lemma 5 for a real argument, including `x = 0`. -/
theorem eval_N_real (ht : eval t a = some (x : ℂ)) : eval (N t) a = some ((-x : ℝ) : ℂ) :=
  eval_congr (eval_N ht (neg_pi_lt_ofReal_im x) (ofReal_im_lt_pi x)) (ofReal_neg x).symm

/-! ## Step 3: constants, products and quotients of non-zero numbers -/

/-- `0 := L(1)`. -/
def const_0 : Term n := L one
/-- `-1 := N(1)`. -/
def const_neg : Term n := N one
/-- `2 := S(1, -1)`. -/
def const_2 : Term n := S one const_neg
/-- `3 := S(2, -1)`. -/
def const_3 : Term n := S const_2 const_neg
/-- `1/2 := E(N(L(2)))`. -/
def const_half : Term n := E (N (L const_2))

/-- **Lemma 6**: the value of `L(1)` is `0`. -/
theorem eval_const_0 : eval (const_0 : Term n) a = some 0 := by
  have h := eval_L (eval_one (a := a)) one_ne_zero
  exact eval_congr h (by rw [Lval_of_mem_slitPlane one_mem_slitPlane, log_one])

/-- **Lemma 6**: the value of `N(1)` is `-1`. -/
theorem eval_const_neg : eval (const_neg : Term n) a = some (-1) :=
  eval_N eval_one (by simp [Real.pi_pos]) (by simp [Real.pi_pos])

/-- **Lemma 6**: the value of `S(1,-1)` is `2`. -/
theorem eval_const_2 : eval (const_2 : Term n) a = some 2 :=
  eval_congr (eval_S eval_one eval_const_neg one_ne_zero (by simp [Real.pi_pos])
    (by simp [Real.pi_pos.le])) (by norm_num)

/-- **Lemma 6**: the value of `S(2,-1)` is `3`. -/
theorem eval_const_3 : eval (const_3 : Term n) a = some 3 :=
  eval_congr (eval_S eval_const_2 eval_const_neg two_ne_zero (by simp [Real.pi_pos])
    (by simp [Real.pi_pos.le])) (by norm_num)

/-- **Lemma 6**: the value of `E(N(L(2)))` is `1/2`. -/
theorem eval_const_half : eval (const_half : Term n) a = some (1 / 2) := by
  have h2 : eval (const_2 : Term n) a = some ((2 : ℝ) : ℂ) := eval_congr eval_const_2 (by norm_num)
  have hE := eval_E_real (eval_N_real (eval_L_pos h2 (by norm_num)))
  exact eval_congr hE (by rw [Real.exp_neg, Real.exp_log (by norm_num)]; norm_num)

/-- `M(u, v) := E(S(L(u), N(L(v))))`. -/
def M (s t : Term n) : Term n := E (S (L s) (N (L t)))
/-- `D(u, v) := E(S(L(u), L(v)))`. -/
def D (s t : Term n) : Term n := E (S (L s) (L t))

/-- **Lemma 7** (products). For `u ∉ {0, 1}` and `v ∉ (-∞, 0]`, `M(u,v)` is defined
and `M(u,v) = u v`. (`slitPlane` is Mathlib's name for `ℂ ∖ (-∞, 0]`.) -/
theorem eval_M (hs : eval s a = some u) (ht : eval t a = some v) (hu0 : u ≠ 0) (hu1 : u ≠ 1)
    (hv : v ∈ slitPlane) : eval (M s t) a = some (u * v) := by
  have hv0 : v ≠ 0 := slitPlane_ne_zero hv
  obtain ⟨hi1, hi2⟩ := Lval_im_of_mem_slitPlane hv
  have hN := eval_N (eval_L ht hv0) hi1 hi2
  have hS := eval_S (eval_L hs hu0) hN (Lval_ne_zero hu0 hu1)
    (by simp; linarith) (by simp; linarith)
  exact eval_congr (eval_E hS) (by rw [sub_neg_eq_add, exp_add, exp_Lval hu0, exp_Lval hv0])

/-- **Lemma 7** (quotients). For `u ∉ {0, 1}` and `v ∉ (-∞, 0]`, `D(u,v)` is defined
and `D(u,v) = u / v`. -/
theorem eval_D (hs : eval s a = some u) (ht : eval t a = some v) (hu0 : u ≠ 0) (hu1 : u ≠ 1)
    (hv : v ∈ slitPlane) : eval (D s t) a = some (u / v) := by
  have hv0 : v ≠ 0 := slitPlane_ne_zero hv
  obtain ⟨hi1, hi2⟩ := Lval_im_of_mem_slitPlane hv
  have hS := eval_S (eval_L hs hu0) (eval_L ht hv0) (Lval_ne_zero hu0 hu1) hi1 hi2.le
  exact eval_congr (eval_E hS) (by rw [exp_sub, exp_Lval hu0, exp_Lval hv0])

/-- `-iπ := eml(1, N(E(e)))`. -/
def const_minus_ipi : Term n := eml one (N (E const_e))
/-- `i := E(N(D(-iπ, 2)))`. -/
def const_i : Term n := E (N (D const_minus_ipi const_2))
/-- `π := M(i, -iπ)`. -/
def const_pi : Term n := M const_i const_minus_ipi
/-- `π/2 := D(π, 2)`. -/
def const_pi_half : Term n := D const_pi const_2
/-- `2i := M(2, i)`. -/
def const_2i : Term n := M const_2 const_i
/-- `iπ := M(i, π)`. -/
def const_ipi : Term n := M const_i const_pi
/-- `2πi := M(2, iπ)`. -/
def const_2ipi : Term n := M const_2 const_ipi

/-- **Lemma 8**: the closed term `eml(1, N(E(e)))` has the value `-iπ`. -/
theorem eval_const_minus_ipi : eval (const_minus_ipi : Term n) a = some (-(Real.pi * I)) := by
  -- `E(e) = e^e` and `N(e^e) = -e^e`, a negative real number
  have hN := eval_N_real (eval_E_real (eval_const_e (a := a)))
  have hneg : -Real.exp (Real.exp 1) < 0 := neg_lt_zero.mpr (Real.exp_pos _)
  -- `Log(-e^e) = ln(e^e) + iπ = e + iπ`
  have hlog : log ((-Real.exp (Real.exp 1) : ℝ) : ℂ) = (Real.exp 1 : ℂ) + Real.pi * I := by
    apply Complex.ext
    · simp only [log_re, norm_real, Real.norm_eq_abs, abs_of_neg hneg, neg_neg, Real.log_exp,
        add_re, ofReal_re, mul_I_re, ofReal_im, neg_zero, add_zero]
    · simp only [log_im, arg_ofReal_of_neg hneg, add_im, ofReal_im, mul_I_im, ofReal_re,
        zero_add]
  -- `eml(1, -e^e) = e - (e + iπ) = -iπ`
  refine eval_congr (eval_eml eval_one hN (ofReal_ne_zero.mpr hneg.ne)) ?_
  rw [hlog]
  push_cast
  ring

/-- **Lemma 8**: the closed term `E(N(D(-iπ, 2)))` has the value `i`. -/
theorem eval_const_i : eval (const_i : Term n) a = some I := by
  have h0 : -((Real.pi : ℂ) * I) ≠ 0 := ne_zero_of_im_ne_zero (by simp [Real.pi_ne_zero])
  have h1 : -((Real.pi : ℂ) * I) ≠ 1 := ne_one_of_im_ne_zero (by simp [Real.pi_ne_zero])
  -- `D(-iπ, 2) = -iπ/2`
  have hD := eval_D (eval_const_minus_ipi (a := a)) eval_const_2 h0 h1 (ofNat_mem_slitPlane 2)
  have him : (-((Real.pi : ℂ) * I) / 2).im = -(Real.pi / 2) := by simp; ring
  -- `N(-iπ/2) = iπ/2`, because `Im = -π/2` lies in `(-π, π)`
  have hN := eval_N hD (by rw [him]; linarith [Real.pi_pos]) (by rw [him]; linarith [Real.pi_pos])
  -- `E(iπ/2) = i`
  refine eval_congr (eval_E hN) ?_
  have h2 : -(-((Real.pi : ℂ) * I) / 2) = (Real.pi : ℂ) / 2 * I := by ring
  rw [h2, exp_mul_I, cos_pi_div_two, sin_pi_div_two]
  simp

/-- **Lemma 8**: `M(i, -iπ)` has the value `π`. -/
theorem eval_const_pi : eval (const_pi : Term n) a = some (Real.pi : ℂ) := by
  have h := eval_M (eval_const_i (a := a)) eval_const_minus_ipi I_ne_zero
    (ne_one_of_im_ne_zero (by simp)) (mem_slitPlane_of_im_ne_zero (by simp [Real.pi_ne_zero]))
  exact eval_congr h (by linear_combination (-(Real.pi : ℂ)) * I_sq)

/-- **Lemma 8**: `D(π, 2)` has the value `π/2`. -/
theorem eval_const_pi_half : eval (const_pi_half : Term n) a = some ((Real.pi / 2 : ℝ) : ℂ) := by
  have h := eval_D (eval_const_pi (a := a)) eval_const_2 (ofReal_ne_zero.mpr Real.pi_ne_zero)
    (ofReal_ne_one.mpr (by linarith [Real.pi_gt_three] : (1 : ℝ) < Real.pi).ne')
    (ofNat_mem_slitPlane 2)
  exact eval_congr h (by push_cast; rfl)

/-- **Lemma 8**: `M(2, i)` has the value `2i`. -/
theorem eval_const_2i : eval (const_2i : Term n) a = some (2 * I) :=
  eval_M eval_const_2 eval_const_i two_ne_zero (by norm_num) I_mem_slitPlane

/-- **Lemma 8**: `M(i, π)` has the value `iπ`. -/
theorem eval_const_ipi : eval (const_ipi : Term n) a = some (I * Real.pi) :=
  eval_M eval_const_i eval_const_pi I_ne_zero (ne_one_of_im_ne_zero (by simp))
    (ofReal_mem_slitPlane.mpr Real.pi_pos)

/-- **Lemma 8**: `M(2, iπ)` has the value `2πi`. -/
theorem eval_const_2ipi : eval (const_2ipi : Term n) a = some (2 * (I * Real.pi)) :=
  eval_M eval_const_2 eval_const_ipi two_ne_zero (by norm_num)
    (mem_slitPlane_of_im_ne_zero (by simp [Real.pi_ne_zero]))

/-- The remark after Lemma 8: the closed term `L(-1)` has the same value `-iπ`.
It is not used in the witnesses. -/
theorem eval_L_const_neg : eval (L (const_neg : Term n)) a = some (-(Real.pi * I)) := by
  have h := eval_L (eval_const_neg (a := a)) (by norm_num)
  exact eval_congr h (by rw [Lval_of_negReal (by simp) (by simp), log_neg_one]; ring)

end EML
