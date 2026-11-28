module

public import EML.Arith
public import EML.Real.Strict

/-!
# Corollary 25 (All rationals)

* `rat_term r`, `corollary25_a` — part (a): every rational number `r` is the value of a closed
  term in which every node has a real value and every logarithm is applied to a positive real
  number. The terms are those of the paper: `n + 1 := S(n, -1)` (`pos_int`),
  `1/q := E(N(L(q)))` and `p/q := D(p, q)` (`pos_rat`), `0 := L(1)`, `-r := N(r)`.
* `val_S`, `val_N`, `val_D` — Lemmas 4, 5, 7 for real nodes, in the form of `EML.Real.Strict`.
* `Wfloor`, `Wceil`, `eval_Wfloor`, `eval_Wceil` — part (b): `W_⌊⌋(x) = ⌊x⌋` and
  `N(W_⌊⌋(N(x))) = ⌈x⌉` for every real `x`. `wfloor` is the term `w(x)` of the paper.
* `R`, `eval_R` — part (c): `R(x) = 1/(2⌊x⌋ - x + 1)` for every real `x ≠ -1`.

That `1, R(1), R(R(1)), …` are all positive rational numbers, each once (the theorem of Calkin,
Wilf and Newman that the paper cites), is proved in `EML.CalkinWilf`.
-/

@[expose] public section

namespace EML

open Complex Term

/-! ## Part (a): rationals from real nodes -/

section partA

variable {n : ℕ} {s t : Term n} {u v : ℝ} {V : Term n → ℝ → Prop}

/-- Lemma 4 with real nodes: `S(u, v) = u - v` for real `u > 0` and real `v`. -/
theorem val_S (hV : StrictVal V) (hs : V s u) (ht : V t v) (hu : 0 < u) : V (S s t) (u - v) :=
  val_congr (hV.eml (val_L hV hs hu) (val_E hV ht) (Real.exp_pos _))
    (by rw [Real.exp_log hu, Real.log_exp])

/-- Lemma 5 with real nodes: `N(v) = -v` for every real `v`. The subtractions have the positive
first arguments `e`, `e - 1`, `e`, `1`. -/
theorem val_N (hV : StrictVal V) (ht : V t v) : V (N t) (-v) := by
  have he : (0 : ℝ) < Real.exp 1 := Real.exp_pos 1
  have he1 : (0 : ℝ) < Real.exp 1 - 1 := by linarith [Real.add_one_lt_exp (one_ne_zero (α := ℝ))]
  have h1 := val_S hV (val_S hV (val_const_e hV) hV.one he) ht he1
  have h2 := val_S hV (val_const_e hV) h1 he
  exact val_congr (val_S hV hV.one h2 one_pos) (by ring)

/-- Lemma 7 with real nodes: `D(u, v) = u / v` for real `u > 1` and `v > 0`. The first argument
of `S` is `ln u > 0`. -/
theorem val_D (hV : StrictVal V) (hs : V s u) (ht : V t v) (hu : 1 < u) (hv : 0 < v) :
    V (D s t) (u / v) :=
  val_congr (val_E hV (val_S hV (val_L hV hs (by linarith)) (val_L hV ht hv) (Real.log_pos hu)))
    (by rw [Real.exp_sub, Real.exp_log (by linarith), Real.exp_log hv])

/-- The positive integers: `pos_int k` is the term for `k + 1`, with `n + 1 := S(n, -1)`. -/
def pos_int : ℕ → Term n
  | 0 => one
  | k + 1 => S (pos_int k) const_neg

/-- `pos_int k` has the value `k + 1`. -/
theorem val_pos_int (hV : StrictVal V) (k : ℕ) : V (pos_int k) (k + 1) := by
  induction k with
  | zero => exact val_congr hV.one (by norm_num)
  | succ k ih =>
    exact val_congr (val_S hV ih (val_N hV hV.one) (by positivity)) (by push_cast; ring)

/-- The term for the positive rational number `(p + 1)/(q + 1)`: `1/q := E(N(L(q)))` for the
numerator `1`, and `p/q := D(p, q)` for a numerator `p ≥ 2`. -/
def pos_rat : ℕ → ℕ → Term n
  | 0, q => E (N (L (pos_int q)))
  | p + 1, q => D (pos_int (p + 1)) (pos_int q)

/-- `pos_rat p q` has the value `(p + 1)/(q + 1)`. -/
theorem val_pos_rat (hV : StrictVal V) (p q : ℕ) :
    V (pos_rat p q) (((p : ℝ) + 1) / ((q : ℝ) + 1)) := by
  have hq : (0 : ℝ) < (q : ℝ) + 1 := by positivity
  cases p with
  | zero =>
    exact val_congr (val_E hV (val_N hV (val_L hV (val_pos_int hV q) hq)))
      (by rw [Real.exp_neg, Real.exp_log hq]; simp)
  | succ p =>
    have hp : (1 : ℝ) < ((p + 1 : ℕ) : ℝ) + 1 := by
      push_cast; linarith [Nat.cast_nonneg (α := ℝ) p]
    exact val_D hV (val_pos_int hV (p + 1)) (val_pos_int hV q) hp hq

/-- The term for a rational number: `0 := L(1)`, `pos_rat` for a positive one, and `-r := N(r)`
for a negative one. -/
def rat_term (r : ℚ) : Term n :=
  if r = 0 then const_0
  else if 0 < r then pos_rat (r.num.natAbs - 1) (r.den - 1)
  else N (pos_rat (r.num.natAbs - 1) (r.den - 1))

/-- **Corollary 25 (a).** `rat_term r` has the value `r` in every evaluation that respects strict
real nodes. -/
theorem val_rat_term (hV : StrictVal V) (r : ℚ) : V (rat_term r) (r : ℝ) := by
  have hden : ((r.den - 1 : ℕ) : ℝ) + 1 = r.den := by
    have : 1 ≤ r.den := r.den_pos
    push_cast [Nat.cast_sub this]; ring
  have hr : (r : ℝ) = (r.num : ℝ) / (r.den : ℝ) := by
    exact_mod_cast (Rat.cast_def r : ((r : ℝ)) = _)
  have hval : r ≠ 0 → V (pos_rat (r.num.natAbs - 1) (r.den - 1)) (|(r.num : ℝ)| / (r.den : ℝ)) := by
    intro h0
    have h1 : 1 ≤ r.num.natAbs := Int.natAbs_pos.mpr (Rat.num_ne_zero.mpr h0)
    have habs : ((r.num.natAbs - 1 : ℕ) : ℝ) + 1 = |(r.num : ℝ)| := by
      rw [Nat.cast_sub h1, Nat.cast_natAbs, Int.cast_abs]; simp
    have h := val_pos_rat hV (r.num.natAbs - 1) (r.den - 1)
    rwa [hden, habs] at h
  rw [rat_term]
  split_ifs with h0 hpos
  · subst h0
    exact val_congr (val_const_0 hV) (by simp)
  · have hn : (0 : ℝ) < (r.num : ℝ) := by exact_mod_cast Rat.num_pos.mpr hpos
    exact val_congr (hval h0) (by rw [abs_of_pos hn, hr])
  · have hneg : r < 0 := lt_of_le_of_ne (not_lt.mp hpos) h0
    have hn : (r.num : ℝ) < 0 := by exact_mod_cast Rat.num_neg.mpr hneg
    exact val_congr (val_N hV (hval h0)) (by rw [abs_of_neg hn, hr]; ring)

/-- **Corollary 25 (a)**, in strict real arithmetic (Definition B.1): the closed term `rat_term r`
is defined, so every node has a real value and every logarithm is applied to a positive real
number, and its value is `r`. -/
theorem corollary25_a (r : ℚ) (a : Fin n → ℝ) : evalR (rat_term r : Term n) a = some (r : ℝ) :=
  val_rat_term (strictVal_evalR a) r

/-- **Corollary 25 (a)**, in the complex arithmetic of Theorem 1: the same term has the value
`r`. -/
theorem corollary25_a_complex (r : ℚ) (a : Fin n → ℂ) :
    eval (rat_term r : Term n) a = some ((r : ℝ) : ℂ) :=
  val_rat_term (strictVal_eval a) r

end partA

/-! ## Parts (b) and (c): floor, ceiling and the map `x ↦ 1/(2⌊x⌋ - x + 1)` -/

variable {n : ℕ} {a : Fin n → ℂ} {t : Term n} {x : ℝ}

/-- `w(x) := M(U(W×(π, W₋(1, W×(2, x)))), i)`. -/
def wfloor (t : Term n) : Term n := M (U (Wmul const_pi (Wsub one (Wmul const_2 t)))) const_i

/-- `W_⌊⌋(x) := W_half(W/(S(M(i, S(w(x), S(w(x), 1))), i), π))`. -/
def Wfloor (t : Term n) : Term n :=
  Whalf (Wdiv (S (M const_i (S (wfloor t) (S (wfloor t) one))) const_i) const_pi)

/-- `⌈x⌉ := N(W_⌊⌋(N(x)))`. -/
def Wceil (t : Term n) : Term n := N (Wfloor (N t))

/-- `R(x) := W_inv(W₊(W₋(W×(2, W_⌊⌋(x)), x), 1))`. -/
def R (t : Term n) : Term n := Winv (Wadd (Wsub (Wmul const_2 (Wfloor t)) t) one)

/-- The value of `w(x)`: with the real number `r = π (1 - 2x)`, `w(x) = (r + i) i = -1 + i r`. -/
theorem eval_wfloor (ht : eval t a = some (x : ℂ)) :
    eval (wfloor t) a = some (-1 + ((Real.pi * (1 - 2 * x) : ℝ) : ℂ) * I) := by
  have hr := eval_Wmul (eval_const_pi (a := a))
    (eval_Wsub eval_one_real (eval_Wmul eval_const_2_real ht))
  have h := eval_M (eval_U hr) (eval_const_i (a := a)) (ofReal_add_I_ne_zero _)
    (ofReal_add_I_ne_one _) I_mem_slitPlane
  exact eval_congr h (by linear_combination I_sq)

/-- **Corollary 25 (b).** For every real `x`, `W_⌊⌋(x)` is defined and equals the floor `⌊x⌋`. -/
theorem eval_Wfloor (ht : eval t a = some (x : ℂ)) :
    eval (Wfloor t) a = some (((⌊x⌋ : ℤ) : ℝ) : ℂ) := by
  have hpi : (0 : ℝ) < Real.pi := Real.pi_pos
  -- `w = -1 + i r` with `r = π (1 - 2x)`; `w ≠ 0`
  have hw := eval_wfloor (a := a) ht
  have hw0 : -1 + ((Real.pi * (1 - 2 * x) : ℝ) : ℂ) * I ≠ 0 := by
    intro h
    have := congrArg Complex.re h
    simp at this
  -- `S(w, 1) = w - 1`
  have h1 := eval_S_real hw eval_one_real hw0
  -- `S(w, w - 1) = w - Log(exp(w - 1)) = 1 + 2πi K(w - 1)`, and `K(w - 1) = ⌈-x⌉ = -⌊x⌋`
  have h2 := eval_S_defined hw h1 hw0
  have hK : ⌈((-1 + ((Real.pi * (1 - 2 * x) : ℝ) : ℂ) * I - ((1 : ℝ) : ℂ)).im - Real.pi)
      / (2 * Real.pi)⌉ = -⌊x⌋ := by
    have him : (-1 + ((Real.pi * (1 - 2 * x) : ℝ) : ℂ) * I - ((1 : ℝ) : ℂ)).im
        = Real.pi * (1 - 2 * x) := by
      simp
    rw [him]
    have hx : (Real.pi * (1 - 2 * x) - Real.pi) / (2 * Real.pi) = -x := by
      field_simp; ring
    rw [hx, Int.ceil_neg]
  have h3 : eval (S (wfloor t) (S (wfloor t) one)) a
      = some (1 - ((2 * Real.pi * ⌊x⌋ : ℝ) : ℂ) * I) :=
    eval_congr h2 (by rw [F2, hK]; push_cast; ring)
  -- `M(i, 1 - 2πi⌊x⌋) = 2π⌊x⌋ + i`: the second argument has real part `1`
  have hslit : 1 - ((2 * Real.pi * ⌊x⌋ : ℝ) : ℂ) * I ∈ slitPlane :=
    mem_slitPlane_iff.mpr (Or.inl (by simp))
  have h4 : eval (M const_i (S (wfloor t) (S (wfloor t) one))) a
      = some (((2 * Real.pi * ⌊x⌋ : ℝ) : ℂ) + I) :=
    eval_congr (eval_M (eval_const_i (a := a)) h3 I_ne_zero (ne_one_of_im_ne_zero (by simp)) hslit)
      (by linear_combination (-((2 * Real.pi * ⌊x⌋ : ℝ) : ℂ)) * I_sq)
  -- `S(2π⌊x⌋ + i, i) = 2π⌊x⌋`
  have h5 : eval (S (M const_i (S (wfloor t) (S (wfloor t) one))) const_i) a
      = some ((2 * Real.pi * ⌊x⌋ : ℝ) : ℂ) :=
    eval_congr (eval_S h4 (eval_const_i (a := a)) (ofReal_add_I_ne_zero _) neg_pi_lt_I_im
      I_im_le_pi) (by ring)
  -- division by `π` and halving
  have h6 := eval_Whalf (eval_Wdiv h5 (eval_const_pi (a := a)) hpi.ne')
  exact eval_congr h6 (by congr 1; field_simp)

/-- **Corollary 25 (b).** For every real `x`, `N(W_⌊⌋(N(x)))` equals the ceiling `⌈x⌉`. -/
theorem eval_Wceil (ht : eval t a = some (x : ℂ)) :
    eval (Wceil t) a = some (((⌈x⌉ : ℤ) : ℝ) : ℂ) :=
  eval_congr (eval_N_real (eval_Wfloor (eval_N_real ht)))
    (by rw [Int.floor_neg]; push_cast; ring)

/-- The number `2⌊x⌋ - x + 1 = ⌊x⌋ + 1 - (x - ⌊x⌋)` vanishes only for `x = -1`. -/
theorem two_floor_sub_add_one_ne_zero (hx : x ≠ -1) : 2 * (⌊x⌋ : ℝ) - x + 1 ≠ 0 := by
  intro h
  have hx' : x = ((2 * ⌊x⌋ + 1 : ℤ) : ℝ) := by push_cast; linarith
  have hfl : ⌊x⌋ = 2 * ⌊x⌋ + 1 := by
    conv_lhs => rw [hx']
    exact Int.floor_intCast _
  have hm : ⌊x⌋ = -1 := by omega
  apply hx
  rw [hx', hm]
  norm_num

/-- **Corollary 25 (c).** For every real `x ≠ -1`, `R(x)` is defined and
`R(x) = 1/(2⌊x⌋ - x + 1)`. -/
theorem eval_R (ht : eval t a = some (x : ℂ)) (hx : x ≠ -1) :
    eval (R t) a = some (((2 * (⌊x⌋ : ℝ) - x + 1)⁻¹ : ℝ) : ℂ) :=
  eval_Winv
    (eval_Wadd (eval_Wsub (eval_Wmul eval_const_2_real (eval_Wfloor ht)) ht) eval_one_real)
    (two_floor_sub_add_one_ne_zero hx)

end EML

#print axioms EML.corollary25_a
#print axioms EML.eval_Wfloor
#print axioms EML.eval_Wceil
#print axioms EML.eval_R
