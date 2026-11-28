module

public import EML.Real.Strict

/-!
# Theorem 1c (Appendix C): extended real `eml`

The inputs are finite real numbers. The value of a node may be a real number, `-∞` or `+∞`
(Mathlib's `EReal`, with `⊥ = -∞` and `⊤ = +∞`).

* `emlE p q`  — Definition C.1 and Table C.1: `eml(u, v) = exp u - ln v` with `ln 0 = -∞`,
  `ln(+∞) = +∞`, `exp(-∞) = 0`, `exp(+∞) = +∞`. `none` means "undefined": a negative right
  child (including `-∞`), or the difference `+∞ - (+∞)`.
* `evalE t a` — the value of the term `t` at the real point `a`: an `Option EReal`.
* `strictVal_evalE`, `evalE_of_evalR` — a strict real evaluation is an extended real evaluation,
  node by node. So all witnesses of Table B.1 carry over, also when their operands are terms
  whose finite values were computed through an infinity (Step 1 of the proof of Theorem 1c).
* `evalE_E_bot`, `evalE_L_zero`, … — Lemma C.1: `E`, `L`, `F` at the two ends.
* `evalE_Wsqrt1c` — Lemma C.2: the term (C.2) is equal to `√x` on `[0, ∞)`.
* `val_P1c`, `evalE_P1c_zero` — the sentence after Lemma C.2: positive powers at zero.
* `evalE_Whypot1c`, `evalE_Warcosh1c` — Corollary C.3: the witnesses on the closed domains
  `ℝ²` and `x ≥ 1`.

In the comments `eml` stands for the operator of this appendix (written with an overline in the
paper), and `w_P` for its witnesses (also with an overline there).
-/

@[expose] public section

namespace EML

open Term

variable {n : ℕ} {a : Fin n → ℝ} {s t : Term n} {u v : ℝ} {p q : EReal}

/-- **Definition C.1** and **Table C.1.** `eml(u, v) = exp u - ln v` on the extended reals.
The tests follow the columns of Table C.1 from left to right:
`v < 0` (including `v = -∞`): undefined; `v = 0`: `+∞`; `v = +∞`: `-∞`, but undefined if
also `u = +∞`; and for `0 < v < +∞`: `+∞` if `u = +∞`, `-ln v` if `u = -∞`, and
`exp u - ln v` if `u` is finite. -/
noncomputable def emlE (p q : EReal) : Option EReal :=
  if q < 0 then none
  else if q = 0 then some ⊤
  else if q = ⊤ then (if p = ⊤ then none else some ⊥)
  else if p = ⊤ then some ⊤
  else if p = ⊥ then some ((-Real.log q.toReal : ℝ) : EReal)
  else some ((Real.exp p.toReal - Real.log q.toReal : ℝ) : EReal)

/-! ### The entries of Table C.1 -/

/-- Columns `v = -∞` and `-∞ < v < 0`: undefined in every row. -/
theorem emlE_neg (hq : q < 0) : emlE p q = none := by
  simp [emlE, hq]

/-- Column `v = -∞`. -/
theorem emlE_right_bot : emlE p ⊥ = none :=
  emlE_neg EReal.bot_lt_zero

/-- Column `-∞ < v < 0`. -/
theorem emlE_right_coe_neg (hv : v < 0) : emlE p (v : EReal) = none :=
  emlE_neg (by exact_mod_cast hv)

/-- Column `v = 0`: `+∞` in every row. -/
theorem emlE_right_zero : emlE p ((0 : ℝ) : EReal) = some ⊤ := by
  simp [emlE]

/-- Column `0 < v < +∞`, row `u = -∞`: `-ln v`. -/
theorem emlE_bot_coe (hv : 0 < v) :
    emlE ⊥ (v : EReal) = some ((-Real.log v : ℝ) : EReal) := by
  have h1 : ¬ ((v : EReal) < 0) := not_lt.mpr (EReal.coe_nonneg.mpr hv.le)
  have h2 : (v : EReal) ≠ 0 := by exact_mod_cast hv.ne'
  simp [emlE, h1, h2]

/-- Column `0 < v < +∞`, row `u ∈ ℝ`: `exp u - ln v`. -/
theorem emlE_coe_coe (hv : 0 < v) :
    emlE (u : EReal) (v : EReal) = some ((Real.exp u - Real.log v : ℝ) : EReal) := by
  have h1 : ¬ ((v : EReal) < 0) := not_lt.mpr (EReal.coe_nonneg.mpr hv.le)
  have h2 : (v : EReal) ≠ 0 := by exact_mod_cast hv.ne'
  simp [emlE, h1, h2]

/-- Column `0 < v < +∞`, row `u = +∞`: `+∞`. -/
theorem emlE_top_coe (hv : 0 < v) : emlE ⊤ (v : EReal) = some ⊤ := by
  have h1 : ¬ ((v : EReal) < 0) := not_lt.mpr (EReal.coe_nonneg.mpr hv.le)
  have h2 : (v : EReal) ≠ 0 := by exact_mod_cast hv.ne'
  simp [emlE, h1, h2]

/-- Column `v = +∞`, row `u = -∞`: `-∞`. -/
theorem emlE_bot_top : emlE ⊥ ⊤ = some ⊥ := by
  simp [emlE]

/-- Column `v = +∞`, row `u ∈ ℝ`: `-∞`. -/
theorem emlE_coe_top : emlE (u : EReal) ⊤ = some ⊥ := by
  simp [emlE]

/-- Column `v = +∞`, row `u = +∞`: the difference `+∞ - (+∞)` is undefined. -/
theorem emlE_top_top : emlE ⊤ ⊤ = none := by
  simp [emlE]

/-! ### Terms -/

/-- **Definition C.1.** Value of a term at the finite real point `a` in extended real
arithmetic. A node is defined iff both children are defined and Table C.1 gives a value. -/
noncomputable def evalE : Term n → (Fin n → ℝ) → Option EReal
  | one, _ => some ((1 : ℝ) : EReal)
  | var j, a => some ((a j : ℝ) : EReal)
  | eml s t, a =>
    match evalE s a, evalE t a with
    | some p, some q => emlE p q
    | _, _ => none

/-- The domain of a term in extended real arithmetic: the finite real points where the term has
a value in `ℝ ∪ {-∞, +∞}`. -/
def domE (t : Term n) : Set (Fin n → ℝ) := {a | (evalE t a).isSome}

theorem evalE_one : evalE (one : Term n) a = some ((1 : ℝ) : EReal) := rfl

theorem evalE_var (j : Fin n) : evalE (var j) a = some ((a j : ℝ) : EReal) := rfl

/-- The rule for a node: the value is read off Table C.1. -/
theorem evalE_eml (hs : evalE s a = some p) (ht : evalE t a = some q) :
    evalE (eml s t) a = emlE p q := by
  simp [evalE, hs, ht]

/-- A node with an undefined left child is undefined. -/
theorem evalE_eml_undef_left (hs : evalE s a = none) : evalE (eml s t) a = none := by
  simp [evalE, hs]

/-- A node with an undefined right child is undefined. -/
theorem evalE_eml_undef_right (ht : evalE t a = none) : evalE (eml s t) a = none := by
  cases h : evalE s a <;> simp [evalE, h, ht]

/-- `ValE a t u`: in extended real arithmetic the term `t` is defined at the point `a` and has
the finite value `u`. -/
abbrev ValE (a : Fin n → ℝ) (t : Term n) (u : ℝ) : Prop := evalE t a = some (u : EReal)

/-- Step 1 of the proof of Theorem 1c. The extended real evaluation respects strict real
nodes: for finite children with a positive right child, Table C.1 gives `exp u - ln v`. Hence
every lemma `val_…` of `EML.Real.Strict` holds for `evalE`, for operands with finite values. -/
theorem strictVal_evalE (a : Fin n → ℝ) : StrictVal (ValE a) where
  one := evalE_one
  eml := fun hs ht hv => by
    change evalE (Term.eml _ _) a = some _
    rw [evalE_eml hs ht, emlE_coe_coe hv]

/-- A term that is defined at a point in strict real arithmetic has the same value in extended
real arithmetic. -/
theorem evalE_of_evalR {t : Term n} {w : ℝ} (h : evalR t a = some w) :
    evalE t a = some (w : EReal) := by
  induction t generalizing w with
  | one =>
    rw [evalR_one, Option.some.injEq] at h
    rw [evalE_one, h]
  | var j =>
    rw [evalR_var, Option.some.injEq] at h
    rw [evalE_var, h]
  | eml s t ihs iht =>
    obtain ⟨u, v, hs, ht, hv, rfl⟩ := evalR_eml_eq_some_iff.mp h
    rw [evalE_eml (ihs hs) (iht ht), emlE_coe_coe hv]

/-! ## Lemma C.1: exponential and logarithm at the two ends -/

/-- **Lemma C.1.** `E(-∞) = 0`. -/
theorem evalE_E_bot (ht : evalE t a = some ⊥) : evalE (E t) a = some ((0 : ℝ) : EReal) := by
  rw [E, evalE_eml ht evalE_one, emlE_bot_coe one_pos]
  simp

/-- **Lemma C.1.** `E(+∞) = +∞`. -/
theorem evalE_E_top (ht : evalE t a = some ⊤) : evalE (E t) a = some ⊤ := by
  rw [E, evalE_eml ht evalE_one, emlE_top_coe one_pos]

/-- **Lemma C.1.** `L(0) = -∞`: the successive values are `+∞, +∞, -∞`. -/
theorem evalE_L_zero (ht : evalE t a = some ((0 : ℝ) : EReal)) : evalE (L t) a = some ⊥ := by
  have h1 : evalE (eml one t) a = some ⊤ := by
    rw [evalE_eml evalE_one ht, emlE_right_zero]
  rw [L, evalE_eml evalE_one (evalE_E_top h1), emlE_coe_top]

/-- **Lemma C.1.** `L(+∞) = +∞`: the successive values are `-∞, 0, +∞`. -/
theorem evalE_L_top (ht : evalE t a = some ⊤) : evalE (L t) a = some ⊤ := by
  have h1 : evalE (eml one t) a = some ⊥ := by
    rw [evalE_eml evalE_one ht, emlE_coe_top]
  rw [L, evalE_eml evalE_one (evalE_E_bot h1), emlE_right_zero]

/-- **Lemma C.1.** `F(t, +∞) = e^t - ∞ = -∞` for finite `t`. -/
theorem evalE_F1b_top (hs : evalE s a = some (u : EReal)) (ht : evalE t a = some ⊤) :
    evalE (F1b s t) a = some ⊥ := by
  rw [F1b, evalE_eml hs (evalE_E_top ht), emlE_coe_top]

/-- **Lemma C.1.** `F(t, -∞) = e^t + ∞ = +∞` for finite `t`. -/
theorem evalE_F1b_bot (hs : evalE s a = some (u : EReal)) (ht : evalE t a = some ⊥) :
    evalE (F1b s t) a = some ⊤ := by
  rw [F1b, evalE_eml hs (evalE_E_bot ht), emlE_right_zero]

/-- `+∞ := eml(1, 0)`, a closed term. (The same term as `const_pinf` of Theorem 1a.) -/
def const_pinf1c : Term n := eml one const_0

/-- `-∞ := eml(1, +∞)`, a closed term. (The same term as `const_ninf` of Theorem 1a.) -/
def const_ninf1c : Term n := eml one const_pinf1c

/-- **Lemma C.1.** The closed term `eml(1, 0)` has the value `+∞`. -/
theorem evalE_const_pinf1c : evalE (const_pinf1c : Term n) a = some ⊤ := by
  rw [const_pinf1c, evalE_eml evalE_one (val_const_0 (strictVal_evalE a)), emlE_right_zero]

/-- **Lemma C.1.** The closed term `eml(1, +∞)` has the value `-∞`. -/
theorem evalE_const_ninf1c : evalE (const_ninf1c : Term n) a = some ⊥ := by
  rw [const_ninf1c, evalE_eml evalE_one evalE_const_pinf1c, emlE_coe_top]

/-! ## Lemma C.2: the square root on `[0, ∞)` -/

variable {V : Term n → ℝ → Prop}

/-- `e^x - ln x ≥ 2` for `x > 0`, from `e^x ≥ 1 + x` and `ln x ≤ x - 1`. -/
theorem two_le_exp_sub_log (hu : 0 < u) : 2 ≤ Real.exp u - Real.log u := by
  have h1 := Real.add_one_le_exp u
  have h2 := Real.log_le_sub_one_of_pos hu
  linarith

/-- Equation (C.2): `w_√(x) := E(F(x - L(2), E(eml(L(L(eml(x, x))), 2))))`, where `x - L(2)`
is the witness `w₋(x, L(2))` and `2` is the closed term `w₊(1, 1)` of Appendix B. -/
def Wsqrt1c (t : Term n) : Term n :=
  E (F1b (Wsub1b t (L const_2_1b)) (E (eml (L (L (eml t t))) const_2_1b)))

/-- **Lemma C.2**, the case `x > 0`. All nodes are strict real nodes: with `u = e^x - ln x ≥ 2`
the inner exponential is `u/2`, and the outer exponential has the argument `(ln x)/2`. So this
part holds in every semantics. -/
theorem val_Wsqrt1c (hV : StrictVal V) (ht : V t u) (hu : 0 < u) :
    V (Wsqrt1c t) (Real.sqrt u) := by
  have hsub : V (Wsub1b t (L const_2_1b)) (u - Real.log 2) :=
    val_Wsub1b hV ht (val_L hV (val_const_2_1b hV) two_pos)
  have hw2 := two_le_exp_sub_log hu
  have hw0 : 0 < Real.exp u - Real.log u := by linarith
  have hlw : 0 < Real.log (Real.exp u - Real.log u) := Real.log_pos (by linarith)
  have hU : V (eml t t) (Real.exp u - Real.log u) := hV.eml ht ht hu
  have h3 := hV.eml (val_L hV (val_L hV hU hw0) hlw) (val_const_2_1b hV) two_pos
  refine val_congr (val_E hV (val_F1b hV hsub (val_E hV h3))) ?_
  have e1 : Real.exp (Real.log (Real.exp u - Real.log u) - Real.log 2)
      = (Real.exp u - Real.log u) / 2 := by
    rw [Real.exp_sub, Real.exp_log hw0, Real.exp_log two_pos]
  have e2 : Real.exp (u - Real.log 2) = Real.exp u / 2 := by
    rw [Real.exp_sub, Real.exp_log two_pos]
  rw [Real.exp_log hlw, e1, e2,
    show Real.exp u / 2 - (Real.exp u - Real.log u) / 2 = Real.log u / 2 by ring,
    Real.exp_half, Real.exp_log hu]

/-- **Lemma C.2.** `w_√(x) = √x` for every real `x ≥ 0`. At `x = 0` the successive values are
`eml(0, 0) = +∞`, `L(L(+∞)) = +∞`, `eml(+∞, 2) = +∞`, `E(+∞) = +∞`, `F(-ln 2, +∞) = -∞`,
`E(-∞) = 0`. -/
theorem evalE_Wsqrt1c (ht : evalE t a = some (u : EReal)) (hu : 0 ≤ u) :
    evalE (Wsqrt1c t) a = some ((Real.sqrt u : ℝ) : EReal) := by
  have hV := strictVal_evalE a
  rcases hu.eq_or_lt with rfl | hpos
  · have h2 : ValE a const_2_1b 2 := val_const_2_1b hV
    have hsub : ValE a (Wsub1b t (L const_2_1b)) (0 - Real.log 2) :=
      val_Wsub1b hV ht (val_L hV h2 two_pos)
    have hU : evalE (eml t t) a = some ⊤ := by
      rw [evalE_eml ht ht, emlE_right_zero]
    have h3 : evalE (eml (L (L (eml t t))) const_2_1b) a = some ⊤ := by
      rw [evalE_eml (evalE_L_top (evalE_L_top hU)) h2, emlE_top_coe two_pos]
    rw [Real.sqrt_zero]
    exact evalE_E_bot (evalE_F1b_top hsub (evalE_E_top h3))
  · exact val_Wsqrt1c hV ht hpos

/-! ## The sentence after Lemma C.2: positive powers at zero

For a term `r` with a value `r > 0`, the term `P(x, r) = E(F(x + L(r), E(eml(L(L(eml(x, x))),
1/r))))` is equal to `x^r` for `x > 0` and to `0` at `x = 0`. The term of Lemma C.2 is the case
`r = 1/2`, with `-L(2)` for `L(1/2)` and `2` for `1/(1/2)`. -/

/-- `U(x) := eml(x, x)`. -/
def U1c (t : Term n) : Term n := eml t t

/-- `V(u, r) := E(eml(L(L(u)), 1/r))`. -/
def V1c (s t : Term n) : Term n := E (eml (L (L s)) (Winv1b t))

/-- `P(x, r) := E(F(x + L(r), V(U(x), r)))`. -/
def P1c (s t : Term n) : Term n := E (F1b (Wadd1b s (L t)) (V1c (U1c s) t))

/-- `U(x) = e^x - ln x` for `x > 0`. -/
theorem val_U1c (hV : StrictVal V) (ht : V t u) (hu : 0 < u) :
    V (U1c t) (Real.exp u - Real.log u) :=
  hV.eml ht ht hu

/-- `V(u, r) = r u` for `u > 1` and `r > 0`. -/
theorem val_V1c (hV : StrictVal V) (hs : V s u) (ht : V t v) (hu : 1 < u) (hv : 0 < v) :
    V (V1c s t) (v * u) := by
  have hu0 : 0 < u := by linarith
  have hlu : 0 < Real.log u := Real.log_pos hu
  have h3 := hV.eml (val_L hV (val_L hV hs hu0) hlu) (val_Winv1b hV ht hv.ne') (inv_pos.mpr hv)
  refine val_congr (val_E hV h3) ?_
  rw [Real.exp_log hlu, Real.log_inv, sub_neg_eq_add, Real.exp_add, Real.exp_log hu0,
    Real.exp_log hv]
  ring

/-- `P(x, r) = x^r` for `x > 0` and `r > 0`. All nodes are strict real nodes, so this holds in
every semantics. -/
theorem val_P1c (hV : StrictVal V) (hs : V s u) (ht : V t v) (hu : 0 < u) (hv : 0 < v) :
    V (P1c s t) (u ^ v) := by
  have h2 := two_le_exp_sub_log hu
  have hVv := val_V1c hV (val_U1c hV hs hu) ht (by linarith) hv
  have hsum := val_Wadd1b hV hs (val_L hV ht hv)
  refine val_congr (val_E hV (val_F1b hV hsum hVv)) ?_
  rw [Real.exp_add, Real.exp_log hv, Real.rpow_def_of_pos hu]
  congr 1
  ring

/-- `P(0, r) = 0` for `r > 0`: the successive values are `U(0) = +∞`, `V(+∞, r) = +∞`,
`F(ln r, +∞) = -∞`, `E(-∞) = 0`. -/
theorem evalE_P1c_zero (hs : evalE s a = some ((0 : ℝ) : EReal))
    (ht : evalE t a = some (v : EReal)) (hv : 0 < v) :
    evalE (P1c s t) a = some ((0 : ℝ) : EReal) := by
  have hU : evalE (U1c s) a = some ⊤ := by
    rw [U1c, evalE_eml hs hs, emlE_right_zero]
  have hinv : evalE (Winv1b t) a = some ((v⁻¹ : ℝ) : EReal) :=
    val_Winv1b (strictVal_evalE a) ht hv.ne'
  have h3 : evalE (eml (L (L (U1c s))) (Winv1b t)) a = some ⊤ := by
    rw [evalE_eml (evalE_L_top (evalE_L_top hU)) hinv, emlE_top_coe (inv_pos.mpr hv)]
  have hVv : evalE (V1c (U1c s) t) a = some ⊤ := evalE_E_top h3
  have hsum : evalE (Wadd1b s (L t)) a = some ((0 + Real.log v : ℝ) : EReal) :=
    val_Wadd1b (strictVal_evalE a) hs (val_L (strictVal_evalE a) ht hv)
  exact evalE_E_bot (evalE_F1b_top hsum hVv)

/-- `1/2 := half(1)`, a closed strict real term. -/
def const_half1b : Term n := Whalf1b one

theorem val_const_half1b (hV : StrictVal V) : V const_half1b (1 / 2) :=
  val_Whalf1b hV hV.one

/-- The case `r = 1/2`: `P(x, 1/2) = √x` for every real `x ≥ 0`. -/
theorem evalE_P1c_half (ht : evalE t a = some (u : EReal)) (hu : 0 ≤ u) :
    evalE (P1c t const_half1b) a = some ((Real.sqrt u : ℝ) : EReal) := by
  have hh : ValE a const_half1b (1 / 2) := val_const_half1b (strictVal_evalE a)
  rcases hu.eq_or_lt with rfl | hpos
  · rw [Real.sqrt_zero]
    exact evalE_P1c_zero ht hh (by norm_num)
  · exact val_congr (val_P1c (strictVal_evalE a) ht hh hpos (by norm_num))
      (by rw [Real.sqrt_eq_rpow])

/-! ## Corollary C.3: hypotenuse and inverse hyperbolic cosine -/

/-- Equation (C.3): `w_hypot(x, y) := w_√(x² + y²)`. -/
def Whypot1c (s t : Term n) : Term n := Wsqrt1c (Wadd1b (Wsqr1b s) (Wsqr1b t))

/-- **Corollary C.3.** `w_hypot(x, y) = √(x² + y²)` for all real `x`, `y`, including
`w_hypot(0, 0) = 0`. -/
theorem evalE_Whypot1c (hs : evalE s a = some (u : EReal)) (ht : evalE t a = some (v : EReal)) :
    evalE (Whypot1c s t) a = some ((Real.sqrt (u ^ 2 + v ^ 2) : ℝ) : EReal) :=
  evalE_Wsqrt1c
    (val_Wadd1b (strictVal_evalE a) (val_Wsqr1b (strictVal_evalE a) hs)
      (val_Wsqr1b (strictVal_evalE a) ht))
    (by positivity)

/-- Equation (C.4): `w_arcosh(x) := L(x + w_√(x² - 1))`. -/
def Warcosh1c (t : Term n) : Term n := L (Wadd1b t (Wsqrt1c (Wsub1b (Wsqr1b t) one)))

/-- **Corollary C.3.** `w_arcosh(x) = arcosh x` for every real `x ≥ 1`, including
`w_arcosh(1) = 0`. -/
theorem evalE_Warcosh1c (ht : evalE t a = some (u : EReal)) (hu : 1 ≤ u) :
    evalE (Warcosh1c t) a = some ((Real.arcosh u : ℝ) : EReal) := by
  have hV := strictVal_evalE a
  have h0 : 0 ≤ u ^ 2 - 1 := by nlinarith
  have hs : ValE a (Wsqrt1c (Wsub1b (Wsqr1b t) one)) (Real.sqrt (u ^ 2 - 1)) :=
    evalE_Wsqrt1c (val_Wsub1b hV (val_Wsqr1b hV ht) hV.one) h0
  have hpos : 0 < u + Real.sqrt (u ^ 2 - 1) := by
    have := Real.sqrt_nonneg (u ^ 2 - 1)
    linarith
  exact val_congr (val_L hV (val_Wadd1b hV ht hs) hpos) (by rw [Real.arcosh])

end EML
