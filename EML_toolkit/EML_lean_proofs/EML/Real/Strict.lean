module

public import EML.Core

/-!
# Theorem 1b (Appendix B): strict real `eml`

Every node is evaluated in the real numbers: `eml(u, v) = exp u - ln v` with `u ∈ ℝ`, `v > 0`.
No complex numbers and no infinite values occur.

* `evalR t a` — Definition B.1: the value of the term `t` at the real point `a`.
  It is an `Option ℝ`: `some r` means "defined, with value `r`", `none` means "undefined"
  (the right child of some node is not strictly positive).
* `StrictVal V` — what the witness lemmas need to know about an evaluation: the leaf `1` has the
  value `1`, and a node whose children have the real values `u` and `v > 0` has the value
  `exp u - ln v`. The strict real evaluation has this property (`strictVal_evalR`); so have the
  extended real one (`EML.Real.Extended`) and the complex one of Theorem 1 (`strictVal_eval`).
* `val_E`, `val_L`, `val_Wadd1b`, … — Lemmas B.1–B.4: the witnesses of Table B.1, in the order
  of the paper, proved once for every evaluation with the property `StrictVal`.
* `val_Wsqrt1b`, `val_Whypot1b`, `val_Warcosh1b` — Lemma B.7, first half: the witnesses on the
  restricted domains `x > 0`, `(x, y) ≠ (0, 0)`, `x > 1`.

The terms (`Term n`) are those of Theorem 1; only their evaluation is new. `E` and `L` are the
terms of Theorem 1 (`E t = eml(t, 1)`, `L t = eml(1, E(eml(1, t)))`); the other witnesses differ
from those of Theorem 1 and carry the suffix `1b`. The auxiliary terms `F`, `B`, `G`, `Q` of
Appendix B are `F1b`, `B1b`, `G1b`, `Q1b`, and its witnesses `w_P` are `Wadd1b`, `Wsub1b`, ….

Mathlib's `Real.log` is total (`Real.log 0 = 0`, `Real.log (-x) = Real.log x`). These conventions
are never used: `evalR` returns `none` as soon as the right child of a node is not positive.
-/

@[expose] public section

namespace EML

open Term

variable {n : ℕ} {a : Fin n → ℝ} {s t : Term n} {u v x y : ℝ}

/-- **Definition B.1.** Value of a term at the real point `a` in strict real arithmetic.
`eml(S,T)` is defined iff `S` and `T` are defined and the value of `T` is strictly positive;
its value is then `exp (value of S) - ln (value of T)`. -/
noncomputable def evalR : Term n → (Fin n → ℝ) → Option ℝ
  | one, _ => some 1
  | var j, a => some (a j)
  | eml s t, a =>
    match evalR s a, evalR t a with
    | some u, some v => if 0 < v then some (Real.exp u - Real.log v) else none
    | _, _ => none

/-- The domain of a term in strict real arithmetic: the points where it is defined. -/
def domR (t : Term n) : Set (Fin n → ℝ) := {a | (evalR t a).isSome}

@[simp] theorem evalR_one : evalR (one : Term n) a = some 1 := rfl

@[simp] theorem evalR_var (j : Fin n) : evalR (var j) a = some (a j) := rfl

/-- The rule for a node whose right child is positive. -/
theorem evalR_eml (hs : evalR s a = some u) (ht : evalR t a = some v) (hv : 0 < v) :
    evalR (eml s t) a = some (Real.exp u - Real.log v) := by
  simp [evalR, hs, ht, hv]

/-- A node whose right child is not positive is undefined. -/
theorem evalR_eml_nonpos (ht : evalR t a = some v) (hv : v ≤ 0) : evalR (eml s t) a = none := by
  cases h : evalR s a <;> simp [evalR, h, ht, not_lt.mpr hv]

/-- A node with an undefined left child is undefined. -/
theorem evalR_eml_undef_left (hs : evalR s a = none) : evalR (eml s t) a = none := by
  simp [evalR, hs]

/-- A node with an undefined right child is undefined. -/
theorem evalR_eml_undef_right (ht : evalR t a = none) : evalR (eml s t) a = none := by
  cases h : evalR s a <;> simp [evalR, h, ht]

/-- A node is defined exactly when both children are defined and the right one is positive. -/
theorem evalR_eml_eq_some_iff {r : ℝ} :
    evalR (eml s t) a = some r ↔
      ∃ u v, evalR s a = some u ∧ evalR t a = some v ∧ 0 < v ∧ Real.exp u - Real.log v = r := by
  constructor
  · intro h
    cases hs : evalR s a with
    | none => rw [evalR_eml_undef_left hs] at h; cases h
    | some u =>
      cases ht : evalR t a with
      | none => rw [evalR_eml_undef_right ht] at h; cases h
      | some v =>
        by_cases hv : 0 < v
        · rw [evalR_eml hs ht hv] at h
          exact ⟨u, v, rfl, rfl, hv, Option.some.inj h⟩
        · rw [evalR_eml_nonpos ht (not_lt.mp hv)] at h; cases h
  · rintro ⟨u, v, hs, ht, hv, rfl⟩
    exact evalR_eml hs ht hv

/-! ## Evaluations that respect strict real nodes -/

/-- `V t u` is read "the term `t` has the real value `u`". Such a relation *respects strict real
nodes* if the leaf `1` has the value `1`, and a node `eml(S, T)` whose children have the real
values `u` and `v > 0` has the value `exp u - ln v`. Nothing is assumed about nodes whose right
child is not positive. -/
structure StrictVal (V : Term n → ℝ → Prop) : Prop where
  /-- the leaf `1` has the value `1` -/
  one : V one 1
  /-- a node with real children `u` and `v > 0` has the value `exp u - ln v` -/
  eml : ∀ {s t : Term n} {u v : ℝ}, V s u → V t v → 0 < v →
    V (eml s t) (Real.exp u - Real.log v)

/-- `ValR a t u`: in strict real arithmetic the term `t` is defined at the point `a` and has
the value `u`. -/
abbrev ValR (a : Fin n → ℝ) (t : Term n) (u : ℝ) : Prop := evalR t a = some u

/-- The strict real evaluation at a point respects strict real nodes. -/
theorem strictVal_evalR (a : Fin n → ℝ) : StrictVal (ValR a) where
  one := evalR_one
  eml := fun hs ht hv => evalR_eml hs ht hv

/-- `ValC a t u`: in the complex arithmetic of Theorem 1 the term `t` is defined at the point
`a` and has the real value `u`. -/
abbrev ValC (a : Fin n → ℂ) (t : Term n) (u : ℝ) : Prop := eval t a = some (u : ℂ)

/-- The complex evaluation of Theorem 1 at a point respects strict real nodes: for real `u` and
real `v > 0` the principal logarithm is the real one. Hence every witness of Table B.1 is also
a term of Theorem 1 with the same value, as long as its operands have real values. -/
theorem strictVal_eval (a : Fin n → ℂ) : StrictVal (ValC a) where
  one := by simp [ValC]
  eml := fun {s t u v} hs ht hv => by
    have hv0 : (v : ℂ) ≠ 0 := by exact_mod_cast hv.ne'
    change eval (Term.eml s t) a = some _
    rw [eval_eml hs ht hv0]
    congr 1
    rw [Complex.ofReal_sub, Complex.ofReal_exp, Complex.ofReal_log hv.le]

variable {V : Term n → ℝ → Prop}

/-- If a term has the value `u` and `u = u'`, it has the value `u'`. -/
theorem val_congr (h : V t u) {u' : ℝ} (hu : u = u') : V t u' :=
  hu ▸ h

/-! ## Lemma B.1: exponential, logarithm, addition and subtraction on all of ℝ² -/

/-- **Lemma B.1.** `E(t) = eml(t, 1) = e^t` for every real `t`. -/
theorem val_E (hV : StrictVal V) (ht : V t u) : V (E t) (Real.exp u) :=
  val_congr (hV.eml ht hV.one one_pos) (by simp)

/-- **Lemma B.1.** `L(u) = eml(1, E(eml(1, u))) = ln u` for `u > 0`. -/
theorem val_L (hV : StrictVal V) (ht : V t u) (hu : 0 < u) : V (L t) (Real.log u) :=
  val_congr (hV.eml hV.one (val_E hV (hV.eml hV.one ht hu)) (Real.exp_pos _))
    (by rw [Real.log_exp]; ring)

/-- `F(t, v) := eml(t, E(v))`. -/
def F1b (s t : Term n) : Term n := eml s (E t)

/-- **Lemma B.1.** `F(t, v) = e^t - v` for all real `t`, `v`. -/
theorem val_F1b (hV : StrictVal V) (hs : V s u) (ht : V t v) :
    V (F1b s t) (Real.exp u - v) :=
  val_congr (hV.eml hs (val_E hV ht) (Real.exp_pos _)) (by rw [Real.log_exp])

/-- `B(t) := L(F(t, t))`. -/
def B1b (t : Term n) : Term n := L (F1b t t)

/-- The tangent inequality `e^t ≥ 1 + t` in the form used below: `e^t - t ≥ 1 > 0`. -/
theorem exp_sub_self_pos (u : ℝ) : 0 < Real.exp u - u := by
  linarith [Real.add_one_le_exp u]

/-- **Lemma B.1.** `B(t) = ln(e^t - t)` for every real `t`; in particular `e^{B(t)} = e^t - t`. -/
theorem val_B1b (hV : StrictVal V) (ht : V t u) :
    V (B1b t) (Real.log (Real.exp u - u)) :=
  val_L hV (val_F1b hV ht ht) (exp_sub_self_pos u)

/-- `w₊(x, y) := F(x, F(B(x), y))`. -/
def Wadd1b (s t : Term n) : Term n := F1b s (F1b (B1b s) t)

/-- **Lemma B.1.** `w₊(x, y) = x + y` for all real `x`, `y`. -/
theorem val_Wadd1b (hV : StrictVal V) (hs : V s u) (ht : V t v) :
    V (Wadd1b s t) (u + v) :=
  val_congr (val_F1b hV hs (val_F1b hV (val_B1b hV hs) ht))
    (by rw [Real.exp_log (exp_sub_self_pos u)]; ring)

/-- `w₋(x, y) := F(B(y), F(y, x))`. -/
def Wsub1b (s t : Term n) : Term n := F1b (B1b t) (F1b t s)

/-- **Lemma B.1.** `w₋(x, y) = x - y` for all real `x`, `y`. -/
theorem val_Wsub1b (hV : StrictVal V) (hs : V s u) (ht : V t v) :
    V (Wsub1b s t) (u - v) :=
  val_congr (val_F1b hV (val_B1b hV ht) (val_F1b hV ht hs))
    (by rw [Real.exp_log (exp_sub_self_pos v)]; ring)

/-! ## The constants `e`, `0`, `-1`, `2` and the unary minus -/

/-- `e = E(1)`. (`const_e` of Theorem 1 is the same term `eml(1, 1)`.) -/
theorem val_const_e (hV : StrictVal V) : V const_e (Real.exp 1) :=
  val_E hV hV.one

/-- `0 := L(1)`. (`const_0` of Theorem 1 is the same term.) -/
theorem val_const_0 (hV : StrictVal V) : V const_0 0 :=
  val_congr (val_L hV hV.one one_pos) Real.log_one

/-- `-x := w₋(0, x)`. -/
def Wminus1b (t : Term n) : Term n := Wsub1b const_0 t

/-- `w₋(0, x) = -x` for every real `x`. -/
theorem val_Wminus1b (hV : StrictVal V) (ht : V t u) : V (Wminus1b t) (-u) :=
  val_congr (val_Wsub1b hV (val_const_0 hV) ht) (by ring)

/-- `-1 := w₋(0, 1)`. -/
def const_neg1b : Term n := Wminus1b one

theorem val_const_neg1b (hV : StrictVal V) : V const_neg1b (-1) :=
  val_Wminus1b hV hV.one

/-- `2 := w₊(1, 1)`. -/
def const_2_1b : Term n := Wadd1b one one

theorem val_const_2_1b (hV : StrictVal V) : V const_2_1b 2 :=
  val_congr (val_Wadd1b hV hV.one hV.one) (by norm_num)

/-! ## Lemma B.2: multiplication on all of ℝ² -/

/-- `G(u, y) := E(u + y) - E(u + B(y))`. -/
def G1b (s t : Term n) : Term n := Wsub1b (E (Wadd1b s t)) (E (Wadd1b s (B1b t)))

/-- **Lemma B.2.** `G(u, y) = e^u · y` for all real `u`, `y`. -/
theorem val_G1b (hV : StrictVal V) (hs : V s u) (ht : V t v) :
    V (G1b s t) (Real.exp u * v) :=
  val_congr
    (val_Wsub1b hV (val_E hV (val_Wadd1b hV hs ht))
      (val_E hV (val_Wadd1b hV hs (val_B1b hV ht))))
    (by rw [Real.exp_add, Real.exp_add, Real.exp_log (exp_sub_self_pos v)]; ring)

/-- `w×(x, y) := G(x, y) - G(B(x), y)`. -/
def Wmul1b (s t : Term n) : Term n := Wsub1b (G1b s t) (G1b (B1b s) t)

/-- **Lemma B.2.** `w×(x, y) = x y` for all real `x`, `y`, including `x = 0` and `y = 0`. -/
theorem val_Wmul1b (hV : StrictVal V) (hs : V s u) (ht : V t v) :
    V (Wmul1b s t) (u * v) :=
  val_congr (val_Wsub1b hV (val_G1b hV hs ht) (val_G1b hV (val_B1b hV hs) ht))
    (by rw [Real.exp_log (exp_sub_self_pos u)]; ring)

/-! ## Lemma B.3: square, reciprocal, quotient, half, mean -/

/-- `w_sqr(y) := w×(y, y)`. -/
def Wsqr1b (t : Term n) : Term n := Wmul1b t t

/-- **Lemma B.3.** `w_sqr(y) = y²` for every real `y`. -/
theorem val_Wsqr1b (hV : StrictVal V) (ht : V t u) : V (Wsqr1b t) (u ^ 2) :=
  val_congr (val_Wmul1b hV ht ht) (by ring)

/-- `w_inv(y) := y · E(-L(y²))`. -/
def Winv1b (t : Term n) : Term n := Wmul1b t (E (Wminus1b (L (Wsqr1b t))))

/-- **Lemma B.3.** `w_inv(y) = 1/y` for every real `y ≠ 0`, including `y < 0`. -/
theorem val_Winv1b (hV : StrictVal V) (ht : V t u) (hu : u ≠ 0) : V (Winv1b t) u⁻¹ := by
  have h2 : 0 < u ^ 2 := by positivity
  refine val_congr
    (val_Wmul1b hV ht (val_E hV (val_Wminus1b hV (val_L hV (val_Wsqr1b hV ht) h2)))) ?_
  rw [Real.exp_neg, Real.exp_log h2]
  field_simp

/-- `w/(x, y) := x · w_inv(y)`. -/
def Wdiv1b (s t : Term n) : Term n := Wmul1b s (Winv1b t)

/-- **Lemma B.3.** `w/(x, y) = x / y` for every real `x` and every real `y ≠ 0`. -/
theorem val_Wdiv1b (hV : StrictVal V) (hs : V s u) (ht : V t v) (hv : v ≠ 0) :
    V (Wdiv1b s t) (u / v) :=
  val_congr (val_Wmul1b hV hs (val_Winv1b hV ht hv)) (by rw [div_eq_mul_inv])

/-- `w_half(x) := x / 2`. -/
def Whalf1b (t : Term n) : Term n := Wdiv1b t const_2_1b

/-- **Lemma B.3.** `w_half(x) = x/2` for every real `x`. -/
theorem val_Whalf1b (hV : StrictVal V) (ht : V t u) : V (Whalf1b t) (u / 2) :=
  val_Wdiv1b hV ht (val_const_2_1b hV) two_ne_zero

/-- `w_avg(x, y) := half(x + y)`. -/
def Wavg1b (s t : Term n) : Term n := Whalf1b (Wadd1b s t)

/-- **Lemma B.3.** `w_avg(x, y) = (x + y)/2` for all real `x`, `y`. -/
theorem val_Wavg1b (hV : StrictVal V) (hs : V s u) (ht : V t v) :
    V (Wavg1b s t) ((u + v) / 2) :=
  val_Whalf1b hV (val_Wadd1b hV hs ht)

/-! ## Lemma B.4: exponential and logarithmic witnesses -/

/-- `w_σ(x) := inv(1 + E(-x))`. -/
def Wsigma1b (t : Term n) : Term n := Winv1b (Wadd1b one (E (Wminus1b t)))

/-- **Lemma B.4.** `w_σ(x) = 1/(1 + e^{-x})` for every real `x`. -/
theorem val_Wsigma1b (hV : StrictVal V) (ht : V t u) : V (Wsigma1b t) (Real.sigmoid u) :=
  val_congr (val_Winv1b hV (val_Wadd1b hV hV.one (val_E hV (val_Wminus1b hV ht)))
    (by positivity)) (by rw [Real.sigmoid_def])

/-- `w_pow(x, y) := E(y · L(x))`. -/
def Wpow1b (s t : Term n) : Term n := E (Wmul1b t (L s))

/-- **Lemma B.4.** `w_pow(x, y) = x^y` for real `x > 0` and real `y`. -/
theorem val_Wpow1b (hV : StrictVal V) (hs : V s u) (ht : V t v) (hu : 0 < u) :
    V (Wpow1b s t) (u ^ v) :=
  val_congr (val_E hV (val_Wmul1b hV ht (val_L hV hs hu)))
    (by rw [Real.rpow_def_of_pos hu, mul_comm])

/-- `w_log(b, y) := L(y) / L(b)`. -/
def Wlog1b (s t : Term n) : Term n := Wdiv1b (L t) (L s)

/-- **Lemma B.4.** `w_log(b, y) = log_b y` for real `b > 0`, `b ≠ 1` and real `y > 0`. -/
theorem val_Wlog1b (hV : StrictVal V) (hs : V s u) (ht : V t v)
    (hu : 0 < u) (hu1 : u ≠ 1) (hv : 0 < v) :
    V (Wlog1b s t) (Real.logb u v) :=
  val_congr (val_Wdiv1b hV (val_L hV ht hv) (val_L hV hs hu)
    (Real.log_ne_zero_of_pos_of_ne_one hu hu1)) (by rw [Real.log_div_log])

/-- `w_sinh(x) := half(E(x) - E(-x))`. -/
def Wsinh1b (t : Term n) : Term n := Whalf1b (Wsub1b (E t) (E (Wminus1b t)))

/-- **Lemma B.4.** `w_sinh(x) = sinh x` for every real `x`. -/
theorem val_Wsinh1b (hV : StrictVal V) (ht : V t u) : V (Wsinh1b t) (Real.sinh u) :=
  val_congr (val_Whalf1b hV (val_Wsub1b hV (val_E hV ht) (val_E hV (val_Wminus1b hV ht))))
    (by rw [Real.sinh_eq])

/-- `w_cosh(x) := half(E(x) + E(-x))`. -/
def Wcosh1b (t : Term n) : Term n := Whalf1b (Wadd1b (E t) (E (Wminus1b t)))

/-- **Lemma B.4.** `w_cosh(x) = cosh x` for every real `x`. -/
theorem val_Wcosh1b (hV : StrictVal V) (ht : V t u) : V (Wcosh1b t) (Real.cosh u) :=
  val_congr (val_Whalf1b hV (val_Wadd1b hV (val_E hV ht) (val_E hV (val_Wminus1b hV ht))))
    (by rw [Real.cosh_eq])

/-- `w_tanh(x) := sinh(x) / cosh(x)`. -/
def Wtanh1b (t : Term n) : Term n := Wdiv1b (Wsinh1b t) (Wcosh1b t)

/-- **Lemma B.4.** `w_tanh(x) = tanh x` for every real `x`. -/
theorem val_Wtanh1b (hV : StrictVal V) (ht : V t u) : V (Wtanh1b t) (Real.tanh u) :=
  val_congr (val_Wdiv1b hV (val_Wsinh1b hV ht) (val_Wcosh1b hV ht) (Real.cosh_pos u).ne')
    (by rw [Real.tanh_eq_sinh_div_cosh])

/-- `Q(t) := E(half(L(t)))`, the positive root, for `t > 0`. -/
def Q1b (t : Term n) : Term n := E (Whalf1b (L t))

/-- **Lemma B.4.** `Q(t) = √t` for real `t > 0`. -/
theorem val_Q1b (hV : StrictVal V) (ht : V t u) (hu : 0 < u) : V (Q1b t) (Real.sqrt u) :=
  val_congr (val_E hV (val_Whalf1b hV (val_L hV ht hu)))
    (by rw [Real.exp_half, Real.exp_log hu])

/-- `w_arsinh(x) := L(x + Q(1 + x²))`. -/
def Warsinh1b (t : Term n) : Term n := L (Wadd1b t (Q1b (Wadd1b one (Wsqr1b t))))

/-- **Lemma B.4.** `w_arsinh(x) = arsinh x` for every real `x`, including `x < 0`. -/
theorem val_Warsinh1b (hV : StrictVal V) (ht : V t u) :
    V (Warsinh1b t) (Real.arsinh u) := by
  have h1 := val_Wadd1b hV ht
    (val_Q1b hV (val_Wadd1b hV hV.one (val_Wsqr1b hV ht)) (by positivity))
  have hpos : 0 < u + Real.sqrt (1 + u ^ 2) := by
    have h := Real.exp_pos (Real.arsinh u)
    rwa [Real.exp_arsinh] at h
  exact val_congr (val_L hV h1 hpos) (by rw [Real.arsinh])

/-- `w_artanh(x) := half(L(1 + x) - L(1 - x))`. -/
def Wartanh1b (t : Term n) : Term n :=
  Whalf1b (Wsub1b (L (Wadd1b one t)) (L (Wsub1b one t)))

/-- **Lemma B.4.** `w_artanh(x) = artanh x` for real `-1 < x < 1`. -/
theorem val_Wartanh1b (hV : StrictVal V) (ht : V t u) (h₁ : -1 < u) (h₂ : u < 1) :
    V (Wartanh1b t) (Real.artanh u) := by
  have hp : 0 < 1 + u := by linarith
  have hm : 0 < 1 - u := by linarith
  have h := val_Whalf1b hV (val_Wsub1b hV (val_L hV (val_Wadd1b hV hV.one ht) hp)
    (val_L hV (val_Wsub1b hV hV.one ht) hm))
  exact val_congr h (by
    rw [Real.artanh_eq_half_log ⟨h₁.le, h₂.le⟩, Real.log_div hp.ne' hm.ne']
    ring)

/-! ## Lemma B.7, first half: witnesses on the restricted domains -/

/-- The square root on `x > 0`: `Q(x)`. -/
def Wsqrt1b (t : Term n) : Term n := Q1b t

/-- `Q(x) = √x` for real `x > 0`. -/
theorem val_Wsqrt1b (hV : StrictVal V) (ht : V t u) (hu : 0 < u) :
    V (Wsqrt1b t) (Real.sqrt u) :=
  val_Q1b hV ht hu

/-- The hypotenuse off the origin: `Q(x² + y²)`. -/
def Whypot1b (s t : Term n) : Term n := Q1b (Wadd1b (Wsqr1b s) (Wsqr1b t))

/-- `x² + y² > 0` off the origin. -/
theorem sq_add_sq_pos (h : u ≠ 0 ∨ v ≠ 0) : 0 < u ^ 2 + v ^ 2 := by
  rcases h with h | h
  · have := pow_pos (abs_pos.mpr h) 2
    rw [sq_abs] at this
    nlinarith [sq_nonneg v]
  · have := pow_pos (abs_pos.mpr h) 2
    rw [sq_abs] at this
    nlinarith [sq_nonneg u]

/-- `Q(x² + y²) = √(x² + y²)` for real `(x, y) ≠ (0, 0)`. -/
theorem val_Whypot1b (hV : StrictVal V) (hs : V s u) (ht : V t v) (h : u ≠ 0 ∨ v ≠ 0) :
    V (Whypot1b s t) (Real.sqrt (u ^ 2 + v ^ 2)) :=
  val_Q1b hV (val_Wadd1b hV (val_Wsqr1b hV hs) (val_Wsqr1b hV ht)) (sq_add_sq_pos h)

/-- `arcosh` on `x > 1`: `L(x + Q(x² - 1))`. -/
def Warcosh1b (t : Term n) : Term n := L (Wadd1b t (Q1b (Wsub1b (Wsqr1b t) one)))

/-- **Lemma B.7.** `L(x + Q(x² - 1)) = arcosh x` for real `x > 1`. -/
theorem val_Warcosh1b (hV : StrictVal V) (ht : V t u) (hu : 1 < u) :
    V (Warcosh1b t) (Real.arcosh u) := by
  have h0 : 0 < u ^ 2 - 1 := by nlinarith
  have h1 := val_Wadd1b hV ht (val_Q1b hV (val_Wsub1b hV (val_Wsqr1b hV ht) hV.one) h0)
  have hpos : 0 < u + Real.sqrt (u ^ 2 - 1) := by
    have := Real.sqrt_nonneg (u ^ 2 - 1)
    linarith
  exact val_congr (val_L hV h1 hpos) (by rw [Real.arcosh])

/-! ## Every witness uses each of its operands

In strict real arithmetic a term is undefined wherever one of its subterms is undefined. The
lemmas below record this for the arithmetic witnesses, and that the witnesses of `ln` and of the
quotient are undefined outside the natural domains. They are used in `EML.Real.Class`
(Corollary B.5), where the domains of the compiled terms must be exactly the domains of the
expressions. -/

theorem evalR_E_undef (ht : evalR t a = none) : evalR (E t) a = none :=
  evalR_eml_undef_left ht

theorem evalR_L_undef (ht : evalR t a = none) : evalR (L t) a = none :=
  evalR_eml_undef_right (evalR_E_undef (evalR_eml_undef_right ht))

/-- `L(u)` is undefined for `u ≤ 0`. -/
theorem evalR_L_nonpos (ht : evalR t a = some u) (hu : u ≤ 0) : evalR (L t) a = none :=
  evalR_eml_undef_right (evalR_E_undef (evalR_eml_nonpos ht hu))

theorem evalR_F1b_undef_left (hs : evalR s a = none) : evalR (F1b s t) a = none :=
  evalR_eml_undef_left hs

theorem evalR_F1b_undef_right (ht : evalR t a = none) : evalR (F1b s t) a = none :=
  evalR_eml_undef_right (evalR_E_undef ht)

theorem evalR_B1b_undef (ht : evalR t a = none) : evalR (B1b t) a = none :=
  evalR_L_undef (evalR_F1b_undef_left ht)

theorem evalR_Wadd1b_undef_left (hs : evalR s a = none) : evalR (Wadd1b s t) a = none :=
  evalR_F1b_undef_left hs

theorem evalR_Wadd1b_undef_right (ht : evalR t a = none) : evalR (Wadd1b s t) a = none :=
  evalR_F1b_undef_right (evalR_F1b_undef_right ht)

theorem evalR_Wsub1b_undef_left (hs : evalR s a = none) : evalR (Wsub1b s t) a = none :=
  evalR_F1b_undef_right (evalR_F1b_undef_right hs)

theorem evalR_Wsub1b_undef_right (ht : evalR t a = none) : evalR (Wsub1b s t) a = none :=
  evalR_F1b_undef_left (evalR_B1b_undef ht)

theorem evalR_G1b_undef_left (hs : evalR s a = none) : evalR (G1b s t) a = none :=
  evalR_Wsub1b_undef_left (evalR_E_undef (evalR_Wadd1b_undef_left hs))

theorem evalR_G1b_undef_right (ht : evalR t a = none) : evalR (G1b s t) a = none :=
  evalR_Wsub1b_undef_left (evalR_E_undef (evalR_Wadd1b_undef_right ht))

theorem evalR_Wmul1b_undef_left (hs : evalR s a = none) : evalR (Wmul1b s t) a = none :=
  evalR_Wsub1b_undef_left (evalR_G1b_undef_left hs)

theorem evalR_Wmul1b_undef_right (ht : evalR t a = none) : evalR (Wmul1b s t) a = none :=
  evalR_Wsub1b_undef_left (evalR_G1b_undef_right ht)

theorem evalR_Winv1b_undef (ht : evalR t a = none) : evalR (Winv1b t) a = none :=
  evalR_Wmul1b_undef_left ht

/-- `w_inv(0)` is undefined: the witness meets `L(0)`. -/
theorem evalR_Winv1b_zero (ht : evalR t a = some 0) : evalR (Winv1b t) a = none := by
  have h0 : evalR (Wsqr1b t) a = some 0 :=
    val_congr (val_Wsqr1b (strictVal_evalR a) ht) (by norm_num)
  exact evalR_Wmul1b_undef_right
    (evalR_E_undef (evalR_Wsub1b_undef_right (evalR_L_nonpos h0 le_rfl)))

theorem evalR_Wdiv1b_undef_left (hs : evalR s a = none) : evalR (Wdiv1b s t) a = none :=
  evalR_Wmul1b_undef_left hs

theorem evalR_Wdiv1b_undef_right (ht : evalR t a = none) : evalR (Wdiv1b s t) a = none :=
  evalR_Wmul1b_undef_right (evalR_Winv1b_undef ht)

/-- Division by zero is undefined: the witness meets `L(0)`. -/
theorem evalR_Wdiv1b_zero (ht : evalR t a = some 0) : evalR (Wdiv1b s t) a = none :=
  evalR_Wmul1b_undef_right (evalR_Winv1b_zero ht)

end EML
