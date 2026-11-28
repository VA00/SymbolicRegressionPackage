module

public import EML.Real.Reduction

/-!
# Theorem 1c (Appendix C): every extended real term is continuous on its domain

Paragraph "Absolute value, ReLU and the unit step" of Appendix C.

* `emlExpr`             — `exp p - ln q` written with Mathlib's extended exponential
  `EReal.exp : EReal → ℝ≥0∞` and extended logarithm `ENNReal.log : ℝ≥0∞ → EReal`.
* `emlE_eq_some_emlExpr` — wherever Table C.1 gives a value, the value is `emlExpr`.
* `continuousAt_emlExpr` — extended `exp`, `ln` and subtraction are continuous at each allowed
  pair, for the order topology of `ℝ ∪ {-∞, +∞}`.
* `continuousOn_valE`   — hence every term is continuous on its domain.
* `no_unit_step1c`      — consequently no extended real term is the unit step on all of `ℝ`.
* `evalE_Wabs1c`, `evalE_Wrelu1c`, `evalE_Wmax1c` — `|x| = √(x²)`, `ReLU(x) = (x + |x|)/2` and
  `max(x, y) = (x + y + |x - y|)/2` are extended real terms on all of `ℝ` and `ℝ²`.
-/

@[expose] public section

namespace EML

open Term Filter Topology

variable {n : ℕ} {a : Fin n → ℝ} {s t : Term n} {u v : ℝ} {p q : EReal}

/-- `exp p - ln q` with Mathlib's extended exponential and logarithm. (For `q < 0` the function
`EReal.toENNReal` returns `0`; this plays no role, because Table C.1 gives no value there.) -/
noncomputable def emlExpr (pq : EReal × EReal) : EReal :=
  (EReal.exp pq.1 : EReal) - ENNReal.log pq.2.toENNReal

/-- Wherever Table C.1 gives a value, the value is `exp p - ln q`. -/
theorem emlE_eq_some_emlExpr (h : (emlE p q).isSome) : emlE p q = some (emlExpr (p, q)) := by
  induction q with
  | bot => rw [emlE_right_bot] at h; cases h
  | top =>
    induction p with
    | bot => rw [emlE_bot_top]; simp [emlExpr]
    | coe u => rw [emlE_coe_top]; simp [emlExpr]
    | top => rw [emlE_top_top] at h; cases h
  | coe v =>
    rcases lt_trichotomy v 0 with hv | rfl | hv
    · rw [emlE_right_coe_neg hv] at h; cases h
    · rw [emlE_right_zero]
      simp [emlExpr, EReal.sub_bot (EReal.coe_ennreal_ne_bot _)]
    · have hlog : ENNReal.log (v : EReal).toENNReal = ((Real.log v : ℝ) : EReal) := by
        rw [EReal.toENNReal_of_ne_top (EReal.coe_ne_top v), EReal.toReal_coe,
          ENNReal.log_ofReal_of_pos hv]
      induction p with
      | bot =>
        rw [emlE_bot_coe hv]
        change _ = some ((EReal.exp ⊥ : EReal) - ENNReal.log (v : EReal).toENNReal)
        rw [hlog]
        simp
      | coe u =>
        rw [emlE_coe_coe hv]
        change _ = some ((EReal.exp (u : EReal) : EReal) - ENNReal.log (v : EReal).toENNReal)
        rw [hlog]
        simp [EReal.coe_ennreal_ofReal, max_eq_left (Real.exp_pos u).le]
      | top =>
        rw [emlE_top_coe hv]
        change _ = some ((EReal.exp ⊤ : EReal) - ENNReal.log (v : EReal).toENNReal)
        rw [hlog]
        simp

/-- Extended `exp`, `ln` and subtraction are continuous at each pair at which Table C.1 gives a
value: the only pair excluded by the subtraction is `(+∞, +∞)`. -/
theorem continuousAt_emlExpr (h : (emlE p q).isSome) : ContinuousAt emlExpr (p, q) := by
  have hcond : p ≠ ⊤ ∨ q ≠ ⊤ := by
    by_contra hc
    push Not at hc
    rw [hc.1, hc.2, emlE_top_top] at h
    cases h
  have h1 : ContinuousAt
      (fun pq : EReal × EReal => ((EReal.exp pq.1 : EReal), -ENNReal.log pq.2.toENNReal))
      (p, q) :=
    ((continuous_coe_ennreal_ereal.comp (ENNReal.continuous_exp.comp continuous_fst)).prodMk
      (continuous_neg.comp (ENNReal.continuous_log.comp
        (EReal.continuous_toENNReal.comp continuous_snd)))).continuousAt
  have h2 : ContinuousAt (fun z : EReal × EReal => z.1 + z.2)
      ((EReal.exp p : EReal), -ENNReal.log q.toENNReal) := by
    refine EReal.continuousAt_add ?_ (Or.inl (EReal.coe_ennreal_ne_bot _))
    rcases hcond with hp | hq
    · exact Or.inl (by simpa using hp)
    · refine Or.inr ?_
      rw [Ne, EReal.neg_eq_bot_iff]
      simpa using hq
  exact h2.comp (f := fun pq : EReal × EReal =>
    ((EReal.exp pq.1 : EReal), -ENNReal.log pq.2.toENNReal)) h1

/-- The value of a term as an extended real number, and `0` where the term is undefined. -/
noncomputable def valE (t : Term n) (a : Fin n → ℝ) : EReal := (evalE t a).getD 0

theorem valE_of_evalE {t : Term n} (h : evalE t a = some p) : valE t a = p := by
  simp [valE, h]

/-- On the domain of a node, its value is `exp - ln` of the values of its children. -/
theorem valE_eml (ha : a ∈ domE (eml s t)) :
    a ∈ domE s ∧ a ∈ domE t ∧ (emlE (valE s a) (valE t a)).isSome ∧
      valE (eml s t) a = emlExpr (valE s a, valE t a) := by
  obtain ⟨p, q, hs, ht, hpq⟩ := evalE_eml_isSome ha
  rw [valE_of_evalE hs, valE_of_evalE ht]
  refine ⟨by simp [domE, hs], by simp [domE, ht], hpq, ?_⟩
  exact valE_of_evalE (by rw [evalE_eml hs ht, emlE_eq_some_emlExpr hpq])

/-- **Continuity.** Every extended real term is continuous on its domain, as a map into
`ℝ ∪ {-∞, +∞}` with the order topology. -/
theorem continuousOn_valE (t : Term n) : ContinuousOn (valE t) (domE t) := by
  induction t with
  | one => exact continuousOn_const
  | var j => exact (continuous_coe_real_ereal.comp (continuous_apply j)).continuousOn
  | eml s t ihs iht =>
    intro a ha
    obtain ⟨has, hat, hpq, -⟩ := valE_eml ha
    have hf : ContinuousWithinAt (valE s) (domE (eml s t)) a :=
      (ihs a has).mono fun b hb => (valE_eml hb).1
    have hg : ContinuousWithinAt (valE t) (domE (eml s t)) a :=
      (iht a hat).mono fun b hb => (valE_eml hb).2.1
    have hpair : ContinuousWithinAt (fun b => (valE s b, valE t b)) (domE (eml s t)) a :=
      hf.prodMk hg
    have hΦ : ContinuousAt emlExpr ((fun b => (valE s b, valE t b)) a) :=
      continuousAt_emlExpr hpq
    have hcomp : ContinuousWithinAt (emlExpr ∘ fun b => (valE s b, valE t b))
        (domE (eml s t)) a :=
      ContinuousAt.comp_continuousWithinAt (f := fun b => (valE s b, valE t b)) (x := a) hΦ hpair
    exact hcomp.congr (fun b hb => (valE_eml hb).2.2.2) (valE_eml ha).2.2.2

/-- The curve `x ↦ (x)` in `ℝ¹` is continuous. -/
theorem continuous_vec1 : Continuous (fun x : ℝ => (![x] : Fin 1 → ℝ)) :=
  continuous_pi fun i => by fin_cases i; exact continuous_id

/-- **No unit step.** No extended real term is equal to the unit step (`0` for `x < 0`, `1` for
`x ≥ 0`) on all of `ℝ`: a term is continuous on its domain. -/
theorem no_unit_step1c :
    ¬ ∃ W : Term 1, ∀ x : ℝ,
      evalE W ![x] = some (((if 0 ≤ x then (1 : ℝ) else 0 : ℝ)) : EReal) := by
  rintro ⟨W, hW⟩
  have hdom : ∀ x : ℝ, (![x] : Fin 1 → ℝ) ∈ domE W := fun x => by simp [domE, hW x]
  have hcont : Continuous (fun x : ℝ => valE W ![x]) :=
    (continuousOn_valE W).comp_continuous continuous_vec1 hdom
  have hval : (fun x : ℝ => valE W ![x])
      = fun x : ℝ => (((if 0 ≤ x then (1 : ℝ) else 0 : ℝ)) : EReal) :=
    funext fun x => valE_of_evalE (hW x)
  rw [hval] at hcont
  have hstep : Continuous (fun x : ℝ => (if 0 ≤ x then (1 : ℝ) else 0)) :=
    EReal.isEmbedding_coe.continuous_iff.mpr hcont
  have hl1 : Tendsto (fun x : ℝ => (if 0 ≤ x then (1 : ℝ) else 0)) (𝓝[<] 0) (𝓝 1) := by
    have := (hstep.tendsto 0).mono_left (nhdsWithin_le_nhds (s := Set.Iio 0))
    simpa using this
  have hl0 : Tendsto (fun x : ℝ => (if 0 ≤ x then (1 : ℝ) else 0)) (𝓝[<] 0) (𝓝 0) := by
    refine tendsto_const_nhds.congr' ?_
    filter_upwards [self_mem_nhdsWithin] with x hx
    simp [not_le.mpr hx]
  exact zero_ne_one (tendsto_nhds_unique hl0 hl1)

/-! ## Absolute value, ReLU, maximum -/

/-- `|x| := w_√(x²)`. -/
def Wabs1c (t : Term n) : Term n := Wsqrt1c (Wsqr1b t)

/-- `|x| = √(x²)` for every real `x`. -/
theorem evalE_Wabs1c (ht : evalE t a = some (u : EReal)) :
    evalE (Wabs1c t) a = some ((|u| : ℝ) : EReal) := by
  have h := evalE_Wsqrt1c (val_Wsqr1b (strictVal_evalE a) ht) (sq_nonneg u)
  rwa [Real.sqrt_sq_eq_abs] at h

/-- `ReLU(x) := (x + |x|)/2`. -/
def Wrelu1c (t : Term n) : Term n := Whalf1b (Wadd1b t (Wabs1c t))

/-- `ReLU(x) = max(x, 0)` for every real `x`. -/
theorem evalE_Wrelu1c (ht : evalE t a = some (u : EReal)) :
    evalE (Wrelu1c t) a = some ((max u 0 : ℝ) : EReal) := by
  have hV := strictVal_evalE a
  have h : ValE a (Wrelu1c t) ((u + |u|) / 2) :=
    val_Whalf1b hV (val_Wadd1b hV ht (evalE_Wabs1c ht))
  refine val_congr h ?_
  rcases le_total 0 u with h0 | h0
  · rw [abs_of_nonneg h0, max_eq_left h0]; ring
  · rw [abs_of_nonpos h0, max_eq_right h0]; ring

/-- `max(x, y) := (x + y + |x - y|)/2`. -/
def Wmax1c (s t : Term n) : Term n :=
  Whalf1b (Wadd1b (Wadd1b s t) (Wabs1c (Wsub1b s t)))

/-- `(x + y + |x - y|)/2 = max(x, y)` for all real `x`, `y`. -/
theorem evalE_Wmax1c (hs : evalE s a = some (u : EReal)) (ht : evalE t a = some (v : EReal)) :
    evalE (Wmax1c s t) a = some ((max u v : ℝ) : EReal) := by
  have hV := strictVal_evalE a
  have h : ValE a (Wmax1c s t) ((u + v + |u - v|) / 2) :=
    val_Whalf1b hV (val_Wadd1b hV (val_Wadd1b hV hs ht) (evalE_Wabs1c (val_Wsub1b hV hs ht)))
  refine val_congr h ?_
  rcases le_total v u with h0 | h0
  · rw [abs_of_nonneg (sub_nonneg.mpr h0), max_eq_left h0]; ring
  · rw [abs_of_nonpos (sub_nonpos.mpr h0), max_eq_right h0]; ring

end EML
