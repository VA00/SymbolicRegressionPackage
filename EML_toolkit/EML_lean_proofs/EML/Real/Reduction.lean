module

public import EML.Real.Extended
public import EML.Real.Analytic

/-!
# Theorem 1c (Appendix C): from extended real terms back to strict real terms

* `emlE_eq_coe_iff`, `evalE_eml_eq_coe_iff` — Table C.1 read backwards: a node has a finite
  value only if its right child is finite and positive and its left child is finite or `-∞`.
* `exists_evalR_of_evalE_coe` — at a single point, a finite value of an extended real term is
  the value of a strict real term.
* `closed_finite_values` — paragraph "Finite constants": the closed terms have the same finite
  values in both semantics. In particular a closed term gives `π` in extended real arithmetic
  if and only if a closed term gives `π` in strict real arithmetic (`closed_pi_iff`).
* `exists_strict_on_subinterval` — **Lemma C.4**: an extended real term that is finite on an
  open interval agrees with a strict real term on an open subinterval.

The proof of Lemma C.4 is the one of the paper. It goes through the nodes from the leaves
upwards and shrinks the interval at each node; the only analytic fact used is that a strict
real term is continuous at every point of its domain (Lemma B.6). No o-minimality is needed.
-/

@[expose] public section

namespace EML

open Term Filter Topology

variable {n : ℕ} {a : Fin n → ℝ} {s t : Term n} {p q : EReal}

/-! ## Table C.1 read backwards -/

/-- A finite value occurs in Table C.1 only in the column `0 < v < +∞`, in the rows `u ∈ ℝ`
(value `exp u - ln v`) and `u = -∞` (value `-ln v`). -/
theorem emlE_eq_coe_iff {w : ℝ} :
    emlE p q = some (w : EReal) ↔
      ∃ v : ℝ, q = (v : EReal) ∧ 0 < v ∧
        ((∃ u : ℝ, p = (u : EReal) ∧ Real.exp u - Real.log v = w) ∨
          (p = ⊥ ∧ -Real.log v = w)) := by
  constructor
  · intro h
    induction q with
    | bot => rw [emlE_right_bot] at h; cases h
    | top =>
      induction p with
      | bot =>
        rw [emlE_bot_top] at h
        exact (EReal.coe_ne_bot w (Option.some.inj h).symm).elim
      | coe u =>
        rw [emlE_coe_top] at h
        exact (EReal.coe_ne_bot w (Option.some.inj h).symm).elim
      | top => rw [emlE_top_top] at h; cases h
    | coe v =>
      rcases lt_trichotomy v 0 with hv | rfl | hv
      · rw [emlE_right_coe_neg hv] at h; cases h
      · rw [emlE_right_zero] at h
        exact (EReal.coe_ne_top w (Option.some.inj h).symm).elim
      · refine ⟨v, rfl, hv, ?_⟩
        induction p with
        | bot =>
          rw [emlE_bot_coe hv] at h
          exact Or.inr ⟨rfl, by exact_mod_cast Option.some.inj h⟩
        | coe u =>
          rw [emlE_coe_coe hv] at h
          exact Or.inl ⟨u, rfl, by exact_mod_cast Option.some.inj h⟩
        | top =>
          rw [emlE_top_coe hv] at h
          exact (EReal.coe_ne_top w (Option.some.inj h).symm).elim
  · rintro ⟨v, rfl, hv, ⟨u, rfl, rfl⟩ | ⟨rfl, rfl⟩⟩
    · exact emlE_coe_coe hv
    · exact emlE_bot_coe hv

/-- A node that is defined has two defined children, and Table C.1 gives a value at them. -/
theorem evalE_eml_isSome (h : (evalE (eml s t) a).isSome) :
    ∃ p q, evalE s a = some p ∧ evalE t a = some q ∧ (emlE p q).isSome := by
  cases hs : evalE s a with
  | none => rw [evalE_eml_undef_left hs] at h; cases h
  | some p =>
    cases ht : evalE t a with
    | none => rw [evalE_eml_undef_right ht] at h; cases h
    | some q =>
      rw [evalE_eml hs ht] at h
      exact ⟨p, q, rfl, rfl, h⟩

/-- A node with a finite value has a finite positive right child, and a left child that is
finite or `-∞`. -/
theorem evalE_eml_eq_coe_iff {w : ℝ} :
    evalE (eml s t) a = some (w : EReal) ↔
      ∃ v : ℝ, evalE t a = some (v : EReal) ∧ 0 < v ∧
        ((∃ u : ℝ, evalE s a = some (u : EReal) ∧ Real.exp u - Real.log v = w) ∨
          (evalE s a = some ⊥ ∧ -Real.log v = w)) := by
  constructor
  · intro h
    obtain ⟨p, q, hs, ht, -⟩ := evalE_eml_isSome (by rw [h]; rfl)
    rw [evalE_eml hs ht] at h
    obtain ⟨v, rfl, hv, hcase⟩ := emlE_eq_coe_iff.mp h
    refine ⟨v, ht, hv, ?_⟩
    rcases hcase with ⟨u, rfl, hw⟩ | ⟨rfl, hw⟩
    · exact Or.inl ⟨u, hs, hw⟩
    · exact Or.inr ⟨hs, hw⟩
  · rintro ⟨v, ht, hv, ⟨u, hs, rfl⟩ | ⟨hs, rfl⟩⟩
    · rw [evalE_eml hs ht, emlE_coe_coe hv]
    · rw [evalE_eml hs ht, emlE_bot_coe hv]

/-! ## Finite constants -/

/-- At a single point, a finite value of an extended real term is the value of a strict real
term. (The strict term depends on the point: it is built along the evaluation.) A finite node
has a finite positive right child `v`; its value is `exp u - ln v` with a finite left child `u`,
or `-ln v` with the left child `-∞`. -/
theorem exists_evalR_of_evalE_coe {t : Term n} {w : ℝ} (h : evalE t a = some (w : EReal)) :
    ∃ t' : Term n, evalR t' a = some w := by
  induction t generalizing w with
  | one =>
    refine ⟨one, ?_⟩
    rw [evalE_one, Option.some.injEq, EReal.coe_eq_coe_iff] at h
    rw [evalR_one, h]
  | var j =>
    refine ⟨var j, ?_⟩
    rw [evalE_var, Option.some.injEq, EReal.coe_eq_coe_iff] at h
    rw [evalR_var, h]
  | eml s t ihs iht =>
    obtain ⟨v, ht, hv, hcase⟩ := evalE_eml_eq_coe_iff.mp h
    obtain ⟨t', ht'⟩ := iht ht
    rcases hcase with ⟨u, hs, rfl⟩ | ⟨-, rfl⟩
    · obtain ⟨s', hs'⟩ := ihs hs
      exact ⟨eml s' t', evalR_eml hs' ht' hv⟩
    · exact ⟨Wminus1b (L t'),
        val_Wminus1b (strictVal_evalR a) (val_L (strictVal_evalR a) ht' hv)⟩

/-- **Finite constants.** A real number is the value of a closed term in extended real
arithmetic if and only if it is the value of a closed term in strict real arithmetic. -/
theorem closed_finite_values (w : ℝ) :
    (∃ t : Term 0, ∀ a, evalE t a = some (w : EReal)) ↔
      (∃ t : Term 0, ∀ a, evalR t a = some w) := by
  constructor
  · rintro ⟨t, ht⟩
    obtain ⟨t', ht'⟩ := exists_evalR_of_evalE_coe (ht ![])
    exact ⟨t', fun a => by rw [Subsingleton.elim a ![]]; exact ht'⟩
  · rintro ⟨t, ht⟩
    exact ⟨t, fun a => evalE_of_evalR (ht a)⟩

/-- **Theorem 1c**, last sentence. A closed term gives `π` in extended real arithmetic if and
only if a closed term gives `π` in strict real arithmetic. -/
theorem closed_pi_iff :
    (∃ t : Term 0, ∀ a, evalE t a = some (Real.pi : EReal)) ↔
      (∃ t : Term 0, ∀ a, evalR t a = some Real.pi) :=
  closed_finite_values Real.pi

/-! ## Lemma C.4: reduction on an open interval -/

/-- A strict real term in one variable with a positive value at `x₀` is defined and positive
on an interval around `x₀` (continuity, Lemma B.6). -/
theorem evalR_pos_near {t : Term 1} {x₀ w₀ : ℝ} (h : evalR t ![x₀] = some w₀) (hw : 0 < w₀) :
    ∃ δ > 0, ∀ x : ℝ, |x - x₀| < δ → ∃ w : ℝ, evalR t ![x] = some w ∧ 0 < w := by
  have h1 : ∀ᶠ x in 𝓝 x₀, ∃ w : ℝ, evalR t ![x] = some w ∧ 0 < w :=
    (differentiableAt_vec1 x₀).continuousAt.eventually (evalR_eventually_pos h hw)
  obtain ⟨δ, hδ, hball⟩ := Metric.eventually_nhds_iff.mp h1
  exact ⟨δ, hδ, fun x hx => hball (by rwa [Real.dist_eq])⟩

/-- The three kinds of behaviour of a one-variable term on the interval `(α, β)`: identically
`+∞`, identically `-∞`, or finite and equal to a strict real term. -/
def KindOn (t : Term 1) (α β : ℝ) : Prop :=
  (∀ x ∈ Set.Ioo α β, evalE t ![x] = some ⊤) ∨
  (∀ x ∈ Set.Ioo α β, evalE t ![x] = some ⊥) ∨
  (∃ t' : Term 1, ∀ x ∈ Set.Ioo α β,
    ∃ w : ℝ, evalE t ![x] = some (w : EReal) ∧ evalR t' ![x] = some w)

/-- A kind of behaviour on an interval persists on every subinterval. -/
theorem KindOn.mono {t : Term 1} {α β α' β' : ℝ} (h : KindOn t α β) (hα : α ≤ α')
    (hβ : β' ≤ β) : KindOn t α' β' := by
  have hsub : ∀ x ∈ Set.Ioo α' β', x ∈ Set.Ioo α β := fun x hx =>
    ⟨lt_of_le_of_lt hα hx.1, lt_of_lt_of_le hx.2 hβ⟩
  rcases h with h | h | ⟨t', h⟩
  · exact Or.inl fun x hx => h x (hsub x hx)
  · exact Or.inr (Or.inl fun x hx => h x (hsub x hx))
  · exact Or.inr (Or.inr ⟨t', fun x hx => h x (hsub x hx)⟩)

/-- If a one-variable term is defined at every point of an open interval, then on some open
subinterval it is identically `+∞`, or identically `-∞`, or finite and equal to a strict real
term. Proof by induction on the term; the interval is shrunk at each node. -/
theorem exists_kindOn (t : Term 1) {α β : ℝ} (hαβ : α < β)
    (hdef : ∀ x ∈ Set.Ioo α β, (evalE t ![x]).isSome) :
    ∃ α' β', α ≤ α' ∧ α' < β' ∧ β' ≤ β ∧ KindOn t α' β' := by
  induction t generalizing α β with
  | one =>
    exact ⟨α, β, le_rfl, hαβ, le_rfl, Or.inr (Or.inr ⟨one, fun x _ => ⟨1, rfl, rfl⟩⟩)⟩
  | var j =>
    exact ⟨α, β, le_rfl, hαβ, le_rfl, Or.inr (Or.inr ⟨var j, fun x _ => ⟨_, rfl, rfl⟩⟩)⟩
  | eml s t ihs iht =>
    -- both children are defined on (α, β); shrink the interval for the left child, then the right
    have hs : ∀ x ∈ Set.Ioo α β, (evalE s ![x]).isSome := fun x hx => by
      obtain ⟨p, q, hp, -, -⟩ := evalE_eml_isSome (hdef x hx)
      rw [hp]; rfl
    have ht : ∀ x ∈ Set.Ioo α β, (evalE t ![x]).isSome := fun x hx => by
      obtain ⟨p, q, -, hq, -⟩ := evalE_eml_isSome (hdef x hx)
      rw [hq]; rfl
    obtain ⟨α₁, β₁, h1a, h1, h1b, ks₁⟩ := ihs hαβ hs
    have sub1 : ∀ x ∈ Set.Ioo α₁ β₁, x ∈ Set.Ioo α β := fun x hx =>
      ⟨lt_of_le_of_lt h1a hx.1, lt_of_lt_of_le hx.2 h1b⟩
    obtain ⟨α₂, β₂, h2a, h2, h2b, kt⟩ := iht h1 (fun x hx => ht x (sub1 x hx))
    have sub2 : ∀ x ∈ Set.Ioo α₂ β₂, x ∈ Set.Ioo α β := fun x hx =>
      sub1 x ⟨lt_of_le_of_lt h2a hx.1, lt_of_lt_of_le hx.2 h2b⟩
    have ks : KindOn s α₂ β₂ := ks₁.mono h2a h2b
    have hα₂ : α ≤ α₂ := h1a.trans h2a
    have hβ₂ : β₂ ≤ β := h2b.trans h1b
    -- a point of (α₂, β₂), for the cases that cannot occur
    have hm : (α₂ + β₂) / 2 ∈ Set.Ioo α₂ β₂ := ⟨by linarith, by linarith⟩
    rcases kt with kt | kt | ⟨t', kt⟩
    · -- the right child is +∞ on the interval
      rcases ks with ks | ks | ⟨s', ks⟩
      · -- left child +∞: the node would be +∞ - (+∞), undefined
        have h := hdef _ (sub2 _ hm)
        rw [evalE_eml (ks _ hm) (kt _ hm), emlE_top_top] at h
        cases h
      · -- left child -∞: the node is -∞
        refine ⟨α₂, β₂, hα₂, h2, hβ₂, Or.inr (Or.inl fun x hx => ?_)⟩
        rw [evalE_eml (ks x hx) (kt x hx), emlE_bot_top]
      · -- left child finite: the node is -∞
        refine ⟨α₂, β₂, hα₂, h2, hβ₂, Or.inr (Or.inl fun x hx => ?_)⟩
        obtain ⟨u, hu, -⟩ := ks x hx
        rw [evalE_eml hu (kt x hx), emlE_coe_top]
    · -- the right child is -∞ on the interval: the node would be undefined
      have h := hdef _ (sub2 _ hm)
      obtain ⟨p, q, hp, hq, hpq⟩ := evalE_eml_isSome h
      rw [kt _ hm, Option.some.injEq] at hq
      rw [← hq, emlE_right_bot] at hpq
      cases hpq
    · -- the right child is finite on the interval, equal to the strict term t'
      by_cases hz : ∀ x ∈ Set.Ioo α₂ β₂, evalE t ![x] = some ((0 : ℝ) : EReal)
      · -- right child 0 on the whole interval: the node is +∞
        refine ⟨α₂, β₂, hα₂, h2, hβ₂, Or.inl fun x hx => ?_⟩
        obtain ⟨p, q, hp, -, -⟩ := evalE_eml_isSome (hdef x (sub2 x hx))
        rw [evalE_eml hp (hz x hx), emlE_right_zero]
      · -- the right child is not 0 at some point x₀, hence positive there and near x₀
        push Not at hz
        obtain ⟨x₀, hx₀, hne⟩ := hz
        obtain ⟨w₀, hw₀E, hw₀R⟩ := kt x₀ hx₀
        have hw₀pos : 0 < w₀ := by
          rcases lt_trichotomy w₀ 0 with hneg | rfl | hpos
          · obtain ⟨p, q, hp, hq, hpq⟩ := evalE_eml_isSome (hdef x₀ (sub2 x₀ hx₀))
            rw [hw₀E, Option.some.injEq] at hq
            rw [← hq, emlE_right_coe_neg hneg] at hpq
            cases hpq
          · exact (hne hw₀E).elim
          · exact hpos
        obtain ⟨δ, hδ, hnear⟩ := evalR_pos_near hw₀R hw₀pos
        have h3a : α₂ ≤ max α₂ (x₀ - δ) := le_max_left _ _
        have h3b : min β₂ (x₀ + δ) ≤ β₂ := min_le_left _ _
        have h3 : max α₂ (x₀ - δ) < min β₂ (x₀ + δ) := by
          rw [max_lt_iff, lt_min_iff, lt_min_iff]
          exact ⟨⟨h2, by linarith [hx₀.1]⟩, ⟨by linarith [hx₀.2], by linarith⟩⟩
        -- on the smaller interval the right child is finite and positive
        have kt' : ∀ x ∈ Set.Ioo (max α₂ (x₀ - δ)) (min β₂ (x₀ + δ)),
            ∃ w : ℝ, evalE t ![x] = some (w : EReal) ∧ evalR t' ![x] = some w ∧ 0 < w := by
          intro x hx
          have hx2 : x ∈ Set.Ioo α₂ β₂ :=
            ⟨lt_of_le_of_lt h3a hx.1, lt_of_lt_of_le hx.2 h3b⟩
          obtain ⟨w, hwE, hwR⟩ := kt x hx2
          have hxδ : |x - x₀| < δ := by
            rw [abs_lt]
            have ha := lt_of_le_of_lt (le_max_right α₂ (x₀ - δ)) hx.1
            have hb := lt_of_lt_of_le hx.2 (min_le_right β₂ (x₀ + δ))
            constructor <;> linarith
          obtain ⟨w', hw'R, hw'pos⟩ := hnear x hxδ
          rw [hwR, Option.some.injEq] at hw'R
          exact ⟨w, hwE, hwR, hw'R ▸ hw'pos⟩
        have ks' : KindOn s (max α₂ (x₀ - δ)) (min β₂ (x₀ + δ)) := ks.mono h3a h3b
        refine ⟨_, _, hα₂.trans h3a, h3, h3b.trans hβ₂, ?_⟩
        rcases ks' with ks' | ks' | ⟨s', ks'⟩
        · -- left child +∞: the node is +∞
          refine Or.inl fun x hx => ?_
          obtain ⟨w, hwE, -, hwpos⟩ := kt' x hx
          rw [evalE_eml (ks' x hx) hwE, emlE_top_coe hwpos]
        · -- left child -∞: the node is -ln of the right child, the strict term -L(t')
          refine Or.inr (Or.inr ⟨Wminus1b (L t'), fun x hx => ?_⟩)
          obtain ⟨w, hwE, hwR, hwpos⟩ := kt' x hx
          refine ⟨-Real.log w, ?_, ?_⟩
          · rw [evalE_eml (ks' x hx) hwE, emlE_bot_coe hwpos]
          · exact val_Wminus1b (strictVal_evalR _) (val_L (strictVal_evalR _) hwR hwpos)
        · -- left child finite: the node is the strict node eml(s', t')
          refine Or.inr (Or.inr ⟨eml s' t', fun x hx => ?_⟩)
          obtain ⟨w, hwE, hwR, hwpos⟩ := kt' x hx
          obtain ⟨u, huE, huR⟩ := ks' x hx
          refine ⟨Real.exp u - Real.log w, ?_, ?_⟩
          · rw [evalE_eml huE hwE, emlE_coe_coe hwpos]
          · exact evalR_eml huR hwR hwpos

/-- **Lemma C.4** (reduction on an open interval). If an extended real term in one variable is
finite throughout a nonempty open interval `(α, β)`, it agrees with a strict real term on some
nonempty open subinterval `(α', β')`. -/
theorem exists_strict_on_subinterval (t : Term 1) {α β : ℝ} (hαβ : α < β)
    (hfin : ∀ x ∈ Set.Ioo α β, ∃ w : ℝ, evalE t ![x] = some (w : EReal)) :
    ∃ (t' : Term 1) (α' β' : ℝ), α ≤ α' ∧ α' < β' ∧ β' ≤ β ∧
      ∀ x ∈ Set.Ioo α' β', ∃ w : ℝ, evalE t ![x] = some (w : EReal) ∧ evalR t' ![x] = some w := by
  have hdef : ∀ x ∈ Set.Ioo α β, (evalE t ![x]).isSome := fun x hx => by
    obtain ⟨w, hw⟩ := hfin x hx
    rw [hw]; rfl
  obtain ⟨α', β', ha, hlt, hb, k⟩ := exists_kindOn t hαβ hdef
  have hm : (α' + β') / 2 ∈ Set.Ioo α' β' := ⟨by linarith, by linarith⟩
  have hm0 : (α' + β') / 2 ∈ Set.Ioo α β :=
    ⟨lt_of_le_of_lt ha hm.1, lt_of_lt_of_le hm.2 hb⟩
  obtain ⟨w, hw⟩ := hfin _ hm0
  rcases k with k | k | ⟨t', k⟩
  · rw [k _ hm, Option.some.injEq] at hw
    exact (EReal.coe_ne_top w hw.symm).elim
  · rw [k _ hm, Option.some.injEq] at hw
    exact (EReal.coe_ne_bot w hw.symm).elim
  · exact ⟨t', α', β', ha, hlt, hb, k⟩

/-- The step from strict real to extended real terms in the paragraph "Inverse circular
functions": if no strict real term agrees with a function `f` on any open interval, then no
extended real term agrees with `f` on any open interval. -/
theorem no_extended_of_no_strict (f : ℝ → ℝ)
    (h : ∀ (t : Term 1) (α β : ℝ), α < β →
      ¬ ∀ x ∈ Set.Ioo α β, evalR t ![x] = some (f x)) :
    ∀ (t : Term 1) (α β : ℝ), α < β →
      ¬ ∀ x ∈ Set.Ioo α β, evalE t ![x] = some ((f x : ℝ) : EReal) := by
  intro t α β hαβ hT
  obtain ⟨t', α', β', ha, hlt, hb, hT'⟩ :=
    exists_strict_on_subinterval t hαβ (fun x hx => ⟨f x, hT x hx⟩)
  refine h t' α' β' hlt fun x hx => ?_
  obtain ⟨w, hwE, hwR⟩ := hT' x hx
  have hx0 : x ∈ Set.Ioo α β := ⟨lt_of_le_of_lt ha hx.1, lt_of_lt_of_le hx.2 hb⟩
  rw [hT x hx0, Option.some.injEq, EReal.coe_eq_coe_iff] at hwE
  rw [hwR, hwE]

end EML
