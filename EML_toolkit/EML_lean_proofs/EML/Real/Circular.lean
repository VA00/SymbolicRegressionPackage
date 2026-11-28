module

public import EML.Real.Hardy
public import EML.Real.Reduction

/-!
# Theorem 1c (Appendix C): no extended real term for `sin`, `cos`, `tan`

Paragraph "Direct circular functions" of Appendix C.

* `evalE_eventually` — for every extended real term in one variable there is `x₀` such that on
  `(x₀, ∞)` the term is undefined everywhere, or `+∞` everywhere, or `-∞` everywhere, or finite
  everywhere and equal to a strict real term. Induction on the term with Table C.1; the sign of
  the right child of a node for large `x` is given by `evalR_eventually`.
* `no_extended_term_of_signChanges` — hence a function that changes sign beyond every `x₀` is
  not given by an extended real term.
* `no_sin1c`, `no_cos1c`, `no_tan1c` — no extended real term is equal to `sin` or to `cos` on
  `ℝ`, or to `tan` on its natural domain.

This is the second argument of that paragraph (Hardy's theorem). As in `EML.Real.Hardy`,
Wilkie's theorem is not used.
-/

@[expose] public section

namespace EML

open Term Filter Topology

/-- For every extended real term in one variable there is `x₀` such that on `(x₀, ∞)` the term
is undefined everywhere, or `+∞` everywhere, or `-∞` everywhere, or finite everywhere and equal
to a strict real term `t'`. -/
theorem evalE_eventually (t : Term 1) :
    (∀ᶠ x in atTop, evalE t ![x] = none) ∨ (∀ᶠ x in atTop, evalE t ![x] = some ⊤) ∨
      (∀ᶠ x in atTop, evalE t ![x] = some ⊥) ∨
      ∃ (t' : Term 1) (f : ℝ → ℝ), ∀ᶠ x in atTop,
        evalE t ![x] = some (f x : EReal) ∧ evalR t' ![x] = some (f x) := by
  induction t with
  | one => exact Or.inr (Or.inr (Or.inr ⟨one, fun _ => 1, .of_forall fun _ => ⟨rfl, rfl⟩⟩))
  | var j =>
    refine Or.inr (Or.inr (Or.inr ⟨var j, fun x => x, .of_forall fun x => ?_⟩))
    fin_cases j
    exact ⟨rfl, rfl⟩
  | eml s t ihs iht =>
    rcases ihs with hs | hs
    · exact Or.inl (hs.mono fun x hx => evalE_eml_undef_left hx)
    rcases iht with ht | ht | ht | ⟨t', g, ht⟩
    · exact Or.inl (ht.mono fun x hx => evalE_eml_undef_right hx)
    · -- the right child is +∞ for all large x
      rcases hs with hs | hs | ⟨s', f, hs⟩
      · left
        filter_upwards [hs, ht] with x hx₁ hx₂
        rw [evalE_eml hx₁ hx₂, emlE_top_top]
      · right; right; left
        filter_upwards [hs, ht] with x hx₁ hx₂
        rw [evalE_eml hx₁ hx₂, emlE_bot_top]
      · right; right; left
        filter_upwards [hs, ht] with x hx₁ hx₂
        rw [evalE_eml hx₁.1 hx₂, emlE_coe_top]
    · -- the right child is -∞ for all large x: the node is undefined
      left
      rcases hs with hs | hs | ⟨s', f, hs⟩
      · filter_upwards [hs, ht] with x hx₁ hx₂
        rw [evalE_eml hx₁ hx₂, emlE_right_bot]
      · filter_upwards [hs, ht] with x hx₁ hx₂
        rw [evalE_eml hx₁ hx₂, emlE_right_bot]
      · filter_upwards [hs, ht] with x hx₁ hx₂
        rw [evalE_eml hx₁.1 hx₂, emlE_right_bot]
    · -- the right child is finite and equal to the strict term t': it has a sign for large x
      have hsign : (∀ᶠ x in atTop, g x = 0) ∨ (∀ᶠ x in atTop, 0 < g x) ∨
          (∀ᶠ x in atTop, g x < 0) := by
        rcases evalR_eventually t' with h | ⟨g', hg', h⟩
        · obtain ⟨x, hx₁, hx₂⟩ := (h.and ht).exists
          rw [hx₁] at hx₂
          cases hx₂.2
        · have e : ∀ᶠ x in atTop, g x = g' x := by
            filter_upwards [ht, hg'] with x hx₁ hx₂
            exact Option.some.inj (hx₁.2.symm.trans hx₂)
          rcases h with h | h | h
          · left
            filter_upwards [e, h] with x hx₁ hx₂
            rw [hx₁, hx₂]
          · right; left
            filter_upwards [e, h] with x hx₁ hx₂
            rw [hx₁]
            exact hx₂
          · right; right
            filter_upwards [e, h] with x hx₁ hx₂
            rw [hx₁]
            exact hx₂
      rcases hsign with h0 | hpos | hneg
      · -- right child 0: the node is +∞
        right; left
        rcases hs with hs | hs | ⟨s', f, hs⟩
        · filter_upwards [hs, ht, h0] with x hx₁ hx₂ hx₃
          rw [evalE_eml hx₁ hx₂.1, hx₃, emlE_right_zero]
        · filter_upwards [hs, ht, h0] with x hx₁ hx₂ hx₃
          rw [evalE_eml hx₁ hx₂.1, hx₃, emlE_right_zero]
        · filter_upwards [hs, ht, h0] with x hx₁ hx₂ hx₃
          rw [evalE_eml hx₁.1 hx₂.1, hx₃, emlE_right_zero]
      · -- right child positive
        rcases hs with hs | hs | ⟨s', f, hs⟩
        · -- left child +∞: the node is +∞
          right; left
          filter_upwards [hs, ht, hpos] with x hx₁ hx₂ hx₃
          rw [evalE_eml hx₁ hx₂.1, emlE_top_coe hx₃]
        · -- left child -∞: the node is -ln of the right child, the strict term -L(t')
          right; right; right
          refine ⟨Wminus1b (L t'), fun x => -Real.log (g x), ?_⟩
          filter_upwards [hs, ht, hpos] with x hx₁ hx₂ hx₃
          exact ⟨by rw [evalE_eml hx₁ hx₂.1, emlE_bot_coe hx₃],
            val_Wminus1b (strictVal_evalR _) (val_L (strictVal_evalR _) hx₂.2 hx₃)⟩
        · -- left child finite: the node is the strict node eml(s', t')
          right; right; right
          refine ⟨eml s' t', fun x => Real.exp (f x) - Real.log (g x), ?_⟩
          filter_upwards [hs, ht, hpos] with x hx₁ hx₂ hx₃
          exact ⟨by rw [evalE_eml hx₁.1 hx₂.1, emlE_coe_coe hx₃], evalR_eml hx₁.2 hx₂.2 hx₃⟩
      · -- right child negative: the node is undefined
        left
        rcases hs with hs | hs | ⟨s', f, hs⟩
        · filter_upwards [hs, ht, hneg] with x hx₁ hx₂ hx₃
          rw [evalE_eml hx₁ hx₂.1, emlE_right_coe_neg hx₃]
        · filter_upwards [hs, ht, hneg] with x hx₁ hx₂ hx₃
          rw [evalE_eml hx₁ hx₂.1, emlE_right_coe_neg hx₃]
        · filter_upwards [hs, ht, hneg] with x hx₁ hx₂ hx₃
          rw [evalE_eml hx₁.1 hx₂.1, emlE_right_coe_neg hx₃]

/-- A function that changes sign on `D` beyond every `x₀` is not given on `D` by an extended
real term: for large `x` such a term would be finite, hence equal to a strict real term. -/
theorem no_extended_term_of_signChanges {φ : ℝ → ℝ} {D : Set ℝ} (h : SignChanges φ D) :
    ¬ ∃ W : Term 1, ∀ x ∈ D, evalE W ![x] = some ((φ x : ℝ) : EReal) := by
  rintro ⟨W, hW⟩
  rcases evalE_eventually W with h' | h' | h' | ⟨t', f, h'⟩
  · obtain ⟨x₀, h₀⟩ := eventually_atTop.mp h'
    obtain ⟨x, hxD, hx, -⟩ := h.1 x₀
    have e := hW x hxD
    rw [h₀ x hx] at e
    cases e
  · obtain ⟨x₀, h₀⟩ := eventually_atTop.mp h'
    obtain ⟨x, hxD, hx, -⟩ := h.1 x₀
    have e := hW x hxD
    rw [h₀ x hx, Option.some.injEq] at e
    exact EReal.coe_ne_top _ e.symm
  · obtain ⟨x₀, h₀⟩ := eventually_atTop.mp h'
    obtain ⟨x, hxD, hx, -⟩ := h.1 x₀
    have e := hW x hxD
    rw [h₀ x hx, Option.some.injEq] at e
    exact EReal.coe_ne_bot _ e.symm
  · obtain ⟨x₀, h₀⟩ := eventually_atTop.mp h'
    -- on `D ∩ [x₀, ∞)` the strict term `t'` is equal to `φ`
    refine no_term_of_signChanges (φ := φ) (D := D ∩ Set.Ici x₀) ⟨fun x₁ => ?_, fun x₁ => ?_⟩
      ⟨t', fun x hx => ?_⟩
    · obtain ⟨x, hxD, hx, hφ⟩ := h.1 (max x₀ x₁)
      exact ⟨x, ⟨hxD, (le_max_left _ _).trans hx⟩, (le_max_right _ _).trans hx, hφ⟩
    · obtain ⟨x, hxD, hx, hφ⟩ := h.2 (max x₀ x₁)
      exact ⟨x, ⟨hxD, (le_max_left _ _).trans hx⟩, (le_max_right _ _).trans hx, hφ⟩
    · have e := hW x hx.1
      rw [(h₀ x hx.2).1, Option.some.injEq, EReal.coe_eq_coe_iff] at e
      rw [(h₀ x hx.2).2, e]

/-- **Theorem 1c**, `sin`. No extended real term is equal to `sin x` for every real `x`. -/
theorem no_sin1c : ¬ ∃ W : Term 1, ∀ x : ℝ, evalE W ![x] = some ((Real.sin x : ℝ) : EReal) :=
  fun ⟨W, hW⟩ => no_extended_term_of_signChanges signChanges_sin ⟨W, fun x _ => hW x⟩

/-- **Theorem 1c**, `cos`. No extended real term is equal to `cos x` for every real `x`. -/
theorem no_cos1c : ¬ ∃ W : Term 1, ∀ x : ℝ, evalE W ![x] = some ((Real.cos x : ℝ) : EReal) :=
  fun ⟨W, hW⟩ => no_extended_term_of_signChanges signChanges_cos ⟨W, fun x _ => hW x⟩

/-- **Theorem 1c**, `tan`. No extended real term is equal to `tan x` at every real `x` with
`cos x ≠ 0`. -/
theorem no_tan1c :
    ¬ ∃ W : Term 1, ∀ x : ℝ, Real.cos x ≠ 0 →
      evalE W ![x] = some ((Real.tan x : ℝ) : EReal) :=
  fun ⟨W, hW⟩ => no_extended_term_of_signChanges signChanges_tan ⟨W, fun x hx => hW x hx⟩

end EML
