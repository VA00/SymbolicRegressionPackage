module

public import EML.Real.Strict

/-!
# Theorem 1b (Appendix B): local analyticity and the three endpoint obstructions

* `evalR_analyticAt` — **Lemma B.6**: near every point of its domain a strict real term agrees
  with a real analytic function. Hence its domain is open (`isOpen_domR`).
* `no_sqrt1b`, `no_hypot1b`, `no_arcosh1b` — no strict real term computes `√x` on `[0, ∞)`,
  `√(x² + y²)` on `ℝ²`, or `arcosh x` on `[1, ∞)`: the endpoint `0`, the origin and the
  endpoint `1` are the obstructions (Lemma B.7 and the paragraph after it).

Only the differentiability of a term at the endpoint is used in the three obstructions.
-/

@[expose] public section

namespace EML

open Term Filter Topology

variable {n : ℕ}

/-- **Lemma B.6** (local analyticity). If a strict real term is defined at the point `a`, there
is a function `f`, real analytic at `a`, such that the term is defined and equal to `f` at every
point near `a`. -/
theorem evalR_analyticAt {t : Term n} {a : Fin n → ℝ} {r : ℝ} (h : evalR t a = some r) :
    ∃ f : (Fin n → ℝ) → ℝ, AnalyticAt ℝ f a ∧ ∀ᶠ b in 𝓝 a, evalR t b = some (f b) := by
  induction t generalizing r with
  | one => exact ⟨fun _ => 1, analyticAt_const, Eventually.of_forall fun b => rfl⟩
  | var j =>
    exact ⟨fun b => b j,
      (ContinuousLinearMap.proj (R := ℝ) (φ := fun _ : Fin n => ℝ) j).analyticAt a,
      Eventually.of_forall fun b => rfl⟩
  | eml s t ihs iht =>
    obtain ⟨u, v, hs, ht, hv, -⟩ := evalR_eml_eq_some_iff.mp h
    obtain ⟨f, hf, hfe⟩ := ihs hs
    obtain ⟨g, hg, hge⟩ := iht ht
    have hga : g a = v := by
      have := hge.self_of_nhds
      rw [ht] at this
      exact (Option.some.inj this).symm
    have hgpos0 : 0 < g a := hga ▸ hv
    have hgpos : ∀ᶠ b in 𝓝 a, 0 < g b := hg.continuousAt.eventually (lt_mem_nhds hgpos0)
    have hlog : AnalyticAt ℝ (fun b => Real.log (g b)) a := (analyticAt_log hgpos0).comp hg
    refine ⟨fun b => Real.exp (f b) - Real.log (g b), hf.rexp'.sub hlog, ?_⟩
    filter_upwards [hfe, hge, hgpos] with b hb1 hb2 hb3
    exact evalR_eml hb1 hb2 hb3

/-- **Lemma B.6.** The domain of every strict real term is open. -/
theorem isOpen_domR (t : Term n) : IsOpen (domR t) := by
  rw [isOpen_iff_eventually]
  intro a ha
  obtain ⟨r, hr⟩ := Option.isSome_iff_exists.mp ha
  obtain ⟨f, -, hfe⟩ := evalR_analyticAt hr
  filter_upwards [hfe] with b hb
  simp [domR, hb]

/-- A strict real term is continuous at every point of its domain: if its value at `a` is
positive, it is defined and positive at all points near `a`. -/
theorem evalR_eventually_pos {t : Term n} {a : Fin n → ℝ} {r : ℝ} (h : evalR t a = some r)
    (hr : 0 < r) : ∀ᶠ b in 𝓝 a, ∃ r' : ℝ, evalR t b = some r' ∧ 0 < r' := by
  obtain ⟨f, hf, hfe⟩ := evalR_analyticAt h
  have hfa : f a = r := by
    have := hfe.self_of_nhds
    rw [h] at this
    exact (Option.some.inj this).symm
  have hpos : ∀ᶠ b in 𝓝 a, 0 < f b := hf.continuousAt.eventually (lt_mem_nhds (hfa ▸ hr))
  filter_upwards [hfe, hpos] with b hb1 hb2
  exact ⟨f b, hb1, hb2⟩

/-- A strict real term restricted to a differentiable curve `γ` through a point of its domain:
near that point it is defined and equal to a function that is differentiable there. -/
theorem evalR_comp_differentiableAt {t : Term n} {γ : ℝ → Fin n → ℝ} {x₀ r : ℝ}
    (hγ : DifferentiableAt ℝ γ x₀) (h : evalR t (γ x₀) = some r) :
    ∃ g : ℝ → ℝ, DifferentiableAt ℝ g x₀ ∧ g x₀ = r ∧
      ∀ᶠ x in 𝓝 x₀, evalR t (γ x) = some (g x) := by
  obtain ⟨f, hf, hfe⟩ := evalR_analyticAt h
  have hfa : f (γ x₀) = r := by
    have := hfe.self_of_nhds
    rw [h] at this
    exact (Option.some.inj this).symm
  exact ⟨fun x => f (γ x), hf.differentiableAt.comp x₀ hγ, hfa, hγ.continuousAt.eventually hfe⟩

/-- The curve `x ↦ (x)` in `ℝ¹`. -/
theorem differentiableAt_vec1 (x₀ : ℝ) :
    DifferentiableAt ℝ (fun x : ℝ => (![x] : Fin 1 → ℝ)) x₀ := by
  rw [differentiableAt_pi]
  intro i
  fin_cases i
  exact differentiableAt_id

/-- The curve `x ↦ (x, 0)` in `ℝ²`. -/
theorem differentiableAt_vec2 (x₀ : ℝ) :
    DifferentiableAt ℝ (fun x : ℝ => (![x, 0] : Fin 2 → ℝ)) x₀ := by
  rw [differentiableAt_pi]
  intro i
  fin_cases i
  · exact differentiableAt_id
  · exact differentiableAt_const _

/-- **Square root.** No strict real term is equal to `√x` on all of `[0, ∞)`.
If `T(x)² = x` for `x ≥ 0`, differentiating from the right at `0` gives `2 T(0) T'(0) = 1`,
although `T(0) = 0`. -/
theorem no_sqrt1b : ¬ ∃ W : Term 1, ∀ x : ℝ, 0 ≤ x → evalR W ![x] = some (Real.sqrt x) := by
  rintro ⟨W, hW⟩
  obtain ⟨g, hg, hg0, hge⟩ := evalR_comp_differentiableAt (differentiableAt_vec1 0) (hW 0 le_rfl)
  rw [Real.sqrt_zero] at hg0
  have h1 : ∀ᶠ x in 𝓝[Set.Ici 0] (0 : ℝ), g x ^ 2 = x := by
    filter_upwards [nhdsWithin_le_nhds hge, self_mem_nhdsWithin] with x hx hx0
    have := hW x hx0
    rw [hx] at this
    rw [Option.some.inj this]
    exact Real.sq_sqrt hx0
  have hd : HasDerivWithinAt (fun x => g x ^ 2) (2 * g 0 ^ (2 - 1) * deriv g 0) (Set.Ici 0) 0 :=
    (hg.hasDerivAt.pow 2).hasDerivWithinAt
  have hd' : HasDerivWithinAt (fun x => g x ^ 2) 1 (Set.Ici 0) 0 :=
    ((hasDerivAt_id (0 : ℝ)).hasDerivWithinAt).congr_of_eventuallyEq h1 (by simp [hg0])
  have := (uniqueDiffWithinAt_Ici (0 : ℝ)).eq_deriv _ hd hd'
  simp [hg0] at this

/-- **Hypotenuse.** No strict real term is equal to `√(x² + y²)` on all of `ℝ²`.
Restricted to the line `(t, 0)` it would be `|t|`, which has no derivative at `0`. -/
theorem no_hypot1b :
    ¬ ∃ W : Term 2, ∀ x y : ℝ, evalR W ![x, y] = some (Real.sqrt (x ^ 2 + y ^ 2)) := by
  rintro ⟨W, hW⟩
  obtain ⟨g, hg, -, hge⟩ := evalR_comp_differentiableAt (differentiableAt_vec2 0) (hW 0 0)
  have habs : (fun x : ℝ => |x|) =ᶠ[𝓝 0] g := by
    filter_upwards [hge] with x hx
    have := hW x 0
    rw [hx] at this
    rw [Option.some.inj this]
    simp [Real.sqrt_sq_eq_abs]
  exact not_differentiableAt_abs_zero (hg.congr_of_eventuallyEq habs)

/-- **Lemma B.7**, second half. No strict real term is equal to `arcosh x` on all of `[1, ∞)`.
Differentiating `cosh(T(x)) = x` from the right at `1` gives `sinh(T(1)) T'(1) = 1`, although
`T(1) = 0`. -/
theorem no_arcosh1b :
    ¬ ∃ W : Term 1, ∀ x : ℝ, 1 ≤ x → evalR W ![x] = some (Real.arcosh x) := by
  rintro ⟨W, hW⟩
  obtain ⟨g, hg, hg1, hge⟩ := evalR_comp_differentiableAt (differentiableAt_vec1 1) (hW 1 le_rfl)
  rw [Real.arcosh_zero] at hg1
  have h1 : ∀ᶠ x in 𝓝[Set.Ici 1] (1 : ℝ), Real.cosh (g x) = x := by
    filter_upwards [nhdsWithin_le_nhds hge, self_mem_nhdsWithin] with x hx hx1
    have := hW x hx1
    rw [hx] at this
    rw [Option.some.inj this]
    exact Real.cosh_arcosh hx1
  have hd : HasDerivWithinAt (fun x => Real.cosh (g x)) (Real.sinh (g 1) * deriv g 1)
      (Set.Ici 1) 1 :=
    (hg.hasDerivAt.cosh).hasDerivWithinAt
  have hd' : HasDerivWithinAt (fun x => Real.cosh (g x)) 1 (Set.Ici 1) 1 :=
    ((hasDerivAt_id (1 : ℝ)).hasDerivWithinAt).congr_of_eventuallyEq h1 (by simp [hg1])
  have := (uniqueDiffWithinAt_Ici (1 : ℝ)).eq_deriv _ hd hd'
  simp [hg1] at this

end EML
