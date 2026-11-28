module

public import EML.Real.Strict

/-!
# Theorem 1b (Appendix B): a strict real term in one variable for large `x`

For every strict real term in one variable there is `x₀` such that on `(x₀, ∞)` the term is
undefined everywhere, or `0` everywhere, or positive everywhere, or negative everywhere
(`evalR_eventually`). Hence no strict real term is equal to `sin` or to `cos` on `ℝ`, or to
`tan` on its natural domain: these functions change sign beyond every `x₀`
(`no_sin1b`, `no_cos1b`, `no_tan1b`; paragraph "Sine, cosine and tangent" of Appendix B).

The paper gives two arguments for this paragraph: Wilkie's theorem (the real exponential field
is o-minimal), and Hardy's theorem on the functions built from `exp`, `ln` and the arithmetic
operations (G. H. Hardy, *Properties of logarithmico-exponential functions*, Proc. London Math.
Soc. (2) 10 (1912), 54-90). The second one is proved here, by Hardy's argument. No model theory
is used.

* `HardyClass H` — `H` is a set of functions `ℝ → ℝ`, looked at for large `x` only, that
  contains the constants and is closed under `+`, `×`, reciprocals of functions that do not
  vanish for large `x`, and differentiation; every function of `H` is `0` for all large `x` or
  non-zero for all large `x`.
* `PolyIn H g d` — the functions `c₀ + c₁ g + ⋯ + c_d g^d` with coefficients in `H`.
* `PolyIn.zero_or_ne` — Hardy's lemma: if `g' = k g + l` for large `x`, with `k`, `l` in `H`,
  such a function is `0` for all large `x` or non-zero for all large `x`.
* `Adjoin H g` — the quotients of such functions; `hardyClass_adjoin`: again a `HardyClass`.
  The three cases used are `g = x`, `g = exp f` and `g = ln f` with `f` in `H`.
* `evalR_eventually` — the statement for terms, by induction on the term: each node
  `eml(S, T)` adjoins `exp S` and `ln T`.
-/

@[expose] public section

namespace EML

open Term Filter Topology

/-! ## Functions of constant sign for large `x` -/

/-- A function that is continuous and non-zero for all large `x` is positive for all large `x`
or negative for all large `x` (intermediate value theorem). -/
theorem eventually_pos_or_neg {f : ℝ → ℝ} (hc : ∀ᶠ x in atTop, ContinuousAt f x)
    (hne : ∀ᶠ x in atTop, f x ≠ 0) :
    (∀ᶠ x in atTop, 0 < f x) ∨ (∀ᶠ x in atTop, f x < 0) := by
  obtain ⟨x₀, h₀⟩ := eventually_atTop.mp (hc.and hne)
  have hcont : ∀ x, ContinuousOn f (Set.Icc x₀ x) := fun x z hz =>
    (h₀ z hz.1).1.continuousWithinAt
  rcases lt_or_gt_of_ne (h₀ x₀ le_rfl).2 with h | h
  · right
    filter_upwards [eventually_ge_atTop x₀] with x hx
    by_contra hcon
    obtain ⟨z, hz, hfz⟩ := intermediate_value_Icc hx (hcont x) ⟨h.le, not_lt.mp hcon⟩
    exact (h₀ z hz.1).2 hfz
  · left
    filter_upwards [eventually_ge_atTop x₀] with x hx
    by_contra hcon
    obtain ⟨z, hz, hfz⟩ := intermediate_value_Icc' hx (hcont x) ⟨not_lt.mp hcon, h.le⟩
    exact (h₀ z hz.1).2 hfz

/-- A function that is injective on `[x₀, ∞)` is non-zero for all large `x`: it has at most one
zero there. -/
theorem eventually_ne_zero_of_injOn {f : ℝ → ℝ} {x₀ : ℝ} (h : Set.InjOn f (Set.Ici x₀)) :
    ∀ᶠ x in atTop, f x ≠ 0 := by
  by_cases hz : ∃ z, x₀ ≤ z ∧ f z = 0
  · obtain ⟨z, hz₀, hz⟩ := hz
    filter_upwards [eventually_gt_atTop z] with x hx hx0
    exact hx.ne' (h (hz₀.trans hx.le) hz₀ (hx0.trans hz.symm))
  · filter_upwards [eventually_ge_atTop x₀] with x hx hx0
    exact hz ⟨x, hx, hx0⟩

/-! ## Hardy classes -/

/-- A *Hardy class*: a set `H` of functions `ℝ → ℝ`, looked at for large `x` only, that contains
the constants and is closed under sums, products, reciprocals and differentiation, and in which
every function is `0` for all large `x` or non-zero for all large `x`. -/
structure HardyClass (H : Set (ℝ → ℝ)) : Prop where
  /-- a function that agrees with a function of `H` for all large `x` is in `H` -/
  congr : ∀ {f g : ℝ → ℝ}, f ∈ H → g =ᶠ[atTop] f → g ∈ H
  /-- the constants are in `H` -/
  const : ∀ c : ℝ, (fun _ => c) ∈ H
  /-- sums -/
  add : ∀ {f g : ℝ → ℝ}, f ∈ H → g ∈ H → (fun x => f x + g x) ∈ H
  /-- products -/
  mul : ∀ {f g : ℝ → ℝ}, f ∈ H → g ∈ H → (fun x => f x * g x) ∈ H
  /-- a function of `H` is `0` for all large `x`, or non-zero for all large `x` -/
  zero_or_ne : ∀ {f : ℝ → ℝ}, f ∈ H →
    (∀ᶠ x in atTop, f x = 0) ∨ (∀ᶠ x in atTop, f x ≠ 0)
  /-- reciprocals of functions that are non-zero for all large `x` -/
  inv : ∀ {f : ℝ → ℝ}, f ∈ H → (∀ᶠ x in atTop, f x ≠ 0) → (fun x => (f x)⁻¹) ∈ H
  /-- derivatives: a function of `H` is differentiable for all large `x`, with derivative in `H` -/
  hasDeriv : ∀ {f : ℝ → ℝ}, f ∈ H → ∃ f' ∈ H, ∀ᶠ x in atTop, HasDerivAt f (f' x) x

namespace HardyClass

variable {H : Set (ℝ → ℝ)} {f g : ℝ → ℝ}

/-- Differences. -/
theorem sub (hH : HardyClass H) (hf : f ∈ H) (hg : g ∈ H) : (fun x => f x - g x) ∈ H := by
  have h := hH.add hf (hH.mul (hH.const (-1)) hg)
  have e : (fun x => f x - g x) = fun x => f x + -1 * g x := by funext x; ring
  rw [e]
  exact h

/-- A function of a Hardy class is continuous for all large `x`. -/
theorem continuousAt (hH : HardyClass H) (hf : f ∈ H) : ∀ᶠ x in atTop, ContinuousAt f x := by
  obtain ⟨f', -, hf'⟩ := hH.hasDeriv hf
  exact hf'.mono fun x hx => hx.continuousAt

/-- A function of a Hardy class is `0` for all large `x`, or positive for all large `x`, or
negative for all large `x`. -/
theorem sign (hH : HardyClass H) (hf : f ∈ H) :
    (∀ᶠ x in atTop, f x = 0) ∨ (∀ᶠ x in atTop, 0 < f x) ∨ (∀ᶠ x in atTop, f x < 0) := by
  rcases hH.zero_or_ne hf with h | h
  · exact Or.inl h
  · exact Or.inr (eventually_pos_or_neg (hH.continuousAt hf) h)

end HardyClass

/-- The functions that are constant for all large `x`. -/
def ConstClass : Set (ℝ → ℝ) := {f | ∃ c : ℝ, f =ᶠ[atTop] fun _ => c}

/-- The functions that are constant for all large `x` form a Hardy class. -/
theorem hardyClass_const : HardyClass ConstClass where
  congr := by
    rintro f g ⟨c, hf⟩ hg
    exact ⟨c, hg.trans hf⟩
  const c := ⟨c, EventuallyEq.rfl⟩
  add := by
    rintro f g ⟨c₁, hf⟩ ⟨c₂, hg⟩
    refine ⟨c₁ + c₂, ?_⟩
    filter_upwards [hf, hg] with x h₁ h₂
    rw [h₁, h₂]
  mul := by
    rintro f g ⟨c₁, hf⟩ ⟨c₂, hg⟩
    refine ⟨c₁ * c₂, ?_⟩
    filter_upwards [hf, hg] with x h₁ h₂
    rw [h₁, h₂]
  zero_or_ne := by
    rintro f ⟨c, hf⟩
    by_cases hc : c = 0
    · left
      filter_upwards [hf] with x hx
      rw [hx, hc]
    · right
      filter_upwards [hf] with x hx
      rw [hx]
      exact hc
  inv := by
    rintro f ⟨c, hf⟩ -
    refine ⟨c⁻¹, ?_⟩
    filter_upwards [hf] with x hx
    rw [hx]
  hasDeriv := by
    rintro f ⟨c, hf⟩
    refine ⟨fun _ => 0, ⟨0, EventuallyEq.rfl⟩, ?_⟩
    obtain ⟨x₀, h₀⟩ := eventually_atTop.mp hf
    filter_upwards [eventually_gt_atTop x₀] with x hx
    have hloc : f =ᶠ[𝓝 x] fun _ => c := by
      filter_upwards [Ioi_mem_nhds hx] with z hz using h₀ z (le_of_lt hz)
    exact (hasDerivAt_const x c).congr_of_eventuallyEq hloc

/-! ## Polynomial expressions in a function `g` -/

/-- `PolyIn H g d`: the functions `c₀ + c₁ g + ⋯ + c_d g^d` with coefficients `c₀, …, c_d` in
`H`. -/
def PolyIn (H : Set (ℝ → ℝ)) (g : ℝ → ℝ) : ℕ → Set (ℝ → ℝ)
  | 0 => H
  | d + 1 => {p | ∃ q ∈ PolyIn H g d, ∃ c ∈ H, p = fun x => q x + c x * g x ^ (d + 1)}

/-- `Adjoinable H g`: for all large `x` the function `g` satisfies `g' = k g + l`, where `k` and
`l` are in `H` and `k` is the derivative of some function `K`. The cases used below are `g = x`
(`k = 0`, `l = 1`), `g = exp f` (`k = f'`, `K = f`, `l = 0`) and `g = ln f` (`k = 0`,
`l = f'/f`). -/
def Adjoinable (H : Set (ℝ → ℝ)) (g : ℝ → ℝ) : Prop :=
  ∃ k ∈ H, ∃ l ∈ H, ∃ K : ℝ → ℝ, (∀ᶠ x in atTop, HasDerivAt K (k x) x) ∧
    ∀ᶠ x in atTop, HasDerivAt g (k x * g x + l x) x

namespace PolyIn

variable {H : Set (ℝ → ℝ)} {g p q c : ℝ → ℝ} {d e : ℕ}

/-- A polynomial expression of degree at most `d` is one of degree at most `d + 1`. -/
theorem succ (hH : HardyClass H) (hp : p ∈ PolyIn H g d) : p ∈ PolyIn H g (d + 1) :=
  ⟨p, hp, fun _ => 0, hH.const 0, by funext x; simp⟩

/-- A polynomial expression of degree at most `d` is one of degree at most `e`, for `d ≤ e`. -/
theorem mono (hH : HardyClass H) (hde : d ≤ e) (hp : p ∈ PolyIn H g d) : p ∈ PolyIn H g e := by
  induction e, hde using Nat.le_induction with
  | base => exact hp
  | succ e _ ih => exact succ hH ih

/-- The functions of `H` are the polynomial expressions of degree `0`. -/
theorem of_mem (hH : HardyClass H) (hp : p ∈ H) : p ∈ PolyIn H g d :=
  mono hH (Nat.zero_le d) (show p ∈ PolyIn H g 0 from hp)

/-- Sums. -/
theorem add (hH : HardyClass H) (hp : p ∈ PolyIn H g d) (hq : q ∈ PolyIn H g d) :
    (fun x => p x + q x) ∈ PolyIn H g d := by
  induction d generalizing p q with
  | zero => exact hH.add hp hq
  | succ d ih =>
    obtain ⟨p₁, hp₁, c₁, hc₁, rfl⟩ := hp
    obtain ⟨q₁, hq₁, c₂, hc₂, rfl⟩ := hq
    exact ⟨fun x => p₁ x + q₁ x, ih hp₁ hq₁, fun x => c₁ x + c₂ x, hH.add hc₁ hc₂, by
      funext x; ring⟩

/-- Products with a function of `H`. -/
theorem smul (hH : HardyClass H) (hc : c ∈ H) (hp : p ∈ PolyIn H g d) :
    (fun x => c x * p x) ∈ PolyIn H g d := by
  induction d generalizing p with
  | zero => exact hH.mul hc hp
  | succ d ih =>
    obtain ⟨p₁, hp₁, c₁, hc₁, rfl⟩ := hp
    exact ⟨fun x => c x * p₁ x, ih hp₁, fun x => c x * c₁ x, hH.mul hc hc₁, by funext x; ring⟩

/-- The monomial `c g^d`. -/
theorem monomial (hH : HardyClass H) (hc : c ∈ H) (d : ℕ) :
    (fun x => c x * g x ^ d) ∈ PolyIn H g d := by
  cases d with
  | zero =>
    have e : (fun x => c x * g x ^ 0) = c := by funext x; simp
    rw [e]
    exact hc
  | succ d => exact ⟨fun _ => 0, of_mem hH (hH.const 0), c, hc, by funext x; simp⟩

/-- Products with a power of `g`. -/
theorem shift (hH : HardyClass H) (hp : p ∈ PolyIn H g d) (j : ℕ) :
    (fun x => p x * g x ^ j) ∈ PolyIn H g (d + j) := by
  induction d generalizing p with
  | zero =>
    rw [Nat.zero_add]
    exact monomial hH hp j
  | succ d ih =>
    obtain ⟨p₁, hp₁, c₁, hc₁, rfl⟩ := hp
    rw [show d + 1 + j = d + j + 1 by omega]
    exact ⟨fun x => p₁ x * g x ^ j, ih hp₁, c₁, hc₁, by funext x; ring⟩

/-- Products. -/
theorem mul (hH : HardyClass H) (hp : p ∈ PolyIn H g d) (hq : q ∈ PolyIn H g e) :
    (fun x => p x * q x) ∈ PolyIn H g (d + e) := by
  induction e generalizing q with
  | zero =>
    have h := smul hH hq hp
    have e' : (fun x => p x * q x) = fun x => q x * p x := by funext x; ring
    rw [e']
    exact h
  | succ e ih =>
    obtain ⟨q₁, hq₁, c₁, hc₁, rfl⟩ := hq
    have h := add hH (succ hH (ih hq₁)) (smul hH hc₁ (shift hH hp (e + 1)))
    have e' : (fun x => p x * (q₁ x + c₁ x * g x ^ (e + 1))) =
        fun x => p x * q₁ x + c₁ x * (p x * g x ^ (e + 1)) := by funext x; ring
    rw [e']
    exact h

/-- A polynomial expression in `g` is differentiable for all large `x`, and its derivative is a
polynomial expression of the same degree: `(c g^(d+1))' = (c' + (d+1) c k) g^(d+1) +
(d+1) c l g^d`. -/
theorem hasDeriv (hH : HardyClass H) (hg : Adjoinable H g) (hp : p ∈ PolyIn H g d) :
    ∃ p' ∈ PolyIn H g d, ∀ᶠ x in atTop, HasDerivAt p (p' x) x := by
  obtain ⟨k, hk, l, hl, K, -, hgd⟩ := hg
  induction d generalizing p with
  | zero => exact hH.hasDeriv hp
  | succ d ih =>
    obtain ⟨q, hq, c, hc, rfl⟩ := hp
    obtain ⟨q', hq', hqd⟩ := ih hq
    obtain ⟨c', hc', hcd⟩ := hH.hasDeriv hc
    have h₁ : (fun x => ((d + 1 : ℕ) : ℝ) * c x * l x) ∈ H :=
      hH.mul (hH.mul (hH.const _) hc) hl
    have h₂ : (fun x => c' x + ((d + 1 : ℕ) : ℝ) * c x * k x) ∈ H :=
      hH.add hc' (hH.mul (hH.mul (hH.const _) hc) hk)
    refine ⟨fun x => (q' x + ((d + 1 : ℕ) : ℝ) * c x * l x * g x ^ d) +
        (c' x + ((d + 1 : ℕ) : ℝ) * c x * k x) * g x ^ (d + 1),
      ⟨_, add hH hq' (monomial hH h₁ d), _, h₂, rfl⟩, ?_⟩
    filter_upwards [hqd, hcd, hgd] with x hx₁ hx₂ hx₃
    have h := hx₁.fun_add (hx₂.fun_mul (hx₃.fun_pow (d + 1)))
    convert h using 1
    rw [Nat.add_sub_cancel]
    ring

/-- **Hardy's lemma**, the case of the leading coefficient `1`. Let `w = q + g^(d+1)` with `q`
of degree at most `d`. Then `w' - (d+1) k w` is of degree at most `d`. So the derivative of
`w exp(-(d+1) K)` is `0` for all large `x`, or positive for all large `x`, or negative for all
large `x`, and this function is constant or strictly monotone there. -/
theorem monic_zero_or_ne (hH : HardyClass H) (hg : Adjoinable H g)
    (ih : ∀ {p : ℝ → ℝ}, p ∈ PolyIn H g d →
      (∀ᶠ x in atTop, p x = 0) ∨ (∀ᶠ x in atTop, p x ≠ 0))
    (hq : q ∈ PolyIn H g d) :
    (∀ᶠ x in atTop, q x + g x ^ (d + 1) = 0) ∨ (∀ᶠ x in atTop, q x + g x ^ (d + 1) ≠ 0) := by
  obtain ⟨q', hq', hqd⟩ := hasDeriv hH hg hq
  obtain ⟨k, hk, l, hl, K, hKd, hgd⟩ := hg
  -- `r = w' - (d+1) k w`, of degree at most `d`
  have hr : (fun x => q' x + -((d + 1 : ℕ) : ℝ) * k x * q x + ((d + 1 : ℕ) : ℝ) * l x * g x ^ d)
      ∈ PolyIn H g d :=
    add hH (add hH hq' (smul hH (hH.mul (hH.const _) hk) hq))
      (monomial hH (hH.mul (hH.const _) hl) d)
  set r : ℝ → ℝ :=
    fun x => q' x + -((d + 1 : ℕ) : ℝ) * k x * q x + ((d + 1 : ℕ) : ℝ) * l x * g x ^ d with hrdef
  -- `v = w exp(-(d+1) K)` has the derivative `r exp(-(d+1) K)`
  set v : ℝ → ℝ :=
    fun x => (q x + g x ^ (d + 1)) * Real.exp (-((d + 1 : ℕ) : ℝ) * K x) with hvdef
  have hv : ∀ᶠ x in atTop, HasDerivAt v (r x * Real.exp (-((d + 1 : ℕ) : ℝ) * K x)) x := by
    filter_upwards [hqd, hgd, hKd] with x hx₁ hx₂ hx₃
    have h := (hx₁.fun_add (hx₂.fun_pow (d + 1))).fun_mul
      ((hx₃.const_mul (-((d + 1 : ℕ) : ℝ))).exp)
    convert h using 1
    rw [hrdef, Nat.add_sub_cancel]
    ring
  have hvw : ∀ x, v x = 0 ↔ q x + g x ^ (d + 1) = 0 := fun x => by
    simp [hvdef, Real.exp_ne_zero]
  suffices h : (∀ᶠ x in atTop, v x = 0) ∨ (∀ᶠ x in atTop, v x ≠ 0) by
    rcases h with h | h
    · exact Or.inl (h.mono fun x hx => (hvw x).mp hx)
    · exact Or.inr (h.mono fun x hx h0 => hx ((hvw x).mpr h0))
  rcases ih hr with hr0 | hrne
  · -- `r = 0` for all large `x`: `v` is constant there
    obtain ⟨x₀, h₀⟩ := eventually_atTop.mp (hv.and hr0)
    have hconst : ∀ x, x₀ ≤ x → v x = v x₀ := fun x hx =>
      constant_of_has_deriv_right_zero
        (fun z hz => (h₀ z hz.1).1.continuousAt.continuousWithinAt)
        (fun z hz => by
          have h := (h₀ z hz.1).1
          rw [(h₀ z hz.1).2, zero_mul] at h
          exact h.hasDerivWithinAt) x ⟨hx, le_rfl⟩
    by_cases hz : v x₀ = 0
    · left
      filter_upwards [eventually_ge_atTop x₀] with x hx
      rw [hconst x hx, hz]
    · right
      filter_upwards [eventually_ge_atTop x₀] with x hx
      rw [hconst x hx]
      exact hz
  · -- `r` has a constant sign for all large `x`: `v` is strictly monotone there
    have hrc : ∀ᶠ x in atTop, ContinuousAt r x := by
      obtain ⟨r', -, hr'⟩ := hasDeriv hH ⟨k, hk, l, hl, K, hKd, hgd⟩ hr
      exact hr'.mono fun x hx => hx.continuousAt
    rcases eventually_pos_or_neg hrc hrne with hpos | hneg
    · obtain ⟨x₀, h₀⟩ := eventually_atTop.mp (hv.and hpos)
      have hmono : StrictMonoOn v (Set.Ici x₀) := by
        refine strictMonoOn_of_deriv_pos (convex_Ici x₀)
          (fun z hz => (h₀ z hz).1.continuousAt.continuousWithinAt) fun z hz => ?_
        rw [interior_Ici] at hz
        rw [(h₀ z hz.le).1.deriv]
        exact mul_pos (h₀ z hz.le).2 (Real.exp_pos _)
      exact Or.inr (eventually_ne_zero_of_injOn hmono.injOn)
    · obtain ⟨x₀, h₀⟩ := eventually_atTop.mp (hv.and hneg)
      have hanti : StrictAntiOn v (Set.Ici x₀) := by
        refine strictAntiOn_of_deriv_neg (convex_Ici x₀)
          (fun z hz => (h₀ z hz).1.continuousAt.continuousWithinAt) fun z hz => ?_
        rw [interior_Ici] at hz
        rw [(h₀ z hz.le).1.deriv]
        exact mul_neg_of_neg_of_pos (h₀ z hz.le).2 (Real.exp_pos _)
      exact Or.inr (eventually_ne_zero_of_injOn hanti.injOn)

/-- **Hardy's lemma.** If `g' = k g + l` for all large `x`, with `k`, `l` in the Hardy class
`H`, then a polynomial expression in `g` with coefficients in `H` is `0` for all large `x` or
non-zero for all large `x`. Induction on the degree: divide by the leading coefficient. -/
theorem zero_or_ne (hH : HardyClass H) (hg : Adjoinable H g) (hp : p ∈ PolyIn H g d) :
    (∀ᶠ x in atTop, p x = 0) ∨ (∀ᶠ x in atTop, p x ≠ 0) := by
  induction d generalizing p with
  | zero => exact hH.zero_or_ne hp
  | succ d ih =>
    obtain ⟨q, hq, c, hc, rfl⟩ := hp
    rcases hH.zero_or_ne hc with hc0 | hc0
    · -- the leading coefficient is `0` for all large `x`
      rcases ih hq with h | h
      · left
        filter_upwards [hc0, h] with x hx₁ hx₂
        simp [hx₁, hx₂]
      · right
        filter_upwards [hc0, h] with x hx₁ hx₂
        simpa [hx₁] using hx₂
    · -- divide by the leading coefficient
      have hq₁ : (fun x => (c x)⁻¹ * q x) ∈ PolyIn H g d := smul hH (hH.inv hc hc0) hq
      have e : ∀ x, c x ≠ 0 →
          q x + c x * g x ^ (d + 1) = c x * ((c x)⁻¹ * q x + g x ^ (d + 1)) := fun x hx => by
        field_simp
      rcases monic_zero_or_ne hH hg ih hq₁ with h | h
      · left
        filter_upwards [hc0, h] with x hx₁ hx₂
        rw [e x hx₁, hx₂, mul_zero]
      · right
        filter_upwards [hc0, h] with x hx₁ hx₂
        rw [e x hx₁]
        exact mul_ne_zero hx₁ hx₂

end PolyIn

/-! ## Adjoining a function to a Hardy class -/

/-- `Adjoin H g`: the functions that agree, for all large `x`, with a quotient `p / q` of two
polynomial expressions in `g` with coefficients in `H`, where `q` is non-zero for all large
`x`. -/
def Adjoin (H : Set (ℝ → ℝ)) (g : ℝ → ℝ) : Set (ℝ → ℝ) :=
  {f | ∃ d, ∃ p ∈ PolyIn H g d, ∃ q ∈ PolyIn H g d,
    (∀ᶠ x in atTop, q x ≠ 0) ∧ f =ᶠ[atTop] fun x => p x / q x}

variable {H : Set (ℝ → ℝ)} {f g : ℝ → ℝ}

/-- `H` is contained in `Adjoin H g`. -/
theorem subset_adjoin (hH : HardyClass H) : H ⊆ Adjoin H g := fun f hf =>
  ⟨0, f, hf, fun _ => 1, hH.const 1, .of_forall fun _ => one_ne_zero,
    .of_forall fun x => by simp⟩

/-- `g` is in `Adjoin H g`. -/
theorem mem_adjoin_self (hH : HardyClass H) : g ∈ Adjoin H g :=
  ⟨1, fun x => 0 + 1 * g x ^ 1, ⟨fun _ => 0, hH.const 0, fun _ => 1, hH.const 1, rfl⟩,
    fun _ => 1, PolyIn.of_mem hH (hH.const 1), .of_forall fun _ => one_ne_zero,
    .of_forall fun x => by simp⟩

/-- **Adjunction.** If `g' = k g + l` for all large `x`, with `k`, `l` in the Hardy class `H`,
then the quotients of polynomial expressions in `g` form a Hardy class. The only step that is
not arithmetic is Hardy's lemma (`PolyIn.zero_or_ne`). -/
theorem hardyClass_adjoin (hH : HardyClass H) (hg : Adjoinable H g) :
    HardyClass (Adjoin H g) where
  congr := by
    rintro f₁ f₂ ⟨d, p, hp, q, hq, hq0, hf⟩ h
    exact ⟨d, p, hp, q, hq, hq0, h.trans hf⟩
  const c := subset_adjoin hH (hH.const c)
  add := by
    rintro f₁ f₂ ⟨d₁, p₁, hp₁, q₁, hq₁, h₁, hf₁⟩ ⟨d₂, p₂, hp₂, q₂, hq₂, h₂, hf₂⟩
    refine ⟨d₁ + d₂, fun x => p₁ x * q₂ x + q₁ x * p₂ x,
      PolyIn.add hH (PolyIn.mul hH hp₁ hq₂) (PolyIn.mul hH hq₁ hp₂),
      fun x => q₁ x * q₂ x, PolyIn.mul hH hq₁ hq₂, ?_, ?_⟩
    · filter_upwards [h₁, h₂] with x hx₁ hx₂ using mul_ne_zero hx₁ hx₂
    · filter_upwards [h₁, h₂, hf₁, hf₂] with x hx₁ hx₂ hx₃ hx₄
      rw [hx₃, hx₄]
      field_simp
  mul := by
    rintro f₁ f₂ ⟨d₁, p₁, hp₁, q₁, hq₁, h₁, hf₁⟩ ⟨d₂, p₂, hp₂, q₂, hq₂, h₂, hf₂⟩
    refine ⟨d₁ + d₂, fun x => p₁ x * p₂ x, PolyIn.mul hH hp₁ hp₂,
      fun x => q₁ x * q₂ x, PolyIn.mul hH hq₁ hq₂, ?_, ?_⟩
    · filter_upwards [h₁, h₂] with x hx₁ hx₂ using mul_ne_zero hx₁ hx₂
    · filter_upwards [hf₁, hf₂] with x hx₃ hx₄
      rw [hx₃, hx₄, div_mul_div_comm]
  zero_or_ne := by
    rintro f₁ ⟨d, p, hp, q, hq, hq0, hf⟩
    rcases PolyIn.zero_or_ne hH hg hp with h | h
    · left
      filter_upwards [hf, h] with x hx₁ hx₂
      rw [hx₁, hx₂, zero_div]
    · right
      filter_upwards [hf, h, hq0] with x hx₁ hx₂ hx₃
      rw [hx₁]
      exact div_ne_zero hx₂ hx₃
  inv := by
    rintro f₁ ⟨d, p, hp, q, hq, hq0, hf⟩ hf0
    refine ⟨d, q, hq, p, hp, ?_, ?_⟩
    · filter_upwards [hf, hf0] with x hx₁ hx₂ hx₃
      apply hx₂
      rw [hx₁, hx₃, zero_div]
    · filter_upwards [hf] with x hx₁
      rw [hx₁, inv_div]
  hasDeriv := by
    rintro f₁ ⟨d, p, hp, q, hq, hq0, hf⟩
    obtain ⟨p', hp', hpd⟩ := PolyIn.hasDeriv hH hg hp
    obtain ⟨q', hq', hqd⟩ := PolyIn.hasDeriv hH hg hq
    have hn : (fun x => p' x * q x + -1 * (p x * q' x)) ∈ PolyIn H g (d + d) :=
      PolyIn.add hH (PolyIn.mul hH hp' hq)
        (PolyIn.smul hH (hH.const (-1)) (PolyIn.mul hH hp hq'))
    refine ⟨fun x => (p' x * q x + -1 * (p x * q' x)) / (q x * q x),
      ⟨d + d, _, hn, _, PolyIn.mul hH hq hq, ?_, EventuallyEq.rfl⟩, ?_⟩
    · filter_upwards [hq0] with x hx using mul_ne_zero hx hx
    · obtain ⟨x₀, h₀⟩ := eventually_atTop.mp hf
      filter_upwards [hpd, hqd, hq0, eventually_gt_atTop x₀] with x hx₁ hx₂ hx₃ hx
      have hloc : f₁ =ᶠ[𝓝 x] fun x => p x / q x := by
        filter_upwards [Ioi_mem_nhds hx] with z hz using h₀ z (le_of_lt hz)
      have h := (hx₁.fun_div hx₂ hx₃).congr_of_eventuallyEq hloc
      convert h using 1
      rw [pow_two]
      ring

/-- `g = x`: here `g' = 1`. -/
theorem adjoinable_id (hH : HardyClass H) : Adjoinable H fun x => x :=
  ⟨fun _ => 0, hH.const 0, fun _ => 1, hH.const 1, fun _ => 0,
    .of_forall fun x => hasDerivAt_const x 0,
    .of_forall fun x => by simpa using hasDerivAt_id' x⟩

/-- `g = exp f` with `f` in `H`: here `g' = f' g`. -/
theorem adjoinable_exp (hH : HardyClass H) (hf : f ∈ H) :
    Adjoinable H fun x => Real.exp (f x) := by
  obtain ⟨f', hf', hfd⟩ := hH.hasDeriv hf
  refine ⟨f', hf', fun _ => 0, hH.const 0, f, hfd, ?_⟩
  filter_upwards [hfd] with x hx
  have h := hx.exp
  convert h using 1
  ring

/-- `g = ln f` with `f` in `H` positive for all large `x`: here `g' = f'/f`. -/
theorem adjoinable_log (hH : HardyClass H) (hf : f ∈ H) (hpos : ∀ᶠ x in atTop, 0 < f x) :
    Adjoinable H fun x => Real.log (f x) := by
  obtain ⟨f', hf', hfd⟩ := hH.hasDeriv hf
  have hne : ∀ᶠ x in atTop, f x ≠ 0 := hpos.mono fun x hx => hx.ne'
  refine ⟨fun _ => 0, hH.const 0, fun x => f' x * (f x)⁻¹, hH.mul hf' (hH.inv hf hne),
    fun _ => 0, .of_forall fun x => hasDerivAt_const x 0, ?_⟩
  filter_upwards [hfd, hne] with x hx hx0
  have h := hx.log hx0
  convert h using 1
  rw [zero_mul, zero_add, div_eq_mul_inv]

/-! ## Strict real terms in one variable -/

/-- For all large `x`, a strict real term in one variable is undefined, or it is defined and its
value is a function of a Hardy class. Induction on the term: at a node `eml(S, T)` the value of
`T` is `0`, positive or negative for all large `x`; in the second case `exp S` and `ln T` are
adjoined. -/
theorem evalR_mem_hardyClass (t : Term 1) (hH : HardyClass H) (hid : (fun x : ℝ => x) ∈ H) :
    (∀ᶠ x in atTop, evalR t ![x] = none) ∨
      ∃ H', H ⊆ H' ∧ HardyClass H' ∧ ∃ f ∈ H', ∀ᶠ x in atTop, evalR t ![x] = some (f x) := by
  induction t generalizing H with
  | one => exact Or.inr ⟨H, subset_rfl, hH, fun _ => 1, hH.const 1, .of_forall fun _ => rfl⟩
  | var j =>
    refine Or.inr ⟨H, subset_rfl, hH, fun x => x, hid, .of_forall fun x => ?_⟩
    fin_cases j
    rfl
  | eml s t ihs iht =>
    rcases ihs hH hid with hs | ⟨H₁, h₁, hH₁, f₁, hf₁, hs⟩
    · exact Or.inl (hs.mono fun x hx => evalR_eml_undef_left hx)
    rcases iht hH₁ (h₁ hid) with ht | ⟨H₂, h₂, hH₂, f₂, hf₂, ht⟩
    · exact Or.inl (ht.mono fun x hx => evalR_eml_undef_right hx)
    rcases hH₂.sign hf₂ with h0 | hpos | hneg
    · left
      filter_upwards [ht, h0] with x hx hx0
      exact evalR_eml_nonpos hx hx0.le
    · right
      have hH₃ := hardyClass_adjoin hH₂ (adjoinable_exp hH₂ (h₂ hf₁))
      have hH₄ := hardyClass_adjoin hH₃ (adjoinable_log hH₃ (subset_adjoin hH₂ hf₂) hpos)
      refine ⟨_, (h₁.trans h₂).trans ((subset_adjoin hH₂).trans (subset_adjoin hH₃)), hH₄,
        fun x => Real.exp (f₁ x) - Real.log (f₂ x),
        hH₄.sub (subset_adjoin hH₃ (mem_adjoin_self hH₂)) (mem_adjoin_self hH₃), ?_⟩
      filter_upwards [hs, ht, hpos] with x hx₁ hx₂ hx₃
      exact evalR_eml hx₁ hx₂ hx₃
    · left
      filter_upwards [ht, hneg] with x hx hx0
      exact evalR_eml_nonpos hx hx0.le

/-- **Hardy's theorem for strict real terms.** For a strict real term in one variable there is
`x₀` such that on `(x₀, ∞)` the term is undefined everywhere, or it is defined everywhere and
its value is `0` everywhere, or positive everywhere, or negative everywhere. -/
theorem evalR_eventually (t : Term 1) :
    (∀ᶠ x in atTop, evalR t ![x] = none) ∨
      ∃ f : ℝ → ℝ, (∀ᶠ x in atTop, evalR t ![x] = some (f x)) ∧
        ((∀ᶠ x in atTop, f x = 0) ∨ (∀ᶠ x in atTop, 0 < f x) ∨ (∀ᶠ x in atTop, f x < 0)) := by
  have hH := hardyClass_adjoin hardyClass_const (adjoinable_id hardyClass_const)
  rcases evalR_mem_hardyClass t hH (mem_adjoin_self hardyClass_const) with
    h | ⟨H', -, hH', f, hf, ht⟩
  · exact Or.inl h
  · exact Or.inr ⟨f, ht, hH'.sign hf⟩

/-! ## No strict real term for `sin`, `cos`, `tan` -/

/-- `SignChanges φ D`: beyond every `x₀` the function `φ` is positive at some point of `D` and
negative at some point of `D`. -/
def SignChanges (φ : ℝ → ℝ) (D : Set ℝ) : Prop :=
  (∀ x₀ : ℝ, ∃ x ∈ D, x₀ ≤ x ∧ 0 < φ x) ∧ (∀ x₀ : ℝ, ∃ x ∈ D, x₀ ≤ x ∧ φ x < 0)

/-- A function that changes sign on `D` beyond every `x₀` is not given on `D` by a strict real
term. -/
theorem no_term_of_signChanges {φ : ℝ → ℝ} {D : Set ℝ} (h : SignChanges φ D) :
    ¬ ∃ W : Term 1, ∀ x ∈ D, evalR W ![x] = some (φ x) := by
  obtain ⟨hpos, hneg⟩ := h
  rintro ⟨W, hW⟩
  rcases evalR_eventually W with h | ⟨f, hf, h | h | h⟩
  · obtain ⟨x₀, h₀⟩ := eventually_atTop.mp h
    obtain ⟨x, hxD, hx, -⟩ := hpos x₀
    have h' := hW x hxD
    rw [h₀ x hx] at h'
    cases h'
  · obtain ⟨x₀, h₀⟩ := eventually_atTop.mp (hf.and h)
    obtain ⟨x, hxD, hx, hφ⟩ := hpos x₀
    have e : φ x = f x := Option.some.inj ((hW x hxD).symm.trans (h₀ x hx).1)
    rw [e, (h₀ x hx).2] at hφ
    exact lt_irrefl _ hφ
  · obtain ⟨x₀, h₀⟩ := eventually_atTop.mp (hf.and h)
    obtain ⟨x, hxD, hx, hφ⟩ := hneg x₀
    have e : φ x = f x := Option.some.inj ((hW x hxD).symm.trans (h₀ x hx).1)
    rw [e] at hφ
    exact lt_asymm hφ (h₀ x hx).2
  · obtain ⟨x₀, h₀⟩ := eventually_atTop.mp (hf.and h)
    obtain ⟨x, hxD, hx, hφ⟩ := hpos x₀
    have e : φ x = f x := Option.some.inj ((hW x hxD).symm.trans (h₀ x hx).1)
    rw [e] at hφ
    exact lt_asymm hφ (h₀ x hx).2

/-- Beyond every `x₀` there is a point `y + 2πn` with a natural number `n`. -/
theorem exists_add_nat_mul_two_pi_ge (y x₀ : ℝ) : ∃ n : ℕ, x₀ ≤ y + n * (2 * Real.pi) := by
  obtain ⟨n, hn⟩ := exists_nat_ge (x₀ - y)
  refine ⟨n, ?_⟩
  have h₁ : (0 : ℝ) ≤ n := Nat.cast_nonneg n
  have h₂ : (n : ℝ) ≤ n * (2 * Real.pi) := by nlinarith [Real.pi_gt_three]
  linarith

/-- `sin(π/2 + 2πn) = 1` and `sin(-π/2 + 2πn) = -1`. -/
theorem signChanges_sin : SignChanges Real.sin Set.univ := by
  refine ⟨fun x₀ => ?_, fun x₀ => ?_⟩
  · obtain ⟨n, hn⟩ := exists_add_nat_mul_two_pi_ge (Real.pi / 2) x₀
    refine ⟨_, Set.mem_univ _, hn, ?_⟩
    rw [Real.sin_add_nat_mul_two_pi, Real.sin_pi_div_two]
    exact one_pos
  · obtain ⟨n, hn⟩ := exists_add_nat_mul_two_pi_ge (-(Real.pi / 2)) x₀
    refine ⟨_, Set.mem_univ _, hn, ?_⟩
    rw [Real.sin_add_nat_mul_two_pi, Real.sin_neg, Real.sin_pi_div_two]
    norm_num

/-- `cos(2πn) = 1` and `cos(π + 2πn) = -1`. -/
theorem signChanges_cos : SignChanges Real.cos Set.univ := by
  refine ⟨fun x₀ => ?_, fun x₀ => ?_⟩
  · obtain ⟨n, hn⟩ := exists_add_nat_mul_two_pi_ge 0 x₀
    refine ⟨_, Set.mem_univ _, hn, ?_⟩
    rw [Real.cos_add_nat_mul_two_pi, Real.cos_zero]
    exact one_pos
  · obtain ⟨n, hn⟩ := exists_add_nat_mul_two_pi_ge Real.pi x₀
    refine ⟨_, Set.mem_univ _, hn, ?_⟩
    rw [Real.cos_add_nat_mul_two_pi, Real.cos_pi]
    norm_num

/-- `tan(π/4 + 2πn) = 1` and `tan(-π/4 + 2πn) = -1`, at points where `cos x ≠ 0`. -/
theorem signChanges_tan : SignChanges Real.tan {x | Real.cos x ≠ 0} := by
  have hc : 0 < Real.cos (Real.pi / 4) :=
    Real.cos_pos_of_mem_Ioo ⟨by linarith [Real.pi_pos], by linarith [Real.pi_pos]⟩
  refine ⟨fun x₀ => ?_, fun x₀ => ?_⟩
  · obtain ⟨n, hn⟩ := exists_add_nat_mul_two_pi_ge (Real.pi / 4) x₀
    refine ⟨_, ?_, hn, ?_⟩
    · change Real.cos (Real.pi / 4 + n * (2 * Real.pi)) ≠ 0
      rw [Real.cos_add_nat_mul_two_pi]
      exact hc.ne'
    · rw [Real.tan_eq_sin_div_cos, Real.sin_add_nat_mul_two_pi, Real.cos_add_nat_mul_two_pi,
        ← Real.tan_eq_sin_div_cos, Real.tan_pi_div_four]
      exact one_pos
  · obtain ⟨n, hn⟩ := exists_add_nat_mul_two_pi_ge (-(Real.pi / 4)) x₀
    refine ⟨_, ?_, hn, ?_⟩
    · change Real.cos (-(Real.pi / 4) + n * (2 * Real.pi)) ≠ 0
      rw [Real.cos_add_nat_mul_two_pi, Real.cos_neg]
      exact hc.ne'
    · rw [Real.tan_eq_sin_div_cos, Real.sin_add_nat_mul_two_pi, Real.cos_add_nat_mul_two_pi,
        ← Real.tan_eq_sin_div_cos, Real.tan_neg, Real.tan_pi_div_four]
      norm_num

/-- **Theorem 1b**, `sin`. No strict real term is equal to `sin x` for every real `x`. -/
theorem no_sin1b : ¬ ∃ W : Term 1, ∀ x : ℝ, evalR W ![x] = some (Real.sin x) :=
  fun ⟨W, hW⟩ => no_term_of_signChanges signChanges_sin ⟨W, fun x _ => hW x⟩

/-- **Theorem 1b**, `cos`. No strict real term is equal to `cos x` for every real `x`. -/
theorem no_cos1b : ¬ ∃ W : Term 1, ∀ x : ℝ, evalR W ![x] = some (Real.cos x) :=
  fun ⟨W, hW⟩ => no_term_of_signChanges signChanges_cos ⟨W, fun x _ => hW x⟩

/-- **Theorem 1b**, `tan`. No strict real term is equal to `tan x` at every real `x` with
`cos x ≠ 0`. -/
theorem no_tan1b :
    ¬ ∃ W : Term 1, ∀ x : ℝ, Real.cos x ≠ 0 → evalR W ![x] = some (Real.tan x) :=
  fun ⟨W, hW⟩ => no_term_of_signChanges signChanges_tan ⟨W, fun x hx => hW x hx⟩

end EML
