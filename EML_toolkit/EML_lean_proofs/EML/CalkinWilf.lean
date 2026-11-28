module

public import EML.Rationals

/-!
# Corollary 25 (c), second half: the Calkin–Wilf sequence

* `fusc` — Stern's diatomic sequence: `fusc 0 = 0`, `fusc 1 = 1`, `fusc (2n) = fusc n`,
  `fusc (2n+1) = fusc n + fusc (n+1)`.
* `cw n = fusc n / fusc (n+1)` — the Calkin–Wilf sequence `1, 1/2, 2, 1/3, 3/2, 2/3, 3, …`
  (for `n = 1, 2, 3, …`).
* `newman x = 1 / (2⌊x⌋ - x + 1)` — the map of M. Newman.
* `cw_succ` — `cw (n+1) = newman (cw n)` for `n ≥ 1` (Newman's formula).
* `exists_unique_cw` — every positive rational number is `cw n` for exactly one `n ≥ 1`
  (Calkin and Wilf).
* `exists_unique_newman_iterate` — every positive rational number occurs exactly once among
  `1, newman 1, newman (newman 1), …`.
* `eval_R_iterate`, `corollary25_c` — the closed terms `1, R(1), R(R(1)), …` of Corollary 25
  have the values `cw 1, cw 2, …`, and every positive rational number is the value of exactly
  one of them.

The paper cites this theorem; Mathlib does not have it, so it is proved here from the recursion
of `fusc`. No EML term occurs before the last section.
-/

@[expose] public section

namespace EML

/-- Stern's diatomic sequence. -/
def fusc : ℕ → ℕ
  | 0 => 0
  | 1 => 1
  | n + 2 => if n % 2 = 0 then fusc (n / 2 + 1) else fusc ((n + 1) / 2) + fusc ((n + 1) / 2 + 1)
decreasing_by all_goals omega

@[simp] theorem fusc_zero : fusc 0 = 0 := by rw [fusc]

@[simp] theorem fusc_one : fusc 1 = 1 := by rw [fusc]

/-- `fusc (2n) = fusc n`. -/
theorem fusc_even (n : ℕ) : fusc (2 * n) = fusc n := by
  cases n with
  | zero => simp
  | succ m =>
    have h : 2 * (m + 1) = 2 * m + 2 := by ring
    rw [h, fusc]
    have h1 : 2 * m % 2 = 0 := by omega
    have h2 : 2 * m / 2 + 1 = m + 1 := by omega
    simp only [h1, ↓reduceIte, h2]

/-- `fusc (2n+1) = fusc n + fusc (n+1)`. -/
theorem fusc_odd (n : ℕ) : fusc (2 * n + 1) = fusc n + fusc (n + 1) := by
  cases n with
  | zero => simp
  | succ m =>
    have h : 2 * (m + 1) + 1 = (2 * m + 1) + 2 := by ring
    rw [h, fusc]
    have h1 : ¬ (2 * m + 1) % 2 = 0 := by omega
    have h2 : (2 * m + 1 + 1) / 2 = m + 1 := by omega
    simp only [h1, ↓reduceIte, h2]

/-- `fusc n > 0` for `n ≥ 1`. -/
theorem fusc_pos {n : ℕ} (hn : 0 < n) : 0 < fusc n := by
  induction n using Nat.strong_induction_on with
  | _ n ih =>
    obtain ⟨m, rfl | rfl⟩ := Nat.even_or_odd' n
    · rw [fusc_even]
      exact ih m (by omega) (by omega)
    · rw [fusc_odd]
      have : 0 < fusc (m + 1) := by
        rcases Nat.eq_zero_or_pos m with rfl | hm
        · simp
        · exact ih (m + 1) (by omega) (by omega)
      omega

/-- Consecutive values of `fusc` are coprime. -/
theorem fusc_coprime (n : ℕ) : Nat.Coprime (fusc n) (fusc (n + 1)) := by
  induction n using Nat.strong_induction_on with
  | _ n ih =>
    obtain ⟨m, rfl | rfl⟩ := Nat.even_or_odd' n
    · rcases Nat.eq_zero_or_pos m with rfl | hm
      · simp
      · rw [fusc_even, fusc_odd]
        exact Nat.coprime_self_add_right.mpr (ih m (by omega))
    · have h : 2 * m + 1 + 1 = 2 * (m + 1) := by ring
      rw [h, fusc_even, fusc_odd]
      exact Nat.coprime_add_self_left.mpr (ih m (by omega))

/-- For `m ≥ 1`: `fusc (2m) < fusc (2m+1)`. -/
theorem fusc_even_lt {m : ℕ} (hm : 0 < m) : fusc (2 * m) < fusc (2 * m + 1) := by
  rw [fusc_even, fusc_odd]
  have : 0 < fusc (m + 1) := fusc_pos (by omega)
  omega

/-- `fusc (2m+2) ≤ fusc (2m+1)`. -/
theorem fusc_odd_ge (m : ℕ) : fusc (2 * m + 1 + 1) ≤ fusc (2 * m + 1) := by
  have h : 2 * m + 1 + 1 = 2 * (m + 1) := by ring
  rw [h, fusc_even, fusc_odd]
  omega

/-- Every pair of coprime positive integers is `(fusc n, fusc (n+1))` for some `n ≥ 1`. -/
theorem exists_fusc_pair (p q : ℕ) (hp : 0 < p) (hq : 0 < q) (h : Nat.Coprime p q) :
    ∃ n, 0 < n ∧ fusc n = p ∧ fusc (n + 1) = q := by
  induction hs : p + q using Nat.strong_induction_on generalizing p q with
  | _ s ih =>
    rcases Nat.lt_trichotomy p q with hlt | heq | hgt
    · -- p < q: the pair is the left child of (p, q - p)
      obtain ⟨d, rfl⟩ := Nat.exists_eq_add_of_lt hlt
      have hd : 0 < d + 1 := Nat.succ_pos d
      have hc : Nat.Coprime p (d + 1) := by
        have h' : Nat.Coprime p (p + (d + 1)) := by simpa [Nat.add_assoc] using h
        exact Nat.coprime_self_add_right.mp h'
      obtain ⟨m, hm, h1, h2⟩ := ih (p + (d + 1)) (by omega) p (d + 1) hp hd hc rfl
      refine ⟨2 * m, by omega, ?_, ?_⟩
      · rw [fusc_even, h1]
      · rw [fusc_odd, h1, h2]; omega
    · -- p = q: both are 1
      subst heq
      have hp1 : p = 1 := by simpa using h
      subst hp1
      refine ⟨1, one_pos, fusc_one, ?_⟩
      have : fusc (1 + 1) = fusc (2 * 1) := rfl
      rw [this, fusc_even, fusc_one]
    · -- p > q: the pair is the right child of (p - q, q)
      obtain ⟨d, rfl⟩ := Nat.exists_eq_add_of_lt hgt
      have hd : 0 < d + 1 := Nat.succ_pos d
      have hc : Nat.Coprime (d + 1) q := by
        have h' : Nat.Coprime ((d + 1) + q) q := by
          have e : q + d + 1 = (d + 1) + q := by omega
          rwa [e] at h
        exact Nat.coprime_add_self_left.mp h'
      obtain ⟨m, hm, h1, h2⟩ := ih ((d + 1) + q) (by omega) (d + 1) q hd hq hc rfl
      refine ⟨2 * m + 1, by omega, ?_, ?_⟩
      · rw [fusc_odd, h1, h2]; omega
      · have e : 2 * m + 1 + 1 = 2 * (m + 1) := by ring
        rw [e, fusc_even, h2]

/-- The pairs `(fusc n, fusc (n+1))`, `n ≥ 1`, are all different. -/
theorem fusc_pair_injective {n n' : ℕ} (hn : 0 < n) (hn' : 0 < n') (h1 : fusc n = fusc n')
    (h2 : fusc (n + 1) = fusc (n' + 1)) : n = n' := by
  induction n using Nat.strong_induction_on generalizing n' with
  | _ n ih =>
    obtain ⟨m, rfl | rfl⟩ := Nat.even_or_odd' n <;> obtain ⟨m', rfl | rfl⟩ := Nat.even_or_odd' n'
    · -- both even
      have hm : 0 < m := by omega
      have hm' : 0 < m' := by omega
      rw [fusc_even, fusc_even] at h1
      rw [fusc_odd, fusc_odd] at h2
      have := ih m (by omega) hm hm' h1 (by omega)
      omega
    · -- even and odd: fusc n < fusc (n+1), but fusc n' ≥ fusc (n'+1)
      have hm : 0 < m := by omega
      have := fusc_even_lt hm
      have := fusc_odd_ge m'
      omega
    · -- odd and even
      have hm' : 0 < m' := by omega
      have := fusc_even_lt hm'
      have := fusc_odd_ge m
      omega
    · -- both odd
      have e : 2 * m + 1 + 1 = 2 * (m + 1) := by ring
      have e' : 2 * m' + 1 + 1 = 2 * (m' + 1) := by ring
      rw [fusc_odd, fusc_odd] at h1
      rw [e, e', fusc_even, fusc_even] at h2
      have h0 : fusc m = fusc m' := by omega
      rcases Nat.eq_zero_or_pos m with rfl | hm
      · rcases Nat.eq_zero_or_pos m' with rfl | hm'
        · rfl
        · have := fusc_pos hm'
          simp at h0
          omega
      · rcases Nat.eq_zero_or_pos m' with rfl | hm'
        · have := fusc_pos hm
          simp at h0
          omega
        · have := ih m (by omega) hm hm' h0 h2
          omega

/-- The Calkin–Wilf sequence: `cw n = fusc n / fusc (n+1)`. -/
def cw (n : ℕ) : ℚ := (fusc n : ℚ) / (fusc (n + 1) : ℚ)

/-- The map of M. Newman: `x ↦ 1 / (2⌊x⌋ - x + 1)`. -/
def newman (x : ℚ) : ℚ := 1 / (2 * (⌊x⌋ : ℚ) - x + 1)

/-- `fusc 2 = 1` and `fusc 3 = 2`. -/
theorem fusc_two : fusc 2 = 1 := by
  have h : fusc (2 * 1) = fusc 1 := fusc_even 1
  simpa using h

theorem fusc_three : fusc 3 = 2 := by
  have h : fusc (2 * 1 + 1) = fusc 1 + fusc (1 + 1) := fusc_odd 1
  simpa [fusc_two] using h

/-- Newman's map at a left child of the Calkin–Wilf tree: `a/(a+b) ↦ (a+b)/b`. -/
theorem newman_left {a b : ℚ} (ha : 0 < a) (hb : 0 < b) : newman (a / (a + b)) = (a + b) / b := by
  have hab : a + b ≠ 0 := by positivity
  have hfl : ⌊a / (a + b)⌋ = 0 := by
    rw [Int.floor_eq_zero_iff]
    constructor
    · positivity
    · rw [div_lt_one (by positivity)]; linarith
  have h1 : 2 * ((0 : ℤ) : ℚ) - a / (a + b) + 1 = b / (a + b) := by
    rw [eq_div_iff hab]
    have h2 := div_mul_cancel₀ a hab
    push_cast
    linear_combination (-1) * h2
  rw [newman, hfl, h1, one_div, inv_div]

/-- Newman's map at a right child: if it sends `a/b` to `b/c`, it sends `(a+b)/b` to `b/(b+c)`. -/
theorem newman_right {a b c : ℚ} (hb : 0 < b) (hc : 0 < c) (h : newman (a / b) = b / c) :
    newman ((a + b) / b) = b / (b + c) := by
  have hb' := hb.ne'
  have hc' := hc.ne'
  have hbc : b + c ≠ 0 := by positivity
  have hfl : ⌊(a + b) / b⌋ = ⌊a / b⌋ + 1 := by
    rw [add_div, div_self hb', Int.floor_add_one]
  rw [newman] at h ⊢
  rw [hfl]
  push_cast
  -- with t = 2⌊a/b⌋ - a/b + 1: 1/t = b/c, hence t = c/b
  have ht0 : 2 * (⌊a / b⌋ : ℚ) - a / b + 1 ≠ 0 := by
    intro h0
    rw [h0, div_zero] at h
    have := div_pos hb hc
    linarith
  have ht : 2 * (⌊a / b⌋ : ℚ) - a / b + 1 = c / b := by
    rw [div_eq_div_iff ht0 hc'] at h
    rw [eq_div_iff hb']
    linarith
  have hgoal : 2 * ((⌊a / b⌋ : ℚ) + 1) - (a + b) / b + 1 = (b + c) / b := by
    have : (a + b) / b = a / b + 1 := by rw [add_div, div_self hb']
    rw [this, add_div, div_self hb', ← ht]
    ring
  rw [hgoal, one_div, inv_div]

/-- **Newman's formula.** For `n ≥ 1`: `cw (n+1) = newman (cw n)`. -/
theorem cw_succ {n : ℕ} (hn : 0 < n) : cw (n + 1) = newman (cw n) := by
  induction n using Nat.strong_induction_on with
  | _ n ih =>
    obtain ⟨m, rfl | rfl⟩ := Nat.even_or_odd' n
    · -- n = 2m: cw n = a/(a+b) and cw (n+1) = (a+b)/b
      have hm : 0 < m := by omega
      have ha : (0 : ℚ) < fusc m := by exact_mod_cast fusc_pos hm
      have hb : (0 : ℚ) < fusc (m + 1) := by exact_mod_cast fusc_pos (Nat.succ_pos m)
      have e : 2 * m + 1 + 1 = 2 * (m + 1) := by ring
      have h := newman_left ha hb
      simp only [cw, e, fusc_even, fusc_odd]
      push_cast
      exact h.symm
    · rcases Nat.eq_zero_or_pos m with rfl | hm
      · -- n = 1: cw 1 = 1, cw 2 = 1/2
        simp [cw, newman, fusc_two, fusc_three]
      · -- n = 2m+1: cw n = (a+b)/b, cw (n+1) = b/(b+c); the hypothesis at m: newman (a/b) = b/c
        have hb : (0 : ℚ) < fusc (m + 1) := by exact_mod_cast fusc_pos (Nat.succ_pos m)
        have hc : (0 : ℚ) < fusc (m + 1 + 1) := by
          exact_mod_cast fusc_pos (Nat.succ_pos (m + 1))
        have hih := ih m (by omega) hm
        simp only [cw] at hih
        have e : 2 * m + 1 + 1 = 2 * (m + 1) := by ring
        have h := newman_right (a := (fusc m : ℚ)) hb hc hih.symm
        simp only [cw, e, fusc_even, fusc_odd]
        push_cast
        exact h.symm

/-- `cw n > 0` for `n ≥ 1`. -/
theorem cw_pos {n : ℕ} (hn : 0 < n) : 0 < cw n := by
  have ha : (0 : ℚ) < fusc n := by exact_mod_cast fusc_pos hn
  have hb : (0 : ℚ) < fusc (n + 1) := by exact_mod_cast fusc_pos (Nat.succ_pos n)
  exact div_pos ha hb

/-- The values `cw n`, `n ≥ 1`, are all different. -/
theorem cw_injective {n n' : ℕ} (hn : 0 < n) (hn' : 0 < n') (h : cw n = cw n') : n = n' := by
  have hb : (0 : ℤ) < (fusc (n + 1) : ℤ) := by exact_mod_cast fusc_pos (Nat.succ_pos n)
  have hd : (0 : ℤ) < (fusc (n' + 1) : ℤ) := by exact_mod_cast fusc_pos (Nat.succ_pos n')
  have h' : (((fusc n : ℤ)) : ℚ) / (((fusc (n + 1) : ℤ)) : ℚ)
      = (((fusc n' : ℤ)) : ℚ) / (((fusc (n' + 1) : ℤ)) : ℚ) := by
    simpa [cw] using h
  obtain ⟨h1, h2⟩ := Rat.div_int_inj hb hd (by simpa using fusc_coprime n)
    (by simpa using fusc_coprime n') h'
  exact fusc_pair_injective hn hn' (by exact_mod_cast h1) (by exact_mod_cast h2)

/-- **Calkin and Wilf.** Every positive rational number is `cw n` for exactly one `n ≥ 1`. -/
theorem exists_unique_cw (r : ℚ) (hr : 0 < r) : ∃! n : ℕ, 0 < n ∧ cw n = r := by
  have hnum : 0 < r.num := Rat.num_pos.mpr hr
  obtain ⟨n, hn, h1, h2⟩ := exists_fusc_pair r.num.natAbs r.den (Int.natAbs_pos.mpr hnum.ne')
    r.den_pos r.reduced
  have hcw : cw n = r := by
    have habs : ((r.num.natAbs : ℕ) : ℚ) = (r.num : ℚ) := by
      rw [Nat.cast_natAbs, abs_of_pos hnum]
    rw [cw, h1, h2, habs]
    exact Rat.num_div_den r
  refine ⟨n, ⟨hn, hcw⟩, ?_⟩
  rintro n' ⟨hn', h'⟩
  exact cw_injective hn' hn (h'.trans hcw.symm)

/-- The orbit of `1` under Newman's map is the Calkin–Wilf sequence. -/
theorem newman_iterate (k : ℕ) : newman^[k] 1 = cw (k + 1) := by
  induction k with
  | zero => simp [cw, fusc_two]
  | succ k ih => rw [Function.iterate_succ_apply', ih, ← cw_succ (Nat.succ_pos k)]

/-- Every positive rational number occurs exactly once in the sequence `1, newman 1,
newman (newman 1), …`. -/
theorem exists_unique_newman_iterate (r : ℚ) (hr : 0 < r) : ∃! k : ℕ, newman^[k] 1 = r := by
  obtain ⟨n, ⟨hn, hcw⟩, huniq⟩ := exists_unique_cw r hr
  obtain ⟨k, rfl⟩ := Nat.exists_eq_succ_of_ne_zero hn.ne'
  refine ⟨k, ?_, ?_⟩
  · show newman^[k] 1 = r
    rw [newman_iterate]
    exact hcw
  · intro k' hk'
    have h : newman^[k'] 1 = r := hk'
    rw [newman_iterate] at h
    have := huniq (k' + 1) ⟨Nat.succ_pos k', h⟩
    omega

/-! ## The closed terms `1, R(1), R(R(1)), …` -/

open Complex Term

variable {n : ℕ} {a : Fin n → ℂ}

/-- The closed term `R(R(…R(1)…))` with `k` applications of `R` has the value `cw (k+1)`. -/
theorem eval_R_iterate (k : ℕ) :
    eval (R^[k] (one : Term n)) a = some (((cw (k + 1) : ℚ) : ℝ) : ℂ) := by
  induction k with
  | zero => simp [cw, fusc_two]
  | succ k ih =>
    have hpos : (0 : ℝ) < ((cw (k + 1) : ℚ) : ℝ) := by exact_mod_cast cw_pos (Nat.succ_pos k)
    rw [Function.iterate_succ_apply']
    refine eval_congr (eval_R ih (by linarith)) ?_
    rw [cw_succ (Nat.succ_pos k), newman, Rat.floor_cast]
    push_cast
    rw [one_div]

/-- **Corollary 25 (c).** Every positive rational number is the value of exactly one of the
closed terms `1, R(1), R(R(1)), …`. -/
theorem corollary25_c (r : ℚ) (hr : 0 < r) :
    ∃! k : ℕ, eval (R^[k] (one : Term n)) a = some ((r : ℝ) : ℂ) := by
  obtain ⟨k, hk, huniq⟩ := exists_unique_newman_iterate r hr
  refine ⟨k, ?_, ?_⟩
  · show eval (R^[k] (one : Term n)) a = some ((r : ℝ) : ℂ)
    rw [eval_R_iterate, ← newman_iterate, hk]
  · intro k' hk'
    have h : eval (R^[k'] (one : Term n)) a = some ((r : ℝ) : ℂ) := hk'
    rw [eval_R_iterate, Option.some.injEq] at h
    apply huniq
    show newman^[k'] 1 = r
    rw [newman_iterate]
    exact_mod_cast h

end EML

#print axioms EML.exists_unique_newman_iterate
#print axioms EML.corollary25_c
