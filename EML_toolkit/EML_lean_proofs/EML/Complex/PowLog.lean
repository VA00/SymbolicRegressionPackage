module

public import EML.Complex.Arith

/-!
# Proof of Theorem 1a, Step 3 (Lemmas A.7, A.9, A.10, Corollary A.8)

Power, square root, hypotenuse, logarithm and logarithm to a base, on the whole complex plane.
`z ^ w` is Mathlib's complex power: `z ^ w = exp (w Log z)` for `z ≠ 0`, and `0 ^ w = 0`
for `w ≠ 0`. The square root is `z ^ (1/2)`.
-/

@[expose] public section

namespace EML

open Complex Term XC

variable {n : ℕ} {a : Fin n → ℂ} {s t : Term n} {z w b : ℂ}

/-- Table A.1, row `∞₋`: `EML(∞₋, z) = -Log z` for `z ≠ 0`. -/
theorem evalX_eml_ninf (ht : evalX t a = some (num z)) (hz : z ≠ 0) :
    evalX (eml const_ninf t) a = some (num (-log z)) := by
  rw [evalX_eml evalX_const_ninf ht]
  simp [EMLop, hz]

/-- Table A.1, row `∞₋`: `EML(∞₋, 0) = ∞₊`. -/
theorem evalX_eml_ninf_zero (ht : evalX t a = some (num 0)) :
    evalX (eml const_ninf t) a = some pinf := by
  rw [evalX_eml evalX_const_ninf ht]
  simp [EMLop]

/-- In the proof of Lemma A.7 (b): `W×(w, ∞₊) = ∞₊` for `w ≠ 0`. -/
theorem evalX_Wmul1a_pinf (hs : evalX s a = some (num w)) (hw : w ≠ 0)
    (ht : evalX t a = some pinf) : evalX (Wmul1a s t) a = some pinf := by
  have h1 := evalX_eml_ninf hs hw
  -- `EML(L(-Log w), ∞₊) = ∞₋`, whether `L(-Log w)` is a number or `∞₋` (`w = 1`)
  have h2 : evalX (eml (L (eml const_ninf s)) t) a = some ninf := by
    by_cases hl : -log w = 0
    · rw [hl] at h1
      rw [evalX_eml (evalX_L_zero h1) ht]
      rfl
    · rw [evalX_eml (evalX_L h1 hl) ht]
      rfl
  -- `W_inv(E(∞₋)) = W_inv(0) = ∞₊`
  exact evalX_Winv1a_zero (evalX_E_ninf h2)

/-- `W_pow(z, w) := W_inv(E(W×(w, EML(∞₋, z))))`. -/
def Wpow1a (s t : Term n) : Term n := Winv1a (E (Wmul1a t (eml const_ninf s)))

/-- **Lemma A.7 (a)**: `W_pow(z, w) = z^w = exp(w Log z)` for every `z ≠ 0` and every `w ∈ ℂ`. -/
theorem evalX_Wpow1a (hs : evalX s a = some (num z)) (ht : evalX t a = some (num w))
    (hz : z ≠ 0) : evalX (Wpow1a s t) a = some (num (z ^ w)) := by
  -- `W×(w, -Log z) = -w Log z`, then `W_inv(exp(-w Log z)) = exp(w Log z)`
  have h1 := evalX_E (evalX_Wmul1a ht (evalX_eml_ninf hs hz))
  refine evalX_congr (evalX_Winv1a h1 (exp_ne_zero _)) ?_
  congr 1
  rw [← exp_neg, cpow_def_of_ne_zero hz]
  congr 1
  ring

/-- **Lemma A.7 (b)**: `W_pow(0, w) = 0` for every `w ≠ 0`. -/
theorem evalX_Wpow1a_zero (hs : evalX s a = some (num 0)) (ht : evalX t a = some (num w))
    (hw : w ≠ 0) : evalX (Wpow1a s t) a = some (num 0) :=
  -- `EML(∞₋, 0) = ∞₊`, `W×(w, ∞₊) = ∞₊`, `E(∞₊) = ∞₊`, `W_inv(∞₊) = 0`
  evalX_Winv1a_pinf (evalX_E_pinf (evalX_Wmul1a_pinf ht hw (evalX_eml_ninf_zero hs)))

/-- Lemma A.7 on the domain of `pow` in Table A.2: `z ≠ 0`, or `z = 0` and `Re w > 0`. -/
theorem evalX_Wpow1a_of_mem (hs : evalX s a = some (num z)) (ht : evalX t a = some (num w))
    (h : z ≠ 0 ∨ 0 < w.re) : evalX (Wpow1a s t) a = some (num (z ^ w)) := by
  by_cases hz : z = 0
  · subst hz
    have hw : w ≠ 0 := by
      rintro rfl
      simp at h
    exact evalX_congr (evalX_Wpow1a_zero hs ht hw) (by rw [zero_cpow hw])
  · exact evalX_Wpow1a hs ht hz

/-- `W_√(z) := W_pow(z, 1/2)`. -/
def Wsqrt1a (t : Term n) : Term n := Wpow1a t const_half

/-- **Corollary A.8**: `W_√(z) = z^(1/2)` for every `z ∈ ℂ`, which is `exp(½ Log z)` for
`z ≠ 0` (Lemma A.7 (a)) and `0` for `z = 0` (Lemma A.7 (b)). -/
theorem evalX_Wsqrt1a (ht : evalX t a = some (num z)) :
    evalX (Wsqrt1a t) a = some (num (z ^ (1 / 2 : ℂ))) :=
  evalX_Wpow1a_of_mem ht evalX_const_half (Or.inr (by norm_num))

/-- `W_hypot(z, w) := W_√(W₊(W_sqr(z), W_sqr(w)))`. -/
def Whypot1a (s t : Term n) : Term n := Wsqrt1a (Wadd1a (Wsqr1a s) (Wsqr1a t))

/-- **Corollary A.8**: `W_hypot(z, w) = (z² + w²)^(1/2)` for all `z, w ∈ ℂ`. -/
theorem evalX_Whypot1a (hs : evalX s a = some (num z)) (ht : evalX t a = some (num w)) :
    evalX (Whypot1a s t) a = some (num ((z ^ 2 + w ^ 2) ^ (1 / 2 : ℂ))) :=
  evalX_Wsqrt1a (evalX_Wadd1a (evalX_Wsqr1a hs) (evalX_Wsqr1a ht))

/-- `W_ln(z) := W_minus(EML(∞₋, z))`. -/
def Wln1a (t : Term n) : Term n := Wminus1a (eml const_ninf t)

/-- **Lemma A.9**: `W_ln(z) = Log z` for every `z ≠ 0`. -/
theorem evalX_Wln1a (ht : evalX t a = some (num z)) (hz : z ≠ 0) :
    evalX (Wln1a t) a = some (num (log z)) :=
  evalX_congr (evalX_Wminus1a (evalX_eml_ninf ht hz)) (by rw [neg_neg])

/-- `W_log(b, z) := W/(W_ln(z), W_ln(b))`; the first argument is the base. -/
def Wlog1a (s t : Term n) : Term n := Wdiv1a (Wln1a t) (Wln1a s)

/-- **Lemma A.10**: `W_log(b, z) = Log z / Log b = log_b z` for every `b ∉ {0, 1}` and
every `z ≠ 0`. -/
theorem evalX_Wlog1a (hs : evalX s a = some (num b)) (ht : evalX t a = some (num z))
    (hb0 : b ≠ 0) (hb1 : b ≠ 1) (hz : z ≠ 0) :
    evalX (Wlog1a s t) a = some (num (log z / log b)) := by
  -- `Log b ≠ 0`, because `b ≠ 1`
  have hlog : log b ≠ 0 := by
    intro h
    apply hb1
    rw [← exp_log hb0, h, exp_zero]
  exact evalX_Wdiv1a (evalX_Wln1a ht hz) (evalX_Wln1a hs hb0) hlog

end EML
