module

public import EML.Complex.Semantics

/-!
# Proof of Theorem 1a, Steps 1–2 (Lemmas A.1–A.5, Corollary A.6)

The two infinities, the auxiliary terms `E`, `L`, `S` at `0` and `∞±`, reciprocal,
multiplication, subtraction and the remaining arithmetic, all on the whole complex plane.

The witnesses of Theorem 1a carry the suffix `1a` (`Winv1a`, `Wmul1a`, ...; the paper writes
them `𝒲`). Names without the suffix (`Winv`, `Wmul`, ...) are the witnesses of Theorem 1.
The terms `E`, `L`, `S` and the closed terms of Lemmas 6 and 8 are those of Section 2.
-/

@[expose] public section

namespace EML

open Complex Term XC

variable {n : ℕ} {a : Fin n → ℂ} {s t : Term n} {u v z w : ℂ}

/-! ## Start of the proof: the closed terms of Corollary 2 and Lemmas 6, 8 keep their values -/

theorem evalX_const_e : evalX (const_e : Term n) a = some (num (Real.exp 1)) :=
  evalX_of_eval eval_const_e

theorem evalX_const_0 : evalX (const_0 : Term n) a = some (num 0) := evalX_of_eval eval_const_0

theorem evalX_const_neg : evalX (const_neg : Term n) a = some (num (-1)) :=
  evalX_of_eval eval_const_neg

theorem evalX_const_2 : evalX (const_2 : Term n) a = some (num 2) := evalX_of_eval eval_const_2

theorem evalX_const_half : evalX (const_half : Term n) a = some (num (1 / 2)) :=
  evalX_of_eval eval_const_half

theorem evalX_const_i : evalX (const_i : Term n) a = some (num I) := evalX_of_eval eval_const_i

theorem evalX_const_pi : evalX (const_pi : Term n) a = some (num Real.pi) :=
  evalX_of_eval eval_const_pi

/-- Lemma 1 in Appendix A: `E(z) = exp z` for every `z ∈ ℂ`. -/
theorem evalX_E (ht : evalX t a = some (num z)) : evalX (E t) a = some (num (exp z)) := by
  rw [E, evalX_eml ht evalX_one, EMLop_num one_ne_zero, log_one, sub_zero]

/-- Lemma 3 in Appendix A: `L(z)` for `z ≠ 0`, the same value as in Section 2. -/
theorem evalX_L (ht : evalX t a = some (num z)) (hz : z ≠ 0) :
    evalX (L t) a = some (num (Lval z)) := by
  have h1 : evalX (eml one t) a = some (num (exp 1 - log z)) := by
    rw [evalX_eml evalX_one ht, EMLop_num hz]
  rw [L, evalX_eml evalX_one (evalX_E h1), EMLop_num (exp_ne_zero _), Lval_eq]

/-! ## Step 1: the two infinities, reciprocal and multiplication -/

/-- `∞₊ := EML(1, 0)`, where `0` is the closed term `L(1)` of Lemma 6. -/
def const_pinf : Term n := eml one const_0
/-- `∞₋ := EML(1, ∞₊)`. -/
def const_ninf : Term n := eml one const_pinf

/-- **Lemma A.1**: the closed term `EML(1, 0)` has the value `∞₊`. -/
theorem evalX_const_pinf : evalX (const_pinf : Term n) a = some pinf := by
  rw [const_pinf, evalX_eml evalX_one evalX_const_0]
  simp [EMLop]

/-- **Lemma A.1**: the closed term `EML(1, ∞₊)` has the value `∞₋`. -/
theorem evalX_const_ninf : evalX (const_ninf : Term n) a = some ninf := by
  rw [const_ninf, evalX_eml evalX_one evalX_const_pinf]
  rfl

/-- **Lemma A.2 (a)**: `E(∞₋) = 0`. -/
theorem evalX_E_ninf (ht : evalX t a = some ninf) : evalX (E t) a = some (num 0) := by
  rw [E, evalX_eml ht evalX_one]
  simp [EMLop]

/-- **Lemma A.2 (a)**: `E(∞₊) = ∞₊`. -/
theorem evalX_E_pinf (ht : evalX t a = some pinf) : evalX (E t) a = some pinf := by
  rw [E, evalX_eml ht evalX_one]
  rfl

/-- **Lemma A.2 (b)**: `L(0) = ∞₋`. -/
theorem evalX_L_zero (ht : evalX t a = some (num 0)) : evalX (L t) a = some ninf := by
  have h1 : evalX (eml one t) a = some pinf := by
    rw [evalX_eml evalX_one ht]
    simp [EMLop]
  rw [L, evalX_eml evalX_one (evalX_E_pinf h1)]
  rfl

/-- **Lemma A.2 (b)**: `L(∞₋) = ∞₊`. -/
theorem evalX_L_ninf (ht : evalX t a = some ninf) : evalX (L t) a = some pinf := by
  have h1 : evalX (eml one t) a = some ninf := by
    rw [evalX_eml evalX_one ht]
    rfl
  rw [L, evalX_eml evalX_one (evalX_E_ninf h1)]
  simp [EMLop]

/-- **Lemma A.2 (b)**: `L(∞₊) = ∞₊`. -/
theorem evalX_L_pinf (ht : evalX t a = some pinf) : evalX (L t) a = some pinf := by
  have h1 : evalX (eml one t) a = some ninf := by
    rw [evalX_eml evalX_one ht]
    rfl
  rw [L, evalX_eml evalX_one (evalX_E_ninf h1)]
  simp [EMLop]

/-- **Lemma A.2 (b)**, consequence: `exp(L(u)) = u` for every `u ∈ ℂ`, including `u = 0`,
where `L(0) = ∞₋` and `exp ∞₋ = 0`. Written for a node: `EML(L(u), w) = u - Log w`
for every `u ∈ ℂ` and every `w ≠ 0`. -/
theorem evalX_eml_L (hs : evalX s a = some (num u)) (ht : evalX t a = some (num w)) (hw : w ≠ 0) :
    evalX (eml (L s) t) a = some (num (u - log w)) := by
  by_cases hu : u = 0
  · subst hu
    rw [evalX_eml (evalX_L_zero hs) ht]
    simp [EMLop, hw]
  · rw [evalX_eml (evalX_L hs hu) ht, EMLop_num hw, exp_Lval hu]

/-- The unwinding number `K(w) = ⌈(Im w - π) / (2π)⌉` of (F2). -/
noncomputable def unw (w : ℂ) : ℤ := ⌈(w.im - Real.pi) / (2 * Real.pi)⌉

/-- (F2) written with `unw`: `Log(exp w) = w - 2πi K(w)`. -/
theorem log_exp_unw (w : ℂ) : log (exp w) = w - 2 * Real.pi * I * unw w := by
  rw [F2 w, unw]
  ring

/-- The unwinding number of a real number is `0`. -/
theorem unw_of_im_eq_zero (h : w.im = 0) : unw w = 0 := by
  rw [unw, h, Int.ceil_eq_iff]
  have hpi : (0 : ℝ) < Real.pi := Real.pi_pos
  constructor
  · rw [lt_div_iff₀ (by positivity)]
    push_cast
    linarith
  · rw [div_le_iff₀ (by positivity)]
    push_cast
    linarith

/-- **Lemma A.2 (c)**: for all `u, v ∈ ℂ` the term `S(u, v)` is defined and
`S(u, v) = u - v + 2πi K(v)`. -/
theorem evalX_S (hs : evalX s a = some (num u)) (ht : evalX t a = some (num v)) :
    evalX (S s t) a = some (num (u - v + 2 * Real.pi * I * unw v)) := by
  rw [S, evalX_eml_L hs (evalX_E ht) (exp_ne_zero v), log_exp_unw]
  ring_nf

/-- **Lemma A.2 (c)**, last claim: `S(v, v) = 2πi K(v)`. -/
theorem evalX_S_self (ht : evalX t a = some (num v)) :
    evalX (S t t) a = some (num (2 * Real.pi * I * unw v)) :=
  evalX_congr (evalX_S ht ht) (by ring_nf)

/-- `W_inv(z) := E(EML(∞₋, z))`. -/
def Winv1a (t : Term n) : Term n := E (eml const_ninf t)

/-- **Lemma A.3**: `W_inv(z) = 1/z` for every `z ≠ 0`. -/
theorem evalX_Winv1a (ht : evalX t a = some (num z)) (hz : z ≠ 0) :
    evalX (Winv1a t) a = some (num z⁻¹) := by
  -- `EML(∞₋, z) = -Log z`
  have h1 : evalX (eml const_ninf t) a = some (num (-log z)) := by
    rw [evalX_eml evalX_const_ninf ht]
    simp [EMLop, hz]
  rw [Winv1a, evalX_E h1, exp_neg, exp_log hz]

/-- **Lemma A.3**: `W_inv(0) = ∞₊`. -/
theorem evalX_Winv1a_zero (ht : evalX t a = some (num 0)) : evalX (Winv1a t) a = some pinf := by
  have h1 : evalX (eml const_ninf t) a = some pinf := by
    rw [evalX_eml evalX_const_ninf ht]
    simp [EMLop]
  rw [Winv1a, evalX_E_pinf h1]

/-- **Lemma A.3**: `W_inv(∞₋) = 0`. -/
theorem evalX_Winv1a_ninf (ht : evalX t a = some ninf) : evalX (Winv1a t) a = some (num 0) := by
  have h1 : evalX (eml const_ninf t) a = some ninf := by
    rw [evalX_eml evalX_const_ninf ht]
    rfl
  rw [Winv1a, evalX_E_ninf h1]

/-- **Lemma A.3**: `W_inv(∞₊) = 0`. -/
theorem evalX_Winv1a_pinf (ht : evalX t a = some pinf) : evalX (Winv1a t) a = some (num 0) := by
  have h1 : evalX (eml const_ninf t) a = some ninf := by
    rw [evalX_eml evalX_const_ninf ht]
    rfl
  rw [Winv1a, evalX_E_ninf h1]

/-- `W×(z, w) := W_inv(E(EML(L(EML(∞₋, z)), w)))`. -/
def Wmul1a (s t : Term n) : Term n := Winv1a (E (eml (L (eml const_ninf s)) t))

/-- **Lemma A.4**: for all `z, w ∈ ℂ` the term `W×(z, w)` is defined and `W×(z, w) = z w`. -/
theorem evalX_Wmul1a (hs : evalX s a = some (num z)) (ht : evalX t a = some (num w)) :
    evalX (Wmul1a s t) a = some (num (z * w)) := by
  by_cases hz : z = 0
  · -- `z = 0`: `EML(∞₋, 0) = ∞₊`, `L(∞₊) = ∞₊`, `EML(∞₊, w) = ∞₊`, and `W_inv(E(∞₊)) = 0`
    subst hz
    have h1 : evalX (eml const_ninf s) a = some pinf := by
      rw [evalX_eml evalX_const_ninf hs]
      simp [EMLop]
    have h2 : evalX (eml (L (eml const_ninf s)) t) a = some pinf := by
      rw [evalX_eml (evalX_L_pinf h1) ht]
      rfl
    exact evalX_congr (evalX_Winv1a_pinf (evalX_E_pinf h2)) (by simp)
  · -- `z ≠ 0`: `EML(∞₋, z) = -Log z`
    have h1 : evalX (eml const_ninf s) a = some (num (-log z)) := by
      rw [evalX_eml evalX_const_ninf hs]
      simp [EMLop, hz]
    by_cases hw : w = 0
    · -- `w = 0`: `EML(L(-Log z), 0) = ∞₊`, whether `L(-Log z)` is a number or `∞₋`
      subst hw
      have h2 : evalX (eml (L (eml const_ninf s)) t) a = some pinf := by
        by_cases hl : -log z = 0
        · rw [hl] at h1
          rw [evalX_eml (evalX_L_zero h1) ht]
          simp [EMLop]
        · rw [evalX_eml (evalX_L h1 hl) ht]
          simp [EMLop]
      exact evalX_congr (evalX_Winv1a_pinf (evalX_E_pinf h2)) (by simp)
    · -- `z, w ≠ 0`: `EML(L(-Log z), w) = -Log z - Log w`, and `W_inv(exp(-Log z - Log w)) = z w`
      have h2 := evalX_E (evalX_eml_L h1 ht hw)
      refine evalX_congr (evalX_Winv1a h2 (exp_ne_zero _)) ?_
      congr 1
      rw [← exp_neg, show -(-log z - log w) = log z + log w by ring, exp_add, exp_log hz,
        exp_log hw]

/-! ## Step 2: arithmetic for all complex arguments -/

/-- `-i := W×(-1, i)`. -/
def const_minus_i : Term n := Wmul1a const_neg const_i

/-- The closed term `W×(-1, i)` has the value `-i`. -/
theorem evalX_const_minus_i : evalX (const_minus_i : Term n) a = some (num (-I)) :=
  evalX_congr (evalX_Wmul1a evalX_const_neg evalX_const_i) (by ring_nf)

/-- `W₋(z, w) := W×(-i, S(W×(i, S(z, w)), W×(i, S(w, w))))`. -/
def Wsub1a (s t : Term n) : Term n :=
  Wmul1a const_minus_i (S (Wmul1a const_i (S s t)) (Wmul1a const_i (S t t)))

/-- **Lemma A.5**: for all `z, w ∈ ℂ` the term `W₋(z, w)` is defined and `W₋(z, w) = z - w`. -/
theorem evalX_Wsub1a (hs : evalX s a = some (num z)) (ht : evalX t a = some (num w)) :
    evalX (Wsub1a s t) a = some (num (z - w)) := by
  -- `S(z, w) = z - w + 2πi K(w)` and `S(w, w) = 2πi K(w)` (Lemma A.2 (c))
  have h1 := evalX_Wmul1a evalX_const_i (evalX_S hs ht)
  have h2 := evalX_Wmul1a evalX_const_i (evalX_S_self ht)
  -- `W×(i, S(w, w)) = -2π K(w)` is real, so its unwinding number is `0`
  have hK : unw (I * (2 * Real.pi * I * unw w)) = 0 := by
    apply unw_of_im_eq_zero
    simp
  have h3 := evalX_Wmul1a evalX_const_minus_i (evalX_S h1 h2)
  refine evalX_congr h3 ?_
  rw [hK]
  congr 1
  linear_combination (-(z - w)) * I_sq

/-- **Corollary A.6**: `W_minus(z) := W×(-1, z)`. -/
def Wminus1a (t : Term n) : Term n := Wmul1a const_neg t
/-- **Corollary A.6**: `W₊(z, w) := W₋(z, W_minus(w))`. -/
def Wadd1a (s t : Term n) : Term n := Wsub1a s (Wminus1a t)
/-- **Corollary A.6**: `W_sqr(z) := W×(z, z)`. -/
def Wsqr1a (t : Term n) : Term n := Wmul1a t t
/-- **Corollary A.6**: `W_half(z) := W×(1/2, z)`. -/
def Whalf1a (t : Term n) : Term n := Wmul1a const_half t
/-- **Corollary A.6**: `W_avg(z, w) := W_half(W₊(z, w))`. -/
def Wavg1a (s t : Term n) : Term n := Whalf1a (Wadd1a s t)
/-- **Corollary A.6**: `W/(z, w) := W×(z, W_inv(w))`. -/
def Wdiv1a (s t : Term n) : Term n := Wmul1a s (Winv1a t)

/-- **Corollary A.6**: `W_minus(z) = -z` for every `z ∈ ℂ`. -/
theorem evalX_Wminus1a (ht : evalX t a = some (num z)) : evalX (Wminus1a t) a = some (num (-z)) :=
  evalX_congr (evalX_Wmul1a evalX_const_neg ht) (by ring_nf)

/-- **Corollary A.6**: `W₊(z, w) = z + w` for all `z, w ∈ ℂ`. -/
theorem evalX_Wadd1a (hs : evalX s a = some (num z)) (ht : evalX t a = some (num w)) :
    evalX (Wadd1a s t) a = some (num (z + w)) :=
  evalX_congr (evalX_Wsub1a hs (evalX_Wminus1a ht)) (by ring_nf)

/-- **Corollary A.6**: `W_sqr(z) = z²` for every `z ∈ ℂ`. -/
theorem evalX_Wsqr1a (ht : evalX t a = some (num z)) : evalX (Wsqr1a t) a = some (num (z ^ 2)) :=
  evalX_congr (evalX_Wmul1a ht ht) (by ring_nf)

/-- **Corollary A.6**: `W_half(z) = z/2` for every `z ∈ ℂ`. -/
theorem evalX_Whalf1a (ht : evalX t a = some (num z)) : evalX (Whalf1a t) a = some (num (z / 2)) :=
  evalX_congr (evalX_Wmul1a evalX_const_half ht) (by ring_nf)

/-- **Corollary A.6**: `W_avg(z, w) = (z + w)/2` for all `z, w ∈ ℂ`. -/
theorem evalX_Wavg1a (hs : evalX s a = some (num z)) (ht : evalX t a = some (num w)) :
    evalX (Wavg1a s t) a = some (num ((z + w) / 2)) :=
  evalX_Whalf1a (evalX_Wadd1a hs ht)

/-- **Corollary A.6**: `W/(z, w) = z/w` for every `z ∈ ℂ` and every `w ≠ 0`. -/
theorem evalX_Wdiv1a (hs : evalX s a = some (num z)) (ht : evalX t a = some (num w)) (hw : w ≠ 0) :
    evalX (Wdiv1a s t) a = some (num (z / w)) :=
  evalX_congr (evalX_Wmul1a hs (evalX_Winv1a ht hw)) (by rw [div_eq_mul_inv])

end EML
