module

public import EML.Complex.PowLog

/-!
# Section A.4: values at the exceptional points (Lemma A.12, Corollary A.13, examples)

The test for zero `J`, a prescribed value at one point (`G`), and the three examples of the
paper: the reciprocal and the logarithm in Lean's convention (`1/0 = 0`, `Log 0 = 0`), and the
logarithm in the convention of the Wolfram Language (`Log 0 = ∞₋`). Theorem 1a does not depend
on this file.

The lemmas are stated for an argument term `t` with value `z`, so `J(t)` tests any term `t`
for zero, e.g. `t = z - a` or `t = g(z)` (the remark after Corollary A.13).
-/

@[expose] public section

namespace EML

open Complex Term XC

variable {n : ℕ} {a : Fin n → ℂ} {s t f c : Term n} {u : XC} {z w c₀ : ℂ}

/-- `W×(∞±, w) = 0` for every `w ∈ ℂ`: `EML(∞₋, ∞±) = ∞₋`, `L(∞₋) = ∞₊`, `EML(∞₊, w) = ∞₊`,
`E(∞₊) = ∞₊`, `W_inv(∞₊) = 0` (proof of Lemma A.12, proof of Corollary A.13). -/
theorem evalX_Wmul1a_inf (hs : evalX s a = some ninf ∨ evalX s a = some pinf)
    (ht : evalX t a = some (num w)) : evalX (Wmul1a s t) a = some (num 0) := by
  have h1 : evalX (eml const_ninf s) a = some ninf := by
    rcases hs with hs | hs
    · rw [evalX_eml evalX_const_ninf hs]
      rfl
    · rw [evalX_eml evalX_const_ninf hs]
      rfl
  have h2 : evalX (eml (L (eml const_ninf s)) t) a = some pinf := by
    rw [evalX_eml (evalX_L_ninf h1) ht]
    rfl
  exact evalX_Winv1a_pinf (evalX_E_pinf h2)

/-- The Remark after Table A.3: the witnesses do not treat `∞₊` as a number, `W×(∞₊, 2) = 0`. -/
theorem evalX_Wmul1a_pinf_two : evalX (Wmul1a const_pinf const_2 : Term n) a = some (num 0) :=
  evalX_Wmul1a_inf (Or.inr evalX_const_pinf) evalX_const_2

/-- `J(z) := W×(W_inv(z), z)`, the test for zero. -/
def J (t : Term n) : Term n := Wmul1a (Winv1a t) t

/-- **Lemma A.12**: `J(z) = 1` for every `z ≠ 0`. -/
theorem evalX_J (ht : evalX t a = some (num z)) (hz : z ≠ 0) : evalX (J t) a = some (num 1) :=
  evalX_congr (evalX_Wmul1a (evalX_Winv1a ht hz) ht) (by rw [inv_mul_cancel₀ hz])

/-- **Lemma A.12**: `J(0) = 0`, because `W_inv(0) = ∞₊` and `W×(∞₊, 0) = 0`. -/
theorem evalX_J_zero (ht : evalX t a = some (num 0)) : evalX (J t) a = some (num 0) :=
  evalX_Wmul1a_inf (Or.inr (evalX_Winv1a_zero ht)) ht

/-- `G(z) := W×(F(z), J(z)) + c (1 - J(z))`. Here `f` is the term `F(z)`, `t` the term `z`,
and `c` the term whose value is `c`. -/
def G (f t c : Term n) : Term n := Wadd1a (Wmul1a f (J t)) (Wmul1a c (Wsub1a one (J t)))

/-- **Corollary A.13**: `G(z) = F(z)` for every `z ≠ 0` with `F(z) ∈ ℂ`. -/
theorem evalX_G (hf : evalX f a = some (num w)) (ht : evalX t a = some (num z)) (hz : z ≠ 0)
    (hc : evalX c a = some (num c₀)) : evalX (G f t c) a = some (num w) := by
  have hJ := evalX_J ht hz
  refine evalX_congr (evalX_Wadd1a (evalX_Wmul1a hf hJ) (evalX_Wmul1a hc
    (evalX_Wsub1a evalX_one hJ))) ?_
  congr 1
  ring

/-- **Corollary A.13**: `G(0) = c` for `c ∈ ℂ`, if `F` is defined at `0`; its value there may be
a number, `∞₋` or `∞₊`. -/
theorem evalX_G_zero (hf : evalX f a = some u) (ht : evalX t a = some (num 0))
    (hc : evalX c a = some (num c₀)) : evalX (G f t c) a = some (num c₀) := by
  have hJ := evalX_J_zero ht
  -- `W×(F(0), 0) = 0`
  have h1 : evalX (Wmul1a f (J t)) a = some (num 0) := by
    cases u with
    | num v => exact evalX_congr (evalX_Wmul1a hf hJ) (by rw [mul_zero])
    | ninf => exact evalX_Wmul1a_inf (Or.inl hf) hJ
    | pinf => exact evalX_Wmul1a_inf (Or.inr hf) hJ
  refine evalX_congr (evalX_Wadd1a h1 (evalX_Wmul1a hc (evalX_Wsub1a evalX_one hJ))) ?_
  congr 1
  ring

/-! ## The examples of Section A.4 -/

/-- The reciprocal in Lean's convention: `W×(W_inv(z), J(z))`. -/
def Winv1a_lean (t : Term n) : Term n := Wmul1a (Winv1a t) (J t)

/-- Section A.4: `W×(W_inv(z), J(z))` is `1/z` for `z ≠ 0` and `0` for `z = 0`. This is
Mathlib's `z⁻¹` at every `z ∈ ℂ`, since Mathlib sets `0⁻¹ = 0`. -/
theorem evalX_Winv1a_lean (ht : evalX t a = some (num z)) :
    evalX (Winv1a_lean t) a = some (num z⁻¹) := by
  by_cases hz : z = 0
  · subst hz
    exact evalX_congr (evalX_Wmul1a_inf (Or.inr (evalX_Winv1a_zero ht)) (evalX_J_zero ht))
      (by rw [inv_zero])
  · exact evalX_congr (evalX_Wmul1a (evalX_Winv1a ht hz) (evalX_J ht hz)) (by rw [mul_one])

/-- Section A.4: `W_ln(0) = ∞₊`, the value that the third example replaces by `∞₋`. -/
theorem evalX_Wln1a_zero (ht : evalX t a = some (num 0)) : evalX (Wln1a t) a = some pinf :=
  evalX_Wmul1a_pinf evalX_const_neg (by norm_num) (evalX_eml_ninf_zero ht)

/-- The logarithm in Lean's convention: `W×(W_ln(z), J(z))`. -/
def Wln1a_lean (t : Term n) : Term n := Wmul1a (Wln1a t) (J t)

/-- Section A.4: `W×(W_ln(z), J(z))` is `Log z` for `z ≠ 0` and `0` for `z = 0`. This is
Mathlib's `Complex.log z` at every `z ∈ ℂ`, since Mathlib sets `log 0 = 0`. -/
theorem evalX_Wln1a_lean (ht : evalX t a = some (num z)) :
    evalX (Wln1a_lean t) a = some (num (log z)) := by
  by_cases hz : z = 0
  · subst hz
    exact evalX_congr (evalX_Wmul1a_inf (Or.inr (evalX_Wln1a_zero ht)) (evalX_J_zero ht))
      (by rw [log_zero])
  · exact evalX_congr (evalX_Wmul1a (evalX_Wln1a ht hz) (evalX_J ht hz)) (by rw [mul_one])

/-- The logarithm in the convention of the Wolfram Language and of Definition A.1,
`Log(0) = ∞₋`: `EML(L(W×(W_ln(z), J(z))), W_inv(J(z)))`. -/
def Wln1a_wolfram (t : Term n) : Term n := eml (L (Wln1a_lean t)) (Winv1a (J t))

/-- Section A.4: `EML(L(W×(W_ln(z), J(z))), W_inv(J(z))) = Log z - Log 1 = Log z` for `z ≠ 0`. -/
theorem evalX_Wln1a_wolfram (ht : evalX t a = some (num z)) (hz : z ≠ 0) :
    evalX (Wln1a_wolfram t) a = some (num (log z)) := by
  have h2 : evalX (Winv1a (J t)) a = some (num 1) :=
    evalX_congr (evalX_Winv1a (evalX_J ht hz) one_ne_zero) (by rw [inv_one])
  exact evalX_congr (evalX_eml_L (evalX_Wln1a_lean ht) h2 one_ne_zero)
    (by rw [log_one, sub_zero])

/-- Section A.4: `EML(L(W×(W_ln(0), J(0))), W_inv(J(0))) = EML(∞₋, ∞₊) = ∞₋`. -/
theorem evalX_Wln1a_wolfram_zero (ht : evalX t a = some (num 0)) :
    evalX (Wln1a_wolfram t) a = some ninf := by
  have h1 : evalX (Wln1a_lean t) a = some (num 0) :=
    evalX_congr (evalX_Wln1a_lean ht) (by rw [log_zero])
  rw [Wln1a_wolfram, evalX_eml (evalX_L_zero h1) (evalX_Winv1a_zero (evalX_J_zero ht))]
  rfl

/-! ## A whole branch cut: the zero set of a term need not consist of isolated points -/

variable {f' : Term n} {v v' : ℂ}

/-- The closed term `2πi` of Lemma 8 keeps its value. -/
theorem evalX_const_2ipi : evalX (const_2ipi : Term n) a = some (num (2 * (I * Real.pi))) :=
  evalX_of_eval eval_const_2ipi

/-- `d(z) := (W_ln(z) - L(z))/(2πi)`. -/
def dcut (t : Term n) : Term n := Wdiv1a (Wsub1a (Wln1a t) (L t)) const_2ipi

/-- Section A.4: `d(z) = 1` on the negative real axis and `0` elsewhere in `ℂ*`, by Lemma 3
(`L(z) = Log z - 2πi` on the negative real axis, `Log z` elsewhere) and Lemma A.9
(`W_ln(z) = Log z`). -/
theorem evalX_dcut (ht : evalX t a = some (num z)) (hz : z ≠ 0) :
    evalX (dcut t) a = some (num (if z.re < 0 ∧ z.im = 0 then 1 else 0)) := by
  have h2pi : (2 * (I * Real.pi) : ℂ) ≠ 0 :=
    mul_ne_zero two_ne_zero (mul_ne_zero I_ne_zero (ofReal_ne_zero.mpr Real.pi_ne_zero))
  refine evalX_congr (evalX_Wdiv1a (evalX_Wsub1a (evalX_Wln1a ht hz) (evalX_L ht hz))
    evalX_const_2ipi h2pi) ?_
  congr 1
  by_cases hc : z.re < 0 ∧ z.im = 0
  · rw [Lval_of_negReal hc.1 hc.2, ite_eq_left hc, div_eq_one_iff_eq h2pi]
    ring
  · unfold Lval
    rw [ite_eq_right hc, ite_eq_right hc, sub_self, zero_div]

/-- `F(z)(1 - d(z)) + F'(z) d(z)`. -/
def replace_on_cut (f f' t : Term n) : Term n :=
  Wadd1a (Wmul1a f (Wsub1a one (dcut t))) (Wmul1a f' (dcut t))

/-- Section A.4: `F(z)(1 - d(z)) + F'(z) d(z)` is `F'(z)` on the whole negative real axis and
`F(z)` elsewhere in `ℂ*`, wherever `F(z)` and `F'(z)` are numbers. -/
theorem evalX_replace_on_cut (hf : evalX f a = some (num v)) (hf' : evalX f' a = some (num v'))
    (ht : evalX t a = some (num z)) (hz : z ≠ 0) :
    evalX (replace_on_cut f f' t) a = some (num (if z.re < 0 ∧ z.im = 0 then v' else v)) := by
  have hd := evalX_dcut ht hz
  refine evalX_congr (evalX_Wadd1a (evalX_Wmul1a hf (evalX_Wsub1a evalX_one hd))
    (evalX_Wmul1a hf' hd)) ?_
  congr 1
  split_ifs <;> ring

end EML
