module

public import EML.Complex.PowLog

/-!
# Proof of Theorem 1a, Step 4 (Lemma A.11)

`σ`, the trigonometric and hyperbolic functions, and their inverses.

* `sinh`, `cosh`, `cos`, `sin`, `tan`, `tanh` are Mathlib's complex functions, which Mathlib
  defines through `exp` exactly as in Lemma A.11 (DLMF 4.14.1, 4.14.2, 4.14.4, 4.28.1,
  4.28.2, 4.28.4).
* For `arcsin`, `arccos`, `arsinh`, `arcosh`, `arctan`, `artanh` the value is the formula of
  Lemma A.11; by Section A.2 this is also the value on the branch cuts. Mathlib has no complex
  arcsin, arccos, arsinh, arcosh, artanh, and its `Complex.arctan` takes the other value on
  the lower cut, so it is not used.

Arithmetic written in the usual notation in the paper (convention (i)) is read as the
witnesses of Corollary A.6: `a/2` is `W/(a, 2)`, `iz` is `W×(i, z)`, `-i X` is `W×(-i, X)` with
the closed term `-i` of Lemma A.5, and `π/2`, `2i` are the closed terms of Lemma 8.
-/

@[expose] public section

namespace EML

open Complex Term XC

variable {n : ℕ} {a : Fin n → ℂ} {t : Term n} {z : ℂ}

/-- The closed terms `π/2` and `2i` of Lemma 8 keep their values. -/
theorem evalX_const_pi_half : evalX (const_pi_half : Term n) a = some (num ((Real.pi / 2 : ℝ) : ℂ)) :=
  evalX_of_eval eval_const_pi_half

theorem evalX_const_2i : evalX (const_2i : Term n) a = some (num (2 * I)) :=
  evalX_of_eval eval_const_2i

/-- `(z^(1/2))² = z` for every `z ∈ ℂ`. -/
theorem cpow_half_sq (x : ℂ) : (x ^ (1 / 2 : ℂ)) ^ 2 = x := by
  have h := cpow_nat_inv_pow x (n := 2) two_ne_zero
  rw [one_div]
  exact_mod_cast h

/-! ## `σ`, the hyperbolic and the trigonometric functions -/

/-- `W_σ(z) := W_inv(1 + E(-z))`. -/
def Wsigma1a (t : Term n) : Term n := Winv1a (Wadd1a one (E (Wminus1a t)))

/-- **Lemma A.11**: `W_σ(z) = σ(z) = 1/(1 + e^(-z))` for `e^z ≠ -1`. -/
theorem evalX_Wsigma1a (ht : evalX t a = some (num z)) (hz : exp z ≠ -1) :
    evalX (Wsigma1a t) a = some (num (1 + exp (-z))⁻¹) := by
  -- the divisor `1 + e^(-z)` is not `0`
  have h0 : 1 + exp (-z) ≠ 0 := by
    intro h
    apply hz
    have h1 : (exp z)⁻¹ = -1 := by rw [← exp_neg]; linear_combination h
    rw [inv_eq_iff_eq_inv] at h1
    rw [h1]
    norm_num
  exact evalX_Winv1a (evalX_Wadd1a evalX_one (evalX_E (evalX_Wminus1a ht))) h0

/-- `W_sinh(z) := (E(z) - E(-z))/2`. -/
def Wsinh1a (t : Term n) : Term n := Wdiv1a (Wsub1a (E t) (E (Wminus1a t))) const_2

/-- **Lemma A.11**: `W_sinh(z) = sinh z` for every `z ∈ ℂ`. -/
theorem evalX_Wsinh1a (ht : evalX t a = some (num z)) :
    evalX (Wsinh1a t) a = some (num (sinh z)) :=
  evalX_congr (evalX_Wdiv1a (evalX_Wsub1a (evalX_E ht) (evalX_E (evalX_Wminus1a ht)))
    evalX_const_2 two_ne_zero) (by rw [Complex.sinh])

/-- `W_cosh(z) := (E(z) + E(-z))/2`. -/
def Wcosh1a (t : Term n) : Term n := Wdiv1a (Wadd1a (E t) (E (Wminus1a t))) const_2

/-- **Lemma A.11**: `W_cosh(z) = cosh z` for every `z ∈ ℂ`. -/
theorem evalX_Wcosh1a (ht : evalX t a = some (num z)) :
    evalX (Wcosh1a t) a = some (num (cosh z)) :=
  evalX_congr (evalX_Wdiv1a (evalX_Wadd1a (evalX_E ht) (evalX_E (evalX_Wminus1a ht)))
    evalX_const_2 two_ne_zero) (by rw [Complex.cosh])

/-- `W_cos(z) := (E(iz) + E(-iz))/2`. -/
def Wcos1a (t : Term n) : Term n :=
  Wdiv1a (Wadd1a (E (Wmul1a const_i t)) (E (Wmul1a const_minus_i t))) const_2

/-- **Lemma A.11**: `W_cos(z) = cos z` for every `z ∈ ℂ`. -/
theorem evalX_Wcos1a (ht : evalX t a = some (num z)) :
    evalX (Wcos1a t) a = some (num (cos z)) :=
  evalX_congr (evalX_Wdiv1a (evalX_Wadd1a (evalX_E (evalX_Wmul1a evalX_const_i ht))
    (evalX_E (evalX_Wmul1a evalX_const_minus_i ht))) evalX_const_2 two_ne_zero)
    (by rw [Complex.cos, show -I * z = -z * I by ring, mul_comm I z])

/-- `W_sin(z) := W_cos(z - π/2)`. -/
def Wsin1a (t : Term n) : Term n := Wcos1a (Wsub1a t const_pi_half)

/-- **Lemma A.11**: `W_sin(z) = sin z` for every `z ∈ ℂ`. -/
theorem evalX_Wsin1a (ht : evalX t a = some (num z)) :
    evalX (Wsin1a t) a = some (num (sin z)) :=
  evalX_congr (evalX_Wcos1a (evalX_Wsub1a ht evalX_const_pi_half))
    (by push_cast; rw [cos_sub_pi_div_two])

/-- `W_tan(z) := W_sin(z)/W_cos(z)`. -/
def Wtan1a (t : Term n) : Term n := Wdiv1a (Wsin1a t) (Wcos1a t)

/-- **Lemma A.11**: `W_tan(z) = tan z` for `cos z ≠ 0`. -/
theorem evalX_Wtan1a (ht : evalX t a = some (num z)) (hz : cos z ≠ 0) :
    evalX (Wtan1a t) a = some (num (tan z)) :=
  evalX_congr (evalX_Wdiv1a (evalX_Wsin1a ht) (evalX_Wcos1a ht) hz) (by rw [Complex.tan])

/-- `W_tanh(z) := W_sinh(z)/W_cosh(z)`. -/
def Wtanh1a (t : Term n) : Term n := Wdiv1a (Wsinh1a t) (Wcosh1a t)

/-- **Lemma A.11**: `W_tanh(z) = tanh z` for `cosh z ≠ 0`. -/
theorem evalX_Wtanh1a (ht : evalX t a = some (num z)) (hz : cosh z ≠ 0) :
    evalX (Wtanh1a t) a = some (num (tanh z)) :=
  evalX_congr (evalX_Wdiv1a (evalX_Wsinh1a ht) (evalX_Wcosh1a ht) hz) (by rw [Complex.tanh])

/-! ## The inverse functions -/

/-- `W_arcsin(z) := -i W_ln(iz + √(1 - z²))`. -/
def Warcsin1a (t : Term n) : Term n :=
  Wmul1a const_minus_i (Wln1a (Wadd1a (Wmul1a const_i t) (Wsqrt1a (Wsub1a one (Wsqr1a t)))))

/-- **Lemma A.11**: `W_arcsin(z) = -i Log(iz + √(1 - z²))` for every `z ∈ ℂ`. -/
theorem evalX_Warcsin1a (ht : evalX t a = some (num z)) :
    evalX (Warcsin1a t) a = some (num (-I * log (I * z + (1 - z ^ 2) ^ (1 / 2 : ℂ)))) := by
  have h1 := evalX_Wadd1a (evalX_Wmul1a evalX_const_i ht)
    (evalX_Wsqrt1a (evalX_Wsub1a evalX_one (evalX_Wsqr1a ht)))
  -- the argument of `W_ln` is not `0`: `(iz + √(1 - z²))(-iz + √(1 - z²)) = 1`
  have h0 : I * z + (1 - z ^ 2) ^ (1 / 2 : ℂ) ≠ 0 := by
    apply left_ne_zero_of_mul_eq_one (b := -(I * z) + (1 - z ^ 2) ^ (1 / 2 : ℂ))
    linear_combination cpow_half_sq (1 - z ^ 2) + (-z ^ 2) * I_sq
  exact evalX_Wmul1a evalX_const_minus_i (evalX_Wln1a h1 h0)

/-- `W_arccos(z) := π/2 - W_arcsin(z)`. -/
def Warccos1a (t : Term n) : Term n := Wsub1a const_pi_half (Warcsin1a t)

/-- **Lemma A.11**: `W_arccos(z) = π/2 - W_arcsin(z)` for every `z ∈ ℂ`. -/
theorem evalX_Warccos1a (ht : evalX t a = some (num z)) :
    evalX (Warccos1a t) a =
      some (num (Real.pi / 2 - -I * log (I * z + (1 - z ^ 2) ^ (1 / 2 : ℂ)))) :=
  evalX_congr (evalX_Wsub1a evalX_const_pi_half (evalX_Warcsin1a ht)) (by push_cast; ring_nf)

/-- `W_arsinh(z) := W_ln(z + √(1 + z²))`. -/
def Warsinh1a (t : Term n) : Term n := Wln1a (Wadd1a t (Wsqrt1a (Wadd1a one (Wsqr1a t))))

/-- **Lemma A.11**: `W_arsinh(z) = Log(z + √(1 + z²))` for every `z ∈ ℂ`. -/
theorem evalX_Warsinh1a (ht : evalX t a = some (num z)) :
    evalX (Warsinh1a t) a = some (num (log (z + (1 + z ^ 2) ^ (1 / 2 : ℂ)))) := by
  have h1 := evalX_Wadd1a ht (evalX_Wsqrt1a (evalX_Wadd1a evalX_one (evalX_Wsqr1a ht)))
  -- `(z + √(1 + z²))(-z + √(1 + z²)) = 1`
  have h0 : z + (1 + z ^ 2) ^ (1 / 2 : ℂ) ≠ 0 := by
    apply left_ne_zero_of_mul_eq_one (b := -z + (1 + z ^ 2) ^ (1 / 2 : ℂ))
    linear_combination cpow_half_sq (1 + z ^ 2)
  exact evalX_Wln1a h1 h0

/-- `W_arcosh(z) := W_ln(z + √(z - 1) √(z + 1))`. -/
def Warcosh1a (t : Term n) : Term n :=
  Wln1a (Wadd1a t (Wmul1a (Wsqrt1a (Wsub1a t one)) (Wsqrt1a (Wadd1a t one))))

/-- **Lemma A.11**: `W_arcosh(z) = Log(z + √(z - 1) √(z + 1))` for every `z ∈ ℂ`. -/
theorem evalX_Warcosh1a (ht : evalX t a = some (num z)) :
    evalX (Warcosh1a t) a =
      some (num (log (z + (z - 1) ^ (1 / 2 : ℂ) * (z + 1) ^ (1 / 2 : ℂ)))) := by
  have h1 := evalX_Wadd1a ht (evalX_Wmul1a (evalX_Wsqrt1a (evalX_Wsub1a ht evalX_one))
    (evalX_Wsqrt1a (evalX_Wadd1a ht evalX_one)))
  -- `(z + √(z - 1) √(z + 1))(z - √(z - 1) √(z + 1)) = z² - (z - 1)(z + 1) = 1`
  have h0 : z + (z - 1) ^ (1 / 2 : ℂ) * (z + 1) ^ (1 / 2 : ℂ) ≠ 0 := by
    apply left_ne_zero_of_mul_eq_one (b := z - (z - 1) ^ (1 / 2 : ℂ) * (z + 1) ^ (1 / 2 : ℂ))
    linear_combination (-((z + 1) ^ (1 / 2 : ℂ)) ^ 2) * cpow_half_sq (z - 1)
      + (-(z - 1)) * cpow_half_sq (z + 1)
  exact evalX_Wln1a h1 h0

/-- `W_arctan(z) := (W_ln(1 + iz) - W_ln(1 - iz))/(2i)`. -/
def Warctan1a (t : Term n) : Term n :=
  Wdiv1a (Wsub1a (Wln1a (Wadd1a one (Wmul1a const_i t))) (Wln1a (Wsub1a one (Wmul1a const_i t))))
    const_2i

/-- **Lemma A.11**: `W_arctan(z) = (Log(1 + iz) - Log(1 - iz))/(2i)` for `z ≠ ±i`. -/
theorem evalX_Warctan1a (ht : evalX t a = some (num z)) (h1 : z ≠ I) (h2 : z ≠ -I) :
    evalX (Warctan1a t) a = some (num ((log (1 + I * z) - log (1 - I * z)) / (2 * I))) := by
  -- `1 + iz = 0` only at `z = i`, and `1 - iz = 0` only at `z = -i`
  have hp : 1 + I * z ≠ 0 := by
    intro h
    apply h1
    linear_combination (-I) * h + z * I_sq
  have hm : 1 - I * z ≠ 0 := by
    intro h
    apply h2
    linear_combination I * h + z * I_sq
  have hl := evalX_Wsub1a (evalX_Wln1a (evalX_Wadd1a evalX_one (evalX_Wmul1a evalX_const_i ht)) hp)
    (evalX_Wln1a (evalX_Wsub1a evalX_one (evalX_Wmul1a evalX_const_i ht)) hm)
  exact evalX_Wdiv1a hl evalX_const_2i (mul_ne_zero two_ne_zero I_ne_zero)

/-- `W_artanh(z) := (W_ln(1 + z) - W_ln(1 - z))/2`. -/
def Wartanh1a (t : Term n) : Term n :=
  Wdiv1a (Wsub1a (Wln1a (Wadd1a one t)) (Wln1a (Wsub1a one t))) const_2

/-- **Lemma A.11**: `W_artanh(z) = (Log(1 + z) - Log(1 - z))/2` for `z ≠ ±1`. -/
theorem evalX_Wartanh1a (ht : evalX t a = some (num z)) (h1 : z ≠ 1) (h2 : z ≠ -1) :
    evalX (Wartanh1a t) a = some (num ((log (1 + z) - log (1 - z)) / 2)) := by
  have hp : 1 + z ≠ 0 := by
    intro h
    apply h2
    linear_combination h
  have hm : 1 - z ≠ 0 := by
    intro h
    apply h1
    linear_combination -h
  have hl := evalX_Wsub1a (evalX_Wln1a (evalX_Wadd1a evalX_one ht) hp)
    (evalX_Wln1a (evalX_Wsub1a evalX_one ht) hm)
  exact evalX_Wdiv1a hl evalX_const_2 two_ne_zero

end EML
