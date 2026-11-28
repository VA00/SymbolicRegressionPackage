module

public import EML.Real.Strict

/-!
# Theorem 1b (Appendix B), Corollary B.5: the class of strict real `eml` terms

The strict real `eml` terms compute exactly the partial functions given by the real exp–log
expressions: the expressions built from `1`, the variables, `+`, `-`, `×`, `/`, `exp` and the
real `ln`, where a division by `0` and a logarithm of a non-positive number are undefined.

* `ELExpr n`      — the real exp–log expressions in `n` variables.
* `ELExpr.eval`   — the value of an expression, an `Option ℝ` (`none` = undefined).
* `ELExpr.toTerm` — an expression compiled to an `eml` term with the witnesses of Table B.1.
* `Term.toEL`     — an `eml` term read as an expression: `eml(U, V) = exp U - ln V`.
* `evalR_toTerm`, `eval_toEL` — **Corollary B.5**: both translations preserve the value at every
  real point, and they preserve "undefined". So the domains are the same, not only the values.
-/

@[expose] public section

namespace EML

open Term

variable {n : ℕ}

/-- The real exp–log expressions in the variables `x_0, …, x_{n-1}`. -/
inductive ELExpr (n : ℕ) : Type
  /-- the constant `1` -/
  | one : ELExpr n
  /-- the variable `x_j` -/
  | var : Fin n → ELExpr n
  /-- `F + G` -/
  | add : ELExpr n → ELExpr n → ELExpr n
  /-- `F - G` -/
  | sub : ELExpr n → ELExpr n → ELExpr n
  /-- `F × G` -/
  | mul : ELExpr n → ELExpr n → ELExpr n
  /-- `F / G` -/
  | div : ELExpr n → ELExpr n → ELExpr n
  /-- `exp F` -/
  | exp : ELExpr n → ELExpr n
  /-- `ln F` -/
  | log : ELExpr n → ELExpr n

namespace ELExpr

/-- The value of an expression at the real point `a`. A quotient is undefined when the divisor
is `0`; a logarithm is undefined when its argument is not strictly positive; an expression with
an undefined subexpression is undefined. -/
noncomputable def eval : ELExpr n → (Fin n → ℝ) → Option ℝ
  | one, _ => some 1
  | var j, a => some (a j)
  | add f g, a =>
    match eval f a, eval g a with
    | some x, some y => some (x + y)
    | _, _ => none
  | sub f g, a =>
    match eval f a, eval g a with
    | some x, some y => some (x - y)
    | _, _ => none
  | mul f g, a =>
    match eval f a, eval g a with
    | some x, some y => some (x * y)
    | _, _ => none
  | div f g, a =>
    match eval f a, eval g a with
    | some x, some y => if y = 0 then none else some (x / y)
    | _, _ => none
  | exp f, a =>
    match eval f a with
    | some x => some (Real.exp x)
    | none => none
  | log f, a =>
    match eval f a with
    | some x => if 0 < x then some (Real.log x) else none
    | none => none

/-- An expression compiled to an `eml` term: each operation is replaced by its witness of
Table B.1. -/
def toTerm : ELExpr n → Term n
  | one => Term.one
  | var j => Term.var j
  | add f g => Wadd1b (toTerm f) (toTerm g)
  | sub f g => Wsub1b (toTerm f) (toTerm g)
  | mul f g => Wmul1b (toTerm f) (toTerm g)
  | div f g => Wdiv1b (toTerm f) (toTerm g)
  | exp f => E (toTerm f)
  | log f => L (toTerm f)

end ELExpr

/-- An `eml` term read as an exp–log expression: `eml(U, V) = exp U - ln V`. -/
def Term.toEL : Term n → ELExpr n
  | Term.one => ELExpr.one
  | Term.var j => ELExpr.var j
  | Term.eml s t => ELExpr.sub (ELExpr.exp (Term.toEL s)) (ELExpr.log (Term.toEL t))

/-- **Corollary B.5**, first direction. The compiled term has the same value as the expression
at every real point, and it is undefined exactly where the expression is undefined. -/
theorem evalR_toTerm (F : ELExpr n) (a : Fin n → ℝ) : evalR F.toTerm a = F.eval a := by
  induction F with
  | one => rfl
  | var j => rfl
  | add f g ihf ihg =>
    cases hf : f.eval a with
    | none =>
      rw [hf] at ihf
      simp [ELExpr.toTerm, ELExpr.eval, hf, evalR_Wadd1b_undef_left ihf]
    | some x =>
      rw [hf] at ihf
      cases hg : g.eval a with
      | none =>
        rw [hg] at ihg
        simp [ELExpr.toTerm, ELExpr.eval, hf, hg, evalR_Wadd1b_undef_right ihg]
      | some y =>
        rw [hg] at ihg
        simp [ELExpr.toTerm, ELExpr.eval, hf, hg, val_Wadd1b (strictVal_evalR a) ihf ihg]
  | sub f g ihf ihg =>
    cases hf : f.eval a with
    | none =>
      rw [hf] at ihf
      simp [ELExpr.toTerm, ELExpr.eval, hf, evalR_Wsub1b_undef_left ihf]
    | some x =>
      rw [hf] at ihf
      cases hg : g.eval a with
      | none =>
        rw [hg] at ihg
        simp [ELExpr.toTerm, ELExpr.eval, hf, hg, evalR_Wsub1b_undef_right ihg]
      | some y =>
        rw [hg] at ihg
        simp [ELExpr.toTerm, ELExpr.eval, hf, hg, val_Wsub1b (strictVal_evalR a) ihf ihg]
  | mul f g ihf ihg =>
    cases hf : f.eval a with
    | none =>
      rw [hf] at ihf
      simp [ELExpr.toTerm, ELExpr.eval, hf, evalR_Wmul1b_undef_left ihf]
    | some x =>
      rw [hf] at ihf
      cases hg : g.eval a with
      | none =>
        rw [hg] at ihg
        simp [ELExpr.toTerm, ELExpr.eval, hf, hg, evalR_Wmul1b_undef_right ihg]
      | some y =>
        rw [hg] at ihg
        simp [ELExpr.toTerm, ELExpr.eval, hf, hg, val_Wmul1b (strictVal_evalR a) ihf ihg]
  | div f g ihf ihg =>
    cases hf : f.eval a with
    | none =>
      rw [hf] at ihf
      simp [ELExpr.toTerm, ELExpr.eval, hf, evalR_Wdiv1b_undef_left ihf]
    | some x =>
      rw [hf] at ihf
      cases hg : g.eval a with
      | none =>
        rw [hg] at ihg
        simp [ELExpr.toTerm, ELExpr.eval, hf, hg, evalR_Wdiv1b_undef_right ihg]
      | some y =>
        rw [hg] at ihg
        by_cases hy : y = 0
        · subst hy
          simp [ELExpr.toTerm, ELExpr.eval, hf, hg, evalR_Wdiv1b_zero ihg]
        · simp [ELExpr.toTerm, ELExpr.eval, hf, hg, hy,
            val_Wdiv1b (strictVal_evalR a) ihf ihg hy]
  | exp f ihf =>
    cases hf : f.eval a with
    | none =>
      rw [hf] at ihf
      simp [ELExpr.toTerm, ELExpr.eval, hf, evalR_E_undef ihf]
    | some x =>
      rw [hf] at ihf
      simp [ELExpr.toTerm, ELExpr.eval, hf, val_E (strictVal_evalR a) ihf]
  | log f ihf =>
    cases hf : f.eval a with
    | none =>
      rw [hf] at ihf
      simp [ELExpr.toTerm, ELExpr.eval, hf, evalR_L_undef ihf]
    | some x =>
      rw [hf] at ihf
      by_cases hx : 0 < x
      · simp [ELExpr.toTerm, ELExpr.eval, hf, hx, val_L (strictVal_evalR a) ihf hx]
      · simp [ELExpr.toTerm, ELExpr.eval, hf, hx, evalR_L_nonpos ihf (not_lt.mp hx)]

/-- **Corollary B.5**, second direction. An `eml` term read as an expression has the same value
at every real point, and is undefined exactly where the term is undefined. -/
theorem eval_toEL (t : Term n) (a : Fin n → ℝ) : t.toEL.eval a = evalR t a := by
  induction t with
  | one => rfl
  | var j => rfl
  | eml s t ihs iht =>
    cases hs : evalR s a with
    | none =>
      rw [hs] at ihs
      simp [Term.toEL, ELExpr.eval, ihs, evalR_eml_undef_left hs]
    | some u =>
      rw [hs] at ihs
      cases ht : evalR t a with
      | none =>
        rw [ht] at iht
        simp [Term.toEL, ELExpr.eval, ihs, iht, evalR_eml_undef_right ht]
      | some v =>
        rw [ht] at iht
        by_cases hv : 0 < v
        · simp [Term.toEL, ELExpr.eval, ihs, iht, hv, evalR_eml hs ht hv]
        · simp [Term.toEL, ELExpr.eval, ihs, iht, hv, evalR_eml_nonpos ht (not_lt.mp hv)]

/-- **Corollary B.5.** The strict real `eml` terms and the real exp–log expressions define the
same partial functions `ℝⁿ ⇀ ℝ`, with the same domains. -/
theorem corollary_B5 :
    (∀ F : ELExpr n, ∃ t : Term n, ∀ a : Fin n → ℝ, evalR t a = F.eval a) ∧
    (∀ t : Term n, ∃ F : ELExpr n, ∀ a : Fin n → ℝ, F.eval a = evalR t a) :=
  ⟨fun F => ⟨F.toTerm, evalR_toTerm F⟩, fun t => ⟨t.toEL, eval_toEL t⟩⟩

end EML
