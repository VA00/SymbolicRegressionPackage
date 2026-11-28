module

public import EML.Theorem2
public import EML.Theorem1a

/-!
# Theorem 2 for complex arguments (the remark after Theorem 1a)

Theorem 2 carries over to the complex primitives of Table A.2 for expressions whose intermediate
values are complex numbers.

* `CalcExprC n` — expressions in the variables `z_0, …, z_{n-1}`, built from the variables and
  the 6 constants of Table A.2 by its 28 primitives.
* `CalcExprC.value F a` — the partial complex function of `F`: `some z` means that every
  primitive of `F` is applied inside its domain (Table A.2) and `F(a) = z`.
* `evalX_subst` — a term is evaluated from its leaves upwards, in the arithmetic of Appendix A.
* `CalcExprC.witness F` — every primitive `P` of `F` replaced by its term of Theorem 1a.
* `theorem2a` — at every point where `F` is defined the term is defined and its value is `F(a)`.
-/

@[expose] public section

namespace EML

open Complex Term XC

/-- Expressions over the complex primitives of Table A.2, in the variables `z_0, …, z_{n-1}`. -/
inductive CalcExprC (n : ℕ) : Type
  /-- the variable `z_j` -/
  | var : Fin n → CalcExprC n
  /-- one of the 6 constants of Table A.2 -/
  | const : Prim0a → CalcExprC n
  /-- a unary primitive of Table A.2, applied to an expression -/
  | unary : Prim1a → CalcExprC n → CalcExprC n
  /-- a binary primitive of Table A.2, applied to two expressions -/
  | binary : Prim2a → CalcExprC n → CalcExprC n → CalcExprC n

namespace CalcExprC

variable {n : ℕ}

open Classical in
/-- The value of an expression at the complex point `a`: `F = P(F₁, …, F_k)` is defined at `a`
iff all `F_j` are defined at `a` and their values lie in the domain of `P` (Table A.2); its
value is then `P(F₁(a), …, F_k(a))`. All intermediate values are complex numbers. -/
noncomputable def value : CalcExprC n → (Fin n → ℂ) → Option ℂ
  | var j, a => some (a j)
  | const P, _ => some P.val
  | unary P F, a =>
    match value F a with
    | some u => if u ∈ P.dom then some (P.val u) else none
    | none => none
  | binary P F G, a =>
    match value F a, value G a with
    | some u, some v => if (u, v) ∈ P.dom then some (P.val u v) else none
    | _, _ => none

/-- The term of an expression: every primitive `P` of `F` is replaced by the term `P.witness`
of Theorem 1a, with the terms of the arguments in place of its variables. -/
def witness : CalcExprC n → Term n
  | var j => Term.var j
  | const P => P.witness.subst Fin.elim0
  | unary P F => P.witness.subst ![witness F]
  | binary P F G => P.witness.subst ![witness F, witness G]

end CalcExprC

/-- A term is evaluated from its leaves upwards, in the arithmetic of Appendix A: if the terms
`σ j` are defined at the point `a` and have the complex values `b j`, then `t` with the `σ j` in
place of its variables has at `a` the value, or the undefinedness, of `t` at the point `b`. -/
theorem evalX_subst {k n : ℕ} (t : Term k) {σ : Fin k → Term n} {a : Fin n → ℂ} {b : Fin k → ℂ}
    (h : ∀ j, evalX (σ j) a = some (num (b j))) : evalX (t.subst σ) a = evalX t b := by
  induction t with
  | one => rfl
  | var j => exact h j
  | eml s t hs ht => simp only [Term.subst, evalX, hs, ht]

variable {n : ℕ}

/-- Theorem 2 for the term of a complex expression: if `F` is defined at the point `a` and
`F(a) = z`, then its term is defined at `a` and has the value `z`. -/
theorem evalX_witness (F : CalcExprC n) (a : Fin n → ℂ) (z : ℂ) (h : F.value a = some z) :
    evalX F.witness a = some (num z) := by
  induction F generalizing z with
  | var j =>
    simp only [CalcExprC.value, Option.some.injEq] at h
    subst h
    rfl
  | const P =>
    simp only [CalcExprC.value, Option.some.injEq] at h
    subst h
    rw [CalcExprC.witness, evalX_subst (b := Fin.elim0) _ (fun j => j.elim0)]
    exact theorem1a_const P _
  | unary P F hF =>
    cases hu : F.value a with
    | none => simp [CalcExprC.value, hu] at h
    | some u =>
      by_cases hd : u ∈ P.dom
      · simp only [CalcExprC.value, hu, hd, ite_true, Option.some.injEq] at h
        subst h
        have hσ : ∀ j, evalX (![F.witness] j) a = some (num ((![u] : Fin 1 → ℂ) j)) := by
          intro j
          fin_cases j
          exact hF u hu
        rw [CalcExprC.witness, evalX_subst _ hσ]
        exact theorem1a_unary P u hd
      · simp [CalcExprC.value, hu, hd] at h
  | binary P F G hF hG =>
    cases hu : F.value a with
    | none => simp [CalcExprC.value, hu] at h
    | some u =>
      cases hv : G.value a with
      | none => simp [CalcExprC.value, hu, hv] at h
      | some v =>
        by_cases hd : (u, v) ∈ P.dom
        · simp only [CalcExprC.value, hu, hv, hd, ite_true, Option.some.injEq] at h
          subst h
          have hσ : ∀ j, evalX (![F.witness, G.witness] j) a
              = some (num ((![u, v] : Fin 2 → ℂ) j)) := by
            intro j
            fin_cases j
            · exact hF u hu
            · exact hG v hv
          rw [CalcExprC.witness, evalX_subst _ hσ]
          exact theorem1a_binary P u v hd
        · simp [CalcExprC.value, hu, hv, hd] at h

/-- **Theorem 2 for complex arguments** (the remark after Theorem 1a). For every expression `F`
over the complex primitives of Table A.2 there is an EML term `W` such that, at every point `a`
where `F` is defined with complex intermediate values, `W` is defined in the arithmetic of
Appendix A and its value is the complex number `F(a)`. -/
theorem theorem2a (F : CalcExprC n) :
    ∃ W : Term n, ∀ (a : Fin n → ℂ) (z : ℂ), F.value a = some z →
      evalX W a = some (num z) :=
  ⟨F.witness, evalX_witness F⟩

end EML

#print axioms EML.theorem2a
