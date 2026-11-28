module

public import EML.Theorem1

/-!
# Theorem 2: completeness of `{1, eml}` for the calculator class

* `CalcExpr n` — Definition 3: the calculator expressions in the variables `x_0, …, x_{n-1}`,
  built from the variables and the 5 constants by the 28 primitives of Table 1.
* `CalcExpr.value F a` — Definition 3: the partial real function of the expression `F`.
  It is an `Option ℝ`: `some r` means "`a` lies in the natural domain of `F`, and `F(a) = r`",
  `none` means that some primitive of `F` is applied outside its natural domain.
* `Term.subst t σ` — the term `t` with the terms `σ j` in place of its variables.
* `eval_subst` — a term is evaluated from its leaves upwards.
* `CalcExpr.witness F` — the term `W_F`: every primitive `P` of `F` replaced by its term
  `P.witness` of Theorem 1.
* `theorem2` — at every point of the natural domain of `F` the term `W_F` is defined and its
  value is `F(a)`.
-/

@[expose] public section

namespace EML

open Complex Term

/-- **Definition 3.** Calculator expressions in the variables `x_0, …, x_{n-1}`. -/
inductive CalcExpr (n : ℕ) : Type
  /-- the variable `x_j` -/
  | var : Fin n → CalcExpr n
  /-- one of the 5 constants of Table 1 -/
  | const : Prim0 → CalcExpr n
  /-- a unary primitive of Table 1, applied to an expression -/
  | unary : Prim1 → CalcExpr n → CalcExpr n
  /-- a binary primitive of Table 1, applied to two expressions -/
  | binary : Prim2 → CalcExpr n → CalcExpr n → CalcExpr n

namespace CalcExpr

variable {n : ℕ}

open Classical in
/-- **Definition 3.** The value of an expression at the real point `a`: `F = P(F₁, …, F_k)` is
defined at `a` iff all `F_j` are defined at `a` and their values lie in the natural domain of
`P`; its value is then `P(F₁(a), …, F_k(a))`. -/
noncomputable def value : CalcExpr n → (Fin n → ℝ) → Option ℝ
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

/-- The natural domain of an expression: the points where it is defined. -/
def dom (F : CalcExpr n) : Set (Fin n → ℝ) := {a | (value F a).isSome}

end CalcExpr

/-- The term `t` with the terms `σ j` in place of its variables `x_j`. -/
def Term.subst {k n : ℕ} : Term k → (Fin k → Term n) → Term n
  | one, _ => one
  | var j, σ => σ j
  | eml s t, σ => eml (subst s σ) (subst t σ)

/-- A term is evaluated from its leaves upwards (Definition 2): if the terms `σ j` are defined
at the point `a` and have the values `b j`, then `t` with the `σ j` in place of its variables
has at `a` the value, or the undefinedness, of `t` at the point `b`. -/
theorem eval_subst {k n : ℕ} (t : Term k) {σ : Fin k → Term n} {a : Fin n → ℂ} {b : Fin k → ℂ}
    (h : ∀ j, eval (σ j) a = some (b j)) : eval (t.subst σ) a = eval t b := by
  induction t with
  | one => rfl
  | var j => exact h j
  | eml s t hs ht => simp only [Term.subst, eval, hs, ht]

namespace CalcExpr

variable {n : ℕ}

/-- The term `W_F` of Theorem 2: every primitive `P` of `F` is replaced by the term
`P.witness` of Theorem 1, with the terms of the arguments in place of its variables. -/
def witness : CalcExpr n → Term n
  | var j => Term.var j
  | const P => P.witness.subst Fin.elim0
  | unary P F => P.witness.subst ![witness F]
  | binary P F G => P.witness.subst ![witness F, witness G]

end CalcExpr

variable {n : ℕ}

/-- Theorem 2 for the term `W_F`: if `F` is defined at the real point `a` and `F(a) = x`, then
`W_F` is defined at `a` and `W_F(a) = x`. -/
theorem eval_witness (F : CalcExpr n) (a : Fin n → ℝ) (x : ℝ) (h : F.value a = some x) :
    eval F.witness (fun j => (a j : ℂ)) = some (x : ℂ) := by
  induction F generalizing x with
  | var j =>
    simp only [CalcExpr.value, Option.some.injEq] at h
    subst h
    rfl
  | const P =>
    simp only [CalcExpr.value, Option.some.injEq] at h
    subst h
    rw [CalcExpr.witness, eval_subst (b := Fin.elim0) _ (fun j => j.elim0)]
    exact theorem1_const P _
  | unary P F hF =>
    cases hu : F.value a with
    | none => simp [CalcExpr.value, hu] at h
    | some u =>
      by_cases hd : u ∈ P.dom
      · simp only [CalcExpr.value, hu, hd, ite_true, Option.some.injEq] at h
        subst h
        have hσ : ∀ j, eval (![F.witness] j) (fun j => (a j : ℂ))
            = some ((![(u : ℂ)] : Fin 1 → ℂ) j) := by
          intro j
          fin_cases j
          exact hF u hu
        rw [CalcExpr.witness, eval_subst _ hσ]
        exact theorem1_unary P u hd
      · simp [CalcExpr.value, hu, hd] at h
  | binary P F G hF hG =>
    cases hu : F.value a with
    | none => simp [CalcExpr.value, hu] at h
    | some u =>
      cases hv : G.value a with
      | none => simp [CalcExpr.value, hu, hv] at h
      | some v =>
        by_cases hd : (u, v) ∈ P.dom
        · simp only [CalcExpr.value, hu, hv, hd, ite_true, Option.some.injEq] at h
          subst h
          have hσ : ∀ j, eval (![F.witness, G.witness] j) (fun j => (a j : ℂ))
              = some ((![(u : ℂ), (v : ℂ)] : Fin 2 → ℂ) j) := by
            intro j
            fin_cases j
            · exact hF u hu
            · exact hG v hv
          rw [CalcExpr.witness, eval_subst _ hσ]
          exact theorem1_binary P u v hd
        · simp [CalcExpr.value, hu, hv, hd] at h

/-- **Theorem 2** (completeness of `{1, eml}` for the calculator class). For every calculator
expression `F` in `n` variables there is an EML term `W` in `n` variables such that, at every
point `a` of the natural domain of `F`, `W` is defined and its value is the real number `F(a)`.
The term is `F.witness`: `F` with every primitive replaced by its term of Theorem 1. -/
theorem theorem2 (F : CalcExpr n) :
    ∃ W : Term n, ∀ (a : Fin n → ℝ) (x : ℝ), F.value a = some x →
      eval W (fun j => (a j : ℂ)) = some (x : ℂ) :=
  ⟨F.witness, eval_witness F⟩

/-! ## Two expressions in which an argument of a primitive passes through `0`

`y · 1` and `(x - x) + 1` are defined at every real point, and so are their terms: the
multiplication and addition witnesses of Theorem 1 accept all real arguments. -/

/-- `y · 1` as an expression in the variables `x`, `y`. -/
def y_times_one : CalcExpr 2 := .binary .times (.var 1) (.const .one)

/-- `(x - x) + 1` as an expression in the variables `x`, `y`. -/
def x_minus_x_plus_one : CalcExpr 2 :=
  .binary .plus (.binary .subtract (.var 0) (.var 0)) (.const .one)

example (a : Fin 2 → ℝ) :
    eval y_times_one.witness (fun j => (a j : ℂ)) = some ((a 1 * 1 : ℝ) : ℂ) :=
  eval_witness _ a _ (by simp [y_times_one, CalcExpr.value, Prim2.dom, Prim2.val, Prim0.val])

example (a : Fin 2 → ℝ) :
    eval x_minus_x_plus_one.witness (fun j => (a j : ℂ)) = some ((a 0 - a 0 + 1 : ℝ) : ℂ) :=
  eval_witness _ a _
    (by simp [x_minus_x_plus_one, CalcExpr.value, Prim2.dom, Prim2.val, Prim0.val])

end EML

#print axioms EML.theorem2
