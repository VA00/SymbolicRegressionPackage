(* ::Package:: *)

(* verify_theorem1a.wl

   Theorem 1a (Appendix A, principal values on the complex plane): the witnesses
   of Table A.3 as Wolfram Language code, with exact checks.  Everything is built
   from eml and the constant 1.

   Run:   wolframscript -file verify_theorem1a.wl
   or, in a notebook:   Get["verify_theorem1a.wl"]   and then e.g.  Warcsin[2]

   Names follow verify_theorem1.wl.  E, N, D are protected in the Wolfram
   Language, so the terms E, N, D of the paper are the Greek capitals
   \[CapitalEpsilon], \[CapitalNu], \[CapitalDelta].

   Semantics (Definition A.1, Table A.1).  eml[z, w] = Exp[z] - Log[w] with the
   built-in Exp and Log on C + {-Infinity, +Infinity}:

        Exp[-Infinity] = 0,  Exp[Infinity] = Infinity,
        Log[0] = -Infinity,  Log[Infinity] = Log[-Infinity] = Infinity,

   and Infinity - Infinity = Indeterminate (the term is then undefined).

   Method.

   1. The theorem is about exact numbers.  The kernel applies Log[0] = -Infinity
      only to the literal 0; a zero in disguise such as
            -E^(-I Pi + Log[I Pi]) - I Pi
      is left inside Log.  Therefore eml (a) tests a numeric second argument with
      PossibleZeroQ and (b) passes a numeric value through Simplify (2 s limit).
      Arguments are evaluated first, so a witness is computed from the leaves
      upwards and every intermediate value is short before it is used again.

   2. The checks are at exact points of the domains of Table A.2: Gaussian
      rationals, Pi, E, Sqrt[2], Log[2], the branch points, and points on every
      branch cut.  The reference value is the formula of the paper written with
      the built-in Exp, Log and Sqrt; the built-in inverse functions (ArcSin, ...)
      are compared in addition.  A check counts as "exact" only if the difference
      simplifies to the integer 0, or if it is smaller than 1 in modulus and its
      exponential simplifies to 1.  "numeric" means: equal to 60 digits, but not
      reduced to 0.

   3. No message of the kernel is suppressed.  A run prints two kinds.
      Infinity::indet appears twice, at the two cells Infinity - Infinity of
      Table A.1, where the term is undefined.  N::meprec appears once, at
      artanh: there a difference to the reference value is exactly 0 in a form
      that FullSimplify does not reduce, and the kernel cannot find its
      numerical value; the difference is then shown to be 0 by its exponential.

   What this file does not do: it proves nothing for all z.  The proof is in the
   paper and in Lean (EML/Theorem1a.lean). *)

ClearAll[eml, e, L, S, zero, neg, two, half, M, mipi, img, pi, pi2, twoi, ipi, twoipi, pinf, minf, mimg,
  Winv, Wx, Wminus, Wneg, Wplus, Wsqr, Whalf, Wavg, Wdiv, Wpow, Wsqrt, Whypot, Wln, Wlog, Wsinh, Wcosh,
  Wcos, Wsin, Wtan, Wtanh, Warcsin, Warccos, Warsinh, Warcosh, Warctan, Wartanh, J, WinvLean, WlnLean,
  WlnWolfram, dcut];
ClearAll["\[CapitalEpsilon]", "\[CapitalNu]", "\[CapitalDelta]", "W\[Sigma]"];

emlCount = 0;
eml[z_, w_] := (emlCount++;
   With[{w0 = If[NumericQ[w] && w =!= 0 && PossibleZeroQ[w], 0, w]},
    With[{val = Exp[z] - Log[w0]},
     If[NumericQ[val], TimeConstrained[Simplify[val], 2, val], val]]]);

(* Section 2 (Theorem 1): Lemma 1, Corollary 2, Lemmas 3-8 *)
\[CapitalEpsilon][z_] := eml[z, 1];
e = eml[1, 1];
L[z_] := eml[1, \[CapitalEpsilon][eml[1, z]]];
S[u_, v_] := eml[L[u], \[CapitalEpsilon][v]];
\[CapitalNu][z_] := S[1, S[e, S[S[e, 1], z]]];
zero = L[1];
neg = \[CapitalNu][1];
two = S[1, neg];
half = \[CapitalEpsilon][\[CapitalNu][L[two]]];
M[u_, v_] := \[CapitalEpsilon][S[L[u], \[CapitalNu][L[v]]]];
\[CapitalDelta][u_, v_] := \[CapitalEpsilon][S[L[u], L[v]]];
mipi = eml[1, \[CapitalNu][\[CapitalEpsilon][e]]];
img = \[CapitalEpsilon][\[CapitalNu][\[CapitalDelta][mipi, two]]];
pi = M[img, mipi];
pi2 = \[CapitalDelta][pi, two];
twoi = M[two, img];
ipi = M[img, pi];
twoipi = M[two, ipi];

(* Appendix A, Step 1: Lemmas A.1, A.3, A.4 *)
pinf = eml[1, zero];
minf = eml[1, pinf];
Winv[z_] := \[CapitalEpsilon][eml[minf, z]];
Wx[z_, w_] := Winv[\[CapitalEpsilon][eml[L[eml[minf, z]], w]]];

(* Step 2: Lemma A.5, Corollary A.6 *)
mimg = Wx[neg, img];
Wminus[z_, w_] := Wx[mimg, S[Wx[img, S[z, w]], Wx[img, S[w, w]]]];
Wneg[z_] := Wx[neg, z];
Wplus[z_, w_] := Wminus[z, Wneg[w]];
Wsqr[z_] := Wx[z, z];
Whalf[z_] := Wx[half, z];
Wavg[z_, w_] := Whalf[Wplus[z, w]];
Wdiv[z_, w_] := Wx[z, Winv[w]];

(* Step 3: Lemma A.7, Corollary A.8, Lemmas A.9, A.10 *)
Wpow[z_, w_] := Winv[\[CapitalEpsilon][Wx[w, eml[minf, z]]]];
Wsqrt[z_] := Wpow[z, half];
Whypot[z_, w_] := Wsqrt[Wplus[Wsqr[z], Wsqr[w]]];
Wln[z_] := Wneg[eml[minf, z]];
Wlog[b_, z_] := Wdiv[Wln[z], Wln[b]];

(* Step 4: Lemma A.11 *)
W\[Sigma][z_] := Winv[Wplus[1, \[CapitalEpsilon][Wneg[z]]]];
Wsinh[z_] := Wdiv[Wminus[\[CapitalEpsilon][z], \[CapitalEpsilon][Wneg[z]]], two];
Wcosh[z_] := Wdiv[Wplus[\[CapitalEpsilon][z], \[CapitalEpsilon][Wneg[z]]], two];
Wcos[z_] := Wdiv[Wplus[\[CapitalEpsilon][Wx[img, z]], \[CapitalEpsilon][Wx[mimg, z]]], two];
Wsin[z_] := Wcos[Wminus[z, pi2]];
Wtan[z_] := Wdiv[Wsin[z], Wcos[z]];
Wtanh[z_] := Wdiv[Wsinh[z], Wcosh[z]];
Warcsin[z_] := Wx[mimg, Wln[Wplus[Wx[img, z], Wsqrt[Wminus[1, Wsqr[z]]]]]];
Warccos[z_] := Wminus[pi2, Warcsin[z]];
Warsinh[z_] := Wln[Wplus[z, Wsqrt[Wplus[1, Wsqr[z]]]]];
Warcosh[z_] := Wln[Wplus[z, Wx[Wsqrt[Wminus[z, 1]], Wsqrt[Wplus[z, 1]]]]];
Warctan[z_] := Wdiv[Wminus[Wln[Wplus[1, Wx[img, z]]], Wln[Wminus[1, Wx[img, z]]]], twoi];
Wartanh[z_] := Wdiv[Wminus[Wln[Wplus[1, z]], Wln[Wminus[1, z]]], two];

(* Section A.4: Lemma A.12 and the examples *)
J[z_] := Wx[Winv[z], z];
WinvLean[z_] := Wx[Winv[z], J[z]];
WlnLean[z_] := Wx[Wln[z], J[z]];
WlnWolfram[z_] := eml[L[WlnLean[z]], Winv[J[z]]];
dcut[z_] := Wdiv[Wminus[Wln[z], L[z]], twoipi];

(* ------------------------------------------------------------------ *)
(* Exact checks.  verdict: "exact", "numeric", or {"FAIL", got, want}. *)
infQ[x_] := ! FreeQ[x, DirectedInfinity | Indeterminate | Undefined];
SetAttributes[verdict, HoldFirst];
verdict[got_, want_] := Module[{g = TimeConstrained[got, 120, "TIMEOUT"], d},
   Which[
    g === "TIMEOUT", {"FAIL", "TIMEOUT", want},
    g === want, "exact",
    infQ[g] || infQ[want], {"FAIL", g, want},
    True,
    d = TimeConstrained[FullSimplify[g - want], 20, $Failed];
    If[d =!= 0, d = TimeConstrained[FullSimplify[TrigToExp[g - want]], 20, $Failed]];
    If[d =!= 0, d = TimeConstrained[FullSimplify[ComplexExpand[TrigToExp[g - want]]], 20, $Failed]];
    If[d =!= 0, d = TimeConstrained[FullSimplify[FunctionExpand[g - want]], 20, $Failed]];
    (* a difference d with |d| < 1 and exp(d) = 1 is 0 *)
    If[d =!= 0 && (Abs[N[g - want, 30]] < 1) === True &&
      TimeConstrained[RootReduce[FullSimplify[Exp[TrigToExp[g - want]]]], 20, $Failed] === 1, d = 0];
    Which[d === 0, "exact",
     (Abs[N[g - want, 60]] < 10^-50) === True, "numeric",
     True, {"FAIL", g, want}]]];

allGood = True; nExact = 0; nNumeric = 0; nFail = 0;
report[name_, verdicts_List, pts_List : {}] := Module[{bad = Cases[verdicts, {"FAIL", __}], num = Count[verdicts, "numeric"]},
   nExact += Count[verdicts, "exact"]; nNumeric += num; nFail += Length[bad];
   Print[StringPadRight[name, 46], StringPadLeft[ToString[Length[verdicts]], 5], " points   ",
    Which[bad =!= {}, "FAILURES: " <> ToString[Length[bad]],
     num > 0, ToString[Length[verdicts] - num] <> " exact, " <> ToString[num] <> " numeric" <>
       If[pts === {}, "", " at " <> ToString[Pick[pts, verdicts, "numeric"], InputForm]],
     True, "all exact"]];
   If[bad =!= {},
    allGood = False;
    Do[If[MatchQ[verdicts[[k]], {"FAIL", __}],
      Print["      at ", If[pts === {}, k, InputForm[pts[[k]]]], ":  got ", InputForm[Short[verdicts[[k, 2]], 2]], ",  want ", InputForm[verdicts[[k, 3]]]]],
     {k, Length[verdicts]}]]];
SetAttributes[unary, HoldAll];
unary[name_, W_, F_, xs_] := report[name, Table[verdict[W[x], F[x]], {x, xs}], xs];
SetAttributes[binary, HoldAll];
binary[name_, W_, F_, ps_] := report[name, Table[verdict[W @@ p, F @@ p], {p, ps}], ps];

(* The primitives as the paper defines them (Table A.2): formulas in Exp, Log, Sqrt. *)
pSig[z_] := 1/(1 + Exp[-z]);
pArcsin[z_] := -I Log[I z + Sqrt[1 - z^2]];
pArccos[z_] := Pi/2 - pArcsin[z];
pArsinh[z_] := Log[z + Sqrt[1 + z^2]];
pArcosh[z_] := Log[z + Sqrt[z - 1] Sqrt[z + 1]];
pArctan[z_] := (Log[1 + I z] - Log[1 - I z])/(2 I);
pArtanh[z_] := (Log[1 + z] - Log[1 - z])/2;
pPow[z_, w_] := If[z === 0, 0, Exp[w Log[z]]];
pLog[b_, z_] := Log[z]/Log[b];
pHypot[z_, w_] := Sqrt[z^2 + w^2];

t0 = AbsoluteTime[];

Print["--- the operator on C + {-Infinity, +Infinity} (Table A.1) ---"];
report["12 cells of the table",
  {verdict[eml[2 + 3 I, -5], Exp[2 + 3 I] - (Log[5] + I Pi)], verdict[eml[2 + 3 I, 0], Infinity],
   verdict[eml[2 + 3 I, -Infinity], -Infinity], verdict[eml[2 + 3 I, Infinity], -Infinity],
   verdict[eml[-Infinity, -5], -(Log[5] + I Pi)], verdict[eml[-Infinity, 0], Infinity],
   verdict[eml[-Infinity, -Infinity], -Infinity], verdict[eml[-Infinity, Infinity], -Infinity],
   verdict[eml[Infinity, -5], Infinity], verdict[eml[Infinity, 0], Infinity],
   verdict[eml[Infinity, -Infinity], Indeterminate], verdict[eml[Infinity, Infinity], Indeterminate]}];

Print["--- constants (Section 2 and Lemma A.1) ---"];
report["e, 0, -1, 2, 1/2", MapThread[verdict, {{e, zero, neg, two, half}, {E, 0, -1, 2, 1/2}}]];
report["-i Pi, i, Pi, Pi/2, 2i, i Pi, 2 Pi i, -i",
  MapThread[verdict, {{mipi, img, pi, pi2, twoi, ipi, twoipi, mimg}, {-I Pi, I, Pi, Pi/2, 2 I, I Pi, 2 Pi I, -I}}]];
report["+Infinity, -Infinity", MapThread[verdict, {{pinf, minf}, {Infinity, -Infinity}}]];

Print["--- Lemma A.2: E, L, S ---"];
zs0 = {2, 1/3, -1, -5/2, I, -I, 3 + 4 I, -3 + 4 I, -3 - 4 I, 3 - 4 I};
unw[w_] := Ceiling[(Im[w] - Pi)/(2 Pi)];
report["(a) E(-Inf) = 0, E(+Inf) = +Inf",
  {verdict[\[CapitalEpsilon][-Infinity], 0], verdict[\[CapitalEpsilon][Infinity], Infinity]}];
report["(b) L(0) = -Inf, L(-Inf) = L(+Inf) = +Inf",
  {verdict[L[0], -Infinity], verdict[L[-Infinity], Infinity], verdict[L[Infinity], Infinity]}];
unary["(b) exp(L(u)) = u", Exp[L[#]] &, Identity, Join[{0}, zs0]];
ws = {0, 1, -5/2, I, 3 I, 4 I, 7 I, -7 I, 20 I, -20 I, I Pi, -I Pi, 3 I Pi, -3 I Pi, 5 I Pi, 2 + I Pi, 2 - I Pi, -1 + 3 I Pi, 1 + 10 I};
binary["(c) S(u,v) = u - v + 2 Pi i K(v)", S, (#1 - #2 + 2 Pi I unw[#2]) &, Tuples[{{0, 1, -2 + I}, ws}]];

Print["--- unary primitives on their domains (Table A.2) ---"];
zs = {0, 1, -1, 2, -2, 1/2, -1/2, 3, -3, 10, -10, I, -I, 2 I, -2 I, I/2, -I/2, 3 I, -3 I,
   1 + I, 1 - I, -1 + I, -1 - I, 2 + 3 I, -2 + 3 I, 2 - 3 I, -2 - 3 I, 1/3 + 5 I, -7/2 - 4 I,
   7 I, -7 I, 20 I, -20 I, Pi, -Pi, Pi/2, -Pi/2, I Pi, -I Pi, 2 + I Pi, 2 - I Pi, 3 I Pi, -3 I Pi,
   E, 1/E, Sqrt[2], -Sqrt[2], I Sqrt[2], (1 + I)/Sqrt[2], Log[2], -Log[2] + I Pi, 5/3, -5/3, 5 I/3, -5 I/3};
nz = DeleteCases[zs, 0];
unary["exp", \[CapitalEpsilon], Exp, zs];
unary["minus", Wneg, Minus, zs];
unary["half", Whalf, (#/2) &, zs];
unary["sqr", Wsqr, (#^2) &, zs];
unary["inv, z != 0", Winv, (1/#) &, nz];
unary["sqrt, incl. 0 and the cut", Wsqrt, Sqrt, zs];
unary["ln, z != 0, incl. the cut", Wln, Log, nz];
unary["sigma, e^z != -1", W\[Sigma], pSig, DeleteCases[zs, I Pi | -I Pi | 3 I Pi | -3 I Pi]];
unary["sinh", Wsinh, Sinh, zs];
unary["cosh", Wcosh, Cosh, zs];
unary["tanh, cosh z != 0", Wtanh, Tanh, Join[zs, {I Pi/4, I Pi/3, 3 I Pi/4}]];
unary["sin", Wsin, Sin, zs];
unary["cos", Wcos, Cos, zs];
unary["tan, cos z != 0", Wtan, Tan, Join[DeleteCases[zs, Pi/2 | -Pi/2], {Pi/4, Pi/3, 3 Pi/4}]];
unary["arcsin, incl. +-1 and both cuts", Warcsin, pArcsin, zs];
unary["arccos, incl. +-1 and both cuts", Warccos, pArccos, zs];
unary["arctan, z != +-i, incl. both cuts", Warctan, pArctan, DeleteCases[zs, I | -I]];
unary["arsinh, incl. +-i and both cuts", Warsinh, pArsinh, zs];
unary["arcosh, incl. +-1 and the cut", Warcosh, pArcosh, zs];
unary["artanh, z != +-1, incl. both cuts", Wartanh, pArtanh, DeleteCases[zs, 1 | -1]];

Print["--- binary primitives on their domains ---"];
as = {0, 1, -1, 2, -3, 1/2, I, -I, 1 + I, -2 - 3 I, 7 I, -7 I, I Pi, -I Pi, 2 + 3 I Pi, E, -5/3};
ps = Tuples[as, 2];
binary["+", Wplus, Plus, ps];
binary["-", Wminus, Subtract, ps];
binary["x", Wx, Times, ps];
binary["avg", Wavg, ((#1 + #2)/2) &, ps];
binary["hypot = Sqrt[z^2 + w^2]", Whypot, pHypot, ps];
binary["/, w != 0", Wdiv, Divide, Select[ps, #[[2]] =!= 0 &]];
binary["pow, z != 0 or Re w > 0", Wpow, pPow, Select[ps, ! (#[[1]] === 0 && Re[#[[2]]] <= 0) &]];
bs = {2, 1/2, 10, E, 1/E, -2, -1/2, -1, I, -I, 1 + I, -2 - 3 I, 5/3};
binary["log_b z, b != 0, b != 1, z != 0", Wlog, pLog, Tuples[{bs, DeleteCases[as, 0]}]];

Print["--- 200 random Gaussian rationals (seed 20261119) ---"];
SeedRandom[20261119];
rnd[] := RandomInteger[{-12, 12}]/RandomInteger[{1, 4}] + I RandomInteger[{-12, 12}]/RandomInteger[{1, 4}];
rz = Table[rnd[], {200}]; rw = Table[rnd[], {200}];
rps = Transpose[{rz, rw}];
binary["- (random)", Wminus, Subtract, rps];
binary["+ (random)", Wplus, Plus, rps];
binary["x (random)", Wx, Times, rps];
binary["/ (random)", Wdiv, Divide, Select[rps, #[[2]] =!= 0 &]];
binary["pow (random)", Wpow, pPow, Select[Take[rps, 60], #[[1]] =!= 0 &]];
binary["log (random)", Wlog, pLog, Select[Take[rps, 60], #[[1]] =!= 0 && #[[1]] =!= 1 && #[[2]] =!= 0 &]];
r40 = Take[rz, 40];
unary["sqrt (random)", Wsqrt, Sqrt, r40];
unary["ln (random)", Wln, Log, DeleteCases[r40, 0]];
unary["sin (random)", Wsin, Sin, r40];
unary["cosh (random)", Wcosh, Cosh, r40];
unary["arcsin (random)", Warcsin, pArcsin, r40];
unary["arccos (random)", Warccos, pArccos, r40];
unary["arctan (random)", Warctan, pArctan, DeleteCases[r40, I | -I]];
unary["arsinh (random)", Warsinh, pArsinh, r40];
unary["arcosh (random)", Warcosh, pArcosh, r40];
unary["artanh (random)", Wartanh, pArtanh, DeleteCases[r40, 1 | -1]];

Print["--- the formulas of Table A.2 against the built-in inverse functions, same points ---"];
unary["ArcSin", pArcsin, ArcSin, zs];
unary["ArcCos", pArccos, ArcCos, zs];
unary["ArcTan, z != +-i", pArctan, ArcTan, DeleteCases[zs, I | -I]];
unary["ArcSinh", pArsinh, ArcSinh, zs];
unary["ArcCosh", pArcosh, ArcCosh, zs];
unary["ArcTanh, z != +-1", pArtanh, ArcTanh, DeleteCases[zs, 1 | -1]];

Print["--- Section A.4: values at the exceptional points ---"];
unary["Lemma A.12: J(z) = 1 (z != 0), J(0) = 0", J, If[# === 0, 0, 1] &, Join[{0, 1}, zs0]];
unary["inv with inv(0) = 0 (Lean)", WinvLean, If[# === 0, 0, 1/#] &, Join[{0, 1}, zs0]];
unary["Log with Log(0) = 0 (Lean)", WlnLean, If[# === 0, 0, Log[#]] &, Join[{0, 1}, zs0]];
unary["Log with Log(0) = -Infinity (Wolfram)", WlnWolfram, Log, Join[{0, 1}, zs0]];
unary["d(z) = 1 on the negative real axis, else 0", dcut, If[Im[#] == 0 && Re[#] < 0, 1, 0] &, zs0];

Print[];
Print["eml nodes evaluated: ", emlCount, ";  time ", Round[AbsoluteTime[] - t0], " s"];
Print["checks: ", nExact + nNumeric + nFail, "  (exact ", nExact, ", numeric ", nNumeric, ", failed ", nFail, ")"];
Print[If[allGood, "RESULT: no failure.", "RESULT: THERE ARE FAILURES - see above."]];
