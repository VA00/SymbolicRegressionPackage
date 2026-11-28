(* ::Package:: *)

(* verify_theorem1b_1c.wl

   Theorem 1b (Appendix B, strict real arithmetic) and Theorem 1c (Appendix C,
   extended real arithmetic with ln 0 = -Infinity): the witnesses of Table B.1,
   Table B.2 and Appendix C as Wolfram Language code, with exact checks.
   Everything is built from eml and the constant 1.

   Run:   wolframscript -file verify_theorem1b_1c.wl
   or, in a notebook:   Get["verify_theorem1b_1c.wl"]   and then e.g.  wx[-2, 1/3]

   Names follow the paper: E, L, F, B, G, Q are the auxiliary terms, w... the
   witnesses.  E is protected in the Wolfram Language, so the term E is the
   Greek capital \[CapitalEpsilon].

   Semantics.  The same terms are evaluated in two arithmetics, chosen by mode:

     "strict"    Definition B.1:  eml[u, v] = Exp[u] - Log[v] for real u and
                 real v > 0; anything else is undefined.
     "extended"  Definition C.1, Table C.1:  in addition  Log[0] = -Infinity,
                 Log[Infinity] = Infinity, Exp[-Infinity] = 0,
                 Exp[Infinity] = Infinity; a negative second argument
                 (also -Infinity) and Infinity - Infinity are undefined.

   An undefined node stops the evaluation and the term has the value
   "undefined".  Nothing is taken from the complex logarithm: the sign of the
   second argument is tested before Log is applied.

   Method.  The theorems are about exact numbers.  eml passes its value through
   Simplify (2 s limit); arguments are evaluated first, so a witness is computed
   from the leaves upwards.  A check counts as "exact" only if the difference to
   the built-in function simplifies to the integer 0; "numeric" means equal to
   60 digits, but not reduced to 0.

   The sign of the second argument.  It is taken from the value to 30 digits
   when that value exists.  The witnesses pass through exact numbers that have
   none: sigma(-3) contains the positive number 1/(... + e^(e^21 ...)), below
   10^(-10^17).  For such a number the sign is found from its form: a power of
   e is positive, a product has the product of the signs, and in a sum the
   terms are compared through their logarithms and the largest ones decide
   (sl below).  If this does not decide, the evaluation stops with
   "sign not decided", which counts as a failure.

   Messages.  No message of the kernel is suppressed.  General::ovfl and
   N::meprec appear, first at the auxiliary term Q: the kernel is asked for
   30 digits of a number that it cannot represent, such as
   E^(-E^4 + E^(E^2 (-2 + E^2))), and says so; the sign is then found from
   the form of the number, as described above.  The kernel prints a message
   three times and then General::stop.

   What this file does not do: it proves nothing for all x, and nothing about
   the functions that have no witness.  The proofs are in the paper and in Lean
   (EML/Theorem1b.lean, EML/Theorem1c.lean). *)

ClearAll[eml, mode, sgn, sl, num, simp, e, zero, neg, two, L, F, B, G, Q, wplus, wminus, wneg, wx, wsqr, winv, wdiv,
  whalf, wavg, wpow, wlog, wsinh, wcosh, wtanh, warsinh, wartanh, wsqrtPos, whypotOff, warcoshGt1,
  pinf, minf, wsqrtBar, whypotBar, warcoshBar, wabs, wrelu, wmax];
ClearAll["\[CapitalEpsilon]", "w\[Sigma]"];

mode = "strict";
emlCount = 0;
simp[v_] := TimeConstrained[Simplify[v], 2, v];
num[a_] := With[{n = N[a, 30]}, If[NumberQ[n] && Im[n] == 0, n, Throw["sign not decided", eml]]];
sl[t_Times] := With[{l = sl /@ (List @@ t)}, {Times @@ l[[All, 1]], Total[l[[All, 2]]]}];      (* {sign, ln|t|} *)
sl[Power[E, a_]] := {1, num[a]};
sl[Power[b_, n_Integer]] := With[{l = sl[b]}, {If[EvenQ[n], 1, l[[1]]], n l[[2]]}];
sl[t_Plus] := Module[{l = sl /@ (List @@ t), m, s},
   m = Max[l[[All, 2]]];
   s = Total[If[#[[2]] - m < -200, 0., #[[1]] Exp[N[#[[2]] - m]]] & /@ l];
   If[Abs[s] < 10^-6, Throw["sign not decided", eml]];
   {Sign[s], m + Log[Abs[s]]}];
sl[t_] := With[{n = num[t]}, If[n == 0, Throw["sign not decided", eml]]; {Sign[n], Log[Abs[n]]}];
sgn[v_] := Which[v === 0, 0, v === Infinity, 1, v === -Infinity, -1,
   True, With[{n = N[v, 30]},
     Which[NumberQ[n] && n != 0 && Precision[n] > 10, Sign[n],
      NumberQ[n] && PossibleZeroQ[v], 0,
      True, First[sl[v]]]]];
eml[u_, v_] := (emlCount++;
   If[mode === "strict",
    If[u =!= Infinity && u =!= -Infinity && v =!= Infinity && v =!= -Infinity && sgn[v] > 0,
     simp[Exp[u] - Log[v]], Throw["undefined", eml]],
    Which[
     v === -Infinity || sgn[v] < 0, Throw["undefined", eml],
     sgn[v] == 0, Infinity,
     v === Infinity, If[u === Infinity, Throw["undefined", eml], -Infinity],
     u === Infinity, Infinity,
     u === -Infinity, simp[-Log[v]],
     True, simp[Exp[u] - Log[v]]]]);

(* Appendix B, Step 1: Lemma B.1 *)
\[CapitalEpsilon][t_] := eml[t, 1];
L[u_] := eml[1, \[CapitalEpsilon][eml[1, u]]];
F[t_, v_] := eml[t, \[CapitalEpsilon][v]];
B[t_] := L[F[t, t]];
wplus[x_, y_] := F[x, F[B[x], y]];
wminus[x_, y_] := F[B[y], F[y, x]];
e = \[CapitalEpsilon][1];
zero = L[1];
wneg[x_] := wminus[zero, x];
neg = wminus[zero, 1];
two = wplus[1, 1];

(* Step 2: Lemmas B.2, B.3 *)
G[u_, y_] := wminus[\[CapitalEpsilon][wplus[u, y]], \[CapitalEpsilon][wplus[u, B[y]]]];
wx[x_, y_] := wminus[G[x, y], G[B[x], y]];
wsqr[y_] := wx[y, y];
winv[y_] := wx[y, \[CapitalEpsilon][wneg[L[wsqr[y]]]]];
wdiv[x_, y_] := wx[x, winv[y]];
whalf[x_] := wdiv[x, two];
wavg[x_, y_] := whalf[wplus[x, y]];

(* Step 3: Lemma B.4 *)
w\[Sigma][x_] := winv[wplus[1, \[CapitalEpsilon][wneg[x]]]];
wpow[x_, y_] := \[CapitalEpsilon][wx[y, L[x]]];
wlog[b_, y_] := wdiv[L[y], L[b]];
wsinh[x_] := whalf[wminus[\[CapitalEpsilon][x], \[CapitalEpsilon][wneg[x]]]];
wcosh[x_] := whalf[wplus[\[CapitalEpsilon][x], \[CapitalEpsilon][wneg[x]]]];
wtanh[x_] := wdiv[wsinh[x], wcosh[x]];
Q[t_] := \[CapitalEpsilon][whalf[L[t]]];
warsinh[x_] := L[wplus[x, Q[wplus[1, wsqr[x]]]]];
wartanh[x_] := whalf[wminus[L[wplus[1, x]], L[wminus[1, x]]]];

(* Table B.2 and Lemma B.7: witnesses on restricted domains *)
wsqrtPos[x_] := Q[x];
whypotOff[x_, y_] := Q[wplus[wsqr[x], wsqr[y]]];
warcoshGt1[x_] := L[wplus[x, Q[wminus[wsqr[x], 1]]]];

(* Appendix C: Lemma C.1 (the two infinities), Lemma C.2, Corollary C.3, absolute value *)
pinf := eml[1, zero];
minf := eml[1, pinf];
wsqrtBar[x_] := \[CapitalEpsilon][F[wminus[x, L[two]], \[CapitalEpsilon][eml[L[L[eml[x, x]]], two]]]];
whypotBar[x_, y_] := wsqrtBar[wplus[wsqr[x], wsqr[y]]];
warcoshBar[x_] := L[wplus[x, wsqrtBar[wminus[wsqr[x], 1]]]];
wabs[x_] := wsqrtBar[wsqr[x]];
wrelu[x_] := whalf[wplus[x, wabs[x]]];
wmax[x_, y_] := whalf[wplus[wplus[x, y], wabs[wminus[x, y]]]];

(* ------------------------------------------------------------------ *)
(* Exact checks.  verdict: "exact", "numeric", or {"FAIL", got, want}. *)
SetAttributes[verdict, HoldFirst];
verdict[got_, want_] := Module[{g = TimeConstrained[Catch[got, eml], 600, "TIMEOUT"], d},
   Which[
    g === "TIMEOUT", {"FAIL", "TIMEOUT", want},
    g === want, "exact",
    StringQ[g] || StringQ[want] || MemberQ[{Infinity, -Infinity}, g] || MemberQ[{Infinity, -Infinity}, want],
     {"FAIL", g, want},
    True,
    d = TimeConstrained[FullSimplify[g - want], 20, $Failed];
    If[d =!= 0, d = TimeConstrained[FullSimplify[TrigToExp[g - want]], 20, $Failed]];
    Which[d === 0, "exact",
     Abs[N[g - want, 60]] < 10^-50, "numeric",
     True, {"FAIL", g, want}]]];

allGood = True; nExact = 0; nNumeric = 0; nFail = 0;
report[name_, verdicts_List, pts_List : {}] := Module[{bad = Cases[verdicts, {"FAIL", __}], num = Count[verdicts, "numeric"]},
   nExact += Count[verdicts, "exact"]; nNumeric += num; nFail += Length[bad];
   Print[StringPadRight[name, 52], StringPadLeft[ToString[Length[verdicts]], 4], " points   ",
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
unary[name_, W_, P_, xs_] := report[name, Table[verdict[W[x], P[x]], {x, xs}], xs];
SetAttributes[binary, HoldAll];
binary[name_, W_, P_, ps_] := report[name, Table[verdict[W @@ p, P @@ p], {p, ps}], ps];
und = "undefined" &;

(* The 23 primitives of Table B.1 on their natural domains: the same checks in both arithmetics. *)
table23[] := Module[{xs = {-3, -1, -1/3, 0, 1/3, 1, 5/2}, pos = {1/3, 1, 2, E, 10}, ps = Tuples[{-2, 0, 1/3, 5/2}, 2]},
   report["1, e, -1, 2 and 0", MapThread[verdict, {{1, e, neg, two, zero}, {1, E, -1, 2, 0}}]];
   unary["exp", \[CapitalEpsilon], Exp, xs];
   unary["ln, x > 0", L, Log, pos];
   binary["+", wplus, Plus, ps];
   binary["-", wminus, Subtract, ps];
   unary["minus", wneg, Minus, xs];
   binary["x", wx, Times, ps];
   unary["sqr", wsqr, (#^2) &, xs];
   unary["inv, x != 0", winv, (1/#) &, DeleteCases[xs, 0]];
   binary["/, y != 0", wdiv, Divide, Select[ps, #[[2]] != 0 &]];
   unary["half", whalf, (#/2) &, xs];
   binary["avg", wavg, ((#1 + #2)/2) &, ps];
   unary["sigma", w\[Sigma], LogisticSigmoid, xs];
   binary["pow, x > 0", wpow, Power, Tuples[{{1/3, 1, 2, 5/2}, {-2, 0, 1/3, 5/2}}]];
   binary["log_b y, b > 0, b != 1, y > 0", wlog, Log, Tuples[{{1/3, 2, 5/2}, {1/3, 1, 2, 10}}]];
   unary["sinh", wsinh, Sinh, xs];
   unary["cosh", wcosh, Cosh, xs];
   unary["tanh", wtanh, Tanh, xs];
   unary["arsinh", warsinh, ArcSinh, xs];
   unary["artanh, -1 < x < 1", wartanh, ArcTanh, {-1/2, 0, 1/3, 9/10}]];

t0 = AbsoluteTime[];

Print["=== Theorem 1b: strict real arithmetic (Definition B.1) ==="];
mode = "strict";
Print["--- auxiliary terms (Lemmas B.1, B.2, B.4) ---"];
unary["B(t) = ln(e^t - t)", B, Log[Exp[#] - #] &, {-3, 0, 1/3, 1, 5/2}];
binary["F(t,v) = e^t - v", F, (Exp[#1] - #2) &, Tuples[{{-2, 0, 1/3}, {-2, 0, 5/2}}]];
binary["G(u,y) = y e^u", G, (#2 Exp[#1]) &, Tuples[{{-2, 0, 1/3}, {-2, 0, 5/2}}]];
unary["Q(t) = sqrt t, t > 0", Q, Sqrt, {1/4, 1, 2, 9}];
Print["--- the 23 primitives of Table B.1 ---"];
table23[];
Print["--- Table B.2: witnesses on the restricted domains ---"];
unary["sqrt, x > 0", wsqrtPos, Sqrt, {1/4, 1, 2, 9}];
binary["hypot, (x,y) != (0,0)", whypotOff, Sqrt[#1^2 + #2^2] &, DeleteCases[Tuples[{-2, 0, 1/3, 5/2}, 2], {0, 0}]];
unary["arcosh, x > 1", warcoshGt1, ArcCosh, {3/2, 2, 10}];
Print["--- the same three terms at the excluded points: undefined ---"];
report["Q(0), hypot(0,0), arcosh(1)", {verdict[wsqrtPos[0], "undefined"], verdict[whypotOff[0, 0], "undefined"], verdict[warcoshGt1[1], "undefined"]}];
Print["--- outside the natural domains: undefined ---"];
report["ln 0, ln(-1), inv 0, 1/0, (-2)^(1/3), 0^2, log_1 2",
  {verdict[L[0], "undefined"], verdict[L[-1], "undefined"], verdict[winv[0], "undefined"], verdict[wdiv[1, 0], "undefined"],
   verdict[wpow[-2, 1/3], "undefined"], verdict[wpow[0, 2], "undefined"], verdict[wlog[1, 2], "undefined"]}];

Print[];
Print["=== Theorem 1c: extended real arithmetic (Definition C.1) ==="];
mode = "extended";
Print["--- the operator (Table C.1) ---"];
report["15 cells of the table",
  {verdict[eml[-Infinity, -Infinity], "undefined"], verdict[eml[-Infinity, -2], "undefined"], verdict[eml[-Infinity, 0], Infinity],
   verdict[eml[-Infinity, 3], -Log[3]], verdict[eml[-Infinity, Infinity], -Infinity],
   verdict[eml[2, -Infinity], "undefined"], verdict[eml[2, -2], "undefined"], verdict[eml[2, 0], Infinity],
   verdict[eml[2, 3], E^2 - Log[3]], verdict[eml[2, Infinity], -Infinity],
   verdict[eml[Infinity, -Infinity], "undefined"], verdict[eml[Infinity, -2], "undefined"], verdict[eml[Infinity, 0], Infinity],
   verdict[eml[Infinity, 3], Infinity], verdict[eml[Infinity, Infinity], "undefined"]}];
Print["--- Lemma C.1 ---"];
report["+Infinity = eml(1,0), -Infinity = eml(1,+Infinity)", {verdict[pinf, Infinity], verdict[minf, -Infinity]}];
report["E(-Inf) = 0, E(+Inf) = +Inf", {verdict[\[CapitalEpsilon][-Infinity], 0], verdict[\[CapitalEpsilon][Infinity], Infinity]}];
report["L(0) = -Inf, L(+Inf) = +Inf", {verdict[L[0], -Infinity], verdict[L[Infinity], Infinity]}];
report["F(t,-Inf) = +Inf, F(t,+Inf) = -Inf", {verdict[F[1/3, -Infinity], Infinity], verdict[F[1/3, Infinity], -Infinity]}];
Print["--- Step 1: the 23 primitives of Table B.1 ---"];
table23[];
Print["--- Step 2: Lemma C.2, Corollary C.3 ---"];
unary["sqrt on [0, oo), incl. 0", wsqrtBar, Sqrt, {0, 1/4, 1, 2, 9}];
binary["hypot on R^2, incl. (0,0)", whypotBar, Sqrt[#1^2 + #2^2] &, Tuples[{-2, 0, 1/3, 5/2}, 2]];
unary["arcosh on [1, oo), incl. 1", warcoshBar, ArcCosh, {1, 3/2, 2, 10}];
Print["--- absolute value, ReLU, max ---"];
unary["|x|", wabs, Abs, {-3, -1/3, 0, 1/3, 5/2}];
unary["ReLU", wrelu, Max[#, 0] &, {-3, -1/3, 0, 1/3, 5/2}];
binary["max", wmax, Max, Tuples[{-2, 0, 1/3, 5/2}, 2]];
Print["--- outside the natural domains ---"];
report["ln(-1), sqrt(-1): undefined", {verdict[L[-1], "undefined"], verdict[wsqrtBar[-1], "undefined"]}];

Print[];
Print["eml nodes evaluated: ", emlCount, ";  time ", Round[AbsoluteTime[] - t0], " s"];
Print["checks: ", nExact + nNumeric + nFail, "  (exact ", nExact, ", numeric ", nNumeric, ", failed ", nFail, ")"];
Print[If[allGood, "RESULT: no failure.", "RESULT: THERE ARE FAILURES - see above."]];
