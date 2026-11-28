(* ::Package:: *)

(* verify_theorem1.wl

   The witnesses of Theorem 1 (Lemmas 1-24) and the terms of Corollary 25 as
   Wolfram Language code, with exact checks.  Everything is built from eml and
   the constant 1.

   Run:   wolframscript -file verify_theorem1.wl        (about a minute)
   or, in a notebook:   Get["verify_theorem1.wl"]   and then e.g.  Warctan[1]

   The symbols E, N, D are protected in the Wolfram Language, so the terms
   E, N, D of the paper are the Greek capitals \[CapitalEpsilon], \[CapitalNu],
   \[CapitalDelta] (written here as escapes; a notebook shows them as Greek
   letters).

   Method.

   1. Log is not redefined.  The built-in Log is the principal branch with
      Arg in (-Pi, Pi], which is the Log of the paper.  The only difference is
      Log[0]: undefined in the paper, -Infinity in Mathematica.  So eml tests
      its second argument and stops if it is zero.

   2. eml passes its value through Simplify before returning it (2 s limit;
      after that the value is returned as it is, still exact).  Arguments are
      evaluated first, so a witness is computed from the leaves upwards and
      each intermediate value is short before it is used again.  For an exact
      number q the kernel itself evaluates Log[E^q] with the correct multiple
      of 2 Pi I, also when E^q lies on the branch cut.
      FullSimplify in place of Simplify is not needed and is much slower: it
      spends about a minute on a single exponential such as
      E^(-1 - (1 + E - (1 + E)^(-2))^2) and returns it unchanged.

   3. The checks are at exact points (rationals, Sqrt[2], Pi, ...), among them
      every point that lies on the branch cut: L(-1), H(0), Sqrt 0,
      hypot(0,0), arcosh 1, arcsin(+-1), arccos(+-1).  A check counts as
      "exact" only if the difference to the built-in function simplifies to
      the integer 0.

   4. No message of the kernel is suppressed.  A run prints two kinds.
      eml::log0 appears at the three undefined points near the end; to report
      them is the purpose of that check.  N::meprec appears in the last check
      of Corollary 25, at arguments with the denominator 5: the kernel then
      meets numbers such as Log[1 - (I/5) Pi - Log[-(-1)^(4/5)]], which is
      exactly 0, and cannot find an imaginary part numerically.  The values
      are then left in a longer form, and the comparison at the end still
      reduces the difference to the integer 0.  The kernel prints a message
      three times and then General::stop.

   What this file does not do: it proves nothing for all x.  With a symbolic x,
   Simplify cannot decide which branch of Log[E^q] applies from Lemma 18 on.
   The statements for all x are in the paper and in the Lean proof. *)

ClearAll[eml, e, L, S, zero, neg, two, three, half, M, mipi, img, pi, pi2, twoi, ipi,
  twoipi, U, Wminus, Wplus, Wsqr, Whalf, Wx, Winv, Wdiv, Wavg, X, T, w, H, Wsqrt,
  Whypot, Wpow, Wlog, Wsinh, Wcosh, Wtanh, Warsinh, Warcosh, Wartanh, A, Wcos, Wsin,
  Wtan, Warcsin, Warccos, Warctan, wfl, Wfloor, R];
ClearAll["\[CapitalEpsilon]", "\[CapitalNu]", "\[CapitalDelta]", "\[Tau]", "W\[Sigma]"];
eml::log0 = "Log applied to 0.";
emlCount = 0;
watchReal = False; realNodes = True;   (* used for Corollary 25 (a) only *)
eml[z_, w_] := (emlCount++;
   If[PossibleZeroQ[w], Message[eml::log0]; Throw[$Failed, eml]];
   If[watchReal && ! (Element[z, Reals] === True && Positive[w] === True), realNodes = False];
   With[{v = Exp[z] - Log[w]}, TimeConstrained[Simplify[v], 2, v]]);

(* Step 1: Lemma 1, Corollary 2, Lemmas 3-4 *)
\[CapitalEpsilon][z_] := eml[z, 1];
e = eml[1, 1];
L[z_] := eml[1, \[CapitalEpsilon][eml[1, z]]];
S[u_, v_] := eml[L[u], \[CapitalEpsilon][v]];

(* Step 2: Lemma 5 *)
\[CapitalNu][z_] := S[1, S[e, S[S[e, 1], z]]];

(* Step 3: Lemmas 6-8 *)
zero = L[1];
neg = \[CapitalNu][1];
two = S[1, neg];
three = S[two, neg];
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

(* Step 4: Lemmas 9-16 *)
U[x_] := eml[L[img], \[CapitalEpsilon][\[CapitalNu][x]]];
Wminus[a_, b_] := S[S[U[a], b], img];
Wplus[a_, b_] := S[S[U[a], \[CapitalNu][b]], img];
Wsqr[x_] := Wminus[M[U[x], S[U[x], twoi]], 1];
Whalf[x_] := S[\[CapitalDelta][U[x], two], \[CapitalDelta][img, two]];
Wx[x_, y_] := Whalf[Whalf[Wminus[Wsqr[Wplus[x, y]], Wsqr[Wminus[x, y]]]]];
Winv[x_] := Wx[x, \[CapitalEpsilon][\[CapitalNu][L[Wsqr[x]]]]];
Wdiv[x_, y_] := Wx[x, Winv[y]];
Wavg[x_, y_] := Whalf[Wplus[x, y]];

(* Step 5: Lemmas 17-20 *)
X[x_] := S[M[U[x], img], neg];
T[z_] := eml[L[ipi], \[CapitalEpsilon][eml[L[ipi], z]]];
\[Tau][x_] := Wdiv[\[CapitalNu][x], Wplus[1, Wsqr[x]]];
w[x_] := M[U[\[Tau][x]], img];
H[x_] := S[\[CapitalDelta][T[w[x]], twoipi], \[CapitalDelta][L[w[x]], twoipi]];
Wsqrt[x_] := Wminus[\[CapitalEpsilon][Whalf[L[Wplus[x, H[\[CapitalNu][x]]]]]], H[\[CapitalNu][x]]];
Whypot[x_, y_] := Wsqrt[Wplus[Wsqr[x], Wsqr[y]]];

(* Step 6: Lemma 21 *)
W\[Sigma][x_] := Winv[Wplus[1, \[CapitalEpsilon][\[CapitalNu][x]]]];
Wpow[x_, y_] := \[CapitalEpsilon][Wx[y, L[x]]];
Wlog[x_, y_] := Wdiv[L[y], L[x]];
Wsinh[x_] := Whalf[Wminus[\[CapitalEpsilon][x], \[CapitalEpsilon][\[CapitalNu][x]]]];
Wcosh[x_] := Wavg[\[CapitalEpsilon][x], \[CapitalEpsilon][\[CapitalNu][x]]];
Wtanh[x_] := Wdiv[Wsinh[x], Wcosh[x]];
Warsinh[x_] := L[Wplus[x, Wsqrt[Wplus[Wsqr[x], 1]]]];
Warcosh[x_] := L[Wplus[x, Wsqrt[Wminus[Wsqr[x], 1]]]];
Wartanh[x_] := Whalf[Wminus[L[Wplus[1, x]], L[Wminus[1, x]]]];

(* Corollary 25 (All rationals), parts (b) and (c) *)
wfl[x_] := M[U[Wx[pi, Wminus[1, Wx[two, x]]]], img];
Wfloor[x_] := Whalf[Wdiv[S[M[img, S[wfl[x], S[wfl[x], 1]]], img], pi]];
R[x_] := Winv[Wplus[Wminus[Wx[two, Wfloor[x]], x], 1]];

(* Step 7: Lemmas 22-24 *)
A[u_, v_] := S[S[S[three, \[CapitalNu][u]], \[CapitalNu][v]], three];
Wcos[x_] := Whalf[A[\[CapitalEpsilon][X[x]], \[CapitalEpsilon][X[\[CapitalNu][x]]]]];
Wsin[x_] := Wcos[Wminus[x, pi2]];
Wtan[x_] := Wdiv[Wsin[x], Wcos[x]];
Warcsin[x_] := Wminus[\[CapitalDelta][A[L[A[Wsqrt[Wminus[1, Wsqr[x]]], X[x]]], twoi], img], two];
Warccos[x_] := Wminus[pi2, Warcsin[x]];
Warctan[x_] := Warcsin[Wdiv[x, Wsqrt[Wplus[Wsqr[x], 1]]]];

(* ------------------------------------------------------------------ *)
(* Exact checks.  verdict: "exact" if the difference simplifies to 0,
   "numeric" if it is only numerically 0 (60 digits), "LOG0" if a logarithm
   of 0 was met, otherwise the offending value. *)

SetAttributes[verdict, HoldFirst];
verdict[got_, want_] := Module[{g = Catch[got, eml], d},
   If[g === $Failed, Return["LOG0"]];
   d = TimeConstrained[FullSimplify[g - want], 20, g - want];
   Which[d === 0, "exact",
    Abs[N[d, 60]] < 10^-50, "numeric",
    True, {"FAIL", g, want}]];

report[name_, verdicts_List] := Module[{bad = Cases[verdicts, Except["exact"]]},
   Print[StringPadRight[name, 50], StringPadLeft[ToString[Length[verdicts]], 4], " points   ",
    If[bad === {}, "all exact", {"NOT ALL EXACT", Tally[verdicts]}]];
   If[bad =!= {}, allExact = False]];

allExact = True;
SetAttributes[unary, HoldAll];
unary[name_, W_, P_, xs_] := report[name, Table[verdict[W[x], P[x]], {x, xs}]];
SetAttributes[binary, HoldAll];
binary[name_, W_, P_, ps_] := report[name, Table[verdict[W @@ p, P @@ p], {p, ps}]];

t0 = AbsoluteTime[];

Print["--- constants (Corollary 2, Lemmas 6, 8) ---"];
report["e, 0, -1, 2, 3, 1/2",
  MapThread[verdict, {{e, zero, neg, two, three, half}, {E, 0, -1, 2, 3, 1/2}}]];
report["remark to Lemma 8: L(-1) = -i Pi", {verdict[L[neg], -I Pi]}];
report["-i Pi, i, Pi, Pi/2, 2i, i Pi, 2 Pi i",
  MapThread[verdict, {{mipi, img, pi, pi2, twoi, ipi, twoipi},
    {-I Pi, I, Pi, Pi/2, 2 I, I Pi, 2 Pi I}}]];

Print["--- auxiliary terms ---"];
zs = {2, 1/3, -1, -5/2, I, -I, 3 + 4 I, -3 + 4 I, -3 - 4 I, 3 - 4 I};
Lval[z_] := If[Re[z] < 0 && Im[z] == 0, Log[z] - 2 Pi I, Log[z]];
Tval[z_] := If[Im[z] < 0, Log[z] + 2 Pi I, Log[z]];
unary["Lemma 3: L(z)", L, Lval, zs];
binary["Lemma 4: S(u,v), -Pi < Im v <= Pi", S, Subtract,
  Tuples[{{1, -2, I, 3 + 4 I}, {0, -5/2, I, 1 + I Pi, -1 - 3 I}}]];
binary["Lemma 7: M(u,v) = u v", M, Times, Tuples[{{2, -3, I, 3 + 4 I}, {2, I, -I, 3 - 4 I}}]];
binary["Lemma 7: D(u,v) = u/v", \[CapitalDelta], Divide, Tuples[{{2, -3, I, 3 + 4 I}, {2, I, -I, 3 - 4 I}}]];
binary["Lemma 22: A(u,v) = u + v", A, Plus,
  {{1, I}, {Exp[I/3], Exp[-I/3]}, {I/2, 2 I}, {-I Pi/2, 2 I}, {0, 0}, {-1, -1}}];
unary["Lemma 18: T(z)", T, Tval, zs];
unary["Lemma 5: N(z), |Im z| < Pi", \[CapitalNu], Minus, {0, 1, -7/3, 2 + 3 I, -4 - 31 I/10, 5 I/2}];
unary["Lemma 9: U(x) = x + i", U, (# + I) &, {-3, 0, 1/3, 5/2}];
unary["Lemma 17: X(x) = i x", X, (I #) &, {-3, 0, 1/3, 5/2}];
unary["Lemma 19: H(x), incl. H(0) = 1", H, UnitStep, {-2, -1/10, 0, 1/10, 2}];

Print["--- unary primitives ---"];
xs = {-3, -1, -1/3, 0, 1/3, 1, 5/2};
unary["exp", \[CapitalEpsilon], Exp, xs];
unary["ln", L, Log, {1/3, 1, 2, E, 10}];
unary["minus", \[CapitalNu], Minus, xs];
unary["sqr", Wsqr, (#^2) &, xs];
unary["half", Whalf, (#/2) &, xs];
unary["inv", Winv, (1/#) &, {-3, -1/3, 1/3, 1, 5/2}];
unary["sqrt, incl. 0", Wsqrt, Sqrt, {0, 1/4, 1, 2, 9}];
unary["sigma", W\[Sigma], LogisticSigmoid, xs];
unary["sinh", Wsinh, Sinh, xs];
unary["cosh", Wcosh, Cosh, xs];
unary["tanh", Wtanh, Tanh, xs];
unary["arsinh", Warsinh, ArcSinh, xs];
unary["arcosh, incl. 1", Warcosh, ArcCosh, {1, 3/2, 2, 10}];
unary["artanh", Wartanh, ArcTanh, {-1/2, 0, 1/3, 9/10}];
unary["cos", Wcos, Cos, {-3, -1, 0, 1/3, 1, 5/2, 10, Pi}];
unary["sin", Wsin, Sin, {-3, -1, 0, 1/3, 1, 5/2, 10, Pi/2}];
unary["tan", Wtan, Tan, {-3, -1, 0, 1/3, 1, 5/2, 10}];
unary["arcsin, incl. -1, 1", Warcsin, ArcSin, {-1, -1/2, 0, 1/3, 1/Sqrt[2], 1}];
unary["arccos, incl. -1, 1", Warccos, ArcCos, {-1, -1/2, 0, 1/3, 1/Sqrt[2], 1}];
unary["arctan", Warctan, ArcTan, {-Sqrt[3], -1, 0, 1/3, 1, 10}];

Print["--- binary primitives ---"];
ps = Tuples[{-2, 0, 1/3, 5/2}, 2];
binary["+", Wplus, Plus, ps];
binary["-", Wminus, Subtract, ps];
binary["x", Wx, Times, ps];
binary["avg", Wavg, ((#1 + #2)/2) &, ps];
binary["hypot, incl. (0,0)", Whypot, Sqrt[#1^2 + #2^2] &, ps];
binary["/", Wdiv, Divide, Select[ps, #[[2]] != 0 &]];
binary["pow", Wpow, Power, Tuples[{{1/3, 1, 2, 5/2}, {-2, 0, 1/3, 5/2}}]];
binary["log", Wlog, Log, Tuples[{{1/3, 2, 5/2}, {1/3, 1, 2, 10}}]];

Print["--- undefined points (a logarithm of 0 must be reported) ---"];
Print["S(0,1), ln 0, inv 0:  ", {verdict[S[0, 1], 0], verdict[L[0], 0], verdict[Winv[0], 0]}];

Print["--- Corollary 25 (All rationals) ---"];
(* (a) rationals from terms whose nodes all have real values, every logarithm applied to a positive
   number: n+1 := S(n,-1), 1/q := E(N(L(q))), p/q := D(p,q), -r := N(r).  While watchReal is True,
   realNodes becomes False if a node gets a non-real first argument or a second argument that is not a
   positive real number. *)
watchReal = True; realNodes = True;
int[1] = 1; int[n_Integer /; n > 1] := S[int[n - 1], neg];
rat[1, q_] := \[CapitalEpsilon][\[CapitalNu][L[int[q]]]];
rat[p_ /; p > 1, q_] := \[CapitalDelta][int[p], int[q]];
report["(a) e, -1, 5, 1/3, 7/4, 2/7, -7/4, 0",
  MapThread[verdict, {{eml[1, 1], \[CapitalNu][1], int[5], rat[1, 3], rat[7, 4], rat[2, 7], \[CapitalNu][rat[7, 4]], L[1]},
    {E, -1, 5, 1/3, 7/4, 2/7, -7/4, 0}}]];
Print["    every node real, every logarithm of a positive number: ", realNodes];
If[! realNodes, allExact = False];
realNodes = True; L[neg];
Print["    the same test on L(-1), which takes the logarithm of -1: ", realNodes];
If[realNodes, allExact = False];
watchReal = False;
fl = {-3, -5/2, -1, -1/3, 0, 1/3, 1, 5/2, 7, Sqrt[2], -Pi, E};
unary["(b) floor", Wfloor, Floor, fl];
unary["(b) ceiling = N(floor(N(x)))", \[CapitalNu][Wfloor[\[CapitalNu][#]]] &, Ceiling, fl];
unary["(c) R(x) = 1/(2 floor(x) - x + 1), x != -1", R, (1/(2 Floor[#] - # + 1)) &, {-3, -5/2, -1/3, 0, 1/3, 1, 5/2, 7}];
report["(c) 1, R(1), R(R(1)), ...: Calkin-Wilf, 13 terms",
  MapThread[verdict, {NestList[R, 1, 12], {1, 1/2, 2, 1/3, 3/2, 2/3, 3, 1/4, 4/3, 3/5, 5/2, 2/5, 5/3}}]];

Print[];
Print["eml nodes evaluated: ", emlCount, ";  time ", Round[AbsoluteTime[] - t0], " s"];
Print[If[allExact, "RESULT: all checks exact.", "RESULT: SOME CHECKS ARE NOT EXACT - see above."]];
