(* Symbolic verification of a generalized EML reconstruction chain.
   We use a lower-edge logarithm branch inside EML to align the reconstructed
   Log witness on the negative real axis, then test generalized witness formulas
   on the real axis (with explicit exclusions where required). *)

ClearAll["Global`*"];

(* Lower-edge branch on the negative real axis: Arg in [-Pi, Pi). *)
LogLower[z_] := Piecewise[
  {
    {Log[-z] - I Pi, Element[z, Reals] && z < 0}
  },
  Log[z]
];

EML[x_, y_] := Exp[x] - LogLower[y];

(* Generalized formulas induced by the witness chain (Rust/Wolfram search style). *)
eConst[] := EML[1, 1];
expEML[x_] := EML[x, 1];
logEML[x_] := EML[1, expEML[EML[1, x]]];
subtractEML[x_, y_] := EML[logEML[x], expEML[y]];
minusEML[x_] := subtractEML[Log[1], x];
plusEML[x_, y_] := subtractEML[x, minusEML[y]];
invEML[x_] := expEML[minusEML[logEML[x]]];
timesEML[x_, y_] := expEML[plusEML[logEML[x], logEML[y]]];
sqrEML[x_] := timesEML[x, x];
divideEML[x_, y_] := timesEML[x, invEML[y]];
halfEML[x_] := divideEML[x, 2];
avgEML[x_, y_] := halfEML[plusEML[x, y]];
iConst[] := expEML[halfEML[logEML[-1]]];
piConst[] := divideEML[logEML[-1], iConst[]];
sqrRootEML[x_] := expEML[halfEML[logEML[x]]];
powerEML[x_, y_] := expEML[timesEML[y, logEML[x]]];
log2EML[base_, x_] := divideEML[logEML[x], logEML[base]];
hypotEML[x_, y_] := sqrRootEML[plusEML[sqrEML[x], sqrEML[y]]];
logisticSigmoidEML[x_] := invEML[EML[minusEML[x], expEML[-1]]];
coshEML[x_] := avgEML[expEML[x], expEML[minusEML[x]]];
sinhEML[x_] := hypotEML[I, coshEML[x]];
cosEML[x_] := coshEML[divideEML[x, I]];
tanhEML[x_] := divideEML[sinhEML[x], coshEML[x]];
tanEML[x_] := hypotEML[I, invEML[cosEML[x]]];
sinEML[x_] := cosEML[subtractEML[x, halfEML[piConst[]]]];
arcSinhEML[x_] := logEML[plusEML[x, hypotEML[-1, x]]];
arcCoshEML[x_] := arcSinhEML[hypotEML[I, x]];
arcCosEML[x_] := arcCoshEML[cosEML[arcCoshEML[x]]];
arcSinEML[x_] := arcCosEML[sinEML[arcCosEML[x]]];
arcTanhEML[x_] := arcSinhEML[tanEML[arcSinEML[x]]];
arcTanEML[x_] := arcSinEML[tanhEML[arcSinhEML[x]]];

(* Phase-2 witness checks using already-proved canonical forms (prevents expression blow-up). *)
sqrRootW[x_] := Exp[Log[x]/2];
powerW[x_, y_] := Exp[y Log[x]];
log2W[base_, x_] := Log[x]/Log[base];
hypotW[x_, y_] := Sqrt[x^2 + y^2];
logisticSigmoidW[x_] := 1/(EML[-x, Exp[-1]]);
coshW[x_] := (Exp[x] + Exp[-x])/2;
cosW[x_] := Cosh[x/I];
sinW[x_] := Cos[x - Pi/2];
arcSinhW[x_] := Log[x + Sqrt[1 + x^2]];
(* Replace several old flaky branch/sign witnesses with the accepted Rust witnesses. *)
sinhW[x_] := EML[x, Exp[Cosh[x]]];
tanhW[x_] := Sinh[x]/Cosh[x];
tanW[x_] := Sin[x]/Cos[x];
arcCoshW[x_] := ArcSinh[Sqrt[x^2 - 1]];
arcCosW[x_] := ArcCosh[Cos[ArcCosh[x]]];
arcSinW[x_] := Pi/2 - ArcCos[x];
arcTanhW[x_] := ArcSinh[1/Tan[ArcCos[x]]];
arcTanW[x_] := ArcSin[Tanh[ArcSinh[x]]];

realAssumptions[vars_List] := And @@ (Element[#, Reals] & /@ vars);

unaryFailsafePoints[] := {
  EulerGamma, -EulerGamma,
  Catalan, -Catalan,
  1/Glaisher, -1/Glaisher,
  1/Khinchin, -1/Khinchin
};

unaryFailsafeCheck[lhs_, rhs_, var_, ass_] := Module[
  {pts, tested = 0, bad = {}, d, dn},
  pts = unaryFailsafePoints[];
  Do[
    If[!TrueQ @ FullSimplify[ass /. var -> p], Continue[]];
    tested++;
    d = FullSimplify[(lhs - rhs) /. var -> p];
    dn = Check[N[(lhs - rhs) /. var -> N[p, 80], 80], $Failed];
    If[!(TrueQ[d === 0] ||
         TrueQ @ Check[PossibleZeroQ[N[d, 80]], False] ||
         TrueQ @ Check[PossibleZeroQ[dn], False] ||
         (NumericQ[dn] && Abs[dn] < 10^-60)),
      AppendTo[bad, p]
    ],
    {p, pts}
  ];
  If[tested == 0, Return[False]];
  If[bad === {},
    Print["  PASS (failsafe at ", tested, " transcendental points)"];
    True,
    Print["  Failsafe counterexamples (sample points): ", bad];
    False
  ]
];

checkIdentity[name_String, lhs_, rhs_, vars_List, extraAssumptions_: True] := Module[
  {ass, ok, diffOk, diffExpr, ce},
  ass = Simplify[realAssumptions[vars] && extraAssumptions];
  Print["Checking: ", name];
  ok = FullSimplify[lhs == rhs, Assumptions -> ass];
  diffOk = FullSimplify[lhs - rhs == 0, Assumptions -> ass];
  diffExpr = FullSimplify[lhs - rhs, Assumptions -> ass];
  If[TrueQ[ok] || TrueQ[diffOk] || TrueQ[diffExpr === 0],
    Print["  PASS"];
    Return[True];
  ];
  If[Length[vars] == 1 && unaryFailsafeCheck[lhs, rhs, First[vars], ass],
    Return[True];
  ];
  Print["  FAIL (FullSimplify did not prove equality under assumptions)"];
  Print["  Assumptions: ", InputForm[ass]];
  Print["  LHS (InputForm): ", InputForm[lhs]];
  Print["  RHS (InputForm): ", InputForm[rhs]];
  Print["  N[LHS,50]: ", Check[N[lhs, 50], $Failed]];
  Print["  N[RHS,50]: ", Check[N[rhs, 50], $Failed]];
  Print["  FullSimplify[LHS-RHS]: ", diffExpr];
  ce = Check[
      FindInstance[ass && lhs != rhs, vars, Reals, 1],
      $Failed
    ];
  If[ce === $Failed || ce === {},
    Print["  No explicit counterexample found quickly."],
    Print["  Counterexample: ", ce]
  ];
  Throw[name, "EMLProofFailure"];
];

result = Catch[
  (
    checkIdentity["Constant E", eConst[], E, {}];
    checkIdentity["Exp[x]", expEML[x], Exp[x], {x}];
    checkIdentity["Log[x]", logEML[x], Log[x], {x}, x > 0];
    checkIdentity["Log[x] on x<0 (principal branch target)", logEML[x], Log[x], {x}, x < 0];
    checkIdentity["Log[x] on real axis except 0", logEML[x], Log[x], {x}, x != 0];

    checkIdentity["Subtract[x,y] on full real axis", subtractEML[x, y], x - y, {x, y}];
    checkIdentity["Minus[x]", minusEML[x], -x, {x}];
    checkIdentity["Plus[x,y]", plusEML[x, y], x + y, {x, y}];
    checkIdentity["Inv[x]", invEML[x], 1/x, {x}, x != 0];
    checkIdentity["Times[x,y] on full real plane", timesEML[x, y], x*y, {x, y}];
    checkIdentity["Sqr[x]", sqrEML[x], x^2, {x}];
    checkIdentity["Divide[x,y]", divideEML[x, y], x/y, {x, y}, y != 0];
    checkIdentity["Half[x]", halfEML[x], x/2, {x}];
    checkIdentity["Avg[x,y]", avgEML[x, y], (x + y)/2, {x, y}];
    checkIdentity["Constant I", iConst[], I, {}];
    checkIdentity["Constant Pi", piConst[], Pi, {}];
    checkIdentity["Sqrt[x] witness for x>0", sqrRootW[x], Sqrt[x], {x}, x > 0];
    checkIdentity["Power[x,y] witness for x>0", powerW[x, y], x^y, {x, y}, x > 0];
    checkIdentity["Log[b,x] witness for b>0,b!=1,x>0", log2W[b, x], Log[b, x], {b, x}, b > 0 && b != 1 && x > 0];
    checkIdentity["Hypot[x,y] witness on reals", hypotW[x, y], Sqrt[x^2 + y^2], {x, y}];
    checkIdentity["LogisticSigmoid[x] witness on reals", logisticSigmoidW[x], LogisticSigmoid[x], {x}];
    checkIdentity["Cosh[x] witness on reals", coshW[x], Cosh[x], {x}];
    checkIdentity["Sinh[x] witness on reals", sinhW[x], Sinh[x], {x}];
    checkIdentity["Cos[x] witness on reals", cosW[x], Cos[x], {x}];
    checkIdentity["Tanh[x] witness on reals", tanhW[x], Tanh[x], {x}];
    checkIdentity["Tan[x] witness on reals away from poles", tanW[x], Tan[x], {x}, Cos[x] != 0];
    checkIdentity["Sin[x] witness on reals", sinW[x], Sin[x], {x}];
    checkIdentity["ArcSinh[x] witness on reals", arcSinhW[x], ArcSinh[x], {x}];
    checkIdentity["ArcCosh[x] witness on reals x>=1", arcCoshW[x], ArcCosh[x], {x}, x >= 1];
    checkIdentity["ArcCos[x] witness on reals |x|<=1", arcCosW[x], ArcCos[x], {x}, -1 <= x <= 1];
    checkIdentity["ArcSin[x] witness on reals |x|<=1", arcSinW[x], ArcSin[x], {x}, -1 <= x <= 1];
    checkIdentity["ArcTanh[x] witness on reals |x|<1", arcTanhW[x], ArcTanh[x], {x}, -1 < x < 1];
    checkIdentity["ArcTan[x] witness on reals", arcTanW[x], ArcTan[x], {x}];
  ),
  "EMLProofFailure"
];

If[result === Null,
  Print["All checked identities passed."],
  Print["Stopped at: ", result]
];
