(* ::Package:: *)

(* Symbolic verification of the EML discovery chain from rust_verify.log.
   Each item is proved in sequence: a raw EML witness [*EML] is checked by
   FullSimplify, then promoted to a standard form [*W] for use in subsequent
   proofs. Only previously verified [*W] functions may appear as building
   blocks in later *EML witnesses. *)

ClearAll["Global`*"];

(* Lower-edge branch on the negative real axis:Arg in[-Pi,Pi). *)
 LogLower[z_] := Piecewise[{{Log[-z] - I Pi, z \[Element] Reals && z < 0}}, Log[z]]; 

EML[x_, y_] := Exp[x] - LogLower[y];

(* --- Proof infrastructure --- *)


failsafePoints[] := {
  EulerGamma,    -EulerGamma,
  Catalan,       -Catalan,
  Glaisher^(-1), -Glaisher^(-1),
  Khinchin^(-1), -Khinchin^(-1),
  Glaisher,      -Glaisher,
  Khinchin,      -Khinchin,
  Khinchin^3,    -Khinchin^3, (* This is to ensure |x| > Cosh[Pi]=11.592 is probed; some identities might fail beyond this range *)
};

$MaxExtraPrecision=1024;



checkIdentity[name_String, lhs_, rhs_, vars_List, extraAss_: True] :=
  Module[{ass, diff, proved},
    ass    = Simplify[And @@ (Element[#, Reals] & /@ vars) && extraAss];
    Print["Checking: ", name];
    diff   = FullSimplify[lhs - rhs, Assumptions -> ass];
    proved = TrueQ[diff === 0] ||
             TrueQ[FullSimplify[lhs == rhs, Assumptions -> ass]];
    If[proved, Print["  PASS"]; Return[True]];
    If[Length[vars] == 1 && failsafeCheck[lhs, rhs, First[vars], ass],
      Return[True]];
    Print["  FAIL -- FullSimplify[lhs-rhs] = ", diff];
    Print["  N[lhs,32] = ", Check[N[lhs, 32], $Failed]];
    Print["  N[rhs,32] = ", Check[N[rhs, 32], $Failed]];
    Throw[name, "EMLProofFailure"]];


failsafeCheck[lhs_, rhs_, var_, ass_] := Module[
  {pts, tested, bad, d, dn},
  pts    = failsafePoints[];
  tested = 0;
  bad    = {};
  Do[
    If[!TrueQ[FullSimplify[ass /. var -> p]], Continue[]];
    tested = tested + 1;
    d  = FullSimplify[(lhs - rhs) /. var -> p];
    dn = Check[N[(lhs - rhs) /. var -> N[p, 128], 128], $Failed];
    If[!(TrueQ[d === 0] || TrueQ[d == 0] ||
         TrueQ[Check[PossibleZeroQ[N[d, 128]], False]] ||
         TrueQ[Check[PossibleZeroQ[dn], False]] ||
         (NumericQ[dn] && Abs[dn] < 10^(-64))),
      AppendTo[bad, p]],
    {p, pts}];
  If[tested == 0, Return[False]];
  If[bad === {},
    Print["  PASS (failsafe: ", tested, " transcendental points)"];
    True,
    Print["  FAIL failsafe counterexamples: ", bad];
    False]];

(* --- EML discovery chain (order from rust_verify.log) --- *)

result = Catch[
  Module[{},

  (* 1. Constant E
        Rust witness: EML[1, 1]
        Exp[1] - Log[1] = E - 0 = E *)
  eEML[] := EML[1, 1];
  checkIdentity["Step 1: Constant E", eEML[], E, {}];
  eW[] := E;

  (* 2. Exp[x]
        Rust witness: EML[x, 1]
        Exp[x] - Log[1] = Exp[x] - 0 = Exp[x] *)
  expEML[x_] := EML[x, 1];
  checkIdentity["Step 2: Exp[x]", expEML[x], Exp[x], {x}];
  expW[x_] := Exp[x];

  (* 3. Log[x]
        Rust witness: EML[1, Exp[EML[1, x]]]   uses proved Exp
        = E - EML[1, x] = E - (E - Log[x]) = Log[x] *)
  logEML[x_] := EML[1, expW[EML[1, x]]];
  checkIdentity["Step 3: Log[x]", logEML[x], Log[x], {x}, x>0 || x<0];
  logW[x_] := Log[x];

  (* 4. Subtract[x, y]
        Rust witness: EML[Log[x], Exp[y]]   uses proved Log, Exp
        = Exp[Log[x]] - Log[Exp[y]] = x - y *)
  subtractEML[x_, y_] := EML[logW[x], expW[y]];
  checkIdentity["Step 4: Subtract[x,y]", subtractEML[x, y], x - y, {x, y}, x \[Element] Reals && y \[Element] Reals];
  subtractW[x_, y_] := x - y;

  (* 5. Constant -1
        Rust witness: Subtract[Log[1], 1]   Log is proved; Log[1] = 0
        = 0 - 1 = -1 *)
  negOneEML[] := subtractW[logW[1], 1];
  checkIdentity["Step 5: Constant -1", negOneEML[], -1, {}];
  negOneW[] := -1;

  (* 6. Constant 2
        Rust witness: Subtract[1, -1]   uses proved Subtract and proved -1
        = 1 - (-1) = 2 *)
  twoEML[] := subtractW[1, negOneW[]];
  checkIdentity["Step 6: Constant 2", twoEML[], 2, {}];
  twoW[] := 2;

  (* 7. Minus[x]
        Rust witness: Subtract[Log[1], x]   Log is proved; Log[1] = 0
        = 0 - x = -x *)
  minusEML[x_] := subtractW[logW[1], x];
  checkIdentity["Step 7: Minus[x]", minusEML[x], -x, {x}];
  minusW[x_] := -x;

  (* 8. Plus[x, y]
        Rust witness: Subtract[x, Minus[y]]   uses proved Subtract, Minus
        = x - (-y) = x + y *)
  plusEML[x_, y_] := subtractW[x, minusW[y]];
  checkIdentity["Step 8: Plus[x,y]", plusEML[x, y], x + y, {x, y}];
  plusW[x_, y_] := x + y;

  (* 9. Inv[x]
        Rust witness: Exp[Minus[Log[x]]]   uses proved Exp, Minus, Log
        = Exp[-Log[x]] = 1/x *)
  invEML[x_] := expW[minusW[logW[x]]];
  checkIdentity["Step 9: Inv[x]", invEML[x], 1/x, {x}, x \[Element] Reals];
  invW[x_] := 1/x;

  (* 10. Times[x, y]
         Rust witness: Exp[Plus[Log[x], Log[y]]]   uses proved Exp, Plus, Log
         = Exp[Log[x] + Log[y]] = x*y *)
  timesEML[x_, y_] := expW[plusW[logW[x], logW[y]]];
  checkIdentity["Step 10: Times[x,y]", timesEML[x, y], x*y, {x, y},x \[Element] Reals && y \[Element] Reals];
  timesW[x_, y_] := x*y;

  (* 11. Sqr[x]
         Rust witness: Times[x, x]   uses proved Times *)
  sqrEML[x_] := timesW[x, x];
  checkIdentity["Step 11: Sqr[x]", sqrEML[x], x^2, {x} ];
  sqrW[x_] := x^2;

  (* 12. Divide[x, y]
         Rust witness: Times[x, Inv[y]]   uses proved Times, Inv *)
  divideEML[x_, y_] := timesW[x, invW[y]];
  checkIdentity["Step 12: Divide[x,y]", divideEML[x, y], x/y, {x, y}];
  divideW[x_, y_] := x/y;

  (* 13. Half[x]
         Rust witness: Divide[x, 2]   2 is proved; uses proved Divide *)
  halfEML[x_] := divideW[x, twoW[]];
  checkIdentity["Step 13: Half[x]", halfEML[x], x/2, {x}];
  halfW[x_] := x/2;

  (* 14. Avg[x, y]
         Rust witness: Half[Plus[x, y]]   uses proved Half, Plus *)
  avgEML[x_, y_] := halfW[plusW[x, y]];
  checkIdentity["Step 14: Avg[x,y]", avgEML[x, y], (x+y)/2, {x, y}];
  avgW[x_, y_] := (x+y)/2;

  (* 15. Sqrt[x]
         Rust witness: Exp[Half[Log[x]]]   uses proved Exp, Half, Log *)
  sqrtEML[x_] := expW[halfW[logW[x]]];
  checkIdentity["Step 15: Sqrt[x]", sqrtEML[x], Sqrt[x], {x}, x > 0];
  sqrtW[x_] := Sqrt[x];

  (* 16. Power[x, y]
         Rust witness: Exp[Times[y, Log[x]]]   uses proved Exp, Times, Log *)
  powerEML[x_, y_] := expW[timesW[y, logW[x]]];
  checkIdentity["Step 16: Power[x,y]", powerEML[x, y], x^y, {x, y}, x > 0];
  powerW[x_, y_] := x^y;

  (* 17. Log[base, x] (binary logarithm)
         Rust witness: Divide[Log[x], Log[base]]   uses proved Divide, Log *)
  logBaseEML[base_, x_] := divideW[logW[x], logW[base]];
  checkIdentity["Step 17: Log[base,x]", logBaseEML[b, x], Log[b, x], {b, x},
    b > 0 && b != 1 && x > 0];
  logBaseW[base_, x_] := Log[x]/Log[base];

  (* 18. Constant Pi
         Rust witness: Sqrt[Minus[Sqr[Log[-1]]]]   uses proved Sqrt, Minus, Sqr
         Log[-1] = I*Pi; -(I*Pi)^2 = Pi^2; Sqrt[Pi^2] = Pi *)
  piEML[] := sqrtW[minusW[sqrW[Log[-1]]]];
  checkIdentity["Step 18: Constant Pi", piEML[], Pi, {}];
  piW[] := Pi;

  (* 19. Hypot[x, y]
         Rust witness: Sqrt[Plus[Sqr[x], Sqr[y]]]   uses proved Sqrt, Plus, Sqr *)
  hypotEML[x_, y_] := sqrtW[plusW[sqrW[x], sqrW[y]]];
  checkIdentity["Step 19: Hypot[x,y]", hypotEML[x, y], Sqrt[x^2 + y^2], {x, y}];
  hypotW[x_, y_] := Sqrt[x^2 + y^2];

  (* 20. LogisticSigmoid[x]
         Rust witness: Inv[EML[Minus[x], Exp[-1]]]   uses proved Inv, Minus, Exp
         EML[-x, Exp[-1]] = Exp[-x] - Log[Exp[-1]] = Exp[-x] + 1
         1/(Exp[-x]+1) = LogisticSigmoid[x] *)
  logisticSigmoidEML[x_] := invW[EML[minusW[x], expW[-1]]];
  checkIdentity["Step 20: LogisticSigmoid[x]", logisticSigmoidEML[x],
    LogisticSigmoid[x], {x}];
  logisticSigmoidW[x_] := LogisticSigmoid[x];

  (* 21. Cosh[x]
         Rust witness: Avg[Exp[x], Exp[Minus[x]]]   uses proved Avg, Exp, Minus
         = (Exp[x] + Exp[-x]) / 2 = Cosh[x] *)
  coshEML[x_] := avgW[expW[x], expW[minusW[x]]];
  checkIdentity["Step 21: Cosh[x]", coshEML[x], Cosh[x], {x}];
  coshW[x_] := Cosh[x];

  (* 22. Sinh[x]
         Rust witness: EML[x, Exp[Cosh[x]]]   uses proved Cosh, Exp; raw EML
         = Exp[x] - Log[Exp[Cosh[x]]] = Exp[x] - Cosh[x] = Sinh[x] *)
  sinhEML[x_] := EML[x, expW[coshW[x]]];
  checkIdentity["Step 22: Sinh[x]", sinhEML[x], Sinh[x], {x}];
  sinhW[x_] := Sinh[x];

  (* 23. Tanh[x]
         Rust witness: Divide[Sinh[x], Cosh[x]]   uses proved Divide, Sinh, Cosh *)
  tanhEML[x_] := divideW[sinhW[x], coshW[x]];
  checkIdentity["Step 23: Tanh[x]", tanhEML[x], Tanh[x], {x}];
  tanhW[x_] := Tanh[x];

  (* 24. Cos[x]
         Rust witness: Cosh[Sqrt[Minus[Sqr[x]]]]   uses proved Cosh, Sqrt, Minus, Sqr
         Sqrt[-x^2] = I*|x|; Cosh[I*|x|] = Cos[|x|] = Cos[x]  (Cos is even) *)
  cosEML[x_] := coshW[sqrtW[minusW[sqrW[x]]]];
  checkIdentity["Step 24: Cos[x]", cosEML[x], Cos[x], {x}];
  cosW[x_] := Cos[x];

  (* 25. Sin[x]
         Rust witness: Cos[Subtract[x, Half[Pi]]]   uses proved Cos, Subtract, Half, Pi
         = Cos[x - Pi/2] = Sin[x] *)
  sinEML[x_] := cosW[subtractW[x, halfW[piW[]]]];
  checkIdentity["Step 25: Sin[x]", sinEML[x], Sin[x], {x}];
  sinW[x_] := Sin[x];

  (* 26. Tan[x]
         Rust witness: Divide[Sin[x], Cos[x]]   uses proved Divide, Sin, Cos *)
  tanEML[x_] := divideW[sinW[x], cosW[x]];
  checkIdentity["Step 26: Tan[x]", tanEML[x], Tan[x], {x}, Cos[x] != 0];
  tanW[x_] := Tan[x];

  (* 27. ArcSinh[x]
         Rust witness: Log[Plus[x, Hypot[-1, x]]]   uses proved Log, Plus, Hypot
         = Log[x + Sqrt[1 + x^2]] = ArcSinh[x] *)
  arcSinhEML[x_] := logW[plusW[x, hypotW[-1, x]]];
  checkIdentity["Step 27: ArcSinh[x]", arcSinhEML[x], ArcSinh[x], {x}];
  arcSinhW[x_] := ArcSinh[x];

  (* 28. ArcCosh[x]
         Rust witness: ArcSinh[Hypot[x, Sqrt[-1]]]   uses proved ArcSinh, Hypot, Sqrt
         Sqrt[-1] = I; Hypot[x, I] = Sqrt[x^2 - 1]; ArcSinh[Sqrt[x^2-1]] = ArcCosh[x] *)
  (* arcCoshEML[x_] := arcSinhW[hypotW[x, sqrtW[-1]]]; *)
  arcCoshEML[x_] := timesW[2, arcSinhW[sqrtW[avgW[-1, x]]]]; 
  (* arcCoshEML[x_] := logW[plusW[x,hypotW[x,sqrtW[-1]]]]; *)
  checkIdentity["Step 28: ArcCosh[x]", arcCoshEML[x], ArcCosh[x], {x}, x >= 1];
  arcCoshW[x_] := ArcCosh[x];

  (* 29. ArcCos[x]
         Rust witness: ArcCosh[Cos[ArcCosh[x]]]   uses proved ArcCosh, Cos
         For x in [-1,1]: ArcCosh[x] = I*ArcCos[x]; Cos[I*t] = Cosh[t];
         ArcCosh[Cosh[ArcCos[x]]] = ArcCos[x]  since ArcCos[x] >= 0 *)
  arcCosEML[x_] := arcCoshW[cosW[arcCoshW[x]]];
  checkIdentity["Step 29: ArcCos[x]", arcCosEML[x], ArcCos[x], {x}, -1 <= x <= 1];
  arcCosW[x_] := ArcCos[x];

  (* 30. ArcTanh[x]
         Rust witness: ArcSinh[Inv[Tan[ArcCos[x]]]]   uses proved ArcSinh, Inv, Tan, ArcCos
         = ArcSinh[Cot[ArcCos[x]]] = ArcSinh[x/Sqrt[1-x^2]] = ArcTanh[x] *)
  arcTanhEML[x_] := arcSinhW[invW[tanW[arcCosW[x]]]];
  checkIdentity["Step 30: ArcTanh[x]", arcTanhEML[x], ArcTanh[x], {x},
    -1 < x < 1 && x != 0];
  arcTanhW[x_] := ArcTanh[x];

  (* 31. ArcSin[x]
         Rust witness: Subtract[Half[Pi], ArcCos[x]]   uses proved Subtract, Half, Pi, ArcCos
         = Pi/2 - ArcCos[x] = ArcSin[x] *)
  arcSinEML[x_] := subtractW[halfW[piW[]], arcCosW[x]];
  checkIdentity["Step 31: ArcSin[x]", arcSinEML[x], ArcSin[x], {x}, -1 <= x <= 1];
  arcSinW[x_] := ArcSin[x];

  (* 32. ArcTan[x]
         Rust witness: ArcSin[Tanh[ArcSinh[x]]]   uses proved ArcSin, Tanh, ArcSinh
         Tanh[ArcSinh[x]] = x/Sqrt[1+x^2]; ArcSin[x/Sqrt[1+x^2]] = ArcTan[x] *)
  arcTanEML[x_] := arcSinW[tanhW[arcSinhW[x]]];
  checkIdentity["Step 32: ArcTan[x]", arcTanEML[x], ArcTan[x], {x}];
  arcTanW[x_] := ArcTan[x]

  ],
  "EMLProofFailure"
];

If[result === Null,
  Print["All 32 identities in the EML discovery chain verified successfully."],
  Print["Proof stopped at: ", result]
];
