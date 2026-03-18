EML[x_, y_] := Exp[x] - Log[y];

Off[
 General::ovfl,
 General::munfl,
 General::unfl,
 Infinity::indet,
 Log::indet,
 N::meprec,
 Power::infy
 ];

safeApproxEML[x_, y_] := Check[
   Module[{value = Exp[x] - Log[y]},
    If[NumberQ[value], value, Indeterminate]
    ],
   Indeterminate
   ];

exactMatchQ[expr_, target_] := TrueQ @ Check[
    FullSimplify[expr == target],
    False
    ];

target = 2;
maxTokens = 27;
tolerance = 10.^-10;
targetApprox = N[target];

candidates[1] = {
   <|
    "Expr" -> 1,
    "Approx" -> 1.,
    "Code" -> {1}
    |>
   };

candidates[n_Integer?OddQ] /; n > 1 := candidates[n] = Flatten[
    Table[
     Module[{rightTokens = n - leftTokens - 1},
      Table[
       <|
        "Expr" -> EML[left["Expr"], right["Expr"]],
        "Approx" -> safeApproxEML[left["Approx"], right["Approx"]],
        "Code" -> Join[left["Code"], right["Code"], {EML}]
        |>,
       {left, candidates[leftTokens]},
       {right, candidates[rightTokens]}
       ]
      ],
     {leftTokens, 1, n - 2, 2}
     ],
    2
    ];

Print[Now];

elapsed = AbsoluteTiming[
    result = Catch[
      Do[
       Print["n=", n];
       Do[
        If[
         NumberQ[candidate["Approx"]] &&
          Abs[candidate["Approx"] - targetApprox] < tolerance &&
          exactMatchQ[candidate["Expr"], target],
         Print[candidate["Code"]];
         Print[ToString[candidate["Expr"], InputForm]];
         Throw[candidate]
         ],
        {candidate, candidates[n]}
        ],
       {n, 1, maxTokens, 2}
       ];
      $Failed
      ];
    ][[1]];

Print[Now];
Print[Quantity[elapsed, "Seconds"]];
Print["Finished!\n"];
