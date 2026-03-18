ValidateCodeGeneral[digits_List] := 
 Module[{counter, k, arity}, counter = 0;
  For[k = 1, k <= Length[digits], k++, arity = digits[[k]];
   If[arity == 0, counter++;  (*Operand*), 
    counter -= arity;(*Operator consumes `arity` operands*)
    If[counter < 0, Return[False]];
    counter++;  (*Operator produces one result*)];];
  Return[counter == 1];];


EML[x_, y_] := Exp[x] - Log[y];

functions = {};
binaryOperations = {EML};

(*RPN calculator*)
funs = If[functions == {}, Null, functions /. List -> Alternatives];
ops = binaryOperations /. List -> Alternatives;
language = Join[functions, binaryOperations] /. List -> Alternatives;
rpnRule[{a : Except[language] ..., b : Except[language], 
    c : Except[language], op : ops, d___}] := rpnRule[{a, op[b, c], d}];
rpnRule[{a : Except[language] ..., b : Except[ops | funs], f : funs, 
    c___}] := rpnRule[{a, f[b], c}];
rpnRule[{rest : Except[language]}] := rest;

target = -2;

start = Now;
Print[start];
For[n = 1, n <= 27, n = n + 2,
 Print["n=", n];
 For[k = 0, k < 2^n, k++,
  code = IntegerDigits[k, 2, n] /. 1 -> 2;
  If[ValidateCodeGeneral[code],
   (* Print["n=",n,"\tk=", k,"\t",code]; *)
   rpnCode = code /. {0 -> 1, 2 -> EML};
   formula = rpnRule[rpnCode] // Simplify;
   If[formula == target, Print[rpnCode]; Abort[]]
   ];
  ]
 ];
end = Now;
Print[end];
Print[UnitConvert[DateDifference[start, end], "Seconds"]];
Print["Finished!\n"];