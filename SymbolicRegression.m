BeginPackage["SymbolicRegression`"]

RandomExpression::usage =
        "
		RandomExpression[] - random function of single variable x
		
		RandomExpression[n,vars_consts,functions,operations] - random expression of tree depth 'n', composed of symbols 'vars_consts' (terminal symbols)
		using functions of single variable 'functions' and binary 'operations'
             		
		defaults:  n=7, vars_consts = (-1, x, E}, functions = {Log}, operations = {Plus, Times, Power} 
		"

		
EnumerateExpressions::usage =
        "
		EnumerateExpression[] - list of first functions of single variable x
        
		EnumerateExpression[n,vars_consts,functions,operations]- all expressions up to tree depth 'n', composed of symbols 'vars_consts' (terminal symbols)
		using functions of single variable 'functions' and binary 'operations'
             		
		defaults:  n=1, vars_consts = (-1, x, E}, functions = {Log}, operations = {Plus, Times, Power} 
		"

KolmogorovComplexity::usage = "Not yet implemented. Estimate Kolmogorov complexity of the expression in the language defined by base set, e.g. {-1,x, Log, Power}"

VerifyBaseSet::usage = "Not yet implemented. Check if base set (e.g. Log, Exp, Plus) generates all required types of expressions (Times,Power, Sinh, Sqrt, Pi, etc.) "

RecognizeConstant::usage = " RecognizeConstant[1.38629] - attempt to find best approximation using default setings"

RecognizeFunction::usage = "Not yet implemented"
 
		
Begin["`Private`"]

RandomExpression[depth_Integer:7, var_List:{-1,"Global`x","System`E"}, fun_List:{"System`Log"}, op_List:{"System`Plus","System`Times","System`Power"}] :=
   Module[{vars, funs, ops, lang, i, weights},
    vars = ToString /@ var;
    funs = Table[ToString[fun[[i]]] <> "[left]", {i, 1, Length[fun]}];
    ops = 
     Table[ToString[op[[i]]] <> "[left,right]", {i, 1, Length[op]}];
    lang = Join[vars, funs, ops];
	weights = Join[Table[1, {i, 1, Length[vars]}], Table[3, {i, 1, Length[funs]}], Table[2, {i, 1, Length[ops]}]];
    StringReplace[
      FixedPoint[
       StringReplace[#, 
	   {"left" -> RandomChoice[lang], 
        "right" -> RandomChoice[lang]}] &, 
		RandomChoice[Join[funs,ops]], 
       depth], {"left" -> RandomChoice[vars], 
       "right" -> RandomChoice[vars]}] // ToExpression
    
   ];

   
   
unorderedTuples[l_?ListQ] := 
 Flatten[Table[Table[{l[[i]], l[[j]]}, {i, 1, j}], {j, 1, Length[l]}],
   1];
   
orderedTuples = Tuples[#, {2}] &;

nextLevel[elem_List, var_List, fun_List, operatorsCommutative_List, operatorsNonCommutative_List] := 
(Join[
    elem,
    Flatten@Table[fun[[i]][elem], {i, 1, Length[fun]}], 
    Flatten@Table[
      operatorsCommutative[[i]] @@@ unorderedTuples[elem], {i, 1, 
       Length[operatorsCommutative]}],
    Flatten@
     Table[operatorsNonCommutative[[i]] @@@ orderedTuples[elem], {i, 
       1, Length[operatorsNonCommutative]}]
    ] // DeleteDuplicates);
	
	
EnumerateExpressions[depth_Integer:1,var_List:{-1,x,E}, fun_List:{Log}, op_List:{Plus,Times,Power}]:=
Module[{operatorsCommutative, operatorsNonCommutative},
{operatorsCommutative, operatorsNonCommutative} = {Select[op, MemberQ[Attributes[#], Orderless] &], 
 Select[op, FreeQ[Attributes[#], Orderless] &]};
Nest[nextLevel[#,var,fun,operatorsCommutative,operatorsNonCommutative]&, var, depth] /.x->Global`x
];

RecognizeConstant[target_?NumericQ, constants_List: {-1, I, E, Pi}, 
  functions_List: {Log}, binaryOperations_List: {Plus, Times, Power}, 
  OptionsPattern[]] := 
Module[{i, j, k, n, num, rule, rule2, funs, ops, language, symb, 
   bestError, digits, code, formula, error, rpnRule, 
   currentBestFormula, candidates},
  (* RPN calculator *)
  funs = If[functions == {}, Null, functions /. List -> Alternatives];
  ops = binaryOperations /. List -> Alternatives;
  language = Join[functions, binaryOperations] /. List -> Alternatives;
  rpnRule[{a : Except[language] ..., b : Except[language], 
     c : Except[language], op : ops, d___}] := 
   rpnRule[{a, op[b, c], d}];
  rpnRule[{a : Except[language] ..., b : Except[ops | funs], f : funs,
      c___}] := rpnRule[{a, f[b], c}];
  rpnRule[{rest : Except[language]}] := rest;
  symb = Join[constants, functions, binaryOperations];
  num = Length[symb];
  rule = (# /. List -> Rule &) /@ Transpose[{Range[0, num - 1], symb}];
  rule2 = (# /. List -> Rule &) /@ 
    Transpose[{Range[0, num - 1], 
      Join[Table[1, Length[constants]], Table[0, Length[functions]], 
       Table[-1, Length[binaryOperations]]]}];
  bestError = Infinity;
  i = 0; j = 0;
  candidates = {};
  For[n = 1, n <= OptionValue[MaxCodeLength], n++,
   For[k = 0, k < num^n, k++,
    i++;
    digits = IntegerDigits[k, num, n];
    If[Total[digits /. rule2] != 1, Continue[]];
    j++;
    code = digits /. rule;
    formula = rpnRule[code];
    error = Abs[target - formula];
    If[error < bestError, bestError = error; 
     currentBestFormula = formula;
     AppendTo[candidates, formula]; 
     If[OptionValue[WriteToDisk] == True, 
      Export["candidatesList.m", candidates]];];
    If[bestError <= OptionValue[PrecisionGoal], 
     Return[candidates[[-Min[Length[candidates], 
           OptionValue[Candidates]] ;; -1]]]]
    ]
   ];
  candidates[[-Min[Length[candidates], OptionValue[Candidates]] ;; -1]]
];
  
Options[RecognizeConstant] = {PrecisionGoal -> Sqrt@$MachineEpsilon, MaxCodeLength -> 6, Candidates -> 1, WriteToDisk -> False};
   
End[ ]

EndPackage[ ]