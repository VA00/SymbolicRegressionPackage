BeginPackage["SymbolicRegression`"]

RandomExpression::usage =
        "
		RandomExpression[] - random function of single variable x
		
		RandomExpression[n,vars_consts,functions,operations] - random expression of tree depth 'n', composed of symbols 'vars_consts' (terminal symbols)
		using functions of single variable 'functions' and binary 'operations'
             		
		defaults:  n=7, vars_consts = (-1, x, E}, functions = {Log}, operations = {Plus, Times, Power} 
		"
ZadanieNOF::usage =
        "
		Zadanie na kolokwium z NOF 19.01.2022
		"
		
ZadanieNOF2024::usage =
        "
		Zadanie na kolokwium z NOF 10.01.2024
		"
ZadanieNOF2025::usage =
        "
		Zadanie na kolokwium poprawkowe z NOF 21.02.2025
		"  
		
EnumerateExpressions::usage =
        "
		EnumerateExpression[] - list of first functions of single variable x
        
		EnumerateExpression[n,vars_consts,functions,operations]- all expressions up to tree depth 'n', composed of symbols 'vars_consts' (terminal symbols)
		using functions of single variable 'functions' and binary 'operations'
             		
		defaults:  n=1, vars_consts = (-1, x, E}, functions = {Log}, operations = {Plus, Times, Power} 
		"

EnumerateExpressionsRPN::usage =
        "
		EnumerateExpressionRPN[K] - list of first functions obtained from RPN calculator button sequence of length K
        
		EnumerateExpression[n,vars_consts,functions,operations]- all expressions composed of symbols 'vars_consts' (terminal symbols)
		using functions of single variable 'functions' and binary 'operations'
             		
		defaults:  K=1, vars_consts = (-1, x, E}, functions = {Log}, operations = {Plus, Times, Power} 
		"

KolmogorovComplexity::usage = "Not yet implemented. Estimate Kolmogorov complexity of the expression in the language defined by base set, e.g. {-1,x, Log, Power}"

VerifyBaseSet::usage = "Check if calculator with provided buttons is still fully functional. Default (minimal known) is: VerifyBaseSet[{},{Exp},{Log}]. Check takes 15 minutes."

RecognizeConstant::usage = " RecognizeConstant[1.38629] - attempt to find best approximation using default setings"

NextFunction::usage = "TODO"


Options[RecognizeConstant] = { PrecisionGoal -> 16*$MachineEpsilon, MaxCodeLength -> 11, Candidates -> 1, WriteToDisk -> False, Finalize->{Abs,Re,Im}, MemoryLimit->131072, TimeLimit->8, StartCodeLength->1, StartCodeNumber->0, DisplayProgress->True};

Options[RecognizeFunction] = { PrecisionGoal -> Sqrt@$MachineEpsilon, MaxCodeLength -> 13, WriteToDisk -> True,  MemoryLimit->64*131072, TimeLimit->64, StartCodeLength->1, StartCodeNumber->0};

Options[RecognizeSequence] = { PrecisionGoal -> 0, MaxCodeLength -> 13, WriteToDisk -> False, MemoryLimit -> 64*131072, TimeLimit -> 1, StartCodeLength -> 1, StartCodeNumber -> 0};

RecognizeFunction::usage = "RecognizeFunction[{{0,0},{1,1},{2,1.41421}}] - search for univariate function approximating input data."
 

RecognizeSequence::usage = "RecognizeSequence[{1, 4, 27, 256, 3125}]  - search formula for sequence of exactly known numbers (integers, rationals etc.)."
		
Begin["`Private`"]

ValidateCodeGeneral[digits_List] := 
 Module[{counter, k, arity}, counter = 0;
  For[k = 1, k <= Length[digits], k++,
   arity = digits[[k]];
   If[arity == 0, counter++;  (*Operand*),
    counter -= arity;(*Operator consumes `arity` operands*)
    If[counter < 0, Return[False]];
    counter++;  (*Operator produces one result*)];];
  Return[counter == 1];
  ]


ValidateCodeGeneralCompiled = 
  Compile[{{digits, _Integer, 1}}, 
   Module[{counter = 0, arity, k = 1, len = Length[digits], 
     valid = True}, While[k <= len && valid, arity = digits[[k]];
     If[arity == 0, counter++;  (*Operand*), 
      counter -= arity;(*Operator consumes operands*)
      If[counter < 0, valid = False];
      counter++;  (*Operator produces one result*)];
     k++;];
    valid && counter == 1](*,CompilationTarget->"C",RuntimeOptions->
   "Speed"*)];

NextFunction[kOLD_: 0, nOLD_: 1, constants_List: {x,E}, 
   functions_List: {Log}, binaryOperations_List: {Plus, Times, Power},
    OptionsPattern[]] := 
  Module[{k, n, num, rpnRule, rule, rule2, funs, ops, language, symb, digits, 
    code, formula, numbers},
	
	(*RPN calculator*)
   funs = If[functions == {}, Alternatives[], functions /. List -> Alternatives];

   ops = binaryOperations /. List -> Alternatives;

   language =     Join[funs, ops];

   varconst = Except[language];
   rpnRule[{a : varconst ..., b : varconst, f : funs, c___}] := rpnRule[{a, f[b], c}];
   rpnRule[{a : varconst ..., b : varconst, c : varconst, op : ops, d___}] :=  rpnRule[{a, op[b, c], d}];
   rpnRule[{rest : Except[language]}] := rest;
   
   
   symb = Join[constants, functions, binaryOperations];
   num = Length[symb];
   rule = (# /. List -> Rule &) /@ 
     Transpose[{Range[0, num - 1], symb}];
   rule2 = (# /. List -> Rule &) /@ 
     Transpose[{Range[0, num - 1], 
       Join[Table[1, Length[constants]], 
        Table[0, Length[functions]], 
        Table[-1, Length[binaryOperations]]]}];
   For[n = nOLD, n <= Infinity, n++,
    For[k = kOLD + 1, k < num^n, k++,
     digits = IntegerDigits[k, num, n];
     numbers = digits /. rule2;
     If[! ValidateCode[numbers], Continue[]];
     code = digits /. rule;
     formula = rpnRule[code];
     Goto[END]
     ]
    ];
   Label[END];
   {formula, code, {k, n}}
   ];

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

ZadanieNOF[depth_Integer:6, var_List:{"Global`x", 2}, fun_List:{"System`Exp","System`Sqrt"}, op_List:{"System`Plus","System`Times","System`Divide"}] :=
   Module[{vars, funs, ops, lang, i, weights, zadanie},
    vars = ToString /@ var;
    funs = Table[ToString[fun[[i]]] <> "[left]", {i, 1, Length[fun]}];
    ops = 
     Table[ToString[op[[i]]] <> "[left,right]", {i, 1, Length[op]}];
    lang = Join[vars, funs, ops];
	weights = Join[Table[4, {i, 2, Length[vars]}], Table[1, {i, 1, Length[funs]}], Table[2, {i, 1, Length[ops]}]];
    zadanie=1;
    While[Simplify@D[zadanie,{Global`x,2}]===0 || Simplify@D[1/zadanie,{Global`x,2}]===0 ||LeafCount[zadanie]<24,
    zadanie = StringReplace[
      FixedPoint[
       StringReplace[#, 
	   {"left" -> RandomChoice[lang], 
        "right" -> RandomChoice[lang]}] &, 
		RandomChoice[Join[funs,ops]], 
       depth], {"left" -> RandomChoice[vars], 
       "right" -> RandomChoice[vars]}] // ToExpression // Simplify;
    ];
    Return[zadanie]
    
   ];
   
Global`leafcounter=12;
ZadanieNOF2024[depth_Integer:4, var_List:{"Global`x", 1,2,3,4,5,6,7,8,9}, fun_List:{"System`Exp","System`Sqrt","System`Sinh","System`Cosh","System`Minus"}, op_List:{"System`Plus","System`Times","System`Divide","System`Subtract"}] :=
   Module[{vars, funs, ops, lang, i, weights, zadanie},
    vars = ToString /@ var;
    funs = Table[ToString[fun[[i]]] <> "[left]", {i, 1, Length[fun]}];
    ops = 
     Table[ToString[op[[i]]] <> "[left,right]", {i, 1, Length[op]}];
    lang = Join[vars, funs, ops];
	weights = Join[Table[4, {i, 2, Length[vars]}], Table[1, {i, 1, Length[funs]}], Table[2, {i, 1, Length[ops]}]];
    zadanie=1;
    While[Simplify@D[zadanie,{Global`x,2}]===0 || Simplify@D[1/zadanie,{Global`x,2}]===0 ||LeafCount[zadanie]<Global`leafcounter,
    zadanie = StringReplace[
      FixedPoint[
       StringReplace[#, 
	   {"left" -> RandomChoice[lang], 
        "right" -> RandomChoice[lang]}] &, 
		RandomChoice[Join[funs,ops]], 
       depth], {"left" -> RandomChoice[vars], 
       "right" -> RandomChoice[vars]}] // ToExpression // Simplify;
    ];
    Global`leafcounter++;
    Return[zadanie]
    
   ];

ZadanieNOF2025[depth_Integer:4, var_List:{"Global`x", 1,2,3,"System`Pi","System`E"}, fun_List:{"System`Sqrt","System`Minus"}, op_List:{"System`Plus","System`Times","System`Divide","System`Subtract"}] :=
   Module[{vars, funs, ops, lang, i, weights, zadanie},
    vars = ToString /@ var;
    funs = Table[ToString[fun[[i]]] <> "[left]", {i, 1, Length[fun]}];
    ops = 
     Table[ToString[op[[i]]] <> "[left,right]", {i, 1, Length[op]}];
    lang = Join[vars, funs, ops];
	weights = Join[Table[4, {i, 2, Length[vars]}], Table[1, {i, 1, Length[funs]}], Table[2, {i, 1, Length[ops]}]];
    zadanie=1;
    While[Simplify@D[zadanie,{Global`x,2}]===0 || Simplify@D[1/zadanie,{Global`x,2}]===0 ||LeafCount[zadanie]<Global`leafcounter|| Im[N[zadanie /. Global`x -> Glaisher]] != 0,
    zadanie = StringReplace[
      FixedPoint[
       StringReplace[#, 
	   {"left" -> RandomChoice[lang], 
        "right" -> RandomChoice[lang]}] &, 
		RandomChoice[Join[funs,ops]], 
       depth], {"left" -> RandomChoice[vars], 
       "right" -> RandomChoice[vars]}] // ToExpression // Simplify;
    ];
    Global`leafcounter++;
    Return[zadanie]
    
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

EnumerateExpressionsRPN[K_, constants_List : {-1, I, E, Pi, 2}, 
   functions_List : {Log}, 
   binaryOperations_List : {Plus, Times, Power}] := 
  Module[{k, d, rpnRule, funs, ops, language},
   (*RPN calculator*)
   funs = If[functions == {}, Null, functions /. List -> Alternatives];
   ops = binaryOperations /. List -> Alternatives;
   language = 
    Join[functions, binaryOperations] /. List -> Alternatives;
   rpnRule[{a : Except[language] ..., b : Except[language], 
      c : Except[language], op : ops, d___}] := 
    rpnRule[{a, op[b, c], d}];
   rpnRule[{a : Except[language] ..., b : Except[ops | funs], 
      f : funs, c___}] := rpnRule[{a, f[b], c}];
   rpnRule[{rest : Except[language]}] := rest;
   
   
   Table[
    d = IntegerDigits[k, 3, K];
    If[ValidateCodeGeneralCompiled[d], 
     rpnRule /@ 
      Tuples[d /. {0 -> constants, 1 -> functions, 
         2 -> binaryOperations}], Nothing]
    , {k, 0, -1 + 3^K}]
   
   ];

RecognizeConstant[target_?NumericQ, 
   constants_List : {-1, I, E, Pi, 2}, functions_List : {Log}, 
   binaryOperations_List : {Plus, Times, Power}, OptionsPattern[]] := 
  Module[{k, n, num, rule, rule2, funs, ops, language, symb, x, 
    bestError, digits, code, formula, error, errors, final, formulaN, 
    rpnRule, currentBestFormula, candidates},(*RPN calculator*)
   funs = If[functions == {}, Null, functions /. List -> Alternatives];
   ops = binaryOperations /. List -> Alternatives;
   language = 
    Join[functions, binaryOperations] /. List -> Alternatives;
   rpnRule[{a : Except[language] ..., b : Except[language], 
      c : Except[language], op : ops, d___}] := 
    rpnRule[{a, op[b, c], d}];
   rpnRule[{a : Except[language] ..., b : Except[ops | funs], 
      f : funs, c___}] := rpnRule[{a, f[b], c}];
   rpnRule[{rest : Except[language]}] := rest;
   symb = Join[constants, functions, binaryOperations];
   num = Length[symb];
   rule = (# /. List -> Rule &) /@ 
     Transpose[{Range[0, num - 1], symb}];
  
   
   bestError = Infinity;
   candidates = {};
   If[OptionValue[DisplayProgress], Print["n=",Dynamic[n]," k=",Dynamic[k],"\t",Dynamic[code]];]; 
   Catch[
    
    For[n = OptionValue[StartCodeLength], 
     n <= OptionValue[MaxCodeLength], n++,
     For[k = OptionValue[StartCodeNumber], k < 3^n, k++,
      (*Print["\n\n"];Print[{n,k,num}];*)
      digits = IntegerDigits[k, 3, n];
      (*Print[digits];*)
      If[! ValidateCodeGeneralCompiled[digits], Continue[]];
      
      choices = digits /. {0 -> constants, 1 -> functions, 2 -> ops};
      
      t = Tuples[choices];
      
      For[m = 1, m <= Length[t], m++,
       
       CheckAbort[(
          code = t[[m]];
          (*Print[code];*)
          formula = 
           TimeConstrained[
            MemoryConstrained[
             Check[Block[{Internal`$MinExponent = -1024, 
                Internal`$MaxExponent = 1024}, rpnRule[code]], 
              Infinity], OptionValue[MemoryLimit], Infinity], 
            OptionValue[TimeLimit], Infinity];
          (*Print[formula];*)
          Catch[If[! 
             MachineNumberQ[
              TimeConstrained[formula // N, OptionValue[TimeLimit]]], 
            Continue[]], _SystemException, Continue[] &];
          
          formulaN = 
           Catch[Check[
             MemoryConstrained[
              Block[{Internal`$MinExponent = -1024, 
                Internal`$MaxExponent = 1024}, N[formula, 32]], 
              OptionValue[MemoryLimit], Infinity], 
             Infinity], _SystemException, Infinity &];
          (*Print[formulaN];Print["\n\n"];*)
          errors = 
           Table[{Abs[target - final[formulaN]], final}, {final, 
              Flatten[{OptionValue[Finalize]}]}] // NumericalSort;
          error = errors[[1, 1]] // Chop;
          If[error < bestError, bestError = error;
           currentBestFormula = errors[[1, 2]][formula];
           AppendTo[code, errors[[1, 2]]];
           
           AppendTo[
            candidates, {currentBestFormula, error, code, target}];
           
           If[OptionValue[WriteToDisk] == True, 
            Export["candidatesList_" <> 
              DateString[{"_", "Year", "Month", "Day", "_", "Hour", 
                "Minute"}] <> ".m", candidates];
            
            Print[DateString["ISODateTime"], "\n", code, " err=", 
             error, " n=", n, " k=", k, "\t", currentBestFormula, 
             " = ", errors[[1, 2]][formulaN]]];];
          
          If[bestError <= OptionValue[PrecisionGoal], 
           Throw@Return[
             candidates[[-Min[Length[candidates], 
                  OptionValue[Candidates]] ;; -1]]]];), 
         Print["Best so far:\t", currentBestFormula, "\t", bestError, 
          "\tTested so far:\t", {n, k}, "\tLast code:\t", code]; 
         Abort[];];
       ];
      ]
     (*Print["Level\t",n,"\tcompleted..."];*)
     ];
    candidates[[-Min[Length[candidates], 
         OptionValue[Candidates]] ;; -1]]]];


(* ---------------------- END OF RecognizeConstant ---------------------- *)

RecognizeFunction[data_?ListQ, constants_List : {-1, I, E, Pi, 2}, 
   functions_List : {Log}, 
   binaryOperations_List : {Plus, Times, Power}, OptionsPattern[]] := 
  Module[{k, n, ii, num, rule, constANDvars, funs, ops, 
    language, symb, bestError, digits, code, formula, error, errors, 
    final, formulaN, rpnRule, currentBestFormula, 
    candidates},(*RPN calculator*)
   funs = If[functions == {}, Null, functions /. List -> Alternatives];
   ops = binaryOperations /. List -> Alternatives;
   language = 
    Join[functions, binaryOperations] /. List -> Alternatives;
   rpnRule[{a : Except[language] ..., b : Except[language], 
      c : Except[language], op : ops, d___}] := 
    rpnRule[{a, op[b, c], d}];
   rpnRule[{a : Except[language] ..., b : Except[ops | funs], 
      f : funs, c___}] := rpnRule[{a, f[b], c}];
   rpnRule[{rest : Except[language]}] := rest;
   symb = Join[{x}, constants, functions, binaryOperations];
   constANDvars = Join[{x}, constants];
   num = Length[symb];
   rule = (# /. List -> Rule &) /@ 
     Transpose[{Range[0, num - 1], symb}];

   bestError = Infinity;
   candidates = {};
   Print["n=", Dynamic[n], " k=", Dynamic[k], "\t", Dynamic[code]];
   Catch[
    For[n = OptionValue[StartCodeLength], 
     n <= OptionValue[MaxCodeLength], n++,
     For[k = OptionValue[StartCodeNumber], k < 3^n, k++,
      digits = IntegerDigits[k, 3, n];
      (*Print[digits];*)
      If[! ValidateCodeGeneralCompiled[digits], Continue[]];
      
      choices = 
       digits /. {0 -> constANDvars, 1 -> functions, 2 -> ops};
      
      t = Tuples[choices];
      
      For[m = 1, m <= Length[t], m++,
       
       CheckAbort[(
         
         code = t[[m]];
         
         
         formula = 
          TimeConstrained[
           MemoryConstrained[
            Check[Block[{Internal`$MinExponent = -1024, 
               Internal`$MaxExponent = 1024}, rpnRule[code]], 
             Infinity], OptionValue[MemoryLimit], Infinity], 
           OptionValue[TimeLimit], Infinity];
         (*Print[formula];*)
         Catch[If[! 
            MachineNumberQ[
             TimeConstrained[(formula /. x -> data[[1, 1]]) // N, 
              OptionValue[TimeLimit]]], Continue[]], _SystemException,
           Continue[] &];
         
         formulaN = 
          Catch[Check[
            MemoryConstrained[
             Block[{Internal`$MinExponent = -1024, 
               Internal`$MaxExponent = 1024}, 
              N[(formula /. x -> data[[1, 1]]), 32]], 
             OptionValue[MemoryLimit], Infinity], 
            Infinity], _SystemException, Infinity &];
         (*Print[formulaN];Print["\n\n"];*)
         error = Sum[
           Abs[data[[ii, 2]] - (formula /. 
                x -> data[[ii, 1]])]^2, {ii, 1, Length[data]}];
         If[error < bestError, bestError = error;
          currentBestFormula = formula /. x -> "x";
          
          AppendTo[
           candidates, {currentBestFormula, error, 
            code /. x -> "x"}];
          
          If[OptionValue[WriteToDisk] == True, 
           Export["candidatesList_" <> 
             DateString[{"_", "Year", "Month", "Day", "_", "Hour", 
               "Minute"}] <> ".m", candidates]; 
           Print[DateString["ISODateTime"], "\n", code /. x -> "x", 
            " err=", error, " n=", n, " k=", k, "\t", 
            currentBestFormula]];];
         
         If[bestError <= OptionValue[PrecisionGoal], 
          Throw@Return[candidates[[-1]]]];), 
        Print["Best so far:\t", currentBestFormula, "\t", bestError, 
         "\tTested so far:\t", {n, k}, "\tLast code:\t", code]; 
        Abort[];
        ]
       ];
      ]
     (*Print["Level\t",n,"\tcompleted..."];*)];
    candidates[[-1]]]];


(* ---------------------- END OF RecognizeFunction ---------------------- *)


RecognizeSequence[seq_?ListQ,constants_List:{1},functions_List:{Prime,Factorial,Fibonacci},binaryOperations_List:{Plus,Subtract,Times,Power},OptionsPattern[]]:=Module[{data,k,K,ii,num,rule,constANDvars,funs,ops,language,symb,bestError,digits,code,formula,error,errors,final,formulaN,rpnRule,currentVals, currentBestFormula,candidates},

data=If[VectorQ[seq],Transpose[{Range[Length[seq]],seq}],seq,seq];
Print[data];
(*RPN calculator*)

funs=If[functions=={},Null,functions/. List->Alternatives];
ops=binaryOperations/. List->Alternatives;
language=Join[functions,binaryOperations]/. List->Alternatives;
rpnRule[{a:Except[language]...,b:Except[language],c:Except[language],op:ops,d___}]:=rpnRule[{a,op[b,c],d}];
rpnRule[{a:Except[language]...,b:Except[ops|funs],f:funs,c___}]:=rpnRule[{a,f[b],c}];
rpnRule[{rest:Except[language]}]:=rest;
symb=Join[{n},constants,functions,binaryOperations];
constANDvars=Join[{n},constants];
num=Length[symb];
rule=(#/. List->Rule&)/@Transpose[{Range[0,num-1],symb}];

bestError=Infinity;
candidates={};
(*Print["K=",Dynamic[K]," k=",Dynamic[k],"\t",Dynamic[code]];*)
Catch[For[K=OptionValue[StartCodeLength],K<=OptionValue[MaxCodeLength],K++,
For[k=OptionValue[StartCodeNumber],k<3^K,k++,
digits=IntegerDigits[k,3,K];

If[!ValidateCodeGeneralCompiled[digits],Continue[]];
choices=digits/. {0->constANDvars,1->functions,2->ops};
t=Tuples[choices];
For[m=1,m<=Length[t],m++,
CheckAbort[
(code=t[[m]];
formula=TimeConstrained[
MemoryConstrained[Check[Block[{Internal`$MinExponent=-1024,Internal`$MaxExponent=1024},rpnRule[code]],Infinity],OptionValue[MemoryLimit],Infinity],OptionValue[TimeLimit],Infinity];

For[ii=1,ii<=Length[seq],ii++,
currentVal=

TimeConstrained[
MemoryConstrained[Check[Block[{Internal`$MinExponent=-1024,Internal`$MaxExponent=1024},formula/.n->data[[ii,1]]],Infinity],OptionValue[MemoryLimit],Infinity
],OptionValue[TimeLimit],Infinity
];

If[currentVal!=data[[ii,2]],Break[]];

];

lastOK=If[ii>Length[seq],Length[seq],ii];

error=N[Length[seq]-lastOK+Abs[currentVal-data[[lastOK,2]]]];

If[error<bestError,bestError=error;
currentBestFormula=formula/. n->"n";
AppendTo[candidates,{currentBestFormula,error,code/. n->"n"}];
If[OptionValue[WriteToDisk]==True,Export["candidatesList_"<>DateString[{"_","Year","Month","Day","_","Hour","Minute"}]<>".m",candidates];
Print[DateString["ISODateTime"],"\n",code/. n->"n"," err=",error," K=",K," k=",k,"\t",currentBestFormula]];];
If[bestError<=OptionValue[PrecisionGoal],Throw@Return[candidates[[-1]]]];),Print["Best so far:\t",currentBestFormula,"\t",bestError,"\tTested so far:\t",{K,k},"\tLast code:\t",code];
Abort[];]];];
Print["Level\t",K,"\tcompleted..."
];
];
candidates[[-1]]]];

(* ---------------------- END OF RecognizeSequence ---------------------- *)



VerifyBaseSet[constants_List : {}, functions_List : {Exp}, 
  operations_List : {Log}] := Module[
  {startTime, endTime, CALC3, CALC4, const, fun, op, maxK, K, 
   newCALC3, ii, jj, newItemFound},
  (*Define custom functions*)
  Inv[x_] := 1/x;
  Sqr[x_] := x^2;
  Dbl[x_] := 2 x;
  Half[x_] := x/2;
  Suc[x_] := x + 1;
  Pre[x_] := x - 1;
  Avg[x_, y_] := (x + y)/2;
  Hypot[x_, y_] := Sqrt[x^2 + y^2];

  rename={SymbolicRegression`Private`Inv->"Inv", 
          SymbolicRegression`Private`Sqr->"Sqr", 
          SymbolicRegression`Private`Dbl->"Dbl", 
          SymbolicRegression`Private`Half->"Half", 
          SymbolicRegression`Private`Suc->"Suc", 
          SymbolicRegression`Private`Pre->"Pre", 
          SymbolicRegression`Private`Avg->"Avg", 
          SymbolicRegression`Private`Hypot->"Hypot" 
         };
  
  (*Suppress specific warnings*)
  Off[Power::infy, Power::indet];
  Off[Infinity::indet];
  Off[General::ovfl, General::munfl, General::unfl];
  Off[N::meprec];
  Off[Divide::indet, Divide::infy];
  
  startTime = Now;
  
  (*Set RecognizeConstant options*)
  SetOptions[RecognizeConstant, TimeLimit -> 1, 
   Finalize -> {Identity}, DisplayProgress -> False];
  
  (*Define initial calculators*)
  CALC3 = {Join[{Glaisher, EulerGamma}, constants], functions, 
    operations};
  
  CALC4 = {{Glaisher, EulerGamma, Pi, E, I, 1, -1, 
     2}, {Half, Minus, Log, Exp, Inv, Sqrt, 
     Sqr, Cosh, Cos, Sinh, Sin, Tanh, Tan, ArcSinh, ArcTanh, ArcSin, 
     ArcCos, ArcTan, ArcCosh, LogisticSigmoid}, {Plus, Times, Subtract, 
     Divide, Power, Log, Avg, Hypot}};
  
  (*Prepare constants,functions,and operations*)
  const = 
   Table[{CALC4[[1, jj]], CALC4[[1, jj]]}, {jj, 1, 
     Length[CALC4[[1]]]}];
  fun    = 
   Table[{CALC4[[2, jj]], CALC4[[2, jj]][EulerGamma]}, {jj, 1, 
     Length[CALC4[[2]]]}];
  op       = 
   Table[{CALC4[[3, jj]], CALC4[[3, jj]][EulerGamma, Glaisher]}, {jj, 
     1, Length[CALC4[[3]]]}];
  
  
  (*Initialize variables*)
  newCALC3 = CALC3;
  maxK = 10;  (*Set a maximum value for K to prevent infinite loops*)
  K = 1;      (*Start with K=1*)
  
  (*Main loop*)
  While[K <= 
     maxK && (Length[const] > 0 || Length[fun] > 0 || 
      Length[op] > 0),
   Print["Testing with K = ", K, "\t", Now];
   newItemFound = False;
   
   (*Recognize operations*)
   Print["Recognizing operations started\t", Now];
   ii = 1;
   While[ii <= Length[op], Module[{res},
      res = 
       RecognizeConstant[op[[ii, 2]], newCALC3[[1]], newCALC3[[2]], 
        newCALC3[[3]], StartCodeLength -> K, MaxCodeLength -> K];
      If[Length[res] > 0 && res[[1, 2]] <= 16 $MachineEpsilon,
       (*New operation found*)
       newCALC3[[3]] = Union[newCALC3[[3]], {op[[ii, 1]]}];
       Print["\nFound new operation: ", op[[ii, 1]] /. rename, "\t", res /. rename, 
        "\n"];
       op = Delete[op, ii];
       newItemFound = True;
       K = 1;(*Reset K*)Break[];  (*Exit the inner While loop*), 
       ii++
       ];
      ];
    ];
   
   If[newItemFound,
    Print["Remaining constants: ", const[[All, 1]] /. rename];
    Print["Remaining functions: ",   fun[[All, 1]] /. rename];
    Print["Remaining operations: ",   op[[All, 1]] /. rename];
    Print["Current CALC3: ", newCALC3 /. rename]; 
    Continue[]
    ];
   
   
   (*Recognize constants*)
   Print["Recognizing constants started\t", Now];
   ii = 1;
   While[ii <= Length[const], Module[{res},
      res = 
       RecognizeConstant[const[[ii, 2]], newCALC3[[1]], newCALC3[[2]],
         newCALC3[[3]], StartCodeLength -> K, MaxCodeLength -> K];
      If[Length[res] > 0 && res[[1, 2]] <= 16 $MachineEpsilon,
       (*New constant found*)
       newCALC3[[1]] = Union[newCALC3[[1]], {const[[ii, 1]]}] // Sort // Reverse;
       Print["\nFound new constant: ", const[[ii, 1]], "\t", res /. rename, 
        "\n"];
       const = Delete[const, ii];
       newItemFound = True;
       K = 1;(*Reset K*)
       Break[];  (*Exit the inner While loop*), ii++];
      ];
    ];
   If[newItemFound,
    Print["Remaining constants: ", const[[All, 1]] /. rename];
    Print["Remaining functions: ",   fun[[All, 1]] /. rename];
    Print["Remaining operations: ",   op[[All, 1]] /. rename];
    Print["Current CALC3: ", newCALC3 /. rename]; 
    Continue[]];
   
   (*Recognize functions*)
   Print["Recognizing functions started\t", Now];
   ii = 1;
   While[ii <= Length[fun], Module[{res},
      res = 
       RecognizeConstant[fun[[ii, 2]], newCALC3[[1]], newCALC3[[2]], 
        newCALC3[[3]], StartCodeLength -> K, MaxCodeLength -> K];
      If[Length[res] > 0 && res[[1, 2]] <= 16 $MachineEpsilon,
       (*New function found*)
       newCALC3[[2]] = Union[newCALC3[[2]], {fun[[ii, 1]]}];
       Print["\nFound new function: ", fun[[ii, 1]] /. rename, "\t", res /. rename , 
        "\n"];
       fun = Delete[fun, ii];
       newItemFound = True;
       K = 1;(*Reset K*)Break[];  (*Exit the inner While loop*), 
       ii++];];
    ];
   If[newItemFound,
    Print["Remaining constants: ", const[[All, 1]] /. rename];
    Print["Remaining functions: ",   fun[[All, 1]] /. rename];
    Print["Remaining operations: ",   op[[All, 1]] /. rename];
    Print["Current CALC3: ", newCALC3 /. rename]; 
    Continue[]];
   
   
   (*If no new items found at this K,increment K*)
   If[Not[newItemFound],
    Print["No new items found at K = ", K];
    K++;];
   ];
  
  (*Final results*)
  Print["Final newCALC3: ", newCALC3  /. rename];
  endTime = Now;
  Print[DateDifference[startTime, endTime, "Minutes"]];
  If[Length[const] == 0 && Length[fun] == 0 && Length[op] == 0, 
   Print["All items recognized! Calculator is still fully functional..."];,
   Print["Remaining constants: ", const[[All, 1]]];
   Print["Remaining functions: ", fun[[All, 1]]];
   Print["Remaining operations: ", op[[All, 1]]];];
  ]


  
End[ ]

EndPackage[ ]

