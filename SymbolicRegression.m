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

RecognizeConstant::usage = "Not yet implemented. Use e.g. RIES instead"

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
   
End[ ]

EndPackage[ ]