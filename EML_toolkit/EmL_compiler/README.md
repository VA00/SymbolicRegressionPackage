Compiler is able to convert any elementary formula (''function '') to pure EML form. Input should be proper Wolfram Mathematica expression with correct syntax. For those who do not use Mathematica, this usually means square brackets, and function name starting uppercase, e.g. Sin[x] vs sin(x). 

For example, to obtain EML form for natural logarithm, use

python3 .\eml_compiler_v4.py Log[x]

Expected output:

EML[1,EML[EML[1,x],1]]

Verify in Mathematica by back-substitution of the EML:

EML[1, EML[EML[1, x], 1]] /. EML -> Function[{x, y}, Exp[x] - Log[y]]

Resulting expressions usually require heavy simplification. Try FullSimplify and specify real domain or range for x; use PowerExpand for quick check. 
