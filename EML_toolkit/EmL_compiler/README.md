Compiler is able to convert any elementary function formula to pure EML form. Input expression should be proper Wolfram Mathematica form.

For example, to obtain EML form for natural logarithm, use

python3 .\eml_compiler_v4.py Log[x]

Expected output:

EML[1,EML[EML[1,x],1]]

Verify in Mathematica:

EML[1, EML[EML[1, x], 1]] /. EML -> Function[{x, y}, Exp[x] - Log[y]]

Expressions usually require heavy simplification. Try FullSimplify and specify real domain or range for x, use PowerExpand for quick check. 
