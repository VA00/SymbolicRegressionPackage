# SymbolicRegressionPackage

Basic building blocks for brute-force and random symbolic regression methods in Mathematica


Package will provides various implementations of typical symbolic regression functions:


- numeric constant recognition (Prototype using RPN implemented with up to 2-argument operations) 
- random generators for mathematical expressions
- enumerate all possible expressions composed of given basic building blocks
- tools for testing and verification of methods
- functions estimating Kolmogorov complexity of the given mathematical formula

Typical applications are:
- automated generation of new test problems for students & AI, free of ,,aunt Wikipedia and uncle Google'' solutions
- exhaustive and random tests of Computer Algebra Systems (CAS)
- recreational math
- brute-force search for special solutions of mathematical problems (ODE, PDE, integration, delayed and functional equations etc.)
- statistical analysis of mathematical formulae (e.g. what fraction of integer coefficient quadratic equations have two distinct real solutions)


Main functions are:

RecognizeConstant[x] - search for ,,analytical'' formula given numeric value in the order of growing Kolmogorov complexity expressions. Any function or constant known to Mathematica can be used, e.g. EllipticE, Root[], Gamma etc. 
RecognizeFunction - like previous, but for univariate data with possible errors
RecognizeSequence - like previous, but for integer sequences or other exact symbolic data

RandomExpression[] - generates random mathematical formula [implemented using random-walk string-rewrite approach]
EnumerateExpressions[] - lists all expressions [implemented recursively; this limits search to level 4 at most due to memory-hungry functions, stack machine methodvto be implemented soon]

Some simple examples and applications are presented in SymbolicRegreessionPackage_Examples.nb notebook. 

NOTE: constants (constant functions, like f[x_]:=0) are also proper functions, random generator do not care what you expect!



