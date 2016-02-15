# SymbolicRegressionPackage

Basic building blocks for brute-force and random symbolic regression methods in Mathematica


Package will provides various implementations of typical symbolic regression functions:

- random generators for mathematical expressions
- enumerators for all expressions possible to obtain given soba basic building blocks
- tools for testing and verification of methods, mianimal basic sets  in particular
- functions estimating Kolmogorov complexity of the mathematical formulae
- numeric constant recognition 
- numeric function recognition

Typical applications are:
- automated generation of random tests and tasks for STEM students, free of ,,aunt Wikipedia and uncle Google'' solutions
- exhaustive tests of Computer Alegebra Systems (CAS)
- recreational math
- brute-force search for special solutions of mathematical problems (ODE, PDE, integration, delayed and functional equations etc.)
- statistical analysis of mathematical formulae (e.g. what fraction of integer coefficient quadratic equations have two distinct real solutions)


Main functions are:

RandomExpression[] - generates random mathematical formula [implemented using random-walk string-rewrite approach]
EnumerateExpressions[] - lists all expressions [implemented recursively; this limits searach to level 4 at most due to memory-hungry functions, stack machine method
to be implemented soon]

Some simple examples and applications are presented in SymbolicRegreessionPackage_Examples.nb notebook. 

NOTE: constants are good functions, random generator do not care what you expect!



