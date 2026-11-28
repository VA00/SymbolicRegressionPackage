# The Python scripts of Theorems 1 and 1a against the Lean definitions: the same trees?
# The term definitions of a script are run with eml replaced by the builder of trees.
import sys, os, re
sys.path.insert(0, os.path.dirname(__file__))
from terms import *
lean_dir, py1, py1a = sys.argv[1], sys.argv[2], sys.argv[3]
st = Store(); LE = Defs(st)
for f in ["Core", "Arith", "UnitStep", "ExpLog", "Trig", "Complex/Arith", "Complex/PowLog", "Complex/Functions", "Complex/ZeroTest"]:
    with open(os.path.join(lean_dir, f + ".lean"), encoding="utf-8") as fh: load(LE, lean_defs(fh.read()))

def check(ns, pairs, label):
    bad = 0
    for na, nb, k in pairs:
        args = [Store.X, Store.Y][:k]
        try:
            a = ns[na](*args) if k else ns[na]
            b = LE.value(nb, args)
        except Exception as ex:
            print("  ERROR  %-12s %-18s %r" % (na, nb, ex)); bad += 1; continue
        if a != b: print("  DIFF   %-12s %-18s leaves %s / %s" % (na, nb, st.leaves[a], st.leaves[b])); bad += 1
    print("%s: %d pairs, %d differ" % (label, len(pairs), bad))
    return bad

# ---- verify_theorem1.py: the definitions stand in one block
with open(py1, encoding="utf-8") as fh: src = fh.read()
ns = py_terms(src, st, "one = mpf(1)", "# ---- reference")
p1 = [("E","E",1),("e","const_e",0),("L","L",1),("S","S",2),("N","N",1),("zero","const_0",0),("neg","const_neg",0),
 ("two","const_2",0),("three","const_3",0),("half","const_half",0),("M","M",2),("D","D",2),
 ("mipi","const_minus_ipi",0),("img","const_i",0),("PI","const_pi",0),("PI2","const_pi_half",0),("twoi","const_2i",0),
 ("ipi","const_ipi",0),("twoipi","const_2ipi",0),("U","U",1),("Wminus","Wsub",2),("Wplus","Wadd",2),("Wsqr","Wsqr",1),
 ("Whalf","Whalf",1),("Wx","Wmul",2),("Winv","Winv",1),("Wdiv","Wdiv",2),("Wavg","Wavg",2),("X","X",1),("T","T",1),
 ("tau","tau",1),("w","wH",1),("H","H",1),("Wsqrt","Wsqrt",1),("Whypot","Whypot",2),("Wsigma","Wsigma",1),
 ("Wpow","Wpow",2),("Wlog","Wlog",2),("Wsinh","Wsinh",1),("Wcosh","Wcosh",1),("Wtanh","Wtanh",1),("Warsinh","Warsinh",1),
 ("Warcosh","Warcosh",1),("Wartanh","Wartanh",1),("A","A",2),("Wcos","Wcos",1),("Wsin","Wsin",1),("Wtan","Wtan",1),
 ("Warcsin","Warcsin",1),("Warccos","Warccos",1),("Warctan","Warctan",1)]
bad = check(ns, p1, "Theorem 1, Python against Lean")

# ---- verify_theorem1a.py: the definitions stand in one block
with open(py1a, encoding="utf-8") as fh: src = fh.read()
ns = py_terms(src, st, "one = mpc(1)", "# ---- reference")
p1a = [("E","E",1),("e","const_e",0),("L","L",1),("S","S",2),("N","N",1),("zero","const_0",0),("neg","const_neg",0),
 ("two","const_2",0),("half","const_half",0),("M","M",2),("D","D",2),("mipi","const_minus_ipi",0),("img","const_i",0),
 ("PI","const_pi",0),("PI2","const_pi_half",0),("twoi","const_2i",0),("ipi","const_ipi",0),("twoipi","const_2ipi",0),
 ("pinf","const_pinf",0),("minf","const_ninf",0),("Winv","Winv1a",1),("Wx","Wmul1a",2),("mimg","const_minus_i",0),
 ("Wminus","Wsub1a",2),("Wneg","Wminus1a",1),("Wplus","Wadd1a",2),("Wsqr","Wsqr1a",1),("Whalf","Whalf1a",1),
 ("Wavg","Wavg1a",2),("Wdiv","Wdiv1a",2),("Wpow","Wpow1a",2),("Wsqrt","Wsqrt1a",1),("Whypot","Whypot1a",2),
 ("Wln","Wln1a",1),("Wlog","Wlog1a",2),("Wsigma","Wsigma1a",1),("Wsinh","Wsinh1a",1),("Wcosh","Wcosh1a",1),
 ("Wcos","Wcos1a",1),("Wsin","Wsin1a",1),("Wtan","Wtan1a",1),("Wtanh","Wtanh1a",1),("Warcsin","Warcsin1a",1),
 ("Warccos","Warccos1a",1),("Warsinh","Warsinh1a",1),("Warcosh","Warcosh1a",1),("Warctan","Warctan1a",1),
 ("Wartanh","Wartanh1a",1),("J","J",1),("WinvLean","Winv1a_lean",1),("WlnLean","Wln1a_lean",1),("WlnWolfram","Wln1a_wolfram",1)]
bad += check(ns, p1a, "Theorem 1a, Python against Lean")
sys.exit(1 if bad else 0)
