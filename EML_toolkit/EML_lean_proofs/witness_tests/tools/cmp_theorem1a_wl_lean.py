import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from terms import *
lean_dir, wl_file = sys.argv[1], sys.argv[2]
st = Store(); LE = Defs(st); WL = Defs(st)
for f in ["Core", "Arith", "UnitStep", "ExpLog", "Trig", "Complex/Arith", "Complex/PowLog", "Complex/Functions", "Complex/ZeroTest"]:
    with open(os.path.join(lean_dir, f + ".lean"), encoding="utf-8") as fh: load(LE, lean_defs(fh.read()))
with open(wl_file, encoding="utf-8") as fh: load(WL, wl_defs(fh.read(), until="Exact checks"), skip=("eml",))
pairs = [("EPS","E",1),("e","const_e",0),("L","L",1),("S","S",2),("NU","N",1),("zero","const_0",0),("neg","const_neg",0),
 ("two","const_2",0),("half","const_half",0),("M","M",2),("DELTA","D",2),("mipi","const_minus_ipi",0),("img","const_i",0),
 ("pi","const_pi",0),("pi2","const_pi_half",0),("twoi","const_2i",0),("ipi","const_ipi",0),("twoipi","const_2ipi",0),
 ("pinf","const_pinf",0),("minf","const_ninf",0),("Winv","Winv1a",1),("Wx","Wmul1a",2),("mimg","const_minus_i",0),
 ("Wminus","Wsub1a",2),("Wneg","Wminus1a",1),("Wplus","Wadd1a",2),("Wsqr","Wsqr1a",1),("Whalf","Whalf1a",1),
 ("Wavg","Wavg1a",2),("Wdiv","Wdiv1a",2),("Wpow","Wpow1a",2),("Wsqrt","Wsqrt1a",1),("Whypot","Whypot1a",2),
 ("Wln","Wln1a",1),("Wlog","Wlog1a",2),("WSIGMA","Wsigma1a",1),("Wsinh","Wsinh1a",1),("Wcosh","Wcosh1a",1),
 ("Wcos","Wcos1a",1),("Wsin","Wsin1a",1),("Wtan","Wtan1a",1),("Wtanh","Wtanh1a",1),("Warcsin","Warcsin1a",1),
 ("Warccos","Warccos1a",1),("Warsinh","Warsinh1a",1),("Warcosh","Warcosh1a",1),("Warctan","Warctan1a",1),
 ("Wartanh","Wartanh1a",1),("J","J",1),("WinvLean","Winv1a_lean",1),("WlnLean","Wln1a_lean",1),
 ("WlnWolfram","Wln1a_wolfram",1),("dcut","dcut",1)]
sys.exit(1 if compare(st, WL, LE, pairs, "Theorem 1a, WL against Lean") else 0)
