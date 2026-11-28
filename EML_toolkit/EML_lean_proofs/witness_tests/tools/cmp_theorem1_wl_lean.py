import sys, os, glob
sys.path.insert(0, os.path.dirname(__file__))
from terms import *
lean_dir, wl_file = sys.argv[1], sys.argv[2]
st = Store()
LE = Defs(st)
for f in ["Core", "Arith", "UnitStep", "ExpLog", "Trig"]:
    load(LE, lean_defs(open(os.path.join(lean_dir, f + ".lean"), encoding="utf-8").read()))
WL = Defs(st)
load(WL, wl_defs(open(wl_file, encoding="utf-8").read(), until="Exact checks"), skip=("eml",))
print("Lean definitions:", len(LE.d), " WL definitions:", len(WL.d))
print("Lean:", " ".join(sorted(LE.d)))
print("WL:  ", " ".join(sorted(WL.d)))
pairs = [("EPS","E",1),("e","const_e",0),("L","L",1),("S","S",2),("NU","N",1),("zero","const_0",0),("neg","const_neg",0),
 ("two","const_2",0),("three","const_3",0),("half","const_half",0),("M","M",2),("DELTA","D",2),
 ("mipi","const_minus_ipi",0),("img","const_i",0),("pi","const_pi",0),("pi2","const_pi_half",0),("twoi","const_2i",0),
 ("ipi","const_ipi",0),("twoipi","const_2ipi",0),("U","U",1),("Wminus","Wsub",2),("Wplus","Wadd",2),("Wsqr","Wsqr",1),
 ("Whalf","Whalf",1),("Wx","Wmul",2),("Winv","Winv",1),("Wdiv","Wdiv",2),("Wavg","Wavg",2),("X","X",1),("T","T",1),
 ("TAU","tau",1),("w","wH",1),("H","H",1),("Wsqrt","Wsqrt",1),("Whypot","Whypot",2),("WSIGMA","Wsigma",1),
 ("Wpow","Wpow",2),("Wlog","Wlog",2),("Wsinh","Wsinh",1),("Wcosh","Wcosh",1),("Wtanh","Wtanh",1),("Warsinh","Warsinh",1),
 ("Warcosh","Warcosh",1),("Wartanh","Wartanh",1),("A","A",2),("Wcos","Wcos",1),("Wsin","Wsin",1),("Wtan","Wtan",1),
 ("Warcsin","Warcsin",1),("Warccos","Warccos",1),("Warctan","Warctan",1)]
sys.exit(1 if compare(st, WL, LE, pairs, "Theorem 1, WL against Lean") else 0)
