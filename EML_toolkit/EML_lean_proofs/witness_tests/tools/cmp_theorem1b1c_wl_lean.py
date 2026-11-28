import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from terms import *
lean_dir, wl_file = sys.argv[1], sys.argv[2]
st = Store(); LE = Defs(st); WL = Defs(st)
for f in ["Core", "Real/Strict", "Real/Extended", "Real/Continuity"]:
    with open(os.path.join(lean_dir, f + ".lean"), encoding="utf-8") as fh: load(LE, lean_defs(fh.read()))
with open(wl_file, encoding="utf-8") as fh: load(WL, wl_defs(fh.read(), until="Exact checks"), skip=("eml", "mode", "emlCount", "simp", "sgn"))
pairs = [("EPS","E",1),("L","L",1),("F","F1b",2),("B","B1b",1),("wplus","Wadd1b",2),("wminus","Wsub1b",2),("e","const_e",0),
 ("zero","const_0",0),("wneg","Wminus1b",1),("neg","const_neg1b",0),("two","const_2_1b",0),("G","G1b",2),("wx","Wmul1b",2),
 ("wsqr","Wsqr1b",1),("winv","Winv1b",1),("wdiv","Wdiv1b",2),("whalf","Whalf1b",1),("wavg","Wavg1b",2),("wSIGMA","Wsigma1b",1),
 ("wpow","Wpow1b",2),("wlog","Wlog1b",2),("wsinh","Wsinh1b",1),("wcosh","Wcosh1b",1),("wtanh","Wtanh1b",1),("Q","Q1b",1),
 ("warsinh","Warsinh1b",1),("wartanh","Wartanh1b",1),("wsqrtPos","Wsqrt1b",1),("whypotOff","Whypot1b",2),
 ("warcoshGt1","Warcosh1b",1),("pinf","const_pinf1c",0),("minf","const_ninf1c",0),("wsqrtBar","Wsqrt1c",1),
 ("whypotBar","Whypot1c",2),("warcoshBar","Warcosh1c",1),("wabs","Wabs1c",1),("wrelu","Wrelu1c",1),("wmax","Wmax1c",2)]
sys.exit(1 if compare(st, WL, LE, pairs, "Theorems 1b and 1c, WL against Lean") else 0)
