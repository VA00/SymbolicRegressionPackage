# verify_theorem1b_1c.py (Theorems 1b, 1c) against the Lean definitions: the same trees?
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from terms import *
lean_dir, py = sys.argv[1], sys.argv[2]
st = Store(); LE = Defs(st)
for f in ["Core", "Real/Strict", "Real/Extended", "Real/Continuity"]:
    with open(os.path.join(lean_dir, f + ".lean"), encoding="utf-8") as fh: load(LE, lean_defs(fh.read()))
with open(py, encoding="utf-8") as fh: src = fh.read()
ns = {"eml": st.eml, "ONE": Store.ONE}
exec(src[src.index("# ---- Theorem 1b"):src.index("x, y = var('x'), var('y')")], ns)
pairs = [("E","E",1),("L","L",1),("F","F1b",2),("B","B1b",1),("add","Wadd1b",2),("sub","Wsub1b",2),("ZERO","const_0",0),
 ("neg","Wminus1b",1),("NEG1","const_neg1b",0),("TWO","const_2_1b",0),("G","G1b",2),("mul","Wmul1b",2),("sqr","Wsqr1b",1),
 ("inv","Winv1b",1),("div","Wdiv1b",2),("half","Whalf1b",1),("avg","Wavg1b",2),("sigma","Wsigma1b",1),("pow_","Wpow1b",2),
 ("logb","Wlog1b",2),("sinh_","Wsinh1b",1),("cosh_","Wcosh1b",1),("tanh_","Wtanh1b",1),("Q","Q1b",1),("arsinh_","Warsinh1b",1),
 ("artanh_","Wartanh1b",1),("arcosh_gt1","Warcosh1b",1),("hypot_off","Whypot1b",2),("U","U1c",1),("V","V1c",2),("P","P1c",2),
 ("HALF","const_half1b",0),("sqrt_c","Wsqrt1c",1),("hypot_c","Whypot1c",2),("arcosh_c","Warcosh1c",1),("abs_c","Wabs1c",1),
 ("relu_c","Wrelu1c",1),("max_c","Wmax1c",2)]
bad = 0
for na, nb, k in pairs:
    args = [Store.X, Store.Y][:k]
    a = ns[na](*args) if k else ns[na]; b = LE.value(nb, args)
    if a != b: print("  DIFF   %-12s %-18s leaves %s / %s" % (na, nb, st.leaves[a], st.leaves[b])); bad += 1
print("Theorems 1b and 1c, Python against Lean: %d pairs, %d differ" % (len(pairs), bad))
sys.exit(1 if bad else 0)
