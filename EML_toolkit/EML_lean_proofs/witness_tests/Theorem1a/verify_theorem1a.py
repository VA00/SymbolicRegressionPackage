# verify_theorem1a.py - numerical test (mpmath, 120 digits) of the witnesses of Theorem 1a
# (Appendix A, Table A.3).
#
# Everything is built from EML(z,w) = exp(z) - Log(w) on C + {-inf, +inf} and the
# constant 1, with Table A.1:
#     exp(-inf) = 0, exp(+inf) = +inf, Log 0 = -inf, Log(+-inf) = +inf,
#     inf - inf undefined.
# The theorem is about exact numbers.  Exact arithmetic is emulated here with
# 120 digits and three rules that recognise an exact zero part:
#   exp z   : a part of the value below 1e-70 of its modulus is 0
#             (e^{i pi} is real, e^{i pi/2} is imaginary);
#   Log w   : a part of the value below 1e-70 in absolute value is 0 (|w| = 1);
#   a - b   : a part of the difference below 1e-70 of the larger of the two
#             operand parts is 0 (cancellation).
# No rule is absolute for exp and for the difference, because true values such
# as exp(-e^25) do occur: the subtraction term takes the exponential of a value.
# With inputs of a few digits no true non-zero part comes near these thresholds,
# so exact zeros and exact positions on a branch cut are recognised.
#
# The reference values are the formulas of Table A.2 (Log, sqrt of mpmath on
# exact inputs), not library inverse functions, whose conventions on the cuts
# differ between libraries.  verify_theorem1a.wl compares with Mathematica.
#
# Run: python verify_theorem1a.py        (expect: failures 0)
#      python verify_theorem1a.py 1000   (more points)
import random, sys
from mpmath import mp, mpf, mpc, exp, log, pi

mp.dps = 120
TOL = mpf(10) ** (-70)
PINF, MINF = "+inf", "-inf"


class Undefined(Exception):
    pass


def snap_rel(x):
    x = mpc(x)
    re, im = x.real, x.imag
    m = abs(x)
    if abs(re) < TOL * m:
        re = mpf(0)
    if abs(im) < TOL * m:
        im = mpf(0)
    return mpc(re, im)


def snap_abs(x):
    x = mpc(x)
    re, im = x.real, x.imag
    if abs(re) < TOL:
        re = mpf(0)
    if abs(im) < TOL:
        im = mpf(0)
    return mpc(re, im)


def diff(a, b):
    d = a - b
    re, im = d.real, d.imag
    if abs(re) < TOL * max(abs(a.real), abs(b.real)):
        re = mpf(0)
    if abs(im) < TOL * max(abs(a.imag), abs(b.imag)):
        im = mpf(0)
    return mpc(re, im)


nodes = 0


def eml(z, w):
    global nodes
    nodes += 1
    # exp z
    if z is PINF:
        a = PINF
    elif z is MINF:
        a = mpc(0)
    else:
        a = snap_rel(exp(z))
    # Log w
    if w is PINF or w is MINF:
        b = PINF
    elif w == 0:
        b = MINF
    else:
        b = snap_abs(log(w))
    # a - b
    if a is PINF:
        if b is PINF:
            raise Undefined()
        return PINF
    if b is PINF:
        return MINF
    if b is MINF:
        return PINF
    return diff(a, b)


one = mpc(1)
# Section 2 (Theorem 1): Lemma 1, Corollary 2, Lemmas 3-8
E = lambda z: eml(z, one)
e = eml(one, one)
L = lambda z: eml(one, E(eml(one, z)))
S = lambda u, v: eml(L(u), E(v))
N = lambda z: S(one, S(e, S(S(e, one), z)))
zero = L(one)
neg = N(one)
two = S(one, neg)
half = E(N(L(two)))
M = lambda u, v: E(S(L(u), N(L(v))))
D = lambda u, v: E(S(L(u), L(v)))
mipi = eml(one, N(E(e)))
img = E(N(D(mipi, two)))
PI = M(img, mipi)
PI2 = D(PI, two)
twoi = M(two, img)
ipi = M(img, PI)
twoipi = M(two, ipi)
# Appendix A, Step 1: Lemmas A.1, A.3, A.4
pinf = eml(one, zero)
minf = eml(one, pinf)
Winv = lambda z: E(eml(minf, z))
Wx = lambda z, w: Winv(E(eml(L(eml(minf, z)), w)))
# Step 2: Lemma A.5, Corollary A.6
mimg = Wx(neg, img)
Wminus = lambda z, w: Wx(mimg, S(Wx(img, S(z, w)), Wx(img, S(w, w))))
Wneg = lambda z: Wx(neg, z)
Wplus = lambda z, w: Wminus(z, Wneg(w))
Wsqr = lambda z: Wx(z, z)
Whalf = lambda z: Wx(half, z)
Wavg = lambda z, w: Whalf(Wplus(z, w))
Wdiv = lambda z, w: Wx(z, Winv(w))
# Step 3: Lemma A.7, Corollary A.8, Lemmas A.9, A.10
Wpow = lambda z, w: Winv(E(Wx(w, eml(minf, z))))
Wsqrt = lambda z: Wpow(z, half)
Whypot = lambda z, w: Wsqrt(Wplus(Wsqr(z), Wsqr(w)))
Wln = lambda z: Wneg(eml(minf, z))
Wlog = lambda b, z: Wdiv(Wln(z), Wln(b))
# Step 4: Lemma A.11
Wsigma = lambda z: Winv(Wplus(one, E(Wneg(z))))
Wsinh = lambda z: Wdiv(Wminus(E(z), E(Wneg(z))), two)
Wcosh = lambda z: Wdiv(Wplus(E(z), E(Wneg(z))), two)
Wcos = lambda z: Wdiv(Wplus(E(Wx(img, z)), E(Wx(mimg, z))), two)
Wsin = lambda z: Wcos(Wminus(z, PI2))
Wtan = lambda z: Wdiv(Wsin(z), Wcos(z))
Wtanh = lambda z: Wdiv(Wsinh(z), Wcosh(z))
Warcsin = lambda z: Wx(mimg, Wln(Wplus(Wx(img, z), Wsqrt(Wminus(one, Wsqr(z))))))
Warccos = lambda z: Wminus(PI2, Warcsin(z))
Warsinh = lambda z: Wln(Wplus(z, Wsqrt(Wplus(one, Wsqr(z)))))
Warcosh = lambda z: Wln(Wplus(z, Wx(Wsqrt(Wminus(z, one)), Wsqrt(Wplus(z, one)))))
Warctan = lambda z: Wdiv(Wminus(Wln(Wplus(one, Wx(img, z))), Wln(Wminus(one, Wx(img, z)))), twoi)
Wartanh = lambda z: Wdiv(Wminus(Wln(Wplus(one, z)), Wln(Wminus(one, z))), two)
# Section A.4: Lemma A.12 and the examples
J = lambda z: Wx(Winv(z), z)
WinvLean = lambda z: Wx(Winv(z), J(z))
WlnLean = lambda z: Wx(Wln(z), J(z))
WlnWolfram = lambda z: eml(L(WlnLean(z)), Winv(J(z)))


# ---- reference: principal expressions (exact inputs; mpmath log/sqrt are closed
# on the cut from above: log(-r) = ln r + i pi, sqrt(-r) = i sqrt r) ----
I = mpc(0, 1)
Log = lambda z: log(mpc(z))
Sqrt = lambda z: mpc(0) if z == 0 else exp(Log(z) / 2)
ref = {
    "exp": exp, "minus": lambda z: -z, "half": lambda z: z / 2, "sqr": lambda z: z * z,
    "inv": lambda z: 1 / z, "sqrt": Sqrt, "ln": Log, "sigma": lambda z: 1 / (1 + exp(-z)),
    "sinh": lambda z: (exp(z) - exp(-z)) / 2, "cosh": lambda z: (exp(z) + exp(-z)) / 2,
    "tanh": lambda z: (exp(z) - exp(-z)) / (exp(z) + exp(-z)),
    "sin": lambda z: (exp(I * z) - exp(-I * z)) / (2 * I), "cos": lambda z: (exp(I * z) + exp(-I * z)) / 2,
    "tan": lambda z: (exp(I * z) - exp(-I * z)) / (I * (exp(I * z) + exp(-I * z))),
    "arcsin": lambda z: -I * Log(I * z + Sqrt(1 - z * z)),
    "arccos": lambda z: pi / 2 + I * Log(I * z + Sqrt(1 - z * z)),
    "arctan": lambda z: (Log(1 + I * z) - Log(1 - I * z)) / (2 * I),
    "arsinh": lambda z: Log(z + Sqrt(1 + z * z)),
    "arcosh": lambda z: Log(z + Sqrt(z - 1) * Sqrt(z + 1)),
    "artanh": lambda z: (Log(1 + z) - Log(1 - z)) / 2,
}
wit = {
    "exp": E, "minus": Wneg, "half": Whalf, "sqr": Wsqr, "inv": Winv, "sqrt": Wsqrt, "ln": Wln,
    "sigma": Wsigma, "sinh": Wsinh, "cosh": Wcosh, "tanh": Wtanh, "sin": Wsin, "cos": Wcos, "tan": Wtan,
    "arcsin": Warcsin, "arccos": Warccos, "arctan": Warctan, "arsinh": Warsinh, "arcosh": Warcosh,
    "artanh": Wartanh,
}
ref2 = {
    "+": lambda z, w: z + w, "-": lambda z, w: z - w, "x": lambda z, w: z * w,
    "/": lambda z, w: z / w, "avg": lambda z, w: (z + w) / 2,
    "hypot": lambda z, w: Sqrt(z * z + w * w),
    "pow": lambda z, w: mpc(0) if z == 0 else exp(w * Log(z)),
    "log": lambda b, z: Log(z) / Log(b),
}
wit2 = {"+": Wplus, "-": Wminus, "x": Wx, "/": Wdiv, "avg": Wavg, "hypot": Whypot, "pow": Wpow, "log": Wlog}

fails, checks, worst = 0, 0, mpf(0)
perfun = {}


def check(name, got, want, where):
    global fails, checks, worst
    checks += 1
    perfun[name] = perfun.get(name, 0) + 1
    if isinstance(got, str) or isinstance(want, str):
        ok = got is want
    else:
        d = abs(got - want) / max(mpf(1), abs(want))
        worst = max(worst, d)
        ok = d < mpf(10) ** (-60)
    if not ok:
        fails += 1
        if fails <= 40:
            print("FAIL", name, where, "got", got if isinstance(got, str) else mp.nstr(got, 20),
                  "want", want if isinstance(want, str) else mp.nstr(want, 20))


def run1(name, z):
    try:
        got = wit[name](z)
    except Undefined:
        got = "undefined"
    check(name, got, ref[name](z), z)


def run2(name, z, w):
    try:
        got = wit2[name](z, w)
    except Undefined:
        got = "undefined"
    check(name, got, ref2[name](z, w), (z, w))


# ---- constants ----
for got, want, nm in [(e, exp(1), "e"), (zero, 0, "0"), (neg, -1, "-1"), (two, 2, "2"), (half, mpf(1) / 2, "1/2"),
                      (mipi, -I * pi, "-i pi"), (img, I, "i"), (PI, pi, "pi"), (PI2, pi / 2, "pi/2"), (twoi, 2 * I, "2i"),
                      (ipi, I * pi, "i pi"), (twoipi, 2 * I * pi, "2 pi i"), (mimg, -I, "-i")]:
    check("const", got, mpc(want), nm)
assert pinf is PINF and minf is MINF

# ---- points ----
random.seed(20261119)
rat = lambda: mpf(random.randint(-40, 40)) / random.choice([1, 2, 3, 4, 5, 8])


def generic():
    return mpc(rat(), rat())


def special(kind):
    r = abs(rat()) + mpf(1) / 7          # a positive rational
    k = random.randint(-3, 3)
    return {
        "negreal": mpc(-r, 0), "posreal": mpc(r, 0), "imag+": mpc(0, r), "imag-": mpc(0, -r),
        "big+": mpc(1 + r, 0), "big-": mpc(-1 - r, 0), "ibig+": mpc(0, 1 + r), "ibig-": mpc(0, -1 - r),
        "unit": mpc(mpf(random.randint(-99, 99)) / 100, 0),
        "oddpi": mpc(rat(), (2 * k + 1) * pi), "evenpi": mpc(rat(), 2 * k * pi),
    }[kind]


KINDS = ["negreal", "posreal", "imag+", "imag-", "big+", "big-", "ibig+", "ibig-", "unit", "oddpi", "evenpi"]
fixed = [mpc(x) for x in [0, 1, -1, 2, -2, mpf(1) / 2, -mpf(1) / 2, I, -I, 2 * I, -2 * I, I / 2, -I / 2, 1 + I, -1 - I,
                          pi, -pi, pi / 2, I * pi, -I * pi, 3 * I * pi, 2 + I * pi, exp(1)]]

NGEN = int(sys.argv[1]) if len(sys.argv) > 1 else 300
pts = fixed + [generic() for _ in range(NGEN)] + [special(k) for k in KINDS for _ in range(NGEN // 10)]


def excluded(name, z):
    if name == "inv":
        return z == 0
    if name == "ln":
        return z == 0
    if name == "tan":
        c = (exp(I * z) + exp(-I * z)) / 2
        return abs(c) < mpf(10) ** (-60)
    if name == "tanh":
        c = (exp(z) + exp(-z)) / 2
        return abs(c) < mpf(10) ** (-60)
    if name == "sigma":
        return abs(1 + exp(-z)) < mpf(10) ** (-60)
    if name == "arctan":
        return z == I or z == -I
    if name == "artanh":
        return z == 1 or z == -1
    return False


for name in wit:
    for z in pts:
        # keep exponentials moderate so that the snapping threshold stays far from true values
        if name in ("exp", "sigma", "sinh", "cosh", "tanh") and abs(z.real) > 25:
            continue
        if name in ("sin", "cos", "tan") and abs(z.imag) > 25:
            continue
        if not excluded(name, z):
            run1(name, z)

pairs = [(z, w) for z in fixed[:17] for w in fixed[:17]]
pairs += [(generic(), generic()) for _ in range(NGEN)]
pairs += [(generic(), special("oddpi")) for _ in range(NGEN // 5)]
pairs += [(special("negreal"), generic()) for _ in range(NGEN // 5)]
pairs += [(special("negreal"), special("negreal")) for _ in range(NGEN // 5)]
for z, w in pairs:
    for name in ("+", "-", "x", "avg", "hypot"):
        run2(name, z, w)
    if w != 0:
        run2("/", z, w)
    small = abs(w) < 6 and abs(z) < 12
    if small and (z != 0 or w.real > 0):
        run2("pow", z, w)
    if z != 0 and z != 1 and w != 0:
        run2("log", z, w)          # here z is the base, w the argument

# ---- Section A.4: the test for zero and the three examples ----
for z in [mpc(0)] + fixed[1:12] + [generic() for _ in range(40)]:
    check("J", J(z), mpc(0) if z == 0 else mpc(1), z)
    check("inv, Lean", WinvLean(z), mpc(0) if z == 0 else 1 / z, z)
    check("Log, Lean", WlnLean(z), mpc(0) if z == 0 else Log(z), z)
    check("Log, Wolfram", WlnWolfram(z), MINF if z == 0 else Log(z), z)

print("eml nodes evaluated:", nodes)
for k in sorted(perfun):
    print("  %-10s %6d" % (k, perfun[k]))
print("checks:", checks, " failures:", fails, " worst relative error:", mp.nstr(worst, 3))
