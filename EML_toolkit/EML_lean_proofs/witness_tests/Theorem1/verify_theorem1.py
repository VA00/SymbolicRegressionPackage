# verify_theorem1.py - numerical test (mpmath, 150 digits) of the witnesses of
# Theorem 1 (Lemmas 1-24).
#
# Everything is built from eml(z,w) = exp(z) - Log(w) and the constant 1, with no
# shortcut.  A value that is a real number has the type mpf, so in
# -i*pi := eml(1, N(E(e))) the argument of the logarithm is a negative real
# number, and no rounding decides the side of the branch cut.
#
# Points at which an intermediate value lies exactly on the branch cut
# (H(0), sqrt 0, hypot(0,0), arcosh 1, arcsin(+-1), arccos(+-1)) are not tested
# here: rounding would decide the side of the cut.  They are tested in exact
# arithmetic (verify_theorem1.wl).
#
# The reference values are the functions of mpmath.
#
# Run: python verify_theorem1.py        (expect: failures 0)
from mpmath import (mp, mpf, mpc, exp, log, sqrt, pi, sinh, cosh, tanh, asinh, acosh, atanh,
                    cos, sin, tan, asin, acos, atan)

mp.dps = 150
TOL = mpf(10) ** (-60)

nodes = 0


def eml(z, w):
    global nodes
    nodes += 1
    assert w != 0, "Log 0"
    return exp(z) - log(w)


one = mpf(1)
# Step 1: Lemma 1, Corollary 2, Lemmas 3-4
E = lambda z: eml(z, one)
e = eml(one, one)
L = lambda z: eml(one, E(eml(one, z)))
S = lambda u, v: eml(L(u), E(v))
# Step 2: Lemma 5
N = lambda z: S(one, S(e, S(S(e, one), z)))
# Step 3: Lemmas 6-8
zero = L(one)
neg = N(one)
two = S(one, neg)
three = S(two, neg)
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
# Step 4: Lemmas 9-16
U = lambda x: eml(L(img), E(N(x)))
Wminus = lambda a, b: S(S(U(a), b), img)
Wplus = lambda a, b: S(S(U(a), N(b)), img)
Wsqr = lambda x: Wminus(M(U(x), S(U(x), twoi)), one)
Whalf = lambda x: S(D(U(x), two), D(img, two))
Wx = lambda x, y: Whalf(Whalf(Wminus(Wsqr(Wplus(x, y)), Wsqr(Wminus(x, y)))))
Winv = lambda x: Wx(x, E(N(L(Wsqr(x)))))
Wdiv = lambda x, y: Wx(x, Winv(y))
Wavg = lambda x, y: Whalf(Wplus(x, y))
# Step 5: Lemmas 17-20
X = lambda x: S(M(U(x), img), neg)
T = lambda z: eml(L(ipi), E(eml(L(ipi), z)))
tau = lambda x: Wdiv(N(x), Wplus(one, Wsqr(x)))
w = lambda x: M(U(tau(x)), img)


def H(x):
    v = w(x)
    return S(D(T(v), twoipi), D(L(v), twoipi))


Wsqrt = lambda x: Wminus(E(Whalf(L(Wplus(x, H(N(x)))))), H(N(x)))
Whypot = lambda x, y: Wsqrt(Wplus(Wsqr(x), Wsqr(y)))
# Step 6: Lemma 21
Wsigma = lambda x: Winv(Wplus(one, E(N(x))))
Wpow = lambda x, y: E(Wx(y, L(x)))
Wlog = lambda x, y: Wdiv(L(y), L(x))
Wsinh = lambda x: Whalf(Wminus(E(x), E(N(x))))
Wcosh = lambda x: Wavg(E(x), E(N(x)))
Wtanh = lambda x: Wdiv(Wsinh(x), Wcosh(x))
Warsinh = lambda x: L(Wplus(x, Wsqrt(Wplus(Wsqr(x), one))))
Warcosh = lambda x: L(Wplus(x, Wsqrt(Wminus(Wsqr(x), one))))
Wartanh = lambda x: Whalf(Wminus(L(Wplus(one, x)), L(Wminus(one, x))))
# Step 7: Lemmas 22-24
A = lambda u, v: S(S(S(three, N(u)), N(v)), three)
Wcos = lambda x: Whalf(A(E(X(x)), E(X(N(x)))))
Wsin = lambda x: Wcos(Wminus(x, PI2))
Wtan = lambda x: Wdiv(Wsin(x), Wcos(x))
Warcsin = lambda x: Wminus(D(A(L(A(Wsqrt(Wminus(one, Wsqr(x))), X(x))), twoi), img), two)
Warccos = lambda x: Wminus(PI2, Warcsin(x))
Warctan = lambda x: Warcsin(Wdiv(x, Wsqrt(Wplus(Wsqr(x), one))))

# ---- reference: the functions of mpmath ----
assert isinstance(N(E(e)), mpf)        # -e^e is a real number: Log is applied to it exactly on the cut

I = mpc(0, 1)
fails, checks, worst = 0, 0, mpf(0)
perfun = {}


def check(name, got, want, where):
    global fails, checks, worst
    checks += 1
    perfun[name] = perfun.get(name, 0) + 1
    d = abs(got - want) / max(mpf(1), abs(want))
    worst = max(worst, d)
    if not d < TOL:
        fails += 1
        print("FAIL", name, where, "got", mp.nstr(got, 20), "want", mp.nstr(want, 20))


def reals(*xs):
    return [mpf(x) for x in xs]


# ---- constants ----
for got, want, nm in [(e, exp(1), "e"), (zero, 0, "0"), (neg, -1, "-1"), (two, 2, "2"), (three, 3, "3"),
                      (half, mpf(1) / 2, "1/2"), (mipi, -I * pi, "-i pi"), (img, I, "i"), (PI, pi, "pi"),
                      (PI2, pi / 2, "pi/2"), (twoi, 2 * I, "2i"), (ipi, I * pi, "i pi"), (twoipi, 2 * I * pi, "2 pi i")]:
    check("const", got, mpc(want), nm)

# ---- auxiliary terms: U(x) = x + i, X(x) = i x, the unit step H(x) for x != 0 ----
for x in reals("-50", "-7.3", "-1", "-0.5", "-1e-6", "1e-6", "0.3", "1", "2", "3.14159", "10", "1000"):
    check("U", U(x), x + I, x)
    check("X", X(x), I * x, x)
    check("H", H(x), mpf(1) if x > 0 else mpf(0), x)

# ---- unary primitives ----
ref = {
    "exp": exp, "ln": log, "minus": lambda x: -x, "sqr": lambda x: x * x, "half": lambda x: x / 2,
    "inv": lambda x: 1 / x, "sqrt": sqrt, "sigma": lambda x: 1 / (1 + exp(-x)),
    "sinh": sinh, "cosh": cosh, "tanh": tanh, "arsinh": asinh, "arcosh": acosh, "artanh": atanh,
    "cos": cos, "sin": sin, "tan": tan, "arcsin": asin, "arccos": acos, "arctan": atan,
}
wit = {
    "exp": E, "ln": L, "minus": N, "sqr": Wsqr, "half": Whalf, "inv": Winv, "sqrt": Wsqrt, "sigma": Wsigma,
    "sinh": Wsinh, "cosh": Wcosh, "tanh": Wtanh, "arsinh": Warsinh, "arcosh": Warcosh, "artanh": Wartanh,
    "cos": Wcos, "sin": Wsin, "tan": Wtan, "arcsin": Warcsin, "arccos": Warccos, "arctan": Warctan,
}
xs = reals("-30", "-3.7", "-1", "-0.2", "0", "0.4", "1", "2.5", "7", "40")
pos = reals("1e-6", "0.25", "1", "2", "9", "123.4")
unit = reals("-0.999999", "-0.8414709848", "-0.5", "-1e-9", "0", "1e-9", "0.3", "0.7071", "0.999999")
points = {
    "exp": xs, "minus": xs, "sqr": xs, "half": xs, "inv": [x for x in xs if x != 0], "ln": pos, "sqrt": pos,
    "sigma": xs, "sinh": xs, "cosh": xs, "tanh": xs, "arsinh": xs, "cos": xs, "sin": xs, "tan": xs,
    "arcosh": reals("1.0000001", "1.5", "3", "100"),
    "artanh": reals("-0.999", "-0.5", "0", "0.3", "0.999"),
    "arcsin": unit, "arccos": unit,
    "arctan": reals("-1e4", "-25", "-1", "-0.1", "0", "0.1", "1", "3", "25", "1e4"),
}
for name in wit:
    for x in points[name]:
        check(name, wit[name](x), ref[name](x), x)

# ---- binary primitives ----
ref2 = {
    "+": lambda x, y: x + y, "-": lambda x, y: x - y, "x": lambda x, y: x * y, "/": lambda x, y: x / y,
    "avg": lambda x, y: (x + y) / 2, "hypot": lambda x, y: sqrt(x * x + y * y),
    "pow": lambda x, y: x ** y, "log": lambda x, y: log(y) / log(x),
}
wit2 = {"+": Wplus, "-": Wminus, "x": Wx, "/": Wdiv, "avg": Wavg, "hypot": Whypot, "pow": Wpow, "log": Wlog}
g = reals("-3.7", "-1", "0", "0.4", "2.5", "7")
grid = [(x, y) for x in g for y in g]
points2 = {
    "+": grid, "-": grid, "x": grid, "avg": grid,
    "/": [(x, y) for x, y in grid if y != 0],
    "hypot": [(x, y) for x, y in grid if (x, y) != (0, 0)],
    "pow": [(mpf(x), mpf(y)) for x, y in [("0.5", "-3"), ("2", "0"), ("2", "10"), ("7.5", "0.5"), ("1e-3", "2.5")]],
    "log": [(mpf(x), mpf(y)) for x, y in [("0.5", "3"), ("2", "8"), ("10", "1"), ("7.5", "0.01")]],   # x is the base
}
for name in wit2:
    for x, y in points2[name]:
        check(name, wit2[name](x, y), ref2[name](x, y), (x, y))

print("eml nodes evaluated:", nodes)
for k in sorted(perfun):
    print("  %-10s %6d" % (k, perfun[k]))
print("checks:", checks, " failures:", fails, " worst relative error:", mp.nstr(worst, 3))
