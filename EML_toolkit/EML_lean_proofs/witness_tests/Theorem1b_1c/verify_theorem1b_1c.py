# verify_theorem1b_1c.py - numerical test (mpmath, 2000 digits) of the witnesses of
# Theorems 1b and 1c.
# Terms are built from the leaf 1, variables and eml nodes only (hash-consed DAG),
# and evaluated node by node:  strict real semantics (Definition B.1) and
# extended real semantics (Definition C.1 / Table C.1).  Sizes: leaves, K = 2*leaves - 1.
import sys
sys.setrecursionlimit(100000)
from mpmath import mp, mpf, exp, log, sqrt, sinh, cosh, tanh, asinh, acosh, atanh, inf, isinf
mp.dps = 2000  # the real witnesses cancel hundreds of digits even for |x| <= 2

class T:
    __slots__ = ('kind', 'l', 'r', 'name', 'leaves', 'id')
    table = {}
    count = 0
    def __new__(cls, kind, l=None, r=None, name=None):
        key = (kind, id(l), id(r), name)
        t = cls.table.get(key)
        if t is None:
            t = object.__new__(cls)
            t.kind, t.l, t.r, t.name = kind, l, r, name
            t.leaves = 1 if kind != 'eml' else l.leaves + r.leaves
            T.count += 1
            t.id = T.count
            cls.table[key] = t
        return t

ONE = T('one')
def var(n): return T('var', name=n)
def eml(a, b): return T('eml', a, b)

class Undefined(Exception): pass

def eval_strict(t, env, memo):
    if t.id in memo: return memo[t.id]
    if t.kind == 'one': v = mpf(1)
    elif t.kind == 'var': v = env[t.name]
    else:
        u = eval_strict(t.l, env, memo); w = eval_strict(t.r, env, memo)
        if not (w > 0): raise Undefined()
        v = exp(u) - log(w)
    memo[t.id] = v
    return v

def eval_ext(t, env, memo):
    # Table C.1
    if t.id in memo: return memo[t.id]
    if t.kind == 'one': v = mpf(1)
    elif t.kind == 'var': v = env[t.name]
    else:
        u = eval_ext(t.l, env, memo); w = eval_ext(t.r, env, memo)
        if w < 0: raise Undefined()            # includes -inf
        if w == 0: v = inf
        elif w == inf:
            if u == inf: raise Undefined()
            v = -inf
        elif u == inf: v = inf
        elif u == -inf: v = -log(w)
        else: v = exp(u) - log(w)
    memo[t.id] = v
    return v

# ---- Theorem 1b (Lemmas B.1-B.4, B.7); F, B, G, Q are the auxiliary terms of Appendix B ----
def E(t): return eml(t, ONE)
def L(u): return eml(ONE, E(eml(ONE, u)))
def F(t, v): return eml(t, E(v))
def B(t): return L(F(t, t))
def add(x, y): return F(x, F(B(x), y))
def sub(x, y): return F(B(y), F(y, x))
ZERO = L(ONE)
def neg(x): return sub(ZERO, x)
NEG1 = neg(ONE)
TWO = add(ONE, ONE)
def G(u, y): return sub(E(add(u, y)), E(add(u, B(y))))
def mul(x, y): return sub(G(x, y), G(B(x), y))
def sqr(y): return mul(y, y)
def inv(y): return mul(y, E(neg(L(sqr(y)))))
def div(x, y): return mul(x, inv(y))
def half(x): return div(x, TWO)
def avg(x, y): return half(add(x, y))
def sigma(x): return inv(add(ONE, E(neg(x))))
def pow_(x, y): return E(mul(y, L(x)))
def logb(b, y): return div(L(y), L(b))
def sinh_(x): return half(sub(E(x), E(neg(x))))
def cosh_(x): return half(add(E(x), E(neg(x))))
def tanh_(x): return div(sinh_(x), cosh_(x))
def Q(t): return E(half(L(t)))
def arsinh_(x): return L(add(x, Q(add(ONE, sqr(x)))))
def artanh_(x): return half(sub(L(add(ONE, x)), L(sub(ONE, x))))
def arcosh_gt1(x): return L(add(x, Q(sub(sqr(x), ONE))))
def hypot_off(x, y): return Q(add(sqr(x), sqr(y)))
# ---- Theorem 1c (Lemma C.2, the sentence after it, Corollary C.3) ----
def U(x): return eml(x, x)
def V(u, r): return E(eml(L(L(u)), inv(r)))
def P(x, r): return E(F(add(x, L(r)), V(U(x), r)))
HALF = half(ONE)
def sqrt_P(x): return P(x, HALF)                                              # sentence after Lemma C.2, r = 1/2
def sqrt_c(x): return E(F(sub(x, L(TWO)), E(eml(L(L(eml(x, x))), TWO))))       # Lemma C.2, equation (C.2)
def hypot_c(x, y): return sqrt_c(add(sqr(x), sqr(y)))
def arcosh_c(x): return L(add(x, sqrt_c(sub(sqr(x), ONE))))
def abs_c(x): return sqrt_c(sqr(x))
def relu_c(x): return half(add(x, abs_c(x)))
def max_c(x, y): return half(add(add(x, y), abs_c(sub(x, y))))

x, y = var('x'), var('y')
pts1 = [mpf(s) for s in ['-1.5', '-1', '-0.5', '0', '0.5', '1', '1.5', '2']]
ptsS = [mpf(s) for s in ['-0.5', '0', '0.5', '1', '2']]            # sigma: 1 + e^{-x} must stay small
pts2 = [(a, b) for a in [mpf(s) for s in ['-2', '-0.5', '0', '0.5', '2']]
               for b in [mpf(s) for s in ['-2', '-0.5', '0', '0.5', '2']]]
worst = mpf(0); nchecks = 0; fails = []

def chk(name, term, ref, pts, dom, ev=eval_strict, vars_=('x',)):
    global worst, nchecks
    for p in pts:
        env = dict(zip(vars_, p if isinstance(p, tuple) else (p,)))
        args = tuple(env[v] for v in vars_)
        if not dom(*args):
            continue
        try:
            v = ev(term, env, {})
        except Undefined:
            fails.append((name, args, 'undefined')); continue
        r = ref(*args)
        err = abs(v - r)
        if isinf(v) or err > mpf(10) ** (-100) * (1 + abs(r)):
            fails.append((name, args, v, r))
        worst = max(worst, err); nchecks += 1
    print('  done:', name, flush=True)

always = lambda *a: True
rows = [
 ('exp', E(x), exp, always), ('ln', L(x), log, lambda a: a > 0),
 ('minus', neg(x), lambda a: -a, always), ('sqr', sqr(x), lambda a: a * a, always),
 ('inv', inv(x), lambda a: 1 / a, lambda a: a != 0), ('half', half(x), lambda a: a / 2, always),
 ('sigma', sigma(x), lambda a: 1 / (1 + exp(-a)), always),
 ('sinh', sinh_(x), sinh, always), ('cosh', cosh_(x), cosh, always), ('tanh', tanh_(x), tanh, always),
 ('arsinh', arsinh_(x), asinh, always), ('artanh', artanh_(x), atanh, lambda a: -1 < a < 1),
]
for n_, t_, r_, d_ in rows:
    chk(n_, t_, r_, ptsS if n_ == 'sigma' else pts1 + [mpf('0.9'), mpf('-0.9')], d_)
rows2 = [
 ('plus', add(x, y), lambda a, b: a + b, always), ('subtract', sub(x, y), lambda a, b: a - b, always),
 ('times', mul(x, y), lambda a, b: a * b, always), ('avg', avg(x, y), lambda a, b: (a + b) / 2, always),
 ('divide', div(x, y), lambda a, b: a / b, lambda a, b: b != 0),
 ('power', pow_(x, y), lambda a, b: a ** b, lambda a, b: a > 0),
 ('logarithm', logb(x, y), lambda a, b: log(b) / log(a), lambda a, b: a > 0 and a != 1 and b > 0),
]
for n_, t_, r_, d_ in rows2: chk(n_, t_, r_, pts2, d_, vars_=('x', 'y'))
consts = [('1', ONE, mpf(1)), ('e', E(ONE), exp(1)), ('0', ZERO, mpf(0)), ('-1', NEG1, mpf(-1)), ('2', TWO, mpf(2)), ('1/2', HALF, mpf(1) / 2)]
for n_, t_, r_ in consts:
    v = eval_strict(t_, {}, {}); worst = max(worst, abs(v - r_)); nchecks += 1
    if abs(v - r_) > mpf(10) ** -100: fails.append((n_, v))
# restricted strict witnesses
chk('sqrt x>0 (strict)', Q(x), sqrt, pts1, lambda a: a > 0)
chk('arcosh x>1 (strict)', arcosh_gt1(x), acosh, pts1 + [mpf('1.01'), mpf(3)], lambda a: a > 1)
chk('hypot off origin (strict)', hypot_off(x, y), lambda a, b: sqrt(a * a + b * b), pts2, lambda a, b: (a, b) != (0, 0), vars_=('x', 'y'))
# extended real: the 26 witnesses; the three new ones with endpoints
chk('sqrt x>=0 (ext)', sqrt_c(x), sqrt, pts1, lambda a: a >= 0, ev=eval_ext)
chk('sqrt P(x,1/2) (ext)', sqrt_P(x), sqrt, pts1, lambda a: a >= 0, ev=eval_ext)
chk('arcosh x>=1 (ext)', arcosh_c(x), acosh, pts1 + [mpf('1.01'), mpf(3)], lambda a: a >= 1, ev=eval_ext)
chk('hypot (ext)', hypot_c(x, y), lambda a, b: sqrt(a * a + b * b), pts2, always, ev=eval_ext, vars_=('x', 'y'))
chk('abs (ext)', abs_c(x), abs, pts1, always, ev=eval_ext)
chk('relu (ext)', relu_c(x), lambda a: max(a, 0), pts1, always, ev=eval_ext)
chk('max (ext)', max_c(x, y), lambda a, b: max(a, b), pts2, lambda a, b: abs(a - b) <= 2.5, ev=eval_ext, vars_=('x', 'y'))
for n_, t_, r_, d_ in rows: chk(n_ + ' (ext)', t_, r_, ptsS if n_ == 'sigma' else pts1, d_, ev=eval_ext)
for n_, t_, r_, d_ in rows2: chk(n_ + ' (ext)', t_, r_, pts2, d_, ev=eval_ext, vars_=('x', 'y'))

# expected failures of the strict semantics at the endpoints (obstructions)
def undefined_strict(term, env):
    try: eval_strict(term, env, {}); return False
    except Undefined: return True
print('strict: Q(0) undefined      :', undefined_strict(Q(x), {'x': mpf(0)}))
print('strict: sqrt_c(0) undefined :', undefined_strict(sqrt_c(x), {'x': mpf(0)}))
chk('sqrt (C.2) on x>0 (strict)', sqrt_c(x), sqrt, pts1, lambda a: a > 0)
print('strict: inv(0) undefined    :', undefined_strict(inv(x), {'x': mpf(0)}))
print('strict: L(0), L(-1) undefined:', undefined_strict(L(x), {'x': mpf(0)}), undefined_strict(L(x), {'x': mpf(-1)}))
print('ext: L(0) =', eval_ext(L(x), {'x': mpf(0)}, {}), '  eml(1,0) =', eval_ext(eml(ONE, ZERO), {}, {}), '  eml(1,+inf) =', eval_ext(eml(ONE, eml(ONE, ZERO)), {}, {}))
print('checks:', nchecks, ' worst abs error: %s' % mp.nstr(worst, 5), ' failures:', len(fails))
for f in fails[:20]: print('  FAIL', f)

print()
print('%-28s %40s %42s' % ('witness', 'leaves', 'K = 2*leaves - 1'))
size_rows = [('e = E(1)', E(ONE)), ('0 = L(1)', ZERO), ('-1', NEG1), ('2', TWO), ('1/2', HALF),
 ('exp x', E(x)), ('ln x', L(x)), ('x + y', add(x, y)), ('x - y', sub(x, y)), ('-x', neg(x)),
 ('x y', mul(x, y)), ('x^2', sqr(x)), ('1/x', inv(x)), ('x / y', div(x, y)), ('x/2', half(x)),
 ('avg', avg(x, y)), ('sigma', sigma(x)), ('x^y', pow_(x, y)), ('log_x y', logb(x, y)),
 ('sinh', sinh_(x)), ('cosh', cosh_(x)), ('tanh', tanh_(x)), ('arsinh', arsinh_(x)), ('artanh', artanh_(x)),
 ('sqrt x>0 : Q(x)', Q(x)), ('hypot off 0 (1b)', hypot_off(x, y)), ('arcosh x>1 (1b)', arcosh_gt1(x)),
 ('sqrt (1c), eq. (C.2)', sqrt_c(x)), ('P(x,1/2), after Lemma C.2', sqrt_P(x)), ('hypot (1c)', hypot_c(x, y)), ('arcosh (1c)', arcosh_c(x)),
 ('|x| (1c)', abs_c(x)), ('+inf = eml(1,0)', eml(ONE, ZERO)), ('-inf = eml(1,+inf)', eml(ONE, eml(ONE, ZERO)))]
def fmt(n):
    s = str(n)
    return s if len(s) <= 15 else '%s.%se+%d' % (s[0], s[1:4], len(s) - 1)
for n_, t_ in size_rows:
    print('%-28s %40s %42s' % (n_, fmt(t_.leaves), fmt(2 * t_.leaves - 1)))
print('distinct DAG nodes built in total:', T.count)
