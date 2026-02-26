# eml_compiler_v4.py
# Standalone EML compiler CLI (default mode: compile a Wolfram-style expression)
import os, argparse, re, sys
from sympy import (
    sympify, Symbol, Integer, Rational, Float, E, I, pi,
    Add, Mul, Pow, exp, log, sin, cos, tan, cot, sec, csc,
    asin, acos, atan, sinh, cosh, tanh, asinh, acosh, atanh,
    sqrt, Abs
)

# =========================
# EML string emit primitives
# =========================
def EML(a, b):        return f"EML[{a},{b}]"
def eml_exp(z):       return EML(z, "1")                                   # Exp[z]
def eml_log(z):       return EML("1", eml_exp(EML("1", z)))                # Log[z]
def eml_zero():       return eml_log("1")                                  # 0 = Log[1]
def eml_sub(a, b):    return EML(eml_log(a), eml_exp(b))                   # a - b
def eml_neg(z):       return eml_sub(eml_zero(), z)                        # -z
def eml_add(a, b):    return eml_sub(a, eml_neg(b))                        # a + b
def eml_inv(z):       return eml_exp(eml_neg(eml_log(z)))                  # 1/z
def eml_mul(a, b):    return eml_exp(eml_add(eml_log(a), eml_log(b)))      # a*b
def eml_div(a, b):    return eml_mul(a, eml_inv(b))                        # a/b
def eml_pow(a, b):    return eml_exp(eml_mul(b, eml_log(a)))               # a^b

def eml_one(): return "1"
def eml_two(): return eml_add("1", "1")

def eml_int(n:int):
    if n == 1: return "1"
    if n == 0: return eml_zero()
    if n < 0:  return eml_neg(eml_int(-n))
    acc, term, k = None, "1", n
    while k > 0:
        if k & 1: acc = term if acc is None else eml_add(acc, term)
        term = eml_add(term, term)
        k >>= 1
    return acc

def eml_rational(p:int, q:int):
    if q == 1: return eml_int(p)
    num = eml_int(abs(p)); den = eml_int(q)
    val = eml_mul(num, eml_inv(den))
    return val if p >= 0 else eml_neg(val)

# ---- canonical EML for special constants ----
def eml_const_E():
    return eml_exp("1")  # e = Exp[1]

def eml_const_I():
    #return eml_pow(eml_neg("1"), eml_rational(1, 2))  # I = (-1)^(1/2)
    #return eml_exp(eml_mul(eml_log(eml_neg("1")),eml_rational(1,2)))  # I = Exp[Log[-1]*(1/2)]
    #print(eml_neg("1"))
    #print(eml_log(eml_neg("1")))
    return eml_exp(eml_div(eml_log("-1"),"2"))  # I = Exp[Log[-1]/2]

def eml_const_Pi():
    i_eml = eml_const_I()
    minus_i = eml_neg(i_eml)
    log_minus1 = eml_log(eml_neg("1"))
    return eml_mul(minus_i, log_minus1)  # Pi = -I*Log[-1]

def eml_const_GoldenRatio():
    # φ = (1 + sqrt(5))/2 = (1 + 5^(1/2)) / 2
    sqrt5 = eml_pow(eml_int(5), eml_rational(1, 2))
    num   = eml_add("1", sqrt5)
    return eml_div(num, eml_int(2))

# =========================
# Helpers for sympify locals
# =========================
def SEC(x):  return 1/cos(x)
def CSC(x):  return 1/sin(x)
def COT(x):  return cos(x)/sin(x)
def ASEC(x): return acos(1/x)
def ACSC(x): return asin(1/x)
def ACOT(x): return atan(1/x)

def ASIN_LOG(z):   return -I*log(I*z + sqrt(1 - z**2))
def ACOS_LOG(z):   return -I*log(z + sqrt(z - 1)*sqrt(z + 1))
def ATAN_LOG(z):   return (I/2)*log((I + z)/(I - z))
def ASINH_LOG(z):  return log(z + sqrt(z**2 + 1))
def ACOSH_LOG(z):  return log(z + sqrt(z + 1)*sqrt(z - 1))
def ATANH_LOG(z):  return Rational(1,2)*log((1 + z)/(1 - z))

def HALF(z):  return z/2
def MINUS(z): return -z
def INV(z):   return 1/z
def SQR(z):   return z**2
def AVG(a,b): return (a + b)/2
def HYPOT(a,b): return sqrt(a**2 + b**2)
def LOGISTIC_SIGMOID(z): return 1/(1 + exp(-z))

def LOG_DISPATCH(*args):
    if len(args) == 1: return log(args[0])
    if len(args) == 2:
        b, z = args
        return log(z)/log(b)
    raise TypeError("Log expects 1 or 2 arguments")

def PLUS(a,b):     return a + b
def TIMES(a,b):    return a * b
def SUBTRACT(a,b): return a - b
def DIVIDE(a,b):   return a / b
def POWER(a,b):    return a ** b

from sympy import GoldenRatio as _SymGolden

LOCALS = {
    "exp": exp, "log": log, "ln": log,
    "Exp": exp, "Log": LOG_DISPATCH, "Ln": log,

    "sin": sin, "cos": cos, "tan": tan, "cot": cot, "sec": sec, "csc": csc,
    "Sin": sin, "Cos": cos, "Tan": tan, "Cot": COT, "Sec": SEC, "Csc": CSC,

    "sinh": sinh, "cosh": cosh, "tanh": tanh,
    "Sinh": sinh, "Cosh": cosh, "Tanh": tanh,

    "asin": ASIN_LOG, "acos": ACOS_LOG, "atan": ATAN_LOG,
    "asinh": ASINH_LOG, "acosh": ACOSH_LOG, "atanh": ATANH_LOG,
    "ArcSin": ASIN_LOG, "ArcCos": ACOS_LOG, "ArcTan": ATAN_LOG,
    "ArcSinh": ASINH_LOG, "ArcCosh": ACOSH_LOG, "ArcTanh": ATANH_LOG,

    "asec": ASEC, "acsc": ACSC, "acot": ACOT,
    "ArcSec": ASEC, "ArcCsc": ACSC, "ArcCot": ACOT,

    "Half": HALF, "Minus": MINUS, "Inv": INV, "Sqrt": sqrt, "Sqr": SQR,
    "Avg": AVG, "Hypot": HYPOT, "LogisticSigmoid": LOGISTIC_SIGMOID,

    "Plus": PLUS, "Times": TIMES, "Subtract": SUBTRACT,
    "Divide": DIVIDE, "Power": POWER,

    "sqrt": sqrt, "Sqrt": sqrt, "Abs": Abs, "abs": Abs,

    "E": E, "I": I, "Pi": pi, "pi": pi, "GoldenRatio": _SymGolden,
}

# =========================
# Normalization & compiler
# =========================
def normalize_to_exp_log(expr, max_iter=8):
    e = expr
    for _ in range(max_iter):
        e2 = e.rewrite(log).rewrite(exp).rewrite(Pow)
        if e2 == e: break
        e = e2
    return e

def eml_from_number(x):
    if isinstance(x, Integer):  return eml_int(int(x))
    if isinstance(x, Rational): return eml_rational(int(x.p), int(x.q))
    if isinstance(x, Float):
        r = Rational(str(x))  # exact via decimal string
        return eml_rational(int(r.p), int(r.q))
    raise TypeError(f"Unsupported numeric: {x!r}")

def compile_to_eml(expr):
    # atoms (numbers/symbols/constants)
    if expr.is_Atom:
        if isinstance(expr, (Integer, Rational, Float)):
            return eml_from_number(expr)
        if expr is E:         return eml_const_E()
        if expr is I:         return eml_const_I()
        if expr == pi:        return eml_const_Pi()
        if expr == _SymGolden:return eml_const_GoldenRatio()
        if isinstance(expr, Symbol):
            return expr.name
        raise TypeError(f"Unexpected atom: {repr(expr)}")

    f = getattr(expr, 'func', None)

    if f is exp and len(expr.args) == 1:
        return eml_exp(compile_to_eml(expr.args[0]))
    if f is log and len(expr.args) == 1:
        return eml_log(compile_to_eml(expr.args[0]))

    if isinstance(expr, Pow):
        base, power = expr.as_base_exp()
        return eml_pow(compile_to_eml(base), compile_to_eml(power))

    if isinstance(expr, Mul):
        facs = list(expr.args)
        acc = compile_to_eml(facs[0])
        for g in facs[1:]:
            acc = eml_mul(acc, compile_to_eml(g))
        return acc

    if isinstance(expr, Add):
        terms = list(expr.args)
        acc = compile_to_eml(terms[0])
        for t in terms[1:]:
            acc = eml_add(acc, compile_to_eml(t))
        return acc

    e2 = normalize_to_exp_log(expr, max_iter=8)
    if e2 == expr:
        raise ValueError(f"Unhandled node after normalization: {expr} (type {type(expr)})")
    return compile_to_eml(e2)

def wl_expr_to_sympy_source(s: str) -> str:
    """Translate common Wolfram Mathematica syntax to a SymPy-parseable source string."""
    t = s.strip()
    if not t:
        raise ValueError("Empty input expression")
    # Mathematica function calls use brackets and powers use ^.
    # This is a pragmatic converter for symbolic math expressions handled by this compiler.
    t = t.replace("[", "(").replace("]", ")")
    t = t.replace("^", "**")
    return t

def eml_compile_from_string(s: str):
    expr = sympify(wl_expr_to_sympy_source(s), locals=LOCALS)
    expr = normalize_to_exp_log(expr, max_iter=8)
    return compile_to_eml(expr)

# =========================
# Test-suite definitions
# =========================
CONST_STRINGS = ["Pi", "E", "I", "0", "1", "-1", "2", "1/2", "-1/2", "GoldenRatio"]

FUNC_STRINGS = [
  "Half(x)", "Minus(x)", "Log(x)", "Exp(x)", "Inv(x)", "Sqrt(x)", "Sqr(x)",
  "Cosh(x)", "Cos(x)", "Sinh(x)", "Sin(x)", "Tanh(x)", "Tan(x)",
  "ArcSinh(x)", "ArcTanh(x)", "ArcSin(x)", "ArcCos(x)", "ArcTan(x)", "ArcCosh(x)",
  "LogisticSigmoid(x)", "x"
]

OP_STRINGS = [
  "Plus(x,y)", "Times(x,y)", "Subtract(x,y)", "Divide(x,y)",
  "Power(x,y)", "Log(x,y)", "Avg(x,y)", "Hypot(x,y)"
]

# =========================
# Emitters (WL files)
# =========================
def _pure_list(items):
    body = ",\n  ".join(items)
    return "{\n  " + body + "\n}\n"

def _orig_to_wl_brackets(s: str) -> str:
    return re.sub(r"([A-Za-z$][A-Za-z0-9$]*)\(", r"\1[", s).replace(")", "]")

def emit_test_wl_files(outdir):
    os.makedirs(outdir, exist_ok=True)
    # Remove legacy combined outputs; they are redundant and no longer emitted.
    for legacy_name in ("test_suite_eml.wl", "test_suite_orig.wl"):
        legacy_path = os.path.join(outdir, legacy_name)
        if os.path.exists(legacy_path):
            os.remove(legacy_path)

    # compile to EML
    const_eml = [eml_compile_from_string(s) for s in CONST_STRINGS]
    func_eml  = [eml_compile_from_string(s) for s in FUNC_STRINGS]
    op_eml    = [eml_compile_from_string(s) for s in OP_STRINGS]

    # originals, WL bracket syntax
    const_orig = [_orig_to_wl_brackets(s) for s in CONST_STRINGS]
    func_orig  = [_orig_to_wl_brackets(s) for s in FUNC_STRINGS]
    op_orig    = [_orig_to_wl_brackets(s) for s in OP_STRINGS]

    # compiled lists
    with open(os.path.join(outdir, "const_eml.wl"), "w", encoding="utf-8") as f:
        f.write("(* Auto-generated EML tests: constants *)\n")
        f.write(_pure_list(const_eml))
    with open(os.path.join(outdir, "funcs_eml.wl"), "w", encoding="utf-8") as f:
        f.write("(* Auto-generated EML tests: unary functions of x *)\n")
        f.write(_pure_list(func_eml))
    with open(os.path.join(outdir, "ops_eml.wl"), "w", encoding="utf-8") as f:
        f.write("(* Auto-generated EML tests: binary operations of x,y *)\n")
        f.write(_pure_list(op_eml))

    # original lists
    with open(os.path.join(outdir, "const_orig.wl"), "w", encoding="utf-8") as f:
        f.write("(* Original (uncompiled) expressions: constants *)\n")
        f.write(_pure_list(const_orig))
    with open(os.path.join(outdir, "funcs_orig.wl"), "w", encoding="utf-8") as f:
        f.write("(* Original (uncompiled) expressions: unary functions of x *)\n")
        f.write(_pure_list(func_orig))
    with open(os.path.join(outdir, "ops_orig.wl"), "w", encoding="utf-8") as f:
        f.write("(* Original (uncompiled) expressions: binary operations of x,y *)\n")
        f.write(_pure_list(op_orig))

    return {
        "const_eml.wl": const_eml,
        "funcs_eml.wl": func_eml,
        "ops_eml.wl": op_eml,
        "const_orig.wl": const_orig,
        "funcs_orig.wl": func_orig,
        "ops_orig.wl": op_orig,
    }

# =========================
# CLI
# =========================
if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Standalone EML compiler. Default mode compiles one Wolfram Mathematica-style expression and prints EML."
    )
    ap.add_argument(
        "expr",
        nargs="?",
        help="Wolfram Mathematica-style expression, e.g. 'Sin[x]' or 'Plus[x, y]'",
    )
    ap.add_argument(
        "--emit-test",
        action="store_true",
        help="write *.wl test lists to ./eml_tests_out",
    )
    args = ap.parse_args()

    if args.emit_test:
        outdir = "eml_tests_out"
        emit_test_wl_files(outdir)
        print(f"Wrote EML + original lists to: {os.path.abspath(outdir)}")
        sys.exit(0)

    expr = args.expr
    if expr is None and not sys.stdin.isatty():
        expr = sys.stdin.read().strip()

    if not expr:
        ap.error("missing expression (provide EXPR or use --emit-test)")

    print(eml_compile_from_string(expr))
