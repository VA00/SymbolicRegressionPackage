#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path


DIR = Path(__file__).resolve().parent
PARENT = DIR.parent
FUNC_NAME = "eml_f"


def infer_ref(expr: str) -> str:
    fn = expr.split("[", 1)[0].lower()
    mapping = {
        "half": "ref_half",
        "minus": "ref_minus",
        "inv": "ref_inv",
        "sqr": "ref_sqr",
        "logisticsigmoid": "ref_logistic_sigmoid",
        "x": "ref_identity",
        "sin": "np.sin",
        "cos": "np.cos",
        "tan": "np.tan",
        "sinh": "np.sinh",
        "cosh": "np.cosh",
        "tanh": "np.tanh",
        "exp": "np.exp",
        "log": "np.log",
        "sqrt": "np.sqrt",
        "asin": "np.arcsin",
        "acos": "np.arccos",
        "atan": "np.arctan",
        "asinh": "np.arcsinh",
        "acosh": "np.arccosh",
        "atanh": "np.arctanh",
        "arcsin": "np.arcsin",
        "arccos": "np.arccos",
        "arctan": "np.arctan",
        "arcsinh": "np.arcsinh",
        "arccosh": "np.arccosh",
        "arctanh": "np.arctanh",
    }
    return mapping.get(fn, "np.sin")


def build_parser() -> argparse.ArgumentParser:
    return argparse.ArgumentParser(
        description="Generate a NumPy test for one Wolfram-style expression.",
        usage="%(prog)s F[x] [xMin xMax dx]",
        epilog=(
            "Examples:\n"
            "  python make_eml_numpy.py Sqrt[x] 0 4 0.1\n"
            "  python make_eml_numpy.py ArcCos[x] -1 1 0.01\n\n"
            "Defaults: xMin=-4.0, xMax=4.0, dx=0.0625\n"
            "Write the expression in Wolfram form, for example Sqrt[x], not Sqrt."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )


def parse_range(ap: argparse.ArgumentParser, bounds: list[str]) -> tuple[str, str, str]:
    x_min = "-4.0"
    x_max = "4.0"
    step = "0.0625"
    if len(bounds) > 3:
        ap.error("expected at most 3 numbers: xMin xMax dx")
    if bounds:
        x_min = bounds[0]
    if len(bounds) > 1:
        x_max = bounds[1]
    if len(bounds) > 2:
        step = bounds[2]
    return x_min, x_max, step


def load_compiler_namespace() -> dict[str, object]:
    namespace: dict[str, object] = {}
    source = (PARENT / "eml_compiler_v4.py").read_text(encoding="utf-8")
    exec(compile(source, str(PARENT / "eml_compiler_v4.py"), "exec"), namespace)
    return namespace


def main() -> int:
    ap = build_parser()
    ap.add_argument("expr", metavar="F[x]", help="expression to compile, for example Sqrt[x]")
    ap.add_argument("bounds", nargs="*", help=argparse.SUPPRESS)
    args = ap.parse_args()

    x_min, x_max, step = parse_range(ap, args.bounds)
    ref = infer_ref(args.expr)

    namespace = load_compiler_namespace()
    try:
        eml_expr = namespace["eml_compile_from_string"](args.expr)
    except Exception as exc:
        ap.error(str(exc))

    py_expr = eml_expr.replace("[", "(").replace("]", ")").replace("EML(", "eml(")

    text = (DIR / "test_eml_numpy.py.in").read_text(encoding="utf-8")
    repl = {
        "@NAME@": FUNC_NAME,
        "@REF@": ref,
        "@REF_LABEL@": ref,
        "@XMIN@": x_min,
        "@XMAX@": x_max,
        "@STEP@": step,
        "@EXPR@": args.expr,
        "@PY_EXPR@": py_expr,
    }
    for key, value in repl.items():
        text = text.replace(key, value)
    out = DIR / "test_eml_numpy.py"
    out.write_text(text, encoding="utf-8")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
