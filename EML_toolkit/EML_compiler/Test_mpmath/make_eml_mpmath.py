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
        "sin": "mp.sin",
        "cos": "mp.cos",
        "tan": "mp.tan",
        "sinh": "mp.sinh",
        "cosh": "mp.cosh",
        "tanh": "mp.tanh",
        "exp": "mp.exp",
        "log": "mp.log",
        "sqrt": "mp.sqrt",
        "asin": "mp.asin",
        "acos": "mp.acos",
        "atan": "mp.atan",
        "asinh": "mp.asinh",
        "acosh": "mp.acosh",
        "atanh": "mp.atanh",
        "arcsin": "mp.asin",
        "arccos": "mp.acos",
        "arctan": "mp.atan",
        "arcsinh": "mp.asinh",
        "arccosh": "mp.acosh",
        "arctanh": "mp.atanh",
    }
    return mapping.get(fn, "mp.sin")


def build_parser() -> argparse.ArgumentParser:
    return argparse.ArgumentParser(
        description="Generate an mpmath test for one Wolfram-style expression.",
        usage="%(prog)s F[x] [xMin xMax dx [digits]]",
        epilog=(
            "Examples:\n"
            "  python make_eml_mpmath.py Sqrt[x] 0 4 0.1 64\n"
            "  python make_eml_mpmath.py ArcCos[x] -1 1 0.01 128\n\n"
            "Defaults: xMin=-4.0, xMax=4.0, dx=0.0625, digits=64\n"
            "Write the expression in Wolfram form, for example Sqrt[x], not Sqrt."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )


def parse_range(ap: argparse.ArgumentParser, bounds: list[str]) -> tuple[str, str, str, str]:
    x_min = "-4.0"
    x_max = "4.0"
    step = "0.0625"
    digits = "64"
    if len(bounds) > 4:
        ap.error("expected at most 4 values: xMin xMax dx digits")
    if bounds:
        x_min = bounds[0]
    if len(bounds) > 1:
        x_max = bounds[1]
    if len(bounds) > 2:
        step = bounds[2]
    if len(bounds) > 3:
        digits = bounds[3]
    return x_min, x_max, step, digits


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

    x_min, x_max, step, digits = parse_range(ap, args.bounds)
    ref = infer_ref(args.expr)

    namespace = load_compiler_namespace()
    try:
        eml_expr = namespace["eml_compile_from_string"](args.expr)
    except Exception as exc:
        ap.error(str(exc))

    py_expr = eml_expr.replace("[", "(").replace("]", ")").replace("EML(", "eml(")

    text = (DIR / "test_eml_mpmath.py.in").read_text(encoding="utf-8")
    repl = {
        "@NAME@": FUNC_NAME,
        "@REF@": ref,
        "@XMIN@": x_min,
        "@XMAX@": x_max,
        "@STEP@": step,
        "@EXPR@": args.expr,
        "@C_EXPR@": py_expr,
        "@DPS@": digits,
    }
    for key, value in repl.items():
        text = text.replace(key, value)
    out = DIR / "test_eml_mpmath.py"
    out.write_text(text, encoding="utf-8")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
