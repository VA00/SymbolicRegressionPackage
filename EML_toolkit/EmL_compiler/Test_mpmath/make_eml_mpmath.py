#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path


DIR = Path(__file__).resolve().parent
PARENT = DIR.parent


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


def main() -> int:
    ap = argparse.ArgumentParser(description="Generate a single-function mpmath EML test.")
    ap.add_argument("expr")
    ap.add_argument("name")
    ap.add_argument("rest", nargs="*")
    args = ap.parse_args()

    ref = infer_ref(args.expr)
    x_min = "-4.0"
    x_max = "4.0"
    step = "0.0625"
    dps = "64"
    if args.rest:
        if any(c.isalpha() for c in args.rest[0]):
            ref = args.rest[0]
            if len(args.rest) > 1:
                x_min = args.rest[1]
            if len(args.rest) > 2:
                x_max = args.rest[2]
            if len(args.rest) > 3:
                step = args.rest[3]
            if len(args.rest) > 4:
                dps = args.rest[4]
        else:
            x_min = args.rest[0]
            if len(args.rest) > 1:
                x_max = args.rest[1]
            if len(args.rest) > 2:
                step = args.rest[2]
            if len(args.rest) > 3:
                dps = args.rest[3]

    namespace: dict[str, object] = {}
    source = (PARENT / "eml_compiler_v4.py").read_text(encoding="utf-8")
    exec(compile(source, str(PARENT / "eml_compiler_v4.py"), "exec"), namespace)
    eml_expr = namespace["eml_compile_from_string"](args.expr)
    py_expr = eml_expr.replace("[", "(").replace("]", ")").replace("EML(", "eml(")

    text = (DIR / "test_eml_mpmath.py.in").read_text(encoding="utf-8")
    repl = {
        "@NAME@": args.name,
        "@REF@": ref,
        "@XMIN@": x_min,
        "@XMAX@": x_max,
        "@STEP@": step,
        "@EXPR@": args.expr,
        "@C_EXPR@": py_expr,
        "@DPS@": dps,
    }
    for key, value in repl.items():
        text = text.replace(key, value)
    out = DIR / "test_eml_mpmath.py"
    out.write_text(text, encoding="utf-8")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
