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


def main() -> int:
    ap = argparse.ArgumentParser(description="Generate a single-function numpy EML test.")
    ap.add_argument("expr")
    ap.add_argument("name")
    ap.add_argument("rest", nargs="*")
    args = ap.parse_args()

    ref = infer_ref(args.expr)
    x_min = "-4.0"
    x_max = "4.0"
    step = "0.0625"
    if args.rest:
        if any(c.isalpha() for c in args.rest[0]):
            ref = args.rest[0]
            if len(args.rest) > 1:
                x_min = args.rest[1]
            if len(args.rest) > 2:
                x_max = args.rest[2]
            if len(args.rest) > 3:
                step = args.rest[3]
        else:
            x_min = args.rest[0]
            if len(args.rest) > 1:
                x_max = args.rest[1]
            if len(args.rest) > 2:
                step = args.rest[2]

    namespace: dict[str, object] = {}
    source = (PARENT / "eml_compiler_v4.py").read_text(encoding="utf-8")
    exec(compile(source, str(PARENT / "eml_compiler_v4.py"), "exec"), namespace)
    eml_expr = namespace["eml_compile_from_string"](args.expr)
    py_expr = eml_expr.replace("[", "(").replace("]", ")").replace("EML(", "eml(")

    text = (DIR / "test_eml_numpy.py.in").read_text(encoding="utf-8")
    repl = {
        "@NAME@": args.name,
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
