#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path


DIR = Path(__file__).resolve().parent
PARENT = DIR.parent


def infer_ref(expr: str) -> tuple[str, str]:
    fn = expr.split("[", 1)[0].lower()
    mapping = {
        "half": ("ref_half", "ref_half"),
        "minus": ("ref_minus", "ref_minus"),
        "inv": ("ref_inv", "ref_inv"),
        "sqr": ("ref_sqr", "ref_sqr"),
        "logisticsigmoid": ("ref_logistic_sigmoid", "ref_logistic_sigmoid"),
        "x": ("ref_identity", "ref_identity"),
        "sin": ("ref_torch_sin", "torch.sin"),
        "cos": ("ref_torch_cos", "torch.cos"),
        "tan": ("ref_torch_tan", "torch.tan"),
        "sinh": ("ref_torch_sinh", "torch.sinh"),
        "cosh": ("ref_torch_cosh", "torch.cosh"),
        "tanh": ("ref_torch_tanh", "torch.tanh"),
        "exp": ("ref_torch_exp", "torch.exp"),
        "log": ("ref_torch_log", "torch.log"),
        "sqrt": ("ref_torch_sqrt", "torch.sqrt"),
        "asin": ("ref_torch_asin", "torch.asin"),
        "acos": ("ref_torch_acos", "torch.acos"),
        "atan": ("ref_torch_atan", "torch.atan"),
        "asinh": ("ref_torch_asinh", "torch.asinh"),
        "acosh": ("ref_torch_acosh", "torch.acosh"),
        "atanh": ("ref_torch_atanh", "torch.atanh"),
        "arcsin": ("ref_torch_asin", "torch.asin"),
        "arccos": ("ref_torch_acos", "torch.acos"),
        "arctan": ("ref_torch_atan", "torch.atan"),
        "arcsinh": ("ref_torch_asinh", "torch.asinh"),
        "arccosh": ("ref_torch_acosh", "torch.acosh"),
        "arctanh": ("ref_torch_atanh", "torch.atanh"),
    }
    return mapping.get(fn, ("ref_torch_sin", "torch.sin"))


def main() -> int:
    ap = argparse.ArgumentParser(description="Generate a single-function torch EML test.")
    ap.add_argument("expr")
    ap.add_argument("name")
    ap.add_argument("rest", nargs="*")
    args = ap.parse_args()

    ref, ref_label = infer_ref(args.expr)
    x_min = "-4.0"
    x_max = "4.0"
    step = "0.0625"
    if args.rest:
        if any(c.isalpha() for c in args.rest[0]):
            ref = args.rest[0]
            ref_label = args.rest[0]
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

    text = (DIR / "test_eml_torch.py.in").read_text(encoding="utf-8")
    repl = {
        "@NAME@": args.name,
        "@REF@": ref,
        "@REF_LABEL@": ref_label,
        "@XMIN@": x_min,
        "@XMAX@": x_max,
        "@STEP@": step,
        "@EXPR@": args.expr,
        "@PY_EXPR@": py_expr,
    }
    for key, value in repl.items():
        text = text.replace(key, value)
    out = DIR / "test_eml_torch.py"
    out.write_text(text, encoding="utf-8")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
