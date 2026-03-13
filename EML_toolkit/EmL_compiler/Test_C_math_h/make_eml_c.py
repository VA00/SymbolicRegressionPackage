#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
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
        "sin": "sin",
        "cos": "cos",
        "tan": "tan",
        "sinh": "sinh",
        "cosh": "cosh",
        "tanh": "tanh",
        "exp": "exp",
        "log": "log",
        "sqrt": "sqrt",
        "asin": "asin",
        "acos": "acos",
        "atan": "atan",
        "asinh": "asinh",
        "acosh": "acosh",
        "atanh": "atanh",
        "arcsin": "asin",
        "arccos": "acos",
        "arctan": "atan",
        "arcsinh": "asinh",
        "arccosh": "acosh",
        "arctanh": "atanh",
    }
    return mapping.get(fn, "sin")


def compile_test_program() -> Path | None:
    if os.name == "nt":
        compiler = shutil.which("icx")
        if compiler is None:
            return None
        out = DIR / "test_eml.exe"
        subprocess.run(
            [compiler, "/nologo", "/Qstd:c11", "/TC", "test_eml.c", f"/Fe:{out.name}"],
            cwd=DIR,
            check=True,
            text=True,
        )
        return out

    compiler = shutil.which("cc")
    if compiler is None:
        return None

    out = DIR / "test_eml"
    cmd = [compiler, "-std=c11", "-O0", "test_eml.c", "-o", out.name]
    try:
        subprocess.run(cmd, cwd=DIR, check=True, text=True)
    except subprocess.CalledProcessError:
        subprocess.run(cmd[:-2] + ["-lm", "-o", out.name], cwd=DIR, check=True, text=True)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="Generate a single-function C EML test.")
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
    c_expr = eml_expr.replace("[", "(").replace("]", ")").replace("EML(", "eml(")

    text = (DIR / "eml_math.h.in").read_text(encoding="utf-8")
    repl = {
        "@NAME@": args.name,
        "@REF@": ref,
        "@XMIN@": x_min,
        "@XMAX@": x_max,
        "@STEP@": step,
        "@EXPR@": args.expr,
        "@C_EXPR@": c_expr,
    }
    for key, value in repl.items():
        text = text.replace(key, value)
    header_out = DIR / "eml_math.h"
    header_out.write_text(text, encoding="utf-8")

    text = (DIR / "test_eml.c.in").read_text(encoding="utf-8")
    for key, value in repl.items():
        text = text.replace(key, value)
    source_out = DIR / "test_eml.c"
    source_out.write_text(text, encoding="utf-8")

    print(f"wrote {header_out}")
    print(f"wrote {source_out}")

    binary_out = compile_test_program()
    if binary_out is not None:
        print(f"built {binary_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
