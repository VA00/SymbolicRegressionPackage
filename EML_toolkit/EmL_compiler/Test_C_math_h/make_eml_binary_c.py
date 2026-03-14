#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
from pathlib import Path


DIR = Path(__file__).resolve().parent
PARENT = DIR.parent
FUNC_NAME = "eml_f"


def infer_ref(expr: str) -> tuple[str, str, str]:
    key = expr.replace(" ", "").lower()
    mapping = {
        "plus[x,y]": ("ref_plus", "x + y", "1"),
        "times[x,y]": ("ref_times", "x * y", "1"),
        "subtract[x,y]": ("ref_subtract", "x - y", "1"),
        "divide[x,y]": ("ref_divide", "x / y", "(fabs(y) > 1e-12)"),
        "power[x,y]": (
            "ref_power",
            "pow(x, y)",
            "((x > 0.0) || (fabs(x) <= 1e-12 && y > 0.0) || (x < 0.0 && is_int_like(y)))",
        ),
        "log[x,y]": (
            "ref_logxy",
            "log(y) / log(x)",
            "(x > 0.0 && y > 0.0 && fabs(x - 1.0) > 1e-12)",
        ),
        "avg[x,y]": ("ref_avg", "(x + y) / 2", "1"),
        "hypot[x,y]": ("ref_hypot_xy", "hypot(x, y)", "1"),
    }
    return mapping.get(key, ("ref_plus", "x + y", "1"))


def build_parser() -> argparse.ArgumentParser:
    return argparse.ArgumentParser(
        description="Generate C test files for one Wolfram-style binary expression.",
        usage="%(prog)s F[x,y] [xMin xMax dx yMin yMax dy]",
        epilog=(
            "Examples:\n"
            "  python make_eml_binary_c.py Plus[x,y]\n"
            "  python make_eml_binary_c.py Hypot[x,y] -4 4 0.25 -4 4 0.25\n\n"
            "Defaults: xMin=-4.0, xMax=4.0, dx=0.5, yMin=-4.0, yMax=4.0, dy=0.5\n"
            "Write the expression in Wolfram form, for example Plus[x,y], not x + y."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )


def parse_ranges(ap: argparse.ArgumentParser, bounds: list[str]) -> tuple[str, str, str, str, str, str]:
    x_min = "-4.0"
    x_max = "4.0"
    x_step = "0.5"
    y_min = "-4.0"
    y_max = "4.0"
    y_step = "0.5"
    if len(bounds) not in (0, 6):
        ap.error("expected either no bounds or exactly 6 numbers: xMin xMax dx yMin yMax dy")
    if bounds:
        x_min, x_max, x_step, y_min, y_max, y_step = bounds
    return x_min, x_max, x_step, y_min, y_max, y_step


def load_compiler_namespace() -> dict[str, object]:
    namespace: dict[str, object] = {}
    source = (PARENT / "eml_compiler_v4.py").read_text(encoding="utf-8")
    exec(compile(source, str(PARENT / "eml_compiler_v4.py"), "exec"), namespace)
    return namespace


def compile_test_program() -> Path | None:
    if os.name == "nt":
        compiler = shutil.which("icx")
        if compiler is None:
            return None
        out = DIR / "test_eml_binary.exe"
        subprocess.run(
            [compiler, "/nologo", "/Qstd:c11", "/TC", "test_eml_binary.c", f"/Fe:{out.name}"],
            cwd=DIR,
            check=True,
            text=True,
        )
        return out

    compiler = shutil.which("cc")
    if compiler is None:
        return None

    out = DIR / "test_eml_binary"
    cmd = [compiler, "-std=c11", "-O0", "test_eml_binary.c", "-o", out.name]
    try:
        subprocess.run(cmd, cwd=DIR, check=True, text=True)
    except subprocess.CalledProcessError:
        subprocess.run(cmd[:-2] + ["-lm", "-o", out.name], cwd=DIR, check=True, text=True)
    return out


def main() -> int:
    ap = build_parser()
    ap.add_argument("expr", metavar="F[x,y]", help="binary expression to compile, for example Plus[x,y]")
    ap.add_argument("bounds", nargs="*", help=argparse.SUPPRESS)
    args = ap.parse_args()

    x_min, x_max, x_step, y_min, y_max, y_step = parse_ranges(ap, args.bounds)
    ref, ref_label, domain = infer_ref(args.expr)

    namespace = load_compiler_namespace()
    try:
        eml_expr = namespace["eml_compile_from_string"](args.expr)
    except Exception as exc:
        ap.error(str(exc))

    c_expr = eml_expr.replace("[", "(").replace("]", ")").replace("EML(", "eml(")

    text = (DIR / "eml_math_binary.h.in").read_text(encoding="utf-8")
    repl = {
        "@NAME@": FUNC_NAME,
        "@EXPR@": args.expr,
        "@C_EXPR@": c_expr,
    }
    for key, value in repl.items():
        text = text.replace(key, value)
    header_out = DIR / "eml_math_binary.h"
    header_out.write_text(text, encoding="utf-8")

    text = (DIR / "test_eml_binary.c.in").read_text(encoding="utf-8")
    repl = {
        "@NAME@": FUNC_NAME,
        "@REF@": ref,
        "@REF_LABEL@": ref_label,
        "@DOMAIN@": domain,
        "@XMIN@": x_min,
        "@XMAX@": x_max,
        "@XSTEP@": x_step,
        "@YMIN@": y_min,
        "@YMAX@": y_max,
        "@YSTEP@": y_step,
        "@EXPR@": args.expr,
        "@C_EXPR@": c_expr,
    }
    for key, value in repl.items():
        text = text.replace(key, value)
    source_out = DIR / "test_eml_binary.c"
    source_out.write_text(text, encoding="utf-8")

    print(f"wrote {header_out}")
    print(f"wrote {source_out}")

    binary_out = compile_test_program()
    if binary_out is not None:
        print(f"built {binary_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
