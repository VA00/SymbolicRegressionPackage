#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path


DIR = Path(__file__).resolve().parent
PARENT = DIR.parent
FUNC_NAME = "eml_f"


def infer_ref(expr: str) -> tuple[str, str]:
    key = expr.replace(" ", "").lower()
    mapping = {
        "plus[x,y]": ("ref_plus", "x + y"),
        "times[x,y]": ("ref_times", "x * y"),
        "subtract[x,y]": ("ref_subtract", "x - y"),
        "divide[x,y]": ("ref_divide", "x / y"),
        "power[x,y]": ("ref_power", "np.power(x, y)"),
        "log[x,y]": ("ref_logxy", "np.log(y) / np.log(x)"),
        "avg[x,y]": ("ref_avg", "(x + y) / 2"),
        "hypot[x,y]": ("ref_hypot_xy", "np.hypot(x, y)"),
    }
    return mapping.get(key, ("ref_plus", "x + y"))


def build_parser() -> argparse.ArgumentParser:
    return argparse.ArgumentParser(
        description="Generate a NumPy test for one Wolfram-style binary expression.",
        usage="%(prog)s F[x,y] [xMin xMax dx yMin yMax dy]",
        epilog=(
            "Examples:\n"
            "  python make_eml_binary_numpy.py Plus[x,y]\n"
            "  python make_eml_binary_numpy.py Hypot[x,y] 0.25 4 0.25 0.25 4 0.25\n\n"
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


def main() -> int:
    ap = build_parser()
    ap.add_argument("expr", metavar="F[x,y]", help="binary expression to compile, for example Plus[x,y]")
    ap.add_argument("bounds", nargs="*", help=argparse.SUPPRESS)
    args = ap.parse_args()

    x_min, x_max, x_step, y_min, y_max, y_step = parse_ranges(ap, args.bounds)
    ref, ref_label = infer_ref(args.expr)

    namespace = load_compiler_namespace()
    try:
        eml_expr = namespace["eml_compile_from_string"](args.expr)
    except Exception as exc:
        ap.error(str(exc))

    py_expr = eml_expr.replace("[", "(").replace("]", ")").replace("EML(", "eml(")

    text = (DIR / "test_eml_binary_numpy.py.in").read_text(encoding="utf-8")
    repl = {
        "@NAME@": FUNC_NAME,
        "@REF@": ref,
        "@REF_LABEL@": ref_label,
        "@XMIN@": x_min,
        "@XMAX@": x_max,
        "@XSTEP@": x_step,
        "@YMIN@": y_min,
        "@YMAX@": y_max,
        "@YSTEP@": y_step,
        "@EXPR@": args.expr,
        "@PY_EXPR@": py_expr,
    }
    for key, value in repl.items():
        text = text.replace(key, value)
    out = DIR / "test_eml_binary_numpy.py"
    out.write_text(text, encoding="utf-8")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
