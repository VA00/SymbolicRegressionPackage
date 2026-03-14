#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path


DIR = Path(__file__).resolve().parent
PARENT = DIR.parent
FUNC_NAME = "eml_f"


def infer_ref(expr: str) -> tuple[str, str, str]:
    key = expr.replace(" ", "").lower()
    mapping = {
        "plus[x,y]": ("ref_plus", "torch.add(x, y)", "torch.ones_like(x, dtype=torch.bool)"),
        "times[x,y]": ("ref_times", "torch.mul(x, y)", "torch.ones_like(x, dtype=torch.bool)"),
        "subtract[x,y]": ("ref_subtract", "torch.sub(x, y)", "torch.ones_like(x, dtype=torch.bool)"),
        "divide[x,y]": ("ref_divide", "torch.div(x, y)", "~torch.isclose(y, tr(0.0), atol=1e-12, rtol=0.0)"),
        "power[x,y]": (
            "ref_power",
            "torch.pow(x, y)",
            "((x > 0.0) | (torch.isclose(x, tr(0.0), atol=1e-12, rtol=0.0) & (y > 0.0)) | ((x < 0.0) & torch.isclose(y, torch.round(y), atol=1e-12, rtol=0.0)))",
        ),
        "log[x,y]": (
            "ref_logxy",
            "torch.log(y) / torch.log(x)",
            "((x > 0.0) & (y > 0.0) & (~torch.isclose(x, tr(1.0), atol=1e-12, rtol=0.0)))",
        ),
        "avg[x,y]": ("ref_avg", "torch.div(torch.add(x, y), 2.0)", "torch.ones_like(x, dtype=torch.bool)"),
        "hypot[x,y]": ("ref_hypot_xy", "torch.hypot(x, y)", "torch.ones_like(x, dtype=torch.bool)"),
    }
    return mapping.get(key, ("ref_plus", "torch.add(x, y)", "torch.ones_like(x, dtype=torch.bool)"))


def build_parser() -> argparse.ArgumentParser:
    return argparse.ArgumentParser(
        description="Generate a PyTorch test for one Wolfram-style binary expression.",
        usage="%(prog)s F[x,y] [xMin xMax dx yMin yMax dy]",
        epilog=(
            "Examples:\n"
            "  python make_eml_binary_torch.py Plus[x,y]\n"
            "  python make_eml_binary_torch.py Hypot[x,y] -4 4 0.25 -4 4 0.25\n\n"
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
    ref, ref_label, domain = infer_ref(args.expr)

    namespace = load_compiler_namespace()
    try:
        eml_expr = namespace["eml_compile_from_string"](args.expr)
    except Exception as exc:
        ap.error(str(exc))

    py_expr = eml_expr.replace("[", "(").replace("]", ")").replace("EML(", "eml(")

    text = (DIR / "test_eml_binary_torch.py.in").read_text(encoding="utf-8")
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
        "@PY_EXPR@": py_expr,
    }
    for key, value in repl.items():
        text = text.replace(key, value)
    out = DIR / "test_eml_binary_torch.py"
    out.write_text(text, encoding="utf-8")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
