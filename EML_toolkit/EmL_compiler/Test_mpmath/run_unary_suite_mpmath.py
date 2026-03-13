#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import subprocess
import sys
from pathlib import Path

DIR = Path(__file__).resolve().parent

CASES = [
    ("Half[x]", "eml_half", -4.0, 4.0, 0.125),
    ("Minus[x]", "eml_minus", -4.0, 4.0, 0.125),
    ("Log[x]", "eml_log", 0.125, 8.0, 0.125),
    ("Exp[x]", "eml_exp", -4.0, 4.0, 0.125),
    ("Inv[x]", "eml_inv", 0.125, 8.0, 0.125),
    ("Sqrt[x]", "eml_sqrt", 0.0, 8.0, 0.125),
    ("Sqr[x]", "eml_sqr", -4.0, 4.0, 0.125),
    ("Cosh[x]", "eml_cosh", -4.0, 4.0, 0.125),
    ("Cos[x]", "eml_cos", -20.0, 20.0, 0.125),
    ("Sinh[x]", "eml_sinh", -4.0, 4.0, 0.125),
    ("Sin[x]", "eml_sin", -20.0, 20.0, 0.125),
    ("Tanh[x]", "eml_tanh", -4.0, 4.0, 0.125),
    ("Tan[x]", "eml_tan", -1.25, 1.25, 0.03125),
    ("ArcSinh[x]", "eml_arcsinh", -8.0, 8.0, 0.125),
    ("ArcTanh[x]", "eml_arctanh", -0.95, 0.95, 0.01),
    ("ArcSin[x]", "eml_arcsin", -1.0, 1.0, 0.01),
    ("ArcCos[x]", "eml_arccos", -1.0, 1.0, 0.01),
    ("ArcTan[x]", "eml_arctan", -8.0, 8.0, 0.125),
    ("ArcCosh[x]", "eml_arccosh", 1.0, 9.0, 0.125),
    ("LogisticSigmoid[x]", "eml_logistic_sigmoid", -8.0, 8.0, 0.125),
    ("x", "eml_identity", -8.0, 8.0, 0.125),
]


def run(cmd: list[str]) -> str:
    return subprocess.run(cmd, cwd=DIR, check=True, text=True, capture_output=True).stdout


def parse_output(text: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for line in text.splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        out[key.strip()] = value.strip()
    return out


def parse_err(value: str) -> tuple[float, float]:
    err_s, x_s = value.split(" at x=")
    return float(err_s), float(x_s)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dps", type=int, default=64)
    args = ap.parse_args()

    results = []
    for expr, name, xmin, xmax, step in CASES:
        run([sys.executable, "./make_eml_mpmath.py", expr, name, str(xmin), str(xmax), str(step), str(args.dps)])
        out = parse_output(run([sys.executable, "./test_eml_mpmath.py"]))
        re_err, re_x = parse_err(out["worst re err"])
        im_err, im_x = parse_err(out["worst im err"])
        results.append({
            "expr": expr,
            "samples": int(out["samples"]),
            "valid": int(out["valid"]),
            "nonfinite": int(out["nonfinite"]),
            "re": abs(re_err),
            "re_x": re_x,
            "im": abs(im_err),
            "im_x": im_x,
        })

    lines = [f"Unary EML Suite: mpmath (mp.dps={args.dps})", ""]
    for r in results:
        lines.append(
            f"{r['expr']:<20} valid={r['valid']:>4}/{r['samples']:<4} nonfinite={r['nonfinite']:<3} "
            f"max|re|={r['re']:.3e} @ {r['re_x']:.6g} max|im|={r['im']:.3e} @ {r['im_x']:.6g}"
        )
    worst_re = max(results, key=lambda r: (math.inf if math.isnan(r["re"]) else r["re"]))
    worst_im = max(results, key=lambda r: (math.inf if math.isnan(r["im"]) else r["im"]))
    lines.extend([
        "",
        f"Worst real error : {worst_re['expr']} -> {worst_re['re']:.3e} at x={worst_re['re_x']:.6g}",
        f"Worst imag error : {worst_im['expr']} -> {worst_im['im']:.3e} at x={worst_im['im_x']:.6g}",
    ])
    report = "\n".join(lines) + "\n"
    path = DIR / "unary_suite_report_mpmath.txt"
    path.write_text(report, encoding="utf-8")
    print(report, end="")
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
