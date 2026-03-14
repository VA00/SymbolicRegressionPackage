#!/usr/bin/env python3
from __future__ import annotations

import math
import subprocess
import sys
from pathlib import Path


DIR = Path(__file__).resolve().parent


CASES = [
    ("Plus[x,y]", -8.0, 8.0, 0.5, -8.0, 8.0, 0.5),
    ("Times[x,y]", 0.25, 8.0, 0.25, 0.25, 8.0, 0.25),
    ("Subtract[x,y]", -8.0, 8.0, 0.5, -8.0, 8.0, 0.5),
    ("Divide[x,y]", 0.25, 8.0, 0.25, 0.25, 8.0, 0.25),
    ("Power[x,y]", 0.25, 8.0, 0.25, 0.25, 3.0, 0.25),
    ("Log[x,y]", 2.0, 8.0, 0.25, 0.25, 8.0, 0.25),
    ("Avg[x,y]", -8.0, 8.0, 0.5, -8.0, 8.0, 0.5),
    ("Hypot[x,y]", 0.25, 8.0, 0.25, 0.25, 8.0, 0.25),
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


def parse_err(value: str) -> tuple[float, float, float]:
    err_s, point_s = value.split(" at x=")
    x_s, y_s = point_s.split(", y=")
    return float(err_s), float(x_s), float(y_s)


def main() -> int:
    results = []
    for expr, xmin, xmax, xstep, ymin, ymax, ystep in CASES:
        run(
            [
                sys.executable,
                "./make_eml_binary_torch.py",
                expr,
                str(xmin),
                str(xmax),
                str(xstep),
                str(ymin),
                str(ymax),
                str(ystep),
            ]
        )
        out = parse_output(run([sys.executable, "./test_eml_binary_torch.py"]))
        re_err, re_x, re_y = parse_err(out["worst re err"])
        im_err, im_x, im_y = parse_err(out["worst im err"])
        results.append(
            {
                "expr": expr,
                "samples": int(out["samples"]),
                "valid": int(out["valid"]),
                "nonfinite": int(out["nonfinite"]),
                "re": abs(re_err),
                "re_x": re_x,
                "re_y": re_y,
                "im": abs(im_err),
                "im_x": im_x,
                "im_y": im_y,
            }
        )

    lines = ["Binary EML Suite: torch", ""]
    for r in results:
        lines.append(
            f"{r['expr']:<20} valid={r['valid']:>4}/{r['samples']:<4} nonfinite={r['nonfinite']:<3} "
            f"max|re|={r['re']:.3e} @ ({r['re_x']:.6g}, {r['re_y']:.6g}) "
            f"max|im|={r['im']:.3e} @ ({r['im_x']:.6g}, {r['im_y']:.6g})"
        )
    worst_re = max(results, key=lambda r: (math.inf if math.isnan(r["re"]) else r["re"]))
    worst_im = max(results, key=lambda r: (math.inf if math.isnan(r["im"]) else r["im"]))
    lines.extend(
        [
            "",
            f"Worst real error : {worst_re['expr']} -> {worst_re['re']:.3e} at (x,y)=({worst_re['re_x']:.6g}, {worst_re['re_y']:.6g})",
            f"Worst imag error : {worst_im['expr']} -> {worst_im['im']:.3e} at (x,y)=({worst_im['im_x']:.6g}, {worst_im['im_y']:.6g})",
        ]
    )
    report = "\n".join(lines) + "\n"
    path = DIR / "binary_suite_report_torch.txt"
    path.write_text(report, encoding="utf-8")
    print(report, end="")
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
