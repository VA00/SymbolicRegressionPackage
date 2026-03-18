from __future__ import annotations

import argparse
import math
import time
from dataclasses import dataclass
from datetime import datetime

import sympy as sp
from sympy.parsing.sympy_parser import parse_expr


@dataclass(frozen=True, slots=True)
class Candidate:
    value: float
    left_tokens: int
    left_index: int
    right_tokens: int
    right_index: int


def eml_expr(x: sp.Expr, y: sp.Expr) -> sp.Expr:
    return sp.exp(x) - sp.log(y)


def parse_target(text: str) -> sp.Expr:
    locals_dict = {
        "E": sp.E,
        "GoldenRatio": sp.GoldenRatio,
        "Pi": sp.pi,
        "pi": sp.pi,
        "Sqrt": sp.sqrt,
        "sqrt": sp.sqrt,
    }
    return parse_expr(text, local_dict=locals_dict, evaluate=True)


def reconstruct_expression(
    levels: dict[int, list[Candidate]], tokens: int, index: int
) -> tuple[sp.Expr, list[object]]:
    if tokens == 1:
        return sp.Integer(1), [1]

    candidate = levels[tokens][index]
    left_expr, left_code = reconstruct_expression(
        levels, candidate.left_tokens, candidate.left_index
    )
    right_expr, right_code = reconstruct_expression(
        levels, candidate.right_tokens, candidate.right_index
    )
    return eml_expr(left_expr, right_expr), left_code + right_code + ["EML"]


def format_wolfram_rpn(code: list[object]) -> str:
    formatted = ", ".join("EML" if token == "EML" else str(token) for token in code)
    return f"rpnRule[{{{formatted}}}]"


def search(target: sp.Expr, max_tokens: int, tolerance: float) -> tuple[int, list[object], sp.Expr, float]:
    target_float = float(sp.N(target, 50))
    levels: dict[int, list[Candidate]] = {
        1: [Candidate(1.0, 0, 0, 0, 0)]
    }
    started = time.perf_counter()

    if abs(1.0 - target_float) < tolerance and sp.Integer(1) == target:
        return 1, [1], sp.Integer(1), time.perf_counter() - started

    for tokens in range(3, max_tokens + 1, 2):
        level: list[Candidate] = []
        seen_values: set[float] = set()
        for left_tokens in range(1, tokens - 1, 2):
            right_tokens = tokens - left_tokens - 1
            left_level = levels[left_tokens]
            right_level = levels[right_tokens]
            for left_index, left in enumerate(left_level):
                try:
                    exp_left = math.exp(left.value)
                except OverflowError:
                    continue
                for right_index, right in enumerate(right_level):
                    try:
                        value = exp_left - math.log(right.value)
                    except (OverflowError, ValueError):
                        continue
                    if not math.isfinite(value) or value in seen_values:
                        continue
                    seen_values.add(value)
                    level.append(
                        Candidate(
                            value=value,
                            left_tokens=left_tokens,
                            left_index=left_index,
                            right_tokens=right_tokens,
                            right_index=right_index,
                        )
                    )
                    if abs(value - target_float) >= tolerance:
                        continue
                    levels[tokens] = level
                    expression, code = reconstruct_expression(levels, tokens, len(level) - 1)
                    if sp.simplify(expression - target) == 0:
                        return tokens, code, sp.simplify(expression), time.perf_counter() - started
        levels[tokens] = level

    raise RuntimeError(
        f"no exact expression found for target={target} up to max_tokens={max_tokens}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", default="2")
    parser.add_argument("--max-tokens", type=int, default=27)
    parser.add_argument("--tolerance", type=float, default=1e-10)
    args = parser.parse_args()

    target = parse_target(args.target)
    wall_start = datetime.now()
    print(wall_start.isoformat(sep=" "))

    tokens, code, expression, elapsed = search(
        target=target,
        max_tokens=args.max_tokens,
        tolerance=args.tolerance,
    )

    wall_end = datetime.now()
    print(format_wolfram_rpn(code))
    print(expression)
    print(f"tokens={tokens}")
    print(wall_end.isoformat(sep=" "))
    print(f"{elapsed:.9f} Seconds")
    print("Finished!")


if __name__ == "__main__":
    main()
