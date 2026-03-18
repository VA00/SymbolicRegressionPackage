import argparse
import ast
import os
import pathlib
import subprocess
import time
from datetime import datetime

import sympy as sp
from sympy.parsing.sympy_parser import parse_expr

def parse_target(text: str) -> sp.Expr:
    locals_dict = {
        "E": sp.E,
        "GoldenRatio": sp.GoldenRatio,
        "Pi": sp.pi,
        "pi": sp.pi,
        "Sqrt": sp.sqrt,
        "sqrt": sp.sqrt,
    }
    # Convert Mathematica-style square brackets to Python parentheses
    text = text.replace("[", "(").replace("]", ")")
    return parse_expr(text, local_dict=locals_dict, evaluate=True)

def eml_expr(x: sp.Expr, y: sp.Expr) -> sp.Expr:
    return sp.exp(x) - sp.log(y)

def evaluate_rpn(tokens: list) -> sp.Expr:
    stack = []
    for token in tokens:
        if token == 1:
            stack.append(sp.Integer(1))
        elif token == "EML":
            if len(stack) < 2:
                raise ValueError("Invalid RPN: insufficient values for EML")
            right = stack.pop()
            left = stack.pop()
            stack.append(eml_expr(left, right))
        else:
            raise ValueError(f"Unknown token: {token}")
    if len(stack) != 1:
        raise ValueError("Invalid RPN: stray values left on stack")
    return stack[0]

def format_wolfram_rpn(code: list) -> str:
    formatted = ", ".join("EML" if token == "EML" else str(token) for token in code)
    return f"rpnRule[{{{formatted}}}]"


def rust_executable_path(engine_dir: str) -> str:
    exe_suffix = ".exe" if os.name == "nt" else ""
    exe_name = f"eml_core{exe_suffix}"
    return str(pathlib.Path(engine_dir) / "target" / "release" / exe_name)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", default="2")
    parser.add_argument("--max-tokens", type=int, default=41)
    parser.add_argument("--tolerance", type=float, default=1e-10)
    parser.add_argument("--engine-dir", default="./eml_core")
    args = parser.parse_args()

    target_expr = parse_target(args.target)
    target_float = float(sp.N(target_expr, 50))

    # Compile the Rust engine first
    print("Compiling Rust engine (release mode)...", flush=True)
    subprocess.run(["cargo", "build", "--release"], cwd=args.engine_dir, check=True)

    wall_start = datetime.now()
    started = time.perf_counter()
    print(wall_start.isoformat(sep=" "), flush=True)

    # Run the Rust engine
    exe_path = os.path.abspath(rust_executable_path(args.engine_dir))
    rust_process = subprocess.Popen(
        [exe_path, "--target", str(target_float), "--max-tokens", str(args.max_tokens), "--tolerance", str(args.tolerance)],
        cwd=args.engine_dir,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )

    found = False
    last_completed_level = 1
    for line in iter(rust_process.stdout.readline, ''):
        line = line.strip()
        if not line:
            continue
        if line.startswith("DEBUG:"):
            print(line, flush=True)
            if "Level " in line:
                try:
                    level_text = line.split("Level ", 1)[1].split(" ", 1)[0]
                    last_completed_level = int(level_text)
                except (IndexError, ValueError):
                    pass
            continue
        if line.startswith("CANDIDATE:"):
            # CANDIDATE: [1, 1, 'EML', 1, 'EML']
            # Parse it securely
            rpn_str = line[len("CANDIDATE:"):].strip()
            # replace 'EML' or "EML" to just string so ast can parse
            try:
                tokens = ast.literal_eval(rpn_str)
            except Exception as e:
                print(f"Failed to parse candidate: {rpn_str} ({e})", flush=True)
                continue
            
            # Reconstruct sympify and check
            try:
                candidate_expr = evaluate_rpn(tokens)
                if sp.simplify(candidate_expr - target_expr) == 0:
                    elapsed = time.perf_counter() - started
                    wall_end = datetime.now()
                    
                    print(format_wolfram_rpn(tokens), flush=True)
                    print(sp.simplify(candidate_expr), flush=True)
                    print(f"tokens={len(tokens)}", flush=True)
                    print(wall_end.isoformat(sep=" "), flush=True)
                    print(f"{elapsed:.9f} Seconds", flush=True)
                    print("Finished!", flush=True)
                    found = True
                    rust_process.terminate()
                    break
            except Exception as e:
                pass

    if not found:
        return_code = rust_process.wait()
        # Check stderr if nothing was found
        err = rust_process.stderr.read()
        if err:
            print("Rust engine error output:", err, flush=True)
        if return_code == 0:
            print(f"no exact expression found for target={args.target} up to max_tokens={args.max_tokens}")
        else:
            print(
                f"Rust engine exited early with code {return_code} after completing level {last_completed_level}; "
                f"requested max_tokens={args.max_tokens}",
                flush=True,
            )

if __name__ == "__main__":
    main()
