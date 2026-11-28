#!/usr/bin/env python3
import os
import sys

import eml_compiler_v4 as base


def eml_neg_clean(z):
    # Clean variant that avoids Log[0] in the negation primitive.
    e = base.eml_const_E()
    e_minus_one = base.eml_sub(e, "1")
    one_plus_z = base.eml_sub(e, base.eml_sub(e_minus_one, z))
    return base.eml_sub("1", one_plus_z)


base.eml_neg = eml_neg_clean


if __name__ == "__main__":
    args = base.parse_cli_args()

    if args.emit_test:
        outdir = "eml_tests_out_clean_math"
        base.emit_test_wl_files(outdir)
        print(f"Wrote EML + original lists to: {os.path.abspath(outdir)}")
        sys.exit(0)

    expr = args.expr
    if expr is None and not sys.stdin.isatty():
        expr = sys.stdin.read().strip()

    if not expr:
        base._build_arg_parser().error("missing expression (provide EXPR or use --emit-test)")

    print(base.eml_compile_from_string(expr))
