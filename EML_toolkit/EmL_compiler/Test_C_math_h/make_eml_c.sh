#!/bin/sh
set -eu

if [ "$#" -lt 2 ] || [ "$#" -gt 6 ]; then
  echo "usage: $0 'Sin[x]' eml_sin [ref_fn] [x_min] [x_max] [step]" >&2
  exit 1
fi

DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
PARENT=$(CDPATH= cd -- "$DIR/.." && pwd)
EXPR=$1
NAME=$2

is_number() {
  printf '%s\n' "$1" | grep -Eq '^[+-]?([0-9]+([.][0-9]*)?|[.][0-9]+)([eE][+-]?[0-9]+)?$'
}

infer_ref() {
  fn=$(printf '%s\n' "$1" | sed -E 's/\[.*$//' | tr 'A-Z' 'a-z')
  case "$fn" in
    half) printf '%s\n' "ref_half" ;;
    minus) printf '%s\n' "ref_minus" ;;
    inv) printf '%s\n' "ref_inv" ;;
    sqr) printf '%s\n' "ref_sqr" ;;
    logisticsigmoid) printf '%s\n' "ref_logistic_sigmoid" ;;
    x) printf '%s\n' "ref_identity" ;;
    sin|cos|tan|sinh|cosh|tanh|exp|log|sqrt|asin|acos|atan|asinh|acosh|atanh) printf '%s\n' "$fn" ;;
    arcsin) printf '%s\n' "asin" ;;
    arccos) printf '%s\n' "acos" ;;
    arctan) printf '%s\n' "atan" ;;
    arcsinh) printf '%s\n' "asinh" ;;
    arccosh) printf '%s\n' "acosh" ;;
    arctanh) printf '%s\n' "atanh" ;;
    *) printf '%s\n' "sin" ;;
  esac
}

if [ "$#" -ge 3 ] && ! is_number "$3"; then
  REF=$3
  XMIN=${4:--4.0}
  XMAX=${5:-4.0}
  STEP=${6:-0.0625}
else
  REF=$(infer_ref "$EXPR")
  XMIN=${3:--4.0}
  XMAX=${4:-4.0}
  STEP=${5:-0.0625}
fi

EML_EXPR=$(python3 "$PARENT/eml_compiler_v4.py" "$EXPR")
C_EXPR=$(printf '%s\n' "$EML_EXPR" | tr '[]' '()')
C_EXPR=$(printf '%s\n' "$C_EXPR" | sed 's/EML(/eml(/g')
TMP_C_EXPR=$(mktemp)
trap 'rm -f "$TMP_C_EXPR"' EXIT
printf '%s\n' "$C_EXPR" > "$TMP_C_EXPR"

python3 - "$DIR" "$NAME" "$REF" "$XMIN" "$XMAX" "$STEP" "$EXPR" "$TMP_C_EXPR" <<'PY'
from pathlib import Path
import sys

dir_path = Path(sys.argv[1])
name, ref, xmin, xmax, step, expr, c_expr_path = sys.argv[2:]
c_expr = Path(c_expr_path).read_text(encoding="utf-8").strip()

repl = {
    "@NAME@": name,
    "@REF@": ref,
    "@XMIN@": xmin,
    "@XMAX@": xmax,
    "@STEP@": step,
    "@EXPR@": expr,
    "@C_EXPR@": c_expr,
}

for template_name, output_name in [
    ("eml_math.h.in", "eml_math.h"),
    ("test_eml.c.in", "test_eml.c"),
]:
    text = (dir_path / template_name).read_text(encoding="utf-8")
    for key, value in repl.items():
        text = text.replace(key, value)
    (dir_path / output_name).write_text(text, encoding="utf-8")
PY

if command -v cc >/dev/null 2>&1; then
  if cc -std=c11 -O0 "$DIR/test_eml.c" -o "$DIR/test_eml" >/dev/null 2>&1; then
    :
  else
    cc -std=c11 -O0 "$DIR/test_eml.c" -lm -o "$DIR/test_eml"
  fi
  echo "wrote $DIR/eml_math.h"
  echo "wrote $DIR/test_eml.c"
  echo "built $DIR/test_eml"
else
  echo "wrote $DIR/eml_math.h"
  echo "wrote $DIR/test_eml.c"
fi
