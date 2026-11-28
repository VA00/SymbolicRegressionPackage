Figure 1: Spiral dependency graph of elementary functions from EML
==================================================================

This directory is self-contained. Two source files:

  rust_verify.log        Input data — EML bootstrapping chain (Rust verification)
  gen_spiral_graph.py    Generator script (Python)

Everything else is generated output.


Requirements
------------
  Python 3.10+
  numpy, matplotlib, networkx

  For TikZ compilation: pdflatex with tikz package


Generating the figure
---------------------
  cd Fig_1
  python gen_spiral_graph.py

Outputs (in the same directory):

  graph_spiral.png   Raster (250 dpi, matplotlib, Bezier edges)
  graph_spiral.pdf   Vector (matplotlib, Bezier edges)
  graph_spiral.tex   Standalone TikZ (circular arc edges, compile with pdflatex)

Timestamped copies (e.g. graph_spiral_20260302_131617.png) are also saved
for version tracking.

To compile the TikZ version:

  pdflatex graph_spiral.tex


Custom paths
------------
  python gen_spiral_graph.py <logfile> <output_dir>

Example:
  python gen_spiral_graph.py rust_verify.log ../../../../figures/


Design notes
------------
- Archimedean spiral layout with uniform arc-length spacing
- EML at the center, constant 1 second, then discoveries in bootstrapping order
- Matplotlib: outward-convex quadratic Bezier curves with arrowheads
- TikZ: circular arcs (bend left/right) with Stealth arrows, background layer
- Color scheme:
    Red    = EML (the Sheffer operator)
    Green  = constants and primitives (1, e, -1, 2, pi)
    Gray   = arithmetic operations (unary, binary)
    Blue   = trigonometric functions (cos, sin, tan, arcsin, arccos, arctan)
    Pink   = hyperbolic functions (cosh, sinh, tanh, arsinh, arcosh, artanh)
- Labels use ISO 80000-2 notation for inverse functions
