# Figure 2: EML trees

This directory contains the script that generates the standalone LaTeX/TikZ
source for Fig. 2 in the PNAS package.

## Files

- `fig2_trees.py`: generator script
- `Fig2_trees.tex`: generated standalone LaTeX file
- `Fig2_trees.pdf`: generated PDF after LaTeX compilation

## Requirements

- Python 3.10+
- `numpy`
- `scipy`
- Optional for PDF compilation: `pdflatex` with `standalone`, `tikz`,
  `amsmath`, `amssymb`, and `newtx`

## Generate the LaTeX source

Run in this directory:

```powershell
python3 fig2_trees.py
```

This writes `Fig2_trees.tex` in the same directory.


## Compile the PDF

```powershell
pdflatex  Fig2_trees.tex
```
