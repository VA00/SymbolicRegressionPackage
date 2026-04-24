# EML symbolic verification

This directory contains the Wolfram Language materials used to verify the
bootstrapped EML reconstruction chain.

## Files

- `verify_eml_symbolic_chain.wl`: headless Wolfram Language script for the
  witness chain generated from `rust_verify` and stored in `rust_verify.log`
- `EML_verify.nb`: minimal notebook that imports `SymbolicRegression.m` and
  reruns `VerifyBaseSet[{1}, {}, {EML}]`
- `EML_verify_evaluated.nb`: notebook with outputs already saved
- `mathematica_verify.log`: plain-text record of the discovery verification run
- `rust_verify.log`: Rust-side witness chain used as symbolic input
- `Symbolic_Simplification_Discovery_Chain.nb` symbolic simplification of formulas from `rust_verify.log` 

## Requirements

- Wolfram Mathematica 14.3 or compatible Wolfram Engine / WolframScript
- Windows 11 PowerShell 7, Linux bash, or macOS zsh
- For the notebook workflow, a licensed Mathematica front end is recommended

On a fresh machine, you may need to activate or sign in to the Wolfram
installation once before `wolframscript` can evaluate code headlessly.

## Notebook rerun

Open `EML_verify.nb` in Mathematica and evaluate the three input cells with
`Shift+Enter`, or use `Evaluation -> Evaluate Notebook`.

This notebook imports `SymbolicRegression.m` from the public repository and
runs:

```wolfram
VerifyBaseSet[{1}, {}, {EML}]
```

## Headless rerun

From this directory:

```sh
wolframscript -file verify_eml_symbolic_chain.wl
```

If `wolframscript` is not on `PATH` on macOS, use:

```sh
/Applications/Wolfram.app/Contents/MacOS/wolframscript -file verify_eml_symbolic_chain.wl
```

The script checks the accepted symbolic witnesses and prints `PASS`/`FAIL`
lines for the identities.

## Reference outputs

- `mathematica_verify.log` is the plain-text reference log for readers who do
  not have local Mathematica access.
- `EML_verify_evaluated.nb` is the notebook form of the same accepted run.
