# rust_verify

Rust re-implementation of the Wolfram Mathematica procedure `VerifyBaseSet` of `SymbolicRegression.m`.

## Requirements

- Rust toolchain with `cargo`
- Windows 11, Linux, or macOS

## The EML chain

Run from this directory:

```sh
cargo run --release -- --constants 1 --functions "" --operations EML --explain
cargo run --release -- --constants 1 --functions "" --operations EML --explain --branch-aware-witness
```

Reference outputs:

- `rust_verify_1.log`: the first command (discovery mode)
- `rust_verify_2.log`: the second command (branch-aware witnesses)

A run should print the same lines as the reference output except the last one  (`Elapsed`).
Times: 35 s and 30 minutes on an AMD Ryzen 9 5900X, 20 s and 14 minutes on an Apple M3 Max.
Without `--explain` option it takes 2 to 4 s.

The witnesses of the chain are checked symbolically in Wolfram Mathematica by
`EML_toolkit/EmL_verification/verify_eml_symbolic_chain.wl`.
