# rust_autogen_parallel_search

Parallel exhaustive Sheffer auto-generation checker using Rayon.

## Requirements

- Rust toolchain with `cargo`
- Internet access on first build to fetch crates (`rayon` and dependencies)
- Windows 11, Linux, or macOS

## Notes before running

- There is no `--help` output implemented in this binary.
- This tool can become expensive for larger `--gen-k`, `--verify-k`, and broad
  generation languages.
- The startup line reports both requested and effective Rayon thread count.

## Tested smoke test (fast)

Run from this directory:

```sh
cargo run --release -- \
  --profiles 0 \
  --gen-k 3 \
  --verify-k 4 \
  --max-keep-per-level 10000 \
  --families S3_Exp_LogBinary \
  --gen-unary Exp \
  --gen-binary Log \
  --rayon-threads 4
```

This command is intentionally constrained and completes quickly while still
exercising parallel profile-0 unary/binary search.

Expected key output lines:

- `Run config: gen_k=3 ... verify_k=4 ... rayon_threads_request=4 ...`
- `Enabled families: ["S3_Exp_LogBinary"]`
- `Profile 0 hits: ...`
- `Done. hits=..., elapsed=...`

## Main flags supported by this binary

All flags from `rust_autogen_search` plus:

- `--rayon-threads <usize>`
  `0` (default) means automatic thread count.

Common flags:

- `--gen-k <usize>`
- `--gen-k-const <usize>`
- `--verify-k <usize>`
- `--max-keep-per-level <usize>`
- `--profile <0|A|B|C>` (repeatable)
- `--profiles <csv>` (e.g. `0,A`)
- `--families <csv>`
- `--disable-families <csv>`
- `--gen-const <csv>`
- `--gen-unary <csv>`
- `--gen-binary <csv>`
- `--gen-ternary <csv>`
- `--allow-partial-deps` or `--require-full-deps`
- `--no-quotient-var-permutations` or `--quotient-var-permutations`

Family names currently available:

- `S1_ExpLog_Subtract`
- `S2_E_Power_Log`
- `S3_Exp_LogBinary`
- `S4_Cosh_ArcCosh_Divide`
- `Wolfram_Mathematica`
