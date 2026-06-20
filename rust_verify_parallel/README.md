# rust_verify_parallel

Parallel numerical sieve for candidate reconstruction formulas.

This program is intentionally not a verifier. It evaluates expressions at one
numerical substitution, deduplicates approximately equal values, and reports
candidate witnesses. False identities are possible and expected; validate hits
with `rust_verify`, Mathematica, or another symbolic/formal method.

## Build

```sh
cargo build --release
```

## Sinh/ArcSinh search

```sh
cargo run --release -- \
  --constants 0,1 \
  --functions "" \
  --operations SinhArcSinh \
  --max-k 11 \
  --threads 8
```

The two variable placeholders are `EulerGamma` and `Catalan`. Their default
probe values are both real and between 0 and 1. Override them with
`--x-probe` and `--y-probe`.

Available EML-family binary operators include:

- `EML(x,y) = exp(x) - log(y)`
- `EDL(x,y) = exp(x) / log(y)`
- `LDE(x,y) = log(x) / exp(y)`
- `PLI(x,y) = log(x)^(1/y)`
- `PLM(x,y) = log(x)^(-y)`

To isolate multiplication after supplying already recovered primitives:

```sh
cargo run --release -- \
  --constants 0,1,2 \
  --functions Sinh,ArcSinh,Minus \
  --operations SinhArcSinh,Plus,Subtract \
  --target-constants "" \
  --target-functions "" \
  --target-operations Times \
  --max-k 12 \
  --threads 24
```

Useful flags:

- `--threads N`: Rayon thread count; default `4`, and `0` uses the machine default.
- `--ulp N`: target tolerance per component; default `8` ULP.
- `--dedup-ulp N`: numerical signature bucket; default `8` ULP.
- `--max-keep-per-level N`: default `2000000`; `0` means unlimited and can exhaust RAM.
- `--no-bootstrap`: report the first candidate set without promoting hits.

Every reported `Found ...` witness is numerical evidence only.

The default retention limit is intended to keep workstation runs responsive.
Large-memory servers can raise it or set it to `0`, accepting the risk of very
large allocations.
