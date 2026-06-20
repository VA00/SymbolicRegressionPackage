use rayon::prelude::*;
use std::collections::{HashMap, HashSet};
use std::env;
use std::f64::consts::{E, PI};
use std::process;
use std::time::Instant;

const EULER_GAMMA: f64 = 0.577_215_664_901_532_9;
const CATALAN: f64 = 0.915_965_594_177_219_0;
const GLAISHER: f64 = 1.282_427_129_100_622_6;

#[derive(Clone, Copy, Debug)]
struct C {
    re: f64,
    im: f64,
}

impl C {
    fn real(x: f64) -> Self {
        Self { re: x, im: 0.0 }
    }

    fn i() -> Self {
        Self { re: 0.0, im: 1.0 }
    }

    fn finite(self) -> bool {
        self.re.is_finite() && self.im.is_finite()
    }

    fn abs(self) -> f64 {
        self.re.hypot(self.im)
    }

    fn arg(self) -> f64 {
        self.im.atan2(self.re)
    }

    fn add(self, b: Self) -> Self {
        Self {
            re: self.re + b.re,
            im: self.im + b.im,
        }
    }

    fn sub(self, b: Self) -> Self {
        Self {
            re: self.re - b.re,
            im: self.im - b.im,
        }
    }

    fn mul(self, b: Self) -> Self {
        Self {
            re: self.re * b.re - self.im * b.im,
            im: self.re * b.im + self.im * b.re,
        }
    }

    fn div(self, b: Self) -> Option<Self> {
        let d = b.re * b.re + b.im * b.im;
        if d == 0.0 {
            return None;
        }
        Some(Self {
            re: (self.re * b.re + self.im * b.im) / d,
            im: (self.im * b.re - self.re * b.im) / d,
        })
    }

    fn neg(self) -> Self {
        Self {
            re: -self.re,
            im: -self.im,
        }
    }

    fn exp(self) -> Self {
        let e = self.re.exp();
        Self {
            re: e * self.im.cos(),
            im: e * self.im.sin(),
        }
    }

    fn ln(self) -> Option<Self> {
        let r = self.abs();
        (r != 0.0).then(|| Self {
            re: r.ln(),
            im: self.arg(),
        })
    }

    fn sqrt(self) -> Self {
        let m = self.abs().sqrt();
        let a = self.arg() * 0.5;
        Self {
            re: m * a.cos(),
            im: m * a.sin(),
        }
    }

    fn sin(self) -> Self {
        Self {
            re: self.re.sin() * self.im.cosh(),
            im: self.re.cos() * self.im.sinh(),
        }
    }

    fn cos(self) -> Self {
        Self {
            re: self.re.cos() * self.im.cosh(),
            im: -self.re.sin() * self.im.sinh(),
        }
    }

    fn sinh(self) -> Self {
        Self {
            re: self.re.sinh() * self.im.cos(),
            im: self.re.cosh() * self.im.sin(),
        }
    }

    fn cosh(self) -> Self {
        Self {
            re: self.re.cosh() * self.im.cos(),
            im: self.re.sinh() * self.im.sin(),
        }
    }

    fn pow(self, b: Self) -> Option<Self> {
        if self.re == 0.0 && self.im == 0.0 {
            if b.re == 0.0 && b.im == 0.0 {
                return Some(Self::real(1.0));
            }
            if b.im == 0.0 && b.re > 0.0 {
                return Some(Self::real(0.0));
            }
            return None;
        }
        Some(b.mul(self.ln()?).exp())
    }

    fn asin(self) -> Option<Self> {
        // Use libm on its real principal branch to avoid extra ULP error from
        // evaluating the equivalent complex log/sqrt formula.
        if self.im == 0.0 && (-1.0..=1.0).contains(&self.re) {
            return Some(Self::real(self.re.asin()));
        }
        // asin(z) = -i log(iz + sqrt(1-z^2))
        let i = Self::i();
        let one = Self::real(1.0);
        let inside = i.mul(self).add(one.sub(self.mul(self)).sqrt());
        let l = inside.ln()?;
        Some(Self {
            re: l.im,
            im: -l.re,
        })
    }

    fn acos(self) -> Option<Self> {
        // Keep real-axis identities accurate under strict ULP comparisons;
        // use the principal complex formula outside this real branch.
        if self.im == 0.0 && (-1.0..=1.0).contains(&self.re) {
            return Some(Self::real(self.re.acos()));
        }
        let a = self.asin()?;
        Some(Self {
            re: PI * 0.5 - a.re,
            im: -a.im,
        })
    }

    fn asinh(self) -> Option<Self> {
        // The log/sqrt representation can lose several ULPs on real inputs.
        if self.im == 0.0 {
            return Some(Self::real(self.re.asinh()));
        }
        self.add(self.mul(self).add(Self::real(1.0)).sqrt()).ln()
    }

    fn acosh(self) -> Option<Self> {
        // Prefer the native real principal branch for ULP-stable comparisons.
        if self.im == 0.0 && self.re >= 1.0 {
            return Some(Self::real(self.re.acosh()));
        }
        self.add(
            self.add(Self::real(1.0))
                .sqrt()
                .mul(self.sub(Self::real(1.0)).sqrt()),
        )
        .ln()
    }

    fn atan(self) -> Option<Self> {
        // Native real evaluation avoids avoidable rounding from complex logs.
        if self.im == 0.0 {
            return Some(Self::real(self.re.atan()));
        }
        // atan(z) = (log(1+iz)-log(1-iz))/(2i)
        let iz = Self::i().mul(self);
        let a = Self::real(1.0).add(iz).ln()?;
        let b = Self::real(1.0).sub(iz).ln()?;
        let d = a.sub(b);
        Some(Self {
            re: d.im * 0.5,
            im: -d.re * 0.5,
        })
    }

    fn atanh(self) -> Option<Self> {
        // Prefer the native real principal branch for ULP-stable comparisons.
        if self.im == 0.0 && self.re.abs() < 1.0 {
            return Some(Self::real(self.re.atanh()));
        }
        let a = Self::real(1.0).add(self).ln()?;
        let b = Self::real(1.0).sub(self).ln()?;
        Some(a.sub(b).mul(Self::real(0.5)))
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
struct Key(u64, u64);

#[derive(Clone, Copy, Debug)]
struct NodeRef {
    level: u16,
    index: u32,
}

#[derive(Clone, Debug)]
enum Expr {
    Atom(String),
    Unary {
        op: &'static str,
        child: NodeRef,
    },
    Binary {
        op: &'static str,
        left: NodeRef,
        right: NodeRef,
    },
}

#[derive(Clone, Debug)]
struct Node {
    value: C,
    expr: Expr,
}

#[derive(Clone, Debug)]
struct Temp {
    value: C,
    expr: Expr,
}

#[derive(Clone, Copy)]
enum Job {
    Unary {
        op: UnaryOp,
        level: usize,
        start: usize,
        end: usize,
    },
    Binary {
        op: BinaryOp,
        left_level: usize,
        right_level: usize,
        start: usize,
        end: usize,
    },
}

#[derive(Clone, Copy)]
struct UnaryOp {
    name: &'static str,
    f: fn(C) -> Option<C>,
}

#[derive(Clone, Copy)]
struct BinaryOp {
    name: &'static str,
    f: fn(C, C) -> Option<C>,
    commutative: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
enum TargetKind {
    Constant,
    Unary,
    Binary,
}

#[derive(Clone, Debug)]
struct Target {
    name: String,
    kind: TargetKind,
    value: C,
}

#[derive(Debug)]
struct Args {
    constants: String,
    functions: String,
    operations: String,
    target_constants: Option<String>,
    target_functions: Option<String>,
    target_operations: Option<String>,
    max_k: usize,
    threads: usize,
    ulp: u64,
    dedup_ulp: u64,
    max_keep_per_level: usize,
    x_probe: f64,
    y_probe: f64,
    no_bootstrap: bool,
}

fn parse_csv(s: &str) -> Vec<String> {
    s.split(',')
        .map(str::trim)
        .filter(|x| !x.is_empty())
        .map(ToOwned::to_owned)
        .collect()
}

fn take_value(argv: &[String], i: &mut usize, flag: &str) -> String {
    *i += 1;
    if *i >= argv.len() {
        eprintln!("Error: {flag} requires a value");
        process::exit(2);
    }
    argv[*i].clone()
}

fn parse_args() -> Args {
    let argv: Vec<String> = env::args().collect();
    let mut a = Args {
        constants: "Pi".to_string(),
        functions: "Exp,Log,Minus".to_string(),
        operations: "Plus".to_string(),
        target_constants: None,
        target_functions: None,
        target_operations: None,
        max_k: 10,
        threads: 4,
        ulp: 8,
        dedup_ulp: 8,
        max_keep_per_level: 2_000_000,
        x_probe: EULER_GAMMA,
        y_probe: CATALAN,
        no_bootstrap: false,
    };
    let mut i = 1;
    while i < argv.len() {
        match argv[i].as_str() {
            "--constants" => a.constants = take_value(&argv, &mut i, "--constants"),
            "--functions" => a.functions = take_value(&argv, &mut i, "--functions"),
            "--operations" => a.operations = take_value(&argv, &mut i, "--operations"),
            "--target-constants" => {
                a.target_constants = Some(take_value(&argv, &mut i, "--target-constants"))
            }
            "--target-functions" => {
                a.target_functions = Some(take_value(&argv, &mut i, "--target-functions"))
            }
            "--target-operations" => {
                a.target_operations = Some(take_value(&argv, &mut i, "--target-operations"))
            }
            "--max-k" => {
                a.max_k = take_value(&argv, &mut i, "--max-k")
                    .parse()
                    .unwrap_or_else(|_| {
                        eprintln!("Error: --max-k requires an integer");
                        process::exit(2);
                    })
            }
            "--threads" | "--rayon-threads" => {
                a.threads = take_value(&argv, &mut i, "--threads")
                    .parse()
                    .unwrap_or_else(|_| {
                        eprintln!("Error: --threads requires an integer");
                        process::exit(2);
                    })
            }
            "--ulp" => {
                a.ulp = take_value(&argv, &mut i, "--ulp")
                    .parse()
                    .unwrap_or_else(|_| {
                        eprintln!("Error: --ulp requires an integer");
                        process::exit(2);
                    })
            }
            "--dedup-ulp" => {
                a.dedup_ulp = take_value(&argv, &mut i, "--dedup-ulp")
                    .parse()
                    .unwrap_or_else(|_| {
                        eprintln!("Error: --dedup-ulp requires an integer");
                        process::exit(2);
                    })
            }
            "--max-keep-per-level" => {
                a.max_keep_per_level = take_value(&argv, &mut i, "--max-keep-per-level")
                    .parse()
                    .unwrap_or_else(|_| {
                        eprintln!("Error: --max-keep-per-level requires an integer");
                        process::exit(2);
                    })
            }
            "--x-probe" => {
                a.x_probe = take_value(&argv, &mut i, "--x-probe")
                    .parse()
                    .unwrap_or_else(|_| {
                        eprintln!("Error: --x-probe requires a number");
                        process::exit(2);
                    })
            }
            "--y-probe" => {
                a.y_probe = take_value(&argv, &mut i, "--y-probe")
                    .parse()
                    .unwrap_or_else(|_| {
                        eprintln!("Error: --y-probe requires a number");
                        process::exit(2);
                    })
            }
            "--no-bootstrap" => a.no_bootstrap = true,
            "--help" | "-h" => {
                print_help();
                process::exit(0);
            }
            x => {
                eprintln!("Error: unknown argument {x}");
                process::exit(2);
            }
        }
        i += 1;
    }
    if a.ulp == 0 || a.dedup_ulp == 0 {
        eprintln!("Error: --ulp and --dedup-ulp must be positive");
        process::exit(2);
    }
    a
}

fn print_help() {
    println!("rust_verify_parallel - parallel numerical formula sieve");
    println!();
    println!("Usage:");
    println!("  rust_verify_parallel [options]");
    println!();
    println!("Language:");
    println!("  --constants CSV");
    println!("  --functions CSV");
    println!("  --operations CSV");
    println!("  --target-constants CSV");
    println!("  --target-functions CSV");
    println!("  --target-operations CSV");
    println!();
    println!("Search:");
    println!("  --max-k N                  Maximum level per bootstrap round");
    println!("  --threads N                Rayon threads (default: 4; 0 means automatic)");
    println!("  --ulp N                    Target tolerance in ULPs (default: 8)");
    println!("  --dedup-ulp N              Signature bucket width in ULPs (default: 8)");
    println!("  --max-keep-per-level N     Retention cap (default: 2000000; 0 unlimited)");
    println!("  --x-probe X                EulerGamma placeholder value");
    println!("  --y-probe Y                Catalan placeholder value");
    println!("  --no-bootstrap             Report hits without promoting them");
    println!();
    println!("The sieve uses one numerical probe and can report false identities.");
}

fn unary_catalog() -> HashMap<&'static str, UnaryOp> {
    [
        UnaryOp {
            name: "Half",
            f: |x| Some(x.mul(C::real(0.5))),
        },
        UnaryOp {
            name: "Minus",
            f: |x| Some(x.neg()),
        },
        UnaryOp {
            name: "Log",
            f: C::ln,
        },
        UnaryOp {
            name: "Exp",
            f: |x| Some(x.exp()),
        },
        UnaryOp {
            name: "Inv",
            f: |x| C::real(1.0).div(x),
        },
        UnaryOp {
            name: "Sqrt",
            f: |x| Some(x.sqrt()),
        },
        UnaryOp {
            name: "Sqr",
            f: |x| Some(x.mul(x)),
        },
        UnaryOp {
            name: "Cosh",
            f: |x| Some(x.cosh()),
        },
        UnaryOp {
            name: "Cos",
            f: |x| Some(x.cos()),
        },
        UnaryOp {
            name: "Sinh",
            f: |x| Some(x.sinh()),
        },
        UnaryOp {
            name: "Sin",
            f: |x| Some(x.sin()),
        },
        UnaryOp {
            name: "Tanh",
            f: |x| x.sinh().div(x.cosh()),
        },
        UnaryOp {
            name: "Tan",
            f: |x| x.sin().div(x.cos()),
        },
        UnaryOp {
            name: "ArcSinh",
            f: C::asinh,
        },
        UnaryOp {
            name: "ArcTanh",
            f: C::atanh,
        },
        UnaryOp {
            name: "ArcSin",
            f: C::asin,
        },
        UnaryOp {
            name: "ArcCos",
            f: C::acos,
        },
        UnaryOp {
            name: "ArcTan",
            f: C::atan,
        },
        UnaryOp {
            name: "ArcCosh",
            f: C::acosh,
        },
        UnaryOp {
            name: "LogisticSigmoid",
            f: |x| C::real(1.0).div(C::real(1.0).add(x.neg().exp())),
        },
    ]
    .into_iter()
    .map(|x| (x.name, x))
    .collect()
}

fn binary_catalog() -> HashMap<&'static str, BinaryOp> {
    [
        BinaryOp {
            name: "Plus",
            f: |a, b| Some(a.add(b)),
            commutative: true,
        },
        BinaryOp {
            name: "Times",
            f: |a, b| Some(a.mul(b)),
            commutative: true,
        },
        BinaryOp {
            name: "Subtract",
            f: |a, b| Some(a.sub(b)),
            commutative: false,
        },
        BinaryOp {
            name: "Divide",
            f: C::div,
            commutative: false,
        },
        BinaryOp {
            name: "Power",
            f: C::pow,
            commutative: false,
        },
        BinaryOp {
            name: "Log",
            f: |base, x| x.ln()?.div(base.ln()?),
            commutative: false,
        },
        BinaryOp {
            name: "Avg",
            f: |a, b| Some(a.add(b).mul(C::real(0.5))),
            commutative: true,
        },
        BinaryOp {
            name: "Hypot",
            f: |a, b| Some(a.mul(a).add(b.mul(b)).sqrt()),
            commutative: true,
        },
        BinaryOp {
            name: "EML",
            f: |a, b| Some(a.exp().sub(b.ln()?)),
            commutative: false,
        },
        BinaryOp {
            name: "EDL",
            f: |a, b| a.exp().div(b.ln()?),
            commutative: false,
        },
        BinaryOp {
            name: "LDE",
            f: |a, b| a.ln()?.div(b.exp()),
            commutative: false,
        },
        BinaryOp {
            name: "PLI",
            f: |a, b| a.ln()?.pow(C::real(1.0).div(b)?),
            commutative: false,
        },
        BinaryOp {
            name: "PLM",
            f: |a, b| a.ln()?.pow(b.neg()),
            commutative: false,
        },
        BinaryOp {
            name: "CosArcCos",
            f: |a, b| Some(a.cos().sub(b.acos()?)),
            commutative: false,
        },
        BinaryOp {
            name: "SinhArcSinh",
            f: |a, b| Some(a.sinh().sub(b.asinh()?)),
            commutative: false,
        },
    ]
    .into_iter()
    .map(|x| (x.name, x))
    .collect()
}

fn constant_value(name: &str, x: f64, y: f64) -> Option<C> {
    match name {
        "EulerGamma" => Some(C::real(x)),
        "Catalan" => Some(C::real(y)),
        "Glaisher" => Some(C::real(GLAISHER)),
        "0" => Some(C::real(0.0)),
        "1" => Some(C::real(1.0)),
        "-1" => Some(C::real(-1.0)),
        "2" => Some(C::real(2.0)),
        "3" => Some(C::real(3.0)),
        "E" => Some(C::real(E)),
        "Pi" => Some(C::real(PI)),
        "I" => Some(C::i()),
        _ => None,
    }
}

fn ordered_f64_bits(x: f64) -> u64 {
    let bits = x.to_bits();
    if bits & (1_u64 << 63) != 0 {
        !bits
    } else {
        bits | (1_u64 << 63)
    }
}

fn ulp_distance_f64(a: f64, b: f64) -> Option<u64> {
    if !a.is_finite() || !b.is_finite() {
        return None;
    }
    Some(ordered_f64_bits(a).abs_diff(ordered_f64_bits(b)))
}

fn key(v: C, dedup_ulp: u64) -> Option<Key> {
    if !v.finite() {
        return None;
    }
    let width = dedup_ulp.saturating_add(1);
    Some(Key(
        ordered_f64_bits(v.re) / width,
        ordered_f64_bits(v.im) / width,
    ))
}

fn near(a: C, b: C, ulp: u64) -> bool {
    ulp_distance_f64(a.re, b.re).is_some_and(|d| d <= ulp)
        && ulp_distance_f64(a.im, b.im).is_some_and(|d| d <= ulp)
}

fn render(r: NodeRef, levels: &[Vec<Node>]) -> String {
    let node = &levels[usize::from(r.level)][r.index as usize];
    match &node.expr {
        Expr::Atom(x) => x.clone(),
        Expr::Unary { op, child } => format!("{op}[{}]", render(*child, levels)),
        Expr::Binary { op, left, right } => {
            format!(
                "{op}[{}, {}]",
                render(*left, levels),
                render(*right, levels)
            )
        }
    }
}

fn merge_map_limited(
    mut a: HashMap<Key, Temp>,
    mut b: HashMap<Key, Temp>,
    limit: usize,
) -> HashMap<Key, Temp> {
    if a.len() < b.len() {
        std::mem::swap(&mut a, &mut b);
    }
    if limit > 0 && a.len() > limit {
        a = a.into_iter().take(limit).collect();
    }
    if limit > 0 && a.len() >= limit {
        return a;
    }
    for (k, v) in b {
        a.entry(k).or_insert(v);
        if limit > 0 && a.len() >= limit {
            break;
        }
    }
    a
}

fn run_job(job: Job, levels: &[Vec<Node>], dedup_ulp: u64) -> HashMap<Key, Temp> {
    let mut out = HashMap::new();
    match job {
        Job::Unary {
            op,
            level,
            start,
            end,
        } => {
            for i in start..end {
                if let Some(v) = (op.f)(levels[level][i].value) {
                    if let Some(k) = key(v, dedup_ulp) {
                        out.entry(k).or_insert_with(|| Temp {
                            value: v,
                            expr: Expr::Unary {
                                op: op.name,
                                child: NodeRef {
                                    level: u16::try_from(level)
                                        .expect("expression level exceeds u16"),
                                    index: u32::try_from(i)
                                        .expect("level contains more than u32::MAX nodes"),
                                },
                            },
                        });
                    }
                }
            }
        }
        Job::Binary {
            op,
            left_level,
            right_level,
            start,
            end,
        } => {
            let left = &levels[left_level];
            let right = &levels[right_level];
            let right_len = right.len();
            for flat in start..end {
                let i = flat / right_len;
                let j = flat % right_len;
                if op.commutative && left_level == right_level && i > j {
                    continue;
                }
                if let Some(v) = (op.f)(left[i].value, right[j].value) {
                    if let Some(k) = key(v, dedup_ulp) {
                        out.entry(k).or_insert_with(|| Temp {
                            value: v,
                            expr: Expr::Binary {
                                op: op.name,
                                left: NodeRef {
                                    level: u16::try_from(left_level)
                                        .expect("expression level exceeds u16"),
                                    index: u32::try_from(i)
                                        .expect("level contains more than u32::MAX nodes"),
                                },
                                right: NodeRef {
                                    level: u16::try_from(right_level)
                                        .expect("expression level exceeds u16"),
                                    index: u32::try_from(j)
                                        .expect("level contains more than u32::MAX nodes"),
                                },
                            },
                        });
                    }
                }
            }
        }
    }
    out
}

fn generate_level(
    k: usize,
    levels: &[Vec<Node>],
    unary_ops: &[UnaryOp],
    binary_ops: &[BinaryOp],
    dedup_ulp: u64,
    max_keep_per_level: usize,
) -> (HashMap<Key, Temp>, usize, bool) {
    const CHUNK_SIZE: usize = 1 << 15;
    let mut jobs = Vec::new();

    if !levels[k - 1].is_empty() {
        for &op in unary_ops {
            let len = levels[k - 1].len();
            for start in (0..len).step_by(CHUNK_SIZE) {
                jobs.push(Job::Unary {
                    op,
                    level: k - 1,
                    start,
                    end: (start + CHUNK_SIZE).min(len),
                });
            }
        }
    }

    for &op in binary_ops {
        for left_k in 1..k - 1 {
            let right_k = k - 1 - left_k;
            if op.commutative && left_k > right_k {
                continue;
            }
            if levels[left_k].is_empty() || levels[right_k].is_empty() {
                continue;
            }
            let total = levels[left_k].len().saturating_mul(levels[right_k].len());
            for start in (0..total).step_by(CHUNK_SIZE) {
                jobs.push(Job::Binary {
                    op,
                    left_level: left_k,
                    right_level: right_k,
                    start,
                    end: (start + CHUNK_SIZE).min(total),
                });
            }
        }
    }

    let job_count = jobs.len();
    let map = jobs
        .into_par_iter()
        .map(|job| run_job(job, levels, dedup_ulp))
        .reduce(HashMap::new, |a, b| {
            merge_map_limited(a, b, max_keep_per_level)
        });
    let cap_reached = max_keep_per_level > 0 && map.len() >= max_keep_per_level;
    (map, job_count, cap_reached)
}

fn make_targets(
    constants: &[String],
    unary: &[String],
    binary: &[String],
    uc: &HashMap<&str, UnaryOp>,
    bc: &HashMap<&str, BinaryOp>,
    x: f64,
    y: f64,
) -> Vec<Target> {
    let mut out = Vec::new();
    for name in constants {
        let Some(value) = constant_value(name, x, y) else {
            eprintln!("Error: unknown target constant {name}");
            process::exit(2);
        };
        out.push(Target {
            name: name.clone(),
            kind: TargetKind::Constant,
            value,
        });
    }
    for name in unary {
        let Some(op) = uc.get(name.as_str()) else {
            eprintln!("Error: unknown target function {name}");
            process::exit(2);
        };
        let Some(value) = (op.f)(C::real(x)) else {
            continue;
        };
        out.push(Target {
            name: name.clone(),
            kind: TargetKind::Unary,
            value,
        });
    }
    for name in binary {
        let Some(op) = bc.get(name.as_str()) else {
            eprintln!("Error: unknown target operation {name}");
            process::exit(2);
        };
        let Some(value) = (op.f)(C::real(x), C::real(y)) else {
            continue;
        };
        out.push(Target {
            name: name.clone(),
            kind: TargetKind::Binary,
            value,
        });
    }
    out
}

fn target_defaults() -> (Vec<String>, Vec<String>, Vec<String>) {
    (
        parse_csv("-1,1,2,E,Pi"),
        parse_csv(
            "Half,Minus,Log,Exp,Inv,Sqrt,Sqr,Cosh,Cos,Sinh,Sin,Tanh,Tan,ArcSinh,ArcTanh,ArcSin,ArcCos,ArcTan,ArcCosh,LogisticSigmoid",
        ),
        parse_csv("Plus,Times,Subtract,Divide,Power,Log,Avg,Hypot"),
    )
}

fn names_for_kind(targets: &[Target], kind: TargetKind) -> Vec<String> {
    let mut names: Vec<String> = targets
        .iter()
        .filter(|target| target.kind == kind)
        .map(|target| target.name.clone())
        .collect();
    names.sort();
    names
}

fn print_remaining(targets: &[Target]) {
    println!(
        "Remaining constants: {:?}",
        names_for_kind(targets, TargetKind::Constant)
    );
    println!(
        "Remaining unary: {:?}",
        names_for_kind(targets, TargetKind::Unary)
    );
    println!(
        "Remaining binary: {:?}",
        names_for_kind(targets, TargetKind::Binary)
    );
    println!("Remaining ternary: []");
}

fn displayed_constants(constants: &[String]) -> Vec<String> {
    let mut names = constants.to_vec();
    names.push("EulerGamma".to_string());
    names.push("Catalan".to_string());
    names.sort();
    names.dedup();
    names
}

fn build_initial_levels(
    constants: &[String],
    x: f64,
    y: f64,
    dedup_ulp: u64,
) -> (Vec<Vec<Node>>, HashSet<Key>) {
    let mut names = constants.to_vec();
    names.push("EulerGamma".to_string());
    names.push("Catalan".to_string());
    names.sort();
    names.dedup();

    let mut level = Vec::new();
    let mut seen = HashSet::new();
    for name in names {
        let Some(value) = constant_value(&name, x, y) else {
            eprintln!("Error: unknown constant {name}");
            process::exit(2);
        };
        let Some(k) = key(value, dedup_ulp) else {
            continue;
        };
        if seen.insert(k) {
            level.push(Node {
                value,
                expr: Expr::Atom(name),
            });
        }
    }
    (vec![Vec::new(), level], seen)
}

fn find_hits(
    level_index: usize,
    levels: &[Vec<Node>],
    targets: &[Target],
    ulp: u64,
) -> Vec<(usize, NodeRef)> {
    let nodes = &levels[level_index];
    targets
        .par_iter()
        .enumerate()
        .filter_map(|(ti, target)| {
            nodes
                .iter()
                .position(|n| near(n.value, target.value, ulp))
                .map(|index| {
                    (
                        ti,
                        NodeRef {
                            level: u16::try_from(level_index)
                                .expect("expression level exceeds u16"),
                            index: u32::try_from(index)
                                .expect("level contains more than u32::MAX nodes"),
                        },
                    )
                })
        })
        .collect()
}

fn main() {
    let args = parse_args();
    if args.threads > 0 {
        rayon::ThreadPoolBuilder::new()
            .num_threads(args.threads)
            .build_global()
            .expect("failed to initialize Rayon thread pool");
    }

    let unary_catalog = unary_catalog();
    let binary_catalog = binary_catalog();
    let mut constants = parse_csv(&args.constants);
    let mut unary_names = parse_csv(&args.functions);
    let mut binary_names = parse_csv(&args.operations);
    constants.sort();
    constants.dedup();
    unary_names.sort();
    unary_names.dedup();
    binary_names.sort();
    binary_names.dedup();

    for name in &unary_names {
        if !unary_catalog.contains_key(name.as_str()) {
            eprintln!("Error: unknown function {name}");
            process::exit(2);
        }
    }
    for name in &binary_names {
        if !binary_catalog.contains_key(name.as_str()) {
            eprintln!("Error: unknown operation {name}");
            process::exit(2);
        }
    }

    let defaults = target_defaults();
    let target_constant_names = args
        .target_constants
        .as_deref()
        .map(parse_csv)
        .unwrap_or(defaults.0);
    let target_unary_names = args
        .target_functions
        .as_deref()
        .map(parse_csv)
        .unwrap_or(defaults.1);
    let target_binary_names = args
        .target_operations
        .as_deref()
        .map(parse_csv)
        .unwrap_or(defaults.2);

    let all_targets = make_targets(
        &target_constant_names,
        &target_unary_names,
        &target_binary_names,
        &unary_catalog,
        &binary_catalog,
        args.x_probe,
        args.y_probe,
    );
    let mut remaining: Vec<Target> = all_targets
        .into_iter()
        .filter(|t| match t.kind {
            TargetKind::Constant => !constants.contains(&t.name),
            TargetKind::Unary => !unary_names.contains(&t.name),
            TargetKind::Binary => !binary_names.contains(&t.name),
        })
        .collect();

    println!(
        "Loaded base constants: {:?}",
        displayed_constants(&constants)
    );
    println!("Loaded base unary functions: {unary_names:?}");
    println!("Loaded base binary operations: {binary_names:?}");
    println!("Loaded base ternary operations: []");
    println!("Target constants: {target_constant_names:?}");
    println!("Target unary functions: {target_unary_names:?}");
    println!("Target binary operations: {target_binary_names:?}");
    println!("Target ternary operations: []");
    println!(
        "Parallel sieve: threads={}, ULP tolerance={}, dedup bucket={} ULP, max keep per level={}.",
        rayon::current_num_threads(),
        args.ulp,
        args.dedup_ulp,
        if args.max_keep_per_level == 0 {
            "unlimited".to_string()
        } else {
            args.max_keep_per_level.to_string()
        }
    );
    println!(
        "Probe placeholders: EulerGamma={:.17}, Catalan={:.17}",
        args.x_probe, args.y_probe
    );
    print_remaining(&remaining);

    let total_start = Instant::now();
    loop {
        let unary_ops: Vec<UnaryOp> = unary_names
            .iter()
            .map(|n| unary_catalog[n.as_str()])
            .collect();
        let binary_ops: Vec<BinaryOp> = binary_names
            .iter()
            .map(|n| binary_catalog[n.as_str()])
            .collect();
        let (mut levels, mut seen) =
            build_initial_levels(&constants, args.x_probe, args.y_probe, args.dedup_ulp);
        levels.resize_with(args.max_k + 1, Vec::new);

        let mut promoted: Vec<(Target, String, usize)> = Vec::new();
        println!("Testing with K = 1");
        println!("No new items at K = 1 (level contains only known constants)");
        for k in 2..=args.max_k {
            println!("Testing with K = {k}");
            let level_start = Instant::now();
            let (mut current, job_count, cap_reached) = generate_level(
                k,
                &levels,
                &unary_ops,
                &binary_ops,
                args.dedup_ulp,
                args.max_keep_per_level,
            );

            current.retain(|key, _| !seen.contains(key));
            let mut entries: Vec<(Key, Temp)> = current.into_iter().collect();
            if args.max_keep_per_level > 0 && entries.len() > args.max_keep_per_level {
                entries.truncate(args.max_keep_per_level);
            }
            let kept = entries.len();
            let mut level = Vec::with_capacity(kept);
            for (key, temp) in entries {
                seen.insert(key);
                level.push(Node {
                    value: temp.value,
                    expr: temp.expr,
                });
            }
            levels[k] = level;

            let hits = find_hits(k, &levels, &remaining, args.ulp);
            if !hits.is_empty() {
                let mut hit_target_indices = HashSet::new();
                for (target_index, node_ref) in hits {
                    if !hit_target_indices.insert(target_index) {
                        continue;
                    }
                    let target = remaining[target_index].clone();
                    let witness = render(node_ref, &levels);
                    match target.kind {
                        TargetKind::Constant => println!("Found constant: {}", target.name),
                        TargetKind::Unary => println!("Found unary function: {}", target.name),
                        TargetKind::Binary => {
                            println!("Found binary operation: {}", target.name)
                        }
                    }
                    println!("  witness[k={k}]: {witness}");
                    promoted.push((target, witness, k));
                }
                break;
            }
            println!(
                "No new items at K = {k} (kept={kept}, seen={}, jobs={job_count}, elapsed={:.2?}{})",
                seen.len(),
                level_start.elapsed(),
                if cap_reached { ", retention cap reached" } else { "" }
            );
        }

        if promoted.is_empty() {
            break;
        }

        let promoted_names: HashSet<(TargetKind, String)> = promoted
            .iter()
            .map(|(t, _, _)| (t.kind, t.name.clone()))
            .collect();
        remaining.retain(|t| !promoted_names.contains(&(t.kind, t.name.clone())));
        print_remaining(&remaining);

        if args.no_bootstrap {
            println!("Bootstrap disabled; stopping after candidate report.");
            break;
        }
        for (target, _, _) in promoted {
            match target.kind {
                TargetKind::Constant => constants.push(target.name),
                TargetKind::Unary => unary_names.push(target.name),
                TargetKind::Binary => binary_names.push(target.name),
            }
        }
        constants.sort();
        constants.dedup();
        unary_names.sort();
        unary_names.dedup();
        binary_names.sort();
        binary_names.dedup();

        if remaining.is_empty() {
            break;
        }
    }

    println!("Known constants: {:?}", displayed_constants(&constants));
    println!("Known unary functions: {unary_names:?}");
    println!("Known binary operations: {binary_names:?}");
    println!("Known ternary operations: []");
    println!("Target constants: {target_constant_names:?}");
    println!("Target unary functions: {target_unary_names:?}");
    println!("Target binary operations: {target_binary_names:?}");
    println!("Target ternary operations: []");
    print_remaining(&remaining);
    println!("Elapsed: {:.2?}", total_start.elapsed());
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn real_asinh_sinh_round_trip() {
        for x in [-3.0, -0.5, 0.0, EULER_GAMMA, CATALAN, 4.0] {
            let z = C::real(x).sinh().asinh().unwrap();
            assert!(near(z, C::real(x), 16));
        }
    }

    #[test]
    fn sinh_arc_sinh_recovers_sinh_at_zero_seed() {
        let catalog = binary_catalog();
        let op = catalog["SinhArcSinh"];
        let x = C::real(EULER_GAMMA);
        let got = (op.f)(x, C::real(0.0)).unwrap();
        assert!(near(got, x.sinh(), 16));
    }

    #[test]
    fn ulp_key_rejects_non_finite_values() {
        assert!(key(C::real(f64::INFINITY), 8).is_none());
        assert!(key(C::real(f64::NAN), 8).is_none());
    }

    #[test]
    fn reported_cos_arc_cos_near_miss_is_rejected() {
        let binary = binary_catalog();
        let x = C::real(EULER_GAMMA);
        let y = C::real(CATALAN);
        let left = x.cos().sub(x);
        let right = y.sub(C::real(1.0));
        let false_candidate = (binary["CosArcCos"].f)(left, right)
            .unwrap()
            .neg()
            .asin()
            .unwrap();
        let target = x.sqrt();
        assert!(!near(false_candidate, target, 8));
    }

    #[test]
    fn eml_variants_match_their_definitions() {
        let binary = binary_catalog();
        let x = C::real(2.0);
        let y = C::real(3.0);
        assert!(near(
            (binary["EDL"].f)(x, y).unwrap(),
            x.exp().div(y.ln().unwrap()).unwrap(),
            1
        ));
        assert!(near(
            (binary["LDE"].f)(x, y).unwrap(),
            x.ln().unwrap().div(y.exp()).unwrap(),
            1
        ));
        assert!(near(
            (binary["PLI"].f)(x, y).unwrap(),
            x.ln().unwrap().pow(C::real(1.0).div(y).unwrap()).unwrap(),
            1
        ));
        assert!(near(
            (binary["PLM"].f)(x, y).unwrap(),
            x.ln().unwrap().pow(y.neg()).unwrap(),
            1
        ));
    }
}
