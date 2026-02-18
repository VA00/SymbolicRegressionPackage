use std::collections::{HashMap, HashSet};
use std::env;
use std::f64::consts::{FRAC_PI_2, PI};
use std::process;
use std::time::Instant;

const GLAISHER: f64 = 1.282_427_129_100_622_6;
const EULER_GAMMA: f64 = 0.577_215_664_901_532_9;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum EquivMode {
    Rel,
    Ulp,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum DomainMode {
    Complex,
    Real,
}

#[derive(Clone, Copy, Debug)]
struct EquivCfg {
    mode: EquivMode,
    eps: f64,
    ulp_tol: u64,
}

#[derive(Debug)]
struct Args {
    constants: String,
    functions: String,
    operations: String,
    ternary: String,
    target_constants: Option<String>,
    target_functions: Option<String>,
    target_operations: Option<String>,
    target_ternary: Option<String>,
    max_k: usize,
    explain: bool,
    equiv: EquivCfg,
    domain: DomainMode,
}

#[derive(Clone, Copy, Debug)]
struct C {
    re: f64,
    im: f64,
}

#[derive(Clone)]
struct Unary {
    f: fn(C) -> Option<C>,
}

#[derive(Clone)]
struct Binary {
    f: fn(C, C) -> Option<C>,
    commutative: bool,
}

#[derive(Clone)]
struct Ternary {
    f: fn(C, C, C) -> Option<C>,
}

impl C {
    fn new(re: f64, im: f64) -> Self {
        Self { re, im }
    }

    fn real(x: f64) -> Self {
        Self { re: x, im: 0.0 }
    }

    fn i() -> Self {
        Self { re: 0.0, im: 1.0 }
    }

    fn is_finite(self) -> bool {
        self.re.is_finite() && self.im.is_finite()
    }

    fn abs(self) -> f64 {
        self.re.hypot(self.im)
    }

    fn arg(self) -> f64 {
        self.im.atan2(self.re)
    }

    fn add(self, other: C) -> C {
        C::new(self.re + other.re, self.im + other.im)
    }

    fn sub(self, other: C) -> C {
        C::new(self.re - other.re, self.im - other.im)
    }

    fn mul(self, other: C) -> C {
        C::new(
            self.re * other.re - self.im * other.im,
            self.re * other.im + self.im * other.re,
        )
    }

    fn div(self, other: C) -> Option<C> {
        let den = other.re * other.re + other.im * other.im;
        if den == 0.0 {
            return None;
        }
        Some(C::new(
            (self.re * other.re + self.im * other.im) / den,
            (self.im * other.re - self.re * other.im) / den,
        ))
    }

    fn neg(self) -> C {
        C::new(-self.re, -self.im)
    }

    fn exp(self) -> C {
        let ea = self.re.exp();
        C::new(ea * self.im.cos(), ea * self.im.sin())
    }

    fn ln(self) -> Option<C> {
        let r = self.abs();
        if r == 0.0 {
            return None;
        }
        Some(C::new(r.ln(), self.arg()))
    }

    fn sqrt(self) -> C {
        let r = self.abs();
        let t = self.arg() / 2.0;
        let m = r.sqrt();
        C::new(m * t.cos(), m * t.sin())
    }

    fn sin(self) -> C {
        C::new(
            self.re.sin() * self.im.cosh(),
            self.re.cos() * self.im.sinh(),
        )
    }

    fn cos(self) -> C {
        C::new(
            self.re.cos() * self.im.cosh(),
            -self.re.sin() * self.im.sinh(),
        )
    }

    fn tan(self) -> Option<C> {
        self.sin().div(self.cos())
    }

    fn sinh(self) -> C {
        C::new(
            self.re.sinh() * self.im.cos(),
            self.re.cosh() * self.im.sin(),
        )
    }

    fn cosh(self) -> C {
        C::new(
            self.re.cosh() * self.im.cos(),
            self.re.sinh() * self.im.sin(),
        )
    }

    fn tanh(self) -> Option<C> {
        self.sinh().div(self.cosh())
    }

    fn pow(self, w: C) -> Option<C> {
        if self.re == 0.0 && self.im == 0.0 {
            if w.re == 0.0 && w.im == 0.0 {
                return Some(C::real(1.0));
            }
            if w.im == 0.0 && w.re > 0.0 {
                return Some(C::real(0.0));
            }
            return None;
        }
        let l = self.ln()?;
        Some(w.mul(l).exp())
    }

    fn asin(self) -> Option<C> {
        let i = C::i();
        let one = C::real(1.0);
        let iz = i.mul(self);
        let inside = one.sub(self.mul(self)).sqrt();
        let ln_arg = iz.add(inside);
        let ln_val = ln_arg.ln()?;
        Some(i.neg().mul(ln_val))
    }

    fn acos(self) -> Option<C> {
        Some(C::real(FRAC_PI_2).sub(self.asin()?))
    }

    fn atan(self) -> Option<C> {
        let i = C::i();
        let one = C::real(1.0);
        let l1 = one.sub(i.mul(self)).ln()?;
        let l2 = one.add(i.mul(self)).ln()?;
        let diff = l1.sub(l2);
        Some(C::new(0.0, 0.5).mul(diff))
    }

    fn asinh(self) -> Option<C> {
        let one = C::real(1.0);
        self.add(self.mul(self).add(one).sqrt()).ln()
    }

    fn acosh(self) -> Option<C> {
        let one = C::real(1.0);
        let term = self.add(one).sqrt().mul(self.sub(one).sqrt());
        self.add(term).ln()
    }

    fn atanh(self) -> Option<C> {
        let one = C::real(1.0);
        let num = one.add(self).ln()?;
        let den = one.sub(self).ln()?;
        Some(num.sub(den).mul(C::real(0.5)))
    }
}

fn parse_args() -> Args {
    let mut args = Args {
        constants: "Pi".to_string(),
        functions: "Exp,Log,Minus".to_string(),
        operations: "Plus".to_string(),
        ternary: "".to_string(),
        target_constants: None,
        target_functions: None,
        target_operations: None,
        target_ternary: None,
        max_k: 10,
        explain: false,
        equiv: EquivCfg {
            mode: EquivMode::Rel,
            eps: 16.0 * f64::EPSILON,
            ulp_tol: 4,
        },
        domain: DomainMode::Complex,
    };

    let mut it = env::args().skip(1);
    while let Some(flag) = it.next() {
        match flag.as_str() {
            "--constants" => {
                if let Some(v) = it.next() {
                    args.constants = v;
                }
            }
            "--functions" => {
                if let Some(v) = it.next() {
                    args.functions = v;
                }
            }
            "--operations" => {
                if let Some(v) = it.next() {
                    args.operations = v;
                }
            }
            "--ternary" => {
                if let Some(v) = it.next() {
                    args.ternary = v;
                }
            }
            "--target-constants" => {
                if let Some(v) = it.next() {
                    args.target_constants = Some(v);
                }
            }
            "--target-functions" => {
                if let Some(v) = it.next() {
                    args.target_functions = Some(v);
                }
            }
            "--target-operations" => {
                if let Some(v) = it.next() {
                    args.target_operations = Some(v);
                }
            }
            "--target-ternary" => {
                if let Some(v) = it.next() {
                    args.target_ternary = Some(v);
                }
            }
            "--max-k" => {
                if let Some(v) = it.next() {
                    if let Ok(n) = v.parse::<usize>() {
                        args.max_k = n;
                    }
                }
            }
            "--eps" => {
                if let Some(v) = it.next() {
                    if let Ok(x) = v.parse::<f64>() {
                        if x.is_finite() && x >= 0.0 {
                            args.equiv.eps = x;
                        }
                    }
                }
            }
            "--equiv" => {
                if let Some(v) = it.next() {
                    args.equiv.mode = match v.as_str() {
                        "ulp" | "ULP" => EquivMode::Ulp,
                        _ => EquivMode::Rel,
                    };
                }
            }
            "--ulp" => {
                if let Some(v) = it.next() {
                    if let Ok(x) = v.parse::<u64>() {
                        args.equiv.ulp_tol = x;
                    }
                }
            }
            "--domain" => {
                if let Some(v) = it.next() {
                    args.domain = match v.as_str() {
                        "real" | "REAL" => DomainMode::Real,
                        _ => DomainMode::Complex,
                    };
                }
            }
            "--explain" => {
                args.explain = true;
            }
            _ => {}
        }
    }
    args
}

fn qkey(v: C) -> (i64, i64) {
    ((v.re * 1e12).round() as i64, (v.im * 1e12).round() as i64)
}

fn ulp_distance_f64(a: f64, b: f64) -> Option<u64> {
    if !a.is_finite() || !b.is_finite() {
        return None;
    }
    let ia = a.to_bits() as i64;
    let ib = b.to_bits() as i64;
    let oa = if ia < 0 { i64::MIN - ia } else { ia };
    let ob = if ib < 0 { i64::MIN - ib } else { ib };
    Some((oa as i128 - ob as i128).unsigned_abs() as u64)
}

fn near(a: C, b: C, equiv: EquivCfg) -> bool {
    match equiv.mode {
        EquivMode::Rel => a.sub(b).abs() <= equiv.eps * (1.0 + a.abs() + b.abs()),
        EquivMode::Ulp => {
            let Some(dre) = ulp_distance_f64(a.re, b.re) else {
                return false;
            };
            let Some(dim) = ulp_distance_f64(a.im, b.im) else {
                return false;
            };
            dre <= equiv.ulp_tol && dim <= equiv.ulp_tol
        }
    }
}

fn imag_is_zero(v: C, equiv: EquivCfg) -> bool {
    match equiv.mode {
        EquivMode::Rel => v.im.abs() <= equiv.eps * (1.0 + v.re.abs() + v.im.abs()),
        EquivMode::Ulp => ulp_distance_f64(v.im, 0.0).is_some_and(|d| d <= equiv.ulp_tol),
    }
}

fn value_ok(v: C, domain: DomainMode, equiv: EquivCfg) -> bool {
    if !v.is_finite() {
        return false;
    }
    match domain {
        DomainMode::Complex => true,
        DomainMode::Real => imag_is_zero(v, equiv),
    }
}

fn logistic_sigmoid(z: C) -> Option<C> {
    C::real(1.0).div(C::real(1.0).add(z.neg().exp()))
}

fn unary_catalog() -> HashMap<&'static str, Unary> {
    [
        (
            "Half",
            Unary {
                f: |x| Some(x.mul(C::real(0.5))),
            },
        ),
        (
            "Minus",
            Unary {
                f: |x| Some(x.neg()),
            },
        ),
        ("Log", Unary { f: |x| x.ln() }),
        (
            "Exp",
            Unary {
                f: |x| Some(x.exp()),
            },
        ),
        (
            "Inv",
            Unary {
                f: |x| C::real(1.0).div(x),
            },
        ),
        (
            "Sqrt",
            Unary {
                f: |x| Some(x.sqrt()),
            },
        ),
        (
            "Sqr",
            Unary {
                f: |x| Some(x.mul(x)),
            },
        ),
        (
            "Cosh",
            Unary {
                f: |x| Some(x.cosh()),
            },
        ),
        (
            "Cos",
            Unary {
                f: |x| Some(x.cos()),
            },
        ),
        (
            "Sinh",
            Unary {
                f: |x| Some(x.sinh()),
            },
        ),
        (
            "Sin",
            Unary {
                f: |x| Some(x.sin()),
            },
        ),
        ("Tanh", Unary { f: |x| x.tanh() }),
        ("Tan", Unary { f: |x| x.tan() }),
        ("ArcSinh", Unary { f: |x| x.asinh() }),
        ("ArcTanh", Unary { f: |x| x.atanh() }),
        ("ArcSin", Unary { f: |x| x.asin() }),
        ("ArcCos", Unary { f: |x| x.acos() }),
        ("ArcTan", Unary { f: |x| x.atan() }),
        ("ArcCosh", Unary { f: |x| x.acosh() }),
        (
            "LogisticSigmoid",
            Unary {
                f: |x| logistic_sigmoid(x),
            },
        ),
    ]
    .into_iter()
    .collect()
}

fn binary_catalog() -> HashMap<&'static str, Binary> {
    [
        (
            "Plus",
            Binary {
                f: |a, b| Some(a.add(b)),
                commutative: true,
            },
        ),
        (
            "Times",
            Binary {
                f: |a, b| Some(a.mul(b)),
                commutative: true,
            },
        ),
        (
            "Subtract",
            Binary {
                f: |a, b| Some(a.sub(b)),
                commutative: false,
            },
        ),
        (
            "Divide",
            Binary {
                f: |a, b| a.div(b),
                commutative: false,
            },
        ),
        (
            "Power",
            Binary {
                f: |a, b| a.pow(b),
                commutative: false,
            },
        ),
        (
            "Log",
            Binary {
                f: |base, x| x.ln()?.div(base.ln()?),
                commutative: false,
            },
        ),
        (
            "Avg",
            Binary {
                f: |a, b| Some(a.add(b).mul(C::real(0.5))),
                commutative: true,
            },
        ),
        (
            "Hypot",
            Binary {
                f: |a, b| Some(a.mul(a).add(b.mul(b)).sqrt()),
                commutative: true,
            },
        ),
    ]
    .into_iter()
    .collect()
}

fn constant_catalog() -> HashMap<&'static str, C> {
    [
        ("0", C::real(0.0)),
        ("Glaisher", C::real(GLAISHER)),
        ("EulerGamma", C::real(EULER_GAMMA)),
        ("Pi", C::real(PI)),
        ("E", C::real(std::f64::consts::E)),
        ("I", C::i()),
        ("1", C::real(1.0)),
        ("-1", C::real(-1.0)),
        ("2", C::real(2.0)),
    ]
    .into_iter()
    .collect()
}

fn ternary_catalog() -> HashMap<&'static str, Ternary> {
    [
        (
            "FMA",
            Ternary {
                f: |a, b, c| Some(a.mul(b).add(c)),
            },
        ),
        (
            "FMS",
            Ternary {
                f: |a, b, c| Some(a.mul(b).sub(c)),
            },
        ),
        (
            "FNMA",
            Ternary {
                f: |a, b, c| Some(a.mul(b).neg().add(c)),
            },
        ),
        (
            "FNMS",
            Ternary {
                f: |a, b, c| Some(a.mul(b).neg().sub(c)),
            },
        ),
        (
            "FSD",
            Ternary {
                f: |a, b, c| a.sub(b).div(c),
            },
        ),
    ]
    .into_iter()
    .collect()
}

fn validate_symbols(
    constants: &[String],
    functions: &[String],
    operations: &[String],
    ternary: &[String],
    const_all: &HashMap<&'static str, C>,
    unary_all: &HashMap<&'static str, Unary>,
    binary_all: &HashMap<&'static str, Binary>,
    ternary_all: &HashMap<&'static str, Ternary>,
) {
    let unknown_constants: Vec<String> = constants
        .iter()
        .filter(|name| !const_all.contains_key(name.as_str()))
        .cloned()
        .collect();
    let unknown_functions: Vec<String> = functions
        .iter()
        .filter(|name| !unary_all.contains_key(name.as_str()))
        .cloned()
        .collect();
    let unknown_operations: Vec<String> = operations
        .iter()
        .filter(|name| !binary_all.contains_key(name.as_str()))
        .cloned()
        .collect();
    let unknown_ternary: Vec<String> = ternary
        .iter()
        .filter(|name| !ternary_all.contains_key(name.as_str()))
        .cloned()
        .collect();

    if !unknown_constants.is_empty()
        || !unknown_functions.is_empty()
        || !unknown_operations.is_empty()
        || !unknown_ternary.is_empty()
    {
        eprintln!("Error: unknown symbols in input arguments.");
        if !unknown_constants.is_empty() {
            eprintln!("  unknown constants: {unknown_constants:?}");
        }
        if !unknown_functions.is_empty() {
            eprintln!("  unknown functions: {unknown_functions:?}");
        }
        if !unknown_operations.is_empty() {
            eprintln!("  unknown operations: {unknown_operations:?}");
        }
        if !unknown_ternary.is_empty() {
            eprintln!("  unknown ternary operations: {unknown_ternary:?}");
        }
        process::exit(2);
    }
}

fn default_target_constants() -> Vec<String> {
    vec!["Glaisher", "EulerGamma", "Pi", "E", "1", "-1", "2"]
        .into_iter()
        .map(ToOwned::to_owned)
        .collect()
}

fn default_target_functions() -> Vec<String> {
    vec![
        "Half",
        "Minus",
        "Log",
        "Exp",
        "Inv",
        "Sqrt",
        "Sqr",
        "Cosh",
        "Cos",
        "Sinh",
        "Sin",
        "Tanh",
        "Tan",
        "ArcSinh",
        "ArcTanh",
        "ArcSin",
        "ArcCos",
        "ArcTan",
        "ArcCosh",
        "LogisticSigmoid",
    ]
    .into_iter()
    .map(ToOwned::to_owned)
    .collect()
}

fn default_target_operations() -> Vec<String> {
    vec![
        "Plus", "Times", "Subtract", "Divide", "Power", "Log", "Avg", "Hypot",
    ]
    .into_iter()
    .map(ToOwned::to_owned)
    .collect()
}

fn print_remaining(
    todo_constants: &[String],
    todo_unary: &[String],
    todo_binary: &[String],
    todo_ternary: &[String],
) {
    println!("Remaining constants: {todo_constants:?}");
    println!("Remaining unary: {todo_unary:?}");
    println!("Remaining binary: {todo_binary:?}");
    println!("Remaining ternary: {todo_ternary:?}");
}

fn parse_csv(s: &str) -> Vec<String> {
    s.split(',')
        .map(|x| x.trim())
        .filter(|x| !x.is_empty())
        .map(ToOwned::to_owned)
        .collect()
}

fn can_represent(
    target: C,
    constants: &[C],
    unary: &[Unary],
    binary: &[Binary],
    ternary: &[Ternary],
    max_k: usize,
    equiv: EquivCfg,
    domain: DomainMode,
) -> bool {
    let mut levels: Vec<Vec<C>> = vec![vec![]; max_k + 1];
    let mut seen: HashSet<(i64, i64)> = HashSet::new();

    for &c in constants {
        if !value_ok(c, domain, equiv) {
            continue;
        }
        let key = qkey(c);
        if seen.insert(key) {
            levels[1].push(c);
            if near(c, target, equiv) {
                return true;
            }
        }
    }

    for k in 2..=max_k {
        let mut next: Vec<C> = Vec::new();

        for u in unary {
            for &x in &levels[k - 1] {
                if let Some(y) = (u.f)(x) {
                    if value_ok(y, domain, equiv) {
                        let key = qkey(y);
                        if seen.insert(key) {
                                            if near(y, target, equiv) {
                                                return true;
                                            }
                            next.push(y);
                        }
                    }
                }
            }
        }

        for b in binary {
            for left_k in 1..k - 1 {
                let right_k = k - 1 - left_k;
                for &a in &levels[left_k] {
                    for &bb in &levels[right_k] {
                        if b.commutative && left_k == right_k && qkey(a) > qkey(bb) {
                            continue;
                        }
                        if let Some(y) = (b.f)(a, bb) {
                            if value_ok(y, domain, equiv) {
                                let key = qkey(y);
                                if seen.insert(key) {
                                    if near(y, target, equiv) {
                                        return true;
                                    }
                                    next.push(y);
                                }
                            }
                        }
                    }
                }
            }
        }

        if k >= 4 {
            for t in ternary {
                for left_k in 1..=k - 3 {
                    for mid_k in 1..=k - 2 - left_k {
                        let right_k = k - 1 - left_k - mid_k;
                        if right_k < 1 {
                            continue;
                        }
                        for &a in &levels[left_k] {
                            for &b in &levels[mid_k] {
                                for &c in &levels[right_k] {
                                    if let Some(y) = (t.f)(a, b, c) {
                                        if value_ok(y, domain, equiv) {
                                            let key = qkey(y);
                                            if seen.insert(key) {
                                                if near(y, target, equiv) {
                                                    return true;
                                                }
                                                next.push(y);
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        levels[k] = next;
    }

    false
}

fn find_representation(
    target: C,
    constants: &[(String, C)],
    unary: &[(String, Unary)],
    binary: &[(String, Binary)],
    ternary: &[(String, Ternary)],
    max_k: usize,
    equiv: EquivCfg,
    domain: DomainMode,
) -> Option<(String, usize)> {
    let mut levels: Vec<Vec<(C, String)>> = vec![vec![]; max_k + 1];
    let mut seen: HashSet<(i64, i64)> = HashSet::new();

    for (name, &c) in constants.iter().map(|(n, c)| (n, c)) {
        if !value_ok(c, domain, equiv) {
            continue;
        }
        let key = qkey(c);
        if seen.insert(key) {
            levels[1].push((c, name.clone()));
            if near(c, target, equiv) {
                return Some((name.clone(), 1));
            }
        }
    }

    for k in 2..=max_k {
        let mut next: Vec<(C, String)> = Vec::new();

        for (u_name, u) in unary {
            for (x, x_expr) in &levels[k - 1] {
                if let Some(y) = (u.f)(*x) {
                    if value_ok(y, domain, equiv) {
                        let key = qkey(y);
                        if seen.insert(key) {
                                    let expr = format!("{u_name}[{x_expr}]");
                            if near(y, target, equiv) {
                                return Some((expr, k));
                            }
                            next.push((y, expr));
                        }
                    }
                }
            }
        }

        for (b_name, b) in binary {
            for left_k in 1..k - 1 {
                let right_k = k - 1 - left_k;
                for (a, a_expr) in &levels[left_k] {
                    for (bb, bb_expr) in &levels[right_k] {
                        if b.commutative && left_k == right_k && qkey(*a) > qkey(*bb) {
                            continue;
                        }
                        if let Some(y) = (b.f)(*a, *bb) {
                            if value_ok(y, domain, equiv) {
                                let key = qkey(y);
                                if seen.insert(key) {
                                    let expr = format!("{b_name}[{a_expr}, {bb_expr}]");
                                    if near(y, target, equiv) {
                                        return Some((expr, k));
                                    }
                                    next.push((y, expr));
                                }
                            }
                        }
                    }
                }
            }
        }

        if k >= 4 {
            for (t_name, t) in ternary {
                for left_k in 1..=k - 3 {
                    for mid_k in 1..=k - 2 - left_k {
                        let right_k = k - 1 - left_k - mid_k;
                        if right_k < 1 {
                            continue;
                        }
                        for (a, a_expr) in &levels[left_k] {
                            for (b, b_expr) in &levels[mid_k] {
                                for (c, c_expr) in &levels[right_k] {
                                    if let Some(y) = (t.f)(*a, *b, *c) {
                                        if value_ok(y, domain, equiv) {
                                            let key = qkey(y);
                                            if seen.insert(key) {
                                                let expr = format!(
                                                    "{t_name}[{a_expr}, {b_expr}, {c_expr}]"
                                                );
                                                if near(y, target, equiv) {
                                                    return Some((expr, k));
                                                }
                                                next.push((y, expr));
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        levels[k] = next;
    }

    None
}

fn main() {
    let args = parse_args();
    let start = Instant::now();

    let unary_all = unary_catalog();
    let binary_all = binary_catalog();
    let ternary_all = ternary_catalog();
    let const_all = constant_catalog();

    let mut known_constants: Vec<String> = vec!["Glaisher".to_string(), "EulerGamma".to_string()];
    known_constants.extend(parse_csv(&args.constants));
    known_constants.sort();
    known_constants.dedup();

    let mut known_unary = parse_csv(&args.functions);
    known_unary.sort();
    known_unary.dedup();

    let mut known_binary = parse_csv(&args.operations);
    known_binary.sort();
    known_binary.dedup();
    let mut known_ternary = parse_csv(&args.ternary);
    known_ternary.sort();
    known_ternary.dedup();

    let mut target_constants = match &args.target_constants {
        Some(v) => parse_csv(v),
        None => default_target_constants(),
    };
    target_constants.sort();
    target_constants.dedup();

    let mut target_unary = match &args.target_functions {
        Some(v) => parse_csv(v),
        None => default_target_functions(),
    };
    target_unary.sort();
    target_unary.dedup();

    let mut target_binary = match &args.target_operations {
        Some(v) => parse_csv(v),
        None => default_target_operations(),
    };
    target_binary.sort();
    target_binary.dedup();

    let mut target_ternary = match &args.target_ternary {
        Some(v) => parse_csv(v),
        None => Vec::new(),
    };
    target_ternary.sort();
    target_ternary.dedup();

    validate_symbols(
        &known_constants,
        &known_unary,
        &known_binary,
        &known_ternary,
        &const_all,
        &unary_all,
        &binary_all,
        &ternary_all,
    );
    validate_symbols(
        &target_constants,
        &target_unary,
        &target_binary,
        &target_ternary,
        &const_all,
        &unary_all,
        &binary_all,
        &ternary_all,
    );

    let mut todo_constants = target_constants.clone();
    todo_constants.retain(|c| !known_constants.contains(c));

    let mut todo_unary = target_unary.clone();
    todo_unary.retain(|f| !known_unary.contains(f));

    let mut todo_binary = target_binary.clone();
    todo_binary.retain(|op| !known_binary.contains(op));
    let mut todo_ternary = target_ternary.clone();
    todo_ternary.retain(|op| !known_ternary.contains(op));

    println!("Target constants: {target_constants:?}");
    println!("Target unary functions: {target_unary:?}");
    println!("Target binary operations: {target_binary:?}");
    println!("Target ternary operations: {target_ternary:?}");
    match args.domain {
        DomainMode::Complex => {
            println!("Domain mode: complex (complex branches are allowed).");
        }
        DomainMode::Real => {
            println!("Domain mode: real (values with nonzero imaginary part are rejected).");
        }
    }
    print_remaining(&todo_constants, &todo_unary, &todo_binary, &todo_ternary);

    let mut k = 1usize;
    while k <= args.max_k
        && (!todo_constants.is_empty()
            || !todo_unary.is_empty()
            || !todo_binary.is_empty()
            || !todo_ternary.is_empty())
    {
        println!("Testing with K = {k}");
        let mut new_item = false;

        let numeric_constants: Vec<C> = known_constants
            .iter()
            .filter_map(|name| const_all.get(name.as_str()).copied())
            .collect();
        let named_constants: Vec<(String, C)> = known_constants
            .iter()
            .filter_map(|name| const_all.get(name.as_str()).copied().map(|v| (name.clone(), v)))
            .collect();
        let unary_set: Vec<Unary> = known_unary
            .iter()
            .filter_map(|name| unary_all.get(name.as_str()).cloned())
            .collect();
        let named_unary: Vec<(String, Unary)> = known_unary
            .iter()
            .filter_map(|name| unary_all.get(name.as_str()).cloned().map(|u| (name.clone(), u)))
            .collect();
        let binary_set: Vec<Binary> = known_binary
            .iter()
            .filter_map(|name| binary_all.get(name.as_str()).cloned())
            .collect();
        let named_binary: Vec<(String, Binary)> = known_binary
            .iter()
            .filter_map(|name| binary_all.get(name.as_str()).cloned().map(|b| (name.clone(), b)))
            .collect();
        let ternary_set: Vec<Ternary> = known_ternary
            .iter()
            .filter_map(|name| ternary_all.get(name.as_str()).cloned())
            .collect();
        let named_ternary: Vec<(String, Ternary)> = known_ternary
            .iter()
            .filter_map(|name| {
                ternary_all
                    .get(name.as_str())
                    .cloned()
                    .map(|t| (name.clone(), t))
            })
            .collect();

        let mut found_binary: Option<(usize, Option<(String, usize)>)> = None;
        for (idx, op_name) in todo_binary.iter().enumerate() {
            let op = binary_all.get(op_name.as_str()).unwrap();
            let target = (op.f)(C::real(EULER_GAMMA), C::real(GLAISHER)).unwrap();
            if args.explain {
                if let Some(witness) = find_representation(
                    target,
                    &named_constants,
                    &named_unary,
                    &named_binary,
                    &named_ternary,
                    k,
                    args.equiv,
                    args.domain,
                ) {
                    found_binary = Some((idx, Some(witness)));
                    break;
                }
            } else if can_represent(
                target,
                &numeric_constants,
                &unary_set,
                &binary_set,
                &ternary_set,
                k,
                args.equiv,
                args.domain,
            ) {
                found_binary = Some((idx, None));
                break;
            }
        }
        if let Some((idx, witness)) = found_binary {
            let found = todo_binary.remove(idx);
            println!("Found binary operation: {found}");
            if let Some((expr, expr_k)) = witness {
                println!("  witness[k={expr_k}]: {expr}");
            }
            known_binary.push(found);
            known_binary.sort();
            known_binary.dedup();
            print_remaining(&todo_constants, &todo_unary, &todo_binary, &todo_ternary);
            k = 1;
            new_item = true;
        }

        if new_item {
            continue;
        }

        let mut found_ternary: Option<(usize, Option<(String, usize)>)> = None;
        for (idx, op_name) in todo_ternary.iter().enumerate() {
            let op = ternary_all.get(op_name.as_str()).unwrap();
            let target = (op.f)(
                C::real(EULER_GAMMA),
                C::real(GLAISHER),
                C::real(PI),
            )
            .unwrap();
            if args.explain {
                if let Some(witness) = find_representation(
                    target,
                    &named_constants,
                    &named_unary,
                    &named_binary,
                    &named_ternary,
                    k,
                    args.equiv,
                    args.domain,
                ) {
                    found_ternary = Some((idx, Some(witness)));
                    break;
                }
            } else if can_represent(
                target,
                &numeric_constants,
                &unary_set,
                &binary_set,
                &ternary_set,
                k,
                args.equiv,
                args.domain,
            ) {
                found_ternary = Some((idx, None));
                break;
            }
        }
        if let Some((idx, witness)) = found_ternary {
            let found = todo_ternary.remove(idx);
            println!("Found ternary operation: {found}");
            if let Some((expr, expr_k)) = witness {
                println!("  witness[k={expr_k}]: {expr}");
            }
            known_ternary.push(found);
            known_ternary.sort();
            known_ternary.dedup();
            print_remaining(&todo_constants, &todo_unary, &todo_binary, &todo_ternary);
            k = 1;
            new_item = true;
        }

        if new_item {
            continue;
        }

        let mut found_constant: Option<(usize, Option<(String, usize)>)> = None;
        for (idx, c_name) in todo_constants.iter().enumerate() {
            let target = *const_all.get(c_name.as_str()).unwrap();
            if args.explain {
                if let Some(witness) = find_representation(
                    target,
                    &named_constants,
                    &named_unary,
                    &named_binary,
                    &named_ternary,
                    k,
                    args.equiv,
                    args.domain,
                ) {
                    found_constant = Some((idx, Some(witness)));
                    break;
                }
            } else if can_represent(
                target,
                &numeric_constants,
                &unary_set,
                &binary_set,
                &ternary_set,
                k,
                args.equiv,
                args.domain,
            ) {
                found_constant = Some((idx, None));
                break;
            }
        }
        if let Some((idx, witness)) = found_constant {
            let found = todo_constants.remove(idx);
            println!("Found constant: {found}");
            if let Some((expr, expr_k)) = witness {
                println!("  witness[k={expr_k}]: {expr}");
            }
            known_constants.push(found);
            known_constants.sort();
            known_constants.dedup();
            print_remaining(&todo_constants, &todo_unary, &todo_binary, &todo_ternary);
            k = 1;
            new_item = true;
        }

        if new_item {
            continue;
        }

        let mut found_unary: Option<(usize, Option<(String, usize)>)> = None;
        for (idx, f_name) in todo_unary.iter().enumerate() {
            let f = unary_all.get(f_name.as_str()).unwrap();
            let Some(target) = (f.f)(C::real(EULER_GAMMA)) else {
                continue;
            };
            if args.explain {
                if let Some(witness) = find_representation(
                    target,
                    &named_constants,
                    &named_unary,
                    &named_binary,
                    &named_ternary,
                    k,
                    args.equiv,
                    args.domain,
                ) {
                    found_unary = Some((idx, Some(witness)));
                    break;
                }
            } else if can_represent(
                target,
                &numeric_constants,
                &unary_set,
                &binary_set,
                &ternary_set,
                k,
                args.equiv,
                args.domain,
            ) {
                found_unary = Some((idx, None));
                break;
            }
        }
        if let Some((idx, witness)) = found_unary {
            let found = todo_unary.remove(idx);
            println!("Found unary function: {found}");
            if let Some((expr, expr_k)) = witness {
                println!("  witness[k={expr_k}]: {expr}");
            }
            known_unary.push(found);
            known_unary.sort();
            known_unary.dedup();
            print_remaining(&todo_constants, &todo_unary, &todo_binary, &todo_ternary);
            k = 1;
            new_item = true;
        }

        if !new_item {
            println!("No new items at K = {k}");
            k += 1;
        }
    }

    println!("Known constants: {known_constants:?}");
    println!("Known unary functions: {known_unary:?}");
    println!("Known binary operations: {known_binary:?}");
    println!("Known ternary operations: {known_ternary:?}");
    println!("Target constants: {target_constants:?}");
    println!("Target unary functions: {target_unary:?}");
    println!("Target binary operations: {target_binary:?}");
    println!("Target ternary operations: {target_ternary:?}");
    print_remaining(&todo_constants, &todo_unary, &todo_binary, &todo_ternary);
    println!("Elapsed: {:.2?}", start.elapsed());
}
