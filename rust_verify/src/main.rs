use std::collections::{HashMap, HashSet};
use std::env;
use std::f64::consts::{FRAC_PI_2, PI};
use std::process;
use std::process::Command;
use std::sync::Arc;
use std::time::Instant;


const EULER_GAMMA: f64 = 0.577_215_664_901_532_9;
const CATALAN:     f64 = 0.915_965_594_177_219_0;
const GLAISHER:    f64 = 1.282_427_129_100_622_6;
const KHINCHIN:    f64 = 2.685_452_001_065_306_4;

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
    check_involution: Option<String>,
    involution_samples: usize,
    validate_witness: bool,
    validate_witness_highprec: bool,
    scan_family: bool,
    scan_g: String,
    scan_h: String,
    scan_params: String,
    scan_top: usize,
}

#[derive(Clone, Copy, Debug)]
struct C {
    re: f64,
    im: f64,
}

#[derive(Clone)]
struct Unary {
    f: Arc<dyn Fn(C) -> Option<C> + Send + Sync>,
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

fn unary<F>(f: F) -> Unary
where
    F: Fn(C) -> Option<C> + Send + Sync + 'static,
{
    Unary { f: Arc::new(f) }
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
        check_involution: None,
        involution_samples: 24,
        validate_witness: true,
        validate_witness_highprec: false,
        scan_family: false,
        scan_g: "Half,Minus,Log,Exp,Inv,Sqrt,Sqr,Cosh,Cos,Sinh,Sin,Tanh,Tan,ArcSinh,ArcTanh,ArcSin,ArcCos,ArcTan,ArcCosh,LogisticSigmoid".to_string(),
        scan_h: "reflect,recip,mobius,powlog".to_string(),
        scan_params: "-1,0,1,2,E,Pi,I".to_string(),
        scan_top: 20,
    };

    let argv: Vec<String> = env::args().skip(1).collect();
    let mut i = 0usize;
    while i < argv.len() {
        let flag = argv[i].clone();
        i += 1;
        match flag.as_str() {
            "-h" | "--help" => {
                print_help();
                process::exit(0);
            }
            "--constants" => {
                if let Some(v) = collect_csv_flag_values(&argv, &mut i) {
                    args.constants = v;
                } else {
                    eprintln!("Error: --constants expects at least one value.");
                    process::exit(2);
                }
            }
            "--functions" => {
                if let Some(v) = collect_csv_flag_values(&argv, &mut i) {
                    args.functions = v;
                } else {
                    eprintln!("Error: --functions expects at least one value.");
                    process::exit(2);
                }
            }
            "--operations" => {
                if let Some(v) = collect_csv_flag_values(&argv, &mut i) {
                    args.operations = v;
                } else {
                    eprintln!("Error: --operations expects at least one value.");
                    process::exit(2);
                }
            }
            "--ternary" => {
                if let Some(v) = collect_csv_flag_values(&argv, &mut i) {
                    args.ternary = v;
                } else {
                    eprintln!("Error: --ternary expects at least one value.");
                    process::exit(2);
                }
            }
            "--target-constants" => {
                if let Some(v) = collect_csv_flag_values(&argv, &mut i) {
                    args.target_constants = Some(v);
                } else {
                    eprintln!("Error: --target-constants expects at least one value.");
                    process::exit(2);
                }
            }
            "--target-functions" => {
                if let Some(v) = collect_csv_flag_values(&argv, &mut i) {
                    args.target_functions = Some(v);
                } else {
                    eprintln!("Error: --target-functions expects at least one value.");
                    process::exit(2);
                }
            }
            "--target-operations" => {
                if let Some(v) = collect_csv_flag_values(&argv, &mut i) {
                    args.target_operations = Some(v);
                } else {
                    eprintln!("Error: --target-operations expects at least one value.");
                    process::exit(2);
                }
            }
            "--target-ternary" => {
                if let Some(v) = collect_csv_flag_values(&argv, &mut i) {
                    args.target_ternary = Some(v);
                } else {
                    eprintln!("Error: --target-ternary expects at least one value.");
                    process::exit(2);
                }
            }
            "--max-k" => {
                if i < argv.len() {
                    if let Ok(n) = argv[i].parse::<usize>() {
                        args.max_k = n;
                    }
                    i += 1;
                }
            }
            "--eps" => {
                if i < argv.len() {
                    if let Ok(x) = argv[i].parse::<f64>() {
                        if x.is_finite() && x >= 0.0 {
                            args.equiv.eps = x;
                        }
                    }
                    i += 1;
                }
            }
            "--equiv" => {
                if i < argv.len() {
                    args.equiv.mode = match argv[i].as_str() {
                        "ulp" | "ULP" => EquivMode::Ulp,
                        _ => EquivMode::Rel,
                    };
                    i += 1;
                }
            }
            "--ulp" => {
                if i < argv.len() {
                    if let Ok(x) = argv[i].parse::<u64>() {
                        args.equiv.ulp_tol = x;
                    }
                    i += 1;
                }
            }
            "--domain" => {
                if i < argv.len() {
                    args.domain = match argv[i].as_str() {
                        "real" | "REAL" => DomainMode::Real,
                        _ => DomainMode::Complex,
                    };
                    i += 1;
                }
            }
            "--explain" => {
                args.explain = true;
            }
            "--check-involution" => {
                if i < argv.len() {
                    args.check_involution = Some(argv[i].clone());
                    i += 1;
                }
            }
            "--involution-samples" => {
                if i < argv.len() {
                    if let Ok(n) = argv[i].parse::<usize>() {
                        if n >= 1 {
                            args.involution_samples = n;
                        }
                    }
                    i += 1;
                }
            }
            "--no-validate-witness" => {
                args.validate_witness = false;
            }
            "--validate-witness-highprec" => {
                args.validate_witness_highprec = true;
            }
            "--scan-family" => {
                args.scan_family = true;
            }
            "--scan-g" => {
                if i < argv.len() {
                    args.scan_g = argv[i].clone();
                    i += 1;
                }
            }
            "--scan-h" => {
                if i < argv.len() {
                    args.scan_h = argv[i].clone();
                    i += 1;
                }
            }
            "--scan-params" => {
                if let Some(v) = collect_csv_flag_values(&argv, &mut i) {
                    args.scan_params = v;
                }
            }
            "--scan-top" => {
                if i < argv.len() {
                    if let Ok(n) = argv[i].parse::<usize>() {
                        if n >= 1 {
                            args.scan_top = n;
                        }
                    }
                    i += 1;
                }
            }
            _ => {
                eprintln!("Error: unrecognized argument: {flag}");
                eprintln!("Use --help to see available options.");
                process::exit(2);
            }
        }
    }
    args
}

fn print_help() {
    println!("verify_base_set_rs - symbolic base-set reconstruction checker");
    println!();
    println!("Usage:");
    println!("  verify_base_set_rs [options]");
    println!();
    println!("Core options:");
    println!("  --constants CSV             Base constants (default: Pi)");
    println!("  --functions CSV             Base unary functions (default: Exp,Log,Minus)");
    println!("  --operations CSV            Base binary operations (default: Plus)");
    println!("  --ternary CSV               Base ternary operations (default: empty)");
    println!("  --max-k N                   Maximum search code length/depth bound (default: 10)");
    println!("  --domain real|complex       Evaluation domain (default: complex)");
    println!("  --explain                   Print witness expressions for discovered items");
    println!();
    println!("Target selection (defaults to CALC4-like sets):");
    println!("  --target-constants CSV");
    println!("  --target-functions CSV");
    println!("  --target-operations CSV");
    println!("  --target-ternary CSV");
    println!();
    println!("Equivalence / validation:");
    println!("  --equiv rel|ulp             Numeric equivalence mode (default: rel)");
    println!("  --eps X                     Relative epsilon for --equiv rel");
    println!("  --ulp N                     ULP tolerance for --equiv ulp");
    println!("  --no-validate-witness       Skip witness re-validation (faster, less safe)");
    println!("  --validate-witness-highprec Enable extra Python/mpmath witness check (opt-in)");
    println!();
    println!("Diagnostics / exploration:");
    println!("  --check-involution NAME     Test f(f(x)) ~= x for a unary function");
    println!("  --involution-samples N      Sample count for involution checks");
    println!("  --scan-family               Scan generated unary candidates");
    println!("  --scan-g CSV                Candidate generator base functions");
    println!("  --scan-h CSV                Candidate template families");
    println!("  --scan-params CSV           Parameter pool for candidate generation");
    println!("  --scan-top N                Print top-N scan results");
    println!();
    println!("Examples:");
    println!("  verify_base_set_rs --constants Pi --functions Exp,Log,Minus --operations Plus --max-k 10");
    println!("  verify_base_set_rs --constants 1 --functions '' --operations EML --max-k 10 --explain");
    println!("  verify_base_set_rs --check-involution ArcSinh");
}

fn collect_csv_flag_values(argv: &[String], i: &mut usize) -> Option<String> {
    let mut joined = String::new();
    let mut consumed = 0usize;
    while *i < argv.len() && !argv[*i].starts_with("--") {
        if !joined.is_empty() {
            joined.push(',');
        }
        joined.push_str(argv[*i].trim());
        *i += 1;
        consumed += 1;
    }
    if consumed == 0 {
        return None;
    }
    Some(parse_csv(&joined).join(","))
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

fn involution_log_reflect(x: C, c: C) -> Option<C> {
    let y = x.ln()?;
    Some(c.sub(y).exp())
}

fn involution_log_recip(x: C, c: C) -> Option<C> {
    let y = x.ln()?;
    let den = y.sub(c);
    let frac = C::real(1.0).div(den)?;
    Some(c.add(frac).exp())
}

fn unary_catalog() -> HashMap<String, Unary> {
    let mut out: HashMap<String, Unary> = [
        ("Half".to_string(), unary(|x| Some(x.mul(C::real(0.5))))),
        ("Minus".to_string(), unary(|x| Some(x.neg()))),
        ("Log".to_string(), unary(|x| x.ln())),
        ("Exp".to_string(), unary(|x| Some(x.exp()))),
        ("Inv".to_string(), unary(|x| C::real(1.0).div(x))),
        ("Sqrt".to_string(), unary(|x| Some(x.sqrt()))),
        ("Sqr".to_string(), unary(|x| Some(x.mul(x)))),
        ("Cosh".to_string(), unary(|x| Some(x.cosh()))),
        ("Cos".to_string(), unary(|x| Some(x.cos()))),
        ("Sinh".to_string(), unary(|x| Some(x.sinh()))),
        ("Sin".to_string(), unary(|x| Some(x.sin()))),
        ("Tanh".to_string(), unary(|x| x.tanh())),
        ("Tan".to_string(), unary(|x| x.tan())),
        ("ArcSinh".to_string(), unary(|x| x.asinh())),
        ("ArcTanh".to_string(), unary(|x| x.atanh())),
        ("ArcSin".to_string(), unary(|x| x.asin())),
        ("ArcCos".to_string(), unary(|x| x.acos())),
        ("ArcTan".to_string(), unary(|x| x.atan())),
        ("ArcCosh".to_string(), unary(|x| x.acosh())),
        ("LogisticSigmoid".to_string(), unary(logistic_sigmoid)),
        ("LogReflect1".to_string(), unary(|x| involution_log_reflect(x, C::real(1.0)))),
        ("LogReflect0".to_string(), unary(|x| involution_log_reflect(x, C::real(0.0)))),
        ("LogReflectNeg1".to_string(), unary(|x| involution_log_reflect(x, C::real(-1.0)))),
        ("LogReflect2".to_string(), unary(|x| involution_log_reflect(x, C::real(2.0)))),
        (
            "LogReflectE".to_string(),
            unary(|x| involution_log_reflect(x, C::real(std::f64::consts::E))),
        ),
        ("LogReflectPi".to_string(), unary(|x| involution_log_reflect(x, C::real(PI)))),
        ("LogReflectI".to_string(), unary(|x| involution_log_reflect(x, C::i()))),
        ("LogRecip".to_string(), unary(|x| involution_log_recip(x, C::real(0.0)))),
        (
            "LogNegRecip".to_string(),
            unary(|x| {
                let y = x.ln()?;
                let frac = C::real(-1.0).div(y)?;
                Some(frac.exp())
            }),
        ),
        ("LogRecipShift1".to_string(), unary(|x| involution_log_recip(x, C::real(1.0)))),
        ("LogRecipShift0".to_string(), unary(|x| involution_log_recip(x, C::real(0.0)))),
        (
            "LogRecipShiftNeg1".to_string(),
            unary(|x| involution_log_recip(x, C::real(-1.0))),
        ),
        ("LogRecipShift2".to_string(), unary(|x| involution_log_recip(x, C::real(2.0)))),
        (
            "LogRecipShiftE".to_string(),
            unary(|x| involution_log_recip(x, C::real(std::f64::consts::E))),
        ),
        ("LogRecipShiftPi".to_string(), unary(|x| involution_log_recip(x, C::real(PI)))),
        ("LogRecipShiftI".to_string(), unary(|x| involution_log_recip(x, C::i()))),
        ("AcosReflect1".to_string(), unary(|x| Some(C::real(1.0).sub(x.acos()?).cos()))),
        (
            "AcosReflectPi3".to_string(),
            unary(|x| Some(C::real(PI / 3.0).sub(x.acos()?).cos())),
        ),
        ("AcosReflectI".to_string(), unary(|x| Some(C::i().sub(x.acos()?).cos()))),
        ("AsinhRecip1".to_string(), unary(|x| Some(C::real(1.0).div(x.asinh()?)?.sinh()))),
        ("AsinhRecipI".to_string(), unary(|x| Some(C::i().div(x.asinh()?)?.sinh()))),
        ("AcoshReflect1".to_string(), unary(|x| Some(C::real(1.0).sub(x.acosh()?).cosh()))),
        ("AcoshReflectI".to_string(), unary(|x| Some(C::i().sub(x.acosh()?).cosh()))),
        ("AcoshRecip1".to_string(), unary(|x| Some(C::real(1.0).div(x.acosh()?)?.cosh()))),
        ("AcoshRecipI".to_string(), unary(|x| Some(C::i().div(x.acosh()?)?.cosh()))),
        ("AtanhReflect1".to_string(), unary(|x| C::real(1.0).sub(x.atanh()?).tanh())),
        ("AtanhRecip1".to_string(), unary(|x| C::real(1.0).div(x.atanh()?)?.tanh())),
    ]
    .into_iter()
    .collect();

    for candidate in generated_candidates_default() {
        out.insert(candidate.name, candidate.unary);
    }
    out
}

#[derive(Clone)]
struct GeneratedCandidate {
    name: String,
    formula: String,
    unary: Unary,
}

fn parse_param_token(tok: &str) -> Option<C> {
    match tok {
        "-1" => Some(C::real(-1.0)),
        "0" => Some(C::real(0.0)),
        "1" => Some(C::real(1.0)),
        "2" => Some(C::real(2.0)),
        "E" | "e" => Some(C::real(std::f64::consts::E)),
        "Pi" | "PI" | "pi" => Some(C::real(PI)),
        "I" | "i" => Some(C::i()),
        _ => tok.parse::<f64>().ok().map(C::real),
    }
}

fn encode_param_token(tok: &str) -> String {
    tok.replace('-', "Neg").replace('.', "_")
}

fn apply_g(g: &str, x: C) -> Option<C> {
    match g {
        "Half" => Some(x.mul(C::real(0.5))),
        "Minus" => Some(x.neg()),
        "Log" => x.ln(),
        "Exp" => Some(x.exp()),
        "Inv" => C::real(1.0).div(x),
        "Sqrt" => Some(x.sqrt()),
        "Sqr" => Some(x.mul(x)),
        "Cosh" => Some(x.cosh()),
        "Cos" => Some(x.cos()),
        "Sinh" => Some(x.sinh()),
        "Sin" => Some(x.sin()),
        "Tanh" => x.tanh(),
        "Tan" => x.tan(),
        "ArcCos" => x.acos(),
        "ArcSinh" => x.asinh(),
        "ArcCosh" => x.acosh(),
        "ArcTanh" => x.atanh(),
        "ArcSin" => x.asin(),
        "ArcTan" => x.atan(),
        "LogisticSigmoid" => logistic_sigmoid(x),
        _ => None,
    }
}

fn apply_g_inv(g: &str, y: C) -> Option<C> {
    match g {
        "Half" => Some(y.mul(C::real(2.0))),
        "Minus" => Some(y.neg()),
        "Log" => Some(y.exp()),
        "Exp" => y.ln(),
        "Inv" => C::real(1.0).div(y),
        "Sqrt" => Some(y.mul(y)),
        "Sqr" => Some(y.sqrt()),
        "Cosh" => y.acosh(),
        "Cos" => y.acos(),
        "Sinh" => y.asinh(),
        "Sin" => y.asin(),
        "Tanh" => y.atanh(),
        "Tan" => y.atan(),
        "ArcCos" => Some(y.cos()),
        "ArcSinh" => Some(y.sinh()),
        "ArcCosh" => Some(y.cosh()),
        "ArcTanh" => y.tanh(),
        "ArcSin" => Some(y.sin()),
        "ArcTan" => y.tan(),
        "LogisticSigmoid" => {
            let one = C::real(1.0);
            let den = one.sub(y);
            let frac = y.div(den)?;
            frac.ln()
        }
        _ => None,
    }
}

fn apply_h(h: &str, u: C, c: C) -> Option<C> {
    match h {
        "reflect" => Some(c.sub(u)),
        "recip" => c.div(u),
        "mobius" => {
            let frac = C::real(1.0).div(u.sub(c))?;
            Some(c.add(frac))
        }
        "powlog" => {
            let log_c = c.ln()?;
            let log_u = u.ln()?;
            let expnt = log_c.div(log_u)?;
            c.pow(expnt)
        }
        _ => None,
    }
}

fn generated_candidates_default() -> Vec<GeneratedCandidate> {
    generated_candidates(
        "Half,Minus,Log,Exp,Inv,Sqrt,Sqr,Cosh,Cos,Sinh,Sin,Tanh,Tan,ArcSinh,ArcTanh,ArcSin,ArcCos,ArcTan,ArcCosh,LogisticSigmoid",
        "reflect,recip,mobius,powlog",
        "-1,0,1,2,E,Pi,I",
    )
}

fn generated_candidates(scan_g: &str, scan_h: &str, scan_params: &str) -> Vec<GeneratedCandidate> {
    let gs = parse_csv(scan_g);
    let hs = parse_csv(scan_h);
    let params = parse_csv(scan_params);
    let mut out = Vec::new();
    for g in gs {
        for h in &hs {
            for p in &params {
                let Some(cval) = parse_param_token(p) else {
                    continue;
                };
                let g_name = g.clone();
                let h_name = h.clone();
                let c_tok = p.clone();
                let name = format!("Gen{}_{}{}", g_name, h_name, encode_param_token(&c_tok));
                let formula = format!("{g_name}^-1({h_name}({g_name}(x);{c_tok}))");
                let unary_fn = unary({
                    let g_name = g_name.clone();
                    let h_name = h_name.clone();
                    move |x: C| {
                        let u = apply_g(&g_name, x)?;
                        let v = apply_h(&h_name, u, cval)?;
                        apply_g_inv(&g_name, v)
                    }
                });
                out.push(GeneratedCandidate {
                    name,
                    formula,
                    unary: unary_fn,
                });
            }
        }
    }
    out
}

fn involution_samples(limit: usize) -> Vec<C> {
    let base = [
        C::real(0.2),
        C::real(0.5),
        C::real(0.9),
        C::real(1.3),
        C::real(2.0),
        C::real(3.0),
        C::new(0.8, 0.2),
        C::new(1.2, -0.3),
        C::new(2.0, 0.5),
        C::new(0.4, 1.1),
        C::new(-0.7, 0.6),
        C::new(-1.3, -0.4),
    ];
    if limit <= base.len() {
        return base[..limit].to_vec();
    }
    let mut out = base.to_vec();
    while out.len() < limit {
        let n = out.len() as f64 + 1.0;
        out.push(C::new((0.17 * n).cos() + 1.1, (0.23 * n).sin() * 0.9));
    }
    out
}

fn run_involution_check(
    name: &str,
    unary_all: &HashMap<String, Unary>,
    equiv: EquivCfg,
    samples_n: usize,
) -> i32 {
    let Some(f) = unary_all.get(name).cloned() else {
        eprintln!("Error: unknown function for involution check: {name}");
        return 2;
    };
    let mut tested = 0usize;
    let mut passed = 0usize;
    let mut skipped = 0usize;
    let mut worst = 0.0f64;
    let mut worst_input = C::real(0.0);
    let mut worst_output = C::real(0.0);

    for x in involution_samples(samples_n) {
        let Some(y) = (f.f)(x) else {
            skipped += 1;
            continue;
        };
        let Some(z) = (f.f)(y) else {
            skipped += 1;
            continue;
        };
        tested += 1;
        let err = z.sub(x).abs();
        if err > worst {
            worst = err;
            worst_input = x;
            worst_output = z;
        }
        if near(x, z, equiv) {
            passed += 1;
        }
    }

    println!("Involution check for: {name}");
    println!("tested={tested}, passed={passed}, skipped={skipped}");
    println!(
        "worst_abs_error={worst:.6e} at input=({:.6},{:.6}) -> ({:.6},{:.6})",
        worst_input.re, worst_input.im, worst_output.re, worst_output.im
    );
    if tested > 0 && passed == tested {
        println!("Result: PASS (all tested points satisfy f(f(x)) ~= x)");
        0
    } else {
        println!("Result: FAIL (some tested points violate involution criterion)");
        1
    }
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
        (
            "EML",
            Binary {
                // EML[a,b] = Exp[a] - Log[b]
                f: |a, b| Some(a.exp().sub(b.ln()?)),
                commutative: false,
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
        ("Catalan", C::real(CATALAN)),
        ("Khinchin", C::real(KHINCHIN)),
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
            "TExpLog",
            Ternary {
                // TExpLog[a,b,c] = Exp[a] * Log[b,c]
                f: |a, b, c| Some(a.exp().mul(c.ln()?.div(b.ln()?)?)),
            },
        ),
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
        (
            "LogPowExpSub",
            Ternary {
                // LogPowExpSub[a,b,c] = Log[a, Power[c, Exp[Subtract[a,b]]]]
                // = Exp[a-b] * Log[c] / Log[a]
                f: |a, b, c| a.sub(b).exp().mul(c.ln()?).div(a.ln()?),
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
    unary_all: &HashMap<String, Unary>,
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

fn parse_remaining_count(line: &str) -> Option<usize> {
    let l = line.trim();
    if !l.starts_with("Remaining ") {
        return None;
    }
    let open = l.find('[')?;
    let close = l.rfind(']')?;
    if close <= open {
        return None;
    }
    let body = l[(open + 1)..close].trim();
    if body.is_empty() {
        return Some(0);
    }
    Some(body.split(',').count())
}

#[derive(Clone, Debug)]
enum Expr {
    Atom(String),
    Call(String, Vec<Expr>),
}

fn parse_expr(s: &str) -> Option<Expr> {
    fn skip_ws(bs: &[u8], i: &mut usize) {
        while *i < bs.len() && bs[*i].is_ascii_whitespace() {
            *i += 1;
        }
    }
    fn parse_ident(bs: &[u8], i: &mut usize) -> Option<String> {
        skip_ws(bs, i);
        let start = *i;
        while *i < bs.len() {
            let c = bs[*i] as char;
            if c == '[' || c == ']' || c == ',' || c.is_whitespace() {
                break;
            }
            *i += 1;
        }
        if *i == start {
            return None;
        }
        Some(String::from_utf8_lossy(&bs[start..*i]).to_string())
    }
    fn parse_rec(bs: &[u8], i: &mut usize) -> Option<Expr> {
        let name = parse_ident(bs, i)?;
        skip_ws(bs, i);
        if *i < bs.len() && bs[*i] == b'[' {
            *i += 1;
            let mut args = Vec::new();
            loop {
                skip_ws(bs, i);
                if *i < bs.len() && bs[*i] == b']' {
                    *i += 1;
                    break;
                }
                let e = parse_rec(bs, i)?;
                args.push(e);
                skip_ws(bs, i);
                if *i < bs.len() && bs[*i] == b',' {
                    *i += 1;
                    continue;
                }
                if *i < bs.len() && bs[*i] == b']' {
                    *i += 1;
                    break;
                }
                return None;
            }
            Some(Expr::Call(name, args))
        } else {
            Some(Expr::Atom(name))
        }
    }
    let bs = s.as_bytes();
    let mut i = 0usize;
    let expr = parse_rec(bs, &mut i)?;
    while i < bs.len() && bs[i].is_ascii_whitespace() {
        i += 1;
    }
    if i == bs.len() {
        Some(expr)
    } else {
        None
    }
}

fn eval_expr_c(
    e: &Expr,
    env: &HashMap<String, C>,
    unary_all: &HashMap<String, Unary>,
    binary_all: &HashMap<&'static str, Binary>,
    ternary_all: &HashMap<&'static str, Ternary>,
    const_all: &HashMap<&'static str, C>,
) -> Option<C> {
    match e {
        Expr::Atom(name) => {
            if let Some(v) = env.get(name) {
                return Some(*v);
            }
            const_all.get(name.as_str()).copied()
        }
        Expr::Call(name, args) => match args.len() {
            1 => {
                let x = eval_expr_c(&args[0], env, unary_all, binary_all, ternary_all, const_all)?;
                let u = unary_all.get(name)?;
                (u.f)(x)
            }
            2 => {
                let a = eval_expr_c(&args[0], env, unary_all, binary_all, ternary_all, const_all)?;
                let b = eval_expr_c(&args[1], env, unary_all, binary_all, ternary_all, const_all)?;
                let op = binary_all.get(name.as_str())?;
                (op.f)(a, b)
            }
            3 => {
                let a = eval_expr_c(&args[0], env, unary_all, binary_all, ternary_all, const_all)?;
                let b = eval_expr_c(&args[1], env, unary_all, binary_all, ternary_all, const_all)?;
                let c = eval_expr_c(&args[2], env, unary_all, binary_all, ternary_all, const_all)?;
                let op = ternary_all.get(name.as_str())?;
                (op.f)(a, b, c)
            }
            _ => None,
        },
    }
}

fn validate_witness_highprec_python(witness_expr: &str, target_expr: &str, kind: &str) -> bool {
    let script = r#"
import sys
try:
    import mpmath as mp
except Exception:
    sys.exit(3)

mp.mp.dps = 90
witness, target, kind = sys.argv[1], sys.argv[2], sys.argv[3]

def parse_expr(s):
    i = 0
    n = len(s)
    def skip():
        nonlocal i
        while i < n and s[i].isspace():
            i += 1
    def ident():
        nonlocal i
        skip()
        j = i
        while i < n and s[i] not in '[],' and not s[i].isspace():
            i += 1
        if i == j:
            return None
        return s[j:i]
    def rec():
        nonlocal i
        name = ident()
        if name is None:
            return None
        skip()
        if i < n and s[i] == '[':
            i += 1
            args = []
            while True:
                skip()
                if i < n and s[i] == ']':
                    i += 1
                    break
                a = rec()
                if a is None:
                    return None
                args.append(a)
                skip()
                if i < n and s[i] == ',':
                    i += 1
                    continue
                if i < n and s[i] == ']':
                    i += 1
                    break
                return None
            return ('call', name, args)
        return ('atom', name)
    out = rec()
    skip()
    return out if i == n else None

def const(name):
    tbl = {
      'EulerGamma': mp.mpf('0.5772156649015328606'),
      'Glaisher': mp.mpf('1.2824271291006226369'),
      'Catalan': mp.mpf('0.91596559417721901505'),
      'Khinchin': mp.mpf('2.6854520010653064453'),
      'Pi': mp.pi,
      'E': mp.e,
      '1': mp.mpf(1),
      '-1': mp.mpf(-1),
      '2': mp.mpf(2),
      '0': mp.mpf(0),
    }
    return tbl.get(name)

def apply1(n, x):
    if n == 'Half': return x/2
    if n == 'Minus': return -x
    if n == 'Log': return mp.log(x) if x > 0 else None
    if n == 'Exp': return mp.exp(x)
    if n == 'Inv': return 1/x if x != 0 else None
    if n == 'Sqrt': return mp.sqrt(x) if x >= 0 else None
    if n == 'Sqr': return x*x
    if n == 'Cosh': return mp.cosh(x)
    if n == 'Cos': return mp.cos(x)
    if n == 'Sinh': return mp.sinh(x)
    if n == 'Sin': return mp.sin(x)
    if n == 'Tanh': return mp.tanh(x)
    if n == 'Tan': return mp.tan(x)
    if n == 'ArcSinh': return mp.asinh(x)
    if n == 'ArcTanh': return mp.atanh(x) if abs(x) < 1 else None
    if n == 'ArcSin': return mp.asin(x) if -1 <= x <= 1 else None
    if n == 'ArcCos': return mp.acos(x) if -1 <= x <= 1 else None
    if n == 'ArcTan': return mp.atan(x)
    if n == 'ArcCosh': return mp.acosh(x) if x >= 1 else None
    if n == 'LogisticSigmoid': return 1/(1+mp.e**(-x))
    return None

def apply2(n,a,b):
    if n == 'EML': return mp.e**a - mp.log(b) if b > 0 else None
    if n == 'Plus': return a+b
    if n == 'Times': return a*b
    if n == 'Subtract': return a-b
    if n == 'Divide': return a/b if b != 0 else None
    if n == 'Power': return mp.e**(b*mp.log(a)) if a > 0 else None
    if n == 'Log': return mp.log(b)/mp.log(a) if (a > 0 and a != 1 and b > 0) else None
    if n == 'Avg': return (a+b)/2
    if n == 'Hypot': return mp.sqrt(a*a+b*b)
    return None

def apply3(n,a,b,c):
    if n == 'FMA': return a*b+c
    if n == 'FMS': return a*b-c
    if n == 'FNMA': return -(a*b)+c
    if n == 'FNMS': return -(a*b)-c
    if n == 'FSD': return (a-b)/c if c != 0 else None
    if n == 'TExpLog': return mp.exp(a)*(mp.log(c)/mp.log(b)) if (b > 0 and b != 1 and c > 0) else None
    if n == 'LogPowExpSub': return mp.exp(a-b)*(mp.log(c)/mp.log(a)) if (a > 0 and a != 1 and c > 0) else None
    return None

def eval_expr(e, env):
    if e[0] == 'atom':
        return env.get(e[1], const(e[1]))
    _, name, args = e
    vals = [eval_expr(a, env) for a in args]
    if any(v is None for v in vals):
        return None
    if len(vals) == 1: return apply1(name, vals[0])
    if len(vals) == 2: return apply2(name, vals[0], vals[1])
    if len(vals) == 3: return apply3(name, vals[0], vals[1], vals[2])
    return None

we = parse_expr(witness)
te = parse_expr(target)
if we is None or te is None:
    sys.exit(1)

pairs = [
  (mp.mpf('0.5772156649015328606'), mp.mpf('1.2824271291006226369')),   # +EulerGamma, +Glaisher
  (mp.mpf('-0.5772156649015328606'), mp.mpf('1.2824271291006226369')),  # -EulerGamma, +Glaisher
  (mp.mpf('0.91596559417721901505'), mp.mpf('2.6854520010653064453')),  # +Catalan, +Khinchin
  (mp.mpf('-0.91596559417721901505'), mp.mpf('2.6854520010653064453')), # -Catalan, +Khinchin
  (mp.mpf('1.2824271291006226369'), mp.mpf('0.5772156649015328606')),   # +Glaisher, +EulerGamma
  (mp.mpf('-1.2824271291006226369'), mp.mpf('0.5772156649015328606')),  # -Glaisher, +EulerGamma
  (mp.mpf('2.6854520010653064453'), mp.mpf('0.91596559417721901505')),  # +Khinchin, +Catalan
  (mp.mpf('-2.6854520010653064453'), mp.mpf('0.91596559417721901505')), # -Khinchin, +Catalan
]
triples = [
  (mp.mpf('0.5772156649015328606'), mp.mpf('1.2824271291006226369'), mp.mpf('2.6854520010653064453')),
  (mp.mpf('0.91596559417721901505'), mp.mpf('2.6854520010653064453'), mp.mpf('1.2824271291006226369')),
  (mp.mpf('-0.5772156649015328606'), mp.mpf('1.2824271291006226369'), mp.mpf('2.6854520010653064453')),
]
tol = mp.mpf('1e-60')

tested_pairs = 0
for x,y in pairs:
    env = {
      'EulerGamma': x,
      'Glaisher': y,
      'Catalan': mp.mpf('0.91596559417721901505'),
      'Khinchin': mp.mpf('2.6854520010653064453')
    }
    tv = eval_expr(te, env)
    if tv is None:
        continue
    wv = eval_expr(we, env)
    if wv is None:
        sys.exit(1)
    if not mp.isfinite(wv) or not mp.isfinite(tv) or abs(wv-tv) > tol:
        sys.exit(1)
    tested_pairs += 1
if tested_pairs == 0:
    sys.exit(4)

if kind == 'ternary':
    tested_triples = 0
    for x,y,z in triples:
        env = {'EulerGamma': x, 'Glaisher': y, 'Pi': z}
        tv = eval_expr(te, env)
        if tv is None:
            continue
        wv = eval_expr(we, env)
        if wv is None:
            sys.exit(1)
        if not mp.isfinite(wv) or not mp.isfinite(tv) or abs(wv-tv) > tol:
            sys.exit(1)
        tested_triples += 1
    if tested_triples == 0:
        sys.exit(4)
sys.exit(0)
"#;
    let Ok(out) = Command::new("python")
        .arg("-c")
        .arg(script)
        .arg(witness_expr)
        .arg(target_expr)
        .arg(kind)
        .output()
    else {
        // if python is unavailable, skip high-precision stage
        return true;
    };
    match out.status.code() {
        Some(0) => true,
        Some(4) => true, // complex-domain witness: skip real-only high-precision stage
        _ => false,      // genuine high-precision mismatch
    }
}

enum WitnessKind {
    Constant(String),
    Unary(String),
    Binary(String),
    Ternary(String),
}

fn validate_witness(
    kind: WitnessKind,
    witness_expr: &str,
    equiv: EquivCfg,
    use_highprec_python: bool,
    unary_all: &HashMap<String, Unary>,
    binary_all: &HashMap<&'static str, Binary>,
    ternary_all: &HashMap<&'static str, Ternary>,
    const_all: &HashMap<&'static str, C>,
) -> bool {
    let require_real_target_samples = !matches!(kind, WitnessKind::Constant(_));
    let target_expr = match &kind {
        WitnessKind::Constant(name) => name.clone(),
        WitnessKind::Unary(name) => format!("{name}[EulerGamma]"),
        WitnessKind::Binary(name) => format!("{name}[EulerGamma, Glaisher]"),
        WitnessKind::Ternary(name) => format!("{name}[EulerGamma, Glaisher, Pi]"),
    };
    let Some(w_ast) = parse_expr(witness_expr) else {
        return false;
    };
    let Some(t_ast) = parse_expr(&target_expr) else {
        return false;
    };

    // Primary witness probes use only the four transcendental anchors
    // {EulerGamma, Catalan, Glaisher, Khinchin} with sign-reflected variants.
    // Domain-invalid target samples are skipped below.
    let sample_pairs: Vec<(f64, f64)> = vec![
        (EULER_GAMMA, GLAISHER),
        (-EULER_GAMMA, GLAISHER),
        (CATALAN, KHINCHIN),
        (-CATALAN, KHINCHIN),
        (GLAISHER, EULER_GAMMA),
        (-GLAISHER, EULER_GAMMA),
        (KHINCHIN, CATALAN),
        (-KHINCHIN, CATALAN),
    ];
    let sample_triples: Vec<(f64, f64, f64)> = vec![
        (EULER_GAMMA, GLAISHER, KHINCHIN),
        (CATALAN, KHINCHIN, GLAISHER),
        (-EULER_GAMMA, GLAISHER, KHINCHIN),
    ];

    let mut real_only = true;
    let mut tested_pairs = 0usize;
    for (x, y) in &sample_pairs {
        let mut env_c: HashMap<String, C> = HashMap::new();
        env_c.insert("EulerGamma".to_string(), C::real(*x));
        env_c.insert("Glaisher".to_string(), C::real(*y));
        env_c.insert("Catalan".to_string(), C::real(CATALAN));
        env_c.insert("Khinchin".to_string(), C::real(KHINCHIN));
        let Some(tv) = eval_expr_c(&t_ast, &env_c, unary_all, binary_all, ternary_all, const_all) else {
            continue;
        };
        if require_real_target_samples && !imag_is_zero(tv, equiv) {
            continue;
        }
        let Some(wv) = eval_expr_c(&w_ast, &env_c, unary_all, binary_all, ternary_all, const_all) else {
            return false;
        };
        if !wv.is_finite() || !tv.is_finite() || !near(wv, tv, equiv) {
            return false;
        }
        tested_pairs += 1;
        if !imag_is_zero(wv, equiv) || !imag_is_zero(tv, equiv) {
            real_only = false;
        }
    }
    if tested_pairs == 0 {
        return false;
    }
    let mut tested_triples = 0usize;
    for (x, y, z) in &sample_triples {
        if !matches!(kind, WitnessKind::Ternary(_)) {
            break;
        }
        let mut env_c: HashMap<String, C> = HashMap::new();
        env_c.insert("EulerGamma".to_string(), C::real(*x));
        env_c.insert("Glaisher".to_string(), C::real(*y));
        env_c.insert("Pi".to_string(), C::real(*z));
        let Some(tv) = eval_expr_c(&t_ast, &env_c, unary_all, binary_all, ternary_all, const_all) else {
            continue;
        };
        if require_real_target_samples && !imag_is_zero(tv, equiv) {
            continue;
        }
        let Some(wv) = eval_expr_c(&w_ast, &env_c, unary_all, binary_all, ternary_all, const_all) else {
            return false;
        };
        if !wv.is_finite() || !tv.is_finite() || !near(wv, tv, equiv) {
            return false;
        }
        tested_triples += 1;
        if !imag_is_zero(wv, equiv) || !imag_is_zero(tv, equiv) {
            real_only = false;
        }
    }
    if matches!(kind, WitnessKind::Ternary(_)) && tested_triples == 0 {
        return false;
    }

    // Complex witnesses are accepted/rejected by multi-point complex checks only.
    if !real_only {
        return true;
    }

    // Real-only witnesses may get extra high-precision confirmation using Python/mpmath.
    if !use_highprec_python {
        return true;
    }

    // Real-only witnesses get strict high-precision confirmation using Python/mpmath.
    let kind_tag = match kind {
        WitnessKind::Constant(_) => "constant",
        WitnessKind::Unary(_) => "unary",
        WitnessKind::Binary(_) => "binary",
        WitnessKind::Ternary(_) => "ternary",
    };
    validate_witness_highprec_python(witness_expr, &target_expr, kind_tag)
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

fn find_representation_with_skip(
    target: C,
    constants: &[(String, C)],
    unary: &[(String, Unary)],
    binary: &[(String, Binary)],
    ternary: &[(String, Ternary)],
    max_k: usize,
    equiv: EquivCfg,
    domain: DomainMode,
    skip_expr: &HashSet<String>,
) -> Option<(String, usize)> {
    let mut levels: Vec<Vec<(C, String)>> = vec![vec![]; max_k + 1];
    let mut seen: HashSet<(i64, i64)> = HashSet::new();

    for (name, &c) in constants.iter().map(|(n, c)| (n, c)) {
        if !value_ok(c, domain, equiv) {
            continue;
        }
        if near(c, target, equiv) && skip_expr.contains(name) {
            // Do not let a rejected target witness consume the dedup slot.
            continue;
        }
        let key = qkey(c);
        if seen.insert(key) {
            levels[1].push((c, name.clone()));
            if near(c, target, equiv) && !skip_expr.contains(name) {
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
                        let expr = format!("{u_name}[{x_expr}]");
                        if near(y, target, equiv) {
                            if !skip_expr.contains(&expr) {
                                return Some((expr, k));
                            }
                            // Rejected target witness: do not poison dedup for this value.
                            continue;
                        }
                        let key = qkey(y);
                        if seen.insert(key) {
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
                                let expr = format!("{b_name}[{a_expr}, {bb_expr}]");
                                if near(y, target, equiv) {
                                    if !skip_expr.contains(&expr) {
                                        return Some((expr, k));
                                    }
                                    continue;
                                }
                                let key = qkey(y);
                                if seen.insert(key) {
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
                                            let expr = format!(
                                                "{t_name}[{a_expr}, {b_expr}, {c_expr}]"
                                            );
                                            if near(y, target, equiv) {
                                                if !skip_expr.contains(&expr) {
                                                    return Some((expr, k));
                                                }
                                                continue;
                                            }
                                            let key = qkey(y);
                                            if seen.insert(key) {
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

    let mut unary_all = unary_catalog();
    let binary_all = binary_catalog();
    let ternary_all = ternary_catalog();
    let const_all = constant_catalog();
    let scan_candidates = generated_candidates(&args.scan_g, &args.scan_h, &args.scan_params);
    for candidate in &scan_candidates {
        unary_all
            .entry(candidate.name.clone())
            .or_insert_with(|| candidate.unary.clone());
    }

    if let Some(name) = args.check_involution.as_deref() {
        let code = run_involution_check(name, &unary_all, args.equiv, args.involution_samples);
        process::exit(code);
    }

    if args.scan_family {
        let Ok(exe_path) = env::current_exe() else {
            eprintln!("Error: unable to locate current executable for scan mode.");
            process::exit(2);
        };
        println!(
            "Scan mode: {} generated candidates (g={}, h={}, params={})",
            scan_candidates.len(),
            args.scan_g,
            args.scan_h,
            args.scan_params
        );

        let mut rows: Vec<(usize, usize, usize, usize, f64, f64, String, String)> = Vec::new();
        for candidate in &scan_candidates {
            let Some(f) = unary_all.get(&candidate.name).cloned() else {
                continue;
            };
            let samples = involution_samples(args.involution_samples);
            let mut tested = 0usize;
            let mut passed = 0usize;
            for x in samples {
                let Some(y) = (f.f)(x) else {
                    continue;
                };
                let Some(z) = (f.f)(y) else {
                    continue;
                };
                tested += 1;
                if near(x, z, args.equiv) {
                    passed += 1;
                }
            }
            let inv_rate = if tested == 0 {
                0.0
            } else {
                passed as f64 / tested as f64
            };
            let valid_rate = tested as f64 / args.involution_samples as f64;

            let mut cmd = Command::new(&exe_path);
            cmd.arg("--constants")
                .arg("")
                .arg("--functions")
                .arg(&candidate.name)
                .arg("--operations")
                .arg("Plus,Times,Subtract")
                .arg("--max-k")
                .arg(args.max_k.to_string())
                .arg("--scan-g")
                .arg(&args.scan_g)
                .arg("--scan-h")
                .arg(&args.scan_h)
                .arg("--scan-params")
                .arg(&args.scan_params);
            if matches!(args.domain, DomainMode::Real) {
                cmd.arg("--domain").arg("real");
            }
            let output = match cmd.output() {
                Ok(v) => v,
                Err(_) => continue,
            };
            let stdout = String::from_utf8_lossy(&output.stdout);
            let mut rc = 999usize;
            let mut ru = 999usize;
            let mut rb = 999usize;
            for line in stdout.lines() {
                if line.starts_with("Remaining constants:") {
                    rc = parse_remaining_count(line).unwrap_or(999);
                } else if line.starts_with("Remaining unary:") {
                    ru = parse_remaining_count(line).unwrap_or(999);
                } else if line.starts_with("Remaining binary:") {
                    rb = parse_remaining_count(line).unwrap_or(999);
                }
            }
            let total = rc.saturating_add(ru).saturating_add(rb);
            rows.push((
                total,
                rc,
                ru,
                rb,
                inv_rate,
                valid_rate,
                candidate.name.clone(),
                candidate.formula.clone(),
            ));
            println!(
                "scan {}: rem=({}, {}, {}) inv_rate={:.2} valid_rate={:.2}",
                candidate.name, rc, ru, rb, inv_rate, valid_rate
            );
        }

        rows.sort_by(|a, b| {
            a.0.cmp(&b.0)
                .then_with(|| a.2.cmp(&b.2))
                .then_with(|| b.4.partial_cmp(&a.4).unwrap_or(std::cmp::Ordering::Equal))
        });
        println!("Top {} candidates:", args.scan_top.min(rows.len()));
        for row in rows.iter().take(args.scan_top) {
            println!(
                "{:<28} rem_total={:<3} rem(c/u/b)=({},{},{}) inv_rate={:.2} valid_rate={:.2} formula={}",
                row.6, row.0, row.1, row.2, row.3, row.4, row.5, row.7
            );
        }
        println!("Elapsed: {:.2?}", start.elapsed());
        return;
    }

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

    println!("Loaded base constants: {known_constants:?}");
    println!("Loaded base unary functions: {known_unary:?}");
    println!("Loaded base binary operations: {known_binary:?}");
    println!("Loaded base ternary operations: {known_ternary:?}");
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
                let mut rejected: HashSet<String> = HashSet::new();
                loop {
                    let Some(witness) = find_representation_with_skip(
                        target,
                        &named_constants,
                        &named_unary,
                        &named_binary,
                        &named_ternary,
                        k,
                        args.equiv,
                        args.domain,
                        &rejected,
                    ) else {
                        break;
                    };
                    let valid = if args.validate_witness {
                        validate_witness(
                            WitnessKind::Binary(op_name.clone()),
                            &witness.0,
                            args.equiv,
                            args.validate_witness_highprec,
                            &unary_all,
                            &binary_all,
                            &ternary_all,
                            &const_all,
                        )
                    } else {
                        true
                    };
                    if valid {
                        found_binary = Some((idx, Some(witness)));
                        break;
                    }
                    println!(
                        "Rejected flaky witness for binary operation: {op_name} at k={} -> {}",
                        witness.1, witness.0
                    );
                    rejected.insert(witness.0);
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
                let mut rejected: HashSet<String> = HashSet::new();
                loop {
                    let Some(witness) = find_representation_with_skip(
                        target,
                        &named_constants,
                        &named_unary,
                        &named_binary,
                        &named_ternary,
                        k,
                        args.equiv,
                        args.domain,
                        &rejected,
                    ) else {
                        break;
                    };
                    let valid = if args.validate_witness {
                        validate_witness(
                            WitnessKind::Ternary(op_name.clone()),
                            &witness.0,
                            args.equiv,
                            args.validate_witness_highprec,
                            &unary_all,
                            &binary_all,
                            &ternary_all,
                            &const_all,
                        )
                    } else {
                        true
                    };
                    if valid {
                        found_ternary = Some((idx, Some(witness)));
                        break;
                    }
                    println!(
                        "Rejected flaky witness for ternary operation: {op_name} at k={} -> {}",
                        witness.1, witness.0
                    );
                    rejected.insert(witness.0);
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
                let mut rejected: HashSet<String> = HashSet::new();
                loop {
                    let Some(witness) = find_representation_with_skip(
                        target,
                        &named_constants,
                        &named_unary,
                        &named_binary,
                        &named_ternary,
                        k,
                        args.equiv,
                        args.domain,
                        &rejected,
                    ) else {
                        break;
                    };
                    let valid = if args.validate_witness {
                        validate_witness(
                            WitnessKind::Constant(c_name.clone()),
                            &witness.0,
                            args.equiv,
                            args.validate_witness_highprec,
                            &unary_all,
                            &binary_all,
                            &ternary_all,
                            &const_all,
                        )
                    } else {
                        true
                    };
                    if valid {
                        found_constant = Some((idx, Some(witness)));
                        break;
                    }
                    println!(
                        "Rejected flaky witness for constant: {c_name} at k={} -> {}",
                        witness.1, witness.0
                    );
                    rejected.insert(witness.0);
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
                let mut rejected: HashSet<String> = HashSet::new();
                loop {
                    let Some(witness) = find_representation_with_skip(
                        target,
                        &named_constants,
                        &named_unary,
                        &named_binary,
                        &named_ternary,
                        k,
                        args.equiv,
                        args.domain,
                        &rejected,
                    ) else {
                        break;
                    };
                    let valid = if args.validate_witness {
                        validate_witness(
                            WitnessKind::Unary(f_name.clone()),
                            &witness.0,
                            args.equiv,
                            args.validate_witness_highprec,
                            &unary_all,
                            &binary_all,
                            &ternary_all,
                            &const_all,
                        )
                    } else {
                        true
                    };
                    if valid {
                        found_unary = Some((idx, Some(witness)));
                        break;
                    }
                    println!(
                        "Rejected flaky witness for unary function: {f_name} at k={} -> {}",
                        witness.1, witness.0
                    );
                    rejected.insert(witness.0);
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
