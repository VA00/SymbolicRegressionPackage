use std::collections::{HashMap, HashSet};
use std::env;
use std::f64::consts::{FRAC_PI_2, PI};
use std::time::Instant;

const GLAISHER: f64 = 1.282_427_129_100_622_6;
const EULER_GAMMA: f64 = 0.577_215_664_901_532_9;
const EPS: f64 = 1e-12;

#[derive(Debug)]
struct Args {
    constants: String,
    functions: String,
    operations: String,
    max_k: usize,
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
        max_k: 10,
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
            "--max-k" => {
                if let Some(v) = it.next() {
                    if let Ok(n) = v.parse::<usize>() {
                        args.max_k = n;
                    }
                }
            }
            _ => {}
        }
    }
    args
}

fn qkey(v: C) -> (i64, i64) {
    ((v.re * 1e12).round() as i64, (v.im * 1e12).round() as i64)
}

fn near(a: C, b: C) -> bool {
    a.sub(b).abs() <= EPS * (1.0 + a.abs() + b.abs())
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
    max_k: usize,
) -> bool {
    let mut levels: Vec<Vec<C>> = vec![vec![]; max_k + 1];
    let mut seen: HashSet<(i64, i64)> = HashSet::new();

    for &c in constants {
        if !c.is_finite() {
            continue;
        }
        let key = qkey(c);
        if seen.insert(key) {
            levels[1].push(c);
            if near(c, target) {
                return true;
            }
        }
    }

    for k in 2..=max_k {
        let mut next: Vec<C> = Vec::new();

        for u in unary {
            for &x in &levels[k - 1] {
                if let Some(y) = (u.f)(x) {
                    if y.is_finite() {
                        let key = qkey(y);
                        if seen.insert(key) {
                            if near(y, target) {
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
                            if y.is_finite() {
                                let key = qkey(y);
                                if seen.insert(key) {
                                    if near(y, target) {
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

        levels[k] = next;
    }

    false
}

fn main() {
    let args = parse_args();
    let start = Instant::now();

    let unary_all = unary_catalog();
    let binary_all = binary_catalog();
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

    let mut todo_constants = vec!["Glaisher", "EulerGamma", "Pi", "E", "I", "1", "-1", "2"]
        .into_iter()
        .map(ToOwned::to_owned)
        .collect::<Vec<_>>();
    todo_constants.retain(|c| !known_constants.contains(c));

    let mut todo_unary = vec![
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
    .collect::<Vec<_>>();
    todo_unary.retain(|f| !known_unary.contains(f));

    let mut todo_binary = vec![
        "Plus", "Times", "Subtract", "Divide", "Power", "Log", "Avg", "Hypot",
    ]
    .into_iter()
    .map(ToOwned::to_owned)
    .collect::<Vec<_>>();
    todo_binary.retain(|op| !known_binary.contains(op));

    let mut k = 1usize;
    while k <= args.max_k
        && (!todo_constants.is_empty() || !todo_unary.is_empty() || !todo_binary.is_empty())
    {
        println!("Testing with K = {k}");
        let mut new_item = false;

        let numeric_constants: Vec<C> = known_constants
            .iter()
            .filter_map(|name| const_all.get(name.as_str()).copied())
            .collect();
        let unary_set: Vec<Unary> = known_unary
            .iter()
            .filter_map(|name| unary_all.get(name.as_str()).cloned())
            .collect();
        let binary_set: Vec<Binary> = known_binary
            .iter()
            .filter_map(|name| binary_all.get(name.as_str()).cloned())
            .collect();

        if let Some(idx) = todo_binary.iter().position(|op_name| {
            let op = binary_all.get(op_name.as_str()).unwrap();
            let target = (op.f)(C::real(EULER_GAMMA), C::real(GLAISHER)).unwrap();
            can_represent(target, &numeric_constants, &unary_set, &binary_set, k)
        }) {
            let found = todo_binary.remove(idx);
            println!("Found binary operation: {found}");
            known_binary.push(found);
            known_binary.sort();
            known_binary.dedup();
            k = 1;
            new_item = true;
        }

        if new_item {
            continue;
        }

        if let Some(idx) = todo_constants.iter().position(|c_name| {
            let target = *const_all.get(c_name.as_str()).unwrap();
            can_represent(target, &numeric_constants, &unary_set, &binary_set, k)
        }) {
            let found = todo_constants.remove(idx);
            println!("Found constant: {found}");
            known_constants.push(found);
            known_constants.sort();
            known_constants.dedup();
            k = 1;
            new_item = true;
        }

        if new_item {
            continue;
        }

        if let Some(idx) = todo_unary.iter().position(|f_name| {
            let f = unary_all.get(f_name.as_str()).unwrap();
            let Some(target) = (f.f)(C::real(EULER_GAMMA)) else {
                return false;
            };
            can_represent(target, &numeric_constants, &unary_set, &binary_set, k)
        }) {
            let found = todo_unary.remove(idx);
            println!("Found unary function: {found}");
            known_unary.push(found);
            known_unary.sort();
            known_unary.dedup();
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
    println!("Remaining constants: {todo_constants:?}");
    println!("Remaining unary: {todo_unary:?}");
    println!("Remaining binary: {todo_binary:?}");
    println!("Complex-domain mode is enabled; I and complex inverse-trig paths are reachable.");
    println!("Elapsed: {:.2?}", start.elapsed());
}
