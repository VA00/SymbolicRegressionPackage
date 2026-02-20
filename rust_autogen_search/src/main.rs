use std::collections::HashMap;
use std::env;
use std::f64::consts::{FRAC_PI_2, PI};
use std::sync::Arc;
use std::time::Instant;

const GLAISHER: f64 = 1.282_427_129_100_622_6;
const EULER_GAMMA: f64 = 0.577_215_664_901_532_9;

#[derive(Clone, Copy, Debug)]
struct C {
    re: f64,
    im: f64,
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
    fn add(self, o: C) -> C {
        C::new(self.re + o.re, self.im + o.im)
    }
    fn sub(self, o: C) -> C {
        C::new(self.re - o.re, self.im - o.im)
    }
    fn mul(self, o: C) -> C {
        C::new(self.re * o.re - self.im * o.im, self.re * o.im + self.im * o.re)
    }
    fn div(self, o: C) -> Option<C> {
        let den = o.re * o.re + o.im * o.im;
        if den == 0.0 {
            return None;
        }
        Some(C::new(
            (self.re * o.re + self.im * o.im) / den,
            (self.im * o.re - self.re * o.im) / den,
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
        C::new(self.re.sin() * self.im.cosh(), self.re.cos() * self.im.sinh())
    }
    fn cos(self) -> C {
        C::new(self.re.cos() * self.im.cosh(), -self.re.sin() * self.im.sinh())
    }
    fn tan(self) -> Option<C> {
        self.sin().div(self.cos())
    }
    fn sinh(self) -> C {
        C::new(self.re.sinh() * self.im.cos(), self.re.cosh() * self.im.sin())
    }
    fn cosh(self) -> C {
        C::new(self.re.cosh() * self.im.cos(), self.re.sinh() * self.im.sin())
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
        Some(w.mul(self.ln()?).exp())
    }
    fn asin(self) -> Option<C> {
        let i = C::i();
        let one = C::real(1.0);
        let iz = i.mul(self);
        let inside = one.sub(self.mul(self)).sqrt();
        Some(i.neg().mul(iz.add(inside).ln()?))
    }
    fn acos(self) -> Option<C> {
        Some(C::real(FRAC_PI_2).sub(self.asin()?))
    }
    fn atan(self) -> Option<C> {
        let i = C::i();
        let one = C::real(1.0);
        let l1 = one.sub(i.mul(self)).ln()?;
        let l2 = one.add(i.mul(self)).ln()?;
        Some(C::new(0.0, 0.5).mul(l1.sub(l2)))
    }
    fn asinh(self) -> Option<C> {
        let one = C::real(1.0);
        self.add(self.mul(self).add(one).sqrt()).ln()
    }
    fn acosh(self) -> Option<C> {
        let one = C::real(1.0);
        self.add(self.add(one).sqrt().mul(self.sub(one).sqrt())).ln()
    }
    fn atanh(self) -> Option<C> {
        let one = C::real(1.0);
        Some(one.add(self).ln()?.sub(one.sub(self).ln()?).mul(C::real(0.5)))
    }
}

type F = Arc<dyn Fn(&[C]) -> Option<C> + Send + Sync>;

#[derive(Clone)]
struct Cand {
    expr: String,
    f: F,
    deps: u8,
}

struct GenOutput {
    pool: Vec<Cand>,
    discarded_by_cap: usize,
    total_unique_all_deps: usize,
}

fn empty_gen_output() -> GenOutput {
    GenOutput {
        pool: Vec::new(),
        discarded_by_cap: 0,
        total_unique_all_deps: 0,
    }
}

#[derive(Clone)]
struct Unary {
    f: Arc<dyn Fn(C) -> Option<C> + Send + Sync>,
}
#[derive(Clone)]
struct Binary {
    f: Arc<dyn Fn(C, C) -> Option<C> + Send + Sync>,
    commutative: bool,
}
#[derive(Clone)]
struct Ternary {
    f: Arc<dyn Fn(C, C, C) -> Option<C> + Send + Sync>,
}

fn near(a: C, b: C) -> bool {
    a.sub(b).abs() <= 16.0 * f64::EPSILON * (1.0 + a.abs() + b.abs())
}
fn qkey(v: C) -> (i64, i64) {
    ((v.re * 1e10).round() as i64, (v.im * 1e10).round() as i64)
}

fn sample_inputs(arity: usize) -> Vec<Vec<C>> {
    let s1 = vec![
        C::real(-1.3),
        C::real(-0.4),
        C::real(0.3),
        C::real(0.9),
        C::real(2.1),
        C::new(0.8, 0.2),
    ];
    match arity {
        0 => vec![vec![]],
        1 => s1.into_iter().map(|x| vec![x]).collect(),
        2 => vec![
            vec![C::real(EULER_GAMMA), C::real(GLAISHER)],
            vec![C::real(1.3), C::real(0.7)],
            vec![C::new(0.8, 0.2), C::real(1.1)],
            vec![C::real(-0.7), C::new(0.4, 0.9)],
        ],
        _ => vec![
            vec![C::real(0.8), C::real(1.2), C::real(2.3)],
            vec![C::new(0.6, 0.1), C::real(0.9), C::new(1.1, -0.2)],
        ],
    }
}

fn signature(c: &Cand, arity: usize) -> Option<Vec<(i64, i64)>> {
    let mut out = Vec::new();
    for args in sample_inputs(arity) {
        let v = (c.f)(&args)?;
        if !v.is_finite() {
            return None;
        }
        out.push(qkey(v));
    }
    Some(out)
}

fn unary_ops() -> Vec<(&'static str, Arc<dyn Fn(C) -> Option<C> + Send + Sync>)> {
    vec![
        ("Half", Arc::new(|x| Some(x.mul(C::real(0.5))))),
        ("Minus", Arc::new(|x| Some(x.neg()))),
        ("Log", Arc::new(|x| x.ln())),
        ("Exp", Arc::new(|x| Some(x.exp()))),
        ("Inv", Arc::new(|x| C::real(1.0).div(x))),
        ("Sqrt", Arc::new(|x| Some(x.sqrt()))),
        ("Sqr", Arc::new(|x| Some(x.mul(x)))),
        ("Cosh", Arc::new(|x| Some(x.cosh()))),
        ("Cos", Arc::new(|x| Some(x.cos()))),
        ("Sinh", Arc::new(|x| Some(x.sinh()))),
        ("Sin", Arc::new(|x| Some(x.sin()))),
        ("Tanh", Arc::new(|x| x.tanh())),
        ("Tan", Arc::new(|x| x.tan())),
        ("ArcSinh", Arc::new(|x| x.asinh())),
        ("ArcTanh", Arc::new(|x| x.atanh())),
        ("ArcSin", Arc::new(|x| x.asin())),
        ("ArcCos", Arc::new(|x| x.acos())),
        ("ArcTan", Arc::new(|x| x.atan())),
        ("ArcCosh", Arc::new(|x| x.acosh())),
    ]
}

fn binary_ops() -> Vec<(&'static str, bool, Arc<dyn Fn(C, C) -> Option<C> + Send + Sync>)> {
    vec![
        ("Plus", true, Arc::new(|a, b| Some(a.add(b)))),
        ("Times", true, Arc::new(|a, b| Some(a.mul(b)))),
        ("Subtract", false, Arc::new(|a, b| Some(a.sub(b)))),
        ("Divide", false, Arc::new(|a, b| a.div(b))),
        ("Power", false, Arc::new(|a, b| a.pow(b))),
        ("Log", false, Arc::new(|base, x| x.ln()?.div(base.ln()?))),
    ]
}

fn ternary_ops() -> Vec<(&'static str, Arc<dyn Fn(C, C, C) -> Option<C> + Send + Sync>)> {
    vec![("FSD", Arc::new(|a, b, c| a.sub(b).div(c)))]
}

fn generation_seed_constants() -> Vec<(&'static str, C)> {
    vec![
        ("Pi", C::real(PI)),
        ("E", C::real(std::f64::consts::E)),
        ("I", C::i()),
        ("1", C::real(1.0)),
        ("-1", C::real(-1.0)),
        ("2", C::real(2.0)),
    ]
}

fn generate_candidates(arity: usize, gen_k: usize, max_keep: usize) -> GenOutput {
    let mut levels: Vec<Vec<Cand>> = vec![vec![]; gen_k + 1];
    let mut seen: HashMap<Vec<(i64, i64)>, Cand> = HashMap::new();
    let mut discarded_by_cap = 0usize;
    for var_i in 0..arity {
        let mut deps = 0u8;
        deps |= 1 << var_i;
        let f = Arc::new(move |args: &[C]| Some(args[var_i]));
        let c = Cand {
            expr: format!("v{var_i}"),
            f,
            deps,
        };
        if let Some(sig) = signature(&c, arity) {
            seen.insert(sig, c.clone());
            levels[1].push(c);
        }
    }
    for (n, v) in generation_seed_constants() {
        let vv = v;
        let c = Cand {
            expr: n.to_string(),
            f: Arc::new(move |_| Some(vv)),
            deps: 0,
        };
        if let Some(sig) = signature(&c, arity) {
            seen.insert(sig, c.clone());
            levels[1].push(c);
        }
    }

    for k in 2..=gen_k {
        let mut next: Vec<Cand> = Vec::new();
        for u in unary_ops() {
            for a in &levels[k - 1] {
                let f0 = a.f.clone();
                let op = u.1.clone();
                let cand = Cand {
                    expr: format!("{}({})", u.0, a.expr),
                    f: Arc::new(move |args| op(f0(args)?) ),
                    deps: a.deps,
                };
                if let Some(sig) = signature(&cand, arity) {
                    if !seen.contains_key(&sig) {
                        seen.insert(sig, cand.clone());
                        next.push(cand);
                    }
                }
            }
        }
        for b in binary_ops() {
            for lk in 1..k - 1 {
                let rk = k - 1 - lk;
                for a in &levels[lk] {
                    for bb in &levels[rk] {
                        if b.1 && lk == rk && a.expr > bb.expr {
                            continue;
                        }
                        let fa = a.f.clone();
                        let fb = bb.f.clone();
                        let op = b.2.clone();
                        let cand = Cand {
                            expr: format!("{}({}, {})", b.0, a.expr, bb.expr),
                            f: Arc::new(move |args| op(fa(args)?, fb(args)?)),
                            deps: a.deps | bb.deps,
                        };
                        if let Some(sig) = signature(&cand, arity) {
                            if !seen.contains_key(&sig) {
                                seen.insert(sig, cand.clone());
                                next.push(cand);
                            }
                        }
                    }
                }
            }
        }
        if k >= 4 {
            for t in ternary_ops() {
                for a_k in 1..=k - 3 {
                    for b_k in 1..=k - 2 - a_k {
                        let c_k = k - 1 - a_k - b_k;
                        if c_k < 1 {
                            continue;
                        }
                        for a in &levels[a_k] {
                            for b in &levels[b_k] {
                                for c in &levels[c_k] {
                                    let fa = a.f.clone();
                                    let fb = b.f.clone();
                                    let fc = c.f.clone();
                                    let op = t.1.clone();
                                    let cand = Cand {
                                        expr: format!("{}({}, {}, {})", t.0, a.expr, b.expr, c.expr),
                                        f: Arc::new(move |args| op(fa(args)?, fb(args)?, fc(args)?)),
                                        deps: a.deps | b.deps | c.deps,
                                    };
                                    if let Some(sig) = signature(&cand, arity) {
                                        if !seen.contains_key(&sig) {
                                            seen.insert(sig, cand.clone());
                                            next.push(cand);
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        if next.len() > max_keep {
            discarded_by_cap += next.len() - max_keep;
            next.sort_by(|a, b| {
                let da = a.deps.count_ones() as usize;
                let db = b.deps.count_ones() as usize;
                db.cmp(&da)
                    .then_with(|| a.expr.len().cmp(&b.expr.len()))
                    .then_with(|| a.expr.cmp(&b.expr))
            });
            next.truncate(max_keep);
        }
        levels[k] = next;
    }

    let need = if arity == 0 { 0 } else { (1u8 << arity) - 1 };
    let total_unique_all_deps = seen.len();
    let mut out: Vec<Cand> = seen
        .into_values()
        .filter(|c| c.deps == need)
        .collect();
    out.sort_by(|a, b| a.expr.len().cmp(&b.expr.len()).then_with(|| a.expr.cmp(&b.expr)));
    GenOutput {
        pool: out,
        discarded_by_cap,
        total_unique_all_deps,
    }
}

fn can_represent(
    target: C,
    constants: &[C],
    unary: &[Unary],
    binary: &[Binary],
    ternary: &[Ternary],
    max_k: usize,
) -> bool {
    let mut levels: Vec<Vec<C>> = vec![vec![]; max_k + 1];
    let mut seen: HashMap<(i64, i64), bool> = HashMap::new();
    for &c in constants {
        if !c.is_finite() {
            continue;
        }
        let key = qkey(c);
        if seen.insert(key, true).is_none() {
            levels[1].push(c);
            if near(c, target) {
                return true;
            }
        }
    }
    for k in 2..=max_k {
        let mut next = Vec::new();
        for u in unary {
            for &x in &levels[k - 1] {
                if let Some(y) = (u.f)(x) {
                    if y.is_finite() {
                        let key = qkey(y);
                        if seen.insert(key, true).is_none() {
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
            for lk in 1..k - 1 {
                let rk = k - 1 - lk;
                for &a in &levels[lk] {
                    for &bb in &levels[rk] {
                        if b.commutative && lk == rk && qkey(a) > qkey(bb) {
                            continue;
                        }
                        if let Some(y) = (b.f)(a, bb) {
                            if y.is_finite() {
                                let key = qkey(y);
                                if seen.insert(key, true).is_none() {
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
        if k >= 4 {
            for t in ternary {
                for ak in 1..=k - 3 {
                    for bk in 1..=k - 2 - ak {
                        let ck = k - 1 - ak - bk;
                        if ck < 1 {
                            continue;
                        }
                        for &a in &levels[ak] {
                            for &b in &levels[bk] {
                                for &c in &levels[ck] {
                                    if let Some(y) = (t.f)(a, b, c) {
                                        if y.is_finite() {
                                            let key = qkey(y);
                                            if seen.insert(key, true).is_none() {
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
                }
            }
        }
        levels[k] = next;
    }
    false
}

#[derive(Clone)]
struct Family {
    name: &'static str,
    const_targets: Vec<C>,
    unary_targets: Vec<Unary>,
    binary_targets: Vec<Binary>,
}

fn completeness_families() -> Vec<Family> {
    vec![
        Family {
            name: "S1_ExpLog_Subtract",
            const_targets: vec![],
            unary_targets: vec![
                Unary { f: Arc::new(|x| Some(x.exp())) },
                Unary { f: Arc::new(|x| x.ln()) },
            ],
            binary_targets: vec![Binary {
                f: Arc::new(|a, b| Some(a.sub(b))),
                commutative: false,
            }],
        },
        Family {
            name: "S2_E_Power_Log",
            const_targets: vec![C::real(std::f64::consts::E)],
            unary_targets: vec![],
            binary_targets: vec![
                Binary { f: Arc::new(|a, b| a.pow(b)), commutative: false },
                Binary {
                    f: Arc::new(|base, x| x.ln()?.div(base.ln()?)),
                    commutative: false,
                },
            ],
        },
        Family {
            name: "S3_Exp_LogBinary",
            const_targets: vec![],
            unary_targets: vec![Unary {
                f: Arc::new(|x| Some(x.exp())),
            }],
            binary_targets: vec![
                Binary {
                    f: Arc::new(|base, x| x.ln()?.div(base.ln()?)),
                    commutative: false,
                },
            ],
        },
        Family {
            name: "S4_Cosh_ArcCosh_Divide",
            const_targets: vec![],
            unary_targets: vec![
                Unary { f: Arc::new(|x| Some(x.cosh())) },
                Unary { f: Arc::new(|x| x.acosh()) },
            ],
            binary_targets: vec![Binary {
                f: Arc::new(|a, b| a.div(b)),
                commutative: false,
            }],
        },
    ]
}

fn satisfies_any_family(constants: &[C], unary: &[Unary], binary: &[Binary], ternary: &[Ternary], k: usize) -> Option<&'static str> {
    for fam in completeness_families() {
        let mut ok = true;
        for c in fam.const_targets {
            if !can_represent(c, constants, unary, binary, ternary, k) {
                ok = false;
                break;
            }
        }
        if !ok {
            continue;
        }
        for u in &fam.unary_targets {
            let Some(target) = (u.f)(C::real(EULER_GAMMA)) else {
                ok = false;
                break;
            };
            if !can_represent(target, constants, unary, binary, ternary, k) {
                ok = false;
                break;
            }
        }
        if !ok {
            continue;
        }
        for b in &fam.binary_targets {
            let Some(target) = (b.f)(C::real(EULER_GAMMA), C::real(GLAISHER)) else {
                ok = false;
                break;
            };
            if !can_represent(target, constants, unary, binary, ternary, k) {
                ok = false;
                break;
            }
        }
        if ok {
            return Some(fam.name);
        }
    }
    None
}

#[derive(Clone, Copy)]
struct RunProfiles {
    p0: bool,
    pa: bool,
    pb: bool,
    pc: bool,
}

impl RunProfiles {
    fn all() -> Self {
        Self {
            p0: true,
            pa: true,
            pb: true,
            pc: true,
        }
    }
}

fn parse_profiles() -> RunProfiles {
    let mut selected: Vec<String> = Vec::new();
    let argv: Vec<String> = env::args().skip(1).collect();
    let mut i = 0usize;
    while i < argv.len() {
        match argv[i].as_str() {
            "--profile" => {
                i += 1;
                if i < argv.len() {
                    selected.push(argv[i].clone());
                    i += 1;
                }
            }
            "--profiles" => {
                i += 1;
                if i < argv.len() {
                    selected.extend(
                        argv[i]
                            .split(',')
                            .map(str::trim)
                            .filter(|s| !s.is_empty())
                            .map(ToOwned::to_owned),
                    );
                    i += 1;
                }
            }
            _ => {
                i += 1;
            }
        }
    }

    if selected.is_empty() {
        return RunProfiles::all();
    }

    let mut out = RunProfiles {
        p0: false,
        pa: false,
        pb: false,
        pc: false,
    };
    for s in selected {
        match s.as_str() {
            "0" | "P0" | "p0" => out.p0 = true,
            "A" | "a" | "PA" | "pa" => out.pa = true,
            "B" | "b" | "PB" | "pb" => out.pb = true,
            "C" | "c" | "PC" | "pc" => out.pc = true,
            _ => {}
        }
    }
    out
}

fn main() {
    let start = Instant::now();
    let gen_k = 7usize;
    let verify_k = 6usize;
    let max_keep_per_level = 100_000_000usize;
    //Profile 0: 0 const, 1 unary, 1 binary
    let profile0_unary_take = 1usize;
    let profile0_binary_take = 1usize;
    //Profile A: 1 const, 0 unary, 1 binary
    let profile_a_const_take = 128usize;
    //let profile_a_binary_take = 120usize;
    let profile_a_binary_take = 128usize;
    //Profile B: 0 const, 0 unary, 1 binary
    let profile_b_binary_take = 200usize;
    //Profile C: 0 const, 0 unary, 0 binary, 1 ternary
    let profile_c_ternary_take = usize::MAX;
    let run = parse_profiles();

    println!("Generating unnamed primitives...");
    let const_out = if run.pa {
        generate_candidates(0, 1, max_keep_per_level)
    } else {
        empty_gen_output()
    };
    let unary_out = if run.p0 {
        generate_candidates(1, gen_k, max_keep_per_level)
    } else {
        empty_gen_output()
    };
    let binary_out = if run.p0 || run.pa || run.pb {
        generate_candidates(2, gen_k, max_keep_per_level)
    } else {
        empty_gen_output()
    };
    let ternary_out = if run.pc {
        generate_candidates(3, gen_k, max_keep_per_level)
    } else {
        empty_gen_output()
    };
    let const_pool = const_out.pool;
    let unary_pool = unary_out.pool;
    let binary_pool = binary_out.pool;
    let ternary_pool = ternary_out.pool;
    println!(
        "Pool sizes: const={}, unary={}, binary={}, ternary={}",
        const_pool.len(),
        unary_pool.len(),
        binary_pool.len(),
        ternary_pool.len()
    );
    println!(
        "Profile limits: p0 unary={} binary={}, pA const={} binary={}, pB binary={}, pC ternary={}",
        profile0_unary_take,
        profile0_binary_take,
        profile_a_const_take,
        profile_a_binary_take,
        profile_b_binary_take,
        profile_c_ternary_take
    );
    println!(
        "Discarded by cap(max_keep_per_level={}): const={}, unary={}, binary={}, ternary={}",
        max_keep_per_level,
        const_out.discarded_by_cap,
        unary_out.discarded_by_cap,
        binary_out.discarded_by_cap,
        ternary_out.discarded_by_cap
    );
    println!(
        "Unique before arity-dependency filter: const={}, unary={}, binary={}, ternary={}",
        const_out.total_unique_all_deps,
        unary_out.total_unique_all_deps,
        binary_out.total_unique_all_deps,
        ternary_out.total_unique_all_deps
    );
    println!(
        "Enabled profiles: 0={} A={} B={} C={}",
        run.p0, run.pa, run.pb, run.pc
    );

    let placeholder_constants = vec![C::real(EULER_GAMMA), C::real(GLAISHER)];
    let mut hits = 0usize;
    let mut hits_p0 = 0usize;
    let mut hits_pa = 0usize;
    let mut hits_pb = 0usize;
    let mut hits_pc = 0usize;

    if run.p0 {
        println!("Profile 0: 0 const, 1 unary, 1 binary");
        let p0_u_n = unary_pool.len().min(profile0_unary_take);
        let p0_b_n = binary_pool.len().min(profile0_binary_take);
        println!(
            "Profile 0 coverage: unary {} of {}, binary {} of {}, tested pairs {} of {}",
            p0_u_n,
            unary_pool.len(),
            p0_b_n,
            binary_pool.len(),
            p0_u_n.saturating_mul(p0_b_n),
            unary_pool.len().saturating_mul(binary_pool.len())
        );
        let has_exp = unary_pool.iter().any(|u| u.expr == "Exp(v0)");
        let has_log = binary_pool.iter().any(|b| b.expr == "Log(v0, v1)");
        if has_exp && has_log {
            let constants = vec![placeholder_constants[0], placeholder_constants[1]];
            let unary = vec![Unary {
                f: Arc::new(|x| Some(x.exp())),
            }];
            let binary = vec![Binary {
                f: Arc::new(|base, x| x.ln()?.div(base.ln()?)),
                commutative: false,
            }];
            if let Some(name) = satisfies_any_family(&constants, &unary, &binary, &[], verify_k) {
                hits += 1;
                println!("ASSERTION HIT[{name}] unary=Exp(v0) | binary=Log(v0, v1)");
            } else {
                println!("ASSERTION FAILED: Exp(v0)+Log(v0,v1) not recognized");
            }
        } else {
            println!(
                "ASSERTION SKIPPED: missing generated primitive(s): has Exp(v0)={}, has Log(v0,v1)={}",
                has_exp, has_log
            );
        }
        for u in unary_pool.iter().take(profile0_unary_take) {
            for b in binary_pool.iter().take(profile0_binary_take) {
                let constants = vec![placeholder_constants[0], placeholder_constants[1]];
                let unary = vec![Unary {
                    f: Arc::new({
                        let f = u.f.clone();
                        move |x| f(&[x])
                    }),
                }];
                let binary = vec![Binary {
                    f: Arc::new({
                        let f = b.f.clone();
                        move |x, y| f(&[x, y])
                    }),
                    commutative: false,
                }];
                if let Some(name) = satisfies_any_family(&constants, &unary, &binary, &[], verify_k) {
                    hits += 1;
                    hits_p0 += 1;
                    println!("HIT[{name}] unary={} | binary={}", u.expr, b.expr);
                }
            }
        }
        println!("Profile 0 hits: {hits_p0}");
    }

    if run.pa {
        println!("Profile A: 1 const, 0 unary, 1 binary");
        let pa_c_n = const_pool.len().min(profile_a_const_take);
        let pa_b_n = binary_pool.len().min(profile_a_binary_take);
        println!(
            "Profile A coverage: const {} of {}, binary {} of {}, tested pairs {} of {}",
            pa_c_n,
            const_pool.len(),
            pa_b_n,
            binary_pool.len(),
            pa_c_n.saturating_mul(pa_b_n),
            const_pool.len().saturating_mul(binary_pool.len())
        );
        for c in const_pool.iter().take(profile_a_const_take) {
            let Some(cv) = (c.f)(&[]) else { continue };
            for b in binary_pool.iter().take(profile_a_binary_take) {
                let constants = vec![placeholder_constants[0], placeholder_constants[1], cv];
                let unary = vec![];
                let binary = vec![Binary {
                    f: Arc::new({
                        let f = b.f.clone();
                        move |x, y| f(&[x, y])
                    }),
                    commutative: false,
                }];
                if let Some(name) = satisfies_any_family(&constants, &unary, &binary, &[], verify_k) {
                    hits += 1;
                    hits_pa += 1;
                    println!("HIT[{name}] const={} | binary={}", c.expr, b.expr);
                }
            }
        }
        println!("Profile A hits: {hits_pa}");
    }

    if run.pb {
        println!("Profile B: 0 const, 0 unary, 1 binary");
        let pb_b_n = binary_pool.len().min(profile_b_binary_take);
        println!(
            "Profile B coverage: binary {} of {}",
            pb_b_n,
            binary_pool.len()
        );
        for b in binary_pool.iter().take(profile_b_binary_take) {
            let constants = vec![placeholder_constants[0], placeholder_constants[1]];
            let unary = vec![];
            let binary = vec![Binary {
                f: Arc::new({
                    let f = b.f.clone();
                    move |x, y| f(&[x, y])
                }),
                commutative: false,
            }];
            if let Some(name) = satisfies_any_family(&constants, &unary, &binary, &[], verify_k) {
                hits += 1;
                hits_pb += 1;
                println!("HIT[{name}] binary={}", b.expr);
            }
        }
        println!("Profile B hits: {hits_pb}");
    }

    if run.pc {
        println!("Profile C: 0 const, 0 unary, 0 binary, 1 ternary");
        let pc_t_n = ternary_pool.len().min(profile_c_ternary_take);
        println!(
            "Profile C coverage: ternary {} of {}",
            pc_t_n,
            ternary_pool.len()
        );
        for t in ternary_pool.iter().take(profile_c_ternary_take) {
            let constants = vec![placeholder_constants[0], placeholder_constants[1]];
            let ternary = vec![Ternary {
                f: Arc::new({
                    let f = t.f.clone();
                    move |a, b, c| f(&[a, b, c])
                }),
            }];
            if let Some(name) = satisfies_any_family(&constants, &[], &[], &ternary, verify_k) {
                hits += 1;
                hits_pc += 1;
                println!("HIT[{name}] ternary={}", t.expr);
            }
        }
        println!("Profile C hits: {hits_pc}");
    }

    println!("Done. hits={hits}, elapsed={:.2?}", start.elapsed());
    println!("This is a starter project; tune gen_k/max_keep/profile loops for deeper search.");
}
