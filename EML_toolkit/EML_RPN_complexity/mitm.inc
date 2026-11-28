// mitm.inc - the search above the table (meet in the middle). Included by eml_mitm.cpp before main().
//
// A term with n leaves, n above the table depth NT, is a "spine" of nodes with more than NT leaves; at each
// spine node one child is a table entry (the "side") and the other child is the next spine node, down to a
// node whose two children are both table entries (the "bottom pair"). The target fixes what the next spine
// node must be, step by step ("request"):
//     value request  (KV): the node has the value r
//     exp request    (KE): exp(node) = r
//   KV, side a on the left  (node = exp(a) - Log(X)):  X = exp(exp(a) - r), if Im(exp(a) - r) is in (-pi, pi]
//   KV, side b on the right (node = exp(X) - Log(b)):  exp(X) = r + Log(b)
//   KE, side a on the left:                            X = exp(exp(a)) / r
//   KE, side b on the right:                           exp(exp(X)) = r*b; taken on the principal branch,
//                                                      exp(X) = Log(r) + Log(b) reduced to Im in (-pi, pi]
// (the last line is the one place where a branch is chosen; with --real it is exact, and with --exact every
// branch is covered: see "terms run forward" below).
// The bottom pair is found in one of two ways:
//   B: for every entry of the smaller child the request for the larger child is looked up in the table
//      (by value, by exp(value), or by exp(value) modulo 2 pi i);
//   S: the requests for nodes of a "zone" of sizes are stored, and all pairs exp(a) - Log(b) of table
//      entries of these sizes are run against the store; the terms are recovered in a second run.
// Sieve: double precision with a bound on the error carried along; a request whose bound is above TAU is
// recomputed (long double, quadruple, MPFR at PMAX bits). Every hit is recomputed in quadruple precision
// and then the whole term is evaluated from its leaves in tier 3 and compared with the target at PMAX bits.

struct Zd { double re, im; };
static const double U53 = 1.1102230246251565e-16;
static const double TWOPI_D = 6.283185307179586476925286766559;

static int M_NT = 0, M_NMAX = 0, M_ZONE = -1, M_ESAMPLE = 0, M_SBITS = 31, M_SHOW = 3, M_NMIN = 0, M_KWIN = 0;
static bool M_PRUNE = true;      // a node whose value is a value of a smaller level of the table is not followed (--noprune: it is)
static bool M_EXACT = false;     // --exact: every branch of exp(exp(X)) = c, by running all terms X forward (X up to n-2 leaves)
static bool M_CLEAN = false;     // --clean: no term with structure below the resolution (no S entry, no absorbed operand)
static int M_F32 = 0;            // --f32 1|2: (with --exact --clean) the terms run forward go through a sieve in single precision first: 1 on the processor, 2 on the graphics card
static bool M_F32CHECK = false;  // --f32check: both ways are run, and what the sieve leads to must contain what the plain way finds
static double M_F32REL = 7.8125e-3;          // 2^-7: largest relative error of a number in the sieve (above it the case is handed back)
static double M_TAU = 5.6843418860808015e-14;        // 2^-44: largest relative error bound of a request in double
static const double M_TOLF = 2.842170943040401e-14;  // 2^-45: largest relative error of a pair value in double
static const double M_ACC = 8.881784197001252e-16;   // 2^-50: what a recomputed request must reach
static const double M_STINY = 1e-17;                 // a zone request below this is handled by table look-ups (B)
static const int MBC = 30;                           // bits of a cell of the look-up tables
static const int MAXD = 28;
enum { KV = 0, KE = 1 };
enum { PK_V = 0, PK_E = 1, PK_ER = 2 };
enum { ST_OK = 0, ST_DROP = 1, ST_DOUBT = 2, ST_FAIL = 3 };
static const int T_BAD = 7;      // (tier 3) the term takes Log of 0 or leaves the range: not a term of the class
enum { MF_E = 1, MF_L = 2, MF_EE = 4, MF_ZERO = 8, MF_S = 16, MF_D = 32 };
static const double MAB = 5.048709793414476e-29;     // 2^-94: an operand below this part of the other is "below the resolution" (as in the table)
static const double MDLIM = 7.9e-31;                 // 2^-100: displacements below this part of a value are compared as displacements
enum { MODE_GEN = 0, MODE_REGEN = 1 };
enum { BOT_ENTRY = 0, BOT_PAIR = 1 };

struct MLev { size_t n = 0; std::vector<Zd> V, E, L, EE; std::vector<uint8_t> f, cE, cL; };     // cE, cL: marks of exactness of exp and Log (2 bits per probe)
static std::vector<MLev> ml;

static inline double asumd(const Zd& z) { return std::fabs(z.re) + std::fabs(z.im); }
static inline double amxd(const Zd& z) { return std::max(std::fabs(z.re), std::fabs(z.im)); }
static inline Zd zmul(const Zd& a, const Zd& b) { return Zd{ a.re * b.re - a.im * b.im, a.re * b.im + a.im * b.re }; }
static inline Zd zinv(const Zd& b)
{
    double s = amxd(b), br = b.re / s, bi = b.im / s, d = (br * br + bi * bi) * s;
    return Zd{ br / d, -bi / d };
}
static inline Zd zlog(const Zd& z) { return Zd{ std::log(std::hypot(z.re, z.im)), std::atan2(z.im, z.re) }; }
static inline Zd toZd(const C& z) { return Zd{ (double)z.re, (double)(PIq * z.y) }; }
static inline Zd zdivd(const Zd& a, const Zd& b) { return amxd(b) > 0 ? zmul(a, zinv(b)) : Zd{ 0, 0 }; }
static inline Zd dZd(const D& d) { return Zd{ d.re, PI_D * d.y }; }
static inline uint64_t dbits(double x) { uint64_t b; memcpy(&b, &x, 8); return b; }
static inline int ilogb_fast(double m) { return (int)((dbits(m) >> 52) & 0x7ff) - 1023; }
static inline double pow2_fast(int k) { uint64_t b = (uint64_t)(k + 1023) << 52; double d; memcpy(&d, &b, 8); return d; }
// An operand of a sum is "below the resolution" of the other one when the binary exponents of their sizes differ by
// more than FS: the rule of the table (eval2), so that a request and a table entry are split into digits and
// displacement in the same way. lo, big: the sizes (max of |re|, |im|), lo <= big.
static inline bool absb(double lo, double big) { return lo > 1e-300 && big < 1e300 && ilogb_fast(lo) - ilogb_fast(big) < -FS; }

// ---------------------------------------------------------------- numbers of several precisions

template <class R> struct Cx { R re, im; };

struct MP {
    mpfr_t v;
    static mpfr_prec_t& prec() { static thread_local mpfr_prec_t p = 2304; return p; }
    MP() { mpfr_init2(v, prec()); mpfr_set_zero(v, 1); }
    MP(double d) { mpfr_init2(v, prec()); mpfr_set_d(v, d, MPFR_RNDN); }
    MP(const MP& o) { mpfr_init2(v, prec()); mpfr_set(v, o.v, MPFR_RNDN); }
    MP& operator=(const MP& o) { if (this != &o) mpfr_set(v, o.v, MPFR_RNDN); return *this; }
    ~MP() { mpfr_clear(v); }
    explicit operator double() const { return mpfr_get_d(v, MPFR_RNDN); }
};
static inline MP operator+(const MP& a, const MP& b) { MP r; mpfr_add(r.v, a.v, b.v, MPFR_RNDN); return r; }
static inline MP operator-(const MP& a, const MP& b) { MP r; mpfr_sub(r.v, a.v, b.v, MPFR_RNDN); return r; }
static inline MP operator*(const MP& a, const MP& b) { MP r; mpfr_mul(r.v, a.v, b.v, MPFR_RNDN); return r; }
static inline MP operator/(const MP& a, const MP& b) { MP r; mpfr_div(r.v, a.v, b.v, MPFR_RNDN); return r; }

static inline double U_(double) { return 1.1102230246251565e-16; }
#if __LDBL_MANT_DIG__ >= 64
static inline double U_(long double) { return 5.421010862427522e-20; }
#else   // long double is double here (53 bits, as on arm64 macOS): its unit roundoff is that of double
static inline double U_(long double) { return 1.1102230246251565e-16; }
#endif
static inline double U_(const q&) { return 9.62964972193618e-35; }
static inline double U_(const MP&) { return 1e-300; }      // (a double cannot hold less; enough to tell a zero)

static inline double xexp(double x) { return std::exp(x); }
static inline long double xexp(long double x) { return expl(x); }
static inline q xexp(const q& x) { return expq(x); }
static inline MP xexp(const MP& x) { MP r; mpfr_exp(r.v, x.v, MPFR_RNDN); return r; }
static inline double xlog(double x) { return std::log(x); }
static inline long double xlog(long double x) { return logl(x); }
static inline q xlog(const q& x) { return logq(x); }
static inline MP xlog(const MP& x) { MP r; mpfr_log(r.v, x.v, MPFR_RNDN); return r; }
static inline void xsincos(double x, double& s, double& c) { sincos(x, &s, &c); }
static inline void xsincos(long double x, long double& s, long double& c) { sincosl(x, &s, &c); }
static inline void xsincos(const q& x, q& s, q& c) { sincosq(x, &s, &c); }
static inline void xsincos(const MP& x, MP& s, MP& c) { mpfr_sin_cos(s.v, c.v, x.v, MPFR_RNDN); }
static inline double xatan2(double y, double x) { return std::atan2(y, x); }
static inline long double xatan2(long double y, long double x) { return atan2l(y, x); }
static inline q xatan2(const q& y, const q& x) { return atan2q(y, x); }
static inline MP xatan2(const MP& y, const MP& x) { MP r; mpfr_atan2(r.v, y.v, x.v, MPFR_RNDN); return r; }
static inline double xhypot(double a, double b) { return std::hypot(a, b); }
static inline long double xhypot(long double a, long double b) { return hypotl(a, b); }
static inline q xhypot(const q& a, const q& b) { return hypotq(a, b); }
static inline MP xhypot(const MP& a, const MP& b) { MP r; mpfr_hypot(r.v, a.v, b.v, MPFR_RNDN); return r; }
static inline double xround(double x) { return std::nearbyint(x); }
static inline long double xround(long double x) { return nearbyintl(x); }
static inline q xround(const q& x) { return nearbyintq(x); }
static inline MP xround(const MP& x) { MP r; mpfr_round(r.v, x.v); return r; }
static inline double xtwopi(double) { return TWOPI_D; }
static inline long double xtwopi(long double) { return 6.283185307179586476925286766559005768L; }
static inline q xtwopi(const q&) { return 2 * PIq; }
static inline MP xtwopi(const MP&) { MP r; mpfr_const_pi(r.v, MPFR_RNDN); mpfr_mul_2si(r.v, r.v, 1, MPFR_RNDN); return r; }
static inline double xpi(double) { return PI_D; }
static inline long double xpi(long double) { return 3.141592653589793238462643383279502884L; }
static inline q xpi(const q&) { return PIq; }
static inline MP xpi(const MP&) { MP r; mpfr_const_pi(r.v, MPFR_RNDN); return r; }
static inline bool final_(double) { return false; }       // the last precision decides what the others leave open
static inline bool final_(long double) { return false; }
static inline bool final_(const q&) { return false; }
static inline bool final_(const MP&) { return true; }

template <class R> static inline double amx(const Cx<R>& z) { return std::max(std::fabs((double)z.re), std::fabs((double)z.im)); }
template <class R> static inline double asum(const Cx<R>& z) { return std::fabs((double)z.re) + std::fabs((double)z.im); }
template <class R> static inline Cx<R> cexp_(const Cx<R>& w)
{
    R m = xexp(w.re), s, c;
    xsincos(w.im, s, c);
    return Cx<R>{ m * c, m * s };
}
template <class R> static inline Cx<R> clog_(const Cx<R>& z) { return Cx<R>{ xlog(xhypot(z.re, z.im)), xatan2(z.im, z.re) }; }

// One step of the chain at one probe point: the request r (kind, relative error bound eb) for a node, the
// side entry of that node (sv = exp(a) for side 0, Log(b) for side 1, absolute error serr), gives the request
// `out` for the other child. hint: the double value of `out` on record (it fixes the multiple of 2 pi i); null:
// the principal one, and cut says that the neighbouring multiple is within the error bound.
// dreq (in: of r, out: of `out`) is what the request has below the resolution, to first order, by the rules
// of the table: the displacement of an S side (dside), and an operand below 2^-94 of the other one.
template <class R> static inline Zd zd_(const Cx<R>& z) { return Zd{ (double)z.re, (double)z.im }; }
struct Dsp { Zd z; double m; };      // a displacement to first order, and the sum of the sizes of what went into it
template <class R>
static inline int step_g(int kind, const Cx<R>& r, double eb, int side, const Cx<R>& sv, double serr, const Zd* hint,
                         Cx<R>& out, double& neb, bool& cut, const Zd& dside, Dsp& dreq)
{
    const double u = U_(R(0.0));
    cut = false;
    const bool conv = !final_(R(0.0));         // the digits follow the convention of the table (see dreq), except at PMAX bits
    if (kind == KV && side == 0) {
        Cx<R> w{ sv.re - r.re, sv.im - r.im };
        {
            double nr0 = amx(r), ns0 = amx(sv);
            if (conv && absb(nr0, ns0)) w = sv;
            else if (conv && absb(ns0, nr0)) w = Cx<R>{ R(0.0) - r.re, R(0.0) - r.im };
        }
        double d = eb * asum(r) + serr + 2 * u * (asum(sv) + asum(w));
        double wi = (double)w.im, wr = (double)w.re;
        if (wi > PI_D + 1e-9 || wi < -PI_D - 1e-9) {
            if (wi > PI_D + d || wi < -PI_D - d) return ST_DROP;    // not the imaginary part of a principal Log
        } else if (!REAL_ONLY && (wi > 3.0 || wi < -3.0)) {
            // next to the ends of (-pi, pi]: +pi is a principal value, -pi is not
            double tu = (double)(w.im - xpi(R(0.0))), tl = (double)(w.im + xpi(R(0.0)));
            if (tu > d || tl < -d) return ST_DROP;
            if (tl <= d) return final_(R(0.0)) ? ST_DROP : ST_DOUBT;   // -pi within the error bound: more digits; at PMAX presumed -pi
        }
        if (wr > 668.5 + d || wr < -668.5 - d) return ST_DROP;      // the child would be outside [1e-290, 1e290]
        if (!(d < 0.25)) return ST_DOUBT;
        out = cexp_(w);
        neb = d + 8 * u;
        Zd dw{ dside.re - dreq.z.re, dside.im - dreq.z.im };
        double nr = amx(r), ns = amx(sv), dwm = dreq.m + asumd(dside);
        if (absb(nr, ns)) { dw.re -= (double)r.re; dw.im -= (double)r.im; dwm += asum(r); }
        else if (absb(ns, nr)) { dw.re += (double)sv.re; dw.im += (double)sv.im; dwm += asum(sv); }
        dreq.z = zmul(zd_(out), dw); dreq.m = asum(out) * dwm;
        return ST_OK;
    }
    if (kind == KV) {
        out = Cx<R>{ r.re + sv.re, r.im + sv.im };
        {
            double nr0 = amx(r), ns0 = amx(sv);
            if (conv && absb(nr0, ns0)) out = sv;
            else if (conv && absb(ns0, nr0)) out = r;
        }
        double d = eb * asum(r) + serr + 2 * u * (asum(sv) + asum(out));
        double m = amx(out);
        if (REAL_ONLY && (double)out.re < -d) return ST_DROP;       // exp of a real number is positive
        if (!(d < 0.25 * m)) return ST_DOUBT;                       // cannot be told from 0 (exp is never 0)
        if (REAL_ONLY && !((double)out.re > 0)) return ST_DOUBT;
        neb = d / m;
        Zd d2{ dreq.z.re + dside.re, dreq.z.im + dside.im };
        double nr = amx(r), ns = amx(sv), dm2 = dreq.m + asumd(dside);
        if (absb(nr, ns)) { d2.re += (double)r.re; d2.im += (double)r.im; dm2 += asum(r); }
        else if (absb(ns, nr)) { d2.re += (double)sv.re; d2.im += (double)sv.im; dm2 += asum(sv); }
        dreq.z = d2; dreq.m = dm2;
        return ST_OK;
    }
    Cx<R> lw = clog_(r);
    Zd dl = zdivd(dreq.z, zd_(r));                                  // of Log(r)
    double nl = amx(lw), ns = amx(sv), dlm = amx(r) > 0 ? dreq.m / amx(r) : 0;
    if (side == 0) {
        Cx<R> w{ sv.re - lw.re, sv.im - lw.im };
        if (conv && absb(nl, ns)) w = sv;
        else if (conv && absb(ns, nl)) w = Cx<R>{ R(0.0) - lw.re, R(0.0) - lw.im };
        double d = eb + serr + 2 * u * (asum(sv) + asum(lw) + asum(w) + 2);
        double wr = (double)w.re;
        if (wr > 668.5 + d || wr < -668.5 - d) return ST_DROP;
        if (!(d < 0.25)) return ST_DOUBT;
        out = cexp_(w);
        neb = d + 8 * u;
        Zd dw{ dside.re - dl.re, dside.im - dl.im };
        double dwm = dlm + asumd(dside);
        if (absb(nl, ns)) { dw.re -= (double)lw.re; dw.im -= (double)lw.im; dwm += asum(lw); }
        else if (absb(ns, nl)) { dw.re += (double)sv.re; dw.im += (double)sv.im; dwm += asum(sv); }
        dreq.z = zmul(zd_(out), dw); dreq.m = asum(out) * dwm;
        return ST_OK;
    }
    out = Cx<R>{ lw.re + sv.re, lw.im + sv.im };
    if (conv && absb(nl, ns)) out = sv;
    else if (conv && absb(ns, nl)) out = lw;
    if (hint) {
        R k = xround((out.im - R(hint->im)) / xtwopi(R(0.0)));
        out.im = out.im - k * xtwopi(R(0.0));
    } else {
        double oi = (double)out.im;
        if (oi > PI_D) out.im = out.im - xtwopi(R(0.0));
        else if (oi <= -PI_D) out.im = out.im + xtwopi(R(0.0));
    }
    double d = eb + serr + 2 * u * (2 * asum(lw) + asum(sv) + asum(out) + 8);
    double m = amx(out);
    if (REAL_ONLY && (double)out.re < -d) return ST_DROP;
    if (!(d < 0.25 * m)) return ST_DOUBT;
    if (REAL_ONLY && !((double)out.re > 0)) return ST_DOUBT;
    neb = d / m;
    cut = !hint && std::fabs((double)out.im) > PI_D - d;
    Zd d2{ dl.re + dside.re, dl.im + dside.im };
    double dm2 = dlm + asumd(dside);
    if (absb(nl, ns)) { d2.re += (double)lw.re; d2.im += (double)lw.im; dm2 += asum(lw); }
    else if (absb(ns, nl)) { d2.re += (double)sv.re; d2.im += (double)sv.im; dm2 += asum(sv); }
    dreq.z = d2; dreq.m = dm2;
    return ST_OK;
}

struct MStep { uint8_t lev, side; uint16_t var; uint32_t idx; Zd val[NP]; };

// the side value of a step (exp of the entry for side 0, Log for side 1), a bound on its absolute error, and
// what it has below the resolution (the displacement of an S entry)
template <class R> static inline bool sideval(T3*, int lv, uint32_t idx, int p, int side, Cx<R>& out, double& serr, Zd& sdz)
{
    const Level& L = lev[lv];
    const C& z = side == 0 ? L.E[idx].z[p] : L.L[idx].z[p];
    out.re = (R)z.re; out.im = (R)(PIq * z.y);
    serr = (side == 0 ? L.wE[(size_t)idx * NP + p] : L.wL[(size_t)idx * NP + p]) * ULP;
    sdz = Zd{ 0, 0 };
    uint32_t dk = L.dx[idx];
    if (dk != NOD) {
        sdz = dZd(side == 0 ? L.dE[dk].z[p] : L.dL[dk].z[p]);
        serr += asumd(sdz);                                         // the digits of an S entry are not its value
    }
    return true;
}
template <> inline bool sideval<MP>(T3* T, int lv, uint32_t idx, int p, int side, Cx<MP>& out, double& serr, Zd& sdz)
{
    MV A, X;
    int st = T->entry(lv, idx, p, A);
    if (st == T_FAIL || st == T_ESC) return false;
    if (side == 0) { if (!T->cexp(A, X)) return false; }
    else if (!T->clog(A, X)) return false;
    mpfr_set(out.re.v, X.re, MPFR_RNDN);
    mpfr_mul(out.im.v, X.y, T->pi, MPFR_RNDN);
    serr = 0; sdz = Zd{ 0, 0 };
    return true;
}

// step t of the chain st[] applied to the request cur (all probe points); e: its relative error bound; del: what
// it has below the resolution
template <class R> static int step_all(T3* T, const MStep* st, int t, Cx<R>* cur, double& e, Dsp* del)
{
    const MStep& S = st[t];
    int kind = (t == 0) ? KV : (st[t - 1].side ? KE : KV);
    double ne = 0;
    for (int p = 0; p < NP; p++) {
        Cx<R> sv, out;
        double serr, nep = 0;
        Zd sdz;
        bool cut;
        if (!sideval<R>(T, S.lev, S.idx, p, S.side, sv, serr, sdz)) return ST_FAIL;
        int s1 = step_g<R>(kind, cur[p], e, S.side, sv, serr, &S.val[p], out, nep, cut, sdz, del[p]);
        if (s1 != ST_OK) return s1;
        cur[p] = out;
        ne = std::max(ne, nep);
    }
    e = ne;
    return ST_OK;
}

// ---------------------------------------------------------------- targets of the search

struct MForm { std::string code; double lg; bool pres; int status; };
struct MTarget {
    std::string name;
    int ti = -1;                       // index in targets[], or
    int slev = 0; uint32_t sidx = 0;   // (etalon) the table entry whose value is the target
    Zd v[NP]; Cx<q> vq[NP];
    Val t; uint8_t cls = 0;            // as (re, y) with the marks of exactness
    int found_n = 0;
    unsigned long long nforms = 0, nunres = 0;
    std::vector<MForm> forms;
};
static std::vector<MTarget> mt;

// the target at probe p as (re, y) at PMAX bits
static bool target_mv(T3& X, const MTarget& T, int p, mpfr_t re, mpfr_t y)
{
    if (T.ti >= 0) { int cls; target_at(targets[T.ti], p, re, y, cls); return true; }
    MV V;
    int st = X.entry(T.slev, T.sidx, p, V);
    if (st == T_FAIL || st == T_ESC) return false;
    mpfr_set(re, V.re, MPFR_RNDN); mpfr_set(y, V.y, MPFR_RNDN);
    return true;
}

// The request after D steps, at PMAX bits from the exact target. sig[d] names the path up to depth d; the
// requests along the last path are kept per thread, so that a neighbouring path costs a step or two.
struct MPCache { uint64_t sig[MAXD + 1]; double eb[MAXD + 1]; std::vector<Cx<MP>> v[MAXD + 1]; MPCache() { for (int d = 0; d <= MAXD; d++) sig[d] = 0; } };
static std::vector<std::unique_ptr<MPCache>> mpcache;
struct XC { bool yh, rz; long y2; };
static int chain_mp(const MTarget& T, const MStep* st, const uint64_t* sig, int D, Cx<q>* out, double& e, XC* cls = nullptr, int cmpl = 0,
                    uint32_t cmpk = 0, int cmpkind = 0, int* same = nullptr)
{
    T3& X = ctx(true);
    MP::prec() = PMAX;
    auto& cp = mpcache[omp_get_thread_num()];
    if (!cp) cp.reset(new MPCache());
    MPCache& Cc = *cp;
    int d0 = D;
    while (d0 > 0 && Cc.sig[d0] != sig[d0]) d0--;
    if (Cc.sig[d0] != sig[d0]) {
        std::vector<Cx<MP>> cur(NP);
        mpfr_t re, y;
        mpfr_inits2(PMAX, re, y, (mpfr_ptr)0);
        bool ok = true;
        for (int p = 0; p < NP && ok; p++) {
            ok = target_mv(X, T, p, re, y);
            if (ok) { mpfr_set(cur[p].re.v, re, MPFR_RNDN); mpfr_mul(cur[p].im.v, y, X.pi, MPFR_RNDN); }
        }
        mpfr_clears(re, y, (mpfr_ptr)0);
        if (!ok) return ST_FAIL;
        Cc.v[0] = cur; Cc.eb[0] = 1e-300; Cc.sig[0] = sig[0];
    }
    std::vector<Cx<MP>> cur = Cc.v[d0];
    double eb = Cc.eb[d0];
    Dsp dum[NP];
    for (int t = d0; t < D; t++) {
        for (int p = 0; p < NP; p++) dum[p] = Dsp{ Zd{ 0, 0 }, 0 };
        int s1 = step_all<MP>(&X, st, t, cur.data(), eb, dum);
        if (s1 != ST_OK) return s1;
        Cc.v[t + 1] = cur; Cc.eb[t + 1] = eb; Cc.sig[t + 1] = sig[t + 1];
    }
    if (!(eb < 1e-100)) return ST_DOUBT;
    if (cls) {
        // is the imaginary part a multiple of pi/2, is the real part 0, as far as PMAX bits can tell?
        for (int p = 0; p < NP; p++) {
            MP hp = xpi(MP(0.0)) / MP(2.0), t = cur[p].im / hp, k = xround(t), df = t - k;
            double dd = std::fabs((double)df), tt = std::fabs((double)t), kk = (double)k;
            double sz = std::fabs((double)cur[p].re) + std::fabs((double)cur[p].im);
            cls[p].yh = (dd <= 1e-250 * std::max(1.0, tt)) && std::fabs(kk) < 1e15;
            cls[p].y2 = cls[p].yh ? (long)kk : 0;
            cls[p].rz = std::fabs((double)cur[p].re) <= 1e-250 * sz;
        }
    }
    if (same) {
        // is the request (cmpkind 1: its principal Log) the value of the table entry (cmpl, cmpk), as far as PMAX bits can tell?
        *same = 1;
        for (int p = 0; p < NP && *same; p++) {
            MV W;
            int st3 = X.entry(cmpl, cmpk, p, W);
            if (st3 == T_FAIL || st3 == T_ESC) { *same = 0; break; }
            Cx<MP> c = cmpkind ? clog_(cur[p]) : cur[p];
            MP wre, wim;
            mpfr_set(wre.v, W.re, MPFR_RNDN);
            mpfr_mul(wim.v, W.y, X.pi, MPFR_RNDN);
            MP dre = c.re - wre, dim = c.im - wim;
            MP scale = xhypot(wre, wim);
            if (mpfr_zero_p(scale.v)) { *same = 0; break; }
            MP r1 = dre / scale, r2 = dim / scale;
            if (std::fabs((double)r1) > 1e-280 || std::fabs((double)r2) > 1e-280) *same = 0;
        }
    }
    for (int p = 0; p < NP; p++) { out[p].re = mpfr_get_float128(cur[p].re.v, MPFR_RNDN); out[p].im = mpfr_get_float128(cur[p].im.v, MPFR_RNDN); }
    e = 4 * U_(q(0.0));
    return ST_OK;
}

// The marks of exactness of a request, carried along the chain as in the table: yh: the imaginary part is exactly
// y2*pi/2; rz: the real part is exactly 0. They decide the ends of (-pi, pi] and the cut without rounding.
static inline XC xc_side(int lv, uint32_t idx, int p, int side)
{
    const Level& L = lev[lv];
    uint8_t f = side == 0 ? ml[lv].cE[idx] : ml[lv].cL[idx];
    const C& z = side == 0 ? L.E[idx].z[p] : L.L[idx].z[p];
    XC c;
    c.rz = (f >> (2 * p)) & 1; c.yh = (f >> (2 * p + 1)) & 1; c.y2 = 0;
    if (c.yh) { if (fabsq(z.y) < 1e15) c.y2 = (long)llrintq(2 * z.y); else c.yh = false; }
    return c;
}
static XC class_chain(const MTarget& T, const MStep* st, int upto, int p, int from = 0, const XC* cfrom = nullptr)
{
    XC c;
    if (cfrom) c = *cfrom;
    else {
        from = 0;
        c.rz = (T.cls >> (2 * p)) & 1; c.yh = (T.cls >> (2 * p + 1)) & 1; c.y2 = 0;
        if (c.yh) { if (fabsq(T.t.z[p].y) < 1e15) c.y2 = (long)llrintq(2 * T.t.z[p].y); else c.yh = false; }
    }
    for (int t = from; t < upto; t++) {
        int kind = (t == 0) ? KV : (st[t - 1].side ? KE : KV);
        XC s = xc_side(st[t].lev, st[t].idx, p, st[t].side);
        const Zd& cur = (t == 0) ? T.v[p] : st[t - 1].val[p];
        const Zd& nxt = st[t].val[p];
        XC o{ false, false, 0 };
        if (kind == KV && st[t].side == 0) {
            if (s.yh && c.yh) { if ((((s.y2 - c.y2) % 2) + 2) % 2 == 0) o.yh = true; else o.rz = true; }
        } else if (kind == KV) {
            o.yh = s.yh && c.yh; o.y2 = o.yh ? s.y2 + c.y2 : 0; o.rz = s.rz && c.rz;
        } else {
            int aw = (c.yh && c.y2 == 0) ? 1 : (c.rz ? 2 : 0);          // exp(node) on the real (1) or the imaginary (2) axis
            if (st[t].side == 0) {
                int ae = s.yh ? ((((s.y2 % 2) + 2) % 2 == 0) ? 1 : 2) : 0;
                if (ae && aw) { if (ae == aw) o.yh = true; else o.rz = true; }
            } else if (aw && s.yh) {
                long yl = (aw == 1) ? (cur.re > 0 ? 0 : 2) : (cur.im > 0 ? 1 : -1);
                o.yh = true;
                o.y2 = yl + s.y2;
                o.y2 -= 4 * std::lround(((double)o.y2 - nxt.im / (PI_D / 2)) / 4.0);    // the multiple of 2 pi i on record
            }
        }
        c = o;
    }
    return c;
}

// ---------------------------------------------------------------- statistics

struct alignas(64) MStat {
    unsigned long long req = 0, probes = 0, taghits = 0, dhits = 0, stored = 0, special = 0, dead_acc = 0;
    unsigned long long ref_ld = 0, ref_q = 0, ref_mp = 0, unres = 0, pzero = 0, cuts = 0, qrej = 0, qpass = 0;
    unsigned long long fw_pairs = 0, fw_quad = 0, fw_t3 = 0, fw_tiny = 0, fw_fail = 0, fw_hits = 0, zhits = 0;
    unsigned long long drej = 0, qdrej = 0, xdrop = 0, xkeep = 0, slp = 0, slh = 0, slreq = 0, qp3 = 0, qp4 = 0, prej = 0, ppass = 0, pcls = 0, qk_noq = 0, qk_dx = 0;
    unsigned long long fx_nodes = 0, fx_acc = 0, fx_t3 = 0, fx_dead = 0, fx_hits = 0, e2req = 0, e2hits = 0, pruned = 0, pr_t3 = 0, notnew = 0, fw_drop = 0;
    void add(const MStat& o)
    {
        req += o.req; probes += o.probes; taghits += o.taghits; dhits += o.dhits; stored += o.stored; special += o.special;
        dead_acc += o.dead_acc; ref_ld += o.ref_ld; ref_q += o.ref_q; ref_mp += o.ref_mp; unres += o.unres; pzero += o.pzero; cuts += o.cuts;
        qrej += o.qrej; qpass += o.qpass; fw_pairs += o.fw_pairs; fw_quad += o.fw_quad; fw_t3 += o.fw_t3; fw_tiny += o.fw_tiny;
        fw_fail += o.fw_fail; fw_hits += o.fw_hits; zhits += o.zhits;
        drej += o.drej; qdrej += o.qdrej; xdrop += o.xdrop; xkeep += o.xkeep; slp += o.slp; slh += o.slh; slreq += o.slreq; qp3 += o.qp3; qp4 += o.qp4; prej += o.prej; ppass += o.ppass; pcls += o.pcls; qk_noq += o.qk_noq; qk_dx += o.qk_dx;
        fx_nodes += o.fx_nodes; fx_acc += o.fx_acc; fx_t3 += o.fx_t3; fx_dead += o.fx_dead; fx_hits += o.fx_hits; e2req += o.e2req; e2hits += o.e2hits;
        pruned += o.pruned; pr_t3 += o.pr_t3; notnew += o.notnew; fw_drop += o.fw_drop;
    }
};
static std::vector<MStat> mstat;
static inline MStat& stat_me() { return mstat[omp_get_thread_num()]; }

// ---------------------------------------------------------------- the path of the search and its recomputation

struct Path {
    int len = 0;
    uint64_t sig[MAXD + 1];
    MStep st[MAXD];
    bool hasA[MAXD + 1];
    double ebA[MAXD + 1];
    Cx<q> A[MAXD + 1][NP];
    // the same in quadruple precision, for the checks of hits; kept apart so that the requests themselves (their
    // double values must come out the same in every pass) do not depend on which hits were checked
    bool hasQ[MAXD + 1];
    double ebQ[MAXD + 1];
    Cx<q> Aq[MAXD + 1][NP];
    // the marks of exactness of the requests: 1: by the rules; 2: with what PMAX bits presume
    uint8_t hasPC[MAXD + 1];
    XC pc[MAXD + 1][NP];
};

// The request after the last step of P, recomputed with more digits; its double value replaces the one in P.
static int refresh(const MTarget& T, Path& P, double& eb, MStat& S)
{
    const int D = P.len;
    auto done = [&](const Cx<q>* cur, double e) {
        for (int p = 0; p < NP; p++) {
            P.A[D][p] = cur[p];
            if (D > 0) P.st[D - 1].val[p] = Zd{ (double)cur[p].re, (double)cur[p].im };
        }
        P.ebA[D] = e; P.hasA[D] = true;
        eb = e + 2 * U53;
    };
    if (D == 0) { eb = P.ebA[0] + 2 * U53; return ST_OK; }
    int j = D - 1;
    while (j > 0 && !P.hasA[j]) j--;
    if (P.ebA[j] <= M_ACC * 0.125) {
        Cx<long double> cur[NP];
        for (int p = 0; p < NP; p++) { cur[p].re = (long double)P.A[j][p].re; cur[p].im = (long double)P.A[j][p].im; }
        double e = std::max(P.ebA[j], 4 * U_(0.0L));
        int st = ST_OK;
        Dsp dum[NP];
        for (int p = 0; p < NP; p++) dum[p] = Dsp{ Zd{ 0, 0 }, 0 };
        for (int t = j; t < D && st == ST_OK; t++) st = step_all<long double>(nullptr, P.st, t, cur, e, dum);
        if (st == ST_DROP) { S.dead_acc++; return ST_DROP; }
        if (st == ST_OK && e <= M_ACC) {
            Cx<q> c2[NP];
            for (int p = 0; p < NP; p++) { c2[p].re = cur[p].re; c2[p].im = cur[p].im; }
            done(c2, e);
            S.ref_ld++;
            return ST_OK;
        }
    }
    {
        Cx<q> cur[NP];
        for (int p = 0; p < NP; p++) cur[p] = P.A[0][p];
        double e = P.ebA[0];
        int st = ST_OK;
        Dsp dum[NP];
        for (int p = 0; p < NP; p++) dum[p] = Dsp{ Zd{ 0, 0 }, 0 };
        for (int t = 0; t < D && st == ST_OK; t++) st = step_all<q>(nullptr, P.st, t, cur, e, dum);
        if (st == ST_DROP) { S.dead_acc++; return ST_DROP; }
        if (st == ST_OK && e <= M_ACC) { done(cur, e); S.ref_q++; return ST_OK; }
    }
    {
        Cx<q> cur[NP];
        double e = 0;
        int st = chain_mp(T, P.st, P.sig, D, cur, e);
        S.ref_mp++;
        if (st == ST_DROP && getenv("MITM_DBG2")) {
            static std::atomic<int> cnt(0);
            if (cnt++ % 4001 == 0) {
#pragma omp critical(dbgq)
                {
                    printf("DBGMP %s len %d:", T.name.c_str(), D);
                    for (int d = 0; d < D; d++) {
                        std::string code;
                        rpn(P.st[d].lev, P.st[d].idx, code);
                        XC c = class_chain(T, P.st, d + 1, 0);
                        printf(" [%s s%d -> (%.15g, %.15g) yh%d y2=%ld rz%d]", code.c_str(), P.st[d].side, P.st[d].val[0].re, P.st[d].val[0].im, (int)c.yh, c.y2, (int)c.rz);
                    }
                    printf("\n");
                }
            }
        }
        if (st == ST_DROP) { S.dead_acc++; return ST_DROP; }
        if (st == ST_DOUBT) { S.pzero++; return ST_DROP; }   // exp(X) = 0 within 1e-298 at PMAX bits: presumed 0, no such X
        if (st != ST_OK) { S.unres++; return ST_DROP; }      // tier 3 failed: counted and reported
        done(cur, e);
        return ST_OK;
    }
}

// The request at the end of the path in quadruple precision, kept in P.A (false: quadruple precision cannot give it).
static const double M_QOK = 1e-29;
static bool quad_at(const MTarget& T, Path& P)
{
    const int D = P.len;
    if (P.hasQ[D]) return P.ebQ[D] <= M_QOK;
    // where quadruple precision has lost the digits (large numbers on the way): from PMAX bits
    auto from_mp = [&]() -> bool {
        Cx<q> acc[NP];
        double ea = 0;
        P.hasQ[D] = true; P.ebQ[D] = 1e300;
        if (chain_mp(T, P.st, P.sig, D, acc, ea) != ST_OK) return false;
        for (int p = 0; p < NP; p++) P.Aq[D][p] = acc[p];
        P.ebQ[D] = ea;
        return true;
    };
    int j = D - 1;
    while (j > 0 && !P.hasQ[j]) j--;
    if (j < 0) return false;
    if (P.ebQ[j] > M_QOK) return from_mp();
    Cx<q> cur[NP];
    for (int p = 0; p < NP; p++) cur[p] = P.Aq[j][p];
    double e = P.ebQ[j];
    Dsp dum[NP];
    for (int p = 0; p < NP; p++) dum[p] = Dsp{ Zd{ 0, 0 }, 0 };
    for (int t = j; t < D; t++) {
        if (step_all<q>(nullptr, P.st, t, cur, e, dum) != ST_OK) e = 1e300;      // (quadruple precision cannot give it)
        if (e > M_QOK) return from_mp();
        for (int p = 0; p < NP; p++) P.Aq[t + 1][p] = cur[p];
        P.ebQ[t + 1] = e; P.hasQ[t + 1] = true;
    }
    return true;
}
// As quad_at, but without the fall-back to PMAX bits: false where quadruple precision alone does not give the
// request. Nothing is recorded for the end of the path in that case, so a later quad_at does what it did before.
static bool quad_plain(Path& P)
{
    const int D = P.len;
    if (P.hasQ[D]) return P.ebQ[D] <= M_QOK;
    int j = D - 1;
    while (j > 0 && !P.hasQ[j]) j--;
    if (j < 0 || P.ebQ[j] > M_QOK) return false;
    Cx<q> cur[NP];
    for (int p = 0; p < NP; p++) cur[p] = P.Aq[j][p];
    double e = P.ebQ[j];
    Dsp dum[NP];
    for (int p = 0; p < NP; p++) dum[p] = Dsp{ Zd{ 0, 0 }, 0 };
    for (int t = j; t < D; t++) {
        if (step_all<q>(nullptr, P.st, t, cur, e, dum) != ST_OK) return false;
        if (e > M_QOK) return false;
        for (int p = 0; p < NP; p++) P.Aq[t + 1][p] = cur[p];
        P.ebQ[t + 1] = e; P.hasQ[t + 1] = true;
    }
    return true;
}

// ---------------------------------------------------------------- cells and the look-up table of the entries

static inline uint64_t cellhash(int e, double qr, double qi, uint64_t salt)
{
    uint64_t h = mix64((uint64_t)(int64_t)qr * 0x9E3779B97F4A7C15ULL + salt);
    return mix64(h ^ ((uint64_t)(int64_t)qi * 0xC2B2AE3D27D4EB4FULL) ^ ((uint64_t)(uint32_t)e << 40));
}
// the cell of a stored (accurate) value; false: zero or outside the range
static inline bool cell1(const Zd& z, uint64_t salt, uint64_t& h, int bits = MBC)
{
    double m = amxd(z);
    if (!(m > 1e-300 && m < 1e300)) return false;
    int e = ilogb_fast(m);
    double sc = pow2_fast(bits - e);
    h = cellhash(e, std::nearbyint(z.re * sc), std::nearbyint(z.im * sc), salt);
    return true;
}
// every cell that a value within tol (relative) of z can be in; at most 8
static inline int cells(const Zd& z, double tol, uint64_t salt, uint64_t* out, int bits = MBC)
{
    double m = amxd(z);
    if (!(m > 1e-300 && m < 1e300)) return 0;
    int e = ilogb_fast(m), es[2], ne = 1;
    es[0] = e;
    double lo = pow2_fast(e), t = tol * m;
    if (m - lo <= t) es[ne++] = e - 1;
    else if (2 * lo - m <= t) es[ne++] = e + 1;
    int n = 0;
    for (int j = 0; j < ne; j++) {
        double sc = pow2_fast(bits - es[j]);
        double xr = z.re * sc, xi = z.im * sc, band = t * sc;
        double qr = std::nearbyint(xr), qi = std::nearbyint(xi), fr = xr - qr, fi = xi - qi;
        double ar = (std::fabs(fr) > 0.5 - band) ? (fr > 0 ? 1.0 : -1.0) : 0.0;
        double ai = (std::fabs(fi) > 0.5 - band) ? (fi > 0 ? 1.0 : -1.0) : 0.0;
        out[n++] = cellhash(es[j], qr, qi, salt);
        if (ar != 0) out[n++] = cellhash(es[j], qr + ar, qi, salt);
        if (ai != 0) out[n++] = cellhash(es[j], qr, qi + ai, salt);
        if (ar != 0 && ai != 0) out[n++] = cellhash(es[j], qr + ar, qi + ai, salt);
    }
    return n;
}

// Three look-up tables of the entries, each by value, by exp(value) and by exp(value) mod 2 pi i:
//  A: by the digits: clean entries, and S entries whose displacement is not on record or is not below 2^-100 (marked U);
//  B: S entries with a displacement on record below 2^-100 of the value, by digits and displacement together;
//  C: the entries of B by their digits alone (for a request whose own displacement is not known well).
// A request without structure below the resolution can only be an entry of A; one with a known displacement
// below 2^-100 can only be an entry of B with that displacement, or a U entry of A.
static uint64_t* fpt = nullptr;
static uint64_t* fpc = nullptr;
static uint64_t* fpv = nullptr;                            // the clean values of all levels by their digits: 27 bits tag, 5 bits level, 32 bits index
static uint64_t fpmask = 0, fpcmask = 0, fpvmask = 0;
static const uint64_t SALT_ALL = 0xC1EA9B1E55ED5EEDULL;
static const int MBD = 8;                                  // bits of a cell of the displacements
static const double MTOLD = 6.103515625e-05;               // 2^-14: displacements agree to this
static const uint32_t FLAG_U = 0x80000000u;
static inline uint64_t salt_of(int level, int kind) { return 0x51ED270B0A5A5A5BULL * (uint64_t)(level * 4 + kind + 1); }
static inline uint64_t salt_zone(int s) { return 0x2545F4914F6CDD1DULL * (uint64_t)(s + 1000); }
static inline void fp_insert(uint64_t* tab, uint64_t mask, uint64_t h, uint32_t idx)
{
    uint64_t word = (((h >> 32) | 1ULL) << 32) | idx, pos = h & mask;
    for (;;) {
        uint64_t cur = __atomic_load_n(&tab[pos], __ATOMIC_RELAXED);
        if (cur == 0 && __atomic_compare_exchange_n(&tab[pos], &cur, word, false, __ATOMIC_RELAXED, __ATOMIC_RELAXED)) return;
        pos = (pos + 1) & mask;
    }
}
template <class F> static inline void fp_scan(const uint64_t* tab, uint64_t mask, uint64_t h, F&& f)
{
    uint64_t pos = h & mask;
    uint32_t tag = (uint32_t)((h >> 32) | 1ULL);
    for (;;) {
        uint64_t c = tab[pos];
        if (!c) return;
        if ((uint32_t)(c >> 32) == tag) f((uint32_t)c);
        pos = (pos + 1) & mask;
    }
}
static inline uint64_t hcomb(uint64_t hv, uint64_t hd) { return mix64(hv ^ (hd * 0x9FB21C651E98DF25ULL)); }
static inline Zd ered(const C& E)       // exp-value reduced modulo 2 pi i (exact in units of pi)
{
    q y = E.y - 2 * nearbyintq(E.y / 2);
    return Zd{ (double)E.re, (double)(PIq * y) };
}

static void mitm_arrays(int NT)
{
    double t0 = now();
    ml.assign(NT + 1, MLev());
    unsigned long long tot = 0;
    for (int s = 1; s <= NT; s++) {
        const Level& L = lev[s];
        MLev& M = ml[s];
        size_t m = L.size();
        M.n = m; M.V.resize(m * NP); M.E.resize(m * NP); M.L.resize(m * NP); M.EE.resize(m * NP); M.f.assign(m, 0);
#pragma omp parallel for schedule(static)
        for (long long k = 0; k < (long long)m; k++) {
            uint8_t f = 0;
            bool zero = true;
            for (int p = 0; p < NP; p++) {
                M.V[k * NP + p] = toZd(L.v[k].z[p]);
                if (!(((L.fl[k] >> (2 * p)) & 3) == 3 && L.v[k].z[p].y == 0)) zero = false;
            }
            if (zero) f |= MF_ZERO;
            if (L.fl[k] & F_S) f |= MF_S;
            if (L.dx[k] != NOD) f |= MF_D;
            if (L.fE[k] & F_OK) {
                f |= MF_E;
                bool eeok = true;
                for (int p = 0; p < NP; p++) {
                    M.E[k * NP + p] = toZd(L.E[k].z[p]);
                    C ee;
                    int ec;
                    if (exp2(L.E[k].z[p], (L.fE[k] >> (2 * p)) & 3, ee, ec)) {
                        Zd z = toZd(ee);
                        double mm = amxd(z);
                        if (mm > 1e-300 && mm < 1e300) M.EE[k * NP + p] = z; else eeok = false;
                    } else eeok = false;
                }
                if (eeok) f |= MF_EE;
            }
            if (L.fL[k] & F_OK) {
                f |= MF_L;
                for (int p = 0; p < NP; p++) M.L[k * NP + p] = toZd(L.L[k].z[p]);
            }
            M.f[k] = f;
        }
        tot += 3 * m;
    }
    // The marks of exactness of exp and Log of the entries. The table has them by its rules; where exp or Log of
    // an entry is on an axis or has an imaginary part k*pi/2 within quadruple precision without the mark, tier 3 is
    // asked at PMAX bits (presumed, as in the table).
    unsigned long long npres = 0, ncand = 0;
    for (int s = 1; s <= NT; s++) {
        const Level& L = lev[s];
        MLev& M = ml[s];
        const size_t m = M.n;
        M.cE.assign(m, 0); M.cL.assign(m, 0);
        std::vector<uint32_t> cand;
        for (size_t k = 0; k < m; k++) {
            M.cE[k] = L.fE[k] & CLSMASK; M.cL[k] = L.fL[k] & CLSMASK;
            bool c = false;
            for (int side = 0; side < 2; side++) {
                if (!((side == 0 ? L.fE[k] : L.fL[k]) & F_OK)) continue;
                for (int p = 0; p < NP; p++) {
                    const C& z = side == 0 ? L.E[k].z[p] : L.L[k].z[p];
                    int cls = ((side == 0 ? L.fE[k] : L.fL[k]) >> (2 * p)) & 3;
                    double wabs = (side == 0 ? L.wE[k * NP + p] : L.wL[k * NP + p]) * ULP * 64 + 1e-30 * nrm(z);
                    if (!(cls & 2) && std::fabs((double)z.y) < 1e15) { q k2 = nearbyintq(2 * z.y); if (PI_D * std::fabs((double)(z.y - k2 / 2)) <= wabs) c = true; }
                    if (!(cls & 1) && std::fabs((double)z.re) <= wabs) c = true;
                }
            }
            if (c) cand.push_back((uint32_t)k);
        }
        ncand += cand.size();
#pragma omp parallel for schedule(dynamic, 64) reduction(+ : npres)
        for (long long ci = 0; ci < (long long)cand.size(); ci++) {
            const uint32_t k = cand[ci];
            T3& X = ctx(true);
            for (int side = 0; side < 2; side++) {
                if (!((side == 0 ? L.fE[k] : L.fL[k]) & F_OK)) continue;
                for (int p = 0; p < NP; p++) {
                    MV A, R;
                    int st = X.entry(s, k, p, A);
                    if (st == T_FAIL || st == T_ESC) continue;
                    if (side == 0) { if (!X.cexp(A, R)) continue; }
                    else if (!X.clog(A, R)) continue;
                    bool rz0 = R.rz, yh0 = R.yh;
                    double labs = lsum(R.lrel, xn(R) + 1);
                    X.classify(R, labs, true);
                    uint8_t add = (uint8_t)(((R.rz && !rz0) || R.rz ? 1 : 0) | (R.yh ? 2 : 0));
                    uint8_t& tgt = side == 0 ? M.cE[k] : M.cL[k];
                    uint8_t before = (tgt >> (2 * p)) & 3;
                    (void)yh0;
                    if ((add | before) != before) { tgt |= (uint8_t)(add << (2 * p)); npres++; }
                }
            }
        }
    }
    printf("marks of exactness of exp and Log of the entries: %llu entries looked at in tier 3, %llu marks added (presumed at %ld bits)\n", ncand, npres, (long)PMAX);
    int bits = 16;
    while ((double)(1ULL << bits) * 0.5 < (double)tot) bits++;
    fpmask = (1ULL << bits) - 1;
    fpt = (uint64_t*)aligned_alloc(4096, (size_t)8 << bits);
    if (!fpt) { fprintf(stderr, "no memory for the look-up table\n"); exit(1); }
#pragma omp parallel for schedule(static)
    for (long long k = 0; k < (long long)(1ULL << bits); k++) fpt[k] = 0;
    // which entries go to B and C: count first
    unsigned long long nB = 0;
    auto classify = [&](int s, long long k, int kind, Zd& v, Zd& d) -> int {      // 0: none; 1: A; 2: A marked U; 3: B and C
        const Level& L = lev[s];
        const MLev& M = ml[s];
        bool isS = L.fl[k] & F_S;
        uint32_t dk = L.dx[k];
        d = Zd{ 0, 0 };
        if (kind == PK_V) {
            if (M.f[k] & MF_ZERO) return 0;
            v = M.V[k * NP];
            if (isS && dk != NOD) d = dZd(L.dV[dk].z[0]);
        } else {
            if (!(M.f[k] & MF_E)) return 0;
            v = (kind == PK_E) ? M.E[k * NP] : ered(L.E[k].z[0]);
            if (dk != NOD) d = dZd(L.dE[dk].z[0]);
        }
        double md = asumd(d), mv = amxd(v);
        if (md > 0 && md < MDLIM * mv && std::isfinite(md)) return 3;
        if (md > 0 || (isS && md == 0)) return 2;
        return 1;
    };
    for (int s = 1; s <= NT; s++) {
#pragma omp parallel for schedule(static) reduction(+ : nB)
        for (long long k = 0; k < (long long)ml[s].n; k++)
            for (int kind = 0; kind < 3; kind++) { Zd v, d; if (classify(s, k, kind, v, d) == 3) nB++; }
    }
    int cbits = 12;
    while ((double)(1ULL << cbits) * 0.5 < (double)nB) cbits++;
    fpcmask = (1ULL << cbits) - 1;
    fpc = (uint64_t*)aligned_alloc(4096, (size_t)8 << cbits);
    if (!fpc) { fprintf(stderr, "no memory for the look-up table\n"); exit(1); }
#pragma omp parallel for schedule(static)
    for (long long k = 0; k < (long long)(1ULL << cbits); k++) fpc[k] = 0;
    for (int s = 1; s <= NT; s++) {
#pragma omp parallel for schedule(static)
        for (long long k = 0; k < (long long)ml[s].n; k++) {
            for (int kind = 0; kind < 3; kind++) {
                Zd v, d;
                int cl = classify(s, k, kind, v, d);
                uint64_t h, hd;
                if (!cl || !cell1(v, salt_of(s, kind), h)) continue;
                if (cl == 3) {
                    if (cell1(d, 0x7777, hd, MBD)) fp_insert(fpt, fpmask, hcomb(h, hd), (uint32_t)k);
                    fp_insert(fpc, fpcmask, h, (uint32_t)k);
                } else fp_insert(fpt, fpmask, h, (uint32_t)k | (cl == 2 ? FLAG_U : 0));
            }
        }
    }
    {
        unsigned long long nc = 0;
        for (int s = 1; s <= NT; s++) nc += ml[s].n;
        int vb = 12;
        while ((double)(1ULL << vb) * 0.5 < (double)nc) vb++;
        fpvmask = (1ULL << vb) - 1;
        fpv = (uint64_t*)aligned_alloc(4096, (size_t)8 << vb);
        if (!fpv) { fprintf(stderr, "no memory for the look-up table\n"); exit(1); }
#pragma omp parallel for schedule(static)
        for (long long k = 0; k < (long long)(1ULL << vb); k++) fpv[k] = 0;
        for (int s = 1; s <= NT; s++) {
#pragma omp parallel for schedule(static)
            for (long long k = 0; k < (long long)ml[s].n; k++) {
                if (ml[s].f[k] & (MF_S | MF_ZERO)) continue;
                uint64_t h;
                if (!cell1(ml[s].V[k * NP], SALT_ALL, h)) continue;
                uint64_t word = (h & 0xFFFFFFE000000000ULL) | ((uint64_t)s << 32) | (uint64_t)k, pos = h & fpvmask;
                if (!(word >> 37)) word |= 1ULL << 37;
                for (;;) {
                    uint64_t cur = __atomic_load_n(&fpv[pos], __ATOMIC_RELAXED);
                    if (cur == 0 && __atomic_compare_exchange_n(&fpv[pos], &cur, word, false, __ATOMIC_RELAXED, __ATOMIC_RELAXED)) break;
                    pos = (pos + 1) & fpvmask;
                }
            }
        }
    }
    unsigned long long nU = 0;
    for (int s = 1; s <= NT; s++) {
#pragma omp parallel for schedule(static) reduction(+ : nU)
        for (long long k = 0; k < (long long)ml[s].n; k++)
            for (int kind = 0; kind < 3; kind++) { Zd v, d; if (classify(s, k, kind, v, d) == 2) nU++; }
    }
    printf("search arrays for the levels 1..%d: %llu entries, look-up tables 2^%d and 2^%d slots (%.1f GB; keys by displacement %llu, keys marked U %llu), %.1f s, %.1f GB in use\n", NT,
           tot / 3, bits, cbits, (double)((8ULL << bits) + (8ULL << cbits)) / 1073741824.0, nB, nU, now() - t0, mem_gb());
    fflush(stdout);
}

// ---------------------------------------------------------------- the store of the zone requests

static Zd* rst = nullptr;
static uint64_t rsmask = 0, rsslots = 0;
static const uint64_t RS_EMPTY = 0x7FF8DEADBEEF0001ULL;
static std::atomic<int> rs_overflow(0);

struct FStep { uint8_t lev, side, alt; uint32_t idx; };    // alt: (side 0) the child was on the cut within every error bound: Log taken with -pi
struct FHit { uint64_t kre, kim; uint32_t a, b; uint8_t s, i, c0, nf; FStep st[14]; };    // a term run forward (pair and steps upward) that met a stored exp-exp request
static std::vector<FHit> fhits;
static std::vector<uint32_t> fhitix;
static uint64_t fhitmask = 0;
static inline uint64_t salt_e2(int s) { return 0x3C6EF372FE94F82BULL * (uint64_t)(s + 3000); }
static std::vector<Zd> e2list[64];                         // the requests exp(exp(X)) = c by the size of X, while they are few
static bool e2big[64];
static std::vector<Zd> e2all[64];                          // all of them (for the sieve in single precision)
// The same requests in quadruple precision, by the size of X, sorted by the bits of the double key. A term run
// forward that meets a request in double precision is compared with it here at once: numbers that agree to 13
// digits and differ at 25 are common (values next to popular table values), and each would otherwise be kept,
// found again in the second run, and sent to tier 3.
struct E2Q { uint64_t kre, kim; Cx<q> k; double e; };
static std::vector<E2Q> e2q[64];
static std::atomic<unsigned long long> E2_QREJ(0), E2_QKEEP(0), E2_QNO(0);
static std::string M_SKIP;                                 // --mitmskip a,b,c: targets not searched (found at smaller sizes in an earlier run)
struct XC;
static int fquad_eval(int i, uint32_t a, int j, uint32_t b, const FStep* st, int nf, bool up, Cx<q>& v, double& ev, XC& cls);
struct Hit { uint64_t kre, kim; uint32_t a, b; uint8_t s, i, hk; };    // hk: 0 plain, 1 by displacement, 2 by digits of a displaced request
static std::vector<Hit> hits;
static std::vector<uint32_t> hitix;
static uint64_t hitmask = 0;
static inline uint64_t hitkey(int s, uint64_t kre, uint64_t kim) { return mix64(kre ^ mix64(kim + (uint64_t)s * 0x9E3779B97F4A7C15ULL)); }
static std::atomic<unsigned long long> RS_PRE(0);

// ---- the plain requests of the zone in quadruple precision, for the pairs that meet them.
// A pair of table entries that agrees with a stored request in double precision is compared with it in
// quadruple precision, which rejects all but one in ten thousand. This is done at once: the first run notes
// every plain request of the store with its value in quadruple precision (ZQ), and a pair that meets a request
// is recorded (32 bytes) only if one of the requests with that key passes the formula of rs_check, or if one of
// them is not known in quadruple precision (e < 0). The second run then finds the same candidates as it would
// from all such pairs. With --nozq every such pair is recorded and compared in the second run only: more than
// 1.4e9 records (45 GB) at 34 leaves.
struct ZQ { uint64_t kre, kim; Cx<q> k; double e; uint8_t s, kind; };
static std::vector<ZQ> zq;                          // sorted by (s, kre, kim)
static std::vector<uint32_t> zqix;                  // index + 1 of the first entry of a key
static uint64_t zqmask = 0;
static std::vector<std::vector<ZQ>> zq_tl;          // what the threads note during the first run
static bool M_ZQ = true;                            // --nozq: the comparison in the second run only
static int M_PARTS = 0;                             // --parts k: at least k parts of the store (the fast mode)
static std::atomic<unsigned long long> ZQ_REJ(0), ZQ_KEEP(0), ZQ_NOQ(0);
static void zq_build()
{
    size_t tot = 0;
    for (auto& v : zq_tl) tot += v.size();
    std::vector<ZQ>().swap(zq);
    std::vector<uint32_t>().swap(zqix);
    zqmask = 0;
    if (!tot) return;
    zq.reserve(tot);
    for (auto& v : zq_tl) { zq.insert(zq.end(), v.begin(), v.end()); std::vector<ZQ>().swap(v); }
    std::sort(zq.begin(), zq.end(), [](const ZQ& x, const ZQ& y) {
        if (x.s != y.s) return x.s < y.s;
        if (x.kre != y.kre) return x.kre < y.kre;
        if (x.kim != y.kim) return x.kim < y.kim;
        if (x.kind != y.kind) return x.kind < y.kind;
        if (x.e != y.e) return x.e < y.e;
        if (x.k.re != y.k.re) return x.k.re < y.k.re;
        return x.k.im < y.k.im;
    });
    zq.erase(std::unique(zq.begin(), zq.end(), [](const ZQ& x, const ZQ& y) {
        return x.s == y.s && x.kre == y.kre && x.kim == y.kim && x.kind == y.kind && x.e == y.e && x.k.re == y.k.re && x.k.im == y.k.im; }), zq.end());
    if (zq.size() >= 0xFFFFFFF0ULL) { fprintf(stderr, "too many requests for the table in quadruple precision (use --parts)\n"); exit(1); }
    int hb = 4;
    while ((1ULL << hb) < 2 * zq.size() + 16) hb++;
    zqmask = (1ULL << hb) - 1;
    zqix.assign(1ULL << hb, 0);
    for (size_t k = 0; k < zq.size(); k++) {
        if (k && zq[k].s == zq[k - 1].s && zq[k].kre == zq[k - 1].kre && zq[k].kim == zq[k - 1].kim) continue;
        uint64_t pos = hitkey(zq[k].s, zq[k].kre, zq[k].kim) & zqmask;
        while (zqix[pos]) pos = (pos + 1) & zqmask;
        zqix[pos] = (uint32_t)(k + 1);
    }
}
// the pair exp(a) - Log(b) (E, L from the table, wsum: their error bounds in units of 2^-113) against the plain
// requests of size s with the key (kre, kim): false if quadruple precision says that it is none of them
static inline bool zq_keep(int s, uint64_t kre, uint64_t kim, const C& E, const C& L, double wsum)
{
    if (zq.empty()) return true;
    uint64_t pos = hitkey(s, kre, kim) & zqmask;
    for (;;) {
        uint32_t ix = zqix[pos];
        if (!ix) return true;                                   // (no note of this key: as before)
        const ZQ& z0 = zq[ix - 1];
        if (z0.s == s && z0.kre == kre && z0.kim == kim) {
            const q vr = E.re - L.re, vi = PIq * (E.y - L.y);
            const double av = std::fabs((double)E.re) + std::fabs((double)L.re) + PI_D * (std::fabs((double)E.y) + std::fabs((double)L.y));
            for (size_t k = ix - 1; k < zq.size() && zq[k].s == s && zq[k].kre == kre && zq[k].kim == kim; k++) {
                const ZQ& z = zq[k];
                if (z.e < 0) return true;
                q dr = vr - z.k.re, di = vi - z.k.im;
                if (z.kind == KE) di -= 2 * PIq * nearbyintq(di / (2 * PIq));
                double tol = wsum * ULP + (z.kind == KV ? z.e * asum(z.k) : z.e + 8 * U_(q(0.0)) * (1 + asum(z.k)));
                tol = 4 * tol + 16 * U_(q(0.0)) * (asum(z.k) + av) + 1e-33;
                if (!(std::fabs((double)dr) > tol || std::fabs((double)di) > tol)) return true;
            }
            return false;
        }
        pos = (pos + 1) & zqmask;
    }
}

// ---------------------------------------------------------------- candidates

struct MCand {
    int tgt;
    uint8_t nf = 0, e2 = 0;        // nf: steps st[len..len+nf-1] of a part run forward (from the node of the request down to the pair)
    uint8_t n, len, bot, pk, blev, blev2;
    uint32_t bidx, bidx2;
    MStep st[MAXD];
};
static std::vector<std::vector<MCand>> cand_tl;

// A hit of the sieve against quadruple precision. 0: not the same digits; 2: the same digits, but what the two
// have below the resolution differs (an S entry is not its clean twin, nor another twin); 1: the same, or
// quadruple precision cannot tell.
static bool SQ_DEBUG = false;
static thread_local int SQ_RK = 0, SQ_EK = 0, SQ_WHY = 0;     // (statistics) kind of the request, of the entry, reason
static std::atomic<unsigned long long> SQ_HIST[2][4][4][8];
static int stage_q(const MTarget& T, const MCand& c)
{
    Cx<q> cur[NP];
    Dsp del[NP];
    for (int p = 0; p < NP; p++) { cur[p] = T.vq[p]; del[p] = Dsp{ Zd{ 0, 0 }, 0 }; }
    double e = 4 * U_(q(0.0));
    bool havedel = true;
    int qst = ST_OK;
    for (int t = 0; t < c.len && qst == ST_OK; t++) qst = step_all<q>(nullptr, c.st, t, cur, e, del);
    if (qst == ST_DROP) return 0;
    if (qst != ST_OK || e > 1e-31) {
        // quadruple precision has lost the digits of the request (large numbers on the way): PMAX bits
        uint64_t sig[MAXD + 1];
        sig[0] = mix64(0x1234567ULL + (uint64_t)c.tgt * 0x9E3779B97F4A7C15ULL + (uint64_t)c.n);
        for (int t = 0; t < c.len; t++)
            sig[t + 1] = mix64(sig[t] ^ (((uint64_t)c.st[t].lev << 56) | ((uint64_t)c.st[t].side << 55) | ((uint64_t)c.st[t].var << 36) | (uint64_t)c.st[t].idx));
        Cx<q> acc[NP];
        double ea = 0;
        int mst = chain_mp(T, c.st, sig, c.len, acc, ea);
        if (mst == ST_DROP || mst == ST_DOUBT) return 0;       // no such request (at PMAX: outside the strip, or exp = 0 presumed)
        if (mst != ST_OK) return 3;
        if (qst != ST_OK) havedel = false;
        for (int p = 0; p < NP; p++) {
            cur[p] = acc[p];
            if (havedel) { cur[p].re -= (q)del[p].z.re; cur[p].im -= (q)del[p].z.im; }     // the digits by the convention of the table
        }
        e = ea;
    }
    int kind = c.len ? (c.st[c.len - 1].side ? KE : KV) : KV;
    bool drej = false;
    if (c.e2) {
        // exp(X) for the term X run forward against the request, modulo 2 pi i
        FStep fs[16];
        for (int t = 0; t < c.nf; t++) { fs[t].lev = c.st[c.len + t].lev; fs[t].side = c.st[c.len + t].side; fs[t].idx = c.st[c.len + t].idx; fs[t].alt = (uint8_t)c.st[c.len + t].var; }
        Cx<q> v;
        double ev = 0;
        XC cls;
        int fr = fquad_eval(c.blev, c.bidx, c.blev2, c.bidx2, fs, c.nf, false, v, ev, cls);
        if (fr == 2) return 0;
        if (fr != 0 || (double)v.re > 700.0 || !(ev * asum(v) < 1e-6)) return 3;
        Cx<q> ex = cexp_(v);
        q dq = ex.re - cur[0].re, di = ex.im - cur[0].im;
        di -= 2 * PIq * nearbyintq(di / (2 * PIq));
        double tol = 4 * (e * asum(cur[0]) + (ev * asum(v) + 8 * U_(q(0.0))) * asum(ex)) + 1e-28 * (asum(ex) + asum(cur[0]));
        if (SQ_DEBUG) printf("        SQE2: v=(%.17g, %.17g) ev %.2e  exp(v)=(%.17g, %.17g)  request=(%.17g, %.17g) e %.2e  diff (%.2e, %.2e) tol %.2e  marks yh%d rz%d\n", (double)v.re, (double)v.im, ev,
                             (double)ex.re, (double)ex.im, (double)cur[0].re, (double)cur[0].im, e, (double)dq, (double)di, tol, (int)cls.yh, (int)cls.rz);
        return (std::fabs((double)dq) > tol || std::fabs((double)di) > tol) ? 0 : 1;
    }
    for (int p = 0; p < NP; p++) {
        Cx<q> b, r = cur[p];
        double be, re_abs = e * asum(r);     // absolute error bounds of the bottom value and of the request
        double uslack = 0;                   // what an S entry without a displacement on record can have below its digits
        bool mod = false, known = true;
        Zd de{ 0, 0 }, dr = del[p].z;
        double drm = del[p].m;
        if (c.bot == BOT_PAIR) {
            const Level& A = lev[c.blev];
            const Level& B = lev[c.blev2];
            const C& E = A.E[c.bidx].z[p];
            const C& L = B.L[c.bidx2].z[p];
            b.re = E.re - L.re; b.im = PIq * (E.y - L.y);
            be = B.wL[(size_t)c.bidx2 * NP + p] * ULP + 4 * U_(q(0.0)) * asum(b);
            if (A.dx[c.bidx] != NOD) {
                Zd d = dZd(A.dE[A.dx[c.bidx]].z[p]);
                de.re += d.re; de.im += d.im;
                const bool aS = A.fl[c.bidx] & F_S;
                const double na = nrm(A.v[c.bidx].z[p]);
                if (na > 0 && na < std::ldexp(1.0, -FS)) {                       // a tiny value: exp = 1 + d + d*d/2 with d the value
                    be += asumd(d) * asumd(d) + 4 * U_(q(0.0));
                    if (aS) uslack += 2 * MAB * asumd(d);
                } else {
                    be += A.wE[(size_t)c.bidx * NP + p] * ULP;
                    if (aS && asumd(d) == 0) { known = false; uslack += 2 * MAB * asumd(toZd(E)) * asumd(toZd(A.v[c.bidx].z[p])); }
                }
            } else be += A.wE[(size_t)c.bidx * NP + p] * ULP;
            if (B.dx[c.bidx2] != NOD) {
                Zd d = dZd(B.dL[B.dx[c.bidx2]].z[p]);
                de.re -= d.re; de.im -= d.im;
                if ((B.fl[c.bidx2] & F_S) && asumd(d) == 0) { known = false; uslack += 2 * MAB; }
            }
            if (kind == KE) { dr = zdivd(dr, zd_(cur[p])); drm /= amx(cur[p]); r = clog_(cur[p]); re_abs = e + 8 * U_(q(0.0)) * (1 + asum(r)); mod = true; }
        } else {
            const Level& H = lev[c.blev];
            uint32_t dk = H.dx[c.bidx];
            bool isS = H.fl[c.bidx] & F_S;
            if (c.pk == PK_V) {
                const C& z = H.v[c.bidx].z[p];
                b.re = z.re; b.im = PIq * z.y;
                be = ((double)H.rV[(size_t)c.bidx * NP + p] + 2) * ULP * asum(b);
                if (isS) { if (dk != NOD) de = dZd(H.dV[dk].z[p]); if (asumd(de) == 0) { known = false; uslack += 2 * MAB * asum(b); } }
            } else {
                const C& z = H.E[c.bidx].z[p];
                b.re = z.re; b.im = PIq * z.y;
                be = 2 * U_(q(0.0)) * asum(b);
                if (dk != NOD) {
                    de = dZd(H.dE[dk].z[p]);
                    const double nv = nrm(H.v[c.bidx].z[p]);
                    if (nv > 0 && nv < std::ldexp(1.0, -FS)) {                   // a tiny value: exp = 1 + d + d*d/2 with d the value
                        be += asumd(de) * asumd(de);
                        if (isS) uslack += 2 * MAB * asumd(de);
                    } else {
                        be += H.wE[(size_t)c.bidx * NP + p] * ULP;
                        if (isS && asumd(de) == 0) { known = false; uslack += 2 * MAB * asum(b) * asumd(toZd(H.v[c.bidx].z[p])); }
                    }
                } else be += H.wE[(size_t)c.bidx * NP + p] * ULP;
                mod = (c.pk == PK_ER);
            }
        }
        // the two numbers are digits + displacement: the difference of the digits (quadruple) and of the displacements
        q dq = b.re - r.re, di = b.im - r.im;
        if (mod) di -= 2 * PIq * nearbyintq(di / (2 * PIq));
        double mv = std::max(asum(b), asum(r));
        double me = asumd(de), mr = asumd(dr);
        double tol = 4 * (be + re_abs) + 1e-32 * mv + uslack;
        SQ_RK = !havedel ? 3 : (drm == 0 ? 0 : (drm < MDLIM * mv ? 1 : 2));
        SQ_EK = !known ? 3 : (me == 0 ? 0 : (me < MDLIM * mv ? 1 : 2));
        if (SQ_DEBUG) printf("        SQ: kind %d known %d  |digits diff|/mv %.2e  |de|/mv %.2e |dr|/mv %.2e  be/mv %.2e re_abs/mv %.2e uslack/mv %.2e tol/mv %.2e  e %.2e\n", kind, (int)known,
                             (std::fabs((double)dq) + std::fabs((double)di)) / mv, me / mv, mr / mv, be / mv, re_abs / mv, uslack / mv, tol / mv, e);
        if (!std::isfinite(mr) || !std::isfinite(drm)) return 4;
        if (!havedel) {                                         // (nothing known of what the request has below the resolution)
            if (std::fabs((double)dq) > tol + me || std::fabs((double)di) > tol + me) return 0;
            continue;
        }
        if (std::fabs((double)dq + (de.re - dr.re)) > tol || std::fabs((double)di + (de.im - dr.im)) > tol) return 0;
        // both displacements below 2^-100 of the value: they are compared with each other, to what double precision gave them
        if (known && me < MDLIM * mv && drm < MDLIM * mv && (me > 0 || drm > 0)) {
            double told = 4e-9 + 1e-12 * (mr > 0 ? drm / mr : 1e30);
            if (told < 0.5 && std::fabs(de.re - dr.re) + std::fabs(de.im - dr.im) > told * (me + mr)) drej = true;
        }
    }
    return drej ? 2 : 1;
}

// ---- tier 3 in batches, during the second run
// A candidate that quadruple precision lets through takes 696 bytes, and tier 3 rejects nearly all of them
// (31 leaves: 9.2e6 candidates, 2 confirmed). Kept until the walk is over they would be 13 GB at 31 leaves
// and 54 GB at 32. So a thread that has M_T3BATCH of them hands the batch to tier 3 at once: as a task that
// any thread may take, or, when T3_QMAX batches are waiting already, by doing it itself. What is kept: the
// candidates that tier 3 confirms or cannot evaluate, with what it found (the same rule as in run_size,
// which does the rest of every thread after the walk and makes the report).
struct T3Out { MCand c; double lg; int st; uint8_t pres; };
static std::vector<std::vector<T3Out>> t3keep_tl;
static size_t M_T3BATCH = 2048;
static const long T3_QMAX = 256;
static std::atomic<long> T3_WAITING(0), T3_WAITMAX(0);
static std::atomic<unsigned long long> T3_NB(0), T3_NC(0), T3_NINL(0), T3_REJ(0), T3_BAD(0), T3_US(0);
static double cand_distance(const MTarget& T, const MCand& c, int& status, bool& pres);

static void tier3_batch(const MCand* c, size_t n)
{
    const double t0 = now();
    unsigned long long rej = 0, bad = 0;
    std::vector<T3Out>& keep = t3keep_tl[omp_get_thread_num()];
    for (size_t k = 0; k < n; k++) {
        int status;
        bool pres;
        const double lg = cand_distance(mt[c[k].tgt], c[k], status, pres);
        if (status == T_BAD) { bad++; continue; }
        const bool tok = (status == T_OK || status == T_DEEP);
        const bool good = tok && (lg < -1e8 || lg < -(double)PMAX * 0.30103 + 60);
        if (tok && !good) { rej++; continue; }
        keep.push_back(T3Out{ c[k], lg, status, (uint8_t)pres });
    }
    T3_REJ += rej; T3_BAD += bad; T3_NB++; T3_NC += n;
    T3_US += (unsigned long long)((now() - t0) * 1e6);
}

static inline void emit(const MTarget& T, const MCand& c, MStat& S)
{
    SQ_RK = SQ_EK = 0;
    int r = stage_q(T, c);
    SQ_HIST[c.bot][SQ_RK][SQ_EK][r & 7]++;
    if (r == 3) S.qp3++;
    if (r == 4) S.qp4++;
    if (r == 1 || r == 3 || r == 4) {
        S.qpass++;
        std::vector<MCand>& v = cand_tl[omp_get_thread_num()];
        v.push_back(c);
        if (v.size() >= M_T3BATCH) {
            if (T3_WAITING.load(std::memory_order_relaxed) >= T3_QMAX) {
                T3_NINL++;
                tier3_batch(v.data(), v.size());
                v.clear();
            } else {
                std::vector<MCand>* b = new std::vector<MCand>();
                b->swap(v);
                v.reserve(M_T3BATCH);
                const long w = ++T3_WAITING;
                long m = T3_WAITMAX.load(std::memory_order_relaxed);
                while (w > m && !T3_WAITMAX.compare_exchange_weak(m, w)) {}
#pragma omp task firstprivate(b)
                {
                    --T3_WAITING;
                    tier3_batch(b->data(), b->size());
                    delete b;
                }
            }
        }
    }
    else if (r == 2) S.qdrej++;
    else S.qrej++;
}

// the value of the candidate term at probe p, from its leaves, in the context T; status as T3::entry
static int cand_t3(T3& T, const MCand& c, int p, MV& out, bool hi, bool& pres)
{
    MV V;
    int st;
    if (c.bot == BOT_PAIR) {
        MV A, B, E, Lg;
        st = T.entry(c.blev, c.bidx, p, A);
        if (st == T_ESC || st == T_FAIL) return st;
        int st2 = T.entry(c.blev2, c.bidx2, p, B);
        if (st2 == T_ESC || st2 == T_FAIL) return st2;
        if (!T.cexp(A, E) || !T.clog(B, Lg)) return T_BAD;
        double labs;
        T.sub(E, Lg, V, labs);
        bool unc = T.classify(V, labs, hi);
        if ((unc || T.deep(V)) && !hi) return T_ESC;
        if (unc) pres = true;
        st = (st == T_DEEP || st2 == T_DEEP || T.deep(V)) ? T_DEEP : T_OK;
    } else {
        st = T.entry(c.blev, c.bidx, p, V);
        if (st == T_ESC || st == T_FAIL) return st;
    }
    for (int d = (int)c.len + (int)c.nf - 1; d >= 0; d--) {
        MV Sd, E, Lg, W;
        int s2 = T.entry(c.st[d].lev, c.st[d].idx, p, Sd);
        if (s2 == T_ESC || s2 == T_FAIL) return s2;
        if (c.st[d].side == 0) { if (!T.cexp(Sd, E) || !T.clog(V, Lg)) return T_BAD; }
        else if (!T.cexp(V, E) || !T.clog(Sd, Lg)) return T_BAD;
        double labs;
        T.sub(E, Lg, W, labs);
        bool unc = T.classify(W, labs, hi);
        if ((unc || T.deep(W)) && !hi) return T_ESC;
        if (unc) pres = true;
        if (s2 == T_DEEP || T.deep(W)) st = T_DEEP;
        if (!W.zero()) {                          // a node outside [1e-290, 1e290]: not a term of the class
            long xw = xn(W);
            if (xw < -966 || xw > 965) return T_BAD;
        }
        V.set(W, T.P);
    }
    out.set(V, T.P);
    return st;
}

// log10 of the distance of the candidate from the target at PMAX bits (-1e9: equal); status: tier-3 status
static double cand_distance(const MTarget& T, const MCand& c, int& status, bool& pres)
{
    T3& X = ctx(true);
    double worst = -1e9;
    status = T_OK; pres = false;
    mpfr_t tr, ty, d1, d2;
    mpfr_inits2(PMAX, tr, ty, d1, d2, (mpfr_ptr)0);
    for (int p = 0; p < NP; p++) {
        MV V;
        int st = cand_t3(X, c, p, V, true, pres);
        if (st == T_ESC || st == T_FAIL || st == T_BAD) { status = st; worst = 1e9; break; }
        if (st == T_DEEP) status = T_DEEP;
        if (!target_mv(X, T, p, tr, ty)) { status = T_FAIL; worst = 1e9; break; }
        mpfr_sub(d1, V.re, tr, RN); mpfr_abs(d1, d1, RN);
        mpfr_sub(d2, V.y, ty, RN); mpfr_abs(d2, d2, RN); mpfr_mul(d2, d2, X.pi, RN);
        mpfr_max(d1, d1, d2, RN);
        mpfr_abs(tr, tr, RN); mpfr_abs(ty, ty, RN); mpfr_mul(ty, ty, X.pi, RN); mpfr_max(tr, tr, ty, RN);
        if (mpfr_cmp_ui(tr, 1) > 0) mpfr_div(d1, d1, tr, RN);
        double lg = mpfr_zero_p(d1) ? -1e9 : (double)mpfr_get_exp(d1) * 0.30103;
        worst = std::max(worst, lg);
    }
    mpfr_clears(tr, ty, d1, d2, (mpfr_ptr)0);
    return worst;
}

static void cand_code(const MCand& c, int d, std::string& out)
{
    if (d == c.len + c.nf) {
        if (c.bot == BOT_PAIR) { rpn(c.blev, c.bidx, out); rpn(c.blev2, c.bidx2, out); out.push_back('E'); }
        else rpn(c.blev, c.bidx, out);
        return;
    }
    if (c.st[d].side == 0) { rpn(c.st[d].lev, c.st[d].idx, out); cand_code(c, d + 1, out); }
    else { cand_code(c, d + 1, out); rpn(c.st[d].lev, c.st[d].idx, out); }
    out.push_back('E');
}

// ---------------------------------------------------------------- one pass of the search for terms with n leaves

struct Run {
    int n = 0, mode = MODE_GEN, zlo = 1, zhi = 0, part = 0, npart = 1, fmax = 0;
    bool bprobe = true;
    double work[64];
};

static void node(const Run& R, int tgt, int s, int kind, const Zd* val, double eb, const double* disp, const Zd* del, Path& P);

// Is x (the value that a request without structure below the resolution asks of a node with s leaves; kind KE: the
// principal Log of the requested exp) the value of a clean table entry of a smaller level? Decided at PMAX bits, as
// the duplicates of the table are. Such a node is not part of a shortest term.
static bool lower_value(const MTarget& T, Path& P, int s, int kind, const Zd* x, double tol, MStat& S)
{
    uint64_t hs[8];
    const int n = cells(x[0], tol, SALT_ALL, hs);
    for (int t = 0; t < n; t++) {
        uint64_t pos = hs[t] & fpvmask, tag = hs[t] >> 37;
        if (!tag) tag = 1;
        for (;; pos = (pos + 1) & fpvmask) {
            const uint64_t c = fpv[pos];
            if (!c) break;
            if ((c >> 37) != tag) continue;
            const int l = (int)((c >> 32) & 31);
            const uint32_t k = (uint32_t)c;
            if (l >= s || l < 1 || l > M_NT || k >= ml[l].n) continue;
            bool same = true;
            for (int p = 0; p < NP && same; p++) {
                const Zd& w = ml[l].V[(size_t)k * NP + p];
                double m = amxd(x[p]) * tol, di = w.im - x[p].im;
                if (kind == KE) di -= TWOPI_D * std::nearbyint(di / TWOPI_D);
                if (!(std::fabs(w.re - x[p].re) <= m && std::fabs(di) <= m)) same = false;
            }
            if (!same) continue;
            if (quad_at(T, P)) {
                const double eq = P.ebQ[P.len];
                for (int p = 0; p < NP && same; p++) {
                    Cx<q> r = (kind == KV) ? P.Aq[P.len][p] : clog_(P.Aq[P.len][p]);
                    const C& z = lev[l].v[k].z[p];
                    q dr = z.re - r.re, di = PIq * z.y - r.im;
                    double tq = 4 * ((kind == KV ? eq * asum(r) : eq + 8 * U_(q(0.0)) * (1 + asum(r))) + ((double)lev[l].rV[(size_t)k * NP + p] + 2) * ULP * asum(r)) + 1e-31 * asum(r);
                    if (std::fabs((double)dr) > tq || std::fabs((double)di) > tq) same = false;
                }
                if (!same) continue;
            }
            Cx<q> acc[NP];
            double ea = 0;
            int eq2 = 0;
            S.pr_t3++;
            if (chain_mp(T, P.st, P.sig, P.len, acc, ea, nullptr, l, k, kind == KE ? 1 : 0, &eq2) == ST_OK && eq2) return true;
        }
    }
    return false;
}

// The key of a zone request: the value of the node modulo 2 pi i. 0: key; 1: handled by look-ups instead; 2: no such request.
static int zone_key(int tgt, Path& P, int kind, const Zd* val, double eb, Zd* key, double& kb, MStat& S)
{
    double kmin = 1e300;
    kb = 0;
    for (int p = 0; p < NP; p++) {
        double ae;
        if (kind == KV) {
            key[p] = val[p];
            if (key[p].im > PI_D || key[p].im <= -PI_D) key[p].im -= TWOPI_D * std::nearbyint(key[p].im / TWOPI_D);
            ae = eb * asumd(val[p]) + 4 * U53 * std::fabs(val[p].im);
        } else {
            key[p] = zlog(val[p]);
            ae = eb + 4 * U53 * (1 + asumd(key[p]));
        }
        double m = amxd(key[p]);
        kmin = std::min(kmin, m);
        kb = std::max(kb, m > 0 ? ae / m : 1e300);
    }
    if (kb <= M_TAU && kmin >= M_STINY) return 0;
    const int D = P.len;
    if (!P.hasA[D]) {
        double e2;
        if (refresh(mt[tgt], P, e2, S) != ST_OK) return 2;
    }
    kb = 0; kmin = 1e300;
    for (int p = 0; p < NP; p++) {
        Cx<q> k = P.A[D][p];
        double ae;
        if (kind == KV) {
            k.im -= 2 * PIq * nearbyintq(k.im / (2 * PIq));
            ae = P.ebA[D] * asum(P.A[D][p]) + 4 * U_(q(0.0)) * std::fabs((double)P.A[D][p].im);
        } else {
            k = clog_(P.A[D][p]);
            ae = P.ebA[D] + 8 * U_(q(0.0)) * (1 + asum(k));
        }
        key[p] = Zd{ (double)k.re, (double)k.im };
        double m = amxd(key[p]);
        kmin = std::min(kmin, m);
        kb = std::max(kb, m > 0 ? ae / m : 1e300);
    }
    if (kb <= M_TAU && kmin >= M_STINY) return 0;
    return 1;
}

static inline uint64_t salt_zone_u(int s) { return 0x6C8E9CF570932BD5ULL * (uint64_t)(s + 2000); }
static inline void rs_insert(const Run& R, uint64_t h, const Zd& key, MStat& S)
{
    if (R.npart > 1 && (int)((h >> 44) % (uint64_t)R.npart) != R.part) return;
    uint64_t kre = dbits(key.re), kim = dbits(key.im), pos = h & rsmask;
    for (int probe = 0;; probe++) {
        uint64_t* slot = (uint64_t*)&rst[pos];
        uint64_t cur = __atomic_load_n(slot, __ATOMIC_RELAXED);
        if (cur == RS_EMPTY && __atomic_compare_exchange_n(slot, &cur, kre, false, __ATOMIC_RELAXED, __ATOMIC_RELAXED)) {
            __atomic_store_n(slot + 1, kim, __ATOMIC_RELAXED);
            S.stored++;
            return;
        }
        if (cur == kre && __atomic_load_n(slot + 1, __ATOMIC_RELAXED) == kim) return;
        pos = (pos + 1) & rsmask;
        if (probe > 4000000) { rs_overflow = 1; return; }
    }
}

static inline void rs_check(const Run& R, int tgt, int s, const Zd& key, int rk, double kb, int kind, Path& P, MStat& S)
{
    if (hits.empty()) return;
    int qstate = 0;
    Cx<q> kq[NP];
    double eq = 0;
    uint64_t kre = dbits(key.re), kim = dbits(key.im), pos = hitkey(s, kre, kim) & hitmask;
    for (;;) {
        uint32_t ix = hitix[pos];
        if (!ix) return;
        const Hit& h0 = hits[ix - 1];
        if (h0.s == s && h0.kre == kre && h0.kim == kim) {
            for (size_t k = ix - 1; k < hits.size() && hits[k].s == s && hits[k].kre == kre && hits[k].kim == kim; k++) {
                if ((rk == 0) != (hits[k].hk == 0)) continue;       // a request without displacement: plain hits only; with: the others
                {
                    // the pair again in double, against the error bound of this request (the store used a common bound)
                    const MLev& A = ml[hits[k].i];
                    const MLev& B = ml[s - hits[k].i];
                    const Zd& Ea = A.E[(size_t)hits[k].a * NP];
                    const Zd& Lb = B.L[(size_t)hits[k].b * NP];
                    double vr = Ea.re - Lb.re, vi = Ea.im - Lb.im;
                    double di = vi - key.im;
                    di -= TWOPI_D * std::nearbyint(di / TWOPI_D);
                    double m = amxd(key), t2 = (kb + 16 * U53) * m + 3 * U53 * (asumd(Ea) + asumd(Lb));
                    if (std::fabs(vr - key.re) > t2 || std::fabs(di) > t2) { RS_PRE++; continue; }
                }
                if (rk == 0 && lev[hits[k].i].dx[hits[k].a] == NOD && lev[s - hits[k].i].dx[hits[k].b] == NOD) {
                    // the same in quadruple precision, at the cost of a subtraction
                    if (qstate == 0) {
                        qstate = quad_at(mt[tgt], P) ? 1 : 2;
                        if (qstate == 1) {
                            eq = P.ebQ[P.len];
                            for (int p = 0; p < NP; p++) kq[p] = (kind == KV) ? P.Aq[P.len][p] : clog_(P.Aq[P.len][p]);
                        }
                    }
                    if (qstate == 1) {
                        const Level& LA = lev[hits[k].i];
                        const Level& LB = lev[s - hits[k].i];
                        bool differ = false;
                        for (int p = 0; p < NP && !differ; p++) {
                            const C& E = LA.E[hits[k].a].z[p];
                            const C& L = LB.L[hits[k].b].z[p];
                            q dr = E.re - L.re - kq[p].re, di = PIq * (E.y - L.y) - kq[p].im;
                            if (kind == KE) di -= 2 * PIq * nearbyintq(di / (2 * PIq));
                            double tol = (LA.wE[(size_t)hits[k].a * NP + p] + LB.wL[(size_t)hits[k].b * NP + p]) * ULP
                                         + (kind == KV ? eq * asum(kq[p]) : eq + 8 * U_(q(0.0)) * (1 + asum(kq[p])));
                            tol = 4 * tol + 16 * U_(q(0.0)) * (asum(kq[p]) + std::fabs((double)E.re) + std::fabs((double)L.re) + PI_D * (std::fabs((double)E.y) + std::fabs((double)L.y))) + 1e-33;
                            if (std::fabs((double)dr) > tol || std::fabs((double)di) > tol) differ = true;
                        }
                        if (differ && getenv("MITM_DBG")) {
#pragma omp critical(dbgq)
                            {
                                const C& E = LA.E[hits[k].a].z[0];
                                const C& L = LB.L[hits[k].b].z[0];
                                printf("DBGQ %s s=%d kind=%d len=%d eq=%.3e kq=(%.25g, %.25g) pair=(%.25g, %.25g) key=(%.17g, %.17g)\n", mt[tgt].name.c_str(), s, kind, P.len, eq,
                                       (double)kq[0].re, (double)kq[0].im, (double)(E.re - L.re), (double)(PIq * (E.y - L.y)), key.re, key.im);
                            }
                        }
                        if (differ) { S.prej++; continue; }
                    }
                }
                MCand c;
                c.tgt = tgt; c.n = (uint8_t)R.n; c.len = (uint8_t)P.len; c.bot = BOT_PAIR; c.pk = 0;
                c.blev = hits[k].i; c.bidx = hits[k].a; c.blev2 = (uint8_t)(s - hits[k].i); c.bidx2 = hits[k].b;
                for (int d = 0; d < P.len; d++) c.st[d] = P.st[d];
                S.zhits++;
                emit(mt[tgt], c, S);
            }
            return;
        }
        pos = (pos + 1) & hitmask;
    }
}

// (second run) a request exp(exp(X)) = c that a term run forward has met: the terms
static void e2_check(const Run& R, int tgt, int s, const Zd& key, const Path& P, MStat& S)
{
    if (fhits.empty()) return;
    uint64_t kre = dbits(key.re), kim = dbits(key.im), pos = hitkey(s, kre, kim) & fhitmask;
    for (;;) {
        uint32_t ix = fhitix[pos];
        if (!ix) return;
        const FHit& h0 = fhits[ix - 1];
        if (h0.s == s && h0.kre == kre && h0.kim == kim) {
            for (size_t k = ix - 1; k < fhits.size() && fhits[k].s == s && fhits[k].kre == kre && fhits[k].kim == kim; k++) {
                const FHit& h = fhits[k];
                if (P.len + h.nf > MAXD) continue;
                MCand c;
                c.tgt = tgt; c.n = (uint8_t)R.n; c.len = (uint8_t)P.len; c.nf = h.nf; c.e2 = 1; c.bot = BOT_PAIR; c.pk = 0;
                c.blev = h.i; c.bidx = h.a; c.blev2 = (uint8_t)(h.c0 - h.i); c.bidx2 = h.b;
                for (int d = 0; d < P.len; d++) c.st[d] = P.st[d];
                for (int t = 0; t < h.nf; t++) {
                    MStep& x = c.st[P.len + t];
                    const FStep& fs = h.st[h.nf - 1 - t];
                    x.lev = fs.lev; x.side = fs.side; x.idx = fs.idx; x.var = fs.alt;
                    for (int p = 0; p < NP; p++) x.val[p] = Zd{ 0, 0 };
                }
                S.e2hits++;
                emit(mt[tgt], c, S);
            }
            return;
        }
        pos = (pos + 1) & fhitmask;
    }
}

// The entries lo..hi-1 of level u as the side (side 0: left child, 1: right child) of a node with s leaves that
// has the request (kind, val, eb). del: what the request has below the resolution, to first order; disp: the sum
// of the sizes of what went into del (del is trusted when it is not a small rest of a cancellation).
// probe: the other child is looked up in the table; else it is the next node.
static void side_loop(const Run& R, int tgt, int s, int kind, const Zd* val, double eb, const double* disp, const Zd* del, Path& P, int u, int side,
                      bool probe, size_t lo, size_t hi)
{
    const MLev& M = ml[u];
    const Level& Lu = lev[u];
    MStat& S = stat_me();
    const uint8_t need = side == 0 ? MF_E : MF_L;
    const int cs = s - u, nk = side ? KE : KV, pk = (side == 0) ? PK_V : (kind == KV ? PK_E : PK_ER);
    Zd invW[NP], logW[NP], dlW[NP];
    double mW[NP];
    if (kind == KE) for (int p = 0; p < NP; p++) { invW[p] = zinv(val[p]); logW[p] = zlog(val[p]); mW[p] = amxd(val[p]); dlW[p] = zmul(del[p], invW[p]); }
    const uint64_t salt = salt_of(cs, pk);
    const int BN = 32, BH = 20;
    struct BE { Zd v[NP]; double disp[NP]; double tol; uint32_t idx; uint16_t var; bool clean; int nh; uint64_t h[BH]; uint8_t hk[BH]; };
    BE batch[BN];
    int nb = 0;
    XC rc[NP];
    bool haveC = false;
    auto classes = [&]() {
        if (haveC) return;
        const int D = P.len;
        if (!P.hasPC[D]) {
            int j = D;
            while (j > 0 && !P.hasPC[j]) j--;
            for (int p = 0; p < NP; p++) P.pc[D][p] = class_chain(mt[tgt], P.st, D, p, j, P.hasPC[j] ? &P.pc[j][p] : nullptr);
            P.hasPC[D] = 1;
        }
        for (int p = 0; p < NP; p++) rc[p] = P.pc[D][p];
        haveC = true;
    };
    // the marks that the rules do not give: what PMAX bits presume about this request (once per node)
    auto presume = [&]() {
        const int D = P.len;
        if (P.hasPC[D] == 2) return;
        P.hasPC[D] = 2;
        Cx<q> acc[NP];
        double ea = 0;
        XC c2[NP];
        S.pcls++;
        if (chain_mp(mt[tgt], P.st, P.sig, D, acc, ea, c2) != ST_OK) return;
        for (int p = 0; p < NP; p++) {
            if (c2[p].yh && !P.pc[D][p].yh) { P.pc[D][p].yh = true; P.pc[D][p].y2 = c2[p].y2; }
            if (c2[p].rz) P.pc[D][p].rz = true;
            rc[p] = P.pc[D][p];
        }
    };
    if (P.len >= MAXD - 1) { fprintf(stderr, "path too long\n"); exit(1); }
    const MLev& H = ml[cs];
    const Level& LH = lev[cs];
    // A hit in double against quadruple precision, at the cost of a subtraction: the request of this node in
    // quadruple precision (once per call), the side and the entry from the table. The comparison is made where no
    // exp or Log is needed: Log(entry) against exp(a) - r, exp(entry) against r + Log(b), and the same for an exp request.
    int qstate = 0;                          // 0: not tried; 1: available; 2: not available
    Cx<q> rq[NP], lwq[NP];
    double eq = 0;
    auto quick = [&](uint32_t sk, uint32_t e) -> bool {      // false: surely not the same number
        if (qstate == 0) {
            qstate = quad_at(mt[tgt], P) ? 1 : 2;
            if (qstate == 1) {
                eq = P.ebQ[P.len];
                for (int p = 0; p < NP; p++) { rq[p] = P.Aq[P.len][p]; if (kind == KE) lwq[p] = clog_(rq[p]); }
            }
        }
        if (qstate != 1) { S.qk_noq++; return true; }
        if (LH.dx[e] != NOD || Lu.dx[sk] != NOD) { S.qk_dx++; return true; }           // (displacements: the full comparison)
        for (int p = 0; p < NP; p++) {
            const C& sz = side == 0 ? Lu.E[sk].z[p] : Lu.L[sk].z[p];
            Cx<q> sq{ sz.re, PIq * sz.y }, t, x;
            double tol = (side == 0 ? Lu.wE[(size_t)sk * NP + p] : Lu.wL[(size_t)sk * NP + p]) * ULP;
            bool mod = false;
            if (kind == KV) {
                if (side == 0) { t = Cx<q>{ sq.re - rq[p].re, sq.im - rq[p].im }; tol += eq * asum(rq[p]); }
                else { t = Cx<q>{ rq[p].re + sq.re, rq[p].im + sq.im }; tol += eq * asum(rq[p]); }
            } else {
                if (side == 0) t = Cx<q>{ sq.re - lwq[p].re, sq.im - lwq[p].im };
                else t = Cx<q>{ lwq[p].re + sq.re, lwq[p].im + sq.im };
                tol += eq + 8 * U_(q(0.0)) * (1 + asum(lwq[p]));
                mod = true;
            }
            if (side == 0) {
                if (!(LH.fL[e] & F_OK)) return true;
                const C& l = LH.L[e].z[p];
                x = Cx<q>{ l.re, PIq * l.y };
                tol += LH.wL[(size_t)e * NP + p] * ULP;
            } else {
                const C& ee = LH.E[e].z[p];
                x = Cx<q>{ ee.re, PIq * ee.y };
                tol += LH.wE[(size_t)e * NP + p] * ULP;
            }
            tol = 4 * tol + 16 * U_(q(0.0)) * (asum(t) + asum(x)) + 1e-33;
            q dr = x.re - t.re, di = x.im - t.im;
            if (mod) di -= 2 * PIq * nearbyintq(di / (2 * PIq));
            if (std::fabs((double)dr) > tol || std::fabs((double)di) > tol) return false;
        }
        return true;
    };
    // one cell of a table against the request b; hk: 0 table A, 1 table A (U entries only), 2 table B, 3 table C
    auto scan1 = [&](const BE& b, uint64_t h, int hk) {
        fp_scan(hk == 3 ? fpc : fpt, hk == 3 ? fpcmask : fpmask, h, [&](uint32_t w) {
            if (hk == 1 && !(w & FLAG_U)) return;
            if (M_CLEAN && (w & FLAG_U)) return;
            uint32_t e = w & ~FLAG_U;
            if (e >= H.n) return;                    // (a slot of another level with the same tag by chance)
            S.taghits++;
            bool same = true;
            for (int p = 0; p < NP && same; p++) {
                Zd x = (pk == PK_V) ? H.V[(size_t)e * NP + p] : H.E[(size_t)e * NP + p];
                double m = amxd(b.v[p]) * b.tol, di = x.im - b.v[p].im;
                if (pk == PK_ER) { di -= TWOPI_D * std::nearbyint(di / TWOPI_D); m += 4 * U53 * std::fabs(x.im); }
                if (!(std::fabs(x.re - b.v[p].re) <= m && std::fabs(di) <= m)) same = false;
            }
            if (!same) return;
            S.dhits++;
            if (b.disp[0] > 0) S.slh++;
            if (b.clean && b.var == 0 && !quick(b.idx, e)) { S.prej++; return; }
            S.ppass++;
            // an entry that is displaced from these digits by more than the request can be: not the request
            uint32_t dk = LH.dx[e];
            if (dk != NOD) {
                for (int p = 0; p < NP; p++) {
                    double m = asumd(dZd(pk == PK_V ? LH.dV[dk].z[p] : LH.dE[dk].z[p]));
                    if (m > 0 && m < MDLIM * amxd(b.v[p]) && m > 8 * b.disp[p]) { S.drej++; return; }
                }
            }
            MCand c;
            c.tgt = tgt; c.n = (uint8_t)R.n; c.len = (uint8_t)(P.len + 1); c.bot = BOT_ENTRY; c.pk = (uint8_t)pk;
            c.blev = (uint8_t)cs; c.bidx = e; c.blev2 = 0; c.bidx2 = 0;
            for (int d = 0; d < P.len; d++) c.st[d] = P.st[d];
            MStep& x = c.st[P.len];
            x.lev = (uint8_t)u; x.side = (uint8_t)side; x.idx = b.idx; x.var = b.var;
            for (int p = 0; p < NP; p++) x.val[p] = b.v[p];
            emit(mt[tgt], c, S);
        });
    };
    auto flush = [&]() {
        for (int t = 0; t < nb; t++) for (int j = 0; j < batch[t].nh; j++) scan1(batch[t], batch[t].h[j], batch[t].hk[j]);
        nb = 0;
    };
    for (size_t k = lo; k < hi; k++) {
        if (!(M.f[k] & need)) continue;
        if (M_CLEAN && (M.f[k] & (MF_S | MF_D))) continue;
        Zd ch[NP], cz[NP];
        double cd[NP];
        double neb = 0;
        bool doubt = false, dead = false;
        int cutm = 0;
        const bool hasd = M.f[k] & MF_D;
        for (int p = 0; p < NP; p++) {
            const Zd& rv = val[p];
            double nebp = 0;
            Zd sdz{ 0, 0 };
            if (hasd) sdz = dZd(side == 0 ? Lu.dE[Lu.dx[k]].z[p] : Lu.dL[Lu.dx[k]].z[p]);
            const double sd = asumd(sdz);
            cd[p] = 0; cz[p] = Zd{ 0, 0 };
            if (kind == KV) {
                if (side == 0) {
                    const Zd& Ea = M.E[k * NP + p];
                    double wr = Ea.re - rv.re, wi = Ea.im - rv.im;
                    double nr = asumd(rv), ns = asumd(Ea);
                    double d = eb * nr + 2 * U53 * (ns + std::fabs(wr) + std::fabs(wi));
                    if (wi > PI_D + d || wi < -PI_D - d || wr > 668.5 + d || wr < -668.5 - d) { dead = true; break; }
                    if (!(d < 0.25)) { doubt = true; ch[p] = Zd{ 1, 0 }; continue; }
                    bool snapim = false;
                    if ((wi > PI_D - d || wi < -PI_D + d) && !REAL_ONLY) {
                        // at an end of (-pi, pi]: the marks of exactness decide; without them more digits
                        classes();
                        XC sc = xc_side(u, (uint32_t)k, p, 0);
                        if (sc.yh && !rc[p].yh) presume();
                        if (sc.yh && rc[p].yh) {
                            long y2w = sc.y2 - rc[p].y2;
                            if (y2w <= -2 || y2w > 2) { S.xdrop++; dead = true; break; }
                            S.xkeep++;
                            wi = (double)y2w * (PI_D / 2);
                            snapim = true;
                        } else if (wi < -PI_D + d) { doubt = true; ch[p] = Zd{ 1, 0 }; continue; }
                    }
                    double mm = std::exp(wr), sn, cs2;
                    sincos(wi, &sn, &cs2);
                    ch[p] = Zd{ mm * cs2, mm * sn };
                    if (snapim) ch[p].im = 0;
                    nebp = d + 8 * U53;
                    double dwm = disp[p] + sd;
                    Zd dw{ sdz.re - del[p].re, sdz.im - del[p].im };
                    if (absb(amxd(rv), amxd(Ea))) { dwm += nr; dw.re -= rv.re; dw.im -= rv.im; }
                    else if (absb(amxd(Ea), amxd(rv))) { dwm += ns; dw.re += Ea.re; dw.im += Ea.im; }
                    if (dwm > 0) { cd[p] = asumd(ch[p]) * dwm; cz[p] = zmul(ch[p], dw); }
                } else {
                    const Zd& Lb = M.L[k * NP + p];
                    double xr = rv.re + Lb.re, xi = rv.im + Lb.im;
                    double nr = asumd(rv), ns = asumd(Lb);
                    double d = eb * nr + 2 * U53 * (ns + std::fabs(xr) + std::fabs(xi));
                    double m = std::max(std::fabs(xr), std::fabs(xi));
                    ch[p] = Zd{ xr, xi };
                    if (REAL_ONLY && xr < -d) { dead = true; break; }
                    if (!(d < 0.25 * m) || (REAL_ONLY && !(xr > 0))) doubt = true; else nebp = d / m;
                    cd[p] = disp[p] + sd;
                    cz[p] = Zd{ del[p].re + sdz.re, del[p].im + sdz.im };
                    if (absb(amxd(rv), amxd(Lb))) { cd[p] += nr; cz[p].re += rv.re; cz[p].im += rv.im; }
                    else if (absb(amxd(Lb), amxd(rv))) { cd[p] += ns; cz[p].re += Lb.re; cz[p].im += Lb.im; }
                }
            } else if (side == 0) {
                const Zd& Ea = M.E[k * NP + p];
                if (M.f[k] & MF_EE) {
                    ch[p] = zmul(M.EE[k * NP + p], invW[p]);
                    double m = amxd(ch[p]);
                    if (!(m >= RMIN * 0.25 && m <= RMAX * 4)) { dead = true; break; }
                    nebp = eb + 16 * U53;
                } else {
                    double wr = Ea.re - logW[p].re, wi = Ea.im - logW[p].im;
                    double d = eb + 2 * U53 * (asumd(Ea) + asumd(logW[p]) + std::fabs(wr) + std::fabs(wi) + 2);
                    if (wr > 668.5 + d || wr < -668.5 - d) { dead = true; break; }
                    if (!(d < 0.25)) { doubt = true; ch[p] = Zd{ 1, 0 }; continue; }
                    double mm = std::exp(wr), sn, cs2;
                    sincos(wi, &sn, &cs2);
                    ch[p] = Zd{ mm * cs2, mm * sn };
                    nebp = d + 8 * U53;
                }
                double dwm = disp[p] / mW[p] + sd, nl = asumd(logW[p]), ns = asumd(Ea);
                Zd dw{ sdz.re - dlW[p].re, sdz.im - dlW[p].im };
                if (absb(amxd(logW[p]), amxd(Ea))) { dwm += nl; dw.re -= logW[p].re; dw.im -= logW[p].im; }
                else if (absb(amxd(Ea), amxd(logW[p]))) { dwm += ns; dw.re += Ea.re; dw.im += Ea.im; }
                if (dwm > 0) { cd[p] = asumd(ch[p]) * dwm; cz[p] = zmul(ch[p], dw); }
            } else {
                const Zd& Lb = M.L[k * NP + p];
                double xr = logW[p].re + Lb.re, xi = logW[p].im + Lb.im;
                if (xi > PI_D) xi -= TWOPI_D; else if (xi <= -PI_D) xi += TWOPI_D;
                double nl = asumd(logW[p]), ns = asumd(Lb);
                double d = eb + 2 * U53 * (2 * nl + ns + std::fabs(xr) + std::fabs(xi) + 8);
                double m = std::max(std::fabs(xr), std::fabs(xi));
                if (REAL_ONLY && xr < -d) { dead = true; break; }
                if (!(d < 0.25 * m) || (REAL_ONLY && !(xr > 0))) { doubt = true; ch[p] = Zd{ xr, xi }; }
                else {
                    nebp = d / m;
                    if (!REAL_ONLY && std::fabs(xi) > PI_D - d) {
                        // on the cut: with the marks of exactness the principal value is known; without them both sides
                        classes();
                        XC sc = xc_side(u, (uint32_t)k, p, 1);
                        if (sc.yh && !((rc[p].yh && rc[p].y2 == 0) || rc[p].rz)) presume();
                        int aw = (rc[p].yh && rc[p].y2 == 0) ? 1 : (rc[p].rz ? 2 : 0);
                        if (aw && sc.yh) {
                            long y2 = ((aw == 1) ? (val[p].re > 0 ? 0 : 2) : (val[p].im > 0 ? 1 : -1)) + sc.y2;
                            while (y2 > 2) y2 -= 4;
                            while (y2 <= -2) y2 += 4;
                            xi = (double)y2 * (PI_D / 2);
                            S.xkeep++;
                        } else cutm |= 1 << p;
                    }
                    ch[p] = Zd{ xr, xi };
                }
                cd[p] = disp[p] / mW[p] + sd;
                cz[p] = Zd{ dlW[p].re + sdz.re, dlW[p].im + sdz.im };
                if (absb(amxd(logW[p]), amxd(Lb))) { cd[p] += nl; cz[p].re += logW[p].re; cz[p].im += logW[p].im; }
                else if (absb(amxd(Lb), amxd(logW[p]))) { cd[p] += ns; cz[p].re += Lb.re; cz[p].im += Lb.im; }
            }
            neb = std::max(neb, nebp);
        }
        if (dead) continue;
        if (M_CLEAN && !doubt) { bool sl = false; for (int p = 0; p < NP; p++) if (cd[p] > 0) sl = true; if (sl) continue; }
        // the branches of exp(X) = Log(r*b) + 2 pi i k that are followed: k = 0; the neighbour when on the cut; |k| <= KWIN
        int nvp[NP], nvar = 1;
        for (int p = 0; p < NP; p++) {
            nvp[p] = 1;
            if (kind == KE && side == 1 && !REAL_ONLY && !doubt) {
                if (M_KWIN > 0) nvp[p] = 2 * M_KWIN + 1;
                else if (cutm & (1 << p)) nvp[p] = 2;
            }
            nvar *= nvp[p];
        }
        if (cutm && !doubt) S.cuts++;
        for (int var = 0; var < nvar; var++) {
            MStep& x = P.st[P.len];
            x.lev = (uint8_t)u; x.side = (uint8_t)side; x.idx = (uint32_t)k; x.var = (uint16_t)var;
            int rem = var;
            double nb2 = neb;
            for (int p = 0; p < NP; p++) {
                int vp = rem % nvp[p];
                rem /= nvp[p];
                x.val[p] = ch[p];
                if (nvp[p] == 2) { if (vp) x.val[p].im += (ch[p].im > 0) ? -TWOPI_D : TWOPI_D; }      // the other side of the cut
                else if (nvp[p] > 2) {
                    x.val[p].im += TWOPI_D * (double)(vp - M_KWIN);
                    nb2 = std::max(nb2, neb * amxd(ch[p]) / amxd(x.val[p]) + 4 * U53);
                }
            }
            P.sig[P.len + 1] = mix64(P.sig[P.len] ^ (((uint64_t)u << 56) | ((uint64_t)side << 55) | ((uint64_t)var << 36) | (uint64_t)k));
            P.hasA[P.len + 1] = false;
            P.hasQ[P.len + 1] = false;
            P.hasPC[P.len + 1] = 0;
            P.len++;
            double e2 = nb2;
            bool ok = true;
            if (doubt || nb2 > M_TAU) ok = (refresh(mt[tgt], P, e2, S) == ST_OK);
            if (ok) {
                if (doubt) for (int p = 0; p < NP; p++) { cd[p] = M_CLEAN ? 0 : 1e300; cz[p] = Zd{ 0, 0 }; }     // (nothing known about the displacement)
                if (probe) {
                    BE& b = batch[nb];
                    for (int p = 0; p < NP; p++) { b.v[p] = x.val[p]; b.disp[p] = cd[p]; }
                    b.tol = e2 + 8 * U53; b.idx = (uint32_t)k; b.var = (uint16_t)var;
                    // the cells of the digits
                    uint64_t hv[16];
                    int nv = cells(b.v[0], b.tol, salt, hv);
                    if (pk == PK_ER && std::fabs(b.v[0].im) > PI_D - 2 * b.tol * amxd(b.v[0]) - 1e-12) {
                        Zd w = b.v[0];
                        w.im += (w.im > 0) ? -TWOPI_D : TWOPI_D;
                        nv += cells(w, b.tol, salt, hv + nv);
                    }
                    // the kind of the request: without structure below the resolution (0); with a displacement that is
                    // known and below 2^-100 of the value (1); anything else (2)
                    const double mv = amxd(b.v[0]), mz = asumd(cz[0]);
                    bool allclean = true, allsub = true;
                    for (int p = 0; p < NP; p++) { if (cd[p] != 0) allclean = false; if (!(cd[p] < MDLIM * amxd(b.v[p]))) allsub = false; }
                    int rk = allclean ? 0 : ((allsub && cd[0] > 0 && mz > 1e-3 * cd[0]) ? 1 : 2);
                    b.clean = allclean;
                    S.probes++;
                    if (rk) S.slp++;
                    P.len--;
                    if (rk == 1) {
                        uint64_t hd[8];
                        int nd = cells(cz[0], MTOLD, 0x7777, hd, MBD);
                        if (nv * nd + nv > BH) {                 // (many cells: looked up at once)
                            for (int i = 0; i < nv; i++) { for (int j = 0; j < nd; j++) scan1(b, hcomb(hv[i], hd[j]), 2); scan1(b, hv[i], 1); }
                            continue;
                        }
                        b.nh = 0;
                        for (int i = 0; i < nv; i++) {
                            for (int j = 0; j < nd; j++) { b.h[b.nh] = hcomb(hv[i], hd[j]); b.hk[b.nh++] = 2; }
                            b.h[b.nh] = hv[i]; b.hk[b.nh++] = 1;
                        }
                    } else {
                        if ((rk == 2 ? 2 * nv : nv) > BH) {      // (many cells: looked up at once)
                            for (int i = 0; i < nv; i++) { scan1(b, hv[i], 0); if (rk == 2) scan1(b, hv[i], 3); }
                            continue;
                        }
                        b.nh = 0;
                        for (int i = 0; i < nv; i++) { b.h[b.nh] = hv[i]; b.hk[b.nh++] = 0; }
                        if (rk == 2) for (int i = 0; i < nv; i++) { b.h[b.nh] = hv[i]; b.hk[b.nh++] = 3; }
                    }
                    for (int j = 0; j < b.nh; j++) __builtin_prefetch(b.hk[j] == 3 ? &fpc[b.h[j] & fpcmask] : &fpt[b.h[j] & fpmask]);
                    if (++nb == BN) flush();
                    continue;
                }
                if (M_EXACT && kind == KE && side == 1 && !REAL_ONLY) {
                    // exp(exp(X)) = r*b for the next node X: not followed (every branch of exp(X) is a request of its
                    // own); kept as Log(r*b) modulo 2 pi i for the terms run forward
                    if (var == 0) {
                        S.e2req++;
                        if (R.mode == MODE_GEN) {
                            uint64_t h;
                            if (cell1(x.val[0], salt_e2(cs), h)) {
                                rs_insert(R, h, x.val[0], S);
                                if (!e2big[cs]) {
#pragma omp critical(e2list)
                                    { if (e2list[cs].size() < 64) e2list[cs].push_back(x.val[0]); else e2big[cs] = true; }
                                }
                                if (M_F32) {
#pragma omp critical(e2all)
                                    e2all[cs].push_back(x.val[0]);
                                }
                                {
                                    E2Q rec;
                                    rec.kre = dbits(x.val[0].re); rec.kim = dbits(x.val[0].im); rec.e = 1e300;
                                    rec.k = Cx<q>{ 0, 0 };
                                    if (quad_at(mt[tgt], P)) { rec.k = P.Aq[P.len][0]; rec.e = P.ebQ[P.len]; }
#pragma omp critical(e2q)
                                    e2q[cs].push_back(rec);
                                }
                            }
                        } else e2_check(R, tgt, cs, x.val[0], P, S);
                    }
                    P.len--;
                    continue;
                }
                Zd c2[NP], z2[NP];
                double d2[NP];
                for (int p = 0; p < NP; p++) { c2[p] = x.val[p]; d2[p] = cd[p]; z2[p] = cz[p]; }
                node(R, tgt, cs, nk, c2, e2, d2, z2, P);
            }
            P.len--;
        }
    }
    if (probe) flush();
}

static void run_side(const Run& R, int tgt, int s, int kind, const Zd* val, double eb, const double* disp, const Zd* del, Path& P, int u, int side, bool probe)
{
    const size_t count = ml[u].n;
    const double w = probe ? 1.0 : R.work[s - u];
    if ((double)count * w < 4e5 || !omp_in_parallel()) { side_loop(R, tgt, s, kind, val, eb, disp, del, P, u, side, probe, 0, count); return; }
    size_t chunk = (size_t)std::max(1.0, std::min((double)count, 1e5 / w));
    size_t nchunks = (count + chunk - 1) / chunk;
#pragma omp taskloop grainsize(1) default(none) shared(R, P) firstprivate(tgt, s, kind, val, eb, disp, del, u, side, probe, chunk, count, nchunks)
    for (size_t c = 0; c < nchunks; c++) {
        Path Q = P;
        side_loop(R, tgt, s, kind, val, eb, disp, del, Q, u, side, probe, c * chunk, std::min(count, (c + 1) * chunk));
    }
}

// One loop over the entries of a level as sides, as a task of its own when it is large (with its own copy of the
// path, made here, before the caller goes on). The caller waits for its tasks before it returns (taskwait in node).
static inline void side_task(const Run& R, int tgt, int s, int kind, const Zd* val, double eb, const double* disp, const Zd* del, Path& P, int u, int side,
                             bool probe)
{
    const double tw = (double)ml[u].n * (probe ? 1.0 : R.work[s - u]);
    if (tw < 2e6 || !omp_in_parallel()) { run_side(R, tgt, s, kind, val, eb, disp, del, P, u, side, probe); return; }
    Path* Q = new Path(P);
#pragma omp task default(none) shared(R) firstprivate(Q, tgt, s, kind, val, eb, disp, del, u, side, probe)
    {
        run_side(R, tgt, s, kind, val, eb, disp, del, *Q, u, side, probe);
        delete Q;
    }
}

// the pairs of table entries that make a node with s leaves, by look-ups
static void bmode(const Run& R, int tgt, int s, int kind, const Zd* val, double eb, const double* disp, const Zd* del, Path& P)
{
    for (int u = std::max(1, s - M_NT); 2 * u <= s; u++) {
        if (2 * u < s) { side_task(R, tgt, s, kind, val, eb, disp, del, P, u, 0, true); side_task(R, tgt, s, kind, val, eb, disp, del, P, u, 1, true); }
        else side_task(R, tgt, s, kind, val, eb, disp, del, P, u, kind == KV ? 1 : 0, true);
    }
}

static void node(const Run& R, int tgt, int s, int kind, const Zd* val, double eb, const double* disp, const Zd* del, Path& P)
{
    MStat& S = stat_me();
    S.req++;
    if (disp[0] > 0) S.slreq++;
    bool allclean = true;
    for (int p = 0; p < NP; p++) if (disp[p] != 0) allclean = false;
    // a node whose value is a value of a smaller level of the table is not part of a shortest term
    if (M_PRUNE && kind == KV && P.len > 0 && allclean && lower_value(mt[tgt], P, s, KV, val, eb + 8 * U53, S)) { S.pruned++; return; }
    const bool zone = (s >= R.zlo && s <= R.zhi);
    bool special = false;
    if (zone) {
        Zd key[NP];
        double kb = 0;
        int zk = zone_key(tgt, P, kind, val, eb, key, kb, S);
        if (zk == 2) return;
        special = (zk == 1);
        // The store has the requests by their kind: without structure below the resolution, by the digits; with a
        // known displacement below 2^-100, by digits and displacement (and by the digits under another mark, for the
        // pairs whose displacement is not on record). A request of any other kind is handled by look-ups.
        Zd kz = (kind == KV) ? del[0] : zdivd(del[0], val[0]);
        double kd = (kind == KV) ? disp[0] : disp[0] / amxd(val[0]), mk = amxd(key[0]);
        int rk = (kd == 0) ? 0 : ((kd < MDLIM * mk && asumd(kz) > 1e-3 * kd) ? 1 : 2);
        for (int p = 1; p < NP; p++) {
            double kdp = (kind == KV) ? disp[p] : disp[p] / amxd(val[p]);
            if (rk == 0 && kdp != 0) rk = 2;
            if (rk == 1 && !(kdp < MDLIM * amxd(key[p]))) rk = 2;
        }
        if (rk == 2) special = true;
        if (M_CLEAN && rk != 0) return;
        if (!special && kind == KE && rk == 0 && M_PRUNE) {
            // exp(node) = W with the principal Log W a value of a smaller level: the node itself can still be Log W + 2 pi i k,
            // k not 0. The request is stored with a mark (the last bit of the key), and a pair with the value Log W is not a hit.
            bool nn = lower_value(mt[tgt], P, s, KE, key, kb + 8 * U53, S);
            if (nn) S.notnew++;
            uint64_t b = (dbits(key[0].im) & ~1ULL) | (nn ? 1ULL : 0ULL);
            memcpy(&key[0].im, &b, 8);
        } else if (!special) {
            uint64_t b = dbits(key[0].im) & ~1ULL;
            memcpy(&key[0].im, &b, 8);
        }
        if (special) S.special++;
        else if (R.mode == MODE_GEN) {
            uint64_t h, hd;
            if (cell1(key[0], salt_zone(s), h)) {
                if (rk == 0) {
                    rs_insert(R, h, key[0], S);
                    if (M_ZQ && !(R.npart > 1 && (int)((h >> 44) % (uint64_t)R.npart) != R.part)) {
                        ZQ z;
                        z.kre = dbits(key[0].re); z.kim = dbits(key[0].im); z.s = (uint8_t)s; z.kind = (uint8_t)kind;
                        if (quad_plain(P)) { z.e = P.ebQ[P.len]; z.k = (kind == KV) ? P.Aq[P.len][0] : clog_(P.Aq[P.len][0]); }
                        else { z.e = -1; z.k = Cx<q>{ 0, 0 }; ZQ_NOQ++; }
                        zq_tl[omp_get_thread_num()].push_back(z);
                    }
                }
                else if (cell1(kz, 0x7777, hd, MBD)) {
                    rs_insert(R, hcomb(h, hd), key[0], S);
                    if (cell1(key[0], salt_zone_u(s), h)) rs_insert(R, h, key[0], S);
                }
            }
        } else rs_check(R, tgt, s, key[0], rk, kb, kind, P, S);
    }
    if ((!zone || special) && R.bprobe) bmode(R, tgt, s, kind, val, eb, disp, del, P);
    const int umax = std::min(M_NT, s - M_NT - 1);
    for (int u = 1; u <= umax; u++)
        for (int side = 0; side < 2; side++) side_task(R, tgt, s, kind, val, eb, disp, del, P, u, side, false);
#pragma omp taskwait
}

// ---------------------------------------------------------------- terms run forward (--exact)
//
// Where the search meets exp(exp(X)) = c for a node X above the table, every branch exp(X) = Log c + 2 pi i k is a
// different request, and there is no end to them. With --exact the request is not followed: it is stored as the
// number Log c modulo 2 pi i (by the size of X), and afterwards ALL terms with up to n-2 leaves are run forward
// (pairs of table entries, then entries added on top, one node at a time) and exp of each is looked up in the store.
// This costs as much as a level of the table with n-2 leaves, and makes the search exhaustive in the branches.

static inline XC xc_d(uint8_t marks, const Zd& z)          // marks of an exp or Log of an entry, from its double value
{
    XC c;
    c.rz = marks & 1; c.yh = (marks >> 1) & 1; c.y2 = 0;
    if (c.yh) { if (std::fabs(z.im) < 1e15) c.y2 = std::lround(z.im / (PI_D / 2)); else c.yh = false; }
    return c;
}
static inline XC xc_sub(const XC& a, const XC& b) { XC c; c.yh = a.yh && b.yh; c.y2 = c.yh ? a.y2 - b.y2 : 0; c.rz = a.rz && b.rz; return c; }
static inline XC xc_exp(const XC& a) { XC c{ false, false, 0 }; if (a.yh) { if ((((a.y2 % 2) + 2) % 2) == 0) c.yh = true; else c.rz = true; } return c; }
static inline XC xc_log(const XC& a, double re, double im)
{
    XC c{ false, false, 0 };
    if (a.yh && a.y2 == 0) { c.yh = true; c.y2 = re > 0 ? 0 : 2; }
    else if (a.rz) { c.yh = true; c.y2 = im > 0 ? 1 : -1; }
    return c;
}

// One step upward in quadruple precision: the term with value v (relative error bound ev, marks cls, size m) gets the
// entry of the step on top. 0: done; 1: quadruple precision cannot give the new value (a cancellation); 2: not a
// term of the class; 3: as 1, and the new value is below the resolution (2^-94) of what it is the difference of.
static const double FQ_FWD = 1e-7;       // relative error that quadruple precision may leave on a term run forward (the hits are compared again later)
static int fquad_step(const FStep& s, Cx<q>& v, double& ev, XC& cls, double& m, double tolq = 1e-18)
{
    const double u = U_(q(0.0));
    if (!(m >= RMIN && m <= RMAX)) return 2;
    const Level& LS = lev[s.lev];
    if (cls.yh && cls.y2 == 0) v.im = 0;
    if (cls.rz) v.re = 0;
    Cx<q> p;
    double ap, big;
    XC cp;
    if (s.side == 0) {                                   // exp(w) - Log(v)
        const C& Ew = LS.E[s.idx].z[0];
        Cx<q> lg;
        XC cl = xc_log(cls, (double)v.re, (double)v.im);
        if (cls.yh && cls.y2 == 0) { lg.re = logq(fabsq(v.re)); lg.im = v.re > 0 ? (q)0 : PIq; }
        else if (cls.rz) { lg.re = logq(fabsq(v.im)); lg.im = v.im > 0 ? PIq / 2 : -PIq / 2; }
        else if ((double)v.re < 0 && std::fabs((double)v.im) <= 64 * ev * m) {
            // on the cut within the error, without a mark: the side that the run took (both are run)
            lg.re = logq(fabsq(v.re)); lg.im = s.alt ? -PIq : PIq;
        } else {
            lg = clog_(v);
            if (s.alt != ((double)lg.im < 0 ? 1 : 0) && (double)v.re < 0 && std::fabs((double)v.im) <= 1e-12 * m) return 2;      // (the other side was a guess that is not so)
        }
        p.re = Ew.re - lg.re; p.im = PIq * Ew.y - lg.im;
        ap = ev + 2 * u * (1 + asum(lg)) + LS.wE[(size_t)s.idx * NP] * ULP + 2 * u * asum(p);
        if (LS.dx[s.idx] != NOD) ap += asumd(dZd(LS.dE[LS.dx[s.idx]].z[0]));
        cp = xc_sub(xc_d(ml[s.lev].cE[s.idx] & 3, ml[s.lev].E[(size_t)s.idx * NP]), cl);
        big = std::max(amx(lg), amxd(ml[s.lev].E[(size_t)s.idx * NP]));
    } else {                                             // exp(v) - Log(w)
        const C& Lw = LS.L[s.idx].z[0];
        if ((double)v.re > 700.0) return 2;
        double de = ev * asum(v);
        if (!(de < 1e-3)) return 1;
        Cx<q> ex = cexp_(v);
        XC ce = xc_exp(cls);
        if (ce.yh) ex.im = 0;
        if (ce.rz) ex.re = 0;
        p.re = ex.re - Lw.re; p.im = ex.im - PIq * Lw.y;
        ap = (de + 8 * u) * asum(ex) + LS.wL[(size_t)s.idx * NP] * ULP + 2 * u * asum(p);
        if (LS.dx[s.idx] != NOD) ap += asumd(dZd(LS.dL[LS.dx[s.idx]].z[0]));
        cp = xc_sub(ce, xc_d(ml[s.lev].cL[s.idx] & 3, ml[s.lev].L[(size_t)s.idx * NP]));
        big = std::max(amx(ex), amxd(ml[s.lev].L[(size_t)s.idx * NP]));
    }
    v = p; cls = cp; m = amx(v);
    if (!(ap < tolq * m)) return (m <= MAB * big) ? 3 : 1;
    ev = ap / m;
    return 0;
}
// the pair (entry a of level i, entry b of level j) in quadruple precision; results as fquad_step
static int fquad_pair(int i, uint32_t a, int j, uint32_t b, Cx<q>& v, double& ev, XC& cls, double& m, double tolq = 1e-18)
{
    const Level& LA = lev[i];
    const Level& LB = lev[j];
    const double u = U_(q(0.0));
    const C& E = LA.E[a].z[0];
    const C& L = LB.L[b].z[0];
    v.re = E.re - L.re; v.im = PIq * (E.y - L.y);
    double ae = (LA.wE[(size_t)a * NP] + LB.wL[(size_t)b * NP]) * ULP + 4 * u * asum(v);
    if (LA.dx[a] != NOD) ae += asumd(dZd(LA.dE[LA.dx[a]].z[0]));
    if (LB.dx[b] != NOD) ae += asumd(dZd(LB.dL[LB.dx[b]].z[0]));
    cls = xc_sub(xc_d(ml[i].cE[a] & 3, ml[i].E[(size_t)a * NP]), xc_d(ml[j].cL[b] & 3, ml[j].L[(size_t)b * NP]));
    m = amx(v);
    if (!(ae < tolq * m)) return (m <= MAB * std::max(amxd(ml[i].E[(size_t)a * NP]), amxd(ml[j].L[(size_t)b * NP]))) ? 3 : 1;
    ev = ae / m;
    return 0;
}
// The term made of the pair (entry a of level i, entry b of level j) and nf steps upward (up: st[0] is the first step
// above the pair; else st[nf-1] is) in quadruple precision. 0: v with relative error bound ev and marks cls; 1, 3:
// quadruple precision cannot give it (a cancellation); 2: not a term of the class.
static int fquad_eval(int i, uint32_t a, int j, uint32_t b, const FStep* st, int nf, bool up, Cx<q>& v, double& ev, XC& cls)
{
    double m = 0;
    int r = fquad_pair(i, a, j, b, v, ev, cls, m);
    if (r) return r;
    for (int t = 0; t < nf; t++) {
        r = fquad_step(st[up ? t : nf - 1 - t], v, ev, cls, m);
        if (r) return r;
    }
    if (!(m >= RMIN && m <= RMAX)) return 2;
    if (cls.yh && cls.y2 == 0) v.im = 0;
    if (cls.rz) v.re = 0;
    return 0;
}

// The same term in tier 3, from the leaves (marks of exactness presumed at PMAX bits where it matters). false: not a term.
static bool ft3_eval(int i, uint32_t a, int j, uint32_t b, const FStep* st, int nf, bool up, Cx<q>& v, XC& cls, MStat& S)
{
    MCand c;
    c.tgt = 0; c.n = 0; c.len = 0; c.nf = (uint8_t)nf; c.e2 = 0; c.bot = BOT_PAIR; c.pk = 0;
    c.blev = (uint8_t)i; c.bidx = a; c.blev2 = (uint8_t)j; c.bidx2 = b;
    for (int t = 0; t < nf; t++) {
        const FStep& s = st[up ? nf - 1 - t : t];
        c.st[t].lev = s.lev; c.st[t].side = s.side; c.st[t].idx = s.idx; c.st[t].var = 0;
    }
    S.fx_t3++;
    for (int hi = 0; hi < 2; hi++) {
        T3& X = ctx(hi != 0);
        MV V;
        bool pres = false;
        int st3 = cand_t3(X, c, 0, V, hi != 0, pres);
        if (st3 == T_ESC) continue;
        if (st3 != T_OK && st3 != T_DEEP) return false;
        v.re = mpfr_get_float128(V.re, MPFR_RNDN);
        mpfr_mul(X.t1, V.y, X.pi, MPFR_RNDN);
        v.im = mpfr_get_float128(X.t1, MPFR_RNDN);
        cls.rz = V.rz; cls.yh = V.yh; cls.y2 = 0;
        if (V.yh) { double y = mpfr_get_d(V.y, MPFR_RNDN); if (std::fabs(y) < 1e15) cls.y2 = std::lround(2 * y); else cls.yh = false; }
        if (V.rz) v.re = 0;
        if (cls.yh && cls.y2 == 0) v.im = 0;
        return true;
    }
    return false;
}

static std::atomic<unsigned long long> FX_WHY[8], FX_KEY[8];
struct FCtx {
    const Run* R;
    int fmax, c0, i;
    uint32_t a, b;
    int nf;
    FStep st[15];
    std::vector<FHit>* mine;
    MStat* S;
    // the values along the current term in quadruple precision, where they were computed (depth 0: the pair)
    bool qhas[16];
    Cx<q> qv[16];
    double qe[16];
    XC qc[16];
    // (records of the sieve) the term asked for: its steps, and what is wanted of it
    const FStep* only = nullptr;
    int nonly = 0, omode = 0;
};
// the current term of F in quadruple precision, from the deepest value on record; results as fquad_eval
static int fquad_cur(FCtx& F, Cx<q>& v, double& ev, XC& cls)
{
    const int d = F.nf;
    int j = d;
    while (j >= 0 && !F.qhas[j]) j--;
    double m = 0;
    int r = 0;
    if (j < 0) {
        r = fquad_pair(F.i, F.a, F.c0 - F.i, F.b, v, ev, cls, m, FQ_FWD);
        if (r) return r;
        j = 0;
        F.qhas[0] = true; F.qv[0] = v; F.qe[0] = ev; F.qc[0] = cls;
    } else { v = F.qv[j]; ev = F.qe[j]; cls = F.qc[j]; m = amx(v); }
    for (int t = j; t < d; t++) {
        r = fquad_step(F.st[t], v, ev, cls, m, FQ_FWD);
        if (r) return r;
        F.qhas[t + 1] = true; F.qv[t + 1] = v; F.qe[t + 1] = ev; F.qc[t + 1] = cls;
    }
    if (!(m >= RMIN && m <= RMAX)) return 2;
    if (cls.yh && cls.y2 == 0) v.im = 0;
    if (cls.rz) v.re = 0;
    return 0;
}
static const double M_TOLF2 = 1.4901161193847656e-08;      // 2^-26: largest relative error of a key exp(v) of a term run forward (a number with a large
                                                           // imaginary part reduced modulo 2 pi; such keys are not real, so a wide tolerance costs nothing)

// exp of a term run forward (c leaves) against the stored requests, as numbers modulo 2 pi i
static void e2_probe(FCtx& F, int c, const Zd& Ev, double erel)
{
    Zd k = Ev;
    if (k.im > PI_D || k.im <= -PI_D) k.im -= TWOPI_D * std::nearbyint(k.im / TWOPI_D);
    const double m = amxd(k), tol = M_TAU + erel + 16 * U53;
    if (!(m > 0)) return;
    uint64_t hs[16];
    int qst = 0;                    // the term in quadruple precision: 0 not tried, 1 exq is exp of it (relative error erq), 2 not to be had
    Cx<q> exq{ 0, 0 };
    double erq = 0;
    int n = cells(k, tol, salt_e2(c), hs);
    if (std::fabs(k.im) > PI_D - 2 * tol * m - 1e-12) {
        Zd w = k;
        w.im += (w.im > 0) ? -TWOPI_D : TWOPI_D;
        n += cells(w, tol, salt_e2(c), hs + n);
    }
    for (int t = 0; t < n; t++) {
        if (F.R->npart > 1 && (int)((hs[t] >> 44) % (uint64_t)F.R->npart) != F.R->part) continue;
        uint64_t pos = hs[t] & rsmask;
        for (;;) {
            const Zd& z = rst[pos];
            if (dbits(z.re) == RS_EMPTY) break;
            double di = z.im - k.im;
            if (std::fabs(di) > 3.0) di -= std::copysign(TWOPI_D, di);
            if (std::fabs(z.re - k.re) <= tol * m && std::fabs(di) <= tol * m) {
                // the same comparison in quadruple precision, when the term and every request with this double key can be had there
                if (qst == 0) {
                    qst = 2;
                    double eq = 0;
                    XC c2;
                    Cx<q> vq;
                    if (fquad_cur(F, vq, eq, c2) == 0 && (double)vq.re <= 700.0 && eq * asum(vq) < 1e-22) {
                        exq = cexp_(vq);
                        XC ce2 = xc_exp(c2);
                        if (ce2.yh) exq.im = 0;
                        if (ce2.rz) exq.re = 0;
                        erq = eq * asum(vq) + 16 * U_(q(0.0));
                        qst = 1;
                    }
                }
                if (qst == 1) {
                    const std::vector<E2Q>& v = e2q[c];
                    const uint64_t kre = dbits(z.re), kim = dbits(z.im);
                    size_t lo = 0, hi = v.size();
                    while (lo < hi) { size_t mid = (lo + hi) / 2; if (v[mid].kre < kre || (v[mid].kre == kre && v[mid].kim < kim)) lo = mid + 1; else hi = mid; }
                    bool keep = (lo >= v.size() || v[lo].kre != kre || v[lo].kim != kim), any = false;
                    for (size_t t = lo; !keep && t < v.size() && v[t].kre == kre && v[t].kim == kim; t++) {
                        any = true;
                        if (!(v[t].e <= 1e-25)) { keep = true; break; }
                        const double sc = std::max(amx(exq), amx(v[t].k)), tq = (16 * (erq + v[t].e) + 1e-26) * sc;
                        q dq = exq.im - v[t].k.im;
                        dq -= 2 * PIq * nearbyintq(dq / (2 * PIq));
                        if (std::fabs((double)(exq.re - v[t].k.re)) <= tq && std::fabs((double)dq) <= tq) keep = true;
                    }
                    (void)any;
                    if (!keep) { E2_QREJ++; pos = (pos + 1) & rsmask; continue; }
                    E2_QKEEP++;
                } else E2_QNO++;
                FHit h;
                h.kre = dbits(z.re); h.kim = dbits(z.im); h.a = F.a; h.b = F.b;
                h.s = (uint8_t)c; h.i = (uint8_t)F.i; h.c0 = (uint8_t)F.c0; h.nf = (uint8_t)F.nf;
                for (int t2 = 0; t2 < F.nf; t2++) h.st[t2] = F.st[t2];
                F.mine->push_back(h);
                F.S->fx_hits++;
            }
            pos = (pos + 1) & rsmask;
        }
    }
}
// can exp(v) = Ev (absolute error bound ea) be one of the few requests of size c? (a quick no for most terms)
static inline bool e2_maybe(int c, const Zd& Ev, double ea)
{
    if (e2big[c]) return true;
    for (const Zd& kq : e2list[c]) if (std::fabs(Ev.re - kq.re) <= 1e-9 * (std::fabs(kq.re) + 7.0) + ea) return true;
    return false;
}

// The term of F with more digits: quadruple precision from the pair; tier 3 when that fails, or (marks) when a part
// of v is zero within the error and no mark says so. false: not a term of the class.
static bool facc(FCtx& F, bool marks, Zd& v, double& ev, XC& cls, Zd& Ev, double& eE, bool& okE)
{
    Cx<q> vq;
    double e = 0;
    XC c2;
    F.S->fx_acc++;
    int r = fquad_cur(F, vq, e, c2);
    if (r == 2 || (r == 3 && M_CLEAN)) { F.S->fx_dead++; return false; }      // (below the resolution: not without structure there)
    bool t3 = (r == 1 || r == 3);
    if (t3) FX_WHY[4 + (F.nf ? 1 : 0)]++;
    (void)marks;
    if (t3) {
        if (!ft3_eval(F.i, F.a, F.c0 - F.i, F.b, F.st, F.nf, true, vq, c2, *F.S)) { F.S->fx_dead++; return false; }
        e = 4 * U_(q(0.0));
        F.qhas[F.nf] = true; F.qv[F.nf] = vq; F.qe[F.nf] = e; F.qc[F.nf] = c2;
    }
    double m = amx(vq);
    if (!(m >= RMIN && m <= RMAX)) { F.S->fx_dead++; return false; }
    v = Zd{ (double)vq.re, (double)vq.im }; ev = e + 2 * U53; cls = c2;
    okE = (double)vq.re <= 700.0 && e * asum(vq) < 1e-3;
    if (okE) {
        Cx<q> ex = cexp_(vq);
        XC ce = xc_exp(c2);
        if (ce.yh) ex.im = 0;
        if (ce.rz) ex.re = 0;
        Ev = Zd{ (double)ex.re, (double)ex.im };
        eE = e * asum(vq) + 4 * U53;
    }
    return true;
}

// (the rule of the table for what is below the resolution, for two sizes in any order)
static inline bool absb2(double x, double y) { return x < y ? absb(x, y) : absb(y, x); }
enum { FO_FLAG = 0, FO_SUB = 1, FO_SUB0 = 2, FO_SUB1 = 3 };      // (as F32_FLAG .. F32_SUB1 of the sieve)

// A term run forward: c leaves, value v (relative error bound ev, marks cls); haveE: exp(v) = Ev is given (bound eE).
// With --clean a node is not made of two numbers of which one is below the resolution of the other (the rule of the
// table), and exp is not taken of a number below the resolution of 1.
// With F.only (a record of the sieve in single precision): only the steps of that term are taken up to it; there
// (F.omode) its own request is looked up (FLAG), or that and everything above it (SUB), or everything above it
// that has it under Log (SUB0) or under exp (SUB1).
static void fnode(FCtx& F, int c, Zd v, double ev, XC cls, bool haveE, Zd Ev, double eE)
{
    MStat& S = *F.S;
    S.fx_nodes++;
    F.qhas[F.nf] = false;
    double mv = amxd(v);
    if (!(mv >= RMIN && mv <= RMAX)) return;
    if (cls.yh && cls.y2 == 0) v.im = 0;
    if (cls.rz) v.re = 0;
    const bool pre = F.only && F.nf < F.nonly, at = F.only && F.nf == F.nonly;
    const bool chk = !pre && !(at && (F.omode == FO_SUB0 || F.omode == FO_SUB1));
    const bool expand = c < F.fmax && !(at && F.omode == FO_FLAG);
    // (the value 1e-200 with ev >= 1e8 stands for: the subtraction gave exactly 0 in double precision. That is
    // not the number 0 unless the marks say so: quadruple precision, then tier 3, say what it is.)
    const bool unknown = ev >= 1e8 && v.re == 1e-200 && v.im == 0;
    bool okE = true;
    if (!unknown && M_CLEAN && absb(mv, 1.0)) okE = false;      // exp(v) = 1 + v, v below the resolution of 1
    if (okE && !haveE) {
        if (v.re > 700.0) okE = false;
        else {
            eE = ev * asumd(v) + 4 * U53;
            if (eE < 0.5) {
                double mm = std::exp(v.re), sn, cs;
                sincos(v.im, &sn, &cs);
                Ev = Zd{ mm * cs, mm * sn };
                XC ce = xc_exp(cls);
                if (ce.yh) Ev.im = 0;
                if (ce.rz) Ev.re = 0;
            } else Ev = Zd{ 1, 0 };
        }
    }
    auto keybad = [&]() -> bool {                 // the key exp(v) mod 2 pi i is wanted and is not accurate enough
        if (!chk) return false;
        if (!okE || !(std::fabs(Ev.re) < 1500.0)) return false;
        if (!(eE < 0.5)) return true;
        Zd k = Ev;
        if (k.im > PI_D || k.im <= -PI_D) k.im -= TWOPI_D * std::nearbyint(k.im / TWOPI_D);
        return !(eE * asumd(Ev) <= M_TOLF2 * amxd(k)) && e2_maybe(c, Ev, eE * asumd(Ev));
    };
    auto axis = [&]() -> bool {                   // Log(v) is wanted and v is on the cut within the error, without a mark
        if ((cls.yh && cls.y2 == 0) || cls.rz) return false;
        return v.re < 0 && std::fabs(v.im) <= (64 * ev + 1e-13) * mv;
    };
    if (unknown || keybad() || (expand && (ev > 1e-8 || (okE && !(eE < 0.5))))) {
        if (keybad()) { FX_WHY[0]++; FX_KEY[(expand ? 0 : 4) + (haveE ? 0 : 2) + (eE < 0.5 ? 0 : 1)]++; } else if (ev > 1e-8) FX_WHY[1]++; else FX_WHY[3]++;
        if (getenv("FN_DBG")) {
            static std::atomic<int> cnt(0);
            const int kb = keybad() ? 1 : 0;
            if (cnt++ % atoi(getenv("FN_DBG")) == 0) {
#pragma omp critical(dbgq)
                {
                    printf("FNDBG %s size %d depth %d v=(%.6g, %.6g) ev=%.2e haveE %d Ev=(%.6g, %.6g) eE=%.2e |", kb ? "key" : "val", c, F.nf, v.re, v.im, ev, (int)haveE, Ev.re, Ev.im, eE);
                    if (F.nf > 0) {
                        const FStep& ls = F.st[F.nf - 1];
                        printf(" last: side %d level %d E=(%.6g, %.6g) L=(%.6g, %.6g)", ls.side, ls.lev, ml[ls.lev].E[ls.idx].re, ml[ls.lev].E[ls.idx].im, ml[ls.lev].L[ls.idx].re, ml[ls.lev].L[ls.idx].im);
                    } else printf(" pair");
                    printf("\n");
                }
            }
        }
        if (!facc(F, false, v, ev, cls, Ev, eE, okE)) return;
        mv = amxd(v);
        if (M_CLEAN && absb(mv, 1.0)) okE = false;
    }
    // Log(v) with v on the negative real axis within the error, and no mark of exactness: both sides of the cut are
    // run (the term has one of them; the other gives numbers that no term has, and a hit of theirs fails in tier 3)
    const int ncut = (expand && axis()) ? 2 : 1;
    if (ncut == 2) FX_WHY[2]++;
    if (chk && okE && std::fabs(Ev.re) < 1500.0 && e2_maybe(c, Ev, eE * asumd(Ev))) {
        Zd k = Ev;
        if (k.im > PI_D || k.im <= -PI_D) k.im -= TWOPI_D * std::nearbyint(k.im / TWOPI_D);
        double mk = amxd(k);
        if (mk > 0) e2_probe(F, c, Ev, eE * asumd(Ev) / mk);
    }
    if (!expand) return;
    if (F.nf >= 14) return;
    // Log(v), 1/v, exp(exp(v)): once for all the entries put on top
    Zd Lv;
    const XC cl = xc_log(cls, v.re, v.im), ce = xc_exp(cls);
    if (cls.yh && cls.y2 == 0) Lv = Zd{ std::log(std::fabs(v.re)), v.re > 0 ? 0.0 : PI_D };
    else if (cls.rz) Lv = Zd{ std::log(std::fabs(v.im)), v.im > 0 ? PI_D / 2 : -PI_D / 2 };
    else if (ncut == 2) Lv = Zd{ std::log(std::fabs(v.re)), PI_D };
    else Lv = zlog(v);
    const double eL = ev + 2 * U53 * (1 + asumd(Lv));
    const Zd inv = zinv(v);
    bool haveEE = false;
    Zd EEv{ 0, 0 };
    double eEE = 0;
    if (okE && std::fabs(Ev.re) < 700.0) {
        eEE = eE * asumd(Ev) + 4 * U53;
        if (eEE < M_TOLF2) {
            double mm = std::exp(Ev.re), sn, cs;
            sincos(Ev.im, &sn, &cs);
            EEv = Zd{ mm * cs, mm * sn };
            XC cee = xc_exp(ce);
            if (cee.yh) EEv.im = 0;
            if (cee.rz) EEv.re = 0;
            double me = amxd(EEv);
            haveEE = me > 1e-300 && me < 1e300;
        }
    }
    if (!haveEE && okE && std::fabs(Ev.re) < 700.0) {
        // exp(exp(v)) is needed for the terms eml(v, w) and double precision cannot give it: quadruple precision
        Cx<q> vq;
        double e = 0;
        XC c2;
        FX_WHY[7]++;
        if (fquad_cur(F, vq, e, c2) == 0 && (double)vq.re <= 700.0 && e * asum(vq) < 1e-20) {
            Cx<q> ex = cexp_(vq);
            XC ce2 = xc_exp(c2);
            if (ce2.yh) ex.im = 0;
            if (ce2.rz) ex.re = 0;
            double ee = (e * asum(vq) + 8 * U_(q(0.0))) * asum(ex);
            if ((double)ex.re < 700.0 && ee < 1e-12) {
                Cx<q> eex = cexp_(ex);
                XC cee = xc_exp(ce2);
                if (cee.yh) eex.im = 0;
                if (cee.rz) eex.re = 0;
                EEv = Zd{ (double)eex.re, (double)eex.im };
                eEE = ee + 4 * U53;
                double me = amxd(EEv);
                haveEE = me > 1e-300 && me < 1e300;
            }
        }
    }
    const double aEv = eE * asumd(Ev), sEv = amxd(Ev), sLv = amxd(Lv);
    const bool do0 = pre ? F.only[F.nf].side == 0 : !(at && F.omode == FO_SUB1);
    const bool do1 = pre ? F.only[F.nf].side == 1 : !(at && F.omode == FO_SUB0);
    int u0 = 1, u1 = std::min(M_NT, F.fmax - c);
    if (pre) { u0 = F.only[F.nf].lev; if (u0 > u1) return; u1 = u0; }
    for (int u = u0; u <= u1; u++) {
        const MLev& M = ml[u];
        const int cc = c + u;
        const bool top = (cc == F.fmax) && !e2big[cc];
        size_t k0 = 0, k1 = M.n;
        if (pre) { k0 = F.only[F.nf].idx; k1 = k0 + 1; }
        for (size_t k = k0; k < k1; k++) {
            const uint8_t f = M.f[k];
            if (M_CLEAN && (f & (MF_S | MF_D))) continue;
            FStep& st = F.st[F.nf];
            st.lev = (uint8_t)u; st.idx = (uint32_t)k; st.alt = 0;
            if (do0 && (f & MF_E) && !(M_CLEAN && absb2(amxd(M.E[k * NP]), sLv))) for (int alt = 0; alt < ncut; alt++) {     // eml(w, v) = exp(w) - Log(v);  its exp is exp(exp(w)) / v
                const Zd& Ea = M.E[k * NP];
                st.alt = (uint8_t)alt;
                if (ncut == 2) Lv.im = alt ? -PI_D : PI_D;
                if (top && (f & MF_EE)) {
                    Zd Ec = zmul(M.EE[k * NP], inv);
                    S.fx_nodes++;
                    if (e2_maybe(cc, Ec, (ev + 8 * U53) * asumd(Ec))) {
                        Zd p{ Ea.re - Lv.re, Ea.im - Lv.im };
                        if (p.re == 0 && p.im == 0) p = Zd{ 1e-200, 0 };
                        st.side = 0; F.nf++; fnode(F, cc, p, 1e9, XC{ false, false, 0 }, true, Ec, ev + 8 * U53); F.nf--;
                    }
                } else if (!(cc == F.fmax && !(f & MF_EE))) {
                    Zd p{ Ea.re - Lv.re, Ea.im - Lv.im };
                    XC cp = xc_sub(xc_d(M.cE[k] & 3, Ea), cl);
                    if (p.re == 0 && p.im == 0 && !(cp.rz && cp.yh && cp.y2 == 0)) p = Zd{ 1e-200, 0 };
                    double mp = amxd(p), ap = eL + 2 * U53 * (asumd(Ea) + asumd(p));
                    if (mp >= RMIN && mp <= RMAX) {
                        double ep = (ap < 0.25 * mp) ? ap / mp : 1e9;
                        st.side = 0; F.nf++;
                        if (f & MF_EE) fnode(F, cc, p, ep, cp, true, zmul(M.EE[k * NP], inv), ev + 8 * U53);
                        else fnode(F, cc, p, ep, cp, false, Zd{ 0, 0 }, 0);
                        F.nf--;
                    }
                }
            }
            st.alt = 0;
            if (do1 && (f & MF_L) && okE) {                  // eml(v, w) = exp(v) - Log(w);  its exp is exp(exp(v)) / w
                const Zd& Lb = M.L[k * NP];
                // (exp(v) below every number of the arithmetic is still not 0: below the resolution of Log(w) unless that is 0)
                if (M_CLEAN && (sEv > 1e-300 ? absb2(sEv, amxd(Lb)) : amxd(Lb) > 0)) continue;
                if (top && haveEE) {
                    Zd Ec = zmul(EEv, zinv(M.V[k * NP]));
                    S.fx_nodes++;
                    if (e2_maybe(cc, Ec, (eEE + 8 * U53) * asumd(Ec))) {
                        Zd p{ Ev.re - Lb.re, Ev.im - Lb.im };
                        if (p.re == 0 && p.im == 0) p = Zd{ 1e-200, 0 };
                        st.side = 1; F.nf++; fnode(F, cc, p, 1e9, XC{ false, false, 0 }, true, Ec, eEE + 8 * U53); F.nf--;
                    }
                } else if (!(cc == F.fmax && !haveEE && !(std::fabs(Ev.re) < 700.0))) {
                    Zd p{ Ev.re - Lb.re, Ev.im - Lb.im };
                    XC cp = xc_sub(ce, xc_d(M.cL[k] & 3, Lb));
                    if (p.re == 0 && p.im == 0 && !(cp.rz && cp.yh && cp.y2 == 0)) p = Zd{ 1e-200, 0 };
                    double mp = amxd(p), ap = aEv + 2 * U53 * (asumd(Lb) + asumd(p));
                    if (mp >= RMIN && mp <= RMAX) {
                        double ep = (ap < 0.25 * mp) ? ap / mp : 1e9;
                        st.side = 1; F.nf++;
                        if (haveEE) fnode(F, cc, p, ep, cp, true, zmul(EEv, zinv(M.V[k * NP])), eEE + 8 * U53);
                        else fnode(F, cc, p, ep, cp, false, Zd{ 0, 0 }, 0);
                        F.nf--;
                    }
                }
            }
        }
    }
}

// ---------------------------------------------------------------- the pairs of the zone against the store

static std::atomic<int> FW_OVER(0);
static void forward_pass(const Run& R, std::vector<Hit>& out, std::vector<FHit>& fout)
{
    const double tol = M_TAU + M_TOLF + 16 * U53;
    for (int s = M_NT + 1; s <= std::max(R.zhi, R.fmax); s++) {
        const uint64_t zs = salt_zone(s);
        const bool dozone = (s >= R.zlo && s <= R.zhi), doex = (s <= R.fmax) && (!M_F32 || M_F32CHECK);
        if (!dozone && !doex) continue;
        for (int i = std::max(1, s - M_NT); i <= std::min(M_NT, s - 1); i++) {
            const int j = s - i;
            const MLev& A = ml[i];
            const MLev& B = ml[j];
            const bool overA = A.n >= B.n;                      // the parallel loop runs over blocks of the longer level
            // (blocks small enough that every thread gets many: a block with a deep tree above every pair is long)
            const size_t nlong = overA ? A.n : B.n, BL = std::max((size_t)1, std::min((size_t)2048, nlong / ((size_t)omp_get_max_threads() * 64))), nblocks = (nlong + BL - 1) / BL;
#pragma omp parallel
            {
                MStat& S = stat_me();
                std::vector<Hit> mine;
                std::vector<FHit> fmine;
                FCtx F;
                F.R = &R; F.fmax = R.fmax; F.c0 = s; F.i = i; F.nf = 0; F.mine = &fmine; F.S = &S;
                const int BN = 64;
                struct FE { double vr, vi, vi0; uint32_t a, b; int nh; uint64_t h[16]; uint8_t hk[16]; };
                FE batch[BN];
                int nb = 0;
                unsigned long long nzrej = 0;
                auto probe1 = [&](double vr, double vi, double vi0, uint32_t a, uint32_t b, uint64_t h, int hk) {
                    uint64_t pos = h & rsmask;
                    double m = std::max(std::fabs(vr), std::fabs(vi)) * tol;
                    for (;;) {
                        const Zd& z = rst[pos];
                        if (dbits(z.re) == RS_EMPTY) return;
                        double di = z.im - vi;
                        if (std::fabs(di) > 3.0) di -= std::copysign(TWOPI_D, di);       // the keys are values modulo 2 pi i
                        if (std::fabs(z.re - vr) <= m && std::fabs(di) <= m) {
                            if ((dbits(z.im) & 1) && std::fabs(z.im - vi0) <= m + 1e-300) S.fw_drop++;      // (the value of a smaller level itself)
                            else if (M_ZQ && hk == 0 && lev[i].dx[a] == NOD && lev[j].dx[b] == NOD
                                     && !zq_keep(s, dbits(z.re), dbits(z.im), lev[i].E[a].z[0], lev[j].L[b].z[0],
                                                 lev[i].wE[(size_t)a * NP] + lev[j].wL[(size_t)b * NP])) nzrej++;
                            else if (mine.size() < 50000000) {
                                mine.push_back(Hit{ dbits(z.re), dbits(z.im), a, b, (uint8_t)s, (uint8_t)i, (uint8_t)hk });
                                S.fw_hits++;
                            } else FW_OVER = 1;
                        }
                        pos = (pos + 1) & rsmask;
                    }
                };
                auto flush = [&]() {
                    for (int t = 0; t < nb; t++) for (int k = 0; k < batch[t].nh; k++) probe1(batch[t].vr, batch[t].vi, batch[t].vi0, batch[t].a, batch[t].b, batch[t].h[k], batch[t].hk[k]);
                    nb = 0;
                };
                // pk: the kind of the pair: 0 without structure below the resolution; 1 with the displacement dz, known
                // and below 2^-100 of the value; 2 anything else
                auto enqueue = [&](double vr, double vi, uint32_t a, uint32_t b, int pkind, const Zd& dz, double vi0 = 1e300) {
                    Zd z{ vr, vi };
                    uint64_t hs[16], hd[8], hall[40];
                    uint8_t kall[40];
                    int nall = 0;
                    if (!dozone) return;
                    if (M_CLEAN && pkind != 0) return;
                    for (int pass = 0; pass < (pkind == 2 ? 2 : 1); pass++) {
                        const uint64_t zsalt = pass ? salt_zone_u(s) : zs;
                        int n = cells(z, tol, zsalt, hs);
                        if (std::fabs(vi) > PI_D - 2 * tol * amxd(z) - 1e-12 && !REAL_ONLY) {
                            Zd w{ vr, vi + ((vi > 0) ? -TWOPI_D : TWOPI_D) };
                            n += cells(w, tol, zsalt, hs + n);
                        }
                        if (pkind == 1) {
                            int nd = cells(dz, MTOLD, 0x7777, hd, MBD);
                            for (int k = 0; k < n; k++) for (int j = 0; j < nd; j++) if (nall < 40) { kall[nall] = 1; hall[nall++] = hcomb(hs[k], hd[j]); }
                        } else for (int k = 0; k < n; k++) if (nall < 40) { kall[nall] = pass ? 2 : 0; hall[nall++] = hs[k]; }
                    }
                    int k0 = 0;
                    while (k0 < nall) {
                        FE& e = batch[nb];
                        e.vr = vr; e.vi = vi; e.vi0 = vi0; e.a = a; e.b = b; e.nh = 0;
                        for (; k0 < nall && e.nh < 16; k0++) {
                            if (R.npart > 1 && (int)((hall[k0] >> 44) % (uint64_t)R.npart) != R.part) continue;
                            e.hk[e.nh] = kall[k0];
                            e.h[e.nh++] = hall[k0];
                            __builtin_prefetch(&rst[hall[k0] & rsmask]);
                        }
                        if (e.nh && ++nb == BN) flush();
                    }
                };
                // the kind of the pair (a, b) with the value of size mv, and its displacement
                auto pair_kind = [&](uint32_t a, uint32_t b, double mv, Zd& dz) -> int {
                    const Level& LA = lev[i];
                    const Level& LB = lev[j];
                    dz = Zd{ 0, 0 };
                    bool known = true;
                    if (LA.dx[a] != NOD) {
                        Zd d = dZd(LA.dE[LA.dx[a]].z[0]);
                        dz.re += d.re; dz.im += d.im;
                        if ((LA.fl[a] & F_S) && asumd(d) == 0) known = false;
                    }
                    if (LB.dx[b] != NOD) {
                        Zd d = dZd(LB.dL[LB.dx[b]].z[0]);
                        dz.re -= d.re; dz.im -= d.im;
                        if ((LB.fl[b] & F_S) && asumd(d) == 0) known = false;
                    }
                    int xe = LA.xE[(size_t)a * NP], xl = LB.xL[(size_t)b * NP];
                    if (!(LA.fE[a] & F_EZERO) && xl != XNONE && std::abs(xe - xl) > FS) {      // (--inf: E exactly 0 is not an operand below the resolution)
                        if (xe > xl) { Zd l = toZd(LB.L[b].z[0]); dz.re -= l.re; dz.im -= l.im; }
                        else { Zd e2 = toZd(LA.E[a].z[0]); dz.re += e2.re; dz.im += e2.im; }
                    }
                    double md = asumd(dz);
                    if (!known || !std::isfinite(md)) return 2;
                    if (md == 0) return 0;
                    return md < MDLIM * mv ? 1 : 2;
                };
                // a pair whose value lost digits in the subtraction: quadruple precision, then tier 3
                auto slow = [&](uint32_t a, uint32_t b) {
                    const Level& LA = lev[i];
                    const Level& LB = lev[j];
                    const C& E = LA.E[a].z[0];
                    const C& L = LB.L[b].z[0];
                    q vr = E.re - L.re, vy = E.y - L.y;
                    vy -= 2 * nearbyintq(vy / 2);
                    double ae = (LA.wE[(size_t)a * NP] + LB.wL[(size_t)b * NP]) * ULP;
                    if (LA.dx[a] != NOD) { const D& d = LA.dE[LA.dx[a]].z[0]; ae += std::fabs(d.re) + PI_D * std::fabs(d.y); }
                    if (LB.dx[b] != NOD) { const D& d = LB.dL[LB.dx[b]].z[0]; ae += std::fabs(d.re) + PI_D * std::fabs(d.y); }
                    double dr = (double)vr, di = (double)(PIq * vy), m = std::max(std::fabs(dr), std::fabs(di));
                    ae += 4 * U_(q(0.0)) * m;
                    Zd dz;
                    if (ae <= M_TOLF * m) { S.fw_quad++; int pkd = pair_kind(a, b, m, dz); enqueue(dr, di, a, b, pkd, dz); return; }
                    if (m + ae < 0.25 * M_STINY) { S.fw_tiny++; return; }        // (requests this small are not in the store)
                    T3Res r = t3_candidate(s, i, a, b, false);
                    S.fw_t3++;
                    if (r.status == T_FAIL) { S.fw_fail++; return; }
                    q y3 = r.v.z[0].y - 2 * nearbyintq(r.v.z[0].y / 2);
                    dr = (double)r.v.z[0].re; di = (double)(PIq * y3);
                    if (std::max(std::fabs(dr), std::fabs(di)) < 0.25 * M_STINY) { S.fw_tiny++; return; }
                    enqueue(dr, di, a, b, 2, dz);                                // (its digits are from tier 3: every kind of request)
                };
#pragma omp for schedule(dynamic, 1) nowait
                for (long long blk = 0; blk < (long long)nblocks; blk++) {
                    size_t l0 = (size_t)blk * BL, l1 = std::min(nlong, l0 + BL);
                    size_t a0 = overA ? l0 : 0, a1 = overA ? l1 : A.n, b0 = overA ? 0 : l0, b1 = overA ? B.n : l1;
                    const Zd nodz{ 0, 0 };
                    for (size_t a = a0; a < a1; a++) {
                        if (!(A.f[a] & MF_E)) continue;
                        const Zd Ea = A.E[a * NP];
                        const double sE = asumd(Ea);
                        const bool ad = A.f[a] & MF_D;
                        const bool ez = (i == MINF_LEV && (uint32_t)a == MINF_IDX);       // --inf: the entry -inf, E = 0 exactly
                        unsigned long long np = 0;
                        for (size_t b = b0; b < b1; b++) {
                            const uint8_t fb = B.f[b];
                            if (!(fb & MF_L)) continue;
                            const Zd& Lb = B.L[b * NP];
                            double vr = Ea.re - Lb.re, vi = Ea.im - Lb.im;
                            const double vi0 = vi;
                            if (vi > PI_D || vi <= -PI_D) vi -= TWOPI_D * std::nearbyint(vi / TWOPI_D);
                            double m = std::max(std::fabs(vr), std::fabs(vi)), sL = asumd(Lb);
                            np++;
                            if (doex) {
                                // the pair as a term run forward (its exp against the requests exp(exp(X)) = c; entries on top)
                                const bool sl = ad || (fb & MF_D) || (fb & MF_S) || (A.f[a] & MF_S);
                                if (!(M_CLEAN && (sl || absb2(amxd(Ea), amxd(Lb))))) {
                                    Zd v0{ Ea.re - Lb.re, Ea.im - Lb.im };
                                    double m0 = amxd(v0), a0e = 3 * U53 * (sE + sL);
                                    F.a = (uint32_t)a; F.b = (uint32_t)b; F.nf = 0;
                                    const XC c0 = xc_sub(xc_d(A.cE[a] & 3, Ea), xc_d(B.cL[b] & 3, Lb));
                                    if (m0 > 0 || !(c0.rz && c0.yh && c0.y2 == 0))      // (0 in double precision is not the number 0, unless the marks say so)
                                        fnode(F, s, m0 > 0 ? v0 : Zd{ 1e-200, 0 }, (m0 > 0 && a0e < 0.25 * m0) ? a0e / m0 : 1e9, c0, false, Zd{ 0, 0 }, 0);
                                }
                            }
                            if (!dozone) continue;
                            if (!(3 * U53 * (sE + sL) <= M_TOLF * m)) { slow((uint32_t)a, (uint32_t)b); continue; }
                            if (ad || (fb & MF_D) || (!ez && (sL < 8 * MAB * sE || sE < 8 * MAB * sL))) {
                                Zd dz;
                                int pkd = pair_kind((uint32_t)a, (uint32_t)b, m, dz);
                                enqueue(vr, vi, (uint32_t)a, (uint32_t)b, pkd, dz, vi0);
                            } else enqueue(vr, vi, (uint32_t)a, (uint32_t)b, 0, nodz, vi0);
                        }
                        S.fw_pairs += np;
                    }
                }
                flush();
                ZQ_REJ += nzrej; ZQ_KEEP += mine.size();
#pragma omp critical(hitmerge)
                {
                    out.insert(out.end(), mine.begin(), mine.end());
                    fout.insert(fout.end(), fmine.begin(), fmine.end());
                }
            }
        }
    }
}

// ---------------------------------------------------------------- the sieve in single precision (f32kern.h)
//
// With --f32 the terms run forward are first run in single precision (f32kern.h), on the processor (1) or on the
// graphics card (2). What the sieve reports (a term whose exp may be a request; a part of the tree it could not
// compute) is then run by fnode as before, in double precision and above. Only terms without structure below the
// resolution (--clean), since the sieve sees the digits only.

#define F32_COUNT(x) x
#define F32_WHY(x) x
#define F32_DBG(x) x
struct F32N;
#include "f32kern.h"
#include <x86intrin.h>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <deque>

struct F32Host {
    bool built = false;
    std::vector<F32W> W;                 // the entries put on top (levels 1..UW), with their level and index
    std::vector<uint8_t> Wlev;
    std::vector<uint32_t> Worig;
    int wcnt[24], UW = 0;
    std::vector<F32A> A[32];             // exp sides of the entries of a level (those of the class), and their indices
    std::vector<uint32_t> Aorig[32];
    std::vector<F32B> B[32];             // Log sides
    std::vector<uint32_t> Borig[32];
    std::vector<float> rqre, rqim;
    F32G g;
};
static F32Host f32h;
struct F32Stat { unsigned long long nodes = 0, tops = 0, nrec[4] = { 0, 0, 0, 0 }, cpu[4] = { 0, 0, 0, 0 }, why[8] = { 0, 0, 0, 0, 0, 0, 0, 0 }, fwhy[8] = { 0, 0, 0, 0, 0, 0, 0, 0 }, why2[2] = { 0, 0 }; };
static F32Stat f32tot;

static inline F32f f32_ff(double d) { F32f r; r.h = (float)d; r.l = std::isfinite(r.h) ? (float)(d - (double)r.h) : 0.0f; return r; }
static bool f32_makeA(int s, size_t k, F32A& a)
{
    const MLev& M = ml[s];
    a = F32A{ { 0, 0 }, { 0, 0 }, { 0, 0 }, { 0, 0 }, 0, 0, 0 };
    if (!(M.f[k] & MF_E)) return false;
    const Zd E = M.E[k * NP];
    const double m = amxd(E);
    uint32_t meta = 0;
    const XC c = xc_d(M.cE[k] & 3, E);
    if (c.rz) meta |= F32_M_RZ;
    if (c.yh && std::labs((long)c.y2) < 30000) meta |= F32_M_YH | ((uint32_t)(uint16_t)(int16_t)c.y2 << 8);
    // Log(exp(V)): V with the imaginary part reduced to (-pi, pi] (exact in units of pi)
    const C& vq = lev[s].v[k].z[0];
    q y = vq.y - 2 * nearbyintq(vq.y / 2);
    if (y <= -1) y += 2;
    a.LEr = f32_ff((double)vq.re); a.LEi = f32_ff((double)(PIq * y));
    if (!(m <= 1e30 && m >= 1e-30)) meta |= (m > 1e30) ? F32_A_HUGE : F32_A_TINY;
    else {
        meta |= F32_A_OK;
        a.Er = f32_ff(E.re); a.Ei = f32_ff(E.im);
        if (M.f[k] & MF_EE) {
            const Zd EE = M.EE[k * NP];
            const double mm = amxd(EE);
            if (mm >= 1e-30 && mm <= 1e30) { meta |= F32_A_EE; a.EEr = (float)EE.re; a.EEi = (float)EE.im; }
        }
    }
    a.meta = meta;
    return true;
}
static bool f32_makeB(int s, size_t k, F32B& b)
{
    const MLev& M = ml[s];
    b = F32B{ { 0, 0 }, { 0, 0 }, 0, 0, 0, 0 };
    if (!(M.f[k] & MF_L)) return false;
    const Zd L = M.L[k * NP], V = M.V[k * NP];
    uint32_t meta = F32_B_OK;
    const XC c = xc_d(M.cL[k] & 3, L);
    if (c.rz) meta |= F32_M_RZ;
    if (c.yh && std::labs((long)c.y2) < 30000) meta |= F32_M_YH | ((uint32_t)(uint16_t)(int16_t)c.y2 << 8);
    b.Lr = f32_ff(L.re); b.Li = f32_ff(L.im);
    const double m = amxd(L);
    if (m == 0) { meta |= F32_B_LZ; b.lnL = -1e30f; } else b.lnL = (float)std::log(m);
    const double mv = amxd(V);
    if (mv >= 1e-30 && mv <= 1e30) { const Zd iv = zinv(V); meta |= F32_B_IV; b.IVr = (float)iv.re; b.IVi = (float)iv.im; }
    b.meta = meta;
    return true;
}
static void f32_build()
{
    F32Host& H = f32h;
    if (H.built) return;
    double t0 = now();
    H.UW = std::min(M_NT, 13);
    for (int r = 0; r < 24; r++) H.wcnt[r] = 0;
    unsigned long long na = 0, nb = 0, nhuge = 0, ntiny = 0, nee = 0, niv = 0;
    for (int u = 1; u <= M_NT; u++) {
        const MLev& M = ml[u];
        H.A[u].reserve(M.n); H.Aorig[u].reserve(M.n); H.B[u].reserve(M.n); H.Borig[u].reserve(M.n);
        for (size_t k = 0; k < M.n; k++) {
            if (M.f[k] & (MF_S | MF_D)) continue;
            F32A a;
            F32B b;
            const bool ha = f32_makeA(u, k, a), hb = f32_makeB(u, k, b);
            if (ha) {
                H.A[u].push_back(a); H.Aorig[u].push_back((uint32_t)k); na++;
                if (a.meta & F32_A_HUGE) nhuge++;
                if (a.meta & F32_A_TINY) ntiny++;
                if (a.meta & F32_A_EE) nee++;
            }
            if (hb) { H.B[u].push_back(b); H.Borig[u].push_back((uint32_t)k); nb++; if (b.meta & F32_B_IV) niv++; }
            if (u <= H.UW && (ha || hb)) {
                F32W w;
                w.a = a; w.b = b; w.u = (uint32_t)u;
                H.W.push_back(w); H.Wlev.push_back((uint8_t)u); H.Worig.push_back((uint32_t)k);
            }
        }
        if (u <= H.UW) H.wcnt[u] = (int)H.W.size();
    }
    for (int r = H.UW + 1; r < 24; r++) H.wcnt[r] = H.wcnt[H.UW];
    H.built = true;
    printf("sieve in single precision: %llu exp sides (%llu beyond the range upward, %llu downward; exp(exp) in range: %llu), %llu Log sides (1/value in range: %llu), "
           "%zu entries to put on top (levels 1..%d), %.1f s, %.1f GB in use\n", na, nhuge, ntiny, nee, nb, niv, H.W.size(), H.UW, now() - t0, mem_gb());
    fflush(stdout);
}
static void f32_requests(const Run& R)
{
    F32Host& H = f32h;
    H.rqre.clear(); H.rqim.clear();
    for (int c = 0; c < 72; c++) { H.g.rqoff[c] = 0; H.g.rqn[c] = 0; }
    for (int c = M_NT + 1; c <= R.fmax && c < 64; c++) {
        std::vector<Zd>& v = e2all[c];
        std::sort(v.begin(), v.end(), [](const Zd& x, const Zd& y) { return x.re < y.re || (x.re == y.re && x.im < y.im); });
        v.erase(std::unique(v.begin(), v.end(), [](const Zd& x, const Zd& y) { return x.re == y.re && x.im == y.im; }), v.end());
        H.g.rqoff[c] = (int)H.rqre.size(); H.g.rqn[c] = (int)v.size();
        for (const Zd& z : v) { H.rqre.push_back((float)z.re); H.rqim.push_back((float)z.im); }
    }
    H.g.rqre = H.rqre.data(); H.g.rqim = H.rqim.data();
    H.g.fmax = R.fmax; H.g.UW = H.UW; H.g.W = H.W.data();
    for (int r = 0; r < 24; r++) H.g.wcnt[r] = H.wcnt[r];
    H.g.relmax = (float)M_F32REL; H.g.emax = (float)M_F32REL;
    H.g.flags = getenv("F32_FLAGS") ? atoi(getenv("F32_FLAGS")) : 0;
    H.g.unitfast = (!H.W.empty() && H.wcnt[1] == 1 && (H.W[0].a.meta & F32_A_OK) && (H.W[0].a.meta & F32_A_EE)) ? 1 : 0;
    if (std::min(M_NT, R.fmax - M_NT - 1) > H.UW) { fprintf(stderr, "the sieve has the entries put on top up to level %d only\n", H.UW); exit(1); }
}

// A record of the sieve, run in double precision and above as without the sieve.
static void f32_consume(FCtx& F, const F32Rec& r)
{
    const F32Host& H = f32h;
    FStep only[F32_MAXD + 1];
    for (int t = 0; t < r.nf; t++) {
        const uint32_t w = r.st[t] >> 1;
        only[t].lev = H.Wlev[w]; only[t].idx = H.Worig[w]; only[t].side = (uint8_t)(r.st[t] & 1); only[t].alt = 0;
    }
    const int s = r.s, i = r.i, j = s - i;
    const uint32_t a = H.Aorig[i][r.a], b = H.Borig[j][r.b];
    const MLev& A = ml[i];
    const MLev& B = ml[j];
    const Zd Ea = A.E[(size_t)a * NP], Lb = B.L[(size_t)b * NP];
    if (absb2(amxd(Ea), amxd(Lb))) return;
    Zd v0{ Ea.re - Lb.re, Ea.im - Lb.im };
    const double m0 = amxd(v0), a0e = 3 * U53 * (asumd(Ea) + asumd(Lb));
    const XC c0 = xc_sub(xc_d(A.cE[a] & 3, Ea), xc_d(B.cL[b] & 3, Lb));
    if (!(m0 > 0) && c0.rz && c0.yh && c0.y2 == 0) return;
    F.only = only; F.nonly = r.nf; F.omode = r.type;
    F.c0 = s; F.i = i; F.a = a; F.b = b; F.nf = 0;
    fnode(F, s, m0 > 0 ? v0 : Zd{ 1e-200, 0 }, (m0 > 0 && a0e < 0.25 * m0) ? a0e / m0 : 1e9, c0, false, Zd{ 0, 0 }, 0);
    F.only = nullptr;
}

struct F32OutCpu {
    FCtx* F;
    int s, i, npre = 0;
    uint32_t a, b;
    unsigned long long nodes = 0, tops = 0, nrec[4] = { 0, 0, 0, 0 }, cpu[4] = { 0, 0, 0, 0 }, why[8] = { 0, 0, 0, 0, 0, 0, 0, 0 }, fwhy[8] = { 0, 0, 0, 0, 0, 0, 0, 0 }, why2[2] = { 0, 0 };
    void dbg(int what, const F32N& C, float er, int nf, const uint32_t* st)
    {
        static std::atomic<int> cnt(0);
        if (!getenv("F32_DBG") || (cnt++ % atoi(getenv("F32_DBG"))) != 0) return;
#pragma omp critical(dbgq)
        {
            printf("F32DBG %d: size %d depth %d v=(%.7g, %.7g) ea=%.2e er=%.2e marks %d y2 %d lk %d X=(%.6g, %.6g) Y=(%.6g, %.6g) |", what, (int)C.c, nf, f2_val(C.vr), f2_val(C.vi), C.ea, er, (int)C.mk,
                   C.y2, 0, 0.0, 0.0, 0.0, 0.0);
            printf(" PATH");
            for (int t = 0; t < nf; t++) printf(" %u:%u", st[t] & 1, st[t] >> 1);
            if (nf == 0) printf(" pair");
            printf("\n");
        }
    }
    void rec(int type, int nf, const uint32_t* st)
    {
        F32Rec r;
        r.a = a; r.b = b; r.s = (uint8_t)s; r.i = (uint8_t)i; r.nf = (uint8_t)nf; r.type = (uint8_t)type;
        for (int t = 0; t < nf; t++) r.st[t] = st[t];
        nrec[type]++;
        const unsigned long long n0 = F->S->fx_nodes;
        f32_consume(*F, r);
        cpu[type] += F->S->fx_nodes - n0;
    }
};

#if !defined(F32_GPU)
static void forward_f32_gpu(const Run&, std::vector<FHit>&) { fprintf(stderr, "this build has no graphics card part (--f32 2)\n"); exit(1); }
#else
extern "C" {
int f32gpu_init(unsigned cap, unsigned unitcap);
int f32gpu_level(int lev, const F32A* A, size_t na, const F32B* B, size_t nb);
int f32gpu_wrappers(const F32W* W, size_t nw);
int f32gpu_setup(const F32G* g, size_t nrq);
int f32gpu_run(int s, int i, uint32_t a0, uint32_t na, uint32_t b0, uint32_t nb, uint32_t bstep, int ccut, F32Rec* out, unsigned* nrec, unsigned* nunits, unsigned long long* stats);
void f32gpu_free();
void f32gpu_times(double* tk, double* tc);
}
static double M_F32WORK = 1e10;        // --f32work: nodes of one launch on the card (bounded by the buffer of records)
static unsigned M_F32CAP = 1u << 24;   // records of one launch
static unsigned M_F32UCAP = 1u << 22;  // units of one launch (second phase)
static int M_F32R0 = 100;              // a node with at most this many leaves to go is a unit of a second launch (100: no second launches; they brought no gain)
static double f32_tgpu = 0, f32_tcpu = 0, f32_twait = 0;
static unsigned long long f32_ticks[4] = { 0, 0, 0, 0 };      // processor time of the records by kind (clock ticks, all threads)
static unsigned long long f32_launches = 0, f32_retries = 0, f32_units = 0, f32_bad = 0;

// The sieve on the graphics card. One thread drives the card, launch after launch; the records of a launch are run
// on the processor (all other threads) while the card works on the next one.
static void forward_f32_gpu(const Run& R, std::vector<FHit>& fout)
{
    F32Host& H = f32h;
    static bool up = false;
    static const int NBUF = 3;
    static std::vector<F32Rec> pool[NBUF];
    if (!up) {
        double t0 = now();
        if (f32gpu_init(M_F32CAP, M_F32UCAP)) exit(1);
        for (int u = 1; u <= M_NT; u++) if (f32gpu_level(u, H.A[u].data(), H.A[u].size(), H.B[u].data(), H.B[u].size())) exit(1);
        if (f32gpu_wrappers(H.W.data(), H.W.size())) exit(1);
        for (int k = 0; k < NBUF; k++) pool[k].resize(M_F32CAP);
        up = true;
        printf("graphics card: the table is on it (%.1f s)\n", now() - t0);
        fflush(stdout);
    }
    if (f32gpu_setup(&H.g, H.rqre.size())) exit(1);
    // what one pair costs with r leaves left above it (nodes, roughly)
    double est[32];
    est[0] = 1;
    for (int r = 1; r < 32; r++) {
        double e = 1;
        for (int u = 1; u <= std::min(H.UW, r); u++) e += 2.0 * (double)(H.wcnt[u] - H.wcnt[u - 1]) * est[r - u];
        est[r] = e;
    }
    struct Batch { int buf; unsigned nrec; unsigned long long st[24]; };
    std::mutex mx;
    std::condition_variable cv;
    std::deque<Batch> ready;
    std::deque<int> freeb;
    for (int k = 0; k < NBUF; k++) freeb.push_back(k);
    bool done = false;
    const bool lg = getenv("F32_LAUNCH") != nullptr;
    std::thread card([&]() {
        struct Job { int s, i; uint32_t a0, na, b0, nb, bstep; };
        for (int s = M_NT + 1; s <= R.fmax; s++) {
            const int r = R.fmax - s;
            const int ccut = (r > M_F32R0) ? R.fmax - M_F32R0 : 1000;
            for (int i = std::max(1, s - M_NT); i <= std::min(M_NT, s - 1); i++) {
                const int j = s - i;
                const size_t NA = H.A[i].size(), NB = H.B[j].size();
                if (!NA || !NB) continue;
                const double per = est[r];
                // a launch: a rectangle of pairs worth about M_F32WORK nodes, cut into about 4e5 threads (a thread: one a, a run of b)
                uint32_t nbj = (uint32_t)std::min((double)NB, std::max(1.0, std::floor(M_F32WORK / per)));
                uint32_t naj = (uint32_t)std::min((double)NA, std::max(1.0, std::floor(M_F32WORK / (per * (double)nbj))));
                const double nch = std::min((double)nbj, std::max(1.0, std::ceil(4e5 / (double)naj)));
                uint32_t bstep = (uint32_t)std::ceil((double)nbj / nch);
                bstep = (uint32_t)std::min((double)bstep, std::max(1.0, std::floor(1e6 / per)));
                std::vector<Job> jobs;
                for (size_t b0 = 0; b0 < NB; b0 += nbj)
                    for (size_t a0 = 0; a0 < NA; a0 += naj)
                        jobs.push_back(Job{ s, i, (uint32_t)a0, (uint32_t)std::min((size_t)naj, NA - a0), (uint32_t)b0, (uint32_t)std::min((size_t)nbj, NB - b0), bstep });
                while (!jobs.empty()) {
                    Job jb = jobs.back();
                    jobs.pop_back();
                    int bi;
                    {
                        double tw = now();
                        std::unique_lock<std::mutex> lk(mx);
                        cv.wait(lk, [&] { return !freeb.empty(); });
                        bi = freeb.front();
                        freeb.pop_front();
                        f32_twait += now() - tw;
                    }
                    Batch bt;
                    bt.buf = bi; bt.nrec = 0;
                    unsigned nun = 0;
                    double t0 = now();
                    if (f32gpu_run(jb.s, jb.i, jb.a0, jb.na, jb.b0, jb.nb, jb.bstep, ccut, pool[bi].data(), &bt.nrec, &nun, bt.st)) exit(1);
                    double t1 = now();
                    f32_tgpu += t1 - t0;
                    f32_launches++;
                    if (bt.nrec > M_F32CAP || nun > M_F32UCAP) {              // a buffer was too small: the job in two halves
                        f32_retries++;
                        { std::lock_guard<std::mutex> lk(mx); freeb.push_back(bi); }
                        cv.notify_all();
                        if (jb.na > 1) {
                            uint32_t h = jb.na / 2;
                            jobs.push_back(Job{ jb.s, jb.i, jb.a0, h, jb.b0, jb.nb, jb.bstep });
                            jobs.push_back(Job{ jb.s, jb.i, jb.a0 + h, jb.na - h, jb.b0, jb.nb, jb.bstep });
                        } else if (jb.nb > 1) {
                            uint32_t h = jb.nb / 2;
                            uint32_t bs = std::min(jb.bstep, std::max(1u, h));
                            jobs.push_back(Job{ jb.s, jb.i, jb.a0, 1, jb.b0, h, bs });
                            jobs.push_back(Job{ jb.s, jb.i, jb.a0, 1, jb.b0 + h, jb.nb - h, bs });
                        } else { fprintf(stderr, "the sieve: one pair gives more records than the buffer holds\n"); exit(1); }
                        continue;
                    }
                    f32_units += nun; f32_bad += bt.st[20];
                    if (lg)
                        printf("LAUNCH s=%d i=%d r=%d na=%u nb=%u bstep=%u threads=%.3g est/pair=%.3g units=%u nodes=%.4g (inner %.4g) records=%u time=%.3f s  rate=%.3g nodes/s\n", jb.s, jb.i, r, jb.na, jb.nb,
                               jb.bstep, (double)jb.na * std::ceil((double)jb.nb / jb.bstep), per, nun, (double)(bt.st[0] + bt.st[1]), (double)bt.st[0], bt.nrec, t1 - t0,
                               (double)(bt.st[0] + bt.st[1]) / (t1 - t0));
                    { std::lock_guard<std::mutex> lk(mx); ready.push_back(bt); }
                    cv.notify_all();
                }
            }
        }
        { std::lock_guard<std::mutex> lk(mx); done = true; }
        cv.notify_all();
    });
    const int nth = omp_get_max_threads();
    std::vector<std::vector<FHit>> fm(nth);
    std::vector<FCtx> ctx(nth);
    for (;;) {
        Batch bt;
        {
            std::unique_lock<std::mutex> lk(mx);
            cv.wait(lk, [&] { return !ready.empty() || done; });
            if (ready.empty()) break;
            bt = ready.front();
            ready.pop_front();
        }
        double t1 = now();
        f32tot.nodes += bt.st[0]; f32tot.tops += bt.st[1];
        for (int t = 0; t < 8; t++) { f32tot.why[t] += bt.st[2 + t]; f32tot.fwhy[t] += bt.st[10 + t]; }
        f32tot.why2[0] += bt.st[18]; f32tot.why2[1] += bt.st[19];
        const F32Rec* recs = pool[bt.buf].data();
        const long long nrec = bt.nrec;
        unsigned long long nr[4] = { 0, 0, 0, 0 }, cp[4] = { 0, 0, 0, 0 }, tk[4] = { 0, 0, 0, 0 };
#pragma omp parallel reduction(+ : nr[:4], cp[:4], tk[:4])
        {
            const int me = omp_get_thread_num();
            MStat& S = stat_me();
            FCtx& F = ctx[me];
            F.R = &R; F.fmax = R.fmax; F.mine = &fm[me]; F.S = &S; F.only = nullptr;
#pragma omp for schedule(dynamic, 512)
            for (long long k = 0; k < nrec; k++) {
                const F32Rec& rc = recs[k];
                const unsigned long long n0 = S.fx_nodes, c0 = __rdtsc();
                f32_consume(F, rc);
                nr[rc.type]++; cp[rc.type] += S.fx_nodes - n0; tk[rc.type] += __rdtsc() - c0;
            }
        }
        for (int t = 0; t < 4; t++) { f32tot.nrec[t] += nr[t]; f32tot.cpu[t] += cp[t]; f32_ticks[t] += tk[t]; }
        f32_tcpu += now() - t1;
        { std::lock_guard<std::mutex> lk(mx); freeb.push_back(bt.buf); }
        cv.notify_all();
    }
    card.join();
    size_t nfh = fout.size();
    for (int t = 0; t < nth; t++) nfh += fm[t].size();
    fout.reserve(nfh);
    for (int t = 0; t < nth; t++) { fout.insert(fout.end(), fm[t].begin(), fm[t].end()); std::vector<FHit>().swap(fm[t]); }
}
#endif

// the sieve on the processor (the reference for the graphics card: the same kernel)
static void forward_f32(const Run& R, std::vector<FHit>& fout)
{
    F32Host& H = f32h;
    f32_build();
    f32_requests(R);
    if (M_F32 == 2) { forward_f32_gpu(R, fout); return; }
    for (int s = M_NT + 1; s <= R.fmax; s++) {
        for (int i = std::max(1, s - M_NT); i <= std::min(M_NT, s - 1); i++) {
            const int j = s - i;
            const std::vector<F32A>& A = H.A[i];
            const std::vector<F32B>& B = H.B[j];
            const bool overA = A.size() >= B.size();
            const long long nlong = (long long)(overA ? A.size() : B.size());
#pragma omp parallel
            {
                MStat& S = stat_me();
                std::vector<FHit> fmine;
                FCtx F;
                F.R = &R; F.fmax = R.fmax; F.c0 = s; F.i = i; F.nf = 0; F.mine = &fmine; F.S = &S;
                F32OutCpu out;
                out.F = &F; out.s = s; out.i = i;
                F32N st[F32_MAXD + 2];
                uint32_t steps[F32_MAXD + 1], usteps[F32_MAXD + 1];
                F32NoUnits nu;
                unsigned long long hist[48];
                for (int k = 0; k < 48; k++) hist[k] = 0;
                unsigned long long hmax = 0, hsum = 0, hn = 0;
                auto one = [&](const F32A& a_, const F32B& b_) {
                    const unsigned long long n0 = out.nodes + out.tops;
                    f32_pair(H.g, s, a_, b_, st, steps, usteps, out, 1000, nu, 0);
                    const unsigned long long dn = out.nodes + out.tops - n0;
                    int k = 0;
                    while ((1ULL << k) < dn + 1 && k < 47) k++;
                    hist[k]++; hsum += dn; hn++;
                    if (dn > hmax) hmax = dn;
                };
#pragma omp for schedule(dynamic, 16) nowait
                for (long long l = 0; l < nlong; l++) {
                    if (overA) { out.a = (uint32_t)l; for (size_t b = 0; b < B.size(); b++) { out.b = (uint32_t)b; one(A[l], B[b]); } }
                    else { out.b = (uint32_t)l; for (size_t a = 0; a < A.size(); a++) { out.a = (uint32_t)a; one(A[a], B[l]); } }
                }
                if (getenv("F32_HIST")) {
#pragma omp critical(dbgq)
                    {
                        printf("HIST s=%d i=%d r=%d thread %d: pairs %llu mean %.1f max %llu | by log2:", s, i, R.fmax - s, omp_get_thread_num(), hn, hn ? (double)hsum / hn : 0.0, hmax);
                        for (int k = 0; k < 48; k++) if (hist[k]) printf(" %d:%llu", k, hist[k]);
                        printf("\n");
                    }
                }
#pragma omp critical(hitmerge)
                {
                    fout.insert(fout.end(), fmine.begin(), fmine.end());
                    f32tot.nodes += out.nodes; f32tot.tops += out.tops;
                    for (int t = 0; t < 4; t++) { f32tot.nrec[t] += out.nrec[t]; f32tot.cpu[t] += out.cpu[t]; }
                    for (int t = 0; t < 8; t++) { f32tot.why[t] += out.why[t]; f32tot.fwhy[t] += out.fwhy[t]; }
                    f32tot.why2[0] += out.why2[0]; f32tot.why2[1] += out.why2[1];
                }
            }
        }
    }
}

// (--f32check) the hits of the plain way must all be among those that the sieve leads to
static void f32_compare(std::vector<FHit>& plain, std::vector<FHit>& sieve, int n)
{
    auto key = [](const FHit& h) {
        std::string k((const char*)&h.kre, 8);
        k.append((const char*)&h.kim, 8); k.append((const char*)&h.a, 4); k.append((const char*)&h.b, 4);
        k.push_back((char)h.s); k.push_back((char)h.i); k.push_back((char)h.c0); k.push_back((char)h.nf);
        for (int t = 0; t < h.nf; t++) { k.push_back((char)h.st[t].lev); k.push_back((char)h.st[t].side); k.push_back((char)h.st[t].alt); k.append((const char*)&h.st[t].idx, 4); }
        return k;
    };
    std::vector<std::string> kp, ks;
    for (const FHit& h : plain) kp.push_back(key(h));
    for (const FHit& h : sieve) ks.push_back(key(h));
    std::sort(kp.begin(), kp.end()); kp.erase(std::unique(kp.begin(), kp.end()), kp.end());
    std::sort(ks.begin(), ks.end()); ks.erase(std::unique(ks.begin(), ks.end()), ks.end());
    std::vector<std::string> miss, extra;
    std::set_difference(kp.begin(), kp.end(), ks.begin(), ks.end(), std::back_inserter(miss));
    std::set_difference(ks.begin(), ks.end(), kp.begin(), kp.end(), std::back_inserter(extra));
    printf("      [sieve check n=%d: terms that met a request: plain way %zu, through the sieve %zu; MISSED BY THE SIEVE %zu; only through the sieve %zu]\n", n, kp.size(), ks.size(), miss.size(),
           extra.size());
    int shown = 0;
    for (const FHit& h : plain) {
        if (shown >= 6) break;
        if (!std::binary_search(miss.begin(), miss.end(), key(h))) continue;
        shown++;
        std::string ca, cb;
        rpn(h.i, h.a, ca); rpn(h.c0 - h.i, h.b, cb);
        printf("        missed: size %d pair %s | %s  steps", h.s, ca.c_str(), cb.c_str());
        for (int t = 0; t < h.nf; t++) { std::string cw; rpn(h.st[t].lev, h.st[t].idx, cw); printf(" [%s side %d alt %d]", cw.c_str(), h.st[t].side, h.st[t].alt); }
        printf("\n");
    }
    fflush(stdout);
}

// ---------------------------------------------------------------- the plan of a size

struct MPlan { int F = 0, npart = 1, sbits = 20; double store = 0, fwd = 0, back = 0, gen = 0; double work[64]; };
static MPlan make_plan(int n, int NT, int Fforce, double ntargets)
{
    auto eval = [&](int F, MPlan& pl) {
        std::vector<double> cnt(n + 2, 0.0);
        cnt[n] = 1;
        for (int s = n; s > NT; s--) {
            if (cnt[s] == 0) continue;
            for (int u = 1; u <= std::min(NT, s - NT - 1); u++) cnt[s - u] += 2.0 * (double)ml[u].n * cnt[s];
        }
        auto pairs = [&](int s) { double t = 0; for (int i = std::max(1, s - NT); i <= std::min(NT, s - 1); i++) t += (double)ml[i].n * (double)ml[s - i].n; return t; };
        auto Pb = [&](int s) { double t = 0; for (int u = std::max(1, s - NT); 2 * u <= s; u++) t += (2 * u == s ? 1.0 : 2.0) * (double)ml[u].n; return t; };
        pl.F = F; pl.store = pl.fwd = pl.back = pl.gen = 0;
        for (int s = NT + 1; s <= n; s++) {
            pl.gen += cnt[s];
            if (s <= F) { pl.store += cnt[s]; pl.fwd += pairs(s); }
            else pl.back += cnt[s] * Pb(s);
        }
        for (int s = 0; s < 64; s++) pl.work[s] = 0;
        for (int s = NT + 1; s <= n; s++) {
            double w = 1 + (s <= F ? 0.0 : Pb(s));
            for (int u = 1; u <= std::min(NT, s - NT - 1); u++) w += 2.0 * (double)ml[u].n * pl.work[s - u];
            pl.work[s] = w;
        }
    };
    // the store, and the number of parts it is filled in, for a plan; the cost of the plan in look-ups (a pair costs
    // about as much as a look-up, a request about two; the pairs are run once per part, the requests are made once
    // per part and once more for the terms)
    auto finish = [&](MPlan& pl) -> double {
        pl.store *= ntargets;
        if (M_EXACT && !REAL_ONLY) pl.store += 0.5 * pl.gen * ntargets;      // (the requests exp(exp(X)) = c: at most this)
        pl.sbits = 20;
        while (pl.sbits < M_SBITS && (double)(1ULL << pl.sbits) * 0.45 < pl.store) pl.sbits++;
        pl.npart = std::max(1, (int)std::ceil(pl.store / ((double)(1ULL << pl.sbits) * 0.55)));
        return (pl.npart + (pl.store > 0 ? 1 : 0)) * 2 * pl.gen * ntargets + pl.npart * pl.fwd + pl.back * ntargets;
    };
    MPlan best;
    if (Fforce >= 0) { eval(Fforce > NT ? std::min(Fforce, n) : NT, best); finish(best); }
    else {
        double bt = 1e300;
        for (int F = NT; F <= n; F++) {
            MPlan pl;
            eval(F, pl);
            double tot = finish(pl);
            if (tot < bt) { bt = tot; best = pl; }
        }
    }
    return best;
}

// ---------------------------------------------------------------- one size: all passes, then the verification

struct SizeResult { unsigned long long cands = 0, confirmed = 0, t3rej = 0, t3bad = 0, unres = 0; };

static SizeResult run_size(int n, const std::vector<int>& act, bool verbose)
{
    const int NT = M_NT;
    double t0 = now();
    MPlan pl = make_plan(n, NT, M_ZONE, (double)act.size());
    if (verbose) {
        printf("  n=%d: %zu targets; plan: zone to %d, %d part(s) of 2^%d slots; expected at most: requests %.2e (stored %.2e), pairs %.2e per part, look-ups %.2e\n", n, act.size(), pl.F,
               pl.npart, pl.sbits, pl.gen * (double)act.size(), pl.store, pl.fwd, pl.back * (double)act.size());
        fflush(stdout);
    }
    if (M_PARTS > pl.npart) pl.npart = M_PARTS;
    Run R;
    R.n = n; R.zlo = NT + 1; R.zhi = pl.F; R.npart = pl.npart;
    R.fmax = (M_EXACT && !REAL_ONLY) ? n - 2 : 0;
    if (R.fmax <= NT) R.fmax = 0;
    // (a term run forward has a pair with more than NT leaves below and at most 14 entries put on top of it)
    if (R.fmax - (NT + 1) > 14) { fprintf(stderr, "n=%d: the terms run forward would have more than 14 entries above the pair: not covered\n", n); exit(1); }
    for (int c = 0; c < 64; c++) { e2list[c].clear(); e2big[c] = false; e2all[c].clear(); e2q[c].clear(); }
    std::vector<FHit>().swap(fhits); std::vector<uint32_t>().swap(fhitix);      // (the memory of the size before is given back)
    std::vector<Hit>().swap(hits); std::vector<uint32_t>().swap(hitix);
    T3_WAITING = 0; T3_WAITMAX = 0; T3_NB = 0; T3_NC = 0; T3_NINL = 0; T3_REJ = 0; T3_BAD = 0; T3_US = 0;
    for (auto& v : t3keep_tl) v.clear();
    for (int s = 0; s < 64; s++) R.work[s] = pl.work[s];
    const bool zone = R.zhi >= R.zlo || R.fmax > 0;
    for (auto& s : mstat) s = MStat();
    for (auto& v : cand_tl) v.clear();
    if (zone) {
        uint64_t slots = 1ULL << pl.sbits;
        if (slots != rsslots) {
            aligned_free_portable(rst);
            rst = (Zd*)aligned_alloc(4096, (size_t)slots * sizeof(Zd));
            if (!rst) { fprintf(stderr, "no memory for the request store\n"); exit(1); }
            rsslots = slots; rsmask = slots - 1;
        }
    }
    auto dfs = [&](int mode, int part, bool bprobe) {
        Run Q = R;
        Q.mode = mode; Q.part = part; Q.bprobe = bprobe;
#pragma omp parallel
#pragma omp single
        {
            for (size_t t = 0; t < act.size(); t++) {
#pragma omp task firstprivate(t) shared(Q)
                {
                    const MTarget& T = mt[act[t]];
                    Path* P = new Path();
                    P->len = 0; P->hasA[0] = true; P->ebA[0] = 4 * U_(q(0.0));
                    P->hasPC[0] = 0;
                    P->hasQ[0] = true; P->ebQ[0] = 4 * U_(q(0.0));
                    for (int p = 0; p < NP; p++) P->Aq[0][p] = T.vq[p];
                    P->sig[0] = mix64(0x1234567ULL + (uint64_t)act[t] * 0x9E3779B97F4A7C15ULL + (uint64_t)n);
                    for (int p = 0; p < NP; p++) P->A[0][p] = T.vq[p];
                    double d0[NP];
                    Zd z0[NP];
                    for (int p = 0; p < NP; p++) { d0[p] = 0; z0[p] = Zd{ 0, 0 }; }
                    node(Q, act[t], n, KV, T.v, 4 * U53, d0, z0, *P);
                    delete P;
                }
            }
        }
    };
    double tg = 0, tf = 0, tr = 0;
    std::vector<Hit> allhits;
    std::vector<FHit> allfhits;
    // (address space only, nothing is touched: a vector that grows by doubling holds its content twice while it
    // moves, and the hits of the pairs are 16 GB at 33 leaves)
    if (R.zhi >= R.zlo) allhits.reserve((size_t)std::min(2e9, 8.0 * pl.store + 1024));
    for (int part = 0; part < R.npart; part++) {
        double t1 = now();
        if (zone) {
            uint64_t* w = (uint64_t*)rst;
#pragma omp parallel for schedule(static)
            for (long long k = 0; k < (long long)(2 * rsslots); k++) w[k] = RS_EMPTY;
        }
        dfs(MODE_GEN, part, part == 0);
        double t2 = now();
        if (rs_overflow) { fprintf(stderr, "the request store is full at n=%d (raise --sbits)\n", n); exit(1); }
        if (FW_OVER || allhits.size() > 400000000ULL) { printf("n=%d: too many hits of pairs in the store of requests (%zu): stopped here\n", n, allhits.size()); fflush(stdout); exit(2); }
        for (int c = 0; c < 64; c++) {
            std::vector<E2Q>& v = e2q[c];
            if (v.empty()) continue;
            std::sort(v.begin(), v.end(), [](const E2Q& x, const E2Q& y) {
                if (x.kre != y.kre) return x.kre < y.kre;
                if (x.kim != y.kim) return x.kim < y.kim;
                if (x.e != y.e) return x.e < y.e;
                if ((double)x.k.re != (double)y.k.re) return (double)x.k.re < (double)y.k.re;
                return (double)x.k.im < (double)y.k.im;
            });
            v.erase(std::unique(v.begin(), v.end(), [](const E2Q& x, const E2Q& y) { return x.kre == y.kre && x.kim == y.kim && x.e == y.e && x.k.re == y.k.re && x.k.im == y.k.im; }), v.end());
        }
        if (M_ZQ) zq_build();                            // (the plain requests of this part in quadruple precision)
        if (zone) forward_pass([&]() { Run Q = R; Q.part = part; return Q; }(), allhits, allfhits);
        if (M_ZQ) { std::vector<ZQ>().swap(zq); std::vector<uint32_t>().swap(zqix); zqmask = 0; }
        if (FW_OVER) { printf("n=%d: one thread had more than 5e7 hits of pairs in the store of requests at one pair of levels: stopped here, this size is not complete\n", n); fflush(stdout); exit(2); }
        if (M_F32 && R.fmax > 0) {
            if (R.npart != 1) { fprintf(stderr, "--f32: the requests must be in one part (n=%d)\n", n); exit(1); }
            f32tot = F32Stat();
            Run Q = R;
            Q.part = part;
            if (M_F32CHECK) {
                std::vector<FHit> fh2;
                forward_f32(Q, fh2);
                f32_compare(allfhits, fh2, n);
            } else forward_f32(Q, allfhits);
        }
        double t3 = now();
        tg += t2 - t1; tf += t3 - t2;
    }
    if (!allfhits.empty()) {
        std::sort(allfhits.begin(), allfhits.end(), [](const FHit& x, const FHit& y) {
            if (x.s != y.s) return x.s < y.s;
            if (x.kre != y.kre) return x.kre < y.kre;
            return x.kim < y.kim;
        });
        fhits.swap(allfhits);
        int hb = 4;
        while ((1ULL << hb) < 2 * fhits.size() + 16) hb++;
        fhitmask = (1ULL << hb) - 1;
        fhitix.assign(1ULL << hb, 0);
        for (size_t k = 0; k < fhits.size(); k++) {
            if (k && fhits[k].s == fhits[k - 1].s && fhits[k].kre == fhits[k - 1].kre && fhits[k].kim == fhits[k - 1].kim) continue;
            uint64_t pos = hitkey(fhits[k].s, fhits[k].kre, fhits[k].kim) & fhitmask;
            while (fhitix[pos]) pos = (pos + 1) & fhitmask;
            fhitix[pos] = (uint32_t)(k + 1);
        }
    }
    if (!allhits.empty() || !fhits.empty()) {
        double t1 = now();
        std::sort(allhits.begin(), allhits.end(), [](const Hit& x, const Hit& y) {
            if (x.s != y.s) return x.s < y.s;
            if (x.kre != y.kre) return x.kre < y.kre;
            if (x.kim != y.kim) return x.kim < y.kim;
            if (x.i != y.i) return x.i < y.i;
            if (x.a != y.a) return x.a < y.a;
            return x.b < y.b;
        });
        hits.swap(allhits);
        int hb = 4;
        while ((1ULL << hb) < 2 * hits.size() + 16) hb++;
        hitmask = (1ULL << hb) - 1;
        hitix.assign(1ULL << hb, 0);
        for (size_t k = 0; k < hits.size(); k++) {
            if (k && hits[k].s == hits[k - 1].s && hits[k].kre == hits[k - 1].kre && hits[k].kim == hits[k - 1].kim) continue;
            uint64_t pos = hitkey(hits[k].s, hits[k].kre, hits[k].kim) & hitmask;
            while (hitix[pos]) pos = (pos + 1) & hitmask;
            hitix[pos] = (uint32_t)(k + 1);
        }
        dfs(MODE_REGEN, 0, false);
        tr = now() - t1;
    }
    MStat tot;
    for (auto& s : mstat) tot.add(s);
    // tier 3 for what quadruple precision let through: most of it was done in batches during the second run
    // (tier3_batch); here the rest of every thread, and then the report for all that was not rejected
    std::vector<MCand> cands;
    for (auto& v : cand_tl) { cands.insert(cands.end(), v.begin(), v.end()); std::vector<MCand>().swap(v); }
    SizeResult res;
    if (T3_WAITING.load() != 0) { fprintf(stderr, "tier 3: batches still waiting after the second run (a fault of the program)\n"); exit(1); }
    res.t3rej = T3_REJ.load(); res.t3bad = T3_BAD.load();
    res.cands = cands.size() + T3_NC.load();
    std::vector<double> lgs(cands.size());
    std::vector<int> sts(cands.size());
    std::vector<uint8_t> prs(cands.size());
    double t4 = now();
#pragma omp parallel for schedule(dynamic, 1)
    for (long long k = 0; k < (long long)cands.size(); k++) {
        int status;
        bool pres;
        lgs[k] = cand_distance(mt[cands[k].tgt], cands[k], status, pres);
        sts[k] = status; prs[k] = pres;
    }
    const size_t nrest = cands.size();
    for (auto& v : t3keep_tl) {
        for (const T3Out& o : v) { cands.push_back(o.c); lgs.push_back(o.lg); sts.push_back(o.st); prs.push_back(o.pres); }
        std::vector<T3Out>().swap(v);
    }
    for (size_t k = 0; k < cands.size(); k++) {
        MTarget& T = mt[cands[k].tgt];
        if (sts[k] == T_BAD) { res.t3bad++; continue; }
        bool tok = (sts[k] == T_OK || sts[k] == T_DEEP);
        bool good = tok && (lgs[k] < -1e8 || lgs[k] < -(double)PMAX * 0.30103 + 60);
        if (!tok) {
            res.unres++; T.nunres++;
            static int dbg = 0;
            if (dbg < 8 && getenv("MITM_DBG")) {
                dbg++;
                std::string code, tc;
                cand_code(cands[k], 0, code);
                if (T.ti < 0) rpn(T.slev, T.sidx, tc);
                printf("      DEBUG unresolved: target %s (%s) n=%d status %d bot %d pk %d len %d code %s\n", T.name.c_str(), tc.c_str(), n, sts[k], cands[k].bot, cands[k].pk, cands[k].len, code.c_str());
            }
        }
        else if (!good) {
            res.t3rej++;
            static int dbg2 = 0;
            if (dbg2 < 400 && (k % 37) == 0 && getenv("MITM_DBG")) {
                dbg2++;
                std::string code;
                cand_code(cands[k], 0, code);
                printf("      DEBUG rejected: target %s n=%d lg=%.0f bot %d pk %d len %d hole L%d#%u S=%d steps", T.name.c_str(), n, lgs[k], cands[k].bot, cands[k].pk, cands[k].len, cands[k].blev, cands[k].bidx, (lev[cands[k].blev].fl[cands[k].bidx] & F_S) ? 1 : 0);
                for (int d = 0; d < cands[k].len; d++) printf(" [L%d#%u s%d %s (%.6g,%.6g)]", cands[k].st[d].lev, cands[k].st[d].idx, cands[k].st[d].side, (lev[cands[k].st[d].lev].fl[cands[k].st[d].idx] & F_S) ? "S" : "c", cands[k].st[d].val[0].re, cands[k].st[d].val[0].im);
                printf(" hole value (%.17g, %.17g)\n", (double)lev[cands[k].blev].v[cands[k].bidx].z[0].re, (double)(PIq * lev[cands[k].blev].v[cands[k].bidx].z[0].y));
                SQ_DEBUG = true; int rq = stage_q(T, cands[k]); SQ_DEBUG = false;
                printf("        SQ result %d\n", rq);
            }
        }
        if (!(good || !tok)) continue;
        if (good) { res.confirmed++; T.nforms++; if (!T.found_n) T.found_n = n; }
        if ((int)T.forms.size() < M_SHOW || !tok) {
            MForm f;
            cand_code(cands[k], 0, f.code);
            f.lg = lgs[k]; f.pres = prs[k]; f.status = good ? (sts[k] == T_DEEP ? 2 : 1) : 0;
            bool dup = false;
            for (auto& g : T.forms) if (g.code == f.code) dup = true;
            if (!dup) T.forms.push_back(f);
        }
    }
    double t5 = now();
    if (verbose) {
        if (getenv("MITM_HIST")) for (int b = 0; b < 2; b++) for (int rk = 0; rk < 4; rk++) for (int ek = 0; ek < 4; ek++) {
            unsigned long long tt = 0;
            for (int r = 0; r < 8; r++) tt += SQ_HIST[b][rk][ek][r];
            if (!tt) continue;
            printf("      [quad stage: %s request-kind %d entry-kind %d: digits differ %llu, displacement differs %llu, passed %llu, could not tell %llu]\n", b ? "pair " : "entry", rk, ek,
                   SQ_HIST[b][rk][ek][0].load(), SQ_HIST[b][rk][ek][2].load(), SQ_HIST[b][rk][ek][1].load(), SQ_HIST[b][rk][ek][3].load() + SQ_HIST[b][rk][ek][4].load());
            for (int r = 0; r < 8; r++) SQ_HIST[b][rk][ek][r] = 0;
        }
        if (getenv("MITM_HIST")) printf("      [requests below the resolution: %llu of %llu; their look-ups %llu of %llu; their double hits %llu of %llu; passed quad because it could not tell: %llu, no displacement: %llu; quick check: rejected %llu passed %llu (no quad request %llu, displacement %llu); store hits rejected in double %llu; classes by PMAX %llu]\n", tot.slreq, tot.req, tot.slp, tot.probes, tot.slh, tot.dhits, tot.qp3, tot.qp4, tot.prej, tot.ppass, tot.qk_noq, tot.qk_dx, RS_PRE.load(), tot.pcls);
        RS_PRE = 0;
        if (R.fmax > 0) {
            printf("      [recomputed because: key %llu, error of v %llu, exp %llu, exp(exp) %llu; both sides of the cut %llu; tier 3 because: pair cancels %llu, chain cancels %llu]\n",
                   FX_WHY[0].load(), FX_WHY[1].load(), FX_WHY[3].load(), FX_WHY[7].load(), FX_WHY[2].load(), FX_WHY[4].load(), FX_WHY[5].load());
            printf("      [key recomputed: inner/exp given/small error %llu, large %llu; inner/exp not given %llu, %llu; top/given %llu, %llu; top/not given %llu, %llu]\n", FX_KEY[0].load(),
                   FX_KEY[1].load(), FX_KEY[2].load(), FX_KEY[3].load(), FX_KEY[4].load(), FX_KEY[5].load(), FX_KEY[6].load(), FX_KEY[7].load());
            for (int w = 0; w < 8; w++) { FX_WHY[w] = 0; FX_KEY[w] = 0; }
        }
        printf("      [nodes not followed (their value is a value of a smaller level): %llu; exp requests so marked: %llu; decided at %ld bits: %llu; pairs with such a value dropped: %llu]\n",
               tot.pruned, tot.notnew, (long)PMAX, tot.pr_t3, tot.fw_drop);
#if defined(F32_GPU)
        if (R.fmax > 0 && M_F32 == 2) {
            double tk = 0, tc = 0;
            f32gpu_times(&tk, &tc);
            printf("      [graphics card: %llu launches (%llu again in halves), %llu units in second launches (not rebuilt: %llu), %.1f s on the card (kernel %.1f s, records copied %.1f s; waited for the "
                   "processor %.1f s), %.1f s for the records on the processor]\n", f32_launches, f32_retries, f32_units, f32_bad, f32_tgpu, tk, tc, f32_twait, f32_tcpu);
            printf("      [processor time of the records by kind (of the clock ticks counted): terms %.0f %%, parts %.0f %%, Log side %.0f %%, exp side %.0f %%; ticks per record: %.0f, %.0f, %.0f, %.0f]\n",
                   100.0 * f32_ticks[0] / (double)(f32_ticks[0] + f32_ticks[1] + f32_ticks[2] + f32_ticks[3] + 1), 100.0 * f32_ticks[1] / (double)(f32_ticks[0] + f32_ticks[1] + f32_ticks[2] + f32_ticks[3] + 1),
                   100.0 * f32_ticks[2] / (double)(f32_ticks[0] + f32_ticks[1] + f32_ticks[2] + f32_ticks[3] + 1), 100.0 * f32_ticks[3] / (double)(f32_ticks[0] + f32_ticks[1] + f32_ticks[2] + f32_ticks[3] + 1),
                   f32_ticks[0] / (double)(f32tot.nrec[0] + 1), f32_ticks[1] / (double)(f32tot.nrec[1] + 1), f32_ticks[2] / (double)(f32tot.nrec[2] + 1), f32_ticks[3] / (double)(f32tot.nrec[3] + 1));
            for (int t = 0; t < 4; t++) f32_ticks[t] = 0;
            if (f32_bad) { fprintf(stderr, "the sieve: %llu units could not be rebuilt in the second phase (a fault of the program)\n", f32_bad); exit(1); }
            f32_launches = f32_retries = f32_units = 0; f32_tgpu = f32_tcpu = f32_twait = 0;
        }
#endif
        if (R.fmax > 0 && M_F32)
            printf("      [sieve in single precision: %.3e nodes (%.3e of the top size); reported: terms %llu, parts of the tree %llu + %llu (Log side) + %llu (exp side); "
                   "run in double precision for them: %llu, %llu, %llu, %llu nodes]\n", (double)f32tot.nodes + (double)f32tot.tops, (double)f32tot.tops, f32tot.nrec[0], f32tot.nrec[1], f32tot.nrec[2],
                   f32tot.nrec[3], f32tot.cpu[0], f32tot.cpu[1], f32tot.cpu[2], f32tot.cpu[3]);
        if (R.fmax > 0 && M_F32)
            printf("      [handed back because: Log of a number with a large relative error %llu; exp of a number with a large error %llu, with a moderate one %llu; value beyond the range %llu; "
                   "exp(w) beyond the range %llu; exp(node) beyond the range %llu, and imaginary or on a cut %llu; pair beyond the range %llu. Kept by one function in double precision: Log %llu, exp %llu]\n",
                   f32tot.why[0], f32tot.why[1], f32tot.why[2], f32tot.why[3], f32tot.why[4], f32tot.why[5], f32tot.why[6], f32tot.why[7], f32tot.why2[0], f32tot.why2[1]);
        if (R.fmax > 0 && M_F32) {
            printf("      [terms reported because: own exp near a request %llu, not known %llu, huge and near imaginary %llu; top size: product near a request %llu, product beyond the range %llu, "
                   "phase not known %llu, huge and near imaginary %llu, value near a request %llu]\n      [requests exp(exp(X)) = c by the size of X:", f32tot.fwhy[0], f32tot.fwhy[1], f32tot.fwhy[2], f32tot.fwhy[3],
                   f32tot.fwhy[4], f32tot.fwhy[5], f32tot.fwhy[6], f32tot.fwhy[7]);
            for (int c = M_NT + 1; c <= R.fmax; c++) printf(" %d:%d", c, f32h.g.rqn[c]);
            printf("]\n");
        }
        if (T3_NB.load())
            printf("      [tier 3 during the second run: %llu candidates in %llu batches (%llu of them done by the thread that made them; at most %ld waiting), %.1f thread-seconds; after it: %zu candidates]\n",
                   T3_NC.load(), T3_NB.load(), T3_NINL.load(), T3_WAITMAX.load(), (double)T3_US.load() * 1e-6, nrest);
        if (R.fmax > 0)
            printf("      [a term that met a request in double precision, compared in quadruple precision at once: not the same number %llu, kept %llu, could not be compared %llu]\n",
                   E2_QREJ.load(), E2_QKEEP.load(), E2_QNO.load());
        E2_QREJ = 0; E2_QKEEP = 0; E2_QNO = 0;
        if (ZQ_REJ.load() || ZQ_NOQ.load())
            printf("      [pairs that met a plain request in double precision, compared with it in quadruple precision at once: not the same number %llu, recorded %llu; "
                   "requests not known in quadruple precision %llu]\n", ZQ_REJ.load(), ZQ_KEEP.load(), ZQ_NOQ.load());
        ZQ_REJ = 0; ZQ_KEEP = 0; ZQ_NOQ = 0;
        if (R.fmax > 0)
            printf("      [every branch: requests exp(exp(X)) = c stored %llu; terms run forward up to %d leaves: %.3e (recomputed %llu, in tier 3 %llu, not terms %llu), met a request %llu -> %llu]\n",
                   tot.e2req, R.fmax, (double)tot.fx_nodes, tot.fx_acc, tot.fx_t3, tot.fx_dead, tot.fx_hits, tot.e2hits);
        printf("%3d %3d  zone %d..%d parts %d | requests %.3e stored %.3e special %llu | look-ups %.3e cell hits %llu double hits %llu | pairs %.3e (quad %llu, tier3 %llu, tiny %llu, failed %llu) store hits %llu -> %llu"
               " | recomputed ld %llu q %llu mp %llu, dropped there %llu, presumed zero %llu, unresolved %llu, cut %llu, exact marks: dropped %llu kept %llu | displacement: rejected %llu + %llu | quad: rejected %llu passed %llu | tier 3: rejected %llu not a term %llu unresolved %llu CONFIRMED %llu"
               " | %.1f s (dfs %.1f, pairs %.1f, 2nd dfs %.1f, tier 3 %.1f) %.1f GB\n",
               n, 2 * n - 1, R.zlo, R.zhi, R.npart, (double)tot.req, (double)tot.stored, tot.special, (double)tot.probes, tot.taghits, tot.dhits, (double)tot.fw_pairs,
               tot.fw_quad, tot.fw_t3, tot.fw_tiny, tot.fw_fail, tot.fw_hits, tot.zhits, tot.ref_ld, tot.ref_q, tot.ref_mp, tot.dead_acc, tot.pzero, tot.unres, tot.cuts,
               tot.xdrop, tot.xkeep, tot.drej, tot.qdrej, tot.qrej, tot.qpass, res.t3rej, res.t3bad, res.unres, res.confirmed, t5 - t0, tg, tf, tr, t5 - t4, mem_gb());
        fflush(stdout);
    }
    res.unres += tot.unres;
    return res;
}

static void mt_from_value(MTarget& T, const Val& t, uint8_t cls)
{
    T.t = t; T.cls = cls;
    for (int p = 0; p < NP; p++) {
        T.vq[p].re = t.z[p].re; T.vq[p].im = PIq * t.z[p].y;
        T.v[p] = Zd{ (double)T.vq[p].re, (double)T.vq[p].im };
    }
}

// ---------------------------------------------------------------- the two drivers

// The targets that the table (to NT leaves) does not have, searched at NT+1 .. nmax leaves.
static void mitm_targets(int nmax)
{
    mt.clear();
    for (size_t k = 0; k < targets.size(); k++) {
        const Target& G = targets[k];
        if (G.n_clean && G.n_clean <= M_NT) continue;
        bool real = true;
        for (int p = 0; p < NP; p++) if (!(((G.cls >> (2 * p)) & 2) && G.t.z[p].y == 0)) real = false;
        if (REAL_ONLY && !real) continue;
        if (!M_SKIP.empty() && ("," + M_SKIP + ",").find("," + G.name + ",") != std::string::npos) { printf("target %s: not searched (--mitmskip)\n", G.name.c_str()); continue; }
        MTarget T;
        T.name = G.name; T.ti = (int)k;
        mt_from_value(T, G.t, G.cls);
        mt.push_back(T);
    }
    printf("\nSearch above the table: terms with %d..%d leaves (K = %d..%d), table depth %d, %s. One line per size.\n", std::max(M_NT + 1, M_NMIN), nmax,
           2 * std::max(M_NT + 1, M_NMIN) - 1, 2 * nmax - 1, M_NT,
           REAL_ONLY ? "real numbers only (no branch is chosen)" : (M_EXACT ? "complex, every branch (terms with up to n-2 leaves run forward)" : "complex, principal branch at the one place where a branch is chosen"));
    unsigned long long unres = 0;
    for (int n = std::max(M_NT + 1, M_NMIN); n <= nmax; n++) {
        if (n > 2 * M_NT + 1) { printf("n=%d is above 2*%d+1: not covered by this program\n", n, M_NT); break; }
        std::vector<int> act;
        for (size_t k = 0; k < mt.size(); k++) if (!mt[k].found_n) act.push_back((int)k);
        if (act.empty()) break;
        SizeResult r = run_size(n, act, true);
        unres += r.unres;
        for (int k : act) {
            MTarget& T = mt[k];
            if (T.found_n == n) {
                printf("    FOUND %-9s leaves=%2d K=%2d  (%llu terms)\n", T.name.c_str(), n, 2 * n - 1, T.nforms);
                for (auto& f : T.forms) {
                    char ag[48];
                    snprintf(ag, sizeof ag, f.lg < -1e8 ? "0" : "1e%.0f", f.lg);
                    printf("        agree=%-8s %s%s%s\n", ag, f.pres ? "presumed " : "", f.status == 2 ? "(structure below the resolution) " : "", f.code.c_str());
                }
            } else if (T.nunres) {
                printf("    %-9s: %llu candidates that tier 3 could not evaluate\n", T.name.c_str(), T.nunres);
                for (auto& f : T.forms) printf("        %s\n", f.code.c_str());
            }
        }
        fflush(stdout);
    }
    printf("\nResult of the search above the table (no term means: none with up to %d leaves in the class searched; requests left unresolved: %llu)\n", nmax, unres);
    for (auto& T : mt) {
        if (T.found_n) printf("  %-9s leaves=%2d K=%2d  %s\n", T.name.c_str(), T.found_n, 2 * T.found_n - 1, T.forms.empty() ? "" : T.forms[0].code.c_str());
        else printf("  %-9s no term: K > %d\n", T.name.c_str(), 2 * nmax - 1);
    }
}

// The term on record for the entry (n, idx): the largest |k| over the nodes X2 with more than NT leaves that are
// the left child of a left child (there the search takes exp(X2) = Log(..) + 2 pi i k with |k| <= KWIN).
static long term_kneed(int n, uint32_t idx, bool isleft)
{
    if (n <= M_NT) return 0;
    const Level& L = lev[n];
    int i = L.pi[idx], j = n - i;
    long k = 0;
    if (isleft && i > M_NT) {
        for (int p = 0; p < NP; p++) {
            q y = lev[i].E[L.pa[idx]].z[p].y;          // exp(X2) = re + i*pi*y
            long kk = std::labs((long)llrintq(nearbyintq(y / 2)));
            q r = y - 2 * nearbyintq(y / 2);
            if (r == -1) kk = std::max(0L, kk - 1);    // (on the cut either multiple is taken)
            k = std::max(k, kk);
        }
    }
    k = std::max(k, term_kneed(i, L.pa[idx], true));
    k = std::max(k, term_kneed(j, L.pb[idx], false));
    return k;
}

// Etalon: values of the table at the levels NT+1..N are searched with a table cut at NT. Each must be found
// at its own level and at no smaller one.
static void mitm_etalon(int N, int per_level)
{
    mt.clear();
    std::vector<int> lvl;
    for (int m = M_NT + 1; m <= N && m <= 2 * M_NT + 1; m++) {
        const Level& L = lev[m];
        std::vector<uint32_t> clean;
        for (size_t k = 0; k < L.size(); k++) if (!(L.fl[k] & F_S) && !(m == MINF_LEV && (uint32_t)k == MINF_IDX)) clean.push_back((uint32_t)k);
        size_t take = std::min((size_t)per_level, clean.size());
        uint64_t x = 0x9E3779B97F4A7C15ULL * (uint64_t)(m + 1);
        for (size_t t = 0; t < take; t++) {
            size_t pick;
            if (take == clean.size()) pick = t;
            else { x = mix64(x + t); pick = (size_t)(x % clean.size()); }
            MTarget T;
            char nm[64];
            snprintf(nm, sizeof nm, "L%d#%u", m, clean[pick]);
            T.name = nm; T.slev = m; T.sidx = clean[pick];
            bool zero = true;
            for (int p = 0; p < NP; p++) if (!(((L.fl[T.sidx] >> (2 * p)) & 3) == 3 && L.v[T.sidx].z[p].y == 0)) zero = false;
            if (zero) continue;
            mt_from_value(T, L.v[T.sidx], L.fl[T.sidx] & CLSMASK);
            mt.push_back(T);
            lvl.push_back(m);
        }
    }
    printf("\nEtalon: %zu values of the table at the levels %d..%d, searched with the table cut at %d leaves. One line per size.\n", mt.size(), M_NT + 1,
           std::min(N, 2 * M_NT + 1), M_NT);
    unsigned long long unres = 0;
    for (int n = M_NT + 1; n <= N && n <= 2 * M_NT + 1; n++) {
        std::vector<int> act;
        for (size_t k = 0; k < mt.size(); k++) if (!mt[k].found_n && lvl[k] >= n) act.push_back((int)k);
        if (act.empty()) continue;
        SizeResult r = run_size(n, act, true);
        unres += r.unres;
    }
    printf("\nEtalon result (requests left unresolved: %llu)\n level  targets  found at its level  found EARLIER  NOT found\n", unres);
    unsigned long long bad = 0;
    for (int m = M_NT + 1; m <= N && m <= 2 * M_NT + 1; m++) {
        unsigned long long nt = 0, ok = 0, early = 0, miss = 0;
        for (size_t k = 0; k < mt.size(); k++) if (lvl[k] == m) { nt++; if (mt[k].found_n == m) ok++; else if (mt[k].found_n) early++; else miss++; }
        printf("  %3d  %7llu  %7llu             %7llu        %7llu\n", m, nt, ok, early, miss);
        bad += early + miss;
    }
    // A value not found: is the term on record outside the class searched (it needs |k| > KWIN somewhere)?
    unsigned long long expl = 0, unexpl = 0, early = 0, found_out = 0;
    int shown = 0;
    for (size_t k = 0; k < mt.size(); k++) {
        long kn = term_kneed(mt[k].slev, mt[k].sidx, false);
        if (mt[k].found_n == lvl[k]) { if (kn > M_KWIN) found_out++; continue; }
        bool isearly = mt[k].found_n != 0;
        if (isearly) early++; else if (kn > M_KWIN) expl++; else unexpl++;
        if ((isearly || kn <= M_KWIN) && shown < 40) {
            std::string code;
            rpn(mt[k].slev, mt[k].sidx, code);
            printf("    %s  level %d  %s  (%.17g, %.17g)  table term %s (needs |k| = %ld)%s%s\n", isearly ? "EARLIER" : "MISSED, TERM IN CLASS", lvl[k], mt[k].name.c_str(),
                   mt[k].v[0].re, mt[k].v[0].im, code.c_str(), kn, isearly ? "  found: " : "", isearly && !mt[k].forms.empty() ? mt[k].forms[0].code.c_str() : "");
            shown++;
        }
    }
    printf("not found: %llu; of these the term on record is outside the class (a branch |k| > %d is needed): %llu, inside the class: %llu.  found earlier: %llu\n",
           expl + unexpl, M_KWIN, expl, unexpl, early);
    printf("(found although the term on record is outside the class, through another term: %llu)\n", found_out);
    printf("etalon: %s\n", (unexpl || early) ? "DISAGREEMENTS (see above)" : (expl ? "agrees with the table inside the class searched" : "every value found at its own level and at no smaller one"));
}
