// eml_mitm.cpp - the search for the shortest EML terms of given constants and functions.
// It has two parts. The table of the distinct values of EML terms by number of leaves, with a guard against
// impostors and with three precision tiers, is built in this file. The search above the table (meet in the
// middle) is in mitm.inc, which this file includes. f32kern.h and f32gpu.cu are a sieve in single precision
// for the search, on the processor and on a graphics card.
// Two words of the output: "Step 0" is the table. An "etalon" (--etalon) is the test of the search against
// the table: values that the table has above a level NT are searched with the table cut at NT, and each must
// be found at its own level and at no smaller one.
//
// A term with n leaves has K = 2n-1 symbols in reverse Polish notation. Level n holds every value that a
// term with n leaves takes and that no term with fewer leaves takes, with one term for it. A level is made
// from the smaller ones: all exp(a) - Log(b), a from level i, b from level n-i.
// Semantics: eml(a,b) = exp(a) - Log(b), principal Log, b != 0 (Section 2 of the paper); --real: real
// numbers only, Log of a positive number only. Values outside [1e-290, 1e290] are dropped.
//
// HOW A VALUE IS HELD. A complex number is the pair (re, y) and means re + i*pi*y: the imaginary part is
// kept in units of pi. Then Log of a negative number has y = 1 exactly, and exp of a number whose y is a
// multiple of 1/2 is exactly real or exactly imaginary, with no rounding. Each stored value has two bits per
// probe point: "re is exactly 0" and "y is exactly a multiple of 1/2". They are set by these rules, or by
// tier 3 (below); they are never set because a number merely came out small.
//
// TIER 2 (all pairs): quadruple precision (__float128, 113 bits). With every value goes a bound on its
// relative error, in units of 2^-113, carried through exp, Log and the subtraction. Two values that agree
// within their error bounds may be the same value (see DUPLICATES). A candidate goes to tier 3 when tier 2
// cannot decide: its error bound is above PEND units (cancellation), it is zero within its error bound, or
// one of its parts is zero (or y is a multiple of 1/2) within the error bound without the rules above
// saying so.
//
// TIER 3 (what tier 2 cannot decide): the term is evaluated again from its leaves with MPFR, at P1 bits
// (448) when only accuracy was missing and at PMAX bits (2304) when a decision is needed. A part that is
// still zero within the error bound at PMAX bits is taken as exactly zero ("presumed"; counted in the
// output). That is evidence, not a proof.
//
// THE GUARD (impostors). A term such as exp(-261) - Log(b) differs from -Log(b) by 4e-114: no fixed
// precision sees it. With every value goes the size s of its smallest "ingredient": the smaller operand of
// a subtraction, the argument of exp when it is below 1 (exp(a) = 1 + a + ...), and the ingredients of the
// operands, carried through exp and Log to first order. A term whose smallest ingredient is below 2^-FS of
// its value is marked S ("sub-resolution structure": its digits are not its value). The rules for S:
//  - An S entry and a clean entry are never the same value. A clean term with the same digits is stored
//    next to the S entry, and only a clean entry is reported as a form of a target; an S entry with the
//    digits of a target is reported as an impostor, with its distance from the target at PMAX bits.
//  - S twins. An S entry is held as its digits (the term without what is below the resolution) and a
//    "displacement" d (what is below, to first order, in double precision): exp(-261) - Log(b) is held as
//    (-Log(b), 4e-114). Two S entries are the same value only if digits and displacement agree, so that
//    exp(B) - 1 and exp(B) - e stay two entries. The displacement goes through exp (d*exp) and Log (d/b).
//  - S entries are operands like any other. A candidate that is not S but has an S operand (a
//    cancellation has brought the structure back into view, e.g. A - Log(exp(B) - 1) with A = Log(exp(B)))
//    is computed in tier 3 from the leaves, never from the digits of the S entry.
//  - Such a value "come back", one that is below the resolution of the two operands it is the difference
//    of, carries the square of itself from exp and Log of the operands (Log(1 - t) = -t - t*t/2 - ...),
//    below its own resolution: it is S again.
//  - The digits of an S candidate that lost accuracy in a cancellation are taken from tier 3.
//  - --drop-s leaves S terms out instead (nothing is then built on them).
//
// DUPLICATES. A clean candidate with the digits of a stored clean value (they agree within the error
// bounds of tier 2) is dropped as a duplicate only after tier 3 has found the two equal at PMAX bits.
// exp(t) and 1 - Log(1 - t) agree up to t*t/2 and differ by t*t*t/6; for t = 2.8e-12 that is 4e-36, below
// quadruple precision. Such a candidate is stored as a value of its own. --noconfirm skips this.
//
// build:  g++ -O3 -march=native -std=gnu++17 -fopenmp -fno-math-errno -DNP=1 eml_mitm.cpp -o eml_mitm1 -lmpfr -lgmp -lquadmath
//         (-DNP=2 for functions: two probe points. Needs MPFR with the conversions to and from __float128, and GMP.)
//   with the sieve on a graphics card (CUDA):
//         nvcc -O3 -std=c++17 -maxrregcount=128 -I. -c f32gpu.cu -o f32gpu.o
//         g++ (as above) -DF32_GPU eml_mitm.cpp f32gpu.o -o eml_mitm1_gpu -lmpfr -lgmp -lquadmath -lcudart -lpthread
//   Written for GCC on x86-64 Linux: sincos is taken from glibc, __rdtsc from <x86intrin.h> (a time statistic
//   of the card build), and the memory figures from /proc/self/statm (they are 0 elsewhere). The unit roundoff
//   of long double is taken from the compiler, so the error bounds are right where long double has 53 bits.
// options of the table: --maxn N (leaves), --vars 0|1|2 (1 and 2 need NP=2), --real, --inf (with --real: the
//   extended reals; without: the table over C with the two infinities), --only a,b,c, --hbits B, --threads T,
//   --dump FILE (every stored value), --audit K (every K-th new value is recomputed in tier 3 and compared),
//   --drop-s, --noconfirm, --pend x, --fs bits, --p1 bits, --pmax bits, --band x, --memlimit GB
// options of the search (mitm.inc): --mitm N (terms with up to N leaves), --nt T (table depth used; default:
//   all of it), --nmin n (first size searched), --mitmskip a,b,c (targets left out), --show k (terms printed
//   for a target), --etalon K (the test above, K values of a level), --clean (no term with structure below
//   the resolution), --exact (every branch; constants only), --kwin k (branches |k| <= k where one is chosen),
//   --noprune (a node whose value is a value of a smaller level is followed too),
//   --f32 1|2 (with --exact --clean: the sieve in single precision, on the processor or on the card),
//   --f32check (both ways, compared), --f32rel bits (largest relative error in the sieve, 2^-bits),
//   --zone F (the requests for nodes with up to F leaves go to the store; default: the cheapest plan),
//   --parts k (the store is filled in at least k parts), --sbits b (at most 2^b slots of the store),
//   --tau bits (largest relative error bound of a request in double precision, 2^-bits), --nozq (a pair that
//   meets a request is compared with it in quadruple precision in the second run only), --t3batch n
//   (candidates of a batch for tier 3), --inf-search (the search over C with the two infinities;
//   experimental), and for the card --f32work x (nodes of a launch), --f32cap b, --f32ucap b (2^b records,
//   units of a launch), --f32r0 r (leaves left above a unit of a second launch).
// Diagnostic output, off by default, is switched on by the environment variables MITM_DBG, MITM_DBG2,
//   MITM_HIST, FN_DBG, F32_DBG, F32_HIST, F32_LAUNCH; F32_FLAGS, F32_BD, F32_SMEM are experiments with the sieve.

#define MPFR_WANT_FLOAT128
#include <algorithm>
#include <atomic>
#include <chrono>
#include <climits>
#include <complex>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <memory>
#include <string>
#include <vector>
#include <quadmath.h>
#include <mpfr.h>
#include <omp.h>

#ifndef NP
#define NP 1
#endif

typedef __float128 q;
struct C { q re, y; };      // re + i*pi*y
struct Val { C z[NP]; };
struct D { double re, y; }; // the displacement of an S value (see "S twins"), re + i*pi*y like a value
struct DVal { D z[NP]; };

static const q PIq = M_PIq;
static const double PI_D = 3.14159265358979323846;
static double ULP;                       // 2^-113
static q TWOQB;
static const int QB = 60;                // bits of the cell a value is filed under (the comparison is exact)
static double PEND = 4096.0;             // error bound (units of 2^-113) above which a candidate goes to tier 3
static const double THR = 1024.0;        // a part below THR error bounds cannot be told from zero
static const double TOLF = 4.0;          // two values are the same if they differ by less than TOLF error bounds
static int FS = 94;                      // smallest ingredient below 2^-FS of the value: S
static mpfr_prec_t P1 = 448, PMAX = 2304;
static double BAND = 9.5367431640625e-07; // 2^-20 of a cell: closer to a cell border, the next cell is searched too
static bool REAL_ONLY = false, DROP_S = false, CONFIRM = true;
static const double RMIN = 1e-290, RMAX = 1e290;
// Rounding errors, in units of 2^-113 of the value (a correctly rounded quadruple number is off by up to 1 unit):
static const double CEXP = 12.0, CLOG = 6.0, CABS = 4.0;   // exp; Log (relative part); Log (absolute part)
static const double CSUB = 2.0, CT3 = 2.0;                 // the subtraction; a value that comes from tier 3

static const int SNONE = 32767, XNONE = 32767;
static const uint8_t F_S = 16, F_T3 = 32, F_PRES = 64;   // flags of a stored value (bits 0-3: exactness per probe)
static const uint8_t F_OK = 128, F_SOP = 64, F_DE = 32;   // flags of E and L (bits 0-3: exactness per probe; F_SOP: the value is S;
                                                          // F_DE: E has a displacement)
static const uint32_t NOD = 0xFFFFFFFFu;                  // no displacement
// --inf: the extended reals, with --real. Values -inf and +inf, ln 0 = -inf, ln(+inf) = +inf,
// exp(-inf) = 0, exp(+inf) = +inf, inf - inf undefined. A finite value then has exactly one way to arise that
// the reals do not have: eml(-inf, b) = -ln b for a finite b > 0. The shortest term of +inf is eml(1, 0) (5
// leaves; 0 = 111E1EE), of -inf eml(1, +inf) = 11111E1EEEE (6 leaves). So --inf adds ONE entry to level 6: it
// can only be a left operand, and its exp is exactly 0 (flag F_EZERO of fE). It is not a value: it is not in
// the index, has no Log, and is never a target or a duplicate.
static bool INF_MODE = false;
static bool INF_CSEARCH = false;                          // --inf-search: the search above the table over C_inf
static const int MINF_LEV = 6;
static uint32_t MINF_IDX = NOD;                           // index of the entry -inf in level 6 (NOD: none)
static const uint8_t F_EZERO = 16;                        // flag of fE: E is exactly 0 (the entry is -inf)
static const char* MINF_CODE = "11111E1EEEE";
static const uint8_t CLSMASK = (uint8_t)((1 << (2 * NP)) - 1);

static inline double nrm(const C& z) { return std::max(std::fabs((double)z.re), PI_D * std::fabs((double)z.y)); }
static inline double modulus2(double a, double b)
{
    a = std::fabs(a); b = std::fabs(b);
    double m = std::max(a, b);
    if (m == 0.0) return 0.0;
    double r = std::min(a, b) / m;
    return m * std::sqrt(1.0 + r * r);
}
static inline double modulus(const C& z) { return modulus2((double)z.re, PI_D * (double)z.y); }
static inline int clamp16(int s) { return s < -30000 ? -30000 : (s > 30000 ? 30000 : s); }

struct Level {
    std::vector<Val> v, E, L;
    std::vector<uint8_t> fl, fE, fL;
    std::vector<float> rV;                 // [k*NP+p] error bound of v, units of 2^-113 of |v|
    std::vector<int16_t> sV;               // [k*NP+p] log2 of the smallest ingredient of v; SNONE: none
    std::vector<double> wE, wL;            // [k*NP+p] absolute error bound of E, L, units of 2^-113
    std::vector<int16_t> xE, xL, sE, sL;   // [k*NP+p] log2 of the size of E, L; their smallest ingredients
    std::vector<uint32_t> pa, pb;
    std::vector<uint8_t> pi;
    std::vector<uint32_t> dx;              // [k] index of the displacements of entry k in dV, dE, dL; NOD: none
    std::vector<DVal> dV, dE, dL;
    size_t size() const { return v.size(); }
};
static std::vector<Level> lev;
static const char* LEAFNAME = "1xy";
static int NVARS = 0;
static double PROBE[2][2] = { { 0.57721566490153286, 2.6854520010653064 },    // x: EulerGamma, Khinchin
                              { 1.2824271291006226, 0.91596559417721901 } };  // y: Glaisher, Catalan

static void rpn(int n, uint32_t idx, std::string& out)
{
    if (n == 1) { out.push_back(LEAFNAME[idx]); return; }
    if (n == MINF_LEV && idx == MINF_IDX) { out += MINF_CODE; return; }
    const Level& L = lev[n];
    int i = L.pi[idx];
    rpn(i, L.pa[idx], out);
    rpn(n - i, L.pb[idx], out);
    out.push_back('E');
}

// ---------------------------------------------------------------- tier 2: exp and Log of a stored value

// e = exp(a). cls: exactness bits of a (1: re is 0, 2: y is a multiple of 1/2). false: overflow or underflow.
static bool exp2(const C& a, int cls, C& e, int& ecls)
{
    bool rz = cls & 1, yh = cls & 2;
    if (rz && yh && a.y == 0) { e.re = 1; e.y = 0; ecls = 2; return true; }
    q m = rz ? (q)1 : expq(a.re);
    if (!(m > 0) || !finiteq(m)) return false;
    if (yh) {
        long long k2 = llrintq(2 * a.y);
        int k = (int)(((k2 % 4) + 4) % 4);
        switch (k) {
        case 0: e.re = m; e.y = 0; ecls = 2; break;
        case 1: e.re = 0; e.y = m / PIq; ecls = 1; break;
        case 2: e.re = -m; e.y = 0; ecls = 2; break;
        default: e.re = 0; e.y = -m / PIq; ecls = 1; break;
        }
        return true;
    }
    q k2 = nearbyintq(2 * a.y);
    q t = a.y - k2 / 2;                    // exact
    q sn, cs;
    sincosq(PIq * t, &sn, &cs);
    int k = (int)fmodq(k2, 4);
    if (k < 0) k += 4;
    q cr, ci;
    switch (k) {
    case 0: cr = cs; ci = sn; break;
    case 1: cr = -sn; ci = cs; break;
    case 2: cr = -cs; ci = -sn; break;
    default: cr = sn; ci = -cs; break;
    }
    e.re = m * cr; e.y = m * ci / PIq; ecls = 0;
    return true;
}

// l = Log(b), b != 0. one: b is the leaf 1 (then l = 0 exactly).
static void log2v(const C& b, int cls, bool one, C& l, int& lcls)
{
    bool rz = cls & 1, yh = cls & 2;
    if (one) { l.re = 0; l.y = 0; lcls = 3; return; }
    if (yh && b.y == 0) {
        if (b.re > 0) { l.re = logq(b.re); l.y = 0; }
        else { l.re = logq(-b.re); l.y = 1; }
        lcls = 2;
        return;
    }
    if (rz) { l.re = logq(PIq * fabsq(b.y)); l.y = (b.y > 0) ? (q)0.5 : (q)-0.5; lcls = 2; return; }
    q im = PIq * b.y;
    l.re = logq(hypotq(b.re, im)); l.y = atan2q(im, b.re) / PIq; lcls = 0;
}

static void finish_level(int n)
{
    Level& L = lev[n];
    size_t m = L.size();
    const double TINY = std::ldexp(1.0, -FS);
    // exp of a clean value a below 2^-FS is 1, with the displacement a
    for (size_t k = 0; k < m; k++) {
        if (L.dx[k] != NOD) continue;
        bool tiny = false;
        for (int p = 0; p < NP; p++) { double na = nrm(L.v[k].z[p]); if (na > 0 && na < TINY) tiny = true; }
        if (tiny) { L.dx[k] = (uint32_t)L.dV.size(); L.dV.push_back(DVal{}); }
    }
    L.dE.assign(L.dV.size(), DVal{}); L.dL.assign(L.dV.size(), DVal{});
    L.E.resize(m); L.L.resize(m); L.fE.assign(m, 0); L.fL.assign(m, 0);
    L.wE.resize(m * NP); L.wL.resize(m * NP);
    L.xE.resize(m * NP); L.xL.resize(m * NP); L.sE.resize(m * NP); L.sL.resize(m * NP);
#pragma omp parallel for schedule(static)
    for (long long k = 0; k < (long long)m; k++) {
        uint8_t fe = F_OK, fl = F_OK;
        uint32_t dk = L.dx[k];
        bool isS = L.fl[k] & F_S;
        for (int p = 0; p < NP; p++) {
            const C a = L.v[k].z[p];
            int cls = (L.fl[k] >> (2 * p)) & 3;
            double rv = L.rV[k * NP + p];
            int sv = L.sV[k * NP + p];
            double ma = modulus(a), na = nrm(a);
            bool zero = (cls == 3 && a.y == 0);
            bool tiny = !zero && na < TINY;
            D da = (dk != NOD) ? L.dV[dk].z[p] : D{ 0.0, 0.0 };
            C e; int ec;
            bool okx = true;
            if (tiny) { e.re = 1; e.y = 0; ec = 2; }
            else okx = exp2(a, cls, e, ec);
            if (!okx) fe &= ~F_OK;
            else {
                double ne = nrm(e);
                if (!(ne >= RMIN && ne <= RMAX)) fe &= ~F_OK;
                else {
                    L.E[k].z[p] = e;
                    fe |= (uint8_t)(ec << (2 * p));
                    double w = zero ? 0.0 : modulus(e) * (rv * ma + CEXP);
                    if (tiny) w += 2.0 * ma / ULP;
                    L.wE[k * NP + p] = (w < 1e300) ? w : 1e300;
                    int xe = std::ilogb(ne);
                    L.xE[k * NP + p] = (int16_t)xe;
                    int s = SNONE;
                    if (!zero) {
                        if (sv != SNONE) s = clamp16(sv + xe);
                        if (na < 1.0) s = std::min(s, (int)std::ilogb(na));
                    }
                    L.sE[k * NP + p] = (int16_t)s;
                    if (dk != NOD) {
                        D de;
                        if (tiny) { de.re = (double)a.re + da.re; de.y = (double)a.y + da.y; }
                        else {
                            double er = (double)e.re, ei = PI_D * (double)e.y, dr = da.re, di = PI_D * da.y;
                            de.re = er * dr - ei * di; de.y = (er * di + ei * dr) / PI_D;
                        }
                        L.dE[dk].z[p] = de;
                    }
                }
            }
            bool bad = zero || (REAL_ONLY && !((cls & 2) && a.y == 0 && a.re > 0));
            if (bad) fl &= ~F_OK;
            else {
                C l; int lc;
                bool one = (n == 1 && a.re == 1 && a.y == 0);
                log2v(a, cls, one, l, lc);
                // an S value on the negative real axis whose displacement points below the axis: the other side of the cut
                if (isS && dk != NOD && (cls & 2) && a.y == 0 && a.re < 0 && da.y < 0) l.y = -1;
                bool lzero = (l.re == 0 && l.y == 0);
                double nl = nrm(l);
                if (!lzero && !(nl >= RMIN && nl <= RMAX)) fl &= ~F_OK;
                else {
                    L.L[k].z[p] = l;
                    fl |= (uint8_t)(lc << (2 * p));
                    double w = one ? 0.0 : rv + CABS + CLOG * modulus(l);
                    L.wL[k * NP + p] = (w < 1e300) ? w : 1e300;
                    L.xL[k * NP + p] = lzero ? (int16_t)XNONE : (int16_t)std::ilogb(nl);
                    // (an S value with the digits of 1 has Log 0 and a displacement: its ingredient stays on record)
                    L.sL[k * NP + p] = (one || sv == SNONE) ? (int16_t)SNONE : (int16_t)clamp16(sv - std::ilogb(na));
                    if (dk != NOD && isS) {
                        std::complex<double> dz(da.re, PI_D * da.y), bz((double)a.re, PI_D * (double)a.y);
                        std::complex<double> qz = dz / bz;
                        L.dL[dk].z[p] = D{ qz.real(), qz.imag() / PI_D };
                    }
                }
            }
        }
        if (isS) { fe |= F_SOP; fl |= F_SOP; }
        if (dk != NOD) fe |= F_DE;
        L.fE[k] = fe; L.fL[k] = fl;
    }
}

// --inf: the entry -inf, appended to level 6 after its exp and Log have been made
static void add_minf()
{
    if (!INF_MODE || MINF_IDX != NOD) return;
    if ((int)lev.size() <= MINF_LEV || lev[MINF_LEV].E.size() != lev[MINF_LEV].v.size()) return;
    bool ok = false;
    for (size_t k = 0; k < lev[4].size(); k++) {
        bool z = true;
        for (int p = 0; p < NP; p++) if (!(((lev[4].fl[k] >> (2 * p)) & 3) == 3 && lev[4].v[k].z[p].y == 0)) z = false;
        if (z) { std::string c; rpn(4, (uint32_t)k, c); ok = (c == "111E1EE"); }
    }
    if (!ok) { fprintf(stderr, "--inf: the value 0 was expected as the term 111E1EE with 4 leaves\n"); exit(1); }
    Level& L = lev[MINF_LEV];
    Val v, z;
    uint8_t fv = 0, fe = F_OK | F_EZERO;
    for (int p = 0; p < NP; p++) {
        v.z[p].re = -1e4000Q; v.z[p].y = 0;          // (a place holder: this entry is not a value)
        z.z[p].re = 0; z.z[p].y = 0;
        fv |= (uint8_t)(2 << (2 * p));
        fe |= (uint8_t)(3 << (2 * p));               // E = 0: both parts exact
    }
    L.v.push_back(v); L.fl.push_back(fv);
    L.pa.push_back(0); L.pb.push_back(0); L.pi.push_back(0); L.dx.push_back(NOD);
    L.E.push_back(z); L.L.push_back(z); L.fE.push_back(fe); L.fL.push_back(0);
    for (int p = 0; p < NP; p++) {
        L.rV.push_back(0.0f); L.sV.push_back((int16_t)SNONE);
        L.wE.push_back(0.0); L.wL.push_back(0.0);
        L.xE.push_back((int16_t)XNONE); L.xL.push_back((int16_t)XNONE);
        L.sE.push_back((int16_t)SNONE); L.sL.push_back((int16_t)SNONE);
    }
    MINF_IDX = (uint32_t)(L.v.size() - 1);
}

// ---------------------------------------------------------------- tier 2: one candidate

struct Cand { Val v; DVal d; uint8_t fl; float r[NP]; int16_t s[NP]; };
enum { R_OK = 0, R_PEND_INACC, R_PEND_ZERO, R_PEND_COMP, R_RANGE, R_PEND_S };

// y within thr of a non-zero multiple of 1/2
static inline bool near_half(q y, double thr, q& k2)
{
    double t = 2.0 * (double)y;
    if (std::fabs(t) < 4e15 && thr < 1e-9 && std::fabs(t - std::nearbyint(t)) > 1e-6) return false;
    k2 = nearbyintq(2 * y);
    if (k2 == 0) return false;
    return PI_D * std::fabs((double)(y - k2 / 2)) <= thr;
}

static inline int eval2(const Val& Ea, uint8_t fea, const double* wEa, const int16_t* xEa, const int16_t* sEa, const DVal* dEa,
                        const Val& Lb, uint8_t flb, const double* wLb, const int16_t* xLb, const int16_t* sLb, const DVal* dLb, Cand& c)
{
    double av[NP], mv[NP], ae[NP];
    bool rex[NP], yex[NP];
    bool anyS = false;
    for (int p = 0; p < NP; p++) {
        const C& e = Ea.z[p];
        const C& l = Lb.z[p];
        C& v = c.v.z[p];
        v.re = e.re - l.re; v.y = e.y - l.y;
        rex[p] = ((fea >> (2 * p)) & 1) && ((flb >> (2 * p)) & 1);
        yex[p] = ((fea >> (2 * p + 1)) & 1) && ((flb >> (2 * p + 1)) & 1);
        double a1 = std::fabs((double)v.re), a2 = PI_D * std::fabs((double)v.y);
        mv[p] = std::max(a1, a2);
        av[p] = modulus2(a1, a2);
        ae[p] = wEa[p] + wLb[p] + CSUB * av[p];
        if (mv[p] <= THR * ae[p] * ULP) return R_PEND_ZERO;        // zero within the error bound: tier 3 decides
        int j = (xLb[p] != XNONE) ? std::min((int)xEa[p], (int)xLb[p]) : SNONE;
        if (fea & F_EZERO) j = (xLb[p] != XNONE) ? (int)xLb[p] : SNONE;      // (E is exactly 0: v = -l, nothing is absorbed)
        int s = std::min(std::min((int)sEa[p], (int)sLb[p]), j);
        c.s[p] = (int16_t)s;
        if (s != SNONE && s - std::ilogb(mv[p]) < -FS) anyS = true;
    }
    uint8_t fl = 0;
    int pend = 0;
    bool range = false, sinacc = false;
    for (int p = 0; p < NP; p++) {
        C& v = c.v.z[p];
        q k2;
        if (anyS) {
            // An S term. Its stored value is the value without what is below the resolution; that part is the
            // displacement d (to first order). No decision is possible: the parts are put on the axes by size.
            D dl = { 0.0, 0.0 };
            if (dEa) { dl.re += dEa->z[p].re; dl.y += dEa->z[p].y; }
            if (dLb) { dl.re -= dLb->z[p].re; dl.y -= dLb->z[p].y; }
            if (!(fea & F_EZERO) && xLb[p] != XNONE && std::abs((int)xEa[p] - (int)xLb[p]) > FS) {
                const C& e = Ea.z[p];
                const C& l = Lb.z[p];
                if (xEa[p] > xLb[p]) {
                    v = e; dl.re -= (double)l.re; dl.y -= (double)l.y;
                    rex[p] = (fea >> (2 * p)) & 1; yex[p] = (fea >> (2 * p + 1)) & 1;
                    ae[p] = wEa[p] + 1.0;
                } else {
                    v.re = -l.re; v.y = -l.y; dl.re += (double)e.re; dl.y += (double)e.y;
                    rex[p] = (flb >> (2 * p)) & 1; yex[p] = (flb >> (2 * p + 1)) & 1;
                    ae[p] = wLb[p] + 1.0;
                }
                if (v.re == 0) v.re = 0;
                if (v.y == 0) v.y = 0;
                double a1 = std::fabs((double)v.re), a2 = PI_D * std::fabs((double)v.y);
                mv[p] = std::max(a1, a2);
                av[p] = modulus2(a1, a2);
            }
            c.d.z[p] = dl;
            if (ae[p] > PEND * av[p]) sinacc = true;
            double thr = THR * ae[p] * ULP;
            double dre = std::fabs((double)v.re), dim = PI_D * std::fabs((double)v.y);
            bool snapped = false;
            if (!rex[p] && dre <= thr) { rex[p] = true; snapped = true; }
            if (!yex[p]) {
                if (dim <= thr) { v.y = 0; yex[p] = true; snapped = true; }
                else if (near_half(v.y, thr, k2)) { v.y = k2 / 2; yex[p] = true; snapped = true; }
            }
            if (rex[p]) v.re = 0;
            if (snapped) ae[p] += 2.0 * THR * ae[p];
            double m2 = nrm(v);
            if (!(m2 >= RMIN && m2 <= RMAX)) range = true;
        } else {
            double thr = THR * ae[p] * ULP;
            double dre = std::fabs((double)v.re), dim = PI_D * std::fabs((double)v.y);
            if (!(mv[p] >= RMIN && mv[p] <= RMAX)) range = true;
            if (ae[p] > PEND * av[p] && !pend) pend = R_PEND_INACC;
            if (!rex[p] && dre <= thr) pend = R_PEND_COMP;
            if (!yex[p] && (dim <= thr || near_half(v.y, thr, k2))) pend = R_PEND_COMP;
        }
        if (rex[p]) { v.re = 0; fl |= (uint8_t)(1 << (2 * p)); }
        if (yex[p]) fl |= (uint8_t)(2 << (2 * p));
        double rel = (av[p] > 0) ? ae[p] / av[p] : 0.0;
        c.r[p] = (float)std::min(rel, 1e30);
    }
    if (anyS) {
        if (range) return R_RANGE;
        c.fl = fl | F_S;
        return sinacc ? R_PEND_S : R_OK;      // digits lost in a cancellation: tier 3 gives them
    }
    if (range) return R_RANGE;
    if (pend) return pend;
    if ((fea | flb) & F_SOP) return R_PEND_COMP;     // an S operand: its stored digits are not its value
    c.fl = fl;
    return R_OK;
}

// ---------------------------------------------------------------- the index of stored values

static inline uint64_t mix64(uint64_t x)
{
    x ^= x >> 33; x *= 0xff51afd7ed558ccdULL; x ^= x >> 33; x *= 0xc4ceb9fe1a85ec53ULL; x ^= x >> 33;
    return x;
}

// A slot is 0 (empty) or: 27 bits of the hash, 5 bits level, 32 bits index in the level.
struct Index {
    std::vector<uint64_t> t;
    uint64_t mask;
    uint64_t n = 0;
    explicit Index(int bits) : t(1ULL << bits, 0), mask((1ULL << bits) - 1) {}
    template <class F> inline bool find(uint64_t h, F&& match) const
    {
        uint64_t p = h & mask, tag = h >> 37;
        for (;;) {
            uint64_t c = t[p];
            if (c == 0) return false;
            if ((c >> 37) == tag && match((int)((c >> 32) & 31), (uint32_t)c)) return true;
            p = (p + 1) & mask;
        }
    }
    inline void insert(uint64_t h, int level, uint32_t idx)
    {
        uint64_t p = h & mask;
        while (t[p]) p = (p + 1) & mask;
        t[p] = (h & 0xFFFFFFE000000000ULL) | ((uint64_t)level << 32) | idx;
        n++;
    }
};

// The cell of a value: both parts rounded to QB bits below the size of the value; for an S value also the
// two parts of its displacement, rounded to QBD bits below the size of the displacement.
static const int QBD = 30;
static const double BANDD = 1.0 / 256, TOLD = 9.1e-13;  // displacements are compared to 2^-40
struct Quant { int e[NP], ed[NP]; int64_t r[NP][4]; int8_t d[NP][4]; bool near; };
static inline void quant(const Val& v, const DVal* dv, Quant& Q)
{
    Q.near = false;
    for (int p = 0; p < NP; p++) {
        const C& z = v.z[p];
        q are = fabsq(z.re), ay = 4 * fabsq(z.y);
        q m = are > ay ? are : ay;
        if (m == 0) { Q.e[p] = INT_MIN; Q.r[p][0] = Q.r[p][1] = 0; Q.d[p][0] = Q.d[p][1] = 0; }
        else {
            int e;
            frexpq(m, &e);
            q sr = scalbnq(z.re, QB - e), sy = scalbnq(z.y, QB - e);
            q rr = nearbyintq(sr), ry = nearbyintq(sy);
            if (fabsq(rr) >= TWOQB || 4 * fabsq(ry) >= TWOQB) {
                e++;
                sr = scalbnq(z.re, QB - e); sy = scalbnq(z.y, QB - e);
                rr = nearbyintq(sr); ry = nearbyintq(sy);
            }
            Q.e[p] = e; Q.r[p][0] = (int64_t)rr; Q.r[p][1] = (int64_t)ry;
            double fr = (double)(sr - rr), fy = (double)(sy - ry);
            Q.d[p][0] = (std::fabs(fr) > 0.5 - BAND) ? (fr > 0 ? 1 : -1) : 0;
            Q.d[p][1] = (std::fabs(fy) > 0.5 - BAND) ? (fy > 0 ? 1 : -1) : 0;
            if (Q.d[p][0] | Q.d[p][1]) Q.near = true;
        }
        Q.ed[p] = INT_MIN; Q.r[p][2] = Q.r[p][3] = 0; Q.d[p][2] = Q.d[p][3] = 0;
        if (dv) {
            double dr = dv->z[p].re, dy = dv->z[p].y;
            double md = std::max(std::fabs(dr), 4 * std::fabs(dy));
            if (md > 0 && std::isfinite(md)) {
                int e;
                std::frexp(md, &e);
                double sr = std::ldexp(dr, QBD - e), sy = std::ldexp(dy, QBD - e);
                double rr = std::nearbyint(sr), ry = std::nearbyint(sy);
                if (std::fabs(rr) >= (double)(1LL << QBD) || 4 * std::fabs(ry) >= (double)(1LL << QBD)) {
                    e++;
                    sr = std::ldexp(dr, QBD - e); sy = std::ldexp(dy, QBD - e);
                    rr = std::nearbyint(sr); ry = std::nearbyint(sy);
                }
                Q.ed[p] = e; Q.r[p][2] = (int64_t)rr; Q.r[p][3] = (int64_t)ry;
                double fr = sr - rr, fy = sy - ry;
                Q.d[p][2] = (std::fabs(fr) > 0.5 - BANDD) ? (fr > 0 ? 1 : -1) : 0;
                Q.d[p][3] = (std::fabs(fy) > 0.5 - BANDD) ? (fy > 0 ? 1 : -1) : 0;
                if (Q.d[p][2] | Q.d[p][3]) Q.near = true;
            }
        }
    }
}
// kind 0: a clean value; 1: an S value with its displacement; 2: an S value by its digits alone.
// variant: bit (4p+c) set = part c of probe p is taken in the neighbouring cell.
static inline uint64_t qhash(const Quant& Q, int cls, int kind, int variant)
{
    uint64_t h = 0x9E3779B97F4A7C15ULL + (uint64_t)cls + ((uint64_t)kind << 8);
    for (int p = 0; p < NP; p++) {
        int64_t r0 = Q.r[p][0] + (((variant >> (4 * p)) & 1) ? Q.d[p][0] : 0);
        int64_t r1 = Q.r[p][1] + (((variant >> (4 * p + 1)) & 1) ? Q.d[p][1] : 0);
        h = mix64(h * 0x100000001B3ULL + (uint64_t)(int64_t)Q.e[p]);
        h = mix64(h ^ ((uint64_t)r0 * 0x9E3779B97F4A7C15ULL));
        h = mix64(h + ((uint64_t)r1 * 0xC2B2AE3D27D4EB4FULL));
        if (kind == 1) {
            int64_t r2 = Q.r[p][2] + (((variant >> (4 * p + 2)) & 1) ? Q.d[p][2] : 0);
            int64_t r3 = Q.r[p][3] + (((variant >> (4 * p + 3)) & 1) ? Q.d[p][3] : 0);
            h = mix64(h * 0x100000001B3ULL + (uint64_t)(int64_t)Q.ed[p]);
            h = mix64(h ^ ((uint64_t)r2 * 0x9E3779B97F4A7C15ULL));
            h = mix64(h + ((uint64_t)r3 * 0xC2B2AE3D27D4EB4FULL));
        }
    }
    return h;
}
static inline int nearmask(const Quant& Q, int kind)
{
    int m = 0;
    for (int p = 0; p < NP; p++) for (int cc = 0; cc < (kind == 1 ? 4 : 2); cc++) if (Q.d[p][cc]) m |= 1 << (4 * p + cc);
    return m;
}

static inline bool same(const Val& c, const float* rc, const Val& v, const float* rv)
{
    for (int p = 0; p < NP; p++) {
        double m = nrm(c.z[p]);
        q tol = (q)(TOLF * ((double)rc[p] + (double)rv[p] + 2.0) * ULP * m);
        if (fabsq(c.z[p].re - v.z[p].re) > tol) return false;
        if (PIq * fabsq(c.z[p].y - v.z[p].y) > tol) return false;
    }
    return true;
}
static inline bool same_d(const DVal& a, const DVal& b)
{
    for (int p = 0; p < NP; p++) {
        double m = std::max(std::max(std::fabs(a.z[p].re), 4 * std::fabs(a.z[p].y)), std::max(std::fabs(b.z[p].re), 4 * std::fabs(b.z[p].y)));
        if (std::fabs(a.z[p].re - b.z[p].re) > TOLD * m || 4 * std::fabs(a.z[p].y - b.z[p].y) > TOLD * m) return false;
    }
    return true;
}

// ---------------------------------------------------------------- tier 3: MPFR

static const long LNONE = LONG_MAX;
static const double NEG = -1e300;
static std::atomic<unsigned long long> t3_lo(0), t3_hi(0), t3_esc(0), t3_inc1(0), t3_inc2(0), t3_nodes(0), t3_conf(0);
static int t3_reported = 0;

struct MV {
    mpfr_t re, y;
    bool rz = false, yh = false;   // re exactly 0; y exactly a multiple of 1/2
    double lrel = NEG;             // log2 of the relative error bound; NEG: exact
    long s = LNONE;                // log2 of the smallest ingredient
    bool minf = false;             // the entry -inf of --inf (its exp is exactly 0; it has no Log)
    bool inited = false;
    MV() {}
    MV(const MV&) = delete;
    MV& operator=(const MV&) = delete;
    ~MV() { if (inited) { mpfr_clear(re); mpfr_clear(y); } }
    void init(mpfr_prec_t p)
    {
        if (!inited) { mpfr_init2(re, p); mpfr_init2(y, p); inited = true; }
        else if (mpfr_get_prec(re) != p) { mpfr_set_prec(re, p); mpfr_set_prec(y, p); }
    }
    void set(const MV& o, mpfr_prec_t p)
    {
        init(p);
        mpfr_set(re, o.re, MPFR_RNDN); mpfr_set(y, o.y, MPFR_RNDN);
        rz = o.rz; yh = o.yh; lrel = o.lrel; s = o.s; minf = o.minf;
    }
    bool zero() const { return rz && yh && mpfr_zero_p(y); }
};
// floor(log2) of the size of v; LONG_MIN for 0
static inline long xn(const MV& v)
{
    long x = LONG_MIN;
    if (!mpfr_zero_p(v.re)) x = (long)mpfr_get_exp(v.re) - 1;
    if (!mpfr_zero_p(v.y)) x = std::max(x, (long)mpfr_get_exp(v.y) + 1);
    return x;
}
static inline double lsum(double a, long x) { return (a <= NEG / 2 || x == LONG_MIN) ? NEG : a + (double)x; }

enum { T_OK = 0, T_ESC, T_DEEP, T_FAIL };

struct T3 {
    mpfr_prec_t P;
    mpfr_t pi, t1, t2, m, cs, sn;
    struct Slot { uint64_t key = ~0ULL; int status = 0; MV v; };
    std::vector<Slot> cache;
    explicit T3(mpfr_prec_t p) : P(p), cache(1 << 15)
    {
        mpfr_inits2(P, pi, t1, t2, m, cs, sn, (mpfr_ptr)0);
        mpfr_const_pi(pi, MPFR_RNDN);
    }
    ~T3() { mpfr_clears(pi, t1, t2, m, cs, sn, (mpfr_ptr)0); }

    bool cexp(const MV& a, MV& e)
    {
        e.init(P);
        e.minf = false;
        if (a.minf) {                              // exp(-inf) = 0, exactly
            mpfr_set_zero(e.re, 1); mpfr_set_zero(e.y, 1);
            e.rz = true; e.yh = true; e.lrel = NEG; e.s = LNONE;
            return true;
        }
        if (a.zero()) {
            mpfr_set_ui(e.re, 1, MPFR_RNDN); mpfr_set_zero(e.y, 1);
            e.rz = false; e.yh = true; e.lrel = NEG; e.s = LNONE;
            return true;
        }
        if (a.rz) mpfr_set_ui(m, 1, MPFR_RNDN); else mpfr_exp(m, a.re, MPFR_RNDN);
        if (!mpfr_regular_p(m)) return false;
        if (a.yh) {
            mpfr_mul_2si(t1, a.y, 1, MPFR_RNDN);
            long k2 = mpfr_get_si(t1, MPFR_RNDN);
            int k = (int)(((k2 % 4) + 4) % 4);
            if (k == 0 || k == 2) {
                if (k == 0) mpfr_set(e.re, m, MPFR_RNDN); else mpfr_neg(e.re, m, MPFR_RNDN);
                mpfr_set_zero(e.y, 1); e.rz = false; e.yh = true;
            } else {
                mpfr_set_zero(e.re, 1);
                mpfr_div(e.y, m, pi, MPFR_RNDN);
                if (k == 3) mpfr_neg(e.y, e.y, MPFR_RNDN);
                e.rz = true; e.yh = false;
            }
        } else {
            mpfr_mul_2si(t1, a.y, 1, MPFR_RNDN);
            mpfr_round(t2, t1);                      // the integer nearest to 2y; it can be huge (exp(i*1e114))
            mpfr_set_ui(cs, 4, MPFR_RNDN);
            mpfr_fmod(sn, t2, cs, MPFR_RNDN);        // exact
            int k = (int)(((mpfr_get_si(sn, MPFR_RNDN) % 4) + 4) % 4);
            mpfr_div_2si(t2, t2, 1, MPFR_RNDN);
            mpfr_sub(t1, a.y, t2, MPFR_RNDN);
            mpfr_mul(t1, t1, pi, MPFR_RNDN);
            mpfr_sin_cos(sn, cs, t1, MPFR_RNDN);
            switch (k) {
            case 0: mpfr_mul(e.re, m, cs, MPFR_RNDN); mpfr_mul(e.y, m, sn, MPFR_RNDN); break;
            case 1: mpfr_mul(e.re, m, sn, MPFR_RNDN); mpfr_neg(e.re, e.re, MPFR_RNDN); mpfr_mul(e.y, m, cs, MPFR_RNDN); break;
            case 2: mpfr_mul(e.re, m, cs, MPFR_RNDN); mpfr_neg(e.re, e.re, MPFR_RNDN); mpfr_mul(e.y, m, sn, MPFR_RNDN); mpfr_neg(e.y, e.y, MPFR_RNDN); break;
            default: mpfr_mul(e.re, m, sn, MPFR_RNDN); mpfr_mul(e.y, m, cs, MPFR_RNDN); mpfr_neg(e.y, e.y, MPFR_RNDN); break;
            }
            mpfr_div(e.y, e.y, pi, MPFR_RNDN);
            e.rz = false; e.yh = false;
        }
        long xa = xn(a), xe = xn(e);
        e.lrel = std::max(lsum(a.lrel, xa + 1), -(double)P) + 3;
        long s = LNONE;
        if (a.s != LNONE) s = a.s + xe;
        if (xa < 0) s = std::min(s, xa);
        e.s = s;
        return true;
    }

    // exact: b is known exactly (a leaf), so that Log(1) = 0 exactly
    bool clog(const MV& b, MV& l)
    {
        l.init(P);
        l.minf = false;
        if (b.minf) return false;
        if (b.zero()) return false;
        if (b.yh && mpfr_zero_p(b.y)) {
            if (mpfr_sgn(b.re) > 0) { mpfr_log(l.re, b.re, MPFR_RNDN); mpfr_set_zero(l.y, 1); }
            else { mpfr_neg(t1, b.re, MPFR_RNDN); mpfr_log(l.re, t1, MPFR_RNDN); mpfr_set_ui(l.y, 1, MPFR_RNDN); }
            l.yh = true;
            l.rz = (mpfr_zero_p(l.re) && b.lrel <= NEG / 2 && mpfr_sgn(b.re) > 0);
            if (l.rz) mpfr_set_zero(l.re, 1);
        } else if (b.rz) {
            mpfr_abs(t1, b.y, MPFR_RNDN); mpfr_mul(t1, t1, pi, MPFR_RNDN);
            mpfr_log(l.re, t1, MPFR_RNDN);
            mpfr_set_d(l.y, mpfr_sgn(b.y) > 0 ? 0.5 : -0.5, MPFR_RNDN);
            l.yh = true; l.rz = false;
        } else {
            mpfr_mul(t1, b.y, pi, MPFR_RNDN);
            mpfr_hypot(t2, b.re, t1, MPFR_RNDN);
            mpfr_log(l.re, t2, MPFR_RNDN);
            mpfr_atan2(t2, t1, b.re, MPFR_RNDN);
            mpfr_div(l.y, t2, pi, MPFR_RNDN);
            l.yh = false; l.rz = false;
        }
        long xl = xn(l);
        if (l.zero()) { l.lrel = NEG; l.s = LNONE; }
        else {
            l.lrel = std::max((b.lrel <= NEG / 2) ? NEG : b.lrel - (double)xl, -(double)P) + 3;
            l.s = (b.s == LNONE) ? LNONE : b.s - xn(b);
        }
        return true;
    }

    // v = e - l; labs: log2 of the absolute error bound of v
    void sub(const MV& e, const MV& l, MV& v, double& labs)
    {
        v.init(P);
        v.minf = false;
        mpfr_sub(v.re, e.re, l.re, MPFR_RNDN); mpfr_sub(v.y, e.y, l.y, MPFR_RNDN);
        v.rz = e.rz && l.rz; v.yh = e.yh && l.yh;
        if (v.rz) mpfr_set_zero(v.re, 1);
        long xe = xn(e), xl = xn(l), xv = xn(v);
        double la = std::max(lsum(e.lrel, xe + 1), lsum(l.lrel, xl + 1));
        if (la > NEG / 2) la += 1;
        double lr = (xv == LONG_MIN) ? NEG : -(double)P + (double)xv + 1;
        labs = std::max(la, lr);
        if (labs > NEG / 2) labs += 1;
        long s = std::min(e.s, l.s);
        if (xe != LONG_MIN && xl != LONG_MIN) s = std::min(s, std::min(xe, xl));
        v.s = s;
        t3_nodes++;
    }
    void seal(MV& v, double labs)
    {
        if (v.zero()) { mpfr_set_zero(v.re, 1); v.lrel = NEG; v.s = LNONE; return; }
        long xv = xn(v);
        v.lrel = std::max((labs <= NEG / 2) ? NEG : labs - (double)xv, -(double)P) + 1;
    }
    bool tiny(const mpfr_t x, long shift, double labs, double extra) const
    {
        return mpfr_zero_p(x) || (double)((long)mpfr_get_exp(x) - 1 + shift) <= labs + extra;
    }
    // y within the error bound of a non-zero multiple of 1/2: t2 := that multiple
    bool near_half(const MV& v, double labs, double extra)
    {
        mpfr_mul_2si(t1, v.y, 1, MPFR_RNDN);
        mpfr_round(t2, t1);
        if (mpfr_zero_p(t2)) return false;
        mpfr_div_2si(t2, t2, 1, MPFR_RNDN);
        mpfr_sub(t1, v.y, t2, MPFR_RNDN);
        return tiny(t1, 2, labs, extra);
    }
    // The decisions about v by size alone. true: one of them was not certain (presume: it is then made).
    bool classify(MV& v, double labs, bool presume)
    {
        bool unc = false;
        if (!v.rz && tiny(v.re, 0, labs, 12)) { unc = true; if (presume) { mpfr_set_zero(v.re, 1); v.rz = true; } }
        if (!v.yh) {
            if (tiny(v.y, 2, labs, 12)) { unc = true; if (presume) { mpfr_set_zero(v.y, 1); v.yh = true; } }
            else if (near_half(v, labs, 12)) { unc = true; if (presume) { mpfr_set(v.y, t2, MPFR_RNDN); v.yh = true; } }
        }
        seal(v, labs);
        return unc;
    }
    // The decisions about a clean stored value are the ones on record. Returned: 1 = a part on record as
    // exact does not come out as zero, 2 = a part on record as not exact cannot be told from zero here.
    int impose(MV& v, double labs, const C& rec, int cls)
    {
        int bad = 0;
        if (cls & 1) {
            if (!tiny(v.re, 0, labs, 40)) bad |= 1;
            mpfr_set_zero(v.re, 1); v.rz = true;
        } else {
            if (tiny(v.re, 0, labs, 12)) bad |= 2;
            v.rz = false;
        }
        if (cls & 2) {
            mpfr_set_float128(t2, rec.y, MPFR_RNDN);
            mpfr_sub(t1, v.y, t2, MPFR_RNDN);
            if (!tiny(t1, 2, labs, 40)) bad |= 1;
            mpfr_set(v.y, t2, MPFR_RNDN); v.yh = true;
        } else {
            if (tiny(v.y, 2, labs, 12) || near_half(v, labs, 12)) bad |= 2;
            v.yh = false;
        }
        seal(v, labs);
        return bad;
    }
    bool deep(const MV& v) const
    {
        long xv = xn(v);
        return v.s != LNONE && xv != LONG_MIN && v.s - xv < -((long)P - 100);
    }

    // the value of the stored entry (n, idx) at probe p
    int entry(int n, uint32_t idx, int p, MV& out)
    {
        if (n == MINF_LEV && idx == MINF_IDX) {    // -inf: only exp of it is ever taken (cexp)
            out.init(P);
            mpfr_set_zero(out.re, 1); mpfr_set_zero(out.y, 1);
            out.rz = true; out.yh = true; out.lrel = NEG; out.s = LNONE; out.minf = true;
            return T_OK;
        }
        if (n == 1) {
            out.init(P);
            out.minf = false;
            mpfr_set_d(out.re, idx == 0 ? 1.0 : PROBE[idx - 1][p % 2], MPFR_RNDN);
            mpfr_set_zero(out.y, 1);
            out.rz = false; out.yh = true; out.lrel = NEG; out.s = LNONE;
            return T_OK;
        }
        uint64_t key = ((uint64_t)n << 40) | ((uint64_t)p << 36) | idx;
        Slot& sl = cache[mix64(key) & (cache.size() - 1)];
        if (sl.key == key) { out.set(sl.v, P); return sl.status; }
        const Level& L = lev[n];
        int i = L.pi[idx];
        MV A, B, E, Lg;
        int st = entry(i, L.pa[idx], p, A);
        if (st == T_ESC || st == T_FAIL) return st;
        int st2 = entry(n - i, L.pb[idx], p, B);
        if (st2 == T_ESC || st2 == T_FAIL) return st2;
        if (!cexp(A, E) || !clog(B, Lg)) return T_FAIL;
        double labs;
        sub(E, Lg, out, labs);
        uint8_t fl = L.fl[idx];
        int status = (st == T_DEEP || st2 == T_DEEP) ? T_DEEP : T_OK;
        if (fl & F_S) {
            bool unc = classify(out, labs, P == PMAX);
            if ((unc || deep(out)) && P != PMAX) return T_ESC;
            if (deep(out)) status = T_DEEP;
        } else {
            int bad = impose(out, labs, L.v[idx].z[p], (fl >> (2 * p)) & 3);
            if (bad && P != PMAX) return T_ESC;       // this precision is not enough for this entry
            if (bad) {
                if (bad & 1) t3_inc1++;
                if (bad & 2) t3_inc2++;
#pragma omp critical(report)
                if (t3_reported < 12) {
                    t3_reported++;
                    std::string code;
                    rpn(n, idx, code);
                    char b1[64], b2[64];
                    quadmath_snprintf(b1, sizeof b1, "%.25Qg", L.v[idx].z[p].re);
                    quadmath_snprintf(b2, sizeof b2, "%.25Qg", L.v[idx].z[p].y);
                    printf("    tier 3 against the record (%s%s): n=%d %s  record (%s, %s)\n", (bad & 1) ? "exact part not confirmed " : "",
                           (bad & 2) ? "part of no certain sign" : "", n, code.c_str(), b1, b2);
                    mpfr_printf("        here (%.40Re, %.40Re) error 2^%.0f\n", out.re, out.y, labs);
                }
            }
        }
        sl.key = key; sl.status = status; sl.v.set(out, P);
        return status;
    }
};

static std::vector<std::unique_ptr<T3>> ctx_lo, ctx_hi;
static T3& ctx(bool hi)
{
    int t = omp_get_thread_num();
    auto& c = hi ? ctx_hi[t] : ctx_lo[t];
    if (!c) c.reset(new T3(hi ? PMAX : P1));
    return *c;
}

struct T3Res { Val v; uint8_t fl; uint8_t status; };

// The candidate eml(a, b), a from level i and b from level n-i, evaluated from the leaves.
static T3Res t3_candidate(int n, int i, uint32_t a, uint32_t b, bool hi)
{
    T3Res res;
    for (;;) {
        T3& T = ctx(hi);
        if (hi) t3_hi++; else t3_lo++;
        res.fl = F_T3; res.status = T_OK;
        bool esc = false;
        for (int p = 0; p < NP && !esc; p++) {
            MV A, B, E, Lg, V;
            int st = T.entry(i, a, p, A);
            int st2 = (st == T_ESC || st == T_FAIL) ? st : T.entry(n - i, b, p, B);
            if (st == T_ESC || st2 == T_ESC) { esc = true; break; }
            if (st == T_FAIL || st2 == T_FAIL || !T.cexp(A, E) || !T.clog(B, Lg)) { res.status = T_FAIL; return res; }
            if (st == T_DEEP || st2 == T_DEEP) res.status = T_DEEP;
            double labs;
            T.sub(E, Lg, V, labs);
            bool unc = T.classify(V, labs, hi);
            if ((unc || T.deep(V)) && !hi) { esc = true; break; }
            if (unc) res.fl |= F_PRES;
            if (T.deep(V)) res.status = T_DEEP;
            res.v.z[p].re = mpfr_get_float128(V.re, MPFR_RNDN);
            res.v.z[p].y = mpfr_get_float128(V.y, MPFR_RNDN);
            if (V.rz) { res.v.z[p].re = 0; res.fl |= (uint8_t)(1 << (2 * p)); }
            if (V.yh) res.fl |= (uint8_t)(2 << (2 * p));
        }
        if (!esc) return res;
        t3_esc++;
        hi = true;
    }
}

// The candidate eml(a, b) against the stored clean entry (l, k), both at PMAX bits.
// 1: the same value (to 200 bits above the error bounds); 0: they differ (res: the candidate); 2: tier 3 failed
static int t3_same(int n, int i, uint32_t a, uint32_t b, int l, uint32_t k, T3Res& res)
{
    T3& T = ctx(true);
    t3_conf++;
    res.fl = F_T3; res.status = T_OK;
    int same = 1;
    for (int p = 0; p < NP; p++) {
        MV A, B, E, Lg, V, W;
        int st = T.entry(i, a, p, A);
        if (st == T_ESC || st == T_FAIL) return 2;
        int st2 = T.entry(n - i, b, p, B);
        if (st2 == T_ESC || st2 == T_FAIL || !T.cexp(A, E) || !T.clog(B, Lg)) return 2;
        if (st == T_DEEP || st2 == T_DEEP) res.status = T_DEEP;
        double labs;
        T.sub(E, Lg, V, labs);
        if (T.classify(V, labs, true)) res.fl |= F_PRES;
        if (T.deep(V)) res.status = T_DEEP;
        res.v.z[p].re = mpfr_get_float128(V.re, MPFR_RNDN);
        res.v.z[p].y = mpfr_get_float128(V.y, MPFR_RNDN);
        if (V.rz) { res.v.z[p].re = 0; res.fl |= (uint8_t)(1 << (2 * p)); }
        if (V.yh) res.fl |= (uint8_t)(2 << (2 * p));
        int st3 = T.entry(l, k, p, W);
        if (st3 == T_ESC || st3 == T_FAIL) return 2;
        long xm = std::max(xn(V), xn(W));
        if (xm == LONG_MIN) continue;                 // both are 0
        double lb = std::max(std::max(lsum(V.lrel, xn(V) + 1), lsum(W.lrel, xn(W) + 1)), -(double)T.P + (double)xm + 2);
        mpfr_sub(T.t1, V.re, W.re, MPFR_RNDN);
        if (!T.tiny(T.t1, 0, lb, 200)) same = 0;
        mpfr_sub(T.t1, V.y, W.y, MPFR_RNDN);
        if (!T.tiny(T.t1, 2, lb, 200)) same = 0;
    }
    return same;
}

// ---------------------------------------------------------------- targets

typedef std::function<void(mpfr_t re, mpfr_t im, bool& yunits, const mpfr_t x, const mpfr_t y)> TFun;
struct Target {
    std::string name;
    TFun f;
    Val t;
    uint8_t cls = 0;
    int n_clean = 0, n_s = 0, n_near = 0;
    uint32_t i_clean = 0, i_s = 0, i_near = 0;
    double lg_near = 0;
};
static std::vector<Target> targets;

// the target at probe p as (re, y), at the precision of re
static void target_at(const Target& T, int p, mpfr_t re, mpfr_t y, int& cls)
{
    mpfr_prec_t P = mpfr_get_prec(re);
    mpfr_t x, yy, im, pi;
    mpfr_inits2(P, x, yy, im, pi, (mpfr_ptr)0);
    mpfr_set_d(x, PROBE[0][p % 2], MPFR_RNDN); mpfr_set_d(yy, PROBE[1][p % 2], MPFR_RNDN);
    bool yunits = false;
    T.f(re, im, yunits, x, yy);
    cls = 0;
    if (mpfr_zero_p(re)) cls |= 1;
    if (yunits) { mpfr_set(y, im, MPFR_RNDN); cls |= 2; }
    else if (mpfr_zero_p(im)) { mpfr_set_zero(y, 1); cls |= 2; }
    else { mpfr_const_pi(pi, MPFR_RNDN); mpfr_div(y, im, pi, MPFR_RNDN); }
    mpfr_clears(x, yy, im, pi, (mpfr_ptr)0);
}
static void add_target(const char* name, TFun f)
{
    Target T;
    T.name = name; T.f = f;
    mpfr_t re, y;
    mpfr_inits2(PMAX, re, y, (mpfr_ptr)0);
    for (int p = 0; p < NP; p++) {
        int cls;
        target_at(T, p, re, y, cls);
        T.t.z[p].re = mpfr_get_float128(re, MPFR_RNDN);
        T.t.z[p].y = mpfr_get_float128(y, MPFR_RNDN);
        T.cls |= (uint8_t)(cls << (2 * p));
    }
    mpfr_clears(re, y, (mpfr_ptr)0);
    targets.push_back(T);
}
#define TF [](mpfr_t re, mpfr_t im, bool& yu, const mpfr_t x, const mpfr_t y)
#define RN MPFR_RNDN
static void make_targets()
{
    (void)0;
    if (NVARS == 0) {
        add_target("0", TF { mpfr_set_zero(re, 1); mpfr_set_zero(im, 1); (void)yu; (void)x; (void)y; });
        struct Rat { const char* name; long p, q; };
        static const Rat rats[] = { { "-1", -1, 1 }, { "2", 2, 1 }, { "-2", -2, 1 }, { "3", 3, 1 }, { "1/2", 1, 2 }, { "-1/2", -1, 2 },
                                    { "1/3", 1, 3 }, { "2/3", 2, 3 }, { "-2/3", -2, 3 }, { "4", 4, 1 } };
        for (const Rat& r : rats) {
            long pp = r.p, qq = r.q;
            add_target(r.name, [pp, qq](mpfr_t re, mpfr_t im, bool&, const mpfr_t, const mpfr_t) { mpfr_set_si(re, pp, RN); mpfr_div_si(re, re, qq, RN); mpfr_set_zero(im, 1); });
        }
        add_target("sqrt2", TF { mpfr_sqrt_ui(re, 2, RN); mpfr_set_zero(im, 1); (void)yu; (void)x; (void)y; });
        add_target("phi", TF { mpfr_sqrt_ui(re, 5, RN); mpfr_add_ui(re, re, 1, RN); mpfr_div_ui(re, re, 2, RN); mpfr_set_zero(im, 1); (void)yu; (void)x; (void)y; });
        add_target("i", TF { mpfr_set_zero(re, 1); mpfr_set_ui(im, 1, RN); (void)yu; (void)x; (void)y; });
        add_target("-i", TF { mpfr_set_zero(re, 1); mpfr_set_si(im, -1, RN); (void)yu; (void)x; (void)y; });
        add_target("pi", TF { mpfr_const_pi(re, RN); mpfr_set_zero(im, 1); (void)yu; (void)x; (void)y; });
        add_target("-pi", TF { mpfr_const_pi(re, RN); mpfr_neg(re, re, RN); mpfr_set_zero(im, 1); (void)yu; (void)x; (void)y; });
        add_target("i*pi", TF { mpfr_set_zero(re, 1); mpfr_set_ui(im, 1, RN); yu = true; (void)x; (void)y; });
        add_target("-i*pi", TF { mpfr_set_zero(re, 1); mpfr_set_si(im, -1, RN); yu = true; (void)x; (void)y; });
        add_target("i*pi/2", TF { mpfr_set_zero(re, 1); mpfr_set_d(im, 0.5, RN); yu = true; (void)x; (void)y; });
        add_target("-i*pi/2", TF { mpfr_set_zero(re, 1); mpfr_set_d(im, -0.5, RN); yu = true; (void)x; (void)y; });
        add_target("pi/2", TF { mpfr_const_pi(re, RN); mpfr_div_ui(re, re, 2, RN); mpfr_set_zero(im, 1); (void)yu; (void)x; (void)y; });
        add_target("2*pi*i", TF { mpfr_set_zero(re, 1); mpfr_set_ui(im, 2, RN); yu = true; (void)x; (void)y; });
        add_target("e-1", TF { mpfr_set_ui(re, 1, RN); mpfr_exp(re, re, RN); mpfr_sub_ui(re, re, 1, RN); mpfr_set_zero(im, 1); (void)yu; (void)x; (void)y; });
        add_target("1/e", TF { mpfr_set_si(re, -1, RN); mpfr_exp(re, re, RN); mpfr_set_zero(im, 1); (void)yu; (void)x; (void)y; });
        add_target("ln2", TF { mpfr_const_log2(re, RN); mpfr_set_zero(im, 1); (void)yu; (void)x; (void)y; });
        // more constants for the search above the table
        static const Rat rats2[] = { { "5", 5, 1 }, { "-3", -3, 1 }, { "1/4", 1, 4 }, { "3/2", 3, 2 }, { "3/4", 3, 4 }, { "-1/3", -1, 3 } };
        for (const Rat& r : rats2) {
            long pp = r.p, qq = r.q;
            add_target(r.name, [pp, qq](mpfr_t re, mpfr_t im, bool&, const mpfr_t, const mpfr_t) { mpfr_set_si(re, pp, RN); mpfr_div_si(re, re, qq, RN); mpfr_set_zero(im, 1); });
        }
        add_target("sqrt3", TF { mpfr_sqrt_ui(re, 3, RN); mpfr_set_zero(im, 1); (void)yu; (void)x; (void)y; });
        add_target("1/pi", TF { mpfr_const_pi(re, RN); mpfr_ui_div(re, 1, re, RN); mpfr_set_zero(im, 1); (void)yu; (void)x; (void)y; });
        add_target("pi^2", TF { mpfr_const_pi(re, RN); mpfr_sqr(re, re, RN); mpfr_set_zero(im, 1); (void)yu; (void)x; (void)y; });
        add_target("e^pi", TF { mpfr_const_pi(re, RN); mpfr_exp(re, re, RN); mpfr_set_zero(im, 1); (void)yu; (void)x; (void)y; });
        add_target("ln(pi)", TF { mpfr_const_pi(re, RN); mpfr_log(re, re, RN); mpfr_set_zero(im, 1); (void)yu; (void)x; (void)y; });
        add_target("sqrt(pi)", TF { mpfr_const_pi(re, RN); mpfr_sqrt(re, re, RN); mpfr_set_zero(im, 1); (void)yu; (void)x; (void)y; });
        add_target("2i", TF { mpfr_set_zero(re, 1); mpfr_set_ui(im, 2, RN); (void)yu; (void)x; (void)y; });
        add_target("1+i", TF { mpfr_set_ui(re, 1, RN); mpfr_set_ui(im, 1, RN); (void)yu; (void)x; (void)y; });
    } else if (NVARS == 1) {
        add_target("-x", TF { mpfr_neg(re, x, RN); mpfr_set_zero(im, 1); (void)yu; (void)y; });
        add_target("1/x", TF { mpfr_ui_div(re, 1, x, RN); mpfr_set_zero(im, 1); (void)yu; (void)y; });
        add_target("x^2", TF { mpfr_sqr(re, x, RN); mpfr_set_zero(im, 1); (void)yu; (void)y; });
        add_target("sqrt(x)", TF { mpfr_sqrt(re, x, RN); mpfr_set_zero(im, 1); (void)yu; (void)y; });
        add_target("x/2", TF { mpfr_div_ui(re, x, 2, RN); mpfr_set_zero(im, 1); (void)yu; (void)y; });
        add_target("2x", TF { mpfr_mul_ui(re, x, 2, RN); mpfr_set_zero(im, 1); (void)yu; (void)y; });
        add_target("x+1", TF { mpfr_add_ui(re, x, 1, RN); mpfr_set_zero(im, 1); (void)yu; (void)y; });
        add_target("x-1", TF { mpfr_sub_ui(re, x, 1, RN); mpfr_set_zero(im, 1); (void)yu; (void)y; });
        add_target("1-x", TF { mpfr_ui_sub(re, 1, x, RN); mpfr_set_zero(im, 1); (void)yu; (void)y; });
        add_target("exp(x)", TF { mpfr_exp(re, x, RN); mpfr_set_zero(im, 1); (void)yu; (void)y; });
        add_target("ln(x)", TF { mpfr_log(re, x, RN); mpfr_set_zero(im, 1); (void)yu; (void)y; });
        add_target("exp(-x)", TF { mpfr_neg(re, x, RN); mpfr_exp(re, re, RN); mpfr_set_zero(im, 1); (void)yu; (void)y; });
        add_target("e*x", TF { mpfr_set_ui(re, 1, RN); mpfr_exp(re, re, RN); mpfr_mul(re, re, x, RN); mpfr_set_zero(im, 1); (void)yu; (void)y; });
        add_target("x/e", TF { mpfr_set_si(re, -1, RN); mpfr_exp(re, re, RN); mpfr_mul(re, re, x, RN); mpfr_set_zero(im, 1); (void)yu; (void)y; });
        add_target("x+e", TF { mpfr_set_ui(re, 1, RN); mpfr_exp(re, re, RN); mpfr_add(re, re, x, RN); mpfr_set_zero(im, 1); (void)yu; (void)y; });
        add_target("x-e", TF { mpfr_set_ui(re, 1, RN); mpfr_exp(re, re, RN); mpfr_sub(re, x, re, RN); mpfr_set_zero(im, 1); (void)yu; (void)y; });
        add_target("i*x", TF { mpfr_set_zero(re, 1); mpfr_set(im, x, RN); (void)yu; (void)y; });
        add_target("x+i", TF { mpfr_set(re, x, RN); mpfr_set_ui(im, 1, RN); (void)yu; (void)y; });
    } else {
        add_target("x+y", TF { mpfr_add(re, x, y, RN); mpfr_set_zero(im, 1); (void)yu; });
        add_target("x-y", TF { mpfr_sub(re, x, y, RN); mpfr_set_zero(im, 1); (void)yu; });
        add_target("y-x", TF { mpfr_sub(re, y, x, RN); mpfr_set_zero(im, 1); (void)yu; });
        add_target("x*y", TF { mpfr_mul(re, x, y, RN); mpfr_set_zero(im, 1); (void)yu; });
        add_target("x/y", TF { mpfr_div(re, x, y, RN); mpfr_set_zero(im, 1); (void)yu; });
        add_target("y/x", TF { mpfr_div(re, y, x, RN); mpfr_set_zero(im, 1); (void)yu; });
        add_target("x^y", TF { mpfr_pow(re, x, y, RN); mpfr_set_zero(im, 1); (void)yu; });
        add_target("y^x", TF { mpfr_pow(re, y, x, RN); mpfr_set_zero(im, 1); (void)yu; });
        add_target("log_x(y)", TF { mpfr_log(re, y, RN); mpfr_log(im, x, RN); mpfr_div(re, re, im, RN); mpfr_set_zero(im, 1); (void)yu; });
        add_target("log_y(x)", TF { mpfr_log(re, x, RN); mpfr_log(im, y, RN); mpfr_div(re, re, im, RN); mpfr_set_zero(im, 1); (void)yu; });
        add_target("(x+y)/2", TF { mpfr_add(re, x, y, RN); mpfr_div_ui(re, re, 2, RN); mpfr_set_zero(im, 1); (void)yu; });
        add_target("x^2+y^2", TF { mpfr_sqr(re, x, RN); mpfr_sqr(im, y, RN); mpfr_add(re, re, im, RN); mpfr_set_zero(im, 1); (void)yu; });
        add_target("hypot", TF { mpfr_hypot(re, x, y, RN); mpfr_set_zero(im, 1); (void)yu; });
    }
}

// log10 of the distance of the stored entry from the target, at PMAX bits; status: the tier-3 status
static double entry_distance(int n, uint32_t idx, const Target& T, int& status)
{
    T3& X = ctx(true);
    double worst = -1e9;
    status = T_OK;
    mpfr_t tr, ty, d1, d2;
    mpfr_inits2(PMAX, tr, ty, d1, d2, (mpfr_ptr)0);
    for (int p = 0; p < NP; p++) {
        MV V;
        int st = X.entry(n, idx, p, V);
        if (st == T_ESC || st == T_FAIL) { status = st; worst = 1e9; break; }
        if (st == T_DEEP) status = T_DEEP;
        int cls;
        target_at(T, p, tr, ty, cls);
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

// ---------------------------------------------------------------- main

struct Stats {
    unsigned long long pairs = 0, log0 = 0, tower = 0, range = 0, zeros = 0, newv = 0, dup = 0, snew = 0, sdup = 0, sdrop = 0;
    unsigned long long p_inacc = 0, p_zero = 0, p_comp = 0, p_s = 0, pres = 0, deep = 0, fail = 0, nearb = 0, resur = 0, twins = 0, conffail = 0;
};
struct Pend { uint32_t a, b; uint8_t i, reason; };
struct Conf { uint32_t a, b, k; uint8_t i, l; };

static double mem_gb()
{
    FILE* f = fopen("/proc/self/statm", "r");
    if (!f) return 0;
    unsigned long long a = 0, b = 0;
    if (fscanf(f, "%llu %llu", &a, &b) != 2) b = 0;
    fclose(f);
    return (double)b * 4096.0 / 1073741824.0;
}
static double now() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }

// Windows (MSYS2 UCRT64): no aligned_alloc; _aligned_malloc needs _aligned_free
#ifdef _WIN32
#include <malloc.h>
static inline void* aligned_alloc_portable(size_t a, size_t s) { return _aligned_malloc(s, a); }
static inline void aligned_free_portable(void* p) { _aligned_free(p); }
#define aligned_alloc aligned_alloc_portable
#else
static inline void aligned_free_portable(void* p) { free(p); }
#endif

#include "mitm.inc"

int main(int argc, char** argv)
{
    int N = 14, hbits = 0, threads = 0;
    long long audit = 0;
    double memlimit = 0;
    std::string only, dumpfile;
    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        if (a == "--maxn") N = atoi(argv[++i]);
        else if (a == "--hbits") hbits = atoi(argv[++i]);
        else if (a == "--vars") NVARS = atoi(argv[++i]);
        else if (a == "--threads") threads = atoi(argv[++i]);
        else if (a == "--real") REAL_ONLY = true;
        else if (a == "--inf") INF_MODE = true;
        else if (a == "--inf-search") INF_CSEARCH = true;
        else if (a == "--drop-s") DROP_S = true;
        else if (a == "--noconfirm") CONFIRM = false;
        else if (a == "--only") only = argv[++i];
        else if (a == "--dump") dumpfile = argv[++i];
        else if (a == "--audit") audit = atoll(argv[++i]);
        else if (a == "--pend") PEND = atof(argv[++i]);
        else if (a == "--fs") FS = atoi(argv[++i]);
        else if (a == "--p1") P1 = atoi(argv[++i]);
        else if (a == "--pmax") PMAX = atoi(argv[++i]);
        else if (a == "--band") BAND = atof(argv[++i]);
        else if (a == "--memlimit") memlimit = atof(argv[++i]);
        else if (a == "--mitm") M_NMAX = atoi(argv[++i]);
        else if (a == "--nt") M_NT = atoi(argv[++i]);
        else if (a == "--nmin") M_NMIN = atoi(argv[++i]);
        else if (a == "--zone") M_ZONE = atoi(argv[++i]);
        else if (a == "--etalon") M_ESAMPLE = atoi(argv[++i]);
        else if (a == "--sbits") M_SBITS = atoi(argv[++i]);
        else if (a == "--tau") M_TAU = std::ldexp(1.0, -atoi(argv[++i]));
        else if (a == "--show") M_SHOW = atoi(argv[++i]);
        else if (a == "--kwin") M_KWIN = atoi(argv[++i]);
        else if (a == "--clean") M_CLEAN = true;
        else if (a == "--exact") M_EXACT = true;
        else if (a == "--noprune") M_PRUNE = false;
        else if (a == "--mitmskip") M_SKIP = argv[++i];
        else if (a == "--nozq") M_ZQ = false;
        else if (a == "--parts") M_PARTS = atoi(argv[++i]);
        else if (a == "--t3batch") M_T3BATCH = (size_t)std::max(1, atoi(argv[++i]));
        else if (a == "--f32") M_F32 = atoi(argv[++i]);
        else if (a == "--f32check") M_F32CHECK = true;
        else if (a == "--f32rel") M_F32REL = std::ldexp(1.0, -atoi(argv[++i]));
#if defined(F32_GPU)
        else if (a == "--f32work") M_F32WORK = atof(argv[++i]);
        else if (a == "--f32cap") M_F32CAP = 1u << atoi(argv[++i]);
        else if (a == "--f32ucap") M_F32UCAP = 1u << atoi(argv[++i]);
        else if (a == "--f32r0") M_F32R0 = atoi(argv[++i]);
#endif
        else { fprintf(stderr, "unknown option %s\n", a.c_str()); return 1; }
    }
    if (NP == 1 && NVARS > 0) { fprintf(stderr, "functions need the NP=2 build\n"); return 1; }
    if (N > 31) { fprintf(stderr, "at most 31 leaves\n"); return 1; }
    if (threads > 0) omp_set_num_threads(threads);
    threads = omp_get_max_threads();
    ULP = std::ldexp(1.0, -113);
    TWOQB = scalbnq(1, QB);
    ctx_lo.resize(threads); ctx_hi.resize(threads);

    make_targets();
    if (!only.empty()) {
        std::vector<Target> keep;
        for (auto& T : targets) if (("," + only + ",").find("," + T.name + ",") != std::string::npos) keep.push_back(T);
        targets = keep;
    }

    // --inf WITHOUT --real = C_inf = C u {inf-, inf+} of Appendix A (Table A.1, which is the
    // table of special values of ISO C Annex G / IEEE 754). A finite value comes from an infinite operand only as
    // EML(inf-, w) = -Log w, w in C*, and every inf- subterm can be replaced by the shortest one (6 leaves) without
    // changing the value, so the same single entry of level 6 serves (left operand only, exp exactly 0). The table is
    // the same code as for the extended reals. The search above the table over C_inf only with --inf-search
    // (experimental: it has to pass the etalons first).
    if (INF_MODE && !REAL_ONLY && (M_NMAX > 0 || M_ESAMPLE > 0) && !INF_CSEARCH) {
        fprintf(stderr, "--inf over C: table only (the search above the table needs --inf-search, experimental)\n"); return 1;
    }
    printf("EML table (Step 0): NP=%d vars=%d maxn=%d %s  tier 2: 113 bits, PEND=%.0f FS=%d  tier 3: %ld / %ld bits  threads=%d%s\n",
           NP, NVARS, N, REAL_ONLY ? (INF_MODE ? "extended reals (--inf)" : "real-only") : (INF_MODE ? "complex C_inf (--inf)" : "complex"),
           PEND, FS, (long)P1, (long)PMAX, threads, DROP_S ? "  S terms dropped" : "");
    if (hbits == 0) {
        double est = (NVARS == 0) ? 1.7e8 * std::pow(3.4, N - 19) : (NVARS == 1) ? 2.0e7 * std::pow(6.0, N - 11) : 2.4e7 * std::pow(9.0, N - 9);
        hbits = 20;
        while ((double)(1ULL << hbits) * 0.6 < est && hbits < 36) hbits++;
    }
    printf("index: 2^%d slots, %.2f GB\n", hbits, (double)(1ULL << hbits) * 8.0 / 1073741824.0);
    fflush(stdout);

    lev.resize(N + 1);
    Index ix(hbits);
    const uint64_t hlimit = (uint64_t)((double)(ix.mask + 1) * 0.72);

    // looks the candidate up; true: a stored value is the same value
    auto lookup = [&](const Cand& c, const Quant& Q, Stats& st, int& ml, uint32_t& mk) -> bool {
        bool isS = c.fl & F_S;
        int cls = c.fl & CLSMASK, kind = isS ? 1 : 0;
        auto match = [&](int l, uint32_t k) -> bool {
            const Level& X = lev[l];
            uint8_t fe = X.fl[k];
            if (isS != (bool)(fe & F_S)) return false;      // an S entry and a clean one are never the same value
            if ((fe & CLSMASK) != cls) return false;
            if (!same(c.v, c.r, X.v[k], &X.rV[(size_t)k * NP])) return false;
            if (isS && !same_d(c.d, X.dV[X.dx[k]])) return false;     // S twins: same digits, different displacements
            ml = l; mk = k;
            return true;
        };
        if (ix.find(qhash(Q, cls, kind, 0), match)) return true;
        if (Q.near) {
            int nm = nearmask(Q, kind);
            if (nm) {
                st.nearb++;
                for (int var = 1; var < (1 << (4 * NP)); var++)
                    if ((var & nm) == var && ix.find(qhash(Q, cls, kind, var), match)) return true;
            }
        }
        return false;
    };
    auto store = [&](int n, const Cand& c, const Quant& Q, int i, uint32_t a, uint32_t b) {
        Level& Ln = lev[n];
        uint32_t idx = (uint32_t)Ln.v.size();
        bool isS = c.fl & F_S;
        int cls = c.fl & CLSMASK;
        Ln.v.push_back(c.v); Ln.fl.push_back(c.fl);
        for (int p = 0; p < NP; p++) { Ln.rV.push_back(c.r[p]); Ln.sV.push_back(c.s[p]); }
        Ln.pa.push_back(a); Ln.pb.push_back(b); Ln.pi.push_back((uint8_t)i);
        if (isS) { Ln.dx.push_back((uint32_t)Ln.dV.size()); Ln.dV.push_back(c.d); }
        else Ln.dx.push_back(NOD);
        ix.insert(qhash(Q, cls, isS ? 1 : 0, 0), n, idx);
        if (isS) {
            // the first S entry with these digits can also be found by its digits alone (for the targets)
            auto m2 = [&](int l, uint32_t k) -> bool {
                const Level& X = lev[l];
                uint8_t fe = X.fl[k];
                return (fe & F_S) && (fe & CLSMASK) == cls && same(c.v, c.r, X.v[k], &X.rV[(size_t)k * NP]);
            };
            if (!ix.find(qhash(Q, cls, 2, 0), m2)) ix.insert(qhash(Q, cls, 2, 0), n, idx);
        }
    };

    {
        Level& L1 = lev[1];
        for (int l = 0; l <= NVARS; l++) {
            Cand c;
            c.fl = 0;
            for (int p = 0; p < NP; p++) {
                c.v.z[p].re = (l == 0) ? (q)1 : (q)PROBE[l - 1][p % 2];
                c.v.z[p].y = 0;
                c.fl |= (uint8_t)(2 << (2 * p));
                c.r[p] = 0; c.s[p] = (int16_t)SNONE;
            }
            Quant Q;
            quant(c.v, nullptr, Q);
            store(1, c, Q, 0, 0, 0);
        }
        (void)L1;
        finish_level(1);
    }

    auto check_targets = [&](int n) {
        for (auto& T : targets) {
            if (T.n_clean) continue;
            Cand c;
            c.v = T.t;
            for (int p = 0; p < NP; p++) { c.r[p] = 2; c.s[p] = (int16_t)SNONE; }
            Quant Q;
            quant(c.v, nullptr, Q);
            int cls = T.cls & CLSMASK;
            std::vector<uint32_t> hits;
            auto match = [&](int l, uint32_t k) -> bool {
                const Level& X = lev[l];
                uint8_t fe = X.fl[k];
                if ((fe & CLSMASK) != cls) return false;
                if (!same(c.v, c.r, X.v[k], &X.rV[(size_t)k * NP])) return false;
                if (fe & F_S) { if (!T.n_s || l < T.n_s) { T.n_s = l; T.i_s = k; } }
                else if (l == n) hits.push_back(k);
                return false;        // keep looking
            };
            int nm = nearmask(Q, 0);
            for (int var = 0; var < (1 << (4 * NP)); var++)
                if ((var & nm) == var) { ix.find(qhash(Q, cls, 0, var), match); ix.find(qhash(Q, cls, 2, var), match); }
            // a clean entry with the digits of the target is a form of it only if it agrees at PMAX bits
            for (uint32_t k : hits) {
                if (T.n_clean) break;
                int status;
                double lg = entry_distance(n, k, T, status);
                bool good = (status == T_OK || status == T_DEEP) && (lg < -1e8 || lg < -(double)PMAX * 0.30103 + 60);
                if (good) { T.n_clean = n; T.i_clean = k; }
                else if (!T.n_near) { T.n_near = n; T.i_near = k; T.lg_near = lg; }
            }
        }
    };
    check_targets(1);

    printf("\n  n   K  pairs        log0       tower      range     zeros     tier3(inacc/zero/part/S)      presumed  new          S-new      dup         S-dup      cumulative   real    growth  sec(t2+t3)  GB\n");
    unsigned long long cum = lev[1].size();
    double prev = (double)lev[1].size();
    printf("%3d %3d  %-11s  %-9s  %-9s  %-8s  %-8s  %-28s  %-8s  %-11llu  %-9s  %-10s  %-9s  %-11llu\n", 1, 1, "-", "-", "-", "-", "-", "-", "-",
           (unsigned long long)lev[1].size(), "-", "-", "-", cum);
    double t00 = now();
    unsigned long long tot_near = 0, tot_resur = 0, tot_twins = 0, tot_conffail = 0;
    int fail_shown = 0;
    unsigned long long audit_n = 0, audit_bad = 0, audit_unc = 0;
    double audit_worst = 0;
    std::vector<Pend> auditlist;
    std::vector<Val> auditval;
    std::vector<float> auditr;
    long long audit_ctr = 0;

    for (int n = 2; n <= N; n++) {
        double t0 = now();
        Level& Ln = lev[n];
        Stats st;
        std::vector<Pend> pend, pendS;
        std::vector<Cand> pendScand;
        std::vector<Conf> conf;
        bool full = false;
        {
            unsigned long long pn = 0;
            for (int i = 1; i < n; i++) pn += (unsigned long long)lev[i].size() * lev[n - i].size();
            if (pn > 4200000000ULL) {
                printf("n=%d: %llu pairs, more than the 32-bit indices of a level allow: stop\n", n, pn);
                N = n - 1;
                break;
            }
            if (memlimit > 0 && mem_gb() + (double)pn * 0.85 * (double)(sizeof(Val) + 10 + 6 * NP) / 1073741824.0 > memlimit) {
                printf("n=%d: about %.0f GB more would be needed, above --memlimit %.0f: stop\n", n,
                       (double)pn * 0.85 * (double)(sizeof(Val) + 10 + 6 * NP) / 1073741824.0, memlimit);
                N = n - 1;
                break;
            }
            Ln.v.reserve(pn); Ln.fl.reserve(pn); Ln.rV.reserve(pn * NP); Ln.sV.reserve(pn * NP);
            Ln.pa.reserve(pn); Ln.pb.reserve(pn); Ln.pi.reserve(pn); Ln.dx.reserve(pn);
        }
        const int BATCH = 64;
        Cand cbuf[BATCH]; Quant qbuf[BATCH]; uint32_t abuf[BATCH], bbuf[BATCH]; uint8_t ibuf[BATCH];
        int nb = 0;
        auto flush = [&]() {
            for (int k = 0; k < nb; k++) {
                const Cand& c = cbuf[k];
                bool isS = c.fl & F_S;
                int ml = 0;
                uint32_t mk = 0;
                if (lookup(c, qbuf[k], st, ml, mk)) {
                    if (isS) st.sdup++;
                    else if (CONFIRM) conf.push_back(Conf{ abuf[k], bbuf[k], mk, ibuf[k], (uint8_t)ml });
                    else st.dup++;
                    continue;
                }
                if (isS && DROP_S) { st.sdrop++; continue; }
                store(n, c, qbuf[k], ibuf[k], abuf[k], bbuf[k]);
                if (isS) st.snew++;
                else {
                    st.newv++;
                    if (audit > 0 && !(c.fl & F_T3) && ++audit_ctr % audit == 0) {
                        auditlist.push_back(Pend{ abuf[k], bbuf[k], ibuf[k], 0 });
                        auditval.push_back(c.v);
                        for (int p = 0; p < NP; p++) auditr.push_back(c.r[p]);
                    }
                }
            }
            nb = 0;
            if (ix.n > hlimit) full = true;
        };
        auto push = [&](const Cand& c, int i, uint32_t a, uint32_t b) {
            cbuf[nb] = c;
            quant(c.v, (c.fl & F_S) ? &c.d : nullptr, qbuf[nb]);
            __builtin_prefetch(&ix.t[qhash(qbuf[nb], c.fl & CLSMASK, (c.fl & F_S) ? 1 : 0, 0) & ix.mask]);
            abuf[nb] = a; bbuf[nb] = b; ibuf[nb] = (uint8_t)i;
            if (++nb == BATCH) flush();
        };

        // the candidate as tier 3 gives it: value, marks, smallest ingredient; isS: it is S
        auto from_t3 = [&](const Pend& P, const T3Res& R, Cand& c, bool& range, bool& isS, bool count) {
            c.v = R.v; c.fl = R.fl; c.d = DVal{};
            range = false; isS = (R.status == T_DEEP);
            if (isS && count) st.deep++;
            const Level& A = lev[P.i];
            const Level& B = lev[n - P.i];
            for (int p = 0; p < NP; p++) {
                double m = nrm(c.v.z[p]);
                bool zero = (((c.fl >> (2 * p)) & 3) == 3 && c.v.z[p].y == 0);
                if (zero) { if (p == 0 && count) st.zeros++; }
                else if (!(m >= RMIN && m <= RMAX)) range = true;
                int xe = A.xE[(size_t)P.a * NP + p], xl = B.xL[(size_t)P.b * NP + p];
                int se = A.sE[(size_t)P.a * NP + p], sl = B.sL[(size_t)P.b * NP + p];
                const bool ez = A.fE[P.a] & F_EZERO;                         // (--inf: E is exactly 0, v = -l)
                int j = (xl != XNONE) ? std::min(xe, xl) : SNONE;
                if (ez) j = (xl != XNONE) ? xl : SNONE;
                int s = std::min(std::min(se, sl), j);
                if (zero) s = SNONE;
                if (!zero) {
                    int xv = std::ilogb(m);
                    if (s != SNONE && s - xv < -FS) isS = true;
                    else if (!ez && xl != XNONE && xv < std::max(xe, xl) - FS) {
                        // The value is below the resolution of the two operands it is the difference of: all that
                        // could be seen of them has cancelled, and what is left came back from below. exp and Log
                        // have left the square of it in the operands (Log(1 - t) = -t - t*t/2 - ...), which is below
                        // the resolution of the value: S again.
                        isS = true;
                        s = clamp16(std::min(s, 2 * xv - std::max(xe, xl) - 1));
                        if (count) st.resur++;
                    }
                }
                c.r[p] = (float)CT3; c.s[p] = (int16_t)s;
            }
        };

        for (int i = 1; i < n && !full; i++) {
            int j = n - i;
            const Level& A = lev[i];
            const Level& B = lev[j];
            size_t na = A.size(), nbv = B.size();
            st.pairs += (unsigned long long)na * nbv;
            for (size_t a = 0; a < na && !full; a++) {
                if (!(A.fE[a] & F_OK)) { st.tower += nbv; continue; }
                const Val Ea = A.E[a];
                const uint8_t fea = A.fE[a];
                const double* wEa = &A.wE[a * NP];
                const int16_t* xEa = &A.xE[a * NP];
                const int16_t* sEa = &A.sE[a * NP];
                const DVal* dEa = (fea & F_DE) ? &A.dE[A.dx[a]] : nullptr;
                for (size_t b = 0; b < nbv; b++) {
                    uint8_t flb = B.fL[b];
                    if (!(flb & F_OK)) {
                        bool z = true;
                        for (int p = 0; p < NP; p++) if (!(((B.fl[b] >> (2 * p)) & 3) == 3 && B.v[b].z[p].y == 0)) z = false;
                        if (z || REAL_ONLY) st.log0++; else st.tower++;
                        continue;
                    }
                    Cand c;
                    const DVal* dLb = (flb & F_SOP) ? &B.dL[B.dx[b]] : nullptr;
                    int r = eval2(Ea, fea, wEa, xEa, sEa, dEa, B.L[b], flb, &B.wL[b * NP], &B.xL[b * NP], &B.sL[b * NP], dLb, c);
                    switch (r) {
                    case R_OK: push(c, i, (uint32_t)a, (uint32_t)b); break;
                    case R_RANGE: st.range++; break;
                    case R_PEND_S:
                        if (DROP_S) { st.sdrop++; break; }
                        pendS.push_back(Pend{ (uint32_t)a, (uint32_t)b, (uint8_t)i, (uint8_t)r });
                        pendScand.push_back(c);
                        st.p_s++;
                        break;
                    default:
                        pend.push_back(Pend{ (uint32_t)a, (uint32_t)b, (uint8_t)i, (uint8_t)r });
                        if (r == R_PEND_INACC) st.p_inacc++; else if (r == R_PEND_ZERO) st.p_zero++; else st.p_comp++;
                    }
                    if (full) break;
                }
            }
        }
        if (!full) flush();
        double t1 = now();

        // tier 3 for the candidates tier 2 could not decide, in blocks
        const size_t BLOCK = 1 << 21;
        std::vector<T3Res> res;
        for (size_t off = 0; off < pend.size() && !full; off += BLOCK) {
            size_t cnt = std::min(BLOCK, pend.size() - off);
            res.resize(cnt);
#pragma omp parallel for schedule(dynamic, 64)
            for (long long k = 0; k < (long long)cnt; k++) {
                const Pend& P = pend[off + k];
                res[k] = t3_candidate(n, P.i, P.a, P.b, P.reason != R_PEND_INACC);
            }
            for (size_t k = 0; k < cnt && !full; k++) {
                const Pend& P = pend[off + k];
                const T3Res& R = res[k];
                if (R.status == T_FAIL) {
                    st.fail++;
                    if (fail_shown < 6) {
                        fail_shown++;
                        std::string ca, cb;
                        rpn(P.i, P.a, ca); rpn(n - P.i, P.b, cb);
                        printf("         tier 3 failed for %s%sE (operands %s, %s)\n", ca.c_str(), cb.c_str(),
                               (lev[P.i].fl[P.a] & F_S) ? "S" : "clean", (lev[n - P.i].fl[P.b] & F_S) ? "S" : "clean");
                    }
                    continue;
                }
                if (R.fl & F_PRES) st.pres++;
                Cand c;
                bool range, isS;
                from_t3(P, R, c, range, isS, true);
                if (range) { st.range++; continue; }
                if (isS) c.fl |= F_S;
                push(c, P.i, P.a, P.b);
            }
            if (!full) flush();
        }
        // S candidates whose digits lost accuracy: digits from tier 3, displacement as computed
        for (size_t off = 0; off < pendS.size() && !full; off += BLOCK) {
            size_t cnt = std::min(BLOCK, pendS.size() - off);
            res.resize(cnt);
#pragma omp parallel for schedule(dynamic, 64)
            for (long long k = 0; k < (long long)cnt; k++) {
                const Pend& P = pendS[off + k];
                res[k] = t3_candidate(n, P.i, P.a, P.b, false);
            }
            for (size_t k = 0; k < cnt && !full; k++) {
                const Pend& P = pendS[off + k];
                const T3Res& R = res[k];
                Cand c = pendScand[off + k];
                if (R.status != T_FAIL) {
                    bool range = false, zero = false;
                    uint8_t fl = F_S | F_T3;
                    for (int p = 0; p < NP; p++) {
                        C v = R.v.z[p];
                        int cls = (R.fl >> (2 * p)) & 3;
                        if (cls == 3 && v.y == 0) { zero = true; break; }
                        if (!(cls & 1)) v.re -= (q)c.d.z[p].re;
                        if (!(cls & 2)) v.y -= (q)c.d.z[p].y;
                        double m = nrm(v);
                        if (!(m >= RMIN && m <= RMAX)) range = true;
                        c.v.z[p] = v; c.r[p] = (float)CT3;
                        fl |= (uint8_t)(cls << (2 * p));
                    }
                    if (zero || range) { st.range++; continue; }
                    c.fl = fl;
                }
                push(c, P.i, P.a, P.b);
            }
            if (!full) flush();
        }
        // A clean candidate with the digits of a stored clean value is that value only if tier 3 says so at PMAX
        // bits. If it differs, it is a value of its own (stored, unless it is another stored value with these digits).
        std::vector<int> cres;
        for (size_t off = 0; off < conf.size() && !full; off += BLOCK) {
            size_t cnt = std::min(BLOCK, conf.size() - off);
            res.resize(cnt); cres.resize(cnt);
#pragma omp parallel for schedule(dynamic, 64)
            for (long long k = 0; k < (long long)cnt; k++) {
                const Conf& X = conf[off + k];
                cres[k] = t3_same(n, X.i, X.a, X.b, X.l, X.k, res[k]);
            }
            for (size_t k = 0; k < cnt && !full; k++) {
                const Conf& X = conf[off + k];
                if (cres[k] == 1) { st.dup++; continue; }
                if (cres[k] == 2) { st.dup++; st.conffail++; continue; }
                Pend P{ X.a, X.b, X.i, 0 };
                Cand c;
                bool range, isS;
                from_t3(P, res[k], c, range, isS, false);
                if (range) { st.range++; continue; }
                if (isS) { c.fl |= F_S; push(c, X.i, X.a, X.b); continue; }
                Quant Q;
                quant(c.v, nullptr, Q);
                int cls = c.fl & CLSMASK;
                std::vector<std::pair<int, uint32_t>> ms;
                auto collect = [&](int l, uint32_t kk) -> bool {
                    const Level& Y = lev[l];
                    uint8_t fe = Y.fl[kk];
                    if (!(fe & F_S) && (fe & CLSMASK) == cls && same(c.v, c.r, Y.v[kk], &Y.rV[(size_t)kk * NP])) ms.push_back({ l, kk });
                    return false;
                };
                int nm = nearmask(Q, 0);
                for (int var = 0; var < (1 << (4 * NP)); var++)
                    if ((var & nm) == var) ix.find(qhash(Q, cls, 0, var), collect);
                bool found = false;
                for (auto& m : ms) {
                    if (m.first == X.l && m.second == X.k) continue;
                    T3Res tmp;
                    if (t3_same(n, X.i, X.a, X.b, m.first, m.second, tmp) == 1) { found = true; break; }
                }
                if (found) { st.dup++; continue; }
                flush();
                store(n, c, Q, X.i, X.a, X.b);
                st.newv++; st.twins++;
                if (ix.n > hlimit) full = true;
            }
            if (!full) flush();
        }
        if (full) {
            printf("index full at n=%d: this level is incomplete and is left out (raise --hbits above %d)\n", n, hbits);
            Ln = Level();
            N = n - 1;
            break;
        }

        // audit: tier-2 values against tier 3
        if (!auditlist.empty()) {
            size_t cnt = auditlist.size();
            std::vector<T3Res> ar(cnt);
#pragma omp parallel for schedule(dynamic, 64)
            for (long long k = 0; k < (long long)cnt; k++) {
                const Pend& P = auditlist[k];
                ar[k] = t3_candidate(n, P.i, P.a, P.b, false);
            }
            for (size_t k = 0; k < cnt; k++) {
                audit_n++;
                if (ar[k].status != T_OK || (ar[k].fl & F_PRES)) { audit_unc++; continue; }
                for (int p = 0; p < NP; p++) {
                    const C& x = auditval[k].z[p];
                    const C& y = ar[k].v.z[p];
                    double d = modulus2((double)(x.re - y.re), PI_D * (double)(x.y - y.y));
                    double ratio = d / (ULP * modulus(y) * std::max(1.0f, auditr[k * NP + p]));
                    if (ratio > audit_worst) audit_worst = ratio;
                    if (ratio > 1.0) audit_bad++;
                }
            }
            auditlist.clear(); auditval.clear(); auditr.clear();
        }

        size_t reals = 0, clean = 0;
        for (size_t k = 0; k < Ln.size(); k++) {
            if (Ln.fl[k] & F_S) continue;
            clean++;
            bool r = true;
            for (int p = 0; p < NP; p++) if (!(((Ln.fl[k] >> (2 * p)) & 2) && Ln.v[k].z[p].y == 0)) r = false;
            reals += r;
        }
        cum += clean;
        check_targets(n);
        if (n < N) { finish_level(n); if (n == MINF_LEV) add_minf(); }
        double t2 = now();
        char t3s[64];
        snprintf(t3s, sizeof t3s, "%llu/%llu/%llu/%llu", st.p_inacc, st.p_zero, st.p_comp, st.p_s);
        printf("%3d %3d  %-11llu  %-9llu  %-9llu  %-8llu  %-8llu  %-28s  %-8llu  %-11llu  %-9llu  %-10llu  %-9llu  %-11llu  %-6.4f  %-6.3f  %.1f+%.1f  %.1f\n",
               n, 2 * n - 1, st.pairs, st.log0, st.tower, st.range, st.zeros, t3s, st.pres, (unsigned long long)clean, st.snew, st.dup,
               st.sdup + st.sdrop, cum, clean ? (double)reals / clean : 0.0, (double)clean / prev, t1 - t0, t2 - t1, mem_gb());
        if (st.deep || st.fail)
            printf("         n=%d: tier 3 could not resolve %llu (kept as S), failed %llu\n", n, st.deep, st.fail);
        tot_near += st.nearb; tot_resur += st.resur; tot_twins += st.twins; tot_conffail += st.conffail;
        fflush(stdout);
        prev = (double)clean;
    }
    printf("table build: %.1f s;  tier 3: %llu evaluations at %ld bits, %llu at %ld bits (%llu raised from the lower), %llu nodes\n", now() - t00,
           t3_lo.load(), (long)P1, t3_hi.load(), (long)PMAX, t3_esc.load(), t3_nodes.load());
    printf("consistency of tier 3 with the record: %llu exact parts not confirmed, %llu parts of no certain sign%s\n", t3_inc1.load(), t3_inc2.load(),
           (t3_inc1.load() || t3_inc2.load()) ? "   <-- LOOK AT THIS" : "");
    printf("%llu candidates were next to a cell border (neighbouring cells searched); %llu values were an ingredient come back (kept as S)\n", tot_near, tot_resur);
    if (CONFIRM)
        printf("%llu clean candidates had the digits of a stored clean value and were compared with it at %ld bits: %llu differ and are values of their own%s\n",
               t3_conf.load(), (long)PMAX, tot_twins, tot_conffail ? " (some could not be compared, see below)" : "");
    if (tot_conffail) printf("   %llu comparisons failed in tier 3; those candidates were taken as duplicates\n", tot_conffail);
    if (audit > 0)
        printf("audit: %llu tier-2 values recomputed in tier 3: %llu outside their error bound, %llu where tier 3 was not certain; largest error / bound = %.3f\n",
               audit_n, audit_bad, audit_unc, audit_worst);

    printf("\nShortest forms inside the table (n <= %d). 'agree': distance from the target at %ld bits.\n", N, (long)PMAX);
    for (auto& T : targets) {
        if (T.n_clean) {
            std::string code;
            rpn(T.n_clean, T.i_clean, code);
            int status;
            double lg = entry_distance(T.n_clean, T.i_clean, T, status);
            uint8_t fl = lev[T.n_clean].fl[T.i_clean];
            char ag[48];
            if (status == T_OK || status == T_DEEP) snprintf(ag, sizeof ag, lg < -1e8 ? "0" : "1e%.0f", lg); else snprintf(ag, sizeof ag, "tier 3 failed");
            printf("  %-9s leaves=%2d K=%2d  agree=%-8s %s%s %s\n", T.name.c_str(), T.n_clean, 2 * T.n_clean - 1, ag,
                   (fl & F_PRES) ? "presumed " : "", (fl & F_T3) ? "t3" : "t2", code.c_str());
        } else printf("  %-9s not found: K > %d\n", T.name.c_str(), 2 * N - 1);
        if (T.n_near && (!T.n_clean || T.n_near < T.n_clean)) {
            std::string code;
            rpn(T.n_near, T.i_near, code);
            printf("      a clean term at leaves=%2d K=%2d has the digits of the target but differs from it by 1e%.0f  %s\n", T.n_near, 2 * T.n_near - 1, T.lg_near, code.c_str());
        }
        if (T.n_s && (!T.n_clean || T.n_s < T.n_clean)) {
            std::string code;
            rpn(T.n_s, T.i_s, code);
            int status;
            double lg = entry_distance(T.n_s, T.i_s, T, status);
            if (status == T_ESC || status == T_FAIL) printf("      S entry with the same digits at leaves=%2d K=%2d (tier 3 failed)  %s\n", T.n_s, 2 * T.n_s - 1, code.c_str());
            else if (lg < -1e8 || lg < -(double)PMAX * 0.30103 + 40)
                printf("      S entry with the same digits at leaves=%2d K=%2d, NOT told apart from the target at %ld bits%s  %s\n", T.n_s, 2 * T.n_s - 1, (long)PMAX,
                       status == T_DEEP ? " (structure below that)" : "", code.c_str());
            else printf("      impostor at leaves=%2d K=%2d: differs from the target by 1e%.0f  %s\n", T.n_s, 2 * T.n_s - 1, lg, code.c_str());
        }
    }

    if (M_NMAX > 0 || M_ESAMPLE > 0) {
        if (M_NT <= 0 || M_NT > N) M_NT = N;
        if (M_EXACT && NP != 1) { fprintf(stderr, "--exact is for constants (NP=1) so far\n"); return 1; }
        if (M_F32 && !(M_EXACT && M_CLEAN)) { fprintf(stderr, "--f32 is for --exact --clean\n"); return 1; }
        std::vector<uint64_t>().swap(ix.t);              // the index of the table build is not needed any more
        finish_level(N);
        if (N == MINF_LEV) add_minf();
        if (INF_MODE && (M_NT < MINF_LEV || M_EXACT || M_CLEAN || M_F32)) {
            fprintf(stderr, "--inf with the search above the table: the table must have at least %d leaves (--nt), and --exact, --clean, --f32 are not for it\n", MINF_LEV);
            return 1;
        }
        mstat.assign(threads, MStat());
        cand_tl.assign(threads, std::vector<MCand>());
        zq_tl.assign(threads, std::vector<ZQ>());
        t3keep_tl.assign(threads, std::vector<T3Out>());
        mpcache.resize(threads);
        mitm_arrays(M_NT);
        if (M_ESAMPLE > 0) mitm_etalon(N, M_ESAMPLE);
        if (M_NMAX > 0) mitm_targets(M_NMAX);
    }

    if (!dumpfile.empty()) {
        FILE* f = fopen(dumpfile.c_str(), "w");
        if (!f) { fprintf(stderr, "cannot write %s\n", dumpfile.c_str()); return 1; }
        char b1[64], b2[64];
        for (int n = 1; n <= N; n++) {
            const Level& L = lev[n];
            for (size_t k = 0; k < L.size(); k++) {
                if (n == MINF_LEV && (uint32_t)k == MINF_IDX) continue;
                std::string code;
                rpn(n, (uint32_t)k, code);
                fprintf(f, "%d\t%s%s%s\t%s", n, (L.fl[k] & F_S) ? "S" : "c", (L.fl[k] & F_T3) ? "3" : "2", (L.fl[k] & F_PRES) ? "p" : "", code.c_str());
                for (int p = 0; p < NP; p++) {
                    quadmath_snprintf(b1, sizeof b1, "%.35Qe", L.v[k].z[p].re);
                    quadmath_snprintf(b2, sizeof b2, "%.35Qe", PIq * L.v[k].z[p].y);
                    fprintf(f, "\t%s\t%s", b1, b2);
                }
                fprintf(f, "\n");
            }
        }
        fclose(f);
        printf("\nall stored values written to %s\n", dumpfile.c_str());
    }
    return 0;
}
