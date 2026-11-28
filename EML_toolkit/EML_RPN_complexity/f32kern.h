// f32kern.h - the terms run forward in single precision: a sieve for the requests exp(exp(X)) = c of the
// exhaustive search (see mitm.inc). One source for the processor (reference) and for the graphics card (CUDA).
//
// What it promises: every term of the class whose exp can meet a request is reported, either as that term (FLAG)
// or inside a part of the tree that is handed back whole (SUB: the node and all above it; SUB0, SUB1: the terms
// that have the node under Log / under exp). What is reported is decided afterwards in double precision and above
// by the same code as without the sieve. The sieve never decides that a term is a solution; it only says
// "certainly not" for the rest. Every number carries a bound on its error; where single precision cannot give a
// number (range, cancellation, a cut within the error) the case is handed back.
//
// A term run forward: a pair of table entries (exp(a) - Log(b)), then entries put on top one at a time
// (side 0: exp(w) - Log(node); side 1: exp(node) - Log(w)).
//
// Numbers. Values and Logs are kept as two floats (h + l, "ff": about 14 digits when what went in has them); the
// table gives exp, Log and Log(exp) of its entries with 14 digits. Functions are single precision. EML terms make
// small numbers as x - (x - eps) and exp(eps) - 1 all the time, and single precision alone loses them; two rules
// keep them:
//   Log(X - Y) = Log(X) + log1p(-Y/X) when Y is small against X, with Log(X) known well (Log(exp(V)) = V of the
//   table, or the node below for X = exp(node)); and exp(v) = 1 + expm1(v) for small v.
#pragma once
#include <stdint.h>
#include <math.h>
#ifndef F32_FN
#define F32_FN static inline
#endif
#ifndef F32_COUNT
#define F32_COUNT(x)
#endif
#ifndef F32_WHY
#define F32_WHY(x)
#endif
#ifndef F32_DBG
#define F32_DBG(x)
#endif
#ifndef F32_FN_MEMBER
#define F32_FN_MEMBER
#endif

#define F32_MAXD 14
struct F32f { float h, l; };
enum { F32_A_OK = 1, F32_A_EE = 2, F32_A_HUGE = 4, F32_A_TINY = 8, F32_M_RZ = 16, F32_M_YH = 32 };      // exp side of an entry (y2 of the mark in bits 8..23)
enum { F32_B_OK = 1, F32_B_IV = 2, F32_B_LZ = 4 };                                                  // Log side (marks as above)
struct F32A { F32f Er, Ei, LEr, LEi; float EEr, EEi; uint32_t meta; };      // exp(V); Log(exp(V)) (V with Im in (-pi, pi]); exp(exp(V))
struct F32B { F32f Lr, Li; float IVr, IVi, lnL; uint32_t meta; };           // Log(V); 1/V; ln of the size of Log(V)
struct F32W { F32A a; F32B b; uint32_t u; };                                // an entry put on top; u: its leaves
enum { F32_FLAG = 0, F32_SUB = 1, F32_SUB0 = 2, F32_SUB1 = 3 };
struct F32Rec { uint32_t a, b; uint8_t s, i, nf, type; uint32_t st[F32_MAXD]; };      // st: (entry on top << 1) | side

struct F32G {
    int fmax, UW;                      // the largest size run; the largest level of the entries put on top
    const F32W* W;
    int wcnt[24];                      // wcnt[r]: number of entries on top with at most r leaves
    const float* rqre;                 // the requests by the size of X, sorted by the real part: exp(X) = (re, im) modulo 2 pi i
    const float* rqim;
    int rqoff[72], rqn[72];
    float relmax, emax;                // largest relative error of a value whose Log is taken; of an exp that is used
    int flags;                         // (experiments) 1: no function in double precision; 2: no request looked up below the top size
    int unitfast;                      // the entry 1 (first of the list) can be put on top of a node of the size fmax - 1 by a product alone
};

enum { F32_E_NONE = 0, F32_E_VALID = 1, F32_E_HUGE = 2, F32_E_TINY = 3, F32_E_BAD = 4 };
struct F32N {                          // a node: what its children need
    F32f vr, vi;                       // the value
    float ea, er;                      // bound on the size of its error; the same relative to the size of v
    float Evr, Evi, dE, aE;            // exp(v) (its real part is 1 + Evm when one1), bound on the size of its error, its size
    float Evm;
    F32f Lr, Li;                       // Log(v)
    float eL;
    float ir, ii, eI;                  // 1/v, relative error
    float EEr, EEi, eEE;               // exp(exp(v)), relative error
    F32f Wr, Wi;                       // v with the imaginary part reduced to (-pi, pi]: Log(exp(v))
    int it, nw;
    int y2, ly2;                       // marks: Im v = y2*pi/2 exactly (mk & 2); Im Log(v) = ly2*pi/2 exactly (lmk & 2)
    uint8_t c, mk, lmk, okE, okEE, ok0, ncut, logform, one1, hasL;      // mk & 1: Re v = 0 exactly; hasL: Log(v) was made
};
struct F32M {                          // how a node was made, v = X - Y (lk): Log(X) well known (LX, error eLX), X and Y in single precision
    F32f LXr, LXi;
    float eLX, Xr, Xi, relX, Yr, Yi, eY;
    float eEg;                         // (given exp) its relative error
    uint8_t lk;
};

// ---- two floats
F32_FN F32f f2_two(float a, float b) { F32f r; r.h = a + b; const float bb = r.h - a; r.l = (a - (r.h - bb)) + (b - bb); return r; }
F32_FN F32f f2_fast(float a, float b) { F32f r; r.h = a + b; r.l = b - (r.h - a); return r; }
F32_FN F32f f2_add(F32f a, F32f b)
{
    F32f s = f2_two(a.h, b.h);
    const F32f t = f2_two(a.l, b.l);
    s.l += t.h; s = f2_fast(s.h, s.l);
    s.l += t.l;
    return f2_fast(s.h, s.l);
}
F32_FN F32f f2_neg(F32f a) { F32f r; r.h = -a.h; r.l = -a.l; return r; }
F32_FN F32f f2_sub(F32f a, F32f b) { return f2_add(a, f2_neg(b)); }
F32_FN F32f f2_addf(F32f a, float b) { F32f s = f2_two(a.h, b); s.l += a.l; return f2_fast(s.h, s.l); }
F32_FN F32f f2_mulk(float k, F32f a) { const float p = k * a.h; const float e = fmaf(k, a.h, -p) + k * a.l; return f2_fast(p, e); }      // k a small whole number
F32_FN F32f f2_of(float a) { F32f r; r.h = a; r.l = 0.0f; return r; }
F32_FN F32f f2_d(double d) { F32f r; r.h = (float)d; r.l = (float)(d - (double)r.h); return r; }
F32_FN float f2_val(F32f a) { return a.h + a.l; }
F32_FN F32f f2_const_pi() { F32f r; r.h = 3.1415927f; r.l = -8.742278e-8f; return r; }
F32_FN F32f f2_const_hpi() { F32f r; r.h = 1.5707964f; r.l = -4.371139e-8f; return r; }
F32_FN F32f f2_const_2pi() { F32f r; r.h = 6.2831855f; r.l = -1.7484555e-7f; return r; }

// marks of exp of a number with the marks (mk, y2): 2: imaginary part exactly 0; 1: real part exactly 0
F32_FN int f32_mexp(int mk, int y2) { if (!(mk & 2)) return 0; return (y2 & 1) ? 1 : 2; }

// Is there a request of size c within tol of (kr, ki modulo 2 pi)?
F32_FN bool f32_match(const F32G& g, int c, float kr, float ki, float tol)
{
    const int n = g.rqn[c];
    if (n == 0) return false;
    if (!(tol < 1e30f) || !(fabsf(kr) < 1e30f) || !(fabsf(ki) < 1e30f)) return true;      // not a number to compare: not excluded
    const float* re = g.rqre + g.rqoff[c];
    const float* im = g.rqim + g.rqoff[c];
    const float lo = kr - tol, hi = kr + tol;
    int l = 0, r = n;
    while (l < r) { const int m = (l + r) >> 1; if (re[m] < lo) l = m + 1; else r = m; }
    for (int j = l; j < n && re[j] <= hi; j++) {
        if (tol >= 3.0f) return true;
        float d = ki - im[j];
        d -= 6.2831855f * rintf(d * 0.15915494f);
        if (fabsf(d) <= tol) return true;
    }
    return false;
}

// exp of a term of size c given as the product x*y (relative error rel of the factors together): can it be a request?
// (the answers are reasons, for the counts; 0: no)
F32_FN int f32_top_mul(const F32G& g, int c, float xr, float xi, float yr, float yi, float rel)
{
    const float EPS = 1.1920929e-7f;
    const float mm = (fabsf(xr) + fabsf(xi)) * (fabsf(yr) + fabsf(yi));
    if (!(mm < 1e37f)) return 4;
    const float kr = xr * yr - xi * yi, ki = xr * yi + xi * yr;
    const float tol = (rel + 4.0f * EPS) * mm + 1e-36f;
    if (fabsf(kr) - tol > 1500.0f) return 0;
    return f32_match(g, c, kr, ki, tol) ? 3 : 0;
}

// exp of a term of size c with the value p (error bound ea; em: 2 exp(p) is real for certain, 1 imaginary): can it be a request?
F32_FN int f32_top_val(const F32G& g, int c, float pr, float pi, float ea, int em)
{
    const float EPS = 1.1920929e-7f;
    if (g.rqn[c] == 0) return 0;
    if (!(fabsf(pr) < 1e30f) || !(fabsf(pi) < 1e30f) || !(ea < 1e30f)) return 5;
    if (pr - ea > 700.0f || pr + ea < -700.0f) return 0;      // exp(p) is beyond the range of the search, or below every request
    if (em == 2 && pr - ea > 7.4f) return 0;                  // real and above 1500
    if (!(ea < 0.25f)) return 5;                              // the phase is not known
    if (pr - ea > 7.4f) {
        // |exp(p)| > 1600: a request (real part below 1500) only if exp(p) is close to imaginary
        const float cs = fabsf(cosf(pi)) - ea - 4.0f * EPS * (1.0f + fabsf(pi));
        if (em != 1 && cs > 0.0f && (pr - ea) + logf(cs) > 7.33f) return 0;
        return 6;
    }
    const float m = expf(pr);
    float kr = m * cosf(pi), ki = m * sinf(pi);
    if (em == 2) ki = 0.0f;
    if (em == 1) kr = 0.0f;
    const float tol = (ea * (1.0f + ea) + 4.0f * EPS) * m + 4.0f * EPS * fabsf(ki) + 1e-36f;
    if (fabsf(kr) - tol > 1500.0f) return 0;
    return f32_match(g, c, kr, ki, tol) ? 7 : 0;
}

// A node whose value (vr, vi, ea, marks, c; how it was made if lk) is set, and exp(v) if haveE (Evr, Evi, eEg):
// its own request, then what its children need. false: nothing above it is run here.
template <class Out>
F32_FN bool f32_enter(const F32G& g, F32N& C, const F32M& M, bool haveE, Out& out, const uint32_t* st, int nf)
{
    const float EPS = 1.1920929e-7f, ULP2 = 1.4210855e-14f, PI = 3.1415927f;
    const int r = g.fmax - (int)C.c;
    C.it = 0; C.okEE = 0;
    C.nw = r > 0 ? g.wcnt[r < g.UW ? r : g.UW] : 0;
    F32_COUNT(out.nodes++);
    if (C.logform) {                                   // the value is exp(A) beyond the range of the arithmetic; Lr, Li, eL, lmk, ly2, ncut are set
        C.okE = F32_E_NONE; C.ok0 = 1; C.hasL = 1;
        return r > 0;
    }
    C.ok0 = 0; C.ncut = 1; C.lmk = 0; C.ly2 = 0; C.one1 = 0; C.hasL = 0;
    if (C.mk & 2) C.vi = (C.y2 == 0) ? f2_of(0.0f) : f2_mulk((float)C.y2, f2_const_hpi());
    if (C.mk & 1) C.vr = f2_of(0.0f);
    const float vr = f2_val(C.vr), vi = f2_val(C.vi);
    const float ar = fabsf(vr), ai = fabsf(vi), mv = fmaxf(ar, ai);
    if (!(mv >= 1e-28f && mv <= 1e30f)) {
        if (mv == 0.0f && (C.mk & 1) && (C.mk & 2) && C.y2 == 0) return false;      // the value 0: a value of a smaller level
        F32_WHY(out.why[3]++);
        out.rec(F32_SUB, nf, st);
        return false;
    }
    const float tq = (ar > ai) ? ai / ar : ar / ai;
    const float av = mv * sqrtf(1.0f + tq * tq);
    const int em = f32_mexp(C.mk, C.y2);
    // ---- exp(v): given (a product from below), or from v; for small v as 1 + expm1(v)
    C.okE = F32_E_NONE;
    float dEg = 3e38f;
    if (haveE) {
        const float x = fabsf(C.Evr), y = fabsf(C.Evi), m = fmaxf(x, y);
        if (m >= 1e-30f && m <= 1e30f && M.eEg < 0.25f) {
            C.aE = 1.0000005f * ((x > y) ? x * sqrtf(1.0f + (y / x) * (y / x)) : y * sqrtf(1.0f + (x / y) * (x / y)));
            dEg = M.eEg * C.aE;
            C.okE = F32_E_VALID;
        }
    }
    if (vr > 69.0f) { if (C.okE != F32_E_VALID) C.okE = F32_E_HUGE; }
    else if (vr < -69.0f) { if (C.okE != F32_E_VALID) C.okE = F32_E_TINY; }
    else if (mv <= 0.125f) {
        const float dEs = 1.2f * C.ea + 12.0f * EPS * mv;
        if (C.okE != F32_E_VALID || dEs < 0.5f * dEg) {
            const float ex1 = expm1f(vr), sy = sinf(vi), sh = sinf(0.5f * vi), c1 = -2.0f * sh * sh;
            C.Evm = ex1 + c1 + ex1 * c1;
            C.Evr = 1.0f + C.Evm; C.Evi = (1.0f + ex1) * sy;
            C.one1 = 1;
            C.dE = dEs; C.aE = 1.14f;
            C.okE = F32_E_VALID;
            dEg = -1.0f;
        }
    } else {
        const float dEd0 = C.ea * (1.0f + C.ea) + 3.0f * EPS;
        if (C.okE != F32_E_VALID || M.eEg > 2.0f * dEd0) {
            if (dEd0 < 0.25f) {
                const float m = expf(C.vr.h), cs = cosf(C.vi.h), sn = sinf(C.vi.h);
                const float e0r = m * cs, e0i = m * sn;
                C.Evr = e0r + (e0r * C.vr.l - e0i * C.vi.l); C.Evi = e0i + (e0i * C.vr.l + e0r * C.vi.l);
                const float x = fabsf(C.Evr), y = fabsf(C.Evi);
                C.aE = 1.0000005f * ((x > y) ? x * sqrtf(1.0f + (y / x) * (y / x)) : (y > 0.0f ? y * sqrtf(1.0f + (x / y) * (x / y)) : 0.0f));
                C.dE = dEd0 * C.aE;
                C.okE = F32_E_VALID;
                dEg = -1.0f;
            } else if (C.okE != F32_E_VALID) C.okE = F32_E_BAD;
        }
    }
    if (C.okE == F32_E_VALID) {
        if (dEg >= 0.0f) C.dE = dEg;
        if (em == 2) C.Evi = 0.0f;
        if (em == 1) { C.Evr = 0.0f; C.one1 = 0; }
    }
    // ---- its own request
    if (g.rqn[C.c] > 0 && !((g.flags & 2) && r > 0)) {
        if (C.okE == F32_E_VALID) {
            const float tol = C.dE + 4.0f * EPS * (fabsf(C.Evr) + fabsf(C.Evi)) + 1e-36f;
            if (!(fabsf(C.Evr) - tol > 1500.0f) && f32_match(g, C.c, C.Evr, C.Evi, tol)) { F32_WHY(out.fwhy[0]++); out.rec(F32_FLAG, nf, st); }
        } else if (C.okE == F32_E_BAD) { F32_WHY(out.fwhy[1]++); out.rec(F32_FLAG, nf, st); }
        else if (C.okE == F32_E_HUGE) {
            // |exp(v)| > e^69: a request only if exp(v) is within the range of the search and imaginary to more than 26 digits
            if (!(vr - C.ea > 700.0f) && (em == 1 || (em != 2 && fabsf(cosf(vi)) <= 4.0f * C.ea + 1e-5f))) { F32_WHY(out.fwhy[2]++); out.rec(F32_FLAG, nf, st); }
        }
    }
    if (r <= 0) return false;
    // ---- for the children that take Log(v): Log(v), 1/v
    const float er = C.ea / av * 1.0000005f;
    C.er = er;
    if (er > g.relmax) { F32_WHY(out.why[0]++); F32_DBG(out.dbg(0, C, er, nf, st)); out.rec(F32_SUB0, nf, st); }
    else if (r == 1 && g.unitfast) {
        // one leaf to go: only the entry 1 comes on top, and that test is a product with 1/v; Log(v) is not needed
        C.ok0 = 1;
        const float s = 1.0f / mv, xr = vr * s, xi = vi * s, dd = xr * xr + xi * xi;
        C.ir = xr / dd * s; C.ii = -xi / dd * s;
        C.eI = er * (1.0f + 2.0f * er) + 3.0f * EPS;
    } else {
        C.ok0 = 1; C.hasL = 1;
        const bool real = (C.mk & 2) && C.y2 == 0, imag = (C.mk & 1) != 0;
        if (!real && !imag && vr < 0.0f && ai <= 4.0f * C.ea) C.ncut = 2;      // on the cut within the error: both sides are run
        const float sX = fmaxf(fabsf(M.Xr), fabsf(M.Xi)), sY = fmaxf(fabsf(M.Yr), fabsf(M.Yi));
        if (M.lk && sX >= 1e-30f && sX <= 1e30f && sY <= 0.25f * sX) {
            // Log(X - Y) = Log(X) + log1p(-Y/X)
            const float s = 1.0f / sX, xr = M.Xr * s, xi = M.Xi * s, dd = xr * xr + xi * xi;
            const float qr = (M.Yr * xr + M.Yi * xi) / dd * s, qi = (M.Yi * xr - M.Yr * xi) / dd * s;
            const float aq = fabsf(qr) + fabsf(qi);
            const float l1r = 0.5f * log1pf(-2.0f * qr + qr * qr + qi * qi), l1i = atan2f(-qi, 1.0f - qr);
            const float ex = M.eY * s + (M.relX + 4.0f * EPS) * aq;
            C.Lr = f2_addf(M.LXr, l1r); C.Li = f2_addf(M.LXi, l1i);
            // the sum is Log(v) up to a multiple of 2 pi i; the principal value has the sign of Im v (which is known:
            // v is off the real axis by more than its error, or the case is one of those below)
            const float lif = f2_val(C.Li);
            if (vi > 0.0f && lif < -1.0f) C.Li = f2_add(C.Li, f2_const_2pi());
            else if (vi < 0.0f && lif > 1.0f) C.Li = f2_sub(C.Li, f2_const_2pi());
            else if (vi == 0.0f && fabsf(lif) > 4.0f) C.Li = (lif > 0.0f) ? f2_sub(C.Li, f2_const_2pi()) : f2_add(C.Li, f2_const_2pi());
            C.eL = 1.6f * ex + 2.0f * EPS * (fabsf(l1r) + fabsf(l1i)) + M.eLX + ULP2 * (fabsf(M.LXr.h) + fabsf(M.LXi.h) + 8.0f);
        } else {
            const float la = logf(mv), lb = 0.5f * log1pf(tq * tq);
            C.Lr = f2_of(la + lb);
            C.Li = f2_of(atan2f(vi, vr));
            C.eL = er * (1.0f + 2.0f * er) + 2.0f * EPS * (fabsf(la) + lb + fabsf(C.Li.h)) + 1e-37f;
        }
        if (fabsf(vr - 1.0f) <= 0.25f && ai <= 0.25f) {
            // v is near 1: Log(v) = log1p(v - 1), and v - 1 is exact in two floats
            const float z1 = f2_val(f2_addf(C.vr, -1.0f)), z2 = vi;
            const float l1r = 0.5f * log1pf(2.0f * z1 + z1 * z1 + z2 * z2), l1i = atan2f(z2, 1.0f + z1);
            const float e1 = 1.4f * C.ea + 3.0f * EPS * (fabsf(l1r) + fabsf(l1i)) + 1e-37f;
            if (e1 < C.eL) { C.Lr = f2_of(l1r); C.Li = f2_of(l1i); C.eL = e1; }
        }
        if (real) { C.Li = vr > 0.0f ? f2_of(0.0f) : f2_const_pi(); C.lmk = 2; C.ly2 = vr > 0.0f ? 0 : 2; }
        else if (imag) { C.Li = vi > 0.0f ? f2_const_hpi() : f2_neg(f2_const_hpi()); C.lmk = 2; C.ly2 = vi > 0.0f ? 1 : -1; }
        else if (C.ncut == 2) C.Li = f2_const_pi();
        const float s = 1.0f / mv, xr = vr * s, xi = vi * s, dd = xr * xr + xi * xi;
        C.ir = xr / dd * s; C.ii = -xi / dd * s;
        C.eI = er * (1.0f + 2.0f * er) + 3.0f * EPS;
    }
    // ---- for the children that take exp(v): exp(exp(v)), and v reduced (the Log of exp(v))
    if (C.okE == F32_E_BAD) { F32_WHY(out.why[1]++); out.rec(F32_SUB1, nf, st); C.okE = F32_E_NONE; }
    else if (C.okE == F32_E_VALID) {
        if (C.dE > g.emax * C.aE) { F32_WHY(out.why[2]++); F32_DBG(out.dbg(2, C, C.dE / C.aE, nf, st)); out.rec(F32_SUB1, nf, st); C.okE = F32_E_NONE; }
        else {
            C.eEE = C.dE * (1.0f + C.dE) + 3.0f * EPS;
            if (fabsf(C.Evr) <= 85.0f && C.eEE < 0.25f) {
                const float m = expf(C.Evr);
                C.EEr = m * cosf(C.Evi); C.EEi = m * sinf(C.Evi);
                if (em == 2) C.EEi = 0.0f;
                const float mm = fmaxf(fabsf(C.EEr), fabsf(C.EEi));
                C.okEE = (mm >= 1e-30f && mm <= 1e30f) ? 1 : 0;
            }
            if (r >= 2) {
                const float k = rintf(vi * 0.15915494f);
                C.Wr = C.vr;
                C.Wi = (k == 0.0f) ? C.vi : f2_sub(C.vi, f2_mulk(k, f2_const_2pi()));
            }
        }
    }
    return C.ok0 || C.okE != F32_E_NONE;
}

// One child of the node P: the entry w of the list put on top, on the side `side` (alt: Log(P) taken on the other
// side of the cut). 1: the child C is made and is to be entered; 0: nothing more to do for it (not a term of the
// class, or of the top size and tested, or handed back). stw: the step as it is written in a record.
template <class Out>
F32_FN int f32_step(const F32G& g, const F32N& P, int w, int side, int alt, int d, uint32_t stw, uint32_t* st, Out& out, F32N& C, F32M& M, bool& haveE)
{
    const float EPS = 1.1920929e-7f, ULP2 = 1.4210855e-14f, PI = 3.1415927f, HPI = 1.5707964f;
    const F32W& e = g.W[w];
    const int cc = (int)P.c + (int)e.u;
    const bool top = (cc == g.fmax);
    C.c = (uint8_t)cc; C.logform = 0; M.lk = 0;
    haveE = false;
    if (side == 0) {                                           // exp(w) - Log(P)
        if (!P.ok0) return 0;
        const uint32_t m = e.a.meta;
        if (!P.hasL) {                                         // one leaf to go: the entry 1 on top, by the product
            if (alt || !top || !(m & F32_A_OK) || !(m & F32_A_EE)) return 0;
            F32_COUNT(out.tops++);
            const int hit = f32_top_mul(g, cc, e.a.EEr, e.a.EEi, P.ir, P.ii, P.eI);
            if (hit) { F32_WHY(out.fwhy[hit]++); st[d] = stw; out.rec(F32_FLAG, d + 1, st); }
            return 0;
        }
        const F32f PLi = alt ? f2_neg(f2_const_pi()) : P.Li;
        const float Lrf = f2_val(P.Lr), Lif = f2_val(PLi);
        int mk = 0, y2 = 0;
        if ((m & F32_M_YH) && (P.lmk & 2)) { mk = 2; y2 = (int)(int16_t)(m >> 8) - P.ly2; }
        if (m & F32_A_OK) {
            const float sE = fmaxf(fabsf(e.a.Er.h), fabsf(e.a.Ei.h)), sL = fmaxf(fabsf(Lrf), fabsf(Lif));
            if (sL - P.eL > 0.0f && (sE * 3.1e-30f > sL + P.eL || (sL - P.eL) * 3.1e-30f > sE)) return 0;      // one below the resolution of the other, for certain
            if (top) {
                int hit;
                F32_COUNT(out.tops++);
                if ((m & F32_A_EE) && !P.logform) { if (alt) return 0; hit = f32_top_mul(g, cc, e.a.EEr, e.a.EEi, P.ir, P.ii, P.eI); }
                else hit = f32_top_val(g, cc, e.a.Er.h - Lrf, e.a.Ei.h - Lif, 2.0f * EPS * sE + P.eL + 1e-37f, f32_mexp(mk, y2));
                if (hit) { F32_WHY(out.fwhy[hit]++); st[d] = stw; out.rec(F32_FLAG, d + 1, st); }
                return 0;
            }
            C.vr = f2_sub(e.a.Er, P.Lr); C.vi = f2_sub(e.a.Ei, PLi);
            C.ea = P.eL + ULP2 * (sE + sL) + 1e-37f;
            C.mk = (uint8_t)mk; C.y2 = y2;
            M.lk = 1; M.LXr = e.a.LEr; M.LXi = e.a.LEi; M.eLX = 0.0f; M.Xr = e.a.Er.h; M.Xi = e.a.Ei.h; M.relX = 0.0f; M.Yr = Lrf; M.Yi = Lif; M.eY = P.eL;
            if (!(g.flags & 1) && C.ea > 0.25f * g.relmax * fmaxf(fabsf(C.vr.h), fabsf(C.vi.h)) && !P.logform && P.er < 0.125f * P.eL) {
                // the subtraction cancels and Log(P) was the weak part: Log(P) once more, in double precision, from the two floats of P
                const double pr = (double)P.vr.h + (double)P.vr.l, pi = (double)P.vi.h + (double)P.vi.l;
                const double lr = 0.5 * log(pr * pr + pi * pi);
                const F32f Lr2 = f2_d(lr);
                const F32f Li2 = ((P.lmk & 2) || P.ncut == 2) ? PLi : f2_d(atan2(pi, pr));
                const float eL2 = P.er * (1.0f + 2.0f * P.er) + 4e-15f * ((float)fabs(lr) + 4.0f);
                F32_WHY(out.why2[0]++);
                C.vr = f2_sub(e.a.Er, Lr2); C.vi = f2_sub(e.a.Ei, Li2);
                C.ea = eL2 + ULP2 * (sE + sL) + 1e-37f;
                M.Yr = f2_val(Lr2); M.Yi = f2_val(Li2); M.eY = eL2;
            }
            if ((m & F32_A_EE) && !P.logform) {
                haveE = true;
                C.Evr = e.a.EEr * P.ir - e.a.EEi * P.ii; C.Evi = e.a.EEr * P.ii + e.a.EEi * P.ir;
                M.eEg = P.eI + 3.0f * EPS;
            }
        } else if (m & (F32_A_HUGE | F32_A_TINY)) {
            // exp(w) is beyond the range of the arithmetic (its ln is LEr): below the resolution for certain, or left to double precision
            const float lnE = e.a.LEr.h, sL = fmaxf(fabsf(Lrf), fabsf(Lif));
            if (m & F32_A_HUGE) { if (sL - P.eL > 0.0f && lnE - logf(sL + P.eL) > 67.93f) return 0; }
            else if (sL - P.eL > 0.0f && logf(sL - P.eL) - lnE > 67.93f) return 0;
            F32_WHY(out.why[4]++);
            st[d] = stw; out.rec(F32_SUB, d + 1, st);
            return 0;
        } else return 0;
    } else {                                                   // exp(P) - Log(w)
        const uint32_t m = e.b.meta;
        if (!(m & F32_B_OK)) return 0;
        if (P.okE == F32_E_VALID) {
            const float sE = fmaxf(fabsf(P.Evr), fabsf(P.Evi)), sL = fmaxf(fabsf(e.b.Lr.h), fabsf(e.b.Li.h));
            if (!(m & F32_B_LZ) && (sE * 3.1e-30f > sL || sL * 3.1e-30f > sE)) return 0;
            const int em = f32_mexp(P.mk, P.y2);
            int mk = 0, y2 = 0;
            if (em == 2 && (m & F32_M_YH)) { mk |= 2; y2 = -(int)(int16_t)(m >> 8); }
            if (em == 1 && (m & F32_M_RZ)) mk |= 1;
            const float ap = P.dE + ULP2 * (sE + sL) + 1e-37f;
            if (top) {
                int hit;
                F32_COUNT(out.tops++);
                if (P.okEE && (m & F32_B_IV)) hit = f32_top_mul(g, cc, P.EEr, P.EEi, e.b.IVr, e.b.IVi, P.eEE);
                else hit = f32_top_val(g, cc, P.Evr - e.b.Lr.h, P.Evi - e.b.Li.h, ap + 2.0f * EPS * (sE + sL), f32_mexp(mk, y2));
                if (hit) { F32_WHY(out.fwhy[hit]++); st[d] = stw; out.rec(F32_FLAG, d + 1, st); }
                return 0;
            }
            // exp(P) as two floats: 1 + expm1 for small P, else the single value
            const F32f Er2 = P.one1 ? f2_two(1.0f, P.Evm) : f2_of(P.Evr);
            C.vr = f2_sub(Er2, e.b.Lr); C.vi = f2_sub(f2_of(P.Evi), e.b.Li);
            C.ea = ap;
            C.mk = (uint8_t)mk; C.y2 = y2;
            M.lk = 1; M.LXr = P.Wr; M.LXi = P.Wi; M.eLX = P.ea; M.Xr = P.Evr; M.Xi = P.Evi; M.relX = P.dE / P.aE; M.Yr = e.b.Lr.h; M.Yi = e.b.Li.h; M.eY = ULP2 * sL;
            if (!(g.flags & 1) && ap > 0.25f * g.relmax * fmaxf(fabsf(C.vr.h), fabsf(C.vi.h)) && !P.one1 && P.ea * P.aE < 0.125f * P.dE && fabsf(f2_val(P.vr)) < 80.0f) {
                // the subtraction cancels and exp(P) was the weak part: exp(P) once more, in double precision, from the two floats of P
                const double pr = (double)P.vr.h + (double)P.vr.l, pi = (double)P.vi.h + (double)P.vi.l;
                const double mm = exp(pr);
                double xr = mm * cos(pi), xi = mm * sin(pi);
                if (em == 2) xi = 0.0;
                if (em == 1) xr = 0.0;
                const float dE2 = (P.ea * (1.0f + P.ea) + 2e-15f) * P.aE * 1.001f;
                F32_WHY(out.why2[1]++);
                C.vr = f2_sub(f2_d(xr), e.b.Lr); C.vi = f2_sub(f2_d(xi), e.b.Li);
                C.ea = dE2 + ULP2 * (sE + sL) + 1e-37f;
                M.relX = dE2 / P.aE;
            }
            if (P.okEE && (m & F32_B_IV)) {
                haveE = true;
                C.Evr = P.EEr * e.b.IVr - P.EEi * e.b.IVi; C.Evi = P.EEr * e.b.IVi + P.EEi * e.b.IVr;
                M.eEg = P.eEE + 3.0f * EPS;
            }
        } else if (P.okE == F32_E_HUGE || P.okE == F32_E_TINY) {
            const float Pvr = f2_val(P.vr), Pvi = f2_val(P.vi);
            if (!(m & F32_B_LZ)) {
                if (P.okE == F32_E_HUGE) { if ((Pvr - P.ea) - e.b.lnL > 67.93f) return 0; }      // Log(w) is below the resolution of exp(P), for certain
                else if (e.b.lnL - (Pvr + P.ea) > 67.93f) return 0;                            // exp(P) is below the resolution of Log(w), for certain
                F32_WHY(out.why[5]++);
                st[d] = stw; out.rec(F32_SUB, d + 1, st);
                return 0;
            }
            // the child is exp(P) itself, beyond the range of the arithmetic: known by its Log, which is P reduced
            if (fabsf(Pvr) > 667.5f) { if (fabsf(Pvr) < 668.0f) { st[d] = stw; out.rec(F32_SUB, d + 1, st); } return 0; }
            const int em = f32_mexp(P.mk, P.y2);
            if (em == 1) { F32_WHY(out.why[6]++); st[d] = stw; out.rec(F32_SUB, d + 1, st); return 0; }             // imaginary and huge: its exp is not left to single precision
            float th = Pvi - 6.2831855f * rintf(Pvi * 0.15915494f);
            const float eth = 2.0f * P.ea + 4.0f * EPS * (fabsf(Pvi) + 4.0f);
            C.logform = 1; C.mk = 0; C.y2 = 0; C.lmk = 0; C.ly2 = 0; C.ncut = 1;
            if (em == 2) { C.lmk = 2; C.ly2 = (P.y2 & 3) == 0 ? 0 : 2; th = C.ly2 ? PI : 0.0f; }
            else {
                if (!(eth < 0.25f) || fabsf(fabsf(th) - HPI) <= eth + 1e-5f) { F32_WHY(out.why[6]++); st[d] = stw; out.rec(F32_SUB, d + 1, st); return 0; }
                if (fabsf(fabsf(th) - PI) <= eth) { C.ncut = 2; th = PI; }
            }
            C.Lr = f2_of(Pvr); C.Li = f2_of(th);
            C.eL = eth + 2.0f * EPS * fabsf(Pvr);
            if (top) return 0;                                                  // (its exp is no request)
        } else return 0;
    }
    return 1;
}

// A sink for the walk that only rebuilds a node (the second phase on the graphics card): nothing is reported.
struct F32Quiet {
    unsigned long long nodes, tops, why[8], fwhy[8], why2[2];
    F32_FN_MEMBER void rec(int, int, const uint32_t*) {}
};
struct F32NoUnits { F32_FN_MEMBER void unit(int, const uint32_t*) {} };

// All terms above the node S[0] (entered, with children to run). The node whose children are run is kept in P (in
// registers); the stack S is written when a child is entered and read when the walk comes back.
// ccut: a child of at least this size that has children of its own is not walked here: it is handed to `units` (the
// steps to it, with the side of the cut: ust) and walked by another thread. (No cut: ccut = 1000.)
template <class Out, class Units>
F32_FN void f32_tree(const F32G& g, F32N* S, uint32_t* st, uint32_t* ust, Out& out, int ccut, Units& units)
{
    const float PI = 3.1415927f;
    int d = 0;
    F32N P = S[0];
    for (;;) {
        // ---- the cheap steps (the end of a node's list, an entry that does not apply, a term of the top size tested
        // by a product), until a child has to be made. On the card the threads of a group leave this loop together,
        // so that the long step below is taken by all of them at once.
        int alt = 0, side = 0, w = 0;
        bool fin = false;
        for (;;) {
            const int two = (P.ncut == 2);
            if (P.it >= (two ? 4 : 2) * P.nw) {
                if (d == 0) { fin = true; break; }
                d--;
                P = S[d];
                continue;
            }
            const int it = P.it++;
            if (two) { alt = it & 1; side = (it >> 1) & 1; w = it >> 2; if (alt && side) continue; }
            else { alt = 0; side = it & 1; w = it >> 1; }
            const F32W& e = g.W[w];
            const int cc = (int)P.c + (int)e.u;
            const bool top = (cc == g.fmax);
            if (side == 0) {
                if (!P.ok0) continue;
                const uint32_t m = e.a.meta;
                if (!(m & (F32_A_OK | F32_A_HUGE | F32_A_TINY))) continue;
                if (top && (m & F32_A_OK) && (m & F32_A_EE) && !P.logform) {
                    if (P.hasL) {
                        const float Lrf = f2_val(P.Lr), Lif = alt ? -PI : f2_val(P.Li);
                        const float sE = fmaxf(fabsf(e.a.Er.h), fabsf(e.a.Ei.h)), sL = fmaxf(fabsf(Lrf), fabsf(Lif));
                        if (sL - P.eL > 0.0f && (sE * 3.1e-30f > sL + P.eL || (sL - P.eL) * 3.1e-30f > sE)) continue;
                    }
                    F32_COUNT(out.tops++);
                    if (alt) continue;
                    const int hit = f32_top_mul(g, cc, e.a.EEr, e.a.EEi, P.ir, P.ii, P.eI);
                    if (hit) { F32_WHY(out.fwhy[hit]++); st[d] = ((uint32_t)w << 1); out.rec(F32_FLAG, d + 1, st); }
                    continue;
                }
            } else {
                const uint32_t m = e.b.meta;
                if (!(m & F32_B_OK) || P.okE == F32_E_NONE) continue;
                if (top && P.okE == F32_E_VALID && P.okEE && (m & F32_B_IV)) {
                    const float sE = fmaxf(fabsf(P.Evr), fabsf(P.Evi)), sL = fmaxf(fabsf(e.b.Lr.h), fabsf(e.b.Li.h));
                    if (!(m & F32_B_LZ) && (sE * 3.1e-30f > sL || sL * 3.1e-30f > sE)) continue;
                    F32_COUNT(out.tops++);
                    const int hit = f32_top_mul(g, cc, P.EEr, P.EEi, e.b.IVr, e.b.IVi, P.eEE);
                    if (hit) { F32_WHY(out.fwhy[hit]++); st[d] = ((uint32_t)w << 1) | 1u; out.rec(F32_FLAG, d + 1, st); }
                    continue;
                }
            }
            break;
        }
        if (fin) break;
        // ---- the long step: the child is made (or a rare kind of test is done), and entered
        const uint32_t stw = ((uint32_t)w << 1) | (uint32_t)side;
        F32N C;
        F32M M;
        bool haveE;
        if (!f32_step(g, P, w, side, alt, d, stw, st, out, C, M, haveE)) continue;
        st[d] = stw; ust[d] = ((uint32_t)w << 2) | ((uint32_t)alt << 1) | (uint32_t)side;
        if (f32_enter(g, C, M, haveE, out, st, d + 1)) {
            if ((int)C.c >= ccut) { units.unit(d + 1, ust); continue; }
            if ((int)C.c == g.fmax - 1) {
                // one leaf to go: its children are of the top size and are only tested; done here, without the stack
                const int lim = (C.ncut == 2 ? 4 : 2) * C.nw;
                for (int i2 = 0; i2 < lim; i2++) {
                    int a2 = 0, s2, w2;
                    if (C.ncut == 2) { a2 = i2 & 1; s2 = (i2 >> 1) & 1; w2 = i2 >> 2; if (a2 && s2) continue; }
                    else { s2 = i2 & 1; w2 = i2 >> 1; }
                    F32N C2;
                    F32M M2;
                    bool h2;
                    f32_step(g, C, w2, s2, a2, d + 1, ((uint32_t)w2 << 1) | (uint32_t)s2, st, out, C2, M2, h2);
                }
                continue;
            }
            S[d].it = P.it;
            d++;
            S[d] = C;
            P = C;
        }
    }
}

// The pair (exp side A of an entry of level i, Log side B of an entry of level s-i) and everything above it.
// With build != 0 nothing is walked: the node of the pair is only rebuilt in S[0] (true if it has children).
template <class Out, class Units>
F32_FN bool f32_pair(const F32G& g, int s, const F32A& A, const F32B& B, F32N* S, uint32_t* st, uint32_t* ust, Out& out, int ccut, Units& units, int build)
{
    const float EPS = 1.1920929e-7f, ULP2 = 1.4210855e-14f, PI = 3.1415927f, HPI = 1.5707964f;
    const uint32_t ma = A.meta, mb = B.meta;
    int mk = 0, y2 = 0;
    if ((ma & F32_M_YH) && (mb & F32_M_YH)) { mk |= 2; y2 = (int)(int16_t)(ma >> 8) - (int)(int16_t)(mb >> 8); }
    if ((ma & F32_M_RZ) && (mb & F32_M_RZ)) mk |= 1;
    F32N C;
    F32M M;
    C.c = (uint8_t)s; C.logform = 0; M.lk = 0; C.mk = (uint8_t)mk; C.y2 = y2;
    const bool top = (s == g.fmax);
    if (ma & F32_A_OK) {
        const float sE = fmaxf(fabsf(A.Er.h), fabsf(A.Ei.h)), sL = fmaxf(fabsf(B.Lr.h), fabsf(B.Li.h));
        if (!(mb & F32_B_LZ) && (sE * 3.1e-30f > sL || sL * 3.1e-30f > sE)) return false;      // one below the resolution of the other, for certain
        const bool fast = (ma & F32_A_EE) && (mb & F32_B_IV);
        if (top) {
            int hit;
            F32_COUNT(out.tops++);
            if (fast) hit = f32_top_mul(g, s, A.EEr, A.EEi, B.IVr, B.IVi, 2.0f * EPS);
            else hit = f32_top_val(g, s, A.Er.h - B.Lr.h, A.Ei.h - B.Li.h, 2.0f * EPS * (sE + sL) + 1e-37f, f32_mexp(mk, y2));
            if (hit) { F32_WHY(out.fwhy[hit]++); out.rec(F32_FLAG, 0, st); }
            return false;
        }
        C.vr = f2_sub(A.Er, B.Lr); C.vi = f2_sub(A.Ei, B.Li);
        C.ea = ULP2 * (sE + sL) + 1e-37f;
        M.lk = 1; M.LXr = A.LEr; M.LXi = A.LEi; M.eLX = 0.0f; M.Xr = A.Er.h; M.Xi = A.Ei.h; M.relX = 0.0f; M.Yr = B.Lr.h; M.Yi = B.Li.h; M.eY = ULP2 * sL;
        if (fast) {
            C.Evr = A.EEr * B.IVr - A.EEi * B.IVi; C.Evi = A.EEr * B.IVi + A.EEi * B.IVr;
            M.eEg = 4.0f * EPS;
        }
        if (!f32_enter(g, C, M, fast, out, st, 0)) return false;
        S[0] = C;
        if (build) return true;
        if (s >= ccut) units.unit(0, ust); else f32_tree(g, S, st, ust, out, ccut, units);
        return true;
    }
    if (!(ma & (F32_A_HUGE | F32_A_TINY))) return false;
    // exp(a) is beyond the range of the arithmetic; LEr, LEi hold Log(exp(a))
    const float lnE = A.LEr.h, thE = A.LEi.h;
    if (!(mb & F32_B_LZ)) {
        if (ma & F32_A_HUGE) { if (lnE - B.lnL > 67.93f) return false; }
        else if (B.lnL - lnE > 67.93f) return false;
        F32_WHY(out.why[7]++);
        out.rec(F32_SUB, 0, st);                       // (rare) left to double precision
        return false;
    }
    // Log(b) = 0: the value is exp(a) itself, known by its Log
    if (fabsf(lnE) > 667.5f) { if (fabsf(lnE) < 668.0f) out.rec(F32_SUB, 0, st); return false; }
    if ((ma & F32_M_RZ) || (!(ma & F32_M_YH) && (fabsf(fabsf(thE) - HPI) < 1e-4f || fabsf(fabsf(thE) - PI) < 1e-4f))) { out.rec(F32_SUB, 0, st); return false; }
    if (top) return false;                                   // (its exp is no request: beyond every range, or 1 with structure below the resolution)
    C.logform = 1; C.mk = 0; C.y2 = 0; C.lmk = 0; C.ly2 = 0; C.ncut = 1;
    C.Lr = A.LEr; C.Li = A.LEi;
    C.eL = ULP2 * (fabsf(lnE) + 4.0f);
    if (ma & F32_M_YH) {
        const int y = (int)(int16_t)(ma >> 8);
        if (y == 0) { C.lmk = 2; C.ly2 = fabsf(thE) < 1.5f ? 0 : 2; C.Li = C.ly2 ? f2_const_pi() : f2_of(0.0f); }
        else C.Li = fabsf(thE) < 1.5f ? f2_of(0.0f) : (y > 0 ? f2_const_pi() : f2_neg(f2_const_pi()));      // exp(a) = R + i*y*pi/2 with |R| huge
    }
    if (!f32_enter(g, C, M, false, out, st, 0)) return false;
    S[0] = C;
    if (build) return true;
    if (s >= ccut) units.unit(0, ust); else f32_tree(g, S, st, ust, out, ccut, units);
    return true;
}

// (second phase) The node at the end of the steps ust[0..k-1] above the pair, rebuilt in S[0] without a word; then
// everything above it. false: it could not be rebuilt (cannot be: the first phase made it the same way).
template <class Out>
F32_FN bool f32_unit(const F32G& g, int s, const F32A& A, const F32B& B, int k, const uint32_t* upath, F32N* S, uint32_t* st, uint32_t* ust, Out& out)
{
    F32Quiet q = F32Quiet();
    F32NoUnits nu;
    if (!f32_pair(g, s, A, B, S, st, ust, q, 1000, nu, 1)) return false;
    F32N P = S[0];
    for (int t = 0; t < k; t++) {
        const uint32_t u = upath[t];
        const int w = (int)(u >> 2), alt = (int)((u >> 1) & 1), side = (int)(u & 1);
        const uint32_t stw = ((uint32_t)w << 1) | (uint32_t)side;
        F32N C;
        F32M M;
        bool haveE;
        if (!f32_step(g, P, w, side, alt, t, stw, st, q, C, M, haveE)) return false;
        st[t] = stw; ust[t] = u;
        if (!f32_enter(g, C, M, haveE, q, st, t + 1)) return false;
        P = C;
    }
    // the walk above it; the steps so far stay in front (out.npre: a record carries the whole path)
    S[0] = P;
    F32NoUnits nu2;
    out.npre = k;
    f32_tree(g, S, st + k, ust + k, out, 1000, nu2);
    out.npre = 0;
    return true;
}

