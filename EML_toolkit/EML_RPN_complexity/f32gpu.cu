// f32gpu.cu - the sieve in single precision (f32kern.h) on the graphics card. Built with nvcc, linked into eml_mitm.
// One thread of the card takes one exp side a and a run of Log sides b, and runs every pair with everything above
// it (a depth-first walk with its own small stack). What it reports goes into a buffer of records; the processor
// decides them (mitm.inc).
#include <cuda_runtime.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <chrono>
static double tkern = 0, tcopy = 0;
static inline double nowd() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }

#define F32_FN __host__ __device__ static inline
#define F32_FN_MEMBER __host__ __device__
#define F32_COUNT(x) x
#ifdef F32_GPU_WHY
#define F32_WHY(x) x          // (the counts of the reasons cost 36 registers a thread: only in a build made for them)
#endif
#include "f32kern.h"

struct GOut {
    F32Rec* buf;
    unsigned* cnt;
    unsigned cap;
    uint32_t a, b;
    int s, i, npre;
    unsigned nodes, tops;
#ifdef F32_GPU_WHY
    unsigned why[8], fwhy[8], why2[2];
#endif
    __device__ void rec(int type, int nf, const uint32_t* st)
    {
        const unsigned k = atomicAdd(cnt, 1u);
        if (k >= cap) return;
        F32Rec& r = buf[k];
        st -= npre; nf += npre;                    // (second phase) the steps to the node of the unit are in front
        r.a = a; r.b = b; r.s = (uint8_t)s; r.i = (uint8_t)i; r.nf = (uint8_t)nf; r.type = (uint8_t)type;
        for (int t = 0; t < nf; t++) r.st[t] = st[t];
    }
    __device__ void init(F32Rec* b_, unsigned* c_, unsigned cap_)
    {
        buf = b_; cnt = c_; cap = cap_; a = b = 0; s = i = npre = 0; nodes = tops = 0;
#ifdef F32_GPU_WHY
        for (int k = 0; k < 8; k++) { why[k] = 0; fwhy[k] = 0; }
        why2[0] = why2[1] = 0;
#endif
    }
    __device__ void flush(unsigned long long* stats)
    {
        atomicAdd(&stats[0], (unsigned long long)nodes);
        atomicAdd(&stats[1], (unsigned long long)tops);
#ifdef F32_GPU_WHY
        for (int k = 0; k < 8; k++) {
            if (why[k]) atomicAdd(&stats[2 + k], (unsigned long long)why[k]);
            if (fwhy[k]) atomicAdd(&stats[10 + k], (unsigned long long)fwhy[k]);
        }
        if (why2[0]) atomicAdd(&stats[18], (unsigned long long)why2[0]);
        if (why2[1]) atomicAdd(&stats[19], (unsigned long long)why2[1]);
#endif
    }
};
struct GUnits {
    F32Rec* ubuf;
    unsigned* ucnt;
    unsigned ucap;
    uint32_t a, b;
    int s, i;
    __device__ void unit(int k, const uint32_t* ust)
    {
        const unsigned n = atomicAdd(ucnt, 1u);
        if (n >= ucap) return;
        F32Rec& r = ubuf[n];
        r.a = a; r.b = b; r.s = (uint8_t)s; r.i = (uint8_t)i; r.nf = (uint8_t)k; r.type = 4;
        for (int t = 0; t < k; t++) r.st[t] = ust[t];
    }
};

// First phase: a thread takes one exp side a and a run of Log sides b. Below the size ccut it walks everything; a
// node of at least that size with children of its own becomes a unit for the second phase (ccut = 1000: no units).
__global__ void f32_kernel(const F32G* gp, const F32A* A, const F32B* B, uint32_t a0, uint32_t na, uint32_t b0, uint32_t nb, uint32_t bstep, int s, int i, int ccut, F32Rec* buf, unsigned* cnt,
                           unsigned cap, F32Rec* ubuf, unsigned* ucnt, unsigned ucap, unsigned long long* stats)
{
    const unsigned long long t = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t nch = (nb + bstep - 1) / bstep;
    const unsigned long long ai = t / nch;
    const uint32_t ch = (uint32_t)(t % nch);
    if (ai >= na) return;
    const F32G g = *gp;
    F32N S[F32_MAXD + 2];
    uint32_t st[F32_MAXD + 1], ust[F32_MAXD + 1];
    GOut out;
    out.init(buf, cnt, cap);
    out.a = a0 + (uint32_t)ai; out.s = s; out.i = i;
    GUnits un;
    un.ubuf = ubuf; un.ucnt = ucnt; un.ucap = ucap; un.a = out.a; un.b = 0; un.s = s; un.i = i;
    const F32A Aa = A[a0 + ai];
    const uint32_t bb0 = b0 + ch * bstep;
    uint32_t bb1 = bb0 + bstep;
    if (bb1 > b0 + nb) bb1 = b0 + nb;
    for (uint32_t b = bb0; b < bb1; b++) {
        out.b = b; un.b = b;
        f32_pair(g, s, Aa, B[b], S, st, ust, out, ccut, un, 0);
    }
    out.flush(stats);
}

// Second phase: the threads take the units one after another from a common counter, so that none waits while there is work.
__global__ void f32_units_kernel(const F32G* gp, const F32A* A, const F32B* B, const F32Rec* ubuf, unsigned nunits, unsigned* next, F32Rec* buf, unsigned* cnt, unsigned cap,
                                 unsigned long long* stats)
{
    const F32G g = *gp;
    F32N S[F32_MAXD + 2];
    uint32_t st[F32_MAXD + 1], ust[F32_MAXD + 1], path[F32_MAXD + 1];
    GOut out;
    out.init(buf, cnt, cap);
    unsigned long long bad = 0;
    for (;;) {
        const unsigned u = atomicAdd(next, 1u);
        if (u >= nunits) break;
        const F32Rec U = ubuf[u];
        out.a = U.a; out.b = U.b; out.s = U.s; out.i = U.i; out.npre = 0;
        for (int t = 0; t < U.nf; t++) path[t] = U.st[t];
        if (!f32_unit(g, (int)U.s, A[U.a], B[U.b], (int)U.nf, path, S, st, ust, out)) bad++;
    }
    out.flush(stats);
    if (bad) atomicAdd(&stats[20], bad);
}

static F32A* dA[40];
static F32B* dB[40];
static size_t nA[40], nB[40];
static F32W* dW = nullptr;
static float* dRe = nullptr;
static float* dIm = nullptr;
static F32G* dG = nullptr;
static F32Rec* dBuf = nullptr;
static F32Rec* dUnits = nullptr;
static unsigned ucap = 0;
static unsigned* dCnt = nullptr;
static unsigned long long* dStats = nullptr;
static unsigned bufcap = 0;

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(e_), __FILE__, __LINE__); return 1; } } while (0)

extern "C" int f32gpu_init(unsigned cap, unsigned unitcap)
{
    CK(cudaSetDeviceFlags(cudaDeviceScheduleBlockingSync));
    CK(cudaSetDevice(0));
    CK(cudaDeviceSetLimit(cudaLimitStackSize, 16384));
    for (int k = 0; k < 40; k++) { dA[k] = nullptr; dB[k] = nullptr; nA[k] = nB[k] = 0; }
    bufcap = cap;
    CK(cudaMalloc(&dBuf, (size_t)cap * sizeof(F32Rec)));
    ucap = unitcap;
    CK(cudaMalloc(&dUnits, (size_t)ucap * sizeof(F32Rec)));
    CK(cudaMalloc(&dCnt, 4 * sizeof(unsigned)));
    CK(cudaMalloc(&dStats, 24 * sizeof(unsigned long long)));
    CK(cudaMalloc(&dG, sizeof(F32G)));
    size_t fr = 0, tot = 0;
    CK(cudaMemGetInfo(&fr, &tot));
    printf("graphics card: %.1f GB free of %.1f GB\n", fr / 1073741824.0, tot / 1073741824.0);
    return 0;
}
extern "C" int f32gpu_level(int lev, const F32A* A, size_t na, const F32B* B, size_t nb)
{
    if (na) { CK(cudaMalloc(&dA[lev], na * sizeof(F32A))); CK(cudaMemcpy(dA[lev], A, na * sizeof(F32A), cudaMemcpyHostToDevice)); }
    if (nb) { CK(cudaMalloc(&dB[lev], nb * sizeof(F32B))); CK(cudaMemcpy(dB[lev], B, nb * sizeof(F32B), cudaMemcpyHostToDevice)); }
    nA[lev] = na; nB[lev] = nb;
    return 0;
}
extern "C" int f32gpu_wrappers(const F32W* W, size_t nw)
{
    CK(cudaMalloc(&dW, (nw + 1) * sizeof(F32W)));
    CK(cudaMemcpy(dW, W, nw * sizeof(F32W), cudaMemcpyHostToDevice));
    return 0;
}
// the requests of a size n of the search, and the rest of what the kernel reads
extern "C" int f32gpu_setup(const F32G* g, size_t nrq)
{
    if (dRe) { cudaFree(dRe); cudaFree(dIm); dRe = dIm = nullptr; }
    CK(cudaMalloc(&dRe, (nrq + 1) * sizeof(float)));
    CK(cudaMalloc(&dIm, (nrq + 1) * sizeof(float)));
    if (nrq) {
        CK(cudaMemcpy(dRe, g->rqre, nrq * sizeof(float), cudaMemcpyHostToDevice));
        CK(cudaMemcpy(dIm, g->rqim, nrq * sizeof(float), cudaMemcpyHostToDevice));
    }
    F32G h = *g;
    h.W = dW; h.rqre = dRe; h.rqim = dIm;
    CK(cudaMemcpy(dG, &h, sizeof(F32G), cudaMemcpyHostToDevice));
    return 0;
}
// One launch: the exp sides a0..a0+na-1 of level i with the Log sides b0..b0+nb-1 of level s-i, bstep of them per
// thread, and (ccut < 1000) the units it leaves in a second launch. Waits for it. nrec: the records written (more
// than the buffer holds: nothing is valid); nunits: the units (more than their buffer holds: nothing is valid).
extern "C" int f32gpu_run(int s, int i, uint32_t a0, uint32_t na, uint32_t b0, uint32_t nb, uint32_t bstep, int ccut, F32Rec* out, unsigned* nrec, unsigned* nunits, unsigned long long* stats)
{
    const int j = s - i;
    CK(cudaMemset(dCnt, 0, 4 * sizeof(unsigned)));
    CK(cudaMemset(dStats, 0, 24 * sizeof(unsigned long long)));
    const unsigned long long nch = (nb + bstep - 1) / bstep, threads = (unsigned long long)na * nch;
    static const unsigned bd = getenv("F32_BD") ? (unsigned)atoi(getenv("F32_BD")) : 128;
    static const size_t smem = getenv("F32_SMEM") ? (size_t)atol(getenv("F32_SMEM")) : 0;      // (experiment) shared memory asked for per block: fewer blocks at a time on a multiprocessor
    const unsigned long long blocks = (threads + bd - 1) / bd;
    if (blocks > 0x7fffffffULL) { fprintf(stderr, "f32gpu_run: too many blocks\n"); return 1; }
    const double t0 = nowd();
    f32_kernel<<<(unsigned)blocks, bd, smem>>>(dG, dA[i], dB[j], a0, na, b0, nb, bstep, s, i, ccut, dBuf, dCnt, bufcap, dUnits, dCnt + 1, ucap, dStats);
    CK(cudaGetLastError());
    CK(cudaDeviceSynchronize());
    unsigned c4[4] = { 0, 0, 0, 0 };
    CK(cudaMemcpy(c4, dCnt, 4 * sizeof(unsigned), cudaMemcpyDeviceToHost));
    *nunits = c4[1];
    if (c4[1] > 0 && c4[1] <= ucap && c4[0] <= bufcap) {
        const unsigned nt = c4[1] < 262144u ? c4[1] : 262144u;
        f32_units_kernel<<<(nt + bd - 1) / bd, bd>>>(dG, dA[i], dB[j], dUnits, c4[1], dCnt + 2, dBuf, dCnt, bufcap, dStats);
        CK(cudaGetLastError());
        CK(cudaDeviceSynchronize());
        CK(cudaMemcpy(c4, dCnt, 4 * sizeof(unsigned), cudaMemcpyDeviceToHost));
    }
    const double t1 = nowd();
    tkern += t1 - t0;
    const unsigned n = c4[0];
    *nrec = n;
    if (n <= bufcap && *nunits <= ucap) {
        if (n) CK(cudaMemcpy(out, dBuf, (size_t)n * sizeof(F32Rec), cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(stats, dStats, 24 * sizeof(unsigned long long), cudaMemcpyDeviceToHost));
    }
    tcopy += nowd() - t1;
    return 0;
}
extern "C" void f32gpu_times(double* tk, double* tc) { *tk = tkern; *tc = tcopy; tkern = tcopy = 0; }
extern "C" void f32gpu_free()
{
    for (int k = 0; k < 40; k++) { if (dA[k]) cudaFree(dA[k]); if (dB[k]) cudaFree(dB[k]); dA[k] = nullptr; dB[k] = nullptr; }
    if (dW) cudaFree(dW);
    if (dRe) { cudaFree(dRe); cudaFree(dIm); }
    if (dBuf) cudaFree(dBuf);
    if (dUnits) cudaFree(dUnits);
    dUnits = nullptr;
    if (dCnt) cudaFree(dCnt);
    if (dStats) cudaFree(dStats);
    if (dG) cudaFree(dG);
    dW = nullptr; dRe = dIm = nullptr; dBuf = nullptr; dCnt = nullptr; dStats = nullptr; dG = nullptr;
}
