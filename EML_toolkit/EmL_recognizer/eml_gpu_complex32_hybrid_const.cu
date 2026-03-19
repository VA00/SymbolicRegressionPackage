#include <ctype.h>
#include <cuda_runtime.h>
#include <float.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <thrust/complex.h>
#include <time.h>

#define MAX_TOKENS 47
#define MAX_LEAVES ((MAX_TOKENS + 1) / 2)
#define DEFAULT_MAX_TOKENS MAX_TOKENS
#define DEFAULT_THREADS 256
#define DEFAULT_CHUNK_SIZE (1ULL << 26)
#define DEFAULT_MAX_CANDIDATES (1 << 20)
#define DEFAULT_THRESHOLD (512.0f * FLT_EPSILON)
#define DEFAULT_EXACT_TOL (8.0 * DBL_EPSILON)

#define CUDA_CHECK(call)                                                          \
    do {                                                                          \
        cudaError_t err__ = (call);                                               \
        if (err__ != cudaSuccess) {                                               \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,      \
                    cudaGetErrorString(err__));                                    \
            exit(1);                                                               \
        }                                                                          \
    } while (0)

typedef thrust::complex<float> complex32;
typedef thrust::complex<double> complex64;

typedef struct {
    float fp32_error;
    unsigned long long tree_idx;
    int K;
} Candidate;

__constant__ unsigned long long d_shape_count[MAX_LEAVES + 1];

static unsigned long long h_shape_count[MAX_LEAVES + 1];

static void print_timestamp(void)
{
    time_t now = time(NULL);
    struct tm local_tm;
#if defined(_WIN32)
    localtime_s(&local_tm, &now);
#else
    localtime_r(&now, &local_tm);
#endif
    char buf[64];
    strftime(buf, sizeof(buf), "%Y-%m-%d %H:%M:%S", &local_tm);
    printf("%s\n", buf);
}

static int starts_with(const char* s, const char* prefix)
{
    return strncmp(s, prefix, strlen(prefix)) == 0;
}

static const char* exe_name(const char* path)
{
    const char* slash = strrchr(path, '\\');
    const char* forward_slash = strrchr(path, '/');
    const char* base = path;

    if (slash != NULL && slash + 1 > base) {
        base = slash + 1;
    }
    if (forward_slash != NULL && forward_slash + 1 > base) {
        base = forward_slash + 1;
    }
    return base;
}

static void trim_copy(const char* src, char* dst, size_t dst_size)
{
    const char* begin = src;
    while (*begin && isspace((unsigned char)*begin)) {
        begin++;
    }

    const char* end = begin + strlen(begin);
    while (end > begin && isspace((unsigned char)end[-1])) {
        end--;
    }

    size_t len = (size_t)(end - begin);
    if (len >= dst_size) {
        len = dst_size - 1;
    }
    memcpy(dst, begin, len);
    dst[len] = '\0';
}

static int parse_target_complex(const char* expr, complex64* out)
{
    char buf[256];
    trim_copy(expr, buf, sizeof(buf));

    if (buf[0] == '\0') {
        return 0;
    }

    char* comma = strchr(buf, ',');
    if (comma == NULL) {
        char* end = NULL;
        double real_part = strtod(buf, &end);
        if (end != buf && *end == '\0') {
            *out = complex64(real_part, 0.0);
            return 1;
        }
        return 0;
    }

    *comma = '\0';
    char* real_text = buf;
    char* imag_text = comma + 1;
    char* end_real = NULL;
    char* end_imag = NULL;
    double real_part = strtod(real_text, &end_real);
    double imag_part = strtod(imag_text, &end_imag);
    if (end_real == real_text || *end_real != '\0' ||
        end_imag == imag_text || *end_imag != '\0') {
        return 0;
    }

    *out = complex64(real_part, imag_part);
    return 1;
}

static int complex_is_finite64(complex64 z)
{
    return isfinite(z.real()) && isfinite(z.imag());
}

static int complex_is_zero64(complex64 z)
{
    return z.real() == 0.0 && z.imag() == 0.0;
}

static void init_shape_counts(void)
{
    memset(h_shape_count, 0, sizeof(h_shape_count));
    h_shape_count[1] = 1;

    for (int leaves = 2; leaves <= MAX_LEAVES; ++leaves) {
        unsigned long long total = 0;
        for (int left = 1; left < leaves; ++left) {
            int right = leaves - left;
            total += h_shape_count[left] * h_shape_count[right];
        }
        h_shape_count[leaves] = total;
    }
}

static int choose_split_host(
    int leaves,
    unsigned long long rank,
    int* left_leaves,
    unsigned long long* left_rank,
    unsigned long long* right_rank
)
{
    for (int left = 1; left < leaves; ++left) {
        int right = leaves - left;
        unsigned long long right_count = h_shape_count[right];
        unsigned long long block = h_shape_count[left] * right_count;
        if (rank < block) {
            *left_leaves = left;
            *left_rank = rank / right_count;
            *right_rank = rank % right_count;
            return 1;
        }
        rank -= block;
    }
    return 0;
}

static complex64 eml_eval_complex64(complex64 left_value, complex64 right_value)
{
    if (!complex_is_finite64(left_value) || !complex_is_finite64(right_value) ||
        complex_is_zero64(right_value)) {
        return complex64(NAN, NAN);
    }

    complex64 out = thrust::exp(left_value) - thrust::log(right_value);
    return complex_is_finite64(out) ? out : complex64(NAN, NAN);
}

static complex64 evaluate_rank_complex64(int leaves, unsigned long long rank)
{
    if (leaves == 1) {
        return complex64(1.0, 0.0);
    }

    int left_leaves = 0;
    unsigned long long left_rank = 0;
    unsigned long long right_rank = 0;
    if (!choose_split_host(leaves, rank, &left_leaves, &left_rank, &right_rank)) {
        return complex64(NAN, NAN);
    }

    int right_leaves = leaves - left_leaves;
    complex64 left_value = evaluate_rank_complex64(left_leaves, left_rank);
    complex64 right_value = evaluate_rank_complex64(right_leaves, right_rank);
    return eml_eval_complex64(left_value, right_value);
}

static void build_rpn_tokens(
    int leaves,
    unsigned long long rank,
    int* tokens,
    int* len
)
{
    if (leaves == 1) {
        tokens[(*len)++] = 0;
        return;
    }

    int left_leaves = 0;
    unsigned long long left_rank = 0;
    unsigned long long right_rank = 0;
    choose_split_host(leaves, rank, &left_leaves, &left_rank, &right_rank);

    build_rpn_tokens(left_leaves, left_rank, tokens, len);
    build_rpn_tokens(leaves - left_leaves, right_rank, tokens, len);
    tokens[(*len)++] = 1;
}

static void print_rpn_rule_from_rank(int leaves, unsigned long long rank)
{
    int tokens[MAX_TOKENS];
    int len = 0;
    build_rpn_tokens(leaves, rank, tokens, &len);

    printf("rpnRule[{");
    for (int i = 0; i < len; ++i) {
        if (i) {
            printf(", ");
        }
        printf(tokens[i] == 0 ? "1" : "EML");
    }
    printf("}]\n");
}

static int compare_candidates(const void* a, const void* b)
{
    float ea = ((const Candidate*)a)->fp32_error;
    float eb = ((const Candidate*)b)->fp32_error;
    if (ea < eb) {
        return -1;
    }
    if (ea > eb) {
        return 1;
    }

    unsigned long long ia = ((const Candidate*)a)->tree_idx;
    unsigned long long ib = ((const Candidate*)b)->tree_idx;
    if (ia < ib) {
        return -1;
    }
    if (ia > ib) {
        return 1;
    }
    return 0;
}

static int exact_hit(complex64 value, complex64 target, double tol)
{
    double scale = thrust::abs(target);
    if (scale < 1.0) {
        scale = 1.0;
    }
    return thrust::abs(value - target) <= tol * scale;
}

__device__ __forceinline__ int choose_split_device(
    int leaves,
    unsigned long long rank,
    int* left_leaves,
    unsigned long long* left_rank,
    unsigned long long* right_rank
)
{
    for (int left = 1; left < leaves; ++left) {
        int right = leaves - left;
        unsigned long long right_count = d_shape_count[right];
        unsigned long long block = d_shape_count[left] * right_count;
        if (rank < block) {
            *left_leaves = left;
            *left_rank = rank / right_count;
            *right_rank = rank % right_count;
            return 1;
        }
        rank -= block;
    }
    return 0;
}

__device__ __forceinline__ int complex_is_finite32(complex32 z)
{
    return isfinite(z.real()) && isfinite(z.imag());
}

__device__ __forceinline__ int complex_is_zero32(complex32 z)
{
    return z.real() == 0.0f && z.imag() == 0.0f;
}

__device__ __forceinline__ complex32 eml_eval_complex32(complex32 left_value, complex32 right_value)
{
    if (!complex_is_finite32(left_value) || !complex_is_finite32(right_value) ||
        complex_is_zero32(right_value)) {
        return complex32(nanf(""), nanf(""));
    }

    complex32 out = thrust::exp(left_value) - thrust::log(right_value);
    return complex_is_finite32(out) ? out : complex32(nanf(""), nanf(""));
}

__device__ complex32 evaluate_rank_complex32(int leaves, unsigned long long rank)
{
    int node_leaves[MAX_LEAVES];
    int right_leaves[MAX_LEAVES];
    unsigned long long node_rank[MAX_LEAVES];
    unsigned long long right_rank[MAX_LEAVES];
    unsigned char stage[MAX_LEAVES];
    complex32 values[MAX_LEAVES];

    int sp = 0;
    int vsp = 0;

    node_leaves[sp] = leaves;
    node_rank[sp] = rank;
    stage[sp] = 0;
    sp++;

    while (sp > 0) {
        int top = sp - 1;
        int current_leaves = node_leaves[top];

        if (current_leaves == 1) {
            values[vsp++] = complex32(1.0f, 0.0f);
            sp--;
            continue;
        }

        if (stage[top] == 0) {
            int left_leaves = 0;
            unsigned long long left_rank = 0;
            unsigned long long local_right_rank = 0;
            if (!choose_split_device(current_leaves, node_rank[top], &left_leaves,
                                     &left_rank, &local_right_rank)) {
                return complex32(nanf(""), nanf(""));
            }

            stage[top] = 1;
            right_leaves[top] = current_leaves - left_leaves;
            right_rank[top] = local_right_rank;

            node_leaves[sp] = left_leaves;
            node_rank[sp] = left_rank;
            stage[sp] = 0;
            sp++;
            continue;
        }

        if (stage[top] == 1) {
            stage[top] = 2;
            node_leaves[sp] = right_leaves[top];
            node_rank[sp] = right_rank[top];
            stage[sp] = 0;
            sp++;
            continue;
        }

        complex32 right_value = values[--vsp];
        complex32 left_value = values[--vsp];
        complex32 out = eml_eval_complex32(left_value, right_value);
        if (!complex_is_finite32(out)) {
            return complex32(nanf(""), nanf(""));
        }

        values[vsp++] = out;
        sp--;
    }

    return vsp == 1 ? values[0] : complex32(nanf(""), nanf(""));
}

__global__ void search_eml_kernel(
    int leaves,
    unsigned long long count,
    unsigned long long offset,
    complex32 target,
    float candidate_threshold,
    Candidate* candidates,
    int* candidate_count,
    int max_candidates
)
{
    unsigned long long idx = blockIdx.x * (unsigned long long)blockDim.x + threadIdx.x;
    if (idx >= count) {
        return;
    }

    unsigned long long rank = offset + idx;
    complex32 computed = evaluate_rank_complex32(leaves, rank);
    if (!complex_is_finite32(computed)) {
        return;
    }

    float scale = thrust::abs(target);
    if (scale < 1.0f) {
        scale = 1.0f;
    }
    float rel_err = thrust::abs(computed - target) / scale;
    if (rel_err < candidate_threshold) {
        int slot = atomicAdd(candidate_count, 1);
        if (slot < max_candidates) {
            candidates[slot].fp32_error = rel_err;
            candidates[slot].tree_idx = rank;
            candidates[slot].K = 2 * leaves - 1;
        }
    }
}

static void print_usage(const char* argv0)
{
    const char* prog = exe_name(argv0);
    printf("Usage: %s [--target RE,IM] [--max-tokens N] [--threshold X]\n", prog);
    printf("       [--max-candidates N] [--exact-tol X] [--chunk-size N]\n");
    printf("\n");
    printf("Leaves are drawn from {1}; targets are numeric complex pairs RE,IM.\n");
    printf("Complex arithmetic uses the principal branch of log.\n");
    printf("\n");
    printf("Examples:\n");
    printf("  %s --target 0,1 --max-tokens 31\n", prog);
    printf("  %s --target 2,0 --max-tokens 19\n", prog);
    printf("  %s --target 0.5,0 --max-tokens 39 --threshold 1e-3\n", prog);
}

int main(int argc, char** argv)
{
    const char* target_expr = "0,1";
    int max_tokens = DEFAULT_MAX_TOKENS;
    float candidate_threshold = DEFAULT_THRESHOLD;
    int max_candidates = DEFAULT_MAX_CANDIDATES;
    double exact_tol = DEFAULT_EXACT_TOL;
    unsigned long long chunk_size = DEFAULT_CHUNK_SIZE;

    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0) {
            print_usage(argv[0]);
            return 0;
        } else if (strcmp(argv[i], "--target") == 0 && i + 1 < argc) {
            target_expr = argv[++i];
        } else if (starts_with(argv[i], "--target=")) {
            target_expr = argv[i] + 9;
        } else if (strcmp(argv[i], "--max-tokens") == 0 && i + 1 < argc) {
            max_tokens = atoi(argv[++i]);
        } else if (starts_with(argv[i], "--max-tokens=")) {
            max_tokens = atoi(argv[i] + 13);
        } else if (strcmp(argv[i], "--threshold") == 0 && i + 1 < argc) {
            candidate_threshold = (float)atof(argv[++i]);
        } else if (starts_with(argv[i], "--threshold=")) {
            candidate_threshold = (float)atof(argv[i] + 12);
        } else if (strcmp(argv[i], "--max-candidates") == 0 && i + 1 < argc) {
            max_candidates = atoi(argv[++i]);
        } else if (starts_with(argv[i], "--max-candidates=")) {
            max_candidates = atoi(argv[i] + 17);
        } else if (strcmp(argv[i], "--exact-tol") == 0 && i + 1 < argc) {
            exact_tol = atof(argv[++i]);
        } else if (starts_with(argv[i], "--exact-tol=")) {
            exact_tol = atof(argv[i] + 12);
        } else if (strcmp(argv[i], "--chunk-size") == 0 && i + 1 < argc) {
            chunk_size = strtoull(argv[++i], NULL, 10);
        } else if (starts_with(argv[i], "--chunk-size=")) {
            chunk_size = strtoull(argv[i] + 13, NULL, 10);
        } else if (argv[i][0] != '-') {
            target_expr = argv[i];
        } else {
            fprintf(stderr, "Unknown argument: %s\n", argv[i]);
            return 1;
        }
    }

    if (max_tokens < 1) {
        fprintf(stderr, "max-tokens must be >= 1\n");
        return 1;
    }
    if (max_tokens > MAX_TOKENS) {
        fprintf(stderr, "max-tokens=%d exceeds compiled MAX_TOKENS=%d\n",
                max_tokens, MAX_TOKENS);
        return 1;
    }
    if ((max_tokens & 1) == 0) {
        max_tokens -= 1;
    }
    if (candidate_threshold <= 0.0f) {
        fprintf(stderr, "threshold must be > 0\n");
        return 1;
    }
    if (max_candidates <= 0) {
        fprintf(stderr, "max-candidates must be > 0\n");
        return 1;
    }

    complex64 target_value;
    if (!parse_target_complex(target_expr, &target_value)) {
        fprintf(stderr, "Could not parse numeric complex target: %s\n", target_expr);
        return 1;
    }
    complex32 target_value_f((float)target_value.real(), (float)target_value.imag());

    init_shape_counts();
    CUDA_CHECK(cudaMemcpyToSymbol(d_shape_count, h_shape_count, sizeof(h_shape_count)));

    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));

    printf("=== EML Complex Constant Search (complex32 GPU + complex64 CPU verification) ===\n");
    printf("Target:              %s\n", target_expr);
    printf("Target value:        %.17g %+.17g I\n", target_value.real(), target_value.imag());
    printf("Max tokens:          %d\n", max_tokens);
    printf("Branch:              principal complex log\n");
    printf("Candidate threshold: %.9g (%.1f FLT_EPSILON)\n",
           candidate_threshold, candidate_threshold / FLT_EPSILON);
    printf("Exact tolerance:     %.3e (%.1f DBL_EPSILON)\n",
           exact_tol, exact_tol / DBL_EPSILON);
    printf("Chunk size:          %llu\n", chunk_size);
    printf("Max candidates:      %d\n", max_candidates);
    printf("GPU:                 %s (%d SMs)\n", prop.name, prop.multiProcessorCount);
    print_timestamp();
    printf("\n");

    Candidate* d_candidates = NULL;
    int* d_candidate_count = NULL;
    CUDA_CHECK(cudaMalloc(&d_candidates, (size_t)max_candidates * sizeof(Candidate)));
    CUDA_CHECK(cudaMalloc(&d_candidate_count, sizeof(int)));

    Candidate* h_candidates = (Candidate*)malloc((size_t)max_candidates * sizeof(Candidate));
    if (h_candidates == NULL) {
        fprintf(stderr, "Host allocation failed for candidate buffer\n");
        return 1;
    }

    unsigned long long total_evaluated = 0;
    double best_abs_err = HUGE_VAL;
    complex64 best_value(NAN, NAN);
    int best_leaves = 0;
    unsigned long long best_rank = 0;
    int found_exact = 0;

    cudaEvent_t overall_start, overall_stop;
    CUDA_CHECK(cudaEventCreate(&overall_start));
    CUDA_CHECK(cudaEventCreate(&overall_stop));
    CUDA_CHECK(cudaEventRecord(overall_start));

    for (int K = 1; K <= max_tokens; K += 2) {
        int leaves = (K + 1) / 2;
        unsigned long long total_shapes = h_shape_count[leaves];
        unsigned long long level_candidates_total = 0;
        int level_max_chunk_candidates = 0;
        unsigned long long level_dropped_total = 0;

        cudaEvent_t level_start, level_stop;
        CUDA_CHECK(cudaEventCreate(&level_start));
        CUDA_CHECK(cudaEventCreate(&level_stop));
        CUDA_CHECK(cudaEventRecord(level_start));

        for (unsigned long long offset = 0; offset < total_shapes; offset += chunk_size) {
            int zero = 0;
            CUDA_CHECK(cudaMemcpy(d_candidate_count, &zero, sizeof(int), cudaMemcpyHostToDevice));

            unsigned long long count = total_shapes - offset;
            if (count > chunk_size) {
                count = chunk_size;
            }

            int blocks = (int)((count + DEFAULT_THREADS - 1) / DEFAULT_THREADS);
            search_eml_kernel<<<blocks, DEFAULT_THREADS>>>(
                leaves, count, offset, target_value_f, candidate_threshold,
                d_candidates, d_candidate_count, max_candidates
            );
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaDeviceSynchronize());

            int candidate_count = 0;
            CUDA_CHECK(cudaMemcpy(&candidate_count, d_candidate_count, sizeof(int),
                                  cudaMemcpyDeviceToHost));
            level_candidates_total += (unsigned long long)candidate_count;
            if (candidate_count > level_max_chunk_candidates) {
                level_max_chunk_candidates = candidate_count;
            }

            int copied_candidates = candidate_count;
            if (copied_candidates > max_candidates) {
                level_dropped_total += (unsigned long long)(copied_candidates - max_candidates);
                copied_candidates = max_candidates;
            }

            if (copied_candidates > 0) {
                CUDA_CHECK(cudaMemcpy(h_candidates, d_candidates,
                                      (size_t)copied_candidates * sizeof(Candidate),
                                      cudaMemcpyDeviceToHost));
                qsort(h_candidates, copied_candidates, sizeof(Candidate), compare_candidates);
            }

            for (int i = 0; i < copied_candidates; ++i) {
                complex64 value = evaluate_rank_complex64(leaves, h_candidates[i].tree_idx);
                if (!complex_is_finite64(value)) {
                    continue;
                }

                double abs_err = thrust::abs(value - target_value);
                if (abs_err < best_abs_err) {
                    best_abs_err = abs_err;
                    best_value = value;
                    best_leaves = leaves;
                    best_rank = h_candidates[i].tree_idx;
                }

                if (exact_hit(value, target_value, exact_tol)) {
                    found_exact = 1;
                    best_abs_err = abs_err;
                    best_value = value;
                    best_leaves = leaves;
                    best_rank = h_candidates[i].tree_idx;
                    break;
                }
            }

            if (found_exact) {
                break;
            }
        }

        CUDA_CHECK(cudaEventRecord(level_stop));
        CUDA_CHECK(cudaEventSynchronize(level_stop));

        float level_ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&level_ms, level_start, level_stop));

        total_evaluated += total_shapes;
        printf("K=%2d  shapes=%12llu  gpu=%.3f s  candidates=%llu  max_chunk=%d",
               K, total_shapes, level_ms / 1000.0f, level_candidates_total,
               level_max_chunk_candidates);
        if (level_dropped_total > 0) {
            printf("  (overflow: dropped %llu)", level_dropped_total);
        }
        printf("\n");

        CUDA_CHECK(cudaEventDestroy(level_start));
        CUDA_CHECK(cudaEventDestroy(level_stop));

        if (found_exact) {
            break;
        }
    }

    CUDA_CHECK(cudaEventRecord(overall_stop));
    CUDA_CHECK(cudaEventSynchronize(overall_stop));

    float overall_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&overall_ms, overall_start, overall_stop));

    printf("\n=== RESULT ===\n");
    if (best_leaves > 0) {
        print_rpn_rule_from_rank(best_leaves, best_rank);
        printf("%.17g %+.17g I\n", best_value.real(), best_value.imag());
        printf("tokens=%d\n", 2 * best_leaves - 1);
        printf("abs_error=%.17e\n", best_abs_err);
        printf("status=%s\n", found_exact ? "exact-hit" : "best-candidate");
    } else {
        printf("No candidates survived the complex32 threshold.\n");
    }

    printf("\n=== PERFORMANCE ===\n");
    printf("total_shapes=%llu\n", total_evaluated);
    printf("gpu_seconds=%.6f\n", overall_ms / 1000.0f);
    if (overall_ms > 0.0f) {
        printf("throughput=%.3f M eval/s\n", total_evaluated / overall_ms / 1000.0f);
    }
    print_timestamp();

    free(h_candidates);
    CUDA_CHECK(cudaFree(d_candidates));
    CUDA_CHECK(cudaFree(d_candidate_count));
    CUDA_CHECK(cudaEventDestroy(overall_start));
    CUDA_CHECK(cudaEventDestroy(overall_stop));

    return found_exact ? 0 : 1;
}
