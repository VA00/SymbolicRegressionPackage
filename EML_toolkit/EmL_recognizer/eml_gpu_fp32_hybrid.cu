#include <ctype.h>
#include <cuda_runtime.h>
#include <float.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define MAX_TOKENS 47
#define MAX_LEAVES ((MAX_TOKENS + 1) / 2)
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

static int parse_target_expr(const char* expr, double* out)
{
    char buf[256];
    trim_copy(expr, buf, sizeof(buf));

    if (buf[0] == '\0') {
        return 0;
    }

    char* slash = strchr(buf, '/');
    if (slash != NULL) {
        *slash = '\0';
        char* end_a = NULL;
        char* end_b = NULL;
        double a = strtod(buf, &end_a);
        double b = strtod(slash + 1, &end_b);
        if (end_a != buf && *end_a == '\0' && end_b != slash + 1 && *end_b == '\0' &&
            b != 0.0) {
            *out = a / b;
            return 1;
        }
        *slash = '/';
    }

    if (starts_with(buf, "Sqrt[") && buf[strlen(buf) - 1] == ']') {
        buf[strlen(buf) - 1] = '\0';
        char* end = NULL;
        double inner = strtod(buf + 5, &end);
        if (end != buf + 5 && *end == '\0' && inner >= 0.0) {
            *out = sqrt(inner);
            return 1;
        }
        return 0;
    }

    if (starts_with(buf, "sqrt(") && buf[strlen(buf) - 1] == ')') {
        buf[strlen(buf) - 1] = '\0';
        char* end = NULL;
        double inner = strtod(buf + 5, &end);
        if (end != buf + 5 && *end == '\0' && inner >= 0.0) {
            *out = sqrt(inner);
            return 1;
        }
        return 0;
    }

    if (strcmp(buf, "E") == 0) {
        *out = exp(1.0);
        return 1;
    }

    if (strcmp(buf, "Pi") == 0) {
        *out = acos(-1.0);
        return 1;
    }

    char* end = NULL;
    double value = strtod(buf, &end);
    if (end != buf && *end == '\0') {
        *out = value;
        return 1;
    }

    return 0;
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

static double evaluate_rank_double(int leaves, unsigned long long rank)
{
    if (leaves == 1) {
        return 1.0;
    }

    int left_leaves = 0;
    unsigned long long left_rank = 0;
    unsigned long long right_rank = 0;
    if (!choose_split_host(leaves, rank, &left_leaves, &left_rank, &right_rank)) {
        return NAN;
    }

    int right_leaves = leaves - left_leaves;
    double left_value = evaluate_rank_double(left_leaves, left_rank);
    double right_value = evaluate_rank_double(right_leaves, right_rank);
    if (!isfinite(left_value) || !isfinite(right_value) || right_value <= 0.0) {
        return NAN;
    }

    double out = exp(left_value) - log(right_value);
    return isfinite(out) ? out : NAN;
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

static int exact_hit(double value, double target, double tol)
{
    double scale = fabs(target);
    if (scale < 1.0) {
        scale = 1.0;
    }
    return fabs(value - target) <= tol * scale;
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

__device__ float evaluate_rank_fp32(int leaves, unsigned long long rank)
{
    int node_leaves[MAX_LEAVES];
    int right_leaves[MAX_LEAVES];
    unsigned long long node_rank[MAX_LEAVES];
    unsigned long long right_rank[MAX_LEAVES];
    unsigned char stage[MAX_LEAVES];
    float values[MAX_LEAVES];

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
            values[vsp++] = 1.0f;
            sp--;
            continue;
        }

        if (stage[top] == 0) {
            int left_leaves = 0;
            unsigned long long left_rank = 0;
            unsigned long long local_right_rank = 0;
            if (!choose_split_device(current_leaves, node_rank[top], &left_leaves,
                                     &left_rank, &local_right_rank)) {
                return nanf("");
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

        float right_value = values[--vsp];
        float left_value = values[--vsp];
        if (!(right_value > 0.0f)) {
            return nanf("");
        }

        float exp_left = expf(left_value);
        if (!isfinite(exp_left)) {
            return nanf("");
        }

        float log_right = logf(right_value);
        if (!isfinite(log_right)) {
            return nanf("");
        }

        float out = exp_left - log_right;
        if (!isfinite(out)) {
            return nanf("");
        }

        values[vsp++] = out;
        sp--;
    }

    return vsp == 1 ? values[0] : nanf("");
}

__global__ void search_eml_kernel(
    int leaves,
    unsigned long long count,
    unsigned long long offset,
    float target,
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
    float computed = evaluate_rank_fp32(leaves, rank);
    if (!isfinite(computed)) {
        return;
    }

    float rel_err = target == 0.0f ? fabsf(computed) : fabsf(computed / target - 1.0f);
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
    printf("Usage: %s [--target EXPR] [--max-tokens N] [--threshold X]\n", argv0);
    printf("       [--max-candidates N] [--exact-tol X] [--chunk-size N]\n");
    printf("\n");
    printf("Examples:\n");
    printf("  %s --target 2 --max-tokens 19\n", argv0);
    printf("  %s --target 1/2 --max-tokens 39 --threshold 1e-3\n", argv0);
    printf("  %s --target Sqrt[2] --max-tokens 47\n", argv0);
}

int main(int argc, char** argv)
{
    const char* target_expr = "2";
    int max_tokens = MAX_TOKENS;
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

    double target_double = 0.0;
    if (!parse_target_expr(target_expr, &target_double)) {
        fprintf(stderr, "Could not parse target expression: %s\n", target_expr);
        return 1;
    }
    float target_float = (float)target_double;

    init_shape_counts();
    CUDA_CHECK(cudaMemcpyToSymbol(d_shape_count, h_shape_count, sizeof(h_shape_count)));

    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));

    printf("=== EML GPU Search (FP32 GPU + FP64 CPU verification) ===\n");
    printf("Target expr:         %s\n", target_expr);
    printf("Target value:        %.17g\n", target_double);
    printf("Max tokens:          %d\n", max_tokens);
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
    double best_value = NAN;
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
                leaves, count, offset, target_float, candidate_threshold,
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
                double value = evaluate_rank_double(leaves, h_candidates[i].tree_idx);
                if (!isfinite(value)) {
                    continue;
                }

                double abs_err = fabs(value - target_double);
                if (abs_err < best_abs_err) {
                    best_abs_err = abs_err;
                    best_value = value;
                    best_leaves = leaves;
                    best_rank = h_candidates[i].tree_idx;
                }

                if (exact_hit(value, target_double, exact_tol)) {
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
        printf("K=%2d  shapes=%12llu  gpu=%.3f s  candidates=%llu",
               K, total_shapes, level_ms / 1000.0f, level_candidates_total);
        printf("  max_chunk=%d", level_max_chunk_candidates);
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
        printf("%.17g\n", best_value);
        printf("tokens=%d\n", 2 * best_leaves - 1);
        printf("abs_error=%.17e\n", best_abs_err);
        printf("status=%s\n", found_exact ? "exact-hit" : "best-candidate");
    } else {
        printf("No candidates survived the FP32 threshold.\n");
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
