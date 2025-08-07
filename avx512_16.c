// Matrix Multiplication Implementation for x86_64
// Compile with: clang -march=native -mavx512f avx512_16.c -o avx512_16

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <immintrin.h>
#include <time.h>
#include <assert.h>
#include <math.h>
#include <omp.h>

#define TILE_M 16
#define TILE_N 16
#define TILE_K 1

#define WARMUP 10
#define RUN 10

#define UK_PREFETCH_RO(ptr, hint) __builtin_prefetch((ptr), /*rw=*/0, hint)
#define UK_PREFETCH_RW(ptr, hint) __builtin_prefetch((ptr), /*rw=*/1, hint)
// This is how the 0--3 locality hint is interpreted by both Clang and GCC on
// both arm64 and x86-64.
#define UK_PREFETCH_LOCALITY_NONE 0  // No locality. Data accessed once.
#define UK_PREFETCH_LOCALITY_L3 1    // Some locality. Try to keep in L3.
#define UK_PREFETCH_LOCALITY_L2 2    // More locality. Try to keep in L2.
#define UK_PREFETCH_LOCALITY_L1 3    // Most locality. Try to keep in L1.


// Structure to hold tiled matrix data
typedef struct {
    float* data;
    int rows;
    int cols;
    int tile_rows;
    int tile_cols;
} TiledMatrix;
__attribute__((always_inline)) static inline void matmul_tile_16x16x1_avx512(
    float* __restrict__ out_tile, const float* __restrict__ lhs_panel, const float* __restrict__ rhs_panel, const TiledMatrix* __restrict__ rhs);
TiledMatrix* pack_matrix(const float* src, int rows, int cols, int tile_size_m, int tile_size_n);
void unpack_matrix(float* dst, const TiledMatrix* tiled, int tile_size_m, int tile_size_n);
void iree_mmt4d(TiledMatrix* acc, const TiledMatrix* lhs, const TiledMatrix* rhs);
void matmul_reference(float* C, const float* A, const float* B, int M, int N, int K);
void init_matrix(float* mat, int size);
int check_result(const float* mat1, const float* mat2, int size, float tolerance);


int main(int argc, char* argv[]) {
    // Matrix dimensions
    if (argc != 4) {
        printf("Usgae: ./avx512_16 M N K!\n");
        exit(1);
    }
    
    int M = atoi(argv[1]);
    int N = atoi(argv[2]);
    int K = atoi(argv[3]);
    printf("Matrix multiplication: %dx%d * %dx%d = %dx%d\n", M, K, K, N, M, N);
    
    float* A = (float*)malloc(M * K * sizeof(float));
    float* B = (float*)malloc(K * N * sizeof(float));
    float* C_ref = (float*)malloc(M * N * sizeof(float));
    float* C_iree = (float*)malloc(M * N * sizeof(float));

    double flops = 2.0 * M * N * K;

    // Initialize matrices
    srand(137);
    init_matrix(A, M * K);
    init_matrix(B, K * N);
    
    // Reference implementation
    matmul_reference(C_ref, A, B, M, N, K);

    // Sequential mode
    for (int i = 0; i < WARMUP; ++i) {
        memset(C_iree, 0, M * N * sizeof(float));
        // Pack matrices into tiled format
        TiledMatrix* A_tiled = pack_matrix(A, M, K, TILE_M, TILE_K);
        TiledMatrix* B_tiled = pack_matrix(B, K, N, TILE_K, TILE_N);
        TiledMatrix* C_tiled = pack_matrix(C_iree, M, N, TILE_M, TILE_N);
        iree_mmt4d(C_tiled, A_tiled, B_tiled);
        unpack_matrix(C_iree, C_tiled, TILE_M, TILE_N);
        free(A_tiled->data);
        free(A_tiled);
        free(B_tiled->data);
        free(B_tiled);
        free(C_tiled->data);
        free(C_tiled);
    }
    
    clock_t t0, t1, t2, t3;
    double time0[RUN], time1[RUN], time2[RUN], time3[RUN];
    double avg_time0, avg_time1, avg_time2, avg_time3;
    for (int i = 0; i < RUN; ++i) {
        memset(C_iree, 0, M * N * sizeof(float));
        t0 = clock();
        TiledMatrix* A_tiled = pack_matrix(A, M, K, TILE_M, TILE_K);
        TiledMatrix* B_tiled = pack_matrix(B, K, N, TILE_K, TILE_N);
        TiledMatrix* C_tiled = pack_matrix(C_iree, M, N, TILE_M, TILE_N);
        t1 = clock();
        iree_mmt4d(C_tiled, A_tiled, B_tiled);
        t2 = clock();
        unpack_matrix(C_iree, C_tiled, TILE_M, TILE_N);
        t3 = clock();
        time0[i] = (double)(t3 - t0) / CLOCKS_PER_SEC;
        time1[i] = (double)(t2 - t1) / CLOCKS_PER_SEC;
        time2[i] = (double)(t2 - t0) / CLOCKS_PER_SEC;
        time3[i] = (double)(t3 - t1) / CLOCKS_PER_SEC;
        free(A_tiled->data);
        free(A_tiled);
        free(B_tiled->data);
        free(B_tiled);
        free(C_tiled->data);
        free(C_tiled);
    }
    avg_time0 = 0.0; 
    avg_time1 = 0.0; 
    avg_time2 = 0.0;
    avg_time3 = 0.0;
    for (int i = 0; i < RUN; ++i) {
        avg_time0 += time0[i];
        avg_time1 += time1[i];
        avg_time2 += time2[i];
        avg_time3 += time3[i];
    }
    avg_time0 /= RUN;
    avg_time1 /= RUN;
    avg_time2 /= RUN;
    avg_time3 /= RUN;

    // Verify results
    if (!check_result(C_ref, C_iree, M * N, 1e-5)) {
        printf("Sequential mode Results do not match!\n");
        exit(1);
    }

    printf("Sequential 0 : %.2f GFLOPS\n", flops / avg_time0 / 1e9);
    printf("Sequential 1 : %.2f GFLOPS\n", flops / avg_time1 / 1e9);
    printf("Sequential 2 : %.2f GFLOPS\n", flops / avg_time2 / 1e9);
    printf("Sequential 3 : %.2f GFLOPS\n", flops / avg_time3 / 1e9);

    // Cleanup
    free(A);
    free(B);
    free(C_ref);
    free(C_iree);
    
    return 0;
}

void iree_mmt4d(TiledMatrix* acc, const TiledMatrix* lhs, const TiledMatrix* rhs) {
    float* out_tile_row = acc->data;
    const float* lhs_panel = (const float*)lhs->data;
    const float* rhs_panel_start = (const float*)rhs->data;
    int out_tile_size = TILE_M * TILE_N;
    int lhs_panel_stride = lhs->tile_cols * TILE_M;
    int rhs_panel_stride = TILE_N;
    int out_stride = acc->tile_cols * TILE_M * TILE_N;

    for (int i = 0; i < lhs->tile_rows; ++i) {
        float* out_tile = out_tile_row;
        const float* rhs_panel = rhs_panel_start;
        UK_PREFETCH_RW(out_tile_row, UK_PREFETCH_LOCALITY_L3);
        UK_PREFETCH_RO(lhs_panel, UK_PREFETCH_LOCALITY_L1);
        UK_PREFETCH_RO(rhs_panel, UK_PREFETCH_LOCALITY_L1);
        for (int j = 0; j < rhs->tile_cols; ++j) {
            matmul_tile_16x16x1_avx512(out_tile, lhs_panel, rhs_panel, rhs);
            out_tile += out_tile_size;
            rhs_panel += rhs_panel_stride;
        }
        out_tile_row += out_stride;
        lhs_panel += lhs_panel_stride;
    }
}

__attribute__((always_inline)) static inline void matmul_tile_16x16x1_avx512(
    float* __restrict__ out_tile, const float* __restrict__ lhs_panel, const float* __restrict__ rhs_panel, const TiledMatrix* __restrict__ rhs) {
        float* __restrict__ out_ptr = out_tile;
        const float* __restrict__ lhs_ptr = lhs_panel;
        const float* __restrict__ rhs_ptr = rhs_panel;
        // The prefetches in this function are motivated by benchmarking on
        // Skylake; their effect was a > 1.3x speedup on 1024x1024 matmuls. The
        // prefetch-ahead offset of 128*sizeof(float) in the loop was empirically
        // determined. Similar prefetches did not produce any benefit in other
        // kernels, even though they are very similar to this one.
        _mm_prefetch((const char*)lhs_ptr, _MM_HINT_T0);
        _mm_prefetch((const char*)rhs_ptr, _MM_HINT_T0);
        __m512 acc[16];

        #pragma clang loop unroll(full)
        for (int i = 0; i < 16; ++i) {
            acc[i] = _mm512_setzero_ps();
        }
        
        for (int k = 0; k < rhs->rows; ++k) {
            __m512 rhs_loaded = _mm512_loadu_ps(rhs_ptr);
            _mm_prefetch((const char*)(rhs_ptr + 128 * rhs->tile_cols), _MM_HINT_T0);
            rhs_ptr += rhs->tile_cols << 4; // IREE's code: rhs_ptr += 16; // BUG! Could not deal with odd numbers!

            #pragma clang loop unroll(full)
            for (int i = 0; i < 16; ++i) {
                acc[i] = _mm512_fmadd_ps(rhs_loaded, _mm512_set1_ps(lhs_ptr[i]), acc[i]);
            }
            _mm_prefetch((const char*)(lhs_ptr + 128), _MM_HINT_T0);
            lhs_ptr += 16;
        }

        #pragma clang loop unroll(full)
        for (int i = 0; i < 16; ++i) {
            _mm512_storeu_ps(out_ptr + (i << 4), acc[i]);
        }
    }

// Pack matrix into tiled layout
TiledMatrix* pack_matrix(const float* src, int rows, int cols, int tile_size_m, int tile_size_n) {
    TiledMatrix* tiled = (TiledMatrix*)malloc(sizeof(TiledMatrix));
    
    // Calculate number of tiles (with padding)
    tiled->tile_rows = (rows + tile_size_m - 1) / tile_size_m;
    tiled->tile_cols = (cols + tile_size_n - 1) / tile_size_n;
    tiled->rows = rows;
    tiled->cols = cols;
    
    // Allocate tiled data
    int total_elements = tiled->tile_rows * tiled->tile_cols * tile_size_m * tile_size_n;
    tiled->data = (float*)aligned_alloc(64, total_elements * sizeof(float));
    memset(tiled->data, 0, total_elements * sizeof(float));
    
    // Pack data into tiles
    for (int tile_i = 0; tile_i < tiled->tile_rows; tile_i++) {
        for (int tile_j = 0; tile_j < tiled->tile_cols; tile_j++) {
            float* tile_ptr = &tiled->data[(tile_i * tiled->tile_cols + tile_j) * tile_size_m * tile_size_n];
            
            for (int i = 0; i < tile_size_m && tile_i * tile_size_m + i < rows; i++) {
                for (int j = 0; j < tile_size_n && tile_j * tile_size_n + j < cols; j++) {
                    int src_idx = (tile_i * tile_size_m + i) * cols + (tile_j * tile_size_n + j);
                    int tile_idx = i * tile_size_n + j;
                    tile_ptr[tile_idx] = src[src_idx];
                }
            }
        }
    }
    
    return tiled;
}

// Unpack tiled matrix back to regular layout
void unpack_matrix(float* dst, const TiledMatrix* tiled, int tile_size_m, int tile_size_n) {
    for (int tile_i = 0; tile_i < tiled->tile_rows; tile_i++) {
        for (int tile_j = 0; tile_j < tiled->tile_cols; tile_j++) {
            const float* tile_ptr = &tiled->data[(tile_i * tiled->tile_cols + tile_j) * tile_size_m * tile_size_n];
            
            for (int i = 0; i < tile_size_m && tile_i * tile_size_m + i < tiled->rows; i++) {
                for (int j = 0; j < tile_size_n && tile_j * tile_size_n + j < tiled->cols; j++) {
                    int dst_idx = (tile_i * tile_size_m + i) * tiled->cols + (tile_j * tile_size_n + j);
                    int tile_idx = i * tile_size_n + j;
                    dst[dst_idx] = tile_ptr[tile_idx];
                }
            }
        }
    }
}

// Standard matrix multiplication for verification
void matmul_reference(float* C, const float* A, const float* B, int M, int N, int K) {
    for (int i = 0; i < M; ++i) {
        for (int k = 0; k < K; ++k) {
            for (int j = 0; j < N; ++j) {
                C[i * N + j] += A[i * K + k] * B[k * N + j];
            }
        }
    }
}

// Initialize matrix with random values
void init_matrix(float* mat, int size) {
    for (int i = 0; i < size; i++) {
        mat[i] = (float)rand() / (float)RAND_MAX;
    }
}

// Check if two matrices are close enough
int check_result(const float* mat1, const float* mat2, int size, float tolerance) {
    for (int i = 0; i < size; i++) {
        float diff = fabs(mat1[i] - mat2[i]);
        //printf("diff at %d\t : %f\n", i, diff);
        if (diff > tolerance) {
            printf("Mismatch at index %d: %f vs %f (diff: %f)\n", i, mat1[i], mat2[i], diff);
            return 0;
        }
    }
    return 1;
}

