// IREE-inspired Matrix Multiplication Implementation for x86_64
// Based on IREE's mmt4d microkernel optimizations
// Compile with: clang -O3 -march=native -mavx512f avx512_16.c -o avx512_16

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <immintrin.h>
#include <time.h>
#include <assert.h>
#include <math.h>
#include <omp.h>

// Tile sizes inspired by IREE's x86_64 AVX-512 implementation
#define TILE_M 16
#define TILE_N 16
#define TILE_K 1

#define WARMUP 10
#define RUN 10


// Structure to hold tiled matrix data (similar to IREE's mmt4d format)
typedef struct {
    float* data;
    int rows;
    int cols;
    int tile_rows;
    int tile_cols;
} TiledMatrix;

// This is based on iree_uk_mmt4d_tile_f32f32f32_16x16x1_x86_64_avx512_base
__attribute__((always_inline)) static inline void matmul_tile_16x16x1_avx512(
    float* __restrict__ out_tile,
    const float* __restrict__ lhs_tile,
    const float* __restrict__ rhs_tile) {
    
    const float* __restrict__ lhs_ptr = lhs_tile;
    const float* __restrict__ rhs_ptr = rhs_tile;

    _mm_prefetch((const char*)lhs_ptr, _MM_HINT_T0);
    _mm_prefetch((const char*)rhs_ptr, _MM_HINT_T0);
    __m512 acc[16];

    #pragma clang loop unroll(full)
    for (int i = 0; i < 16; i++) {
        acc[i] = _mm512_loadu_ps(out_tile + i * 16);
    }
    
    #pragma clang loop unroll(full)
    for (int i = 0; i < 16; i++) {
        acc[i] = _mm512_fmadd_ps(_mm512_loadu_ps(rhs_ptr), _mm512_set1_ps(lhs_ptr[i]), acc[i]);
    }

    // Store results back to accumulator
    #pragma clang loop unroll(full)
    for (int i = 0; i < 16; i++) {
        _mm512_storeu_ps(out_tile + i * 16, acc[i]);
    }
}

// Pack matrix into tiled layout (similar to IREE's pack operation)
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

void iree_mmt4d(TiledMatrix* acc, const TiledMatrix* lhs, const TiledMatrix* rhs, int K) {
    // Ensure compatible dimensions
    assert(lhs->tile_cols == (K + TILE_K - 1) / TILE_K);
    assert(rhs->tile_rows == (K + TILE_K - 1) / TILE_K);
    assert(acc->tile_rows == lhs->tile_rows);
    assert(acc->tile_cols == rhs->tile_cols);
    
    // Triple nested loop over tiles (following IREE's pattern)
    //#pragma omp parallel for collapse(2)
    for (int m = 0; m < lhs->tile_rows; m++) {
        for (int n = 0; n < rhs->tile_cols; n++) {
            float* acc_tile = &acc->data[(m * acc->tile_cols + n) * TILE_M * TILE_N];
            
            // Prefetch accumulator tile
            _mm_prefetch(acc_tile, _MM_HINT_T1);
            
            // Initialize accumulator to zero for first K tile
            memset(acc_tile, 0, TILE_M * TILE_N * sizeof(float));
            
            for (int k = 0; k < lhs->tile_cols; k++) {
                const float* lhs_tile = &lhs->data[(m * lhs->tile_cols + k) * TILE_M * TILE_K];
                const float* rhs_tile = &rhs->data[(k * rhs->tile_cols + n) * TILE_K * TILE_N];
                
                _mm_prefetch(&lhs->data[(m * lhs->tile_cols + k + 1) * TILE_M * TILE_K], _MM_HINT_T0);
                
                // Call optimized tile multiplication kernel
                matmul_tile_16x16x1_avx512(acc_tile, lhs_tile, rhs_tile);
                
                _mm_prefetch(&rhs->data[((k + 1) * rhs->tile_cols + n) * TILE_K * TILE_N], _MM_HINT_T0);
            }
            

        }
    }
}

void iree_mmt4d_p(TiledMatrix* acc, const TiledMatrix* lhs, const TiledMatrix* rhs, int K) {
    // Ensure compatible dimensions
    assert(lhs->tile_cols == (K + TILE_K - 1) / TILE_K);
    assert(rhs->tile_rows == (K + TILE_K - 1) / TILE_K);
    assert(acc->tile_rows == lhs->tile_rows);
    assert(acc->tile_cols == rhs->tile_cols);
    
    // Triple nested loop over tiles (following IREE's pattern)
    #pragma omp parallel for collapse(2)
    for (int m = 0; m < lhs->tile_rows; m++) {
        for (int n = 0; n < rhs->tile_cols; n++) {
            float* acc_tile = &acc->data[(m * acc->tile_cols + n) * TILE_M * TILE_N];
            
            // Prefetch accumulator tile
            _mm_prefetch(acc_tile, _MM_HINT_T1);
            
            // Initialize accumulator to zero for first K tile
            memset(acc_tile, 0, TILE_M * TILE_N * sizeof(float));
            
            for (int k = 0; k < lhs->tile_cols; k++) {
                const float* lhs_tile = &lhs->data[(m * lhs->tile_cols + k) * TILE_M * TILE_K];
                const float* rhs_tile = &rhs->data[(k * rhs->tile_cols + n) * TILE_K * TILE_N];
                
                _mm_prefetch(&lhs->data[(m * lhs->tile_cols + k + 1) * TILE_M * TILE_K], _MM_HINT_T0);
                
                // Call optimized tile multiplication kernel
                matmul_tile_16x16x1_avx512(acc_tile, lhs_tile, rhs_tile);
                
                _mm_prefetch(&rhs->data[((k + 1) * rhs->tile_cols + n) * TILE_K * TILE_N], _MM_HINT_T0);
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
        mat[i] = 0.5 + (float)rand() / (float)RAND_MAX;
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

int main(int argc, char* argv[]) {
    // Matrix dimensions
    int M = (argc > 1) ? atoi(argv[1]) : 512;
    int N = (argc > 2) ? atoi(argv[2]) : 512;
    int K = (argc > 3) ? atoi(argv[3]) : 512;
    
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
        // Pack matrices into tiled format
        TiledMatrix* A_tiled = pack_matrix(A, M, K, TILE_M, TILE_K);
        TiledMatrix* B_tiled = pack_matrix(B, K, N, TILE_K, TILE_N);
        TiledMatrix* C_tiled = pack_matrix(C_iree, M, N, TILE_M, TILE_N);
        iree_mmt4d(C_tiled, A_tiled, B_tiled, K);
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
        t0 = clock();
        TiledMatrix* A_tiled = pack_matrix(A, M, K, TILE_M, TILE_K);
        TiledMatrix* B_tiled = pack_matrix(B, K, N, TILE_K, TILE_N);
        TiledMatrix* C_tiled = pack_matrix(C_iree, M, N, TILE_M, TILE_N);
        t1 = clock();
        iree_mmt4d(C_tiled, A_tiled, B_tiled, K);
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

    // OpenMP mode
    for (int i = 0; i < WARMUP; ++i) {
        // Pack matrices into tiled format
        TiledMatrix* A_tiled = pack_matrix(A, M, K, TILE_M, TILE_K);
        TiledMatrix* B_tiled = pack_matrix(B, K, N, TILE_K, TILE_N);
        TiledMatrix* C_tiled = pack_matrix(C_iree, M, N, TILE_M, TILE_N);
        iree_mmt4d(C_tiled, A_tiled, B_tiled, K);
        unpack_matrix(C_iree, C_tiled, TILE_M, TILE_N);
        free(A_tiled->data);
        free(A_tiled);
        free(B_tiled->data);
        free(B_tiled);
        free(C_tiled->data);
        free(C_tiled);
    }
    
    for (int i = 0; i < RUN; ++i) {
        t0 = clock();
        TiledMatrix* A_tiled = pack_matrix(A, M, K, TILE_M, TILE_K);
        TiledMatrix* B_tiled = pack_matrix(B, K, N, TILE_K, TILE_N);
        TiledMatrix* C_tiled = pack_matrix(C_iree, M, N, TILE_M, TILE_N);
        t1 = clock();
        iree_mmt4d_p(C_tiled, A_tiled, B_tiled, K);
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
        printf("OpenMP mode Results do not match!\n");
        exit(1);
    }

    printf("OpenMp 0 : %.2f GFLOPS\n", flops / avg_time0 / 1e9);
    printf("OpenMp 1 : %.2f GFLOPS\n", flops / avg_time1 / 1e9);
    printf("OpenMp 2 : %.2f GFLOPS\n", flops / avg_time2 / 1e9);
    printf("OpenMp 3 : %.2f GFLOPS\n", flops / avg_time3 / 1e9);

    // Cleanup
    free(A);
    free(B);
    free(C_ref);
    free(C_iree);
    
    return 0;
}