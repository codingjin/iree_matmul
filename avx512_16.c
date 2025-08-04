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

// Tile sizes inspired by IREE's x86_64 AVX-512 implementation
#define TILE_M 16
#define TILE_N 16
#define TILE_K 1


// Structure to hold tiled matrix data (similar to IREE's mmt4d format)
typedef struct {
    float* data;
    int rows;
    int cols;
    int tile_rows;
    int tile_cols;
} TiledMatrix;

// IREE-inspired tile function for AVX-512
// This is based on iree_uk_mmt4d_tile_f32f32f32_16x16x1_x86_64_avx512_base
static inline void matmul_tile_16x16x1_avx512(
    float* __restrict__ out_tile,
    const float* __restrict__ lhs_tile,
    const float* __restrict__ rhs_tile,
    int K, int accumulate) {
    
    float* __restrict__ out_ptr = out_tile;
    const float* __restrict__ lhs_ptr = lhs_tile;
    const float* __restrict__ rhs_ptr = rhs_tile;

    _mm_prefetch((const char*)lhs_ptr, _MM_HINT_T0);
    _mm_prefetch((const char*)rhs_ptr, _MM_HINT_T0);
    // Load accumulator tiles (16x16 = 16 AVX-512 registers)
    __m512 acc[16];
    
    if (accumulate) {
        // Load existing accumulator values
        #pragma clang loop unroll(full)
        for (int i = 0; i < 16; i++) {
            acc[i] = _mm512_loadu_ps(out_tile + i * 16);
        }
    } else {
        // Zero initialize
        #pragma clang loop unroll(full)
        for (int i = 0; i < 16; i++) {
            acc[i] = _mm512_setzero_ps();
        }
    }
    
    // Main computation loop - similar to IREE's inner loop
    for (int k = 0; k < K; k++) {
        __m512 rhs = _mm512_loadu_ps(rhs_ptr);
        _mm_prefetch((const char*)(rhs_ptr + 128), _MM_HINT_T0);
        rhs_ptr += 16;

        #pragma clang loop unroll(full)
        for (int i = 0; i < 16; i++) {
            acc[i] = _mm512_fmadd_ps(rhs, _mm512_set1_ps(lhs_ptr[i]), acc[i]);
        }
        _mm_prefetch((const char*)(lhs_ptr + 128), _MM_HINT_T0);
        lhs_ptr += 16;
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

// IREE-inspired mmt4d operation
void iree_mmt4d(TiledMatrix* acc, const TiledMatrix* lhs, const TiledMatrix* rhs, int K) {
    // Ensure compatible dimensions
    assert(lhs->tile_cols == (K + TILE_K - 1) / TILE_K);
    assert(rhs->tile_rows == (K + TILE_K - 1) / TILE_K);
    assert(acc->tile_rows == lhs->tile_rows);
    assert(acc->tile_cols == rhs->tile_cols);
    
    // Triple nested loop over tiles (following IREE's pattern)
    for (int m = 0; m < lhs->tile_rows; m++) {
        for (int n = 0; n < rhs->tile_cols; n++) {
            float* acc_tile = &acc->data[(m * acc->tile_cols + n) * TILE_M * TILE_N];
            
            // Prefetch accumulator tile
            _mm_prefetch(acc_tile, _MM_HINT_T1);
            
            // Initialize accumulator to zero for first K tile
            memset(acc_tile, 0, TILE_M * TILE_N * sizeof(float));
            
            // Loop over K dimension tiles
            for (int k = 0; k < lhs->tile_cols; k++) {
                const float* lhs_tile = &lhs->data[(m * lhs->tile_cols + k) * TILE_M * TILE_K];
                const float* rhs_tile = &rhs->data[(k * rhs->tile_cols + n) * TILE_K * TILE_N];
                
                // Prefetch next tiles
                if (k + 1 < lhs->tile_cols) {
                    _mm_prefetch(&lhs->data[(m * lhs->tile_cols + k + 1) * TILE_M * TILE_K], _MM_HINT_T0);
                    _mm_prefetch(&rhs->data[((k + 1) * rhs->tile_cols + n) * TILE_K * TILE_N], _MM_HINT_T0);
                }
                
                // Call optimized tile multiplication kernel
                matmul_tile_16x16x1_avx512(acc_tile, lhs_tile, rhs_tile, TILE_K, k > 0);
            }
        }
    }
}

// Standard matrix multiplication for verification
void matmul_reference(float* C, const float* A, const float* B, int M, int N, int K) {
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float sum = 0.0f;
            for (int k = 0; k < K; k++) {
                sum += A[i * K + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
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
    printf("Using IREE-inspired tiled implementation with AVX-512\n\n");
    
    // Allocate matrices
    float* A = (float*)aligned_alloc(64, M * K * sizeof(float));
    float* B = (float*)aligned_alloc(64, K * N * sizeof(float));
    float* C_ref = (float*)aligned_alloc(64, M * N * sizeof(float));
    float* C_iree = (float*)aligned_alloc(64, M * N * sizeof(float));
    
    // Initialize matrices
    srand(137);
    init_matrix(A, M * K);
    init_matrix(B, K * N);
    
    // Reference implementation
    clock_t start = clock();
    matmul_reference(C_ref, A, B, M, N, K);
    clock_t end = clock();
    double ref_time = (double)(end - start) / CLOCKS_PER_SEC;
    printf("Reference implementation time: %.3f seconds\n", ref_time);
    
    // IREE-inspired implementation
    start = clock();
    
    // Pack matrices into tiled format
    TiledMatrix* A_tiled = pack_matrix(A, M, K, TILE_M, TILE_K);
    TiledMatrix* B_tiled = pack_matrix(B, K, N, TILE_K, TILE_N);
    TiledMatrix* C_tiled = pack_matrix(C_iree, M, N, TILE_M, TILE_N);
    
    // Perform tiled matrix multiplication
    iree_mmt4d(C_tiled, A_tiled, B_tiled, K);
    
    // Unpack result
    unpack_matrix(C_iree, C_tiled, TILE_M, TILE_N);
    
    end = clock();
    double iree_time = (double)(end - start) / CLOCKS_PER_SEC;
    printf("IREE-inspired implementation time: %.3f seconds\n", iree_time);
    printf("Speedup: %.2fx\n\n", ref_time / iree_time);
    
    // Verify results
    if (check_result(C_ref, C_iree, M * N, 1e-3)) {
        printf("Results match!\n");
    } else {
        printf("Results do not match!\n");
    }
    
    // Calculate GFLOPS
    double flops = 2.0 * M * N * K;
    printf("\nPerformance metrics:\n");
    printf("Reference: %.2f GFLOPS\n", flops / ref_time / 1e9);
    printf("IREE-inspired: %.2f GFLOPS\n", flops / iree_time / 1e9);
    
    // Cleanup
    free(A);
    free(B);
    free(C_ref);
    free(C_iree);
    free(A_tiled->data);
    free(B_tiled->data);
    free(C_tiled->data);
    free(A_tiled);
    free(B_tiled);
    free(C_tiled);
    
    return 0;
}