// Reference: https://arxiv.org/pdf/2203.03341
// We use wmma to avoid headaches + demonstration use

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <mma.h>
#include <iostream>
#include <vector>

using namespace nvcuda;  // Too lazy to write nvcuda::whatever

// CUDA check
#define CHECK_CUDA(expr)                                                \
    do {                                                                \
        cudaError_t _err = (expr);                                      \
        if (_err != cudaSuccess) {                                      \
            std::cerr << "CUDA error " << cudaGetErrorString(_err)      \
                      << " at " << __FILE__ << ":" << __LINE__ << "\n"; \
            std::exit(1);                                               \
        }                                                               \
    } while (0)


// The kernel itself
__global__ void matmul(const float* __restrict__ A, const float* __restrict__ B, float* C, const int M, const int N, const int K){  // Takes FP16's, accumulates in FP32 (C) 

    // Main matmul fragments
    // Despite the input is in float, the calculation is done in __half accuracy
    wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> frag_a;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::row_major> frag_b;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> frag_c;

    // Intializing the accumulator fragments
    wmma::fill_fragment(frag_c, 0.0f);

    // Warp indexes
    int warpM = blockIdx.y;  // Row
    int warpN = blockIdx.x;  // Col
    int tileRowBase = warpM * 16;
    int tileColBase = warpN * 16;

    int lane = threadIdx.x % 32;  // Lane in warp

    // Shared-memory tiles in __half for WMMA to load from
    __shared__ __half As_hi[16 * 16];
    __shared__ __half Bs_hi[16 * 16];

    // Loop through K
    for (int k = 0; k < K; k += 16) {
        // Since the input values are float, we need to convert it to __half and put it 
        // on shared memory, just so WMMA can use it.
        for (int idx = lane; idx < 16 * 16; idx += 32) {
            int r = idx / 16;   // Row within tile [0..15]
            int c = idx % 16;   // Col within tile [0..15]

            int globalRowA = tileRowBase + r;
            int globalColA = k + c;
            int globalRowB = k + r;
            int globalColB = tileColBase + c;

            // Load A: [M x K], row-major
            // This is the HIGH A
            float a_val = 0.0f;
            if (globalRowA < M && globalColA < K) {
                a_val = A[globalRowA * K + globalColA];
            }
            __half As_hi_idv = __float2half_rn(a_val);
            As_hi[idx] = As_hi_idv;

            // Load B: [K x N], row-major
            // This is the HIGH B
            float b_val = 0.0f;
            if (globalRowB < K && globalColB < N) {
                b_val = B[globalRowB * N + globalColB];
            }
            __half Bs_hi_idv = __float2half_rn(b_val);
            Bs_hi[idx] = Bs_hi_idv;
        }

        __syncthreads();

        // Now these tiles are 16x16, contiguous, row-major
        wmma::load_matrix_sync(frag_a, As_hi, 16);
        wmma::load_matrix_sync(frag_b, Bs_hi, 16);

        // Matmul
        wmma::mma_sync(frag_c, frag_a, frag_b, frag_c);
        __syncthreads();
    }

    float* tileC  = C + (warpM * 16) * N + (warpN * 16);
    wmma::store_matrix_sync(tileC, frag_c, N, wmma::mem_row_major);  // Is float
}


int main() {
    // Picking something divisible by 16 for now 
    // Because of how Volta's tensor tilings were assumed (16 x 16 x 16)
    const int M = 4096;
    const int N = 128;
    const int K = 4096;

    // Input data is float
    size_t sizeA = M * K * sizeof(float);
    size_t sizeB = K * N * sizeof(float);
    size_t sizeC = M * N * sizeof(float);

    // Host buffers
    // Btw we assumed the data as vectors, not matrices
    std::vector<float> hA(M * K);
    std::vector<float> hB(K * N);
    std::vector<float> hC(M * N, 0.0f);
    std::vector<float> hC_ref(M * N, 0.0f);

    // Initialize A and B with deterministic values
    // So its easier to gauge the differences between runs
    for (int i = 0; i < M; ++i) {
        for (int k = 0; k < K; ++k) {
            float val = (i + k) * 0.001f;       
            hA[i * K + k] = val;  // This is half
        }
    }

    for (int k = 0; k < K; ++k) {
        for (int j = 0; j < N; ++j) {
            float val = (k - j) * 0.002f;
            hB[k * N + j] = val;  // Ditto
        }
    }

    // CPU reference: C_ref = A * B (in float32)
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            float acc = 0.0f;
            for (int k = 0; k < K; ++k) {
                float a = hA[i * K + k];
                float b = hB[k * N + j];
                acc += a * b;
            }
            hC_ref[i * N + j] = acc;
        }
    }

    // Device buffers
    float *dA, *dB;
    float  *dC;

    // Allocate memory
    CHECK_CUDA(cudaMalloc(&dA, sizeA));
    CHECK_CUDA(cudaMalloc(&dB, sizeB));
    CHECK_CUDA(cudaMalloc(&dC, sizeC));

    CHECK_CUDA(cudaMemcpy(dA, hA.data(), sizeA, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dB, hB.data(), sizeB, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemset(dC, 0, sizeC));

    // Launch kernel
    dim3 gridDim((N + 15) / 16, (M + 15) / 16);
    dim3 blockDim(32, 1, 1);   // one warp

    // GPU matmul
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));
    CHECK_CUDA(cudaEventRecord(start));
    matmul<<<gridDim, blockDim>>>(dA, dB, dC, M, N, K);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    // We get the entire time after the devices are sync-ed
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    // Copy result back
    CHECK_CUDA(cudaMemcpy(hC.data(), dC, sizeC, cudaMemcpyDeviceToHost));

    // Check max error vs CPU reference
    float max_err = 0.0f;
    float max_ref = 0.0f;
    for (int i = 0; i < M * N; ++i) {
        float diff = std::fabs(hC[i] - hC_ref[i]);
        if (diff > max_err) max_err = diff;
        if (std::fabs(hC_ref[i]) > max_ref) max_ref = std::fabs(hC_ref[i]);
    }

    // Measure runtime and output difference
    float ms = 0.0f;
    CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop)); 

    printf("Elapsed time: %f ms\n", ms);
    printf("Throughput: %f TFLOP/s\n", ((float) 2*M*N*K)/(ms*1e9));
    std::cout << "Max absolute error: " << max_err << "\n";
    if (max_ref > 0.0f) {
        std::cout << "Max relative error: " << (max_err / max_ref) << "\n";
    }

    // Clean up
    CHECK_CUDA(cudaFree(dA));
    CHECK_CUDA(cudaFree(dB));
    CHECK_CUDA(cudaFree(dC));
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop)); 

    return 0;
}