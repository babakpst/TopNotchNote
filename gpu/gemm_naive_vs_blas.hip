// gemm_naive_vs_blas.hip
// Companion example for: topnotchnote.com/gpu/gemm_libraries.html
//
// Why nobody hand-writes GEMM: the textbook matrix-multiply kernel against
// one call to the vendor BLAS library (hipBLAS, which forwards to rocBLAS
// on AMD and to cuBLAS on Nvidia), on square FP32 matrices from 256 to
// 4096. For each size it prints the time, the achieved GFLOPS and the
// largest difference between the two results.
//
// Compile:  hipcc -O2 -o main gemm_naive_vs_blas.hip -lhipblas
// Nvidia:   hipcc also targets Nvidia (HIP_PLATFORM=nvidia), or swap the hip
//           prefix for cuda, hipblas for cublas, and build with
//           nvcc -O2 ... -lcublas

#include <hip/hip_runtime.h>
#include <hipblas/hipblas.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

#define HIP_CHECK(call)                                                   \
  do {                                                                    \
    hipError_t err_ = (call);                                             \
    if (err_ != hipSuccess) {                                             \
      std::fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__,             \
                   hipGetErrorString(err_));                              \
      std::exit(1);                                                       \
    }                                                                     \
  } while (0)

#define HIPBLAS_CHECK(call)                                               \
  do {                                                                    \
    hipblasStatus_t st_ = (call);                                         \
    if (st_ != HIPBLAS_STATUS_SUCCESS) {                                  \
      std::fprintf(stderr, "%s:%d: hipBLAS error %d\n", __FILE__,         \
                   __LINE__, (int)st_);                                   \
      std::exit(1);                                                       \
    }                                                                     \
  } while (0)

// The kernel everyone writes first: one thread per output element, each
// walking a full row of A and a full column of B straight from global
// memory. Row-major, C = A * B, all n x n.
__global__ void naiveGemm(const float *A, const float *B, float *C, int n) {
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  if (row < n && col < n) {
    float acc = 0.0f;
    for (int k = 0; k < n; ++k) acc += A[row * n + k] * B[k * n + col];
    C[row * n + col] = acc;
  }
}

int main() {
  hipDeviceProp_t prop;
  HIP_CHECK(hipGetDeviceProperties(&prop, 0));
  std::printf("Device: %s\n\n", prop.name);

  hipblasHandle_t handle;
  HIPBLAS_CHECK(hipblasCreate(&handle));
  hipEvent_t start, stop;
  HIP_CHECK(hipEventCreate(&start));
  HIP_CHECK(hipEventCreate(&stop));

  std::printf("%6s %12s %10s %12s %10s %9s %12s\n", "n", "naive ms",
              "GFLOPS", "hipBLAS ms", "GFLOPS", "speedup", "max |diff|");

  for (int n = 256; n <= 4096; n *= 2) {
    size_t elems = (size_t)n * n, bytes = elems * sizeof(float);
    std::vector<float> hA(elems), hB(elems);
    for (size_t i = 0; i < elems; ++i) {
      hA[i] = (float)((i * 7) % 13) / 13.0f - 0.5f;
      hB[i] = (float)((i * 11) % 17) / 17.0f - 0.5f;
    }
    float *A, *B, *Cn, *Cb;
    HIP_CHECK(hipMalloc(&A, bytes));
    HIP_CHECK(hipMalloc(&B, bytes));
    HIP_CHECK(hipMalloc(&Cn, bytes));
    HIP_CHECK(hipMalloc(&Cb, bytes));
    HIP_CHECK(hipMemcpy(A, hA.data(), bytes, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(B, hB.data(), bytes, hipMemcpyHostToDevice));

    const int reps = n <= 1024 ? 10 : 3;
    const double flops = 2.0 * n * (double)n * n;

    // Naive kernel.
    dim3 block(16, 16), grid((n + 15) / 16, (n + 15) / 16);
    naiveGemm<<<grid, block>>>(A, B, Cn, n);  // warm-up
    HIP_CHECK(hipEventRecord(start, 0));
    for (int r = 0; r < reps; ++r) naiveGemm<<<grid, block>>>(A, B, Cn, n);
    HIP_CHECK(hipEventRecord(stop, 0));
    HIP_CHECK(hipEventSynchronize(stop));
    HIP_CHECK(hipGetLastError());
    float naiveMs = 0.0f;
    HIP_CHECK(hipEventElapsedTime(&naiveMs, start, stop));
    naiveMs /= reps;

    // Library call. BLAS is column-major; a row-major C = A * B is the
    // column-major C^T = B^T * A^T, so pass B first and A second and no
    // transposes are needed.
    const float alpha = 1.0f, beta = 0.0f;
    HIPBLAS_CHECK(hipblasSgemm(handle, HIPBLAS_OP_N, HIPBLAS_OP_N, n, n, n,
                               &alpha, B, n, A, n, &beta, Cb, n));  // warm-up
    HIP_CHECK(hipEventRecord(start, 0));
    for (int r = 0; r < reps; ++r)
      HIPBLAS_CHECK(hipblasSgemm(handle, HIPBLAS_OP_N, HIPBLAS_OP_N, n, n, n,
                                 &alpha, B, n, A, n, &beta, Cb, n));
    HIP_CHECK(hipEventRecord(stop, 0));
    HIP_CHECK(hipEventSynchronize(stop));
    float blasMs = 0.0f;
    HIP_CHECK(hipEventElapsedTime(&blasMs, start, stop));
    blasMs /= reps;

    // Same answer? Differences come only from summation order.
    std::vector<float> hN(elems), hBl(elems);
    HIP_CHECK(hipMemcpy(hN.data(), Cn, bytes, hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(hBl.data(), Cb, bytes, hipMemcpyDeviceToHost));
    double maxDiff = 0.0;
    for (size_t i = 0; i < elems; ++i)
      maxDiff = std::fmax(maxDiff, std::fabs((double)hN[i] - hBl[i]));

    std::printf("%6d %12.3f %10.1f %12.3f %10.1f %8.1fx %12.2e\n", n, naiveMs,
                flops / (naiveMs * 1e6), blasMs, flops / (blasMs * 1e6),
                naiveMs / blasMs, maxDiff);

    HIP_CHECK(hipFree(A));
    HIP_CHECK(hipFree(B));
    HIP_CHECK(hipFree(Cn));
    HIP_CHECK(hipFree(Cb));
  }

  HIPBLAS_CHECK(hipblasDestroy(handle));
  HIP_CHECK(hipEventDestroy(start));
  HIP_CHECK(hipEventDestroy(stop));
  return 0;
}
