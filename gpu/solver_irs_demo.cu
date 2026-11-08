// solver_irs_demo.cu
// Companion example for: topnotchnote.com/gpu/solver_libraries.html
//
// cuSOLVER's iterative-refinement solvers (cusolverDn<t1><t2>gesv) factor
// the matrix in a lower precision, then refine the answer back to the main
// precision. This compares, for one random FP64 system:
//   DDgesv : FP64 main precision, FP64 factorization (no mixed precision)
//   DSgesv : FP64 main precision, FP32 factorization + refinement
// and reports time, refinement iterations and the FP64 residual.
// CUDA only: hipSOLVER offers the same-precision variants (DDgesv, SSgesv)
// but no mixed-precision pairs.
//
// Compile: nvcc -O2 -o irs solver_irs_demo.cu -lcusolver

#include <cuda_runtime.h>
#include <cusolverDn.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

#define CUDA_CHECK(call)                                                  \
  do {                                                                    \
    cudaError_t e_ = (call);                                              \
    if (e_ != cudaSuccess) {                                              \
      std::fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__,             \
                   cudaGetErrorString(e_));                               \
      std::exit(1);                                                       \
    }                                                                     \
  } while (0)

#define SOLVER_CHECK(call)                                                \
  do {                                                                    \
    cusolverStatus_t s_ = (call);                                         \
    if (s_ != CUSOLVER_STATUS_SUCCESS) {                                  \
      std::fprintf(stderr, "%s:%d: cuSOLVER status %d\n", __FILE__,       \
                   __LINE__, (int)s_);                                    \
      std::exit(1);                                                       \
    }                                                                     \
  } while (0)

typedef cusolverStatus_t (*gesv_size_fn)(cusolverDnHandle_t, cusolver_int_t, cusolver_int_t,
                                         double*, cusolver_int_t, cusolver_int_t*, double*,
                                         cusolver_int_t, double*, cusolver_int_t, void*, size_t*);
typedef cusolverStatus_t (*gesv_fn)(cusolverDnHandle_t, cusolver_int_t, cusolver_int_t, double*,
                                    cusolver_int_t, cusolver_int_t*, double*, cusolver_int_t,
                                    double*, cusolver_int_t, void*, size_t, cusolver_int_t*,
                                    cusolver_int_t*);

int main() {
  const int n = 4096;
  std::mt19937 rng(1);
  std::uniform_real_distribution<double> u(-1.0, 1.0);
  std::vector<double> A((size_t)n * n), b(n, 1.0), x(n);
  for (auto& v : A) v = u(rng);
  for (int i = 0; i < n; ++i) A[(size_t)i * n + i] += n * 0.05;  // keep it well conditioned

  cusolverDnHandle_t h;
  SOLVER_CHECK(cusolverDnCreate(&h));
  double *dA, *dB, *dX;
  cusolver_int_t *dPiv, *dInfo;
  CUDA_CHECK(cudaMalloc(&dA, sizeof(double) * A.size()));
  CUDA_CHECK(cudaMalloc(&dB, sizeof(double) * n));
  CUDA_CHECK(cudaMalloc(&dX, sizeof(double) * n));
  CUDA_CHECK(cudaMalloc(&dPiv, sizeof(cusolver_int_t) * n));
  CUDA_CHECK(cudaMalloc(&dInfo, sizeof(cusolver_int_t)));

  struct { const char* name; gesv_size_fn size; gesv_fn solve; } runs[] = {
      {"DDgesv (FP64 factor)       ", cusolverDnDDgesv_bufferSize, cusolverDnDDgesv},
      {"DSgesv (FP32 factor + IR)  ", cusolverDnDSgesv_bufferSize, cusolverDnDSgesv},
  };
  std::printf("n = %d, one right-hand side, FP64 answer wanted\n", n);
  for (auto& r : runs) {
    size_t lwork = 0;
    SOLVER_CHECK(r.size(h, n, 1, dA, n, dPiv, dB, n, dX, n, nullptr, &lwork));
    void* dW;
    CUDA_CHECK(cudaMalloc(&dW, lwork));
    cusolver_int_t iters = 0, info = 0;
    float ms = 0;
    for (int rep = 0; rep < 2; ++rep) {  // first run warms up
      CUDA_CHECK(cudaMemcpy(dA, A.data(), sizeof(double) * A.size(), cudaMemcpyHostToDevice));
      CUDA_CHECK(cudaMemcpy(dB, b.data(), sizeof(double) * n, cudaMemcpyHostToDevice));
      cudaEvent_t e0, e1;
      CUDA_CHECK(cudaEventCreate(&e0));
      CUDA_CHECK(cudaEventCreate(&e1));
      CUDA_CHECK(cudaEventRecord(e0));
      SOLVER_CHECK(r.solve(h, n, 1, dA, n, dPiv, dB, n, dX, n, dW, lwork, &iters, dInfo));
      CUDA_CHECK(cudaEventRecord(e1));
      CUDA_CHECK(cudaEventSynchronize(e1));
      CUDA_CHECK(cudaEventElapsedTime(&ms, e0, e1));
    }
    CUDA_CHECK(cudaMemcpy(&info, dInfo, sizeof(info), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(x.data(), dX, sizeof(double) * n, cudaMemcpyDeviceToHost));
    double rn = 0, bn = 0;
    for (int i = 0; i < n; ++i) {
      double s = -b[i];
      for (int j = 0; j < n; ++j) s += A[(size_t)j * n + i] * x[j];
      rn += s * s;
      bn += b[i] * b[i];
    }
    // iters < 0 means refinement did not converge and the solver fell back
    // to a full main-precision factorization.
    std::printf("  %s %8.1f ms  iterations %3d  info %d  residual %.2e\n", r.name, ms,
                (int)iters, (int)info, std::sqrt(rn / bn));
    CUDA_CHECK(cudaFree(dW));
  }
  SOLVER_CHECK(cusolverDnDestroy(h));
  return 0;
}
