// solver_demo.hip
// Companion example for: topnotchnote.com/gpu/solver_libraries.html
//
// Dense linear algebra on the GPU with hipSOLVER, which forwards to rocSOLVER
// on AMD and to cuSOLVER on Nvidia. It uses hipSOLVER's compatibility API
// (hipsolverDn*), whose signatures match cuSOLVER's one for one. Experiments:
//   1. solve A x = b with LU (getrf + getrs), FP32 and FP64, several sizes
//   2. the same system made symmetric positive definite, solved with
//      Cholesky (potrf + potrs) instead of LU
//   3. a singular matrix: the call returns success, and only devInfo says
//      the factorization failed
// Every solve is checked with the relative residual ||A x - b|| / ||b||,
// computed on the host in double precision.
//
// Compile (AMD):    hipcc -O2 -o main solver_demo.hip -lhipsolver
// Compile (Nvidia): HIP_PLATFORM=nvidia hipcc -O2 -o main solver_demo.hip -lhipsolver
//                   or rename hipsolverDn -> cusolverDn, hip -> cuda and
//                   build with nvcc -O2 -x cu -o main solver_demo.hip -lcusolver
// For production code on AMD, move to the regular API (drop "Dn") or call
// rocSOLVER directly; the hipSOLVER docs say both are faster than this one.

#include <hip/hip_runtime.h>
#include <hipsolver/hipsolver.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
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

#define SOLVER_CHECK(call)                                                \
  do {                                                                    \
    hipsolverStatus_t s_ = (call);                                        \
    if (s_ != HIPSOLVER_STATUS_SUCCESS) {                                 \
      std::fprintf(stderr, "%s:%d: hipSOLVER status %d\n", __FILE__,      \
                   __LINE__, (int)s_);                                    \
      std::exit(1);                                                       \
    }                                                                     \
  } while (0)

// Typed wrappers so one template covers float and double.
static hipsolverStatus_t getrf_size(hipsolverDnHandle_t h, int n, float* A, int* lw) {
  return hipsolverDnSgetrf_bufferSize(h, n, n, A, n, lw);
}
static hipsolverStatus_t getrf_size(hipsolverDnHandle_t h, int n, double* A, int* lw) {
  return hipsolverDnDgetrf_bufferSize(h, n, n, A, n, lw);
}
static hipsolverStatus_t getrf(hipsolverDnHandle_t h, int n, float* A, float* w, int* piv, int* info) {
  return hipsolverDnSgetrf(h, n, n, A, n, w, piv, info);
}
static hipsolverStatus_t getrf(hipsolverDnHandle_t h, int n, double* A, double* w, int* piv, int* info) {
  return hipsolverDnDgetrf(h, n, n, A, n, w, piv, info);
}
static hipsolverStatus_t getrs(hipsolverDnHandle_t h, int n, float* A, int* piv, float* b, int* info) {
  return hipsolverDnSgetrs(h, HIPSOLVER_OP_N, n, 1, A, n, piv, b, n, info);
}
static hipsolverStatus_t getrs(hipsolverDnHandle_t h, int n, double* A, int* piv, double* b, int* info) {
  return hipsolverDnDgetrs(h, HIPSOLVER_OP_N, n, 1, A, n, piv, b, n, info);
}
static hipsolverStatus_t potrf_size(hipsolverDnHandle_t h, int n, float* A, int* lw) {
  return hipsolverDnSpotrf_bufferSize(h, HIPSOLVER_FILL_MODE_LOWER, n, A, n, lw);
}
static hipsolverStatus_t potrf_size(hipsolverDnHandle_t h, int n, double* A, int* lw) {
  return hipsolverDnDpotrf_bufferSize(h, HIPSOLVER_FILL_MODE_LOWER, n, A, n, lw);
}
static hipsolverStatus_t potrf(hipsolverDnHandle_t h, int n, float* A, float* w, int lw, int* info) {
  return hipsolverDnSpotrf(h, HIPSOLVER_FILL_MODE_LOWER, n, A, n, w, lw, info);
}
static hipsolverStatus_t potrf(hipsolverDnHandle_t h, int n, double* A, double* w, int lw, int* info) {
  return hipsolverDnDpotrf(h, HIPSOLVER_FILL_MODE_LOWER, n, A, n, w, lw, info);
}
static hipsolverStatus_t potrs(hipsolverDnHandle_t h, int n, float* A, float* b, int* info) {
  return hipsolverDnSpotrs(h, HIPSOLVER_FILL_MODE_LOWER, n, 1, A, n, b, n, info);
}
static hipsolverStatus_t potrs(hipsolverDnHandle_t h, int n, double* A, double* b, int* info) {
  return hipsolverDnDpotrs(h, HIPSOLVER_FILL_MODE_LOWER, n, 1, A, n, b, n, info);
}

// Column-major n x n matrix with entries in [-1, 1]. spd = true returns
// M^T M + n I, which is symmetric positive definite.
static std::vector<double> make_matrix(int n, bool spd, unsigned seed) {
  std::mt19937 rng(seed);
  std::uniform_real_distribution<double> u(-1.0, 1.0);
  std::vector<double> M((size_t)n * n);
  for (auto& v : M) v = u(rng);
  if (!spd) return M;
  std::vector<double> S((size_t)n * n, 0.0);
  for (int j = 0; j < n; ++j)
    for (int i = 0; i <= j; ++i) {
      double s = 0.0;
      for (int k = 0; k < n; ++k) s += M[(size_t)i * n + k] * M[(size_t)j * n + k];
      S[(size_t)j * n + i] = S[(size_t)i * n + j] = s;
    }
  for (int i = 0; i < n; ++i) S[(size_t)i * n + i] += n;
  return S;
}

static double residual(int n, const std::vector<double>& A, const std::vector<double>& x,
                       const std::vector<double>& b) {
  double rn = 0.0, bn = 0.0;
  for (int i = 0; i < n; ++i) {
    double r = -b[i];
    for (int j = 0; j < n; ++j) r += A[(size_t)j * n + i] * x[j];
    rn += r * r;
    bn += b[i] * b[i];
  }
  return std::sqrt(rn / bn);
}

struct Result { double factor_ms, solve_ms, resid; int info; };

// Factor and solve once to warm up, then time a second, identical run.
template <typename T>
static Result run(hipsolverDnHandle_t h, int n, const std::vector<double>& A64,
                  const std::vector<double>& b64, bool cholesky) {
  std::vector<T> A(A64.begin(), A64.end()), b(b64.begin(), b64.end());
  T *dA, *dB, *dW;
  int *dPiv, *dInfo, lwork = 0;
  HIP_CHECK(hipMalloc(&dA, sizeof(T) * A.size()));
  HIP_CHECK(hipMalloc(&dB, sizeof(T) * n));
  HIP_CHECK(hipMalloc(&dPiv, sizeof(int) * n));
  HIP_CHECK(hipMalloc(&dInfo, sizeof(int)));
  HIP_CHECK(hipMemcpy(dA, A.data(), sizeof(T) * A.size(), hipMemcpyHostToDevice));
  if (cholesky) SOLVER_CHECK(potrf_size(h, n, dA, &lwork));
  else          SOLVER_CHECK(getrf_size(h, n, dA, &lwork));
  HIP_CHECK(hipMalloc(&dW, sizeof(T) * (lwork > 0 ? lwork : 1)));

  hipEvent_t e0, e1, e2;
  HIP_CHECK(hipEventCreate(&e0));
  HIP_CHECK(hipEventCreate(&e1));
  HIP_CHECK(hipEventCreate(&e2));
  Result r{};
  for (int rep = 0; rep < 2; ++rep) {
    HIP_CHECK(hipMemcpy(dA, A.data(), sizeof(T) * A.size(), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(dB, b.data(), sizeof(T) * n, hipMemcpyHostToDevice));
    HIP_CHECK(hipEventRecord(e0));
    if (cholesky) SOLVER_CHECK(potrf(h, n, dA, dW, lwork, dInfo));
    else          SOLVER_CHECK(getrf(h, n, dA, dW, dPiv, dInfo));
    HIP_CHECK(hipEventRecord(e1));
    HIP_CHECK(hipMemcpy(&r.info, dInfo, sizeof(int), hipMemcpyDeviceToHost));
    if (r.info != 0) break;  // factorization failed: nothing to solve with
    if (cholesky) SOLVER_CHECK(potrs(h, n, dA, dB, dInfo));
    else          SOLVER_CHECK(getrs(h, n, dA, dPiv, dB, dInfo));
    HIP_CHECK(hipEventRecord(e2));
    HIP_CHECK(hipEventSynchronize(e2));
    float f, s;
    HIP_CHECK(hipEventElapsedTime(&f, e0, e1));
    HIP_CHECK(hipEventElapsedTime(&s, e1, e2));
    r.factor_ms = f;
    r.solve_ms = s;
  }
  if (r.info == 0) {
    std::vector<T> x(n);
    HIP_CHECK(hipMemcpy(x.data(), dB, sizeof(T) * n, hipMemcpyDeviceToHost));
    r.resid = residual(n, A64, std::vector<double>(x.begin(), x.end()), b64);
  }
  HIP_CHECK(hipFree(dA)); HIP_CHECK(hipFree(dB)); HIP_CHECK(hipFree(dW));
  HIP_CHECK(hipFree(dPiv)); HIP_CHECK(hipFree(dInfo));
  return r;
}

int main() {
  hipsolverDnHandle_t h;
  SOLVER_CHECK(hipsolverDnCreate(&h));

  std::printf("1. LU solve (getrf + getrs) of a random n x n system\n");
  std::printf("%6s %5s %12s %10s %10s %12s\n", "n", "type", "factor ms", "GFLOP/s",
              "solve ms", "residual");
  for (int n : {1024, 2048, 4096}) {
    auto A = make_matrix(n, false, 1);
    std::vector<double> b(n, 1.0);
    double flops = 2.0 / 3.0 * n * (double)n * n;
    Result s = run<float>(h, n, A, b, false);
    Result d = run<double>(h, n, A, b, false);
    std::printf("%6d %5s %12.2f %10.1f %10.3f %12.2e\n", n, "FP32", s.factor_ms,
                flops / s.factor_ms / 1e6, s.solve_ms, s.resid);
    std::printf("%6d %5s %12.2f %10.1f %10.3f %12.2e\n", n, "FP64", d.factor_ms,
                flops / d.factor_ms / 1e6, d.solve_ms, d.resid);
  }

  std::printf("\n2. Symmetric positive definite system, n = 2048: LU vs Cholesky\n");
  {
    int n = 2048;
    auto A = make_matrix(n, true, 2);
    std::vector<double> b(n, 1.0);
    Result lu = run<float>(h, n, A, b, false);
    Result ch = run<float>(h, n, A, b, true);
    std::printf("  LU       (getrf) factor %8.2f ms, residual %.2e\n", lu.factor_ms, lu.resid);
    std::printf("  Cholesky (potrf) factor %8.2f ms, residual %.2e\n", ch.factor_ms, ch.resid);
  }

  std::printf("\n3. A singular matrix (row 5 is a copy of row 2), n = 512\n");
  {
    int n = 512;
    auto A = make_matrix(n, false, 3);
    for (int j = 0; j < n; ++j) A[(size_t)j * n + 4] = A[(size_t)j * n + 1];
    std::vector<double> b(n, 1.0);
    Result r = run<double>(h, n, A, b, false);
    std::printf("  getrf returned HIPSOLVER_STATUS_SUCCESS, devInfo = %d\n", r.info);
    std::printf("  devInfo > 0 means U(%d,%d) is exactly zero: the matrix is singular\n",
                r.info, r.info);
  }

  SOLVER_CHECK(hipsolverDnDestroy(h));
  return 0;
}
