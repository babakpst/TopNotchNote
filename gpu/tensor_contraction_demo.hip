// tensor_contraction_demo.hip
// Companion example for: topnotchnote.com/gpu/tensor_libraries.html
//
// A tensor contraction done the "transpose, transpose, GEMM, transpose" way
// (TTGT), to show what a direct contraction library such as cuTENSOR or
// hipTensor saves you. The contraction is the one from cuTENSOR's getting
// started guide:
//
//     C[m,u,n,v] = sum over h,k of  A[m,h,k,n] * B[u,k,v,h]
//
// All tensors are stored column-major (first mode fastest). To use a GEMM,
// the summed modes (h,k) must be contiguous and in the same order in both
// inputs, and the free modes grouped, so the program:
//   1. permutes A[m,h,k,n] -> A'[m,n,h,k]   (an (m*n) x (h*k) matrix)
//   2. permutes B[u,k,v,h] -> B'[h,k,u,v]   (an (h*k) x (u*v) matrix)
//   3. GEMM: C'[m,n,u,v] = A' * B'
//   4. permutes C'[m,n,u,v] -> C[m,u,n,v]
// and times each step, for a "large" contraction (h = k = 64) and a "small"
// one (h = k = 4). It checks a few entries of C against a direct sum on the CPU.
//
// Compile (AMD):    hipcc -O2 -o main tensor_contraction_demo.hip -lhipblas
// Compile (Nvidia): HIP_PLATFORM=nvidia hipcc -O2 -o main tensor_contraction_demo.hip -lhipblas
//                   or rename hipblas -> cublas, hip -> cuda and build with
//                   nvcc -O2 -x cu -o main tensor_contraction_demo.hip -lcublas

#include <hip/hip_runtime.h>
#include <hipblas/hipblas.h>

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

#define BLAS_CHECK(call)                                                  \
  do {                                                                    \
    hipblasStatus_t s_ = (call);                                          \
    if (s_ != HIPBLAS_STATUS_SUCCESS) {                                   \
      std::fprintf(stderr, "%s:%d: hipBLAS status %d\n", __FILE__,        \
                   __LINE__, (int)s_);                                    \
      std::exit(1);                                                       \
    }                                                                     \
  } while (0)

// out has modes (in_mode[p0], in_mode[p1], in_mode[p2], in_mode[p3]).
// One thread per output element; writes are coalesced, reads are not.
struct Perm { int ext[4]; int p[4]; };

__global__ void permute4(const float* __restrict__ in, float* __restrict__ out, Perm P,
                         long long total) {
  long long o = blockIdx.x * (long long)blockDim.x + threadIdx.x;
  if (o >= total) return;
  long long in_stride[4] = {1, P.ext[0], (long long)P.ext[0] * P.ext[1],
                            (long long)P.ext[0] * P.ext[1] * P.ext[2]};
  long long rest = o, src = 0;
  for (int d = 0; d < 4; ++d) {  // output mode d is input mode P.p[d]
    int e = P.ext[P.p[d]];
    src += (rest % e) * in_stride[P.p[d]];
    rest /= e;
  }
  out[o] = in[src];
}

struct Timer {
  hipEvent_t a, b;
  Timer() { HIP_CHECK(hipEventCreate(&a)); HIP_CHECK(hipEventCreate(&b)); }
  void start() { HIP_CHECK(hipEventRecord(a)); }
  float stop() {
    float ms;
    HIP_CHECK(hipEventRecord(b));
    HIP_CHECK(hipEventSynchronize(b));
    HIP_CHECK(hipEventElapsedTime(&ms, a, b));
    return ms;
  }
};

static void run(hipblasHandle_t blas, int m, int n, int u, int v, int h, int k) {
  const long long nA = (long long)m * h * k * n, nB = (long long)u * k * v * h,
                  nC = (long long)m * u * n * v;
  std::vector<float> A(nA), B(nB);
  std::mt19937 rng(7);
  std::uniform_real_distribution<float> U(-1.f, 1.f);
  for (auto& x : A) x = U(rng);
  for (auto& x : B) x = U(rng);

  float *dA, *dB, *dA2, *dB2, *dC2, *dC;
  HIP_CHECK(hipMalloc(&dA, nA * 4)); HIP_CHECK(hipMalloc(&dA2, nA * 4));
  HIP_CHECK(hipMalloc(&dB, nB * 4)); HIP_CHECK(hipMalloc(&dB2, nB * 4));
  HIP_CHECK(hipMalloc(&dC2, nC * 4)); HIP_CHECK(hipMalloc(&dC, nC * 4));
  HIP_CHECK(hipMemcpy(dA, A.data(), nA * 4, hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(dB, B.data(), nB * 4, hipMemcpyHostToDevice));

  // A modes (m,h,k,n) = (0,1,2,3) -> (m,n,h,k) = (0,3,1,2)
  Perm pA{{m, h, k, n}, {0, 3, 1, 2}};
  // B modes (u,k,v,h) = (0,1,2,3) -> (h,k,u,v) = (3,1,0,2)
  Perm pB{{u, k, v, h}, {3, 1, 0, 2}};
  // C' modes (m,n,u,v) = (0,1,2,3) -> (m,u,n,v) = (0,2,1,3)
  Perm pC{{m, n, u, v}, {0, 2, 1, 3}};
  const int T = 256;
  auto blocks = [&](long long t) { return (unsigned)((t + T - 1) / T); };
  const int M = m * n, K = h * k, N = u * v;
  const float one = 1.f, zero = 0.f;

  float tA = 0, tB = 0, tG = 0, tC = 0;
  Timer t;
  for (int rep = 0; rep < 2; ++rep) {  // first pass warms up
    t.start(); permute4<<<blocks(nA), T>>>(dA, dA2, pA, nA); tA = t.stop();
    t.start(); permute4<<<blocks(nB), T>>>(dB, dB2, pB, nB); tB = t.stop();
    t.start();
    BLAS_CHECK(hipblasSgemm(blas, HIPBLAS_OP_N, HIPBLAS_OP_N, M, N, K, &one, dA2, M, dB2, K,
                            &zero, dC2, M));
    tG = t.stop();
    t.start(); permute4<<<blocks(nC), T>>>(dC2, dC, pC, nC); tC = t.stop();
  }
  HIP_CHECK(hipGetLastError());

  // Spot-check against the definition.
  std::vector<float> C(nC);
  HIP_CHECK(hipMemcpy(C.data(), dC, nC * 4, hipMemcpyDeviceToHost));
  double worst = 0;
  for (int s = 0; s < 20; ++s) {
    int im = s * 7 % m, iu = s * 5 % u, in = s * 3 % n, iv = s * 11 % v;
    double ref = 0;
    for (int ih = 0; ih < h; ++ih)
      for (int ik = 0; ik < k; ++ik)
        ref += (double)A[im + (long long)m * (ih + (long long)h * (ik + (long long)k * in))] *
               B[iu + (long long)u * (ik + (long long)k * (iv + (long long)v * ih))];
    double got = C[im + (long long)m * (iu + (long long)u * (in + (long long)n * iv))];
    worst = std::fmax(worst, std::fabs(got - ref));
  }

  double flops = 2.0 * M * (double)N * K;
  float total = tA + tB + tG + tC;
  std::printf("m=n=u=v=%d, h=k=%d   GEMM %d x %d x %d, %.1f GFLOP\n", m, h, M, N, K, flops / 1e9);
  std::printf("  permute A %7.2f ms   permute B %7.2f ms   GEMM %8.2f ms   permute C %7.2f ms\n",
              tA, tB, tG, tC);
  std::printf("  total %.2f ms, of which permutes %.0f%%; GEMM alone %.0f GFLOP/s; max error %.1e\n\n",
              total, 100.0 * (tA + tB + tC) / total, flops / tG / 1e6, worst);

  HIP_CHECK(hipFree(dA)); HIP_CHECK(hipFree(dA2)); HIP_CHECK(hipFree(dB));
  HIP_CHECK(hipFree(dB2)); HIP_CHECK(hipFree(dC2)); HIP_CHECK(hipFree(dC));
}

int main() {
  hipblasHandle_t blas;
  BLAS_CHECK(hipblasCreate(&blas));
  run(blas, 64, 64, 64, 64, 64, 64);  // contraction over 4,096 values: GEMM-heavy
  run(blas, 64, 64, 64, 64, 4, 4);    // contraction over 16 values: data-movement-heavy
  BLAS_CHECK(hipblasDestroy(blas));
  return 0;
}
