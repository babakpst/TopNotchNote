// structured_sparsity_demo.hip
// Companion example for: topnotchnote.com/gpu/structured_sparsity.html
//
// 2:4 structured sparsity: in every group of 4 consecutive values along K in
// a row of matrix A, 2 are zero. Sparse tensor cores (Nvidia sm_80+) and AMD's
// V_SMFMAC instructions (CDNA 3) store only the 2 kept values plus a 2-bit
// index for each, and skip the zero multiplies.
//
// This program does the whole pipeline in plain code, so it runs on any GPU:
//   1. magnitude-prune A to 2:4 (keep the 2 largest |values| of each 4);
//   2. compress it into values + 2-bit indices, and count the bytes;
//   3. multiply C = A x B three ways on the GPU:
//        dense, original A    (the unpruned reference)
//        dense, pruned A      (same math the sparse unit does, zeros included)
//        compressed A         (reads values + indices, does half the FMAs)
//   4. check compressed == pruned-dense, and measure how far pruning moved
//      the output from the unpruned reference.
// Without sparse hardware the compressed kernel is not expected to be 2x
// faster: it does half the multiplies but gathers B rows through indices.
// The point is the format, the storage and the accuracy cost of pruning.
//
// Compile:  hipcc -O2 -o main structured_sparsity_demo.hip
// Nvidia:   hipcc also targets Nvidia (HIP_PLATFORM=nvidia), or swap the hip
//           prefix for cuda in the API calls and build with nvcc.

#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
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

constexpr int M = 1024, N = 1024, K = 2048;  // A is M x K, B is K x N

// Dense C = A x B, one thread per output element (row-major).
__global__ void gemm_dense(const float* A, const float* B, float* C) {
  int r = blockIdx.y * blockDim.y + threadIdx.y;
  int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (r >= M || c >= N) return;
  float acc = 0.f;
  for (int k = 0; k < K; ++k) acc += A[r * K + k] * B[k * N + c];
  C[r * N + c] = acc;
}

// Compressed A: for each row, K/2 kept values and K/2 indices (0..3) that
// say where in its group of 4 each value sat. Half the multiply-adds.
__global__ void gemm_24(const float* Av, const uint8_t* Ai, const float* B,
                        float* C) {
  int r = blockIdx.y * blockDim.y + threadIdx.y;
  int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (r >= M || c >= N) return;
  float acc = 0.f;
  for (int j = 0; j < K / 2; ++j) {
    int k = (j / 2) * 4 + Ai[r * (K / 2) + j];  // group start + 2-bit index
    acc += Av[r * (K / 2) + j] * B[k * N + c];
  }
  C[r * N + c] = acc;
}

static float time_ms(void (*launch)(void*), void* arg) {
  hipEvent_t s, e;
  HIP_CHECK(hipEventCreate(&s));
  HIP_CHECK(hipEventCreate(&e));
  launch(arg);  // warm-up
  std::vector<float> t;
  for (int i = 0; i < 5; ++i) {
    HIP_CHECK(hipEventRecord(s));
    launch(arg);
    HIP_CHECK(hipEventRecord(e));
    HIP_CHECK(hipEventSynchronize(e));
    float ms;
    HIP_CHECK(hipEventElapsedTime(&ms, s, e));
    t.push_back(ms);
  }
  std::sort(t.begin(), t.end());
  return t[2];
}

struct Bufs { float *A, *Ap, *Av, *B, *C; uint8_t* Ai; };
static const dim3 kBlock(16, 16), kGrid(N / 16, M / 16);
static void run_dense(void* p) {
  Bufs* b = (Bufs*)p;
  hipLaunchKernelGGL(gemm_dense, kGrid, kBlock, 0, 0, b->Ap, b->B, b->C);
}
static void run_24(void* p) {
  Bufs* b = (Bufs*)p;
  hipLaunchKernelGGL(gemm_24, kGrid, kBlock, 0, 0, b->Av, b->Ai, b->B, b->C);
}

int main() {
  std::mt19937 rng(42);
  std::normal_distribution<float> nd(0.f, 1.f);
  std::vector<float> A(size_t(M) * K), B(size_t(K) * N);
  for (auto& v : A) v = nd(rng);
  for (auto& v : B) v = nd(rng);

  // 1-2. Prune to 2:4 by magnitude and compress.
  std::vector<float> Ap(A.size()), Av(A.size() / 2);
  std::vector<uint8_t> Ai(A.size() / 2);
  double kept_sq = 0, all_sq = 0;
  for (int r = 0; r < M; ++r)
    for (int g = 0; g < K / 4; ++g) {
      const float* q = &A[size_t(r) * K + g * 4];
      int idx[4] = {0, 1, 2, 3};
      std::sort(idx, idx + 4, [&](int a, int b) { return std::fabs(q[a]) > std::fabs(q[b]); });
      int k0 = std::min(idx[0], idx[1]), k1 = std::max(idx[0], idx[1]);
      for (int i = 0; i < 4; ++i) {
        all_sq += double(q[i]) * q[i];
        Ap[size_t(r) * K + g * 4 + i] = (i == k0 || i == k1) ? q[i] : 0.f;
      }
      kept_sq += double(q[k0]) * q[k0] + double(q[k1]) * q[k1];
      size_t j = size_t(r) * (K / 2) + g * 2;
      Av[j] = q[k0]; Ai[j] = uint8_t(k0);
      Av[j + 1] = q[k1]; Ai[j + 1] = uint8_t(k1);
    }

  std::printf("A is %d x %d; pruned to 2:4 along K by magnitude\n", M, K);
  std::printf("  share of A's squared magnitude kept: %.1f%%\n", 100.0 * kept_sq / all_sq);
  // Storage if A were FP16: dense = 16 bits per element; compressed =
  // half the elements at 16 bits + a 2-bit index per kept element.
  double dense_bits = 16.0 * A.size();
  double comp_bits = 16.0 * Av.size() + 2.0 * Av.size();
  std::printf("  FP16 storage: dense %.1f MiB, compressed %.1f MiB (%.1f%%)\n",
              dense_bits / 8 / 1048576, comp_bits / 8 / 1048576, 100.0 * comp_bits / dense_bits);

  Bufs b;
  HIP_CHECK(hipMalloc(&b.A, A.size() * 4));
  HIP_CHECK(hipMalloc(&b.Ap, A.size() * 4));
  HIP_CHECK(hipMalloc(&b.Av, Av.size() * 4));
  HIP_CHECK(hipMalloc(&b.Ai, Ai.size()));
  HIP_CHECK(hipMalloc(&b.B, B.size() * 4));
  HIP_CHECK(hipMalloc(&b.C, size_t(M) * N * 4));
  HIP_CHECK(hipMemcpy(b.A, A.data(), A.size() * 4, hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(b.Ap, Ap.data(), A.size() * 4, hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(b.Av, Av.data(), Av.size() * 4, hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(b.Ai, Ai.data(), Ai.size(), hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(b.B, B.data(), B.size() * 4, hipMemcpyHostToDevice));

  std::vector<float> C_ref(size_t(M) * N), C_pr(size_t(M) * N), C_24(size_t(M) * N);
  // Unpruned reference: dense kernel on the original A.
  hipLaunchKernelGGL(gemm_dense, kGrid, kBlock, 0, 0, b.A, b.B, b.C);
  HIP_CHECK(hipMemcpy(C_ref.data(), b.C, C_ref.size() * 4, hipMemcpyDeviceToHost));
  float t_dense = time_ms(run_dense, &b);
  HIP_CHECK(hipMemcpy(C_pr.data(), b.C, C_pr.size() * 4, hipMemcpyDeviceToHost));
  float t_24 = time_ms(run_24, &b);
  HIP_CHECK(hipMemcpy(C_24.data(), b.C, C_24.size() * 4, hipMemcpyDeviceToHost));

  double max_diff = 0, err = 0, ref = 0;
  for (size_t i = 0; i < C_ref.size(); ++i) {
    max_diff = std::max(max_diff, double(std::fabs(C_24[i] - C_pr[i])));
    err += double(C_pr[i] - C_ref[i]) * (C_pr[i] - C_ref[i]);
    ref += double(C_ref[i]) * C_ref[i];
  }
  std::printf("\nC = A x B, %d x %d x %d, naive kernels (no sparse hardware used)\n", M, N, K);
  std::printf("  dense on pruned A : %8.2f ms\n", t_dense);
  std::printf("  compressed 2:4    : %8.2f ms  (%.2fx)\n", t_24, t_dense / t_24);
  std::printf("  compressed vs pruned-dense, max |diff|: %.2e\n", max_diff);
  std::printf("  pruned vs unpruned output, relative error: %.1f%%\n",
              100.0 * std::sqrt(err / ref));
  std::printf("  (random, untrained A: real models are fine-tuned after pruning)\n");
  return 0;
}
