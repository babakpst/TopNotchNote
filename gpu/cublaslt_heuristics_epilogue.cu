// cublaslt_heuristics_epilogue.cu
// Companion example for: topnotchnote.com/gpu/cublaslt_hipblaslt.html
//
// Two things the "Lt" GEMM libraries do that classic BLAS does not, measured
// with cuBLASLt on an FP32 problem:
//   1. Heuristics: ask the library for a ranked list of candidate kernels
//      for one shape, then time every candidate. Shows how good the
//      heuristic's first pick is, how much the candidates differ, and what
//      the query itself costs on the host.
//   2. Epilogues: D = GELU(W * X + bias) done two ways. Fused, as one
//      cublasLtMatmul with CUBLASLT_EPILOGUE_GELU_BIAS, and unfused, as a
//      plain matmul followed by a separate bias + GELU kernel.
//
// hipBLASLt's C API mirrors this one name for name (cublasLt -> hipblasLt,
// CUBLASLT_ -> HIPBLASLT_, cuda -> hip, CUBLAS_COMPUTE_ -> HIPBLAS_COMPUTE_,
// CUDA_R_32F -> HIP_R_32F). hipBLASLt runs on AMD GPUs only, so the port was
// not run for this page.
//
// Compile:  nvcc -O2 -o main cublaslt_heuristics_epilogue.cu -lcublasLt
// Run:      ./main

#include <cublasLt.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

#define CUDA_CHECK(call)                                                      \
  do {                                                                        \
    cudaError_t e_ = (call);                                                  \
    if (e_ != cudaSuccess) {                                                  \
      std::fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__,                 \
                   cudaGetErrorString(e_));                                   \
      std::exit(1);                                                           \
    }                                                                         \
  } while (0)

#define LT_CHECK(call)                                                        \
  do {                                                                        \
    cublasStatus_t s_ = (call);                                               \
    if (s_ != CUBLAS_STATUS_SUCCESS) {                                        \
      std::fprintf(stderr, "%s:%d: cuBLASLt status %d\n", __FILE__, __LINE__, \
                   (int)s_);                                                  \
      std::exit(1);                                                           \
    }                                                                         \
  } while (0)

static const size_t kWorkspaceBytes = 32u << 20;  // 32 MiB

// Column-major problem D(m x n) = op(A)(m x k) * B(k x n), plus layouts.
struct Problem {
  int m, n, k;
  cublasLtMatmulDesc_t op = nullptr;
  cublasLtMatrixLayout_t a = nullptr, b = nullptr, c = nullptr;

  Problem(int m_, int n_, int k_, cublasLtEpilogue_t epi, const float* bias)
      : m(m_), n(n_), k(k_) {
    LT_CHECK(cublasLtMatmulDescCreate(&op, CUBLAS_COMPUTE_32F, CUDA_R_32F));
    LT_CHECK(cublasLtMatmulDescSetAttribute(
        op, CUBLASLT_MATMUL_DESC_EPILOGUE, &epi, sizeof(epi)));
    if (bias)
      LT_CHECK(cublasLtMatmulDescSetAttribute(
          op, CUBLASLT_MATMUL_DESC_BIAS_POINTER, &bias, sizeof(bias)));
    LT_CHECK(cublasLtMatrixLayoutCreate(&a, CUDA_R_32F, m, k, m));
    LT_CHECK(cublasLtMatrixLayoutCreate(&b, CUDA_R_32F, k, n, k));
    LT_CHECK(cublasLtMatrixLayoutCreate(&c, CUDA_R_32F, m, n, m));
  }
  ~Problem() {
    cublasLtMatrixLayoutDestroy(a);
    cublasLtMatrixLayoutDestroy(b);
    cublasLtMatrixLayoutDestroy(c);
    cublasLtMatmulDescDestroy(op);
  }
  double gflop() const { return 2.0 * m * n * k * 1e-9; }
};

static int heuristics(cublasLtHandle_t lt, const Problem& p,
                      cublasLtMatmulPreference_t pref, int want,
                      std::vector<cublasLtMatmulHeuristicResult_t>& out) {
  out.resize(want);
  int got = 0;
  LT_CHECK(cublasLtMatmulAlgoGetHeuristic(lt, p.op, p.a, p.b, p.c, p.c, pref,
                                          want, out.data(), &got));
  out.resize(got);
  return got;
}

// Median time in ms of `iters` launches of one algorithm.
static float time_algo(cublasLtHandle_t lt, const Problem& p,
                       const cublasLtMatmulAlgo_t* algo, const float* A,
                       const float* B, float* D, void* ws, int iters) {
  const float alpha = 1.f, beta = 0.f;
  cudaEvent_t e0, e1;
  CUDA_CHECK(cudaEventCreate(&e0));
  CUDA_CHECK(cudaEventCreate(&e1));
  std::vector<float> t;
  for (int i = 0; i < iters + 2; ++i) {
    CUDA_CHECK(cudaEventRecord(e0));
    LT_CHECK(cublasLtMatmul(lt, p.op, &alpha, A, p.a, B, p.b, &beta, D, p.c,
                            D, p.c, algo, ws, kWorkspaceBytes, 0));
    CUDA_CHECK(cudaEventRecord(e1));
    CUDA_CHECK(cudaEventSynchronize(e1));
    float ms;
    CUDA_CHECK(cudaEventElapsedTime(&ms, e0, e1));
    if (i >= 2) t.push_back(ms);  // first two are warm-up
  }
  cudaEventDestroy(e0);
  cudaEventDestroy(e1);
  std::sort(t.begin(), t.end());
  return t[t.size() / 2];
}

static int cfg(const cublasLtMatmulAlgo_t& a, cublasLtMatmulAlgoConfigAttributes_t attr) {
  int v = 0;
  size_t w = 0;
  cublasLtMatmulAlgoConfigGetAttribute(&a, attr, &v, sizeof(v), &w);
  return v;
}

__global__ void bias_gelu(float* d, const float* bias, int m, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= m * n) return;
  float x = d[i] + bias[i % m];  // column-major: row index is i % m
  // tanh approximation, the formula the cuBLASLt docs give for its GELU
  d[i] = 0.5f * x * (1.f + tanhf(0.7978845608f * (x + 0.044715f * x * x * x)));
}

int main() {
  cudaDeviceProp prop;
  CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
  std::printf("GPU: %s (sm_%d%d), cuBLASLt version %zu\n", prop.name,
              prop.major, prop.minor, cublasLtGetVersion());

  cublasLtHandle_t lt;
  LT_CHECK(cublasLtCreate(&lt));
  cublasLtMatmulPreference_t pref;
  LT_CHECK(cublasLtMatmulPreferenceCreate(&pref));
  size_t wsb = kWorkspaceBytes;
  LT_CHECK(cublasLtMatmulPreferenceSetAttribute(
      pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &wsb, sizeof(wsb)));

  const size_t maxElems = 4096u * 4096u;
  float *A, *B, *D, *D2, *bias;
  void* ws;
  CUDA_CHECK(cudaMalloc(&A, maxElems * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&B, maxElems * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&D, maxElems * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&D2, maxElems * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&bias, 4096 * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&ws, kWorkspaceBytes));
  {
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> u(-1.f, 1.f);
    std::vector<float> h(maxElems);
    for (auto& x : h) x = u(rng) * 0.05f;
    CUDA_CHECK(cudaMemcpy(A, h.data(), maxElems * 4, cudaMemcpyHostToDevice));
    for (auto& x : h) x = u(rng);
    CUDA_CHECK(cudaMemcpy(B, h.data(), maxElems * 4, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(bias, h.data(), 4096 * 4, cudaMemcpyHostToDevice));
  }

  // ---- 1. heuristics: ranked candidates, each one timed --------------------
  struct Shape { const char* name; int m, n, k; };
  const Shape shapes[] = {
      {"square 2048^3", 2048, 2048, 2048},
      {"decode: 4096x4096 weight, 16 tokens", 4096, 16, 4096},
      {"prefill: 4096x1024 weight, 512 tokens", 4096, 512, 1024},
  };
  for (const Shape& s : shapes) {
    Problem p(s.m, s.n, s.k, CUBLASLT_EPILOGUE_DEFAULT, nullptr);
    std::vector<cublasLtMatmulHeuristicResult_t> r;

    // Host cost of the query. The first call for a shape runs the heuristic;
    // repeats can be answered from cuBLASLt's heuristics cache.
    auto t0 = std::chrono::steady_clock::now();
    heuristics(lt, p, pref, 1, r);
    auto t1 = std::chrono::steady_clock::now();
    heuristics(lt, p, pref, 1, r);
    auto t2 = std::chrono::steady_clock::now();
    int got = heuristics(lt, p, pref, 32, r);
    auto t3 = std::chrono::steady_clock::now();
    auto us = [](auto a, auto b) {
      return std::chrono::duration<double, std::micro>(b - a).count();
    };

    std::printf("\n== %s  (m=%d n=%d k=%d) ==\n", s.name, s.m, s.n, s.k);
    std::printf("query 1 algo: first %.1f us, repeat %.1f us; 32 algos: %.1f us"
                " -> %d returned\n", us(t0, t1), us(t1, t2), us(t2, t3), got);
    std::printf("rank  algoId tile splitK  waves  workspace   time ms   GFLOPS\n");
    float best = 1e30f;
    int bestRank = -1;
    std::vector<float> times;
    for (int i = 0; i < got; ++i) {
      float ms = time_algo(lt, p, &r[i].algo, A, B, D, ws, 15);
      times.push_back(ms);
      if (ms < best) { best = ms; bestRank = i; }
      std::printf("%4d  %6d %4d %6d %6.2f %10zu %9.3f %8.1f\n", i,
                  cfg(r[i].algo, CUBLASLT_ALGO_CONFIG_ID),
                  cfg(r[i].algo, CUBLASLT_ALGO_CONFIG_TILE_ID),
                  cfg(r[i].algo, CUBLASLT_ALGO_CONFIG_SPLITK_NUM),
                  r[i].wavesCount, r[i].workspaceSize, ms, p.gflop() / ms * 1e3);
    }
    float worst = *std::max_element(times.begin(), times.end());
    std::printf("heuristic #0: %.3f ms | fastest: rank %d, %.3f ms "
                "(#0 is %.1f%% slower) | slowest/fastest: %.2fx\n",
                times[0], bestRank, best, (times[0] / best - 1.f) * 100.f,
                worst / best);
  }

  // ---- 2. epilogue: fused GELU(W X + b) vs matmul + separate kernel --------
  {
    const int m = 4096, n = 512, k = 1024;
    Problem fused(m, n, k, CUBLASLT_EPILOGUE_GELU_BIAS, bias);
    Problem plain(m, n, k, CUBLASLT_EPILOGUE_DEFAULT, nullptr);
    std::vector<cublasLtMatmulHeuristicResult_t> rf, rp;
    if (heuristics(lt, fused, pref, 1, rf) == 0 ||
        heuristics(lt, plain, pref, 1, rp) == 0) {
      std::printf("\nno algorithm for the epilogue test on this GPU\n");
      return 0;
    }
    const float alpha = 1.f, beta = 0.f;
    cudaEvent_t e0, e1;
    CUDA_CHECK(cudaEventCreate(&e0));
    CUDA_CHECK(cudaEventCreate(&e1));
    auto run = [&](bool useFused) {
      if (useFused) {
        LT_CHECK(cublasLtMatmul(lt, fused.op, &alpha, A, fused.a, B, fused.b,
                                &beta, D, fused.c, D, fused.c, &rf[0].algo, ws,
                                kWorkspaceBytes, 0));
      } else {
        LT_CHECK(cublasLtMatmul(lt, plain.op, &alpha, A, plain.a, B, plain.b,
                                &beta, D2, plain.c, D2, plain.c, &rp[0].algo,
                                ws, kWorkspaceBytes, 0));
        bias_gelu<<<(m * n + 255) / 256, 256>>>(D2, bias, m, n);
      }
    };
    auto timeit = [&](bool useFused) {
      std::vector<float> t;
      for (int i = 0; i < 22; ++i) {
        CUDA_CHECK(cudaEventRecord(e0));
        run(useFused);
        CUDA_CHECK(cudaEventRecord(e1));
        CUDA_CHECK(cudaEventSynchronize(e1));
        float ms;
        CUDA_CHECK(cudaEventElapsedTime(&ms, e0, e1));
        if (i >= 2) t.push_back(ms);
      }
      std::sort(t.begin(), t.end());
      return t[t.size() / 2];
    };
    // Time the bias+GELU kernel alone too.
    std::vector<float> tk;
    for (int i = 0; i < 22; ++i) {
      CUDA_CHECK(cudaEventRecord(e0));
      bias_gelu<<<(m * n + 255) / 256, 256>>>(D2, bias, m, n);
      CUDA_CHECK(cudaEventRecord(e1));
      CUDA_CHECK(cudaEventSynchronize(e1));
      float ms;
      CUDA_CHECK(cudaEventElapsedTime(&ms, e0, e1));
      if (i >= 2) tk.push_back(ms);
    }
    std::sort(tk.begin(), tk.end());

    float tf = timeit(true), tu = timeit(false);
    run(true);
    run(false);
    CUDA_CHECK(cudaDeviceSynchronize());
    std::vector<float> hf((size_t)m * n), hu((size_t)m * n);
    CUDA_CHECK(cudaMemcpy(hf.data(), D, hf.size() * 4, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(hu.data(), D2, hu.size() * 4, cudaMemcpyDeviceToHost));
    double maxdiff = 0;
    for (size_t i = 0; i < hf.size(); ++i)
      maxdiff = std::max(maxdiff, (double)std::fabs(hf[i] - hu[i]));

    std::printf("\n== epilogue: D = GELU(W X + b), m=%d n=%d k=%d ==\n", m, n, k);
    std::printf("fused   (1 cublasLtMatmul, GELU_BIAS) : %.3f ms\n", tf);
    std::printf("unfused (matmul + bias_gelu kernel)   : %.3f ms"
                "  (bias_gelu alone %.3f ms)\n", tu, tk[tk.size() / 2]);
    std::printf("fused saves %.1f%%; max |fused - unfused| = %.2e\n",
                (1.f - tf / tu) * 100.f, maxdiff);
    std::printf("extra traffic of the unfused path: D read + written once more"
                " = %.1f MB\n", 2.0 * m * n * 4 / 1e6);
  }

  cudaFree(A); cudaFree(B); cudaFree(D); cudaFree(D2); cudaFree(bias);
  cudaFree(ws);
  cublasLtMatmulPreferenceDestroy(pref);
  cublasLtDestroy(lt);
  return 0;
}
