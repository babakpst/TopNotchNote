// graphs_demo.hip
// Companion example for: topnotchnote.com/gpu/graphs.html
//
// A graph records a sequence of GPU work once and replays it with one call,
// so the per-launch CPU cost is paid at setup instead of on every iteration.
//   1. A chain of K small kernels, run many times: launched one by one into
//      a stream, against captured once into a graph and replayed.
//   2. The same comparison with large kernels, where the GPU time hides the
//      launch cost and a graph buys almost nothing.
//   3. The one-time costs: capture, and instantiate.
//   4. Changing a kernel argument without rebuilding: re-capture and
//      hipGraphExecUpdate, against a fresh instantiate.
//
// Compile:  hipcc -O2 -o main graphs_demo.hip
// Nvidia:   hipcc also targets Nvidia (HIP_PLATFORM=nvidia), or swap the hip
//           prefix for cuda in the API calls and build with nvcc.

#include <hip/hip_runtime.h>

#include <algorithm>
#include <chrono>
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

using Clock = std::chrono::steady_clock;
static double us_since(Clock::time_point t0) {
  return std::chrono::duration<double, std::micro>(Clock::now() - t0).count();
}

__global__ void axpb(float* x, int n, float a, float b) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) x[i] = a * x[i] + b;
}

// K kernels in a row, each depending on the one before (same stream).
static void enqueue_chain(float* x, int n, int k, float b, hipStream_t s) {
  int blocks = (n + 255) / 256;
  for (int j = 0; j < k; ++j)
    hipLaunchKernelGGL(axpb, dim3(blocks), dim3(256), 0, s, x, n, 1.0f, b);
}

static hipGraphExec_t build(float* x, int n, int k, float b, hipStream_t s,
                            double* capture_us, double* inst_us,
                            hipGraph_t* out_graph) {
  hipGraph_t g;
  auto t0 = Clock::now();
  HIP_CHECK(hipStreamBeginCapture(s, hipStreamCaptureModeGlobal));
  enqueue_chain(x, n, k, b, s);
  HIP_CHECK(hipStreamEndCapture(s, &g));
  if (capture_us) *capture_us = us_since(t0);
  hipGraphExec_t e;
  t0 = Clock::now();
  HIP_CHECK(hipGraphInstantiate(&e, g, nullptr, nullptr, 0));
  if (inst_us) *inst_us = us_since(t0);
  if (out_graph) *out_graph = g;
  else HIP_CHECK(hipGraphDestroy(g));
  return e;
}

// Median over 5 trials of (time for `iters` iterations) / iters, wall clock,
// measured from the first launch to the end of a final synchronize.
template <typename F>
static double per_iter_us(F run_once, int iters, hipStream_t s) {
  std::vector<double> t;
  for (int trial = 0; trial < 6; ++trial) {
    auto t0 = Clock::now();
    for (int i = 0; i < iters; ++i) run_once();
    HIP_CHECK(hipStreamSynchronize(s));
    if (trial > 0) t.push_back(us_since(t0) / iters);  // trial 0 = warm-up
  }
  std::sort(t.begin(), t.end());
  return t[t.size() / 2];
}

int main() {
  hipDeviceProp_t prop;
  HIP_CHECK(hipGetDeviceProperties(&prop, 0));
  std::printf("device: %s\n", prop.name);

  hipStream_t s;
  HIP_CHECK(hipStreamCreateWithFlags(&s, hipStreamNonBlocking));
  const int nbig = 1 << 22;
  float* x;
  HIP_CHECK(hipMalloc(&x, nbig * sizeof(float)));
  HIP_CHECK(hipMemset(x, 0, nbig * sizeof(float)));

  // ---- 1 and 2 ----
  std::printf("\n%-10s %4s %14s %14s %8s\n", "elements", "K", "stream us/iter",
              "graph us/iter", "speedup");
  for (int n : {4096, nbig}) {
    for (int k : {1, 10, 100}) {
      int iters = (n == nbig) ? 20 : 2000 / k + 20;
      double t_stream = per_iter_us([&] { enqueue_chain(x, n, k, 0.f, s); },
                                    iters, s);
      hipGraphExec_t e = build(x, n, k, 0.f, s, nullptr, nullptr, nullptr);
      double t_graph = per_iter_us([&] { HIP_CHECK(hipGraphLaunch(e, s)); },
                                   iters, s);
      std::printf("%-10d %4d %14.1f %14.1f %7.2fx\n", n, k, t_stream, t_graph,
                  t_stream / t_graph);
      HIP_CHECK(hipGraphExecDestroy(e));
    }
  }

  // ---- 3 ----
  const int k = 100, n = 4096;
  double cap, inst;
  hipGraph_t g0;
  hipGraphExec_t e = build(x, n, k, 0.f, s, &cap, &inst, &g0);
  std::printf("\none-time cost for a %d-kernel graph: capture %.0f us, "
              "instantiate %.0f us\n", k, cap, inst);

  // ---- 4 ----
  // Same topology, new argument (b = 1): re-capture, then update in place.
  double cap2;
  auto t0 = Clock::now();
  hipGraph_t g1;
  HIP_CHECK(hipStreamBeginCapture(s, hipStreamCaptureModeGlobal));
  enqueue_chain(x, n, k, 1.f, s);
  HIP_CHECK(hipStreamEndCapture(s, &g1));
  cap2 = us_since(t0);
  t0 = Clock::now();
#if defined(__HIP_PLATFORM_AMD__)
  hipGraphNode_t err_node = nullptr;
  hipGraphExecUpdateResult res;
  HIP_CHECK(hipGraphExecUpdate(e, g1, &err_node, &res));
#elif CUDART_VERSION >= 12000
  cudaGraphExecUpdateResultInfo info;
  HIP_CHECK(cudaGraphExecUpdate(e, g1, &info));
#else
  cudaGraphNode_t err_node = nullptr;
  cudaGraphExecUpdateResult res;
  HIP_CHECK(cudaGraphExecUpdate(e, g1, &err_node, &res));
#endif
  double upd = us_since(t0);
  std::printf("change one argument: re-capture %.0f us + update %.0f us "
              "(a fresh instantiate took %.0f us)\n", cap2, upd, inst);

  // Check the updated graph really uses b = 1: x starts at 0, K adds of 1.
  HIP_CHECK(hipMemsetAsync(x, 0, n * sizeof(float), s));
  HIP_CHECK(hipGraphLaunch(e, s));
  float h = 0;
  HIP_CHECK(hipMemcpyAsync(&h, x, sizeof(float), hipMemcpyDeviceToHost, s));
  HIP_CHECK(hipStreamSynchronize(s));
  std::printf("after one replay of the updated graph x[0] = %.0f "
              "(expected %d)\n", h, k);

  HIP_CHECK(hipGraphExecDestroy(e));
  HIP_CHECK(hipGraphDestroy(g0));
  HIP_CHECK(hipGraphDestroy(g1));
  HIP_CHECK(hipFree(x));
  HIP_CHECK(hipStreamDestroy(s));
  return 0;
}
