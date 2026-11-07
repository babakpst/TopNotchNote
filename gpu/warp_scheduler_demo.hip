// warp_scheduler_demo.hip
// Companion example for: topnotchnote.com/gpu/warp_scheduler.html
//
// A warp scheduler issues one instruction per cycle from whichever of its
// warps is ready. A warp waiting on the result of its previous instruction
// is not ready, so the only way to keep the scheduler busy is to give it
// other warps to switch to (or independent instructions inside one warp).
// This program measures that directly. It launches exactly one block per
// SM/CU and grows the block one warp at a time, so the x axis is "resident
// warps per SM/CU" and nothing else changes:
//   1. FMA chain     : every instruction depends on the one before it, so a
//                      lone warp waits out the full arithmetic latency.
//   2. FMA chain, ILP: four independent chains per thread. The scheduler can
//                      issue the next one while the previous is in flight,
//                      so fewer warps reach the same throughput.
//   3. pointer chase : each warp follows its own random linked list in
//                      global memory (all lanes of a warp share one address,
//                      so each step is one memory transaction), and every
//                      load waits for the one before it. Memory latency is far
//                      longer than arithmetic latency, so the curve keeps
//                      climbing long after (1) has flattened.
// Read each table top to bottom: throughput climbs while extra warps are
// filling idle issue slots, then flattens once the scheduler is always busy
// (or, in part 3, once something else such as the memory system saturates).
//
// Compile:  hipcc -O2 -o main warp_scheduler_demo.hip
// Nvidia:   hipcc also targets Nvidia (HIP_PLATFORM=nvidia), or swap the hip
//           prefix for cuda in the API calls and build with nvcc.

#include <hip/hip_runtime.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <numeric>
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

constexpr int kFmaIters = 4096;   // FMAs per chain
constexpr int kChaseSteps = 1024; // dependent loads per lane

// One long chain: x = x * a + b, each FMA needs the previous result.
__global__ void fma_chain(float* out, float a, float b) {
  float x = threadIdx.x;
#pragma unroll 16
  for (int i = 0; i < kFmaIters; ++i) x = fmaf(x, a, b);
  if (x == -1.0f) out[0] = x;  // never true; keeps the chain alive
}

// Four independent chains per thread, same total FMA count per chain.
__global__ void fma_chain_ilp4(float* out, float a, float b) {
  float x0 = threadIdx.x, x1 = x0 + 1, x2 = x0 + 2, x3 = x0 + 3;
#pragma unroll 16
  for (int i = 0; i < kFmaIters; ++i) {
    x0 = fmaf(x0, a, b);
    x1 = fmaf(x1, a, b);
    x2 = fmaf(x2, a, b);
    x3 = fmaf(x3, a, b);
  }
  if (x0 + x1 + x2 + x3 == -1.0f) out[0] = x0;
}

// Each warp walks its own path through a random cycle in global memory.
// Every lane of a warp loads the same address, so one step is one memory
// transaction per warp and the test measures latency, not DRAM throughput.
__global__ void pointer_chase(const unsigned* next, unsigned* out,
                              unsigned stride, int warpSize_) {
  unsigned warp = (blockIdx.x * blockDim.x + threadIdx.x) / warpSize_;
  unsigned p = warp * stride;
  for (int i = 0; i < kChaseSteps; ++i) p = next[p];
  if (p == 0xffffffffu) out[0] = p;
}

template <typename Launch>
float time_ms(Launch launch) {
  hipEvent_t t0, t1;
  HIP_CHECK(hipEventCreate(&t0));
  HIP_CHECK(hipEventCreate(&t1));
  launch();  // warm-up
  HIP_CHECK(hipDeviceSynchronize());
  float best = 1e30f;
  for (int r = 0; r < 5; ++r) {
    HIP_CHECK(hipEventRecord(t0));
    launch();
    HIP_CHECK(hipEventRecord(t1));
    HIP_CHECK(hipEventSynchronize(t1));
    float ms;
    HIP_CHECK(hipEventElapsedTime(&ms, t0, t1));
    best = std::min(best, ms);
  }
  HIP_CHECK(hipGetLastError());
  HIP_CHECK(hipEventDestroy(t0));
  HIP_CHECK(hipEventDestroy(t1));
  return best;
}

int main() {
  hipDeviceProp_t prop;
  HIP_CHECK(hipGetDeviceProperties(&prop, 0));
  const int units = prop.multiProcessorCount;
  const int ws = prop.warpSize;
  const int maxWarps = std::min(prop.maxThreadsPerBlock / ws,
                                prop.maxThreadsPerMultiProcessor / ws);
  std::printf("%s: %d SMs/CUs, warp size %d, up to %d resident warps per "
              "SM/CU, one block per SM/CU below\n\n",
              prop.name, units, ws, prop.maxThreadsPerMultiProcessor / ws);

  float* dOut;
  HIP_CHECK(hipMalloc(&dOut, sizeof(float)));

  // Linked list: one random cycle over 64 MiB, far larger than any L2.
  const unsigned n = 16u << 20;
  std::vector<unsigned> perm(n);
  std::iota(perm.begin(), perm.end(), 0u);
  std::shuffle(perm.begin() + 1, perm.end(), std::mt19937(42));
  std::vector<unsigned> next(n);
  for (unsigned i = 0; i < n; ++i) next[perm[i]] = perm[(i + 1) % n];
  unsigned *dNext, *dSink;
  HIP_CHECK(hipMalloc(&dNext, n * sizeof(unsigned)));
  HIP_CHECK(hipMalloc(&dSink, sizeof(unsigned)));
  HIP_CHECK(hipMemcpy(dNext, next.data(), n * sizeof(unsigned),
                      hipMemcpyHostToDevice));

  std::printf("warps/unit | FMA chain   | FMA chain, ILP 4 | pointer chase\n");
  std::printf("           | GFMA/s      | GFMA/s           | M warp-loads/s  ns/load\n");
  std::printf("-----------+-------------+------------------+------------------\n");

  std::vector<int> counts;
  for (int w = 1; w <= maxWarps; w = (w < 4 ? w + 1 : w * 2)) counts.push_back(w);
  if (counts.back() != maxWarps) counts.push_back(maxWarps);

  for (int w : counts) {
    const int threads = w * ws;
    const double lanes = double(units) * threads;

    float ms1 = time_ms([&] {
      hipLaunchKernelGGL(fma_chain, dim3(units), dim3(threads), 0, 0, dOut,
                         0.999f, 0.001f);
    });
    float ms2 = time_ms([&] {
      hipLaunchKernelGGL(fma_chain_ilp4, dim3(units), dim3(threads), 0, 0,
                         dOut, 0.999f, 0.001f);
    });
    const unsigned stride = n / unsigned(units * w);
    float ms3 = time_ms([&] {
      hipLaunchKernelGGL(pointer_chase, dim3(units), dim3(threads), 0, 0,
                         dNext, dSink, stride, ws);
    });

    double g1 = lanes * kFmaIters / (ms1 * 1e6);
    double g2 = lanes * kFmaIters * 4 / (ms2 * 1e6);
    double m3 = double(units) * w * kChaseSteps / (ms3 * 1e3);
    double nsPerLoad = ms3 * 1e6 / kChaseSteps;  // per step of one warp
    std::printf("%10d | %11.1f | %16.1f | %14.1f  %7.0f\n", w, g1, g2, m3,
                nsPerLoad);
  }

  std::printf("\nns/load is the time one warp spends per step of its chain, "
              "roughly the memory latency. While it stays flat as warps are\n"
              "added, the extra throughput is pure overlap: more warps waiting "
              "at once, not faster loads.\n");

  HIP_CHECK(hipFree(dOut));
  HIP_CHECK(hipFree(dNext));
  HIP_CHECK(hipFree(dSink));
  return 0;
}
