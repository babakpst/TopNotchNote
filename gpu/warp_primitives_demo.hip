// warp_primitives_demo.hip
// Companion example for: topnotchnote.com/gpu/warp_primitives.html
//
// Threads in one warp (Nvidia, 32 lanes) or wavefront (AMD, 32 or 64
// lanes) can trade register values directly, without shared memory:
//   1. shuffle    : a warp sum with shuffle-down, checked lane by lane
//   2. ballot     : one bit per lane for a predicate; popcount gives a
//                   warp-wide count in one instruction
//   3. warp-aggregated atomics: ballot picks a leader lane, which does one
//                   atomicAdd for the whole warp; every lane still gets its
//                   own unique slot (stream compaction)
//   4. cooperative groups: the same shuffle reduction through the
//                   thread_block_tile API, which names the group explicitly
//   5. a block-wide sum repeated many times on register values: shared-
//                   memory tree with a barrier per step, against shuffles.
// Part 3 is timed too, with a twist: for a plain atomicAdd on one address,
// current compilers (nvcc, and LLVM's AMDGPU atomic optimizer) already
// emit this exact ballot/popc/shuffle sequence for you, so the hand-written
// version is no faster. It is shown because it is what the compiler does,
// and the pattern you need when the compiler cannot see it.
//
// Portability: CUDA spells these __shfl_down_sync / __ballot_sync with a
// 32-bit lane mask. HIP on AMD spells them __shfl_down / __ballot, and the
// ballot result is 64 bits wide because a wavefront can have 64 lanes. The
// macros below hide that difference. Always read warpSize at run time.
//
// Compile:  hipcc -O2 -o main warp_primitives_demo.hip
// Nvidia:   hipcc also targets Nvidia (HIP_PLATFORM=nvidia), or swap the hip
//           prefix for cuda in the API calls and build with nvcc.

#include <hip/hip_runtime.h>
#include <hip/hip_cooperative_groups.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

namespace cg = cooperative_groups;

#define HIP_CHECK(call)                                                   \
  do {                                                                    \
    hipError_t err_ = (call);                                             \
    if (err_ != hipSuccess) {                                             \
      std::fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__,             \
                   hipGetErrorString(err_));                              \
      std::exit(1);                                                       \
    }                                                                     \
  } while (0)

#if defined(__HIP_PLATFORM_AMD__)
using LaneMask = unsigned long long;  // up to 64 lanes
#define SHFL_DOWN(v, o) __shfl_down(v, o)
#define SHFL(v, src) __shfl(v, src)
#define BALLOT(p) __ballot(p)
#define POPC(m) __popcll(m)
#define FFS(m) __ffsll((long long)(m))
#else
using LaneMask = unsigned;            // 32 lanes
#define SHFL_DOWN(v, o) __shfl_down_sync(0xffffffffu, v, o)
#define SHFL(v, src) __shfl_sync(0xffffffffu, v, src)
#define BALLOT(p) __ballot_sync(0xffffffffu, p)
#define POPC(m) __popc(m)
#define FFS(m) __ffs(m)
#endif

// 1. Shuffle: after log2(warpSize) steps, lane 0 holds the warp's sum.
__global__ void warpSum(const int *in, int *out) {
  int v = in[blockIdx.x * blockDim.x + threadIdx.x];
  for (int o = warpSize / 2; o > 0; o >>= 1) v += SHFL_DOWN(v, o);
  if (threadIdx.x % warpSize == 0)
    out[(blockIdx.x * blockDim.x + threadIdx.x) / warpSize] = v;
}

// 2. Ballot: each lane votes; the mask has one bit per lane that voted yes.
__global__ void warpCountEven(const int *in, int *out) {
  int v = in[blockIdx.x * blockDim.x + threadIdx.x];
  LaneMask m = BALLOT(v % 2 == 0);
  if (threadIdx.x % warpSize == 0)
    out[(blockIdx.x * blockDim.x + threadIdx.x) / warpSize] = POPC(m);
}

// 3a. Stream compaction, naive: one atomic per kept element.
__global__ void compactAtomic(const float *in, size_t n, float thresh,
                              float *out, unsigned *count) {
  size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
  if (i < n && in[i] > thresh) out[atomicAdd(count, 1u)] = in[i];
}

// 3b. Stream compaction, warp-aggregated: the lowest voting lane reserves
// space for the whole warp with one atomic and broadcasts the base; each
// lane's slot is base + the number of yes-votes in lanes below it.
// (Warps are full here: n is a multiple of the block size.)
__global__ void compactWarp(const float *in, size_t n, float thresh,
                            float *out, unsigned *count) {
  size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
  bool keep = i < n && in[i] > thresh;
  LaneMask m = BALLOT(keep);
  if (m == 0) return;  // whole warp has nothing to keep
  int lane = threadIdx.x % warpSize;
  int leader = FFS(m) - 1;
  unsigned base = 0;
  if (lane == leader) base = atomicAdd(count, (unsigned)POPC(m));
  base = SHFL(base, leader);
  LaneMask below = lane == 0 ? 0 : (m & ((~(LaneMask)0) >> (sizeof(LaneMask) * 8 - lane)));
  if (keep) out[base + POPC(below)] = in[i];
}

// 4. Cooperative groups: the group is a named object, and its size is a
// template parameter the compiler can see.
__global__ void tileSum(const int *in, int *out) {
  cg::thread_block_tile<32> tile =
      cg::tiled_partition<32>(cg::this_thread_block());
  int v = in[blockIdx.x * blockDim.x + threadIdx.x];
  for (int o = tile.size() / 2; o > 0; o >>= 1) v += tile.shfl_down(v, o);
  if (tile.thread_rank() == 0)
    out[(blockIdx.x * blockDim.x + threadIdx.x) / 32] = v;
}

// 5. Where shuffles pay: a block-wide sum done many times over values that
// live in registers, with no global memory traffic to hide behind.
// Shared-memory tree: log2(256) = 8 steps, a barrier after each.
__device__ float blockSumShared(float v, float *s) {
  s[threadIdx.x] = v;
  __syncthreads();
  for (unsigned st = blockDim.x / 2; st > 0; st >>= 1) {
    if (threadIdx.x < st) s[threadIdx.x] += s[threadIdx.x + st];
    __syncthreads();
  }
  float r = s[0];
  __syncthreads();  // s is reused next round
  return r;
}
// Shuffles inside each warp, one shared-memory step across warps.
__device__ float blockSumShuffle(float v, float *s) {
  int lane = threadIdx.x % warpSize, warp = threadIdx.x / warpSize;
  int nWarps = blockDim.x / warpSize;
  for (int o = warpSize / 2; o > 0; o >>= 1) v += SHFL_DOWN(v, o);
  if (lane == 0) s[warp] = v;
  __syncthreads();
  v = lane < nWarps ? s[lane] : 0.0f;
  for (int o = warpSize / 2; o > 0; o >>= 1) v += SHFL_DOWN(v, o);
  v = SHFL(v, 0);
  __syncthreads();  // s is reused next round
  return v;
}
template <bool UseShuffle>
__global__ void repeatedBlockSum(float *out, int rounds) {
  __shared__ float s[256];
  float v = threadIdx.x * 1e-3f, acc = 0.0f;
  for (int r = 0; r < rounds; ++r) {
    float t = UseShuffle ? blockSumShuffle(v, s) : blockSumShared(v, s);
    acc += t;
    v = v * 0.999f + 1e-4f;
  }
  if (threadIdx.x == 0) out[blockIdx.x] = acc;
}

int main() {
  hipDeviceProp_t prop;
  HIP_CHECK(hipGetDeviceProperties(&prop, 0));
  const int ws = prop.warpSize;
  std::printf("Device: %s, warpSize = %d\n\n", prop.name, ws);

  // ---- 1, 2, 4: correctness on 4096 ints ----
  const int n = 4096, block = 256;
  std::vector<int> h(n);
  for (int i = 0; i < n; ++i) h[i] = (i * 37) % 101;
  int *dIn, *dOut;
  HIP_CHECK(hipMalloc(&dIn, n * sizeof(int)));
  HIP_CHECK(hipMalloc(&dOut, n * sizeof(int)));
  HIP_CHECK(hipMemcpy(dIn, h.data(), n * sizeof(int), hipMemcpyHostToDevice));
  auto check = [&](const char *name, int groupSize, auto ref) {
    int groups = n / groupSize, bad = 0;
    std::vector<int> got(groups);
    HIP_CHECK(hipMemcpy(got.data(), dOut, groups * sizeof(int),
                        hipMemcpyDeviceToHost));
    for (int g = 0; g < groups; ++g) {
      int expect = 0;
      for (int j = 0; j < groupSize; ++j) expect += ref(h[g * groupSize + j]);
      bad += got[g] != expect;
    }
    std::printf("%-28s %d groups of %d, %d mismatches (first: %d)\n", name,
                groups, groupSize, bad, got[0]);
  };
  warpSum<<<n / block, block>>>(dIn, dOut);
  check("1. shuffle warp sum", ws, [](int v) { return v; });
  warpCountEven<<<n / block, block>>>(dIn, dOut);
  check("2. ballot + popc, even count", ws, [](int v) { return v % 2 == 0; });
  tileSum<<<n / block, block>>>(dIn, dOut);
  check("4. cooperative groups tile", 32, [](int v) { return v; });

  // ---- 3: stream compaction, timed ----
  const size_t m = 64u << 20;
  std::vector<float> hf(m);
  unsigned x = 1;
  for (size_t i = 0; i < m; ++i) {
    x = x * 1664525u + 1013904223u;
    hf[i] = (float)(x >> 8) / 16777216.0f;
  }
  float *dF, *dKept;
  unsigned *dCount;
  HIP_CHECK(hipMalloc(&dF, m * sizeof(float)));
  HIP_CHECK(hipMalloc(&dKept, m * sizeof(float)));
  HIP_CHECK(hipMalloc(&dCount, sizeof(unsigned)));
  HIP_CHECK(hipMemcpy(dF, hf.data(), m * sizeof(float),
                      hipMemcpyHostToDevice));
  hipEvent_t start, stop;
  HIP_CHECK(hipEventCreate(&start));
  HIP_CHECK(hipEventCreate(&stop));
  std::printf("\n3. Keep elements above a threshold, %zu floats\n", m);
  std::printf("   %-10s %-22s %10s %12s\n", "kept", "method", "time ms",
              "count");
  const float thresholds[] = {0.99f, 0.5f, 0.0f};
  for (float t : thresholds) {
    for (int warpAgg = 0; warpAgg < 2; ++warpAgg) {
      auto launch = [&] {
        HIP_CHECK(hipMemset(dCount, 0, sizeof(unsigned)));
        if (warpAgg)
          compactWarp<<<(unsigned)(m / block), block>>>(dF, m, t, dKept,
                                                        dCount);
        else
          compactAtomic<<<(unsigned)(m / block), block>>>(dF, m, t, dKept,
                                                          dCount);
      };
      launch();  // warm-up
      HIP_CHECK(hipEventRecord(start, 0));
      launch();
      HIP_CHECK(hipEventRecord(stop, 0));
      HIP_CHECK(hipEventSynchronize(stop));
      HIP_CHECK(hipGetLastError());
      float ms = 0.0f;
      unsigned c = 0;
      HIP_CHECK(hipEventElapsedTime(&ms, start, stop));
      HIP_CHECK(hipMemcpy(&c, dCount, sizeof(c), hipMemcpyDeviceToHost));
      char kept[16];
      std::snprintf(kept, sizeof(kept), "%.0f%%", 100.0f * (1.0f - t));
      std::printf("   %-10s %-22s %10.3f %12u\n", kept,
                  warpAgg ? "one atomic per warp" : "one atomic per element",
                  ms, c);
    }
  }

  // ---- 5: block reduction, shared memory vs shuffles, no memory traffic
  {
    const int rounds = 20000, blocks = prop.multiProcessorCount * 8;
    float *dRes;
    HIP_CHECK(hipMalloc(&dRes, blocks * sizeof(float)));
    std::printf("\n5. %d block-wide sums per block, %d blocks of %d, "
                "registers only\n", rounds, blocks, block);
    float res[2];
    for (int shuf = 0; shuf < 2; ++shuf) {
      auto launch = [&] {
        if (shuf) repeatedBlockSum<true><<<blocks, block>>>(dRes, rounds);
        else repeatedBlockSum<false><<<blocks, block>>>(dRes, rounds);
      };
      launch();
      HIP_CHECK(hipEventRecord(start, 0));
      launch();
      HIP_CHECK(hipEventRecord(stop, 0));
      HIP_CHECK(hipEventSynchronize(stop));
      HIP_CHECK(hipGetLastError());
      float ms = 0.0f;
      HIP_CHECK(hipEventElapsedTime(&ms, start, stop));
      HIP_CHECK(hipMemcpy(&res[shuf], dRes, sizeof(float),
                          hipMemcpyDeviceToHost));
      std::printf("   %-34s %9.3f ms  (%.0f ns per block sum)\n",
                  shuf ? "shuffles + 1 shared step" :
                         "shared-memory tree, 8 barriers",
                  ms, ms * 1e6 / rounds);
    }
    std::printf("   results agree: %s (%.4f vs %.4f)\n",
                std::fabs(res[0] - res[1]) <= 1e-3f * std::fabs(res[0])
                    ? "yes" : "NO", res[0], res[1]);
    HIP_CHECK(hipFree(dRes));
  }

  HIP_CHECK(hipFree(dIn));
  HIP_CHECK(hipFree(dOut));
  HIP_CHECK(hipFree(dF));
  HIP_CHECK(hipFree(dKept));
  HIP_CHECK(hipFree(dCount));
  HIP_CHECK(hipEventDestroy(start));
  HIP_CHECK(hipEventDestroy(stop));
  return 0;
}
