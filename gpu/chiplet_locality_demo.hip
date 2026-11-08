// chiplet_locality_demo.hip
// Companion example for: topnotchnote.com/gpu/chiplet_gpus.html
//
// On a chiplet GPU such as the AMD MI300X or MI355X, each compute die (XCD)
// has its own private L2 cache, and in the default single-partition mode the
// hardware hands out workgroups to the XCDs round-robin: block 0 to XCD 0,
// block 1 to XCD 1, and so on. Consecutive blocks therefore land on different
// dies. If consecutive blocks share input data, every die fetches its own copy
// from HBM into its own L2.
//
// This program builds that situation and the usual fix.
//   * Groups of G consecutive logical blocks read the same tile (G-fold reuse).
//   * "natural" mapping : logical block = blockIdx.x.
//   * "die-aware" mapping: logical block is remapped so that the blocks the
//     round-robin dispatcher sends to one die get consecutive logical ids, so
//     each group of G blocks shares one die and one L2.
// It reports effective bandwidth (bytes requested / time) for both.
//
// On a monolithic GPU (one L2 for the whole chip) the natural order is already
// the good one: blocks that run at the same time share a tile and hit in the
// one L2. The die-aware remap spreads each group out in time instead, so
// expect it to be slower there. That is the point: the remap only pays off on
// hardware that really dispatches round-robin across dies with private L2s,
// such as an MI300X or MI355X in SPX mode, and how much it pays depends on
// the tile size and the chip. Run with dies=1 and the two mappings are the
// same code path.
//
// Compile:  hipcc -O2 -o main chiplet_locality_demo.hip
// Nvidia:   hipcc also targets Nvidia (HIP_PLATFORM=nvidia), or swap the hip
//           prefix for cuda in the API calls and build with nvcc.
// Run:      ./main [dies]     (dies = number of XCDs to model, default 8)

#include <hip/hip_runtime.h>

#include <algorithm>
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

constexpr int kThreads = 256;
constexpr int kGroup = 8;                       // blocks that share one tile
constexpr size_t kTileBytes = size_t(512) << 10;  // 512 KiB per tile
constexpr int kBlocks = 4096;                   // 512 tiles, 256 MiB in total

// Every block reads its group's whole tile and writes one partial sum.
__global__ void read_tiles(const float4* __restrict__ data, float* out,
                           int dies, bool die_aware) {
  int b = blockIdx.x;
  int logical = b;
  if (die_aware) {
    // Round-robin sends physical block b to die (b % dies). Give the blocks
    // of one die a contiguous run of logical ids.
    int per_die = gridDim.x / dies;
    logical = (b % dies) * per_die + (b / dies);
  }
  int group = logical / kGroup;
  const float4* tile = data + size_t(group) * (kTileBytes / sizeof(float4));
  float acc = 0.f;
  for (size_t i = threadIdx.x; i < kTileBytes / sizeof(float4); i += kThreads) {
    float4 v = tile[i];
    acc += v.x + v.y + v.z + v.w;
  }
  if (threadIdx.x == 0) out[logical] = acc;  // keeps the loads alive
}

int main(int argc, char** argv) {
  int dies = argc > 1 ? std::atoi(argv[1]) : 8;
  if (dies < 1 || kBlocks % (dies * kGroup) != 0) {
    std::fprintf(stderr, "dies must divide %d evenly\n", kBlocks / kGroup);
    return 1;
  }
  hipDeviceProp_t p;
  HIP_CHECK(hipGetDeviceProperties(&p, 0));
  std::printf("%s: %d SMs/CUs, L2 reported as %.1f MiB\n", p.name,
              p.multiProcessorCount, p.l2CacheSize / 1048576.0);
  std::printf("modelling %d dies, groups of %d blocks share a %zu KiB tile\n\n",
              dies, kGroup, kTileBytes >> 10);

  size_t tiles = kBlocks / kGroup;
  size_t bytes = tiles * kTileBytes;
  float4* data;
  float* out;
  HIP_CHECK(hipMalloc(&data, bytes));
  HIP_CHECK(hipMalloc(&out, kBlocks * sizeof(float)));
  HIP_CHECK(hipMemset(data, 0, bytes));

  hipEvent_t start, stop;
  HIP_CHECK(hipEventCreate(&start));
  HIP_CHECK(hipEventCreate(&stop));
  double requested = double(kBlocks) * kTileBytes;

  for (int pass = 0; pass < 2; ++pass) {
    bool die_aware = pass == 1;
    std::vector<float> ms;
    for (int r = 0; r < 11; ++r) {
      HIP_CHECK(hipEventRecord(start));
      hipLaunchKernelGGL(read_tiles, dim3(kBlocks), dim3(kThreads), 0, 0,
                         data, out, dies, die_aware);
      HIP_CHECK(hipEventRecord(stop));
      HIP_CHECK(hipEventSynchronize(stop));
      float t;
      HIP_CHECK(hipEventElapsedTime(&t, start, stop));
      if (r > 0) ms.push_back(t);  // first launch is warm-up
    }
    std::sort(ms.begin(), ms.end());
    float med = ms[ms.size() / 2];
    std::printf("%-10s mapping: %7.3f ms  %7.1f GB/s requested\n",
                die_aware ? "die-aware" : "natural", med,
                requested / (med * 1e-3) / 1e9);
  }
  std::printf("\nunique data: %.0f MiB; each byte is requested %d times\n",
              bytes / 1048576.0, kGroup);

  HIP_CHECK(hipFree(data));
  HIP_CHECK(hipFree(out));
  return 0;
}
