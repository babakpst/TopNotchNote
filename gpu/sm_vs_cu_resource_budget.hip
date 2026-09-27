// sm_vs_cu_resource_budget.hip
// Companion example for: topnotchnote.com/gpu/sm_vs_cu.html
//
// Prints what one SM/CU on your GPU holds, then asks the runtime how many
// blocks of one kernel fit on a single SM/CU as the block's scratchpad
// (shared memory / LDS) request grows. Every block has to be carved out of
// the same fixed per-unit budget, so the answer drops in steps, and the
// number of blocks the whole GPU can run at once drops with it.
// Compile:  hipcc -O2 -o main sm_vs_cu_resource_budget.hip
// Nvidia:   hipcc also targets Nvidia (HIP_PLATFORM=nvidia), or swap the hip
//           prefix for cuda in the API calls and build with nvcc.

#include <hip/hip_runtime.h>

#include <cstdio>
#include <cstdlib>

#define HIP_CHECK(call)                                                   \
  do {                                                                    \
    hipError_t err_ = (call);                                             \
    if (err_ != hipSuccess) {                                             \
      std::fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__,             \
                   hipGetErrorString(err_));                              \
      std::exit(1);                                                       \
    }                                                                     \
  } while (0)

// A kernel that uses dynamic shared memory, so the request can be varied at launch.
__global__ void scratchpadKernel(float *out) {
  extern __shared__ float tile[];
  tile[threadIdx.x] = static_cast<float>(threadIdx.x);
  __syncthreads();
  out[blockIdx.x * blockDim.x + threadIdx.x] = tile[blockDim.x - 1 - threadIdx.x];
}

int main() {
  hipDeviceProp_t p;
  HIP_CHECK(hipGetDeviceProperties(&p, 0));

  std::printf("Device: %s\n\n", p.name);
  std::printf("SMs / CUs on the die:          %d\n", p.multiProcessorCount);
  std::printf("Warp / wavefront size:         %d threads\n", p.warpSize);
  std::printf("Max resident threads per unit: %d  (= %d warps/wavefronts)\n",
              p.maxThreadsPerMultiProcessor, p.maxThreadsPerMultiProcessor / p.warpSize);
  std::printf("Registers per unit:            %d  (32-bit)\n", p.regsPerMultiprocessor);
  std::printf("Scratchpad per unit:           %zu KB\n", p.sharedMemPerMultiprocessor / 1024);
  std::printf("Scratchpad per block, max:     %zu KB\n", p.sharedMemPerBlock / 1024);
  std::printf("L2 cache, whole GPU:           %d KB\n\n", p.l2CacheSize / 1024);

  const int blockSize = 256;
  std::printf("Blocks of %d threads that fit on one unit, by scratchpad request:\n", blockSize);
  std::printf("  %-12s %-16s %s\n", "scratchpad", "blocks per unit", "blocks on the whole GPU at once");
  const size_t requestsKB[] = {0, 4, 8, 16, 24, 32, 48};
  for (size_t kb : requestsKB) {
    size_t bytes = kb * 1024;
    if (bytes > p.sharedMemPerBlock) break;
    int blocksPerUnit = 0;
    HIP_CHECK(hipOccupancyMaxActiveBlocksPerMultiprocessor(&blocksPerUnit, scratchpadKernel,
                                                           blockSize, bytes));
    std::printf("  %3zu KB       %-16d %d\n", kb, blocksPerUnit,
                blocksPerUnit * p.multiProcessorCount);
  }
  return 0;
}
