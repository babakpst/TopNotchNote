// reduction_demo.hip
// Companion example for: topnotchnote.com/gpu/reduction.html
//
// Summing an array, the hard way. Five versions of the same sum over 64M
// floats (256 MB), each fixing the previous one's main problem:
//   1. interleaved   : tree in shared memory, stride doubling, tid % (2*s)
//                      leaves most threads in each warp idle (divergence)
//   2. sequential    : tree in shared memory, stride halving, active
//                      threads stay contiguous
//   3. first add     : each thread adds two elements while loading, so half
//                      the threads are not idle from the very first step
//   4. grid-stride   : a fixed, small grid; each thread sums many elements
//                      in a register before the tree starts
//   5. warp shuffle  : version 4, with the last 32 steps done in registers
//                      by warp shuffles instead of shared memory
// Every version reduces to one partial per block, then a second, tiny
// launch adds up the partials. Prints time, effective bandwidth (the
// ceiling for a sum is reading the input once) and the result.
//
// Compile:  hipcc -O2 -o main reduction_demo.hip
// Nvidia:   hipcc also targets Nvidia (HIP_PLATFORM=nvidia), or swap the hip
//           prefix for cuda in the API calls and build with nvcc.

#include <hip/hip_runtime.h>

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

// Warp-level sum, used by version 5. On AMD a wavefront is 64 wide and the
// non-_sync shuffle is the portable spelling; see the warp primitives page.
#if defined(__HIP_PLATFORM_AMD__)
#define SHFL_DOWN(v, o) __shfl_down(v, o)
#else
#define SHFL_DOWN(v, o) __shfl_down_sync(0xffffffffu, v, o)
#endif

constexpr int BLOCK = 256;

// 1. Interleaved addressing. Correct, and slow: at each step only threads
// whose index is a multiple of 2*s work, scattered across every warp.
__global__ void reduceInterleaved(const float *in, float *out, size_t n) {
  __shared__ float s[BLOCK];
  unsigned tid = threadIdx.x;
  size_t i = blockIdx.x * (size_t)blockDim.x + tid;
  s[tid] = i < n ? in[i] : 0.0f;
  __syncthreads();
  for (unsigned stride = 1; stride < blockDim.x; stride *= 2) {
    if (tid % (2 * stride) == 0) s[tid] += s[tid + stride];
    __syncthreads();
  }
  if (tid == 0) out[blockIdx.x] = s[0];
}

// 2. Sequential addressing. The first half adds the second half, then the
// first quarter adds the second quarter... Active threads are contiguous,
// so whole warps retire instead of every warp running at partial width.
__global__ void reduceSequential(const float *in, float *out, size_t n) {
  __shared__ float s[BLOCK];
  unsigned tid = threadIdx.x;
  size_t i = blockIdx.x * (size_t)blockDim.x + tid;
  s[tid] = i < n ? in[i] : 0.0f;
  __syncthreads();
  for (unsigned stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride) s[tid] += s[tid + stride];
    __syncthreads();
  }
  if (tid == 0) out[blockIdx.x] = s[0];
}

// 3. First add during load: each block covers 2 * blockDim elements.
__global__ void reduceFirstAdd(const float *in, float *out, size_t n) {
  __shared__ float s[BLOCK];
  unsigned tid = threadIdx.x;
  size_t i = blockIdx.x * (size_t)blockDim.x * 2 + tid;
  float v = i < n ? in[i] : 0.0f;
  if (i + blockDim.x < n) v += in[i + blockDim.x];
  s[tid] = v;
  __syncthreads();
  for (unsigned stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride) s[tid] += s[tid + stride];
    __syncthreads();
  }
  if (tid == 0) out[blockIdx.x] = s[0];
}

// 4. Grid-stride: a few blocks per SM/CU, each thread accumulating many
// elements in a register first. The tree now runs once per block, not once
// per 256 elements.
__global__ void reduceGridStride(const float *in, float *out, size_t n) {
  __shared__ float s[BLOCK];
  unsigned tid = threadIdx.x;
  float v = 0.0f;
  for (size_t i = blockIdx.x * (size_t)blockDim.x + tid; i < n;
       i += (size_t)gridDim.x * blockDim.x)
    v += in[i];
  s[tid] = v;
  __syncthreads();
  for (unsigned stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride) s[tid] += s[tid + stride];
    __syncthreads();
  }
  if (tid == 0) out[blockIdx.x] = s[0];
}

// 5. Grid-stride plus warp shuffles: each warp reduces its own values in
// registers, one value per warp goes to shared memory, and the first warp
// reduces those. Two __syncthreads instead of eight.
__global__ void reduceShuffle(const float *in, float *out, size_t n) {
  __shared__ float warpSums[BLOCK / 32];
  unsigned tid = threadIdx.x;
  float v = 0.0f;
  for (size_t i = blockIdx.x * (size_t)blockDim.x + tid; i < n;
       i += (size_t)gridDim.x * blockDim.x)
    v += in[i];
  const int ws = warpSize;
  for (int o = ws / 2; o > 0; o >>= 1) v += SHFL_DOWN(v, o);
  unsigned lane = tid % ws, warp = tid / ws, nWarps = blockDim.x / ws;
  if (lane == 0) warpSums[warp] = v;
  __syncthreads();
  if (warp == 0) {
    v = lane < nWarps ? warpSums[lane] : 0.0f;
    for (int o = ws / 2; o > 0; o >>= 1) v += SHFL_DOWN(v, o);
    if (lane == 0) out[blockIdx.x] = v;
  }
}

using Kernel = void (*)(const float *, float *, size_t);

int main() {
  hipDeviceProp_t prop;
  HIP_CHECK(hipGetDeviceProperties(&prop, 0));
  const size_t n = 64u << 20;  // 64M floats, 256 MB
  const size_t bytes = n * sizeof(float);
  std::printf("Device: %s, summing %zu floats (%zu MB)\n\n", prop.name, n,
              bytes >> 20);

  // Values in [0, 1) on a 1/1024 grid, so the exact sum is easy to check
  // against. With arbitrary floats the versions would disagree in the last
  // digits, because they add in different orders and float addition is not
  // associative; this data is gentle enough that they usually agree.
  std::vector<float> h(n);
  double exact = 0.0;
  for (size_t i = 0; i < n; ++i) {
    h[i] = (float)((i * 2654435761u) % 1024) / 1024.0f;
    exact += h[i];
  }
  float *dIn, *dPart, *dOut;
  HIP_CHECK(hipMalloc(&dIn, bytes));
  HIP_CHECK(hipMalloc(&dPart, (n / BLOCK + 1) * sizeof(float)));
  HIP_CHECK(hipMalloc(&dOut, sizeof(float)));
  HIP_CHECK(hipMemcpy(dIn, h.data(), bytes, hipMemcpyHostToDevice));

  const unsigned blocksOne = (unsigned)((n + BLOCK - 1) / BLOCK);
  const unsigned blocksTwo = (unsigned)((n + 2 * BLOCK - 1) / (2 * BLOCK));
  const unsigned blocksGrid = prop.multiProcessorCount * 8;
  struct Version { const char *name; Kernel k; unsigned blocks; };
  Version versions[] = {
      {"1 interleaved", reduceInterleaved, blocksOne},
      {"2 sequential", reduceSequential, blocksOne},
      {"3 first add", reduceFirstAdd, blocksTwo},
      {"4 grid-stride", reduceGridStride, blocksGrid},
      {"5 warp shuffle", reduceShuffle, blocksGrid},
  };

  hipEvent_t start, stop;
  HIP_CHECK(hipEventCreate(&start));
  HIP_CHECK(hipEventCreate(&stop));
  std::printf("%-15s %8s %10s %10s %16s\n", "version", "blocks", "time ms",
              "GB/s", "sum");
  for (const Version &v : versions) {
    auto run = [&] {
      v.k<<<v.blocks, BLOCK>>>(dIn, dPart, n);
      // Second pass: one block of grid-stride adds over the partials.
      reduceGridStride<<<1, BLOCK>>>(dPart, dOut, v.blocks);
    };
    run();  // warm-up
    const int reps = 10;
    HIP_CHECK(hipEventRecord(start, 0));
    for (int r = 0; r < reps; ++r) run();
    HIP_CHECK(hipEventRecord(stop, 0));
    HIP_CHECK(hipEventSynchronize(stop));
    HIP_CHECK(hipGetLastError());
    float ms = 0.0f, sum = 0.0f;
    HIP_CHECK(hipEventElapsedTime(&ms, start, stop));
    ms /= reps;
    HIP_CHECK(hipMemcpy(&sum, dOut, sizeof(float), hipMemcpyDeviceToHost));
    std::printf("%-15s %8u %10.3f %10.1f %16.1f\n", v.name, v.blocks, ms,
                bytes / (ms * 1e6), sum);
  }
  std::printf("%-15s %8s %10s %10s %16.1f\n", "exact (double)", "", "", "",
              exact);

  HIP_CHECK(hipEventDestroy(start));
  HIP_CHECK(hipEventDestroy(stop));
  HIP_CHECK(hipFree(dIn));
  HIP_CHECK(hipFree(dPart));
  HIP_CHECK(hipFree(dOut));
  return 0;
}
