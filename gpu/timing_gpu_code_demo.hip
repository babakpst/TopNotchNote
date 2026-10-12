// timing_gpu_code_demo.hip
// Companion example for: topnotchnote.com/gpu/timing_gpu_code.html
//
// Four experiments, each one a way GPU timing goes wrong:
//   1. The missing synchronize: a host clock around a launch measures the
//      enqueue. Adding a synchronize, or using events, measures the kernel.
//   2. The cold first run: the first launch of a kernel pays one-time costs
//      that later launches do not.
//   3. Run-to-run noise: the same kernel, timed 100 times. Report the median
//      and the spread, not one number.
//   4. The warm cache: a buffer that fits in L2, timed back to back, reads
//      from cache. Flushing L2 between runs gives the DRAM number.
// Compile:  hipcc -O2 -o main timing_gpu_code_demo.hip
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

static double microsSince(Clock::time_point start) {
  return std::chrono::duration<double, std::micro>(Clock::now() - start).count();
}

// Enough arithmetic per element that the kernel runs for milliseconds.
__global__ void busyKernel(float *data, int n, int iters) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    float x = data[i];
    for (int k = 0; k < iters; ++k) {
      x = x * 0.999f + 0.001f;
    }
    data[i] = x;
  }
}

// Never launched before experiment 2, so its first launch is a true cold start.
__global__ void coldKernel(float *data, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) data[i] = data[i] * 2.0f + 1.0f;
}

// Memory bound: one read and one write per element.
__global__ void copyKernel(const float4 *in, float4 *out, int n4) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n4) out[i] = in[i];
}

// Writing a buffer larger than L2 evicts whatever was there.
__global__ void flushKernel(int *scratch, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) scratch[i] = i;
}

static float eventMs(hipEvent_t start, hipEvent_t stop) {
  float ms = 0.0f;
  HIP_CHECK(hipEventSynchronize(stop));
  HIP_CHECK(hipEventElapsedTime(&ms, start, stop));
  return ms;
}

int main() {
  hipDeviceProp_t prop;
  HIP_CHECK(hipGetDeviceProperties(&prop, 0));
  std::printf("Device: %s, L2 %d KB\n\n", prop.name, prop.l2CacheSize / 1024);

  const int block = 256;
  hipEvent_t start, stop;
  HIP_CHECK(hipEventCreate(&start));
  HIP_CHECK(hipEventCreate(&stop));

  const int n = 1 << 22;
  const int grid = (n + block - 1) / block;
  float *d_data = nullptr;
  HIP_CHECK(hipMalloc(&d_data, n * sizeof(float)));
  HIP_CHECK(hipMemset(d_data, 0, n * sizeof(float)));

  // 2 runs first, while coldKernel has never been launched.
  // Host clock plus synchronize, so the one-time costs are included.
  std::printf("2. first launch vs later launches of the same kernel\n");
  for (int run = 0; run < 3; ++run) {
    auto t = Clock::now();
    coldKernel<<<grid, block>>>(d_data, n);
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipDeviceSynchronize());
    std::printf("   launch %d: %8.1f us\n", run + 1, microsSince(t));
  }
  std::printf("\n");

  // 1. The missing synchronize.
  const int iters = 2000;
  busyKernel<<<grid, block>>>(d_data, n, iters);  // warmup
  HIP_CHECK(hipDeviceSynchronize());

  auto t0 = Clock::now();
  busyKernel<<<grid, block>>>(d_data, n, iters);
  double noSyncUs = microsSince(t0);
  HIP_CHECK(hipDeviceSynchronize());

  auto t1 = Clock::now();
  busyKernel<<<grid, block>>>(d_data, n, iters);
  HIP_CHECK(hipDeviceSynchronize());
  double syncUs = microsSince(t1);

  HIP_CHECK(hipEventRecord(start));
  busyKernel<<<grid, block>>>(d_data, n, iters);
  HIP_CHECK(hipEventRecord(stop));
  double eventUs = eventMs(start, stop) * 1000.0;

  std::printf("1. one kernel, timed three ways\n");
  std::printf("   host clock, no synchronize:   %10.1f us  (the enqueue)\n", noSyncUs);
  std::printf("   host clock, synchronize:      %10.1f us\n", syncUs);
  std::printf("   events in the stream:         %10.1f us\n\n", eventUs);

  // 3. Run-to-run noise.
  const int runs = 100;
  std::vector<float> times(runs);
  for (int r = 0; r < runs; ++r) {
    HIP_CHECK(hipEventRecord(start));
    busyKernel<<<grid, block>>>(d_data, n, iters);
    HIP_CHECK(hipEventRecord(stop));
    times[r] = eventMs(start, stop) * 1000.0f;
  }
  std::vector<float> sorted = times;
  std::sort(sorted.begin(), sorted.end());
  std::printf("3. the same kernel, %d timed runs\n", runs);
  std::printf("   min %.1f us, median %.1f us, max %.1f us\n\n",
              sorted.front(), sorted[runs / 2], sorted.back());

  // 4. The warm cache. A copy whose input plus output fit in half of L2.
  const size_t smallBytes = static_cast<size_t>(prop.l2CacheSize) / 4;  // per buffer
  const int n4 = static_cast<int>(smallBytes / sizeof(float4));
  const int copyGrid = (n4 + block - 1) / block;
  float4 *d_in = nullptr, *d_out = nullptr;
  HIP_CHECK(hipMalloc(&d_in, smallBytes));
  HIP_CHECK(hipMalloc(&d_out, smallBytes));
  HIP_CHECK(hipMemset(d_in, 0, smallBytes));

  const int flushN = std::max(prop.l2CacheSize, 1 << 20) * 4 / static_cast<int>(sizeof(int));
  int *d_flush = nullptr;
  HIP_CHECK(hipMalloc(&d_flush, flushN * sizeof(int)));

  copyKernel<<<copyGrid, block>>>(d_in, d_out, n4);  // warmup
  HIP_CHECK(hipDeviceSynchronize());

  float warmBest = 1e30f, coldBest = 1e30f;
  for (int r = 0; r < runs; ++r) {
    HIP_CHECK(hipEventRecord(start));
    copyKernel<<<copyGrid, block>>>(d_in, d_out, n4);
    HIP_CHECK(hipEventRecord(stop));
    warmBest = std::min(warmBest, eventMs(start, stop));
  }
  for (int r = 0; r < runs; ++r) {
    flushKernel<<<(flushN + block - 1) / block, block>>>(d_flush, flushN);  // outside the events
    HIP_CHECK(hipEventRecord(start));
    copyKernel<<<copyGrid, block>>>(d_in, d_out, n4);
    HIP_CHECK(hipEventRecord(stop));
    coldBest = std::min(coldBest, eventMs(start, stop));
  }
  const double moved = 2.0 * smallBytes;  // read in, write out
  std::printf("4. copy %zu KB in, %zu KB out (fits in L2), best of %d\n",
              smallBytes / 1024, smallBytes / 1024, runs);
  std::printf("   back to back:        %6.1f us  -> %6.1f GB/s\n",
              warmBest * 1000.0f, moved / (warmBest * 1e-3) / 1e9);
  std::printf("   L2 flushed each run: %6.1f us  -> %6.1f GB/s\n",
              coldBest * 1000.0f, moved / (coldBest * 1e-3) / 1e9);

  HIP_CHECK(hipFree(d_flush));
  HIP_CHECK(hipFree(d_out));
  HIP_CHECK(hipFree(d_in));
  HIP_CHECK(hipFree(d_data));
  HIP_CHECK(hipEventDestroy(start));
  HIP_CHECK(hipEventDestroy(stop));
  return 0;
}
