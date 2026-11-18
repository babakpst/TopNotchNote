// smi_utilization_demo.hip
// Companion example for: topnotchnote.com/gpu/smi_tools.html
//
// Runs four phases of 6 seconds each, with a 2 s idle gap between them, so
// you can watch nvidia-smi or amd-smi / rocm-smi in another terminal:
//   1. idle
//   2. one thread: a kernel with 1 block of 1 thread that spins in a loop.
//      Almost the whole GPU is idle, but a kernel is always running.
//   3. full: a kernel with enough blocks to fill every compute unit, doing
//      arithmetic in a loop.
//   4. bursty: the full kernel for about 10% of the time, then the GPU idles.
// Each phase prints its start time so the samples can be lined up.
//
// Watch it with, for example:
//   nvidia-smi --query-gpu=timestamp,utilization.gpu,power.draw,clocks.sm \
//              --format=csv --loop-ms=500
//   amd-smi metric --usage --power --clock --watch 1      (AMD)
//
// Compile:  hipcc -O2 -o main smi_utilization_demo.hip
// Nvidia:   hipcc also targets Nvidia (HIP_PLATFORM=nvidia), or swap the hip
//           prefix for cuda in the API calls and build with nvcc.

#include <hip/hip_runtime.h>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <thread>

#define HIP_CHECK(call)                                                   \
  do {                                                                    \
    hipError_t err_ = (call);                                             \
    if (err_ != hipSuccess) {                                             \
      std::fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__,             \
                   hipGetErrorString(err_));                              \
      std::exit(1);                                                       \
    }                                                                     \
  } while (0)

__global__ void spin(float* out, int iters) {
  float x = threadIdx.x * 0.001f, y = 1.0f;
  for (int i = 0; i < iters; ++i) {
    x = x * 0.9999f + 0.0001f;
    y = y * 1.0001f - 0.0001f;
  }
  if (x + y == 12345.f) out[threadIdx.x] = x;   // keep the loop alive
}

using Clock = std::chrono::steady_clock;
static double secs(Clock::time_point t0) {
  return std::chrono::duration<double>(Clock::now() - t0).count();
}

// Calibrate: how many loop iterations take about `ms` in a given launch shape.
static int calibrate(float* out, unsigned blocks, unsigned threads, double ms) {
  int iters = 1 << 16;
  while (true) {
    auto t = Clock::now();
    spin<<<blocks, threads>>>(out, iters);
    HIP_CHECK(hipDeviceSynchronize());
    double took = secs(t) * 1e3;
    if (took > ms / 4) return (int)(iters * (ms / took));
    iters *= 2;
  }
}

int main() {
  float* out;
  HIP_CHECK(hipMalloc(&out, 1024 * sizeof(float)));
  int cus = 0;
  HIP_CHECK(hipDeviceGetAttribute(&cus, hipDeviceAttributeMultiprocessorCount, 0));
  const unsigned full_blocks = cus * 8;
  const int it_one = calibrate(out, 1, 1, 50);
  const int it_full = calibrate(out, full_blocks, 256, 10);

  auto t0 = Clock::now();
  auto phase = [&](const char* name) { std::printf("%6.1f s  %s\n", secs(t0), name); std::fflush(stdout); };
  auto idle = [&](double s) { std::this_thread::sleep_for(std::chrono::duration<double>(s)); };

  phase("1. idle");
  idle(6);
  idle(2);
  phase("2. one thread, always running");
  while (secs(t0) < 16) { spin<<<1, 1>>>(out, it_one); HIP_CHECK(hipDeviceSynchronize()); }
  idle(2);
  phase("3. full, always running");
  while (secs(t0) < 24) { spin<<<full_blocks, 256>>>(out, it_full); HIP_CHECK(hipDeviceSynchronize()); }
  idle(2);
  phase("4. bursty: ~10 ms of work, ~90 ms idle");
  while (secs(t0) < 32) {
    spin<<<full_blocks, 256>>>(out, it_full);
    HIP_CHECK(hipDeviceSynchronize());
    idle(0.09);
  }
  phase("done");
  HIP_CHECK(hipFree(out));
  return 0;
}
