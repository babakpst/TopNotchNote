// clock_variance_demo.hip
// Companion example for: topnotchnote.com/gpu/clocks_and_power.html
//
// Runs the same compute-bound kernel back to back for about 30 seconds,
// starting from an idle GPU, and for every launch prints:
//   - how long it took (GPU event timing), and
//   - the clock the SM actually ran at, measured from inside the kernel:
//     cycles counted by clock64() divided by the elapsed time.
// The kernel never changes, so every change in the numbers is the GPU
// changing its clock: ramping up from idle, boosting while cool, and
// backing off once it hits its power or temperature limit.
// Run nvidia-smi or amd-smi in another terminal to watch power and
// temperature move with it (commands on the page).
//
// Compile:  hipcc -O2 -o main clock_variance_demo.hip
// Nvidia:   hipcc also targets Nvidia (HIP_PLATFORM=nvidia), or swap the hip
//           prefix for cuda in the API calls and build with nvcc.

#include <hip/hip_runtime.h>

#include <chrono>
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

// Pure arithmetic, no memory traffic in the loop, so the run time depends
// only on the SM clock. Block 0's first thread records how many cycles of
// its own SM clock the work took.
__global__ void fmaKernel(float *out, int iters, long long *cycles) {
  long long t0 = clock64();
  float a = threadIdx.x * 1e-3f, b = 1.0001f, c = 0.9999f;
  for (int i = 0; i < iters; ++i) {
    a = a * b + c;
    b = b * c + a * 1e-7f;
  }
  long long t1 = clock64();
  if (blockIdx.x == 0 && threadIdx.x == 0) *cycles = t1 - t0;
  if (a == -1.0f) out[threadIdx.x] = a + b;  // keep the loop alive
}

int main() {
  hipDeviceProp_t prop;
  HIP_CHECK(hipGetDeviceProperties(&prop, 0));
  int maxClockKHz = 0;
  HIP_CHECK(hipDeviceGetAttribute(&maxClockKHz, hipDeviceAttributeClockRate, 0));
  std::printf("Device: %s, reported peak SM clock %d MHz\n\n", prop.name,
              maxClockKHz / 1000);

  float *out = nullptr;
  long long *dCycles = nullptr;
  HIP_CHECK(hipMalloc(&out, 1024 * sizeof(float)));
  HIP_CHECK(hipMalloc(&dCycles, sizeof(long long)));

  // One block per SM/CU, so every unit is busy and block 0 runs the whole
  // time. Tune iters so one launch takes roughly 100-200 ms on your card.
  const int grid = prop.multiProcessorCount;
  const int block = 1024;
  const int iters = 1 << 23;

  hipEvent_t start, stop;
  HIP_CHECK(hipEventCreate(&start));
  HIP_CHECK(hipEventCreate(&stop));

  std::printf("%6s %8s %10s %12s\n", "launch", "t (s)", "time (ms)",
              "SM clock MHz");
  auto t0 = std::chrono::steady_clock::now();
  double minMs = 1e30, maxMs = 0.0, minMHz = 1e30, maxMHz = 0.0;
  for (int launch = 0;; ++launch) {
    HIP_CHECK(hipEventRecord(start, 0));
    fmaKernel<<<grid, block>>>(out, iters, dCycles);
    HIP_CHECK(hipEventRecord(stop, 0));
    HIP_CHECK(hipEventSynchronize(stop));
    HIP_CHECK(hipGetLastError());
    float ms = 0.0f;
    long long cycles = 0;
    HIP_CHECK(hipEventElapsedTime(&ms, start, stop));
    HIP_CHECK(hipMemcpy(&cycles, dCycles, sizeof(cycles),
                        hipMemcpyDeviceToHost));
    double secs = std::chrono::duration<double>(
                      std::chrono::steady_clock::now() - t0).count();
    double mhz = cycles / (ms * 1e3);  // cycles per microsecond = MHz
    if (launch >= 3) {  // the first launches include ramp-up from idle
      minMs = ms < minMs ? ms : minMs;
      maxMs = ms > maxMs ? ms : maxMs;
      minMHz = mhz < minMHz ? mhz : minMHz;
      maxMHz = mhz > maxMHz ? mhz : maxMHz;
    }
    // Print every launch for the first few, then about every 2 seconds.
    if (launch < 5 || launch % 15 == 0)
      std::printf("%6d %8.1f %10.1f %12.0f\n", launch, secs, ms, mhz);
    if (secs > 30.0) break;
  }
  std::printf("\nAfter warm-up:\n");
  std::printf("  time : fastest %.1f ms, slowest %.1f ms, spread %.1f%%\n",
              minMs, maxMs, 100.0 * (maxMs - minMs) / minMs);
  std::printf("  clock: lowest %.0f MHz, highest %.0f MHz, spread %.1f%%\n",
              minMHz, maxMHz, 100.0 * (maxMHz - minMHz) / minMHz);
  std::printf("If time spread is much bigger than clock spread, something "
              "other than the clock is varying: another process on the GPU "
              "(a desktop, if this card drives a display), or memory.\n");

  HIP_CHECK(hipEventDestroy(start));
  HIP_CHECK(hipEventDestroy(stop));
  HIP_CHECK(hipFree(out));
  HIP_CHECK(hipFree(dCycles));
  return 0;
}
