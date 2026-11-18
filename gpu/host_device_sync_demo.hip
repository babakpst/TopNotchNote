// host_device_sync_demo.hip
// Companion example for: topnotchnote.com/gpu/host_device_sync.html
//
// An iterative loop of the kind used by solvers and training loops: each
// iteration launches a few small kernels and produces one number (a
// "residual") that the CPU wants to look at. Four ways to look at it:
//   A. copy the number back every iteration (hipMemcpy: the CPU waits)
//   B. copy it back every 10th iteration
//   C. never look (a lower bound on the time)
//   D. copy it with hipMemcpyAsync into a ring of kLag pinned slots, record an
//      event per slot, and read each value kLag iterations later, after
//      waiting on that slot's event (by then it has usually completed). Every
//      value is read, the stopping decision lags by kLag iterations, and the
//      CPU stays up to kLag iterations ahead of the GPU.
// The program prints the time per iteration for each, for a small and a
// large problem size.
//
// Compile:  hipcc -O2 -o main host_device_sync_demo.hip
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

__global__ void step(float* x, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) x[i] = x[i] * 0.999f + 0.001f;
}

// One block reduces the first 1024 elements to one number: enough to stand
// in for "compute a residual" without adding much GPU time.
__global__ void residual(const float* x, float* r) {
  __shared__ float s[1024];
  s[threadIdx.x] = x[threadIdx.x];
  __syncthreads();
  for (int k = 512; k > 0; k >>= 1) {
    if (threadIdx.x < k) s[threadIdx.x] += s[threadIdx.x + k];
    __syncthreads();
  }
  if (threadIdx.x == 0) *r = s[0];
}

constexpr int kKernelsPerIter = 4;
constexpr int kLag = 8;

static void iterate(float* x, int n, float* d_r) {
  for (int k = 0; k < kKernelsPerIter; ++k) step<<<(n + 255) / 256, 256>>>(x, n);
  residual<<<1, 1024>>>(x, d_r);
}

enum Mode { EVERY, EVERY10, NEVER, ASYNC };

static double run(Mode mode, float* x, int n, float* d_r, float* h_pinned, int iters) {
  hipEvent_t ev[kLag];
  for (auto& e : ev) HIP_CHECK(hipEventCreateWithFlags(&e, hipEventDisableTiming));
  HIP_CHECK(hipDeviceSynchronize());
  float seen = 0;
  int checks = 0;
  auto t0 = std::chrono::steady_clock::now();
  for (int it = 0; it < iters; ++it) {
    iterate(x, n, d_r);
    if (mode == EVERY || (mode == EVERY10 && it % 10 == 9)) {
      HIP_CHECK(hipMemcpy(&seen, d_r, sizeof(float), hipMemcpyDeviceToHost));  // waits
      ++checks;
    } else if (mode == ASYNC) {
      const int slot = it % kLag;
      if (it >= kLag) {
        // Read the value this slot got kLag iterations ago before reusing it.
        HIP_CHECK(hipEventSynchronize(ev[slot]));
        seen = h_pinned[slot];
        ++checks;
      }
      HIP_CHECK(hipMemcpyAsync(&h_pinned[slot], d_r, sizeof(float), hipMemcpyDeviceToHost, 0));
      HIP_CHECK(hipEventRecord(ev[slot], 0));
    }
    if (seen < -1.0f) break;   // a stopping test, never true here
  }
  HIP_CHECK(hipDeviceSynchronize());
  double us = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count();
  for (auto& e : ev) HIP_CHECK(hipEventDestroy(e));
  std::printf("  %-44s %8.1f us/iter   (%d values read)\n",
              mode == EVERY ? "A. hipMemcpy every iteration" :
              mode == EVERY10 ? "B. hipMemcpy every 10th iteration" :
              mode == NEVER ? "C. never read (lower bound)" :
              "D. async copy, read 8 iterations later",
              us / iters, checks);
  return us / iters;
}

int main() {
  const int iters = 2000;
  float *x, *d_r, *h_pinned;
  HIP_CHECK(hipMalloc(&x, (1 << 24) * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_r, sizeof(float)));
  HIP_CHECK(hipHostMalloc(&h_pinned, kLag * sizeof(float)));
  HIP_CHECK(hipMemset(x, 0, (1 << 24) * sizeof(float)));
  for (int n : {1 << 14, 1 << 22}) {
    std::printf("n = %d elements, %d kernels + 1 residual per iteration, %d iterations\n",
                n, kKernelsPerIter, iters);
    run(NEVER, x, n, d_r, h_pinned, 200);   // warm-up (printed, but ignore it)
    for (Mode m : {EVERY, EVERY10, NEVER, ASYNC}) run(m, x, n, d_r, h_pinned, iters);
  }
  HIP_CHECK(hipFree(x));
  HIP_CHECK(hipFree(d_r));
  HIP_CHECK(hipHostFree(h_pinned));
  return 0;
}
