// pinned_memory_demo.hip
// Companion example for: topnotchnote.com/gpu/pinned_memory.html
//
// Four experiments comparing ordinary (pageable) host memory with pinned
// (page-locked) host memory:
//   1. What pinning costs: allocating 256 MB with malloc and with
//      hipHostMalloc, with and without touching every page, and pinning an
//      existing buffer with hipHostRegister.
//   2. Copy bandwidth: host-to-device and device-to-host, pageable vs pinned,
//      from 64 KB to 256 MB.
//   3. "Async" in name only: how long hipMemcpyAsync takes to return to the
//      host from pageable vs pinned memory.
//   4. Overlap: a copy queued in one stream, then a kernel queued in another,
//      pageable vs pinned.
// Compile:  hipcc -O2 -o main pinned_memory_demo.hip
// Nvidia:   hipcc also targets Nvidia (HIP_PLATFORM=nvidia), or swap the hip
//           prefix for cuda in the API calls and build with nvcc.

#include <hip/hip_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>

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

static double msSince(Clock::time_point start) {
  return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
}

// Enough arithmetic per element that the kernel runs for a while.
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

// Best of `reps` copies, returned as GB/s. Timed with a host clock around
// the call and a synchronize, because for pageable memory part of the work
// is a host-side copy into a staging buffer that device events do not see.
static double copyGBs(void *dst, const void *src, size_t bytes, hipMemcpyKind kind,
                      int reps) {
  HIP_CHECK(hipMemcpy(dst, src, bytes, kind));  // warmup
  double best = 1e30;
  for (int r = 0; r < reps; ++r) {
    auto t = Clock::now();
    HIP_CHECK(hipMemcpyAsync(dst, src, bytes, kind, 0));
    HIP_CHECK(hipDeviceSynchronize());
    best = std::min(best, msSince(t));
  }
  return bytes / (best * 1e-3) / 1e9;
}

static void printSize(size_t bytes) {
  if (bytes < (1u << 20)) std::printf("   %6zu KB", bytes >> 10);
  else std::printf("   %6zu MB", bytes >> 20);
}

int main() {
  hipDeviceProp_t prop;
  HIP_CHECK(hipGetDeviceProperties(&prop, 0));
  std::printf("Device: %s\n\n", prop.name);
  HIP_CHECK(hipFree(nullptr));  // create the context before timing anything

  const size_t big = 256u << 20;

  // 1. What pinning costs. malloc only reserves address space; the OS backs
  //    each page on first touch. Pinning has to back and lock every page up
  //    front, so compare both the call alone and the call plus a full touch.
  std::printf("1. allocate 256 MB\n");
  auto t = Clock::now();
  char *pageable = static_cast<char *>(std::malloc(big));
  double mallocMs = msSince(t);
  std::memset(pageable, 1, big);
  std::printf("   malloc:                    %7.1f ms   + touch every page: %7.1f ms\n",
              mallocMs, msSince(t));

  t = Clock::now();
  char *pinned = nullptr;
  HIP_CHECK(hipHostMalloc(reinterpret_cast<void **>(&pinned), big, hipHostMallocDefault));
  double pinMs = msSince(t);
  std::memset(pinned, 1, big);
  std::printf("   hipHostMalloc:             %7.1f ms   + touch every page: %7.1f ms\n",
              pinMs, msSince(t));

  char *registered = static_cast<char *>(std::malloc(big));
  std::memset(registered, 1, big);
  t = Clock::now();
  HIP_CHECK(hipHostRegister(registered, big, hipHostRegisterDefault));
  std::printf("   hipHostRegister (existing):%7.1f ms\n", msSince(t));

  t = Clock::now();
  HIP_CHECK(hipHostUnregister(registered));
  std::free(registered);
  std::printf("   unregister + free:         %7.1f ms\n", msSince(t));
  std::printf("\n");

  char *d_buf = nullptr;
  HIP_CHECK(hipMalloc(&d_buf, big));

  // 2. Copy bandwidth.
  std::printf("2. copy bandwidth, best of 10 (GB/s)\n");
  std::printf("   %8s  %12s %12s  %12s %12s\n", "size", "H2D pageable", "H2D pinned",
              "D2H pageable", "D2H pinned");
  const size_t sizes[] = {64u << 10, 1u << 20, 16u << 20, 256u << 20};
  for (size_t bytes : sizes) {
    double hp = copyGBs(d_buf, pageable, bytes, hipMemcpyHostToDevice, 10);
    double hn = copyGBs(d_buf, pinned, bytes, hipMemcpyHostToDevice, 10);
    double dp = copyGBs(pageable, d_buf, bytes, hipMemcpyDeviceToHost, 10);
    double dn = copyGBs(pinned, d_buf, bytes, hipMemcpyDeviceToHost, 10);
    printSize(bytes);
    std::printf("  %12.2f %12.2f  %12.2f %12.2f\n", hp, hn, dp, dn);
  }
  std::printf("\n");

  // 3. How long does the "async" call take to come back?
  const size_t mid = 64u << 20;
  std::printf("3. hipMemcpyAsync of 64 MB, host-to-device: time until the call returns\n");
  for (int which = 0; which < 2; ++which) {
    char *src = which == 0 ? pageable : pinned;
    HIP_CHECK(hipDeviceSynchronize());
    t = Clock::now();
    HIP_CHECK(hipMemcpyAsync(d_buf, src, mid, hipMemcpyHostToDevice, 0));
    double returned = msSince(t);
    HIP_CHECK(hipDeviceSynchronize());
    double done = msSince(t);
    std::printf("   %-8s returned after %7.3f ms, copy finished after %6.1f ms\n",
                which == 0 ? "pageable" : "pinned", returned, done);
  }
  std::printf("\n");

  // 4. Overlap: a 64 MB copy in one stream, then a kernel in another. The copy
  //    is queued first, as it would be when the kernel needs the next chunk.
  const int n = 1 << 22;
  float *d_work = nullptr;
  HIP_CHECK(hipMalloc(&d_work, n * sizeof(float)));
  HIP_CHECK(hipMemset(d_work, 0, n * sizeof(float)));
  hipStream_t computeStream, copyStream;
  HIP_CHECK(hipStreamCreate(&computeStream));
  HIP_CHECK(hipStreamCreate(&copyStream));
  const int block = 256, grid = (n + block - 1) / block;

  // Size the kernel to take about as long as the pinned copy.
  int iters = 200;
  busyKernel<<<grid, block, 0, computeStream>>>(d_work, n, iters);
  HIP_CHECK(hipDeviceSynchronize());
  t = Clock::now();
  busyKernel<<<grid, block, 0, computeStream>>>(d_work, n, iters);
  HIP_CHECK(hipDeviceSynchronize());
  double kernelMs = msSince(t);
  t = Clock::now();
  HIP_CHECK(hipMemcpyAsync(d_buf, pinned, mid, hipMemcpyHostToDevice, copyStream));
  HIP_CHECK(hipDeviceSynchronize());
  double copyMs = msSince(t);
  iters = std::max(1, static_cast<int>(iters * copyMs / kernelMs));
  t = Clock::now();
  busyKernel<<<grid, block, 0, computeStream>>>(d_work, n, iters);
  HIP_CHECK(hipDeviceSynchronize());
  kernelMs = msSince(t);

  std::printf("4. 64 MB copy (%.1f ms alone), then kernel (%.1f ms alone), in two streams\n",
              copyMs, kernelMs);
  for (int which = 0; which < 2; ++which) {
    char *src = which == 0 ? pageable : pinned;
    double best = 1e30;
    for (int r = 0; r < 5; ++r) {
      HIP_CHECK(hipDeviceSynchronize());
      t = Clock::now();
      HIP_CHECK(hipMemcpyAsync(d_buf, src, mid, hipMemcpyHostToDevice, copyStream));
      busyKernel<<<grid, block, 0, computeStream>>>(d_work, n, iters);
      HIP_CHECK(hipDeviceSynchronize());
      best = std::min(best, msSince(t));
    }
    std::printf("   %-8s both done after %6.1f ms\n", which == 0 ? "pageable" : "pinned", best);
  }

  HIP_CHECK(hipStreamDestroy(computeStream));
  HIP_CHECK(hipStreamDestroy(copyStream));
  HIP_CHECK(hipFree(d_work));
  HIP_CHECK(hipFree(d_buf));
  HIP_CHECK(hipHostFree(pinned));
  std::free(pageable);
  return 0;
}
