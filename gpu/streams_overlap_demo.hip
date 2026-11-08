// streams_overlap_demo.hip
// Companion example for: topnotchnote.com/gpu/streams.html
//
// Copy in, compute, copy out, for 256 MiB of data, three ways:
//   1. everything in the default stream, one big chunk;
//   2. split into chunks, still one stream (no overlap possible);
//   3. split into chunks across S streams from pinned memory, so chunk k's
//      kernel can run while chunk k+1 is copied in and chunk k-1 copied out.
// It prints the time of each and the GPU's copy-engine count, which decides
// whether copies in both directions can overlap each other or only overlap
// the kernels.
//
// Compile:  hipcc -O2 -o main streams_overlap_demo.hip
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

// Enough arithmetic per element that the kernel takes about as long as a copy.
__global__ void crunch(float* d, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  float v = d[i];
  for (int k = 0; k < 200; ++k) v = v * 0.999f + 0.001f;
  d[i] = v;
}

using clk = std::chrono::steady_clock;

int main() {
  const int n = 64 << 20;  // 256 MiB of floats
  const size_t bytes = size_t(n) * sizeof(float);
  float *h, *d;
  HIP_CHECK(hipHostMalloc(&h, bytes, hipHostMallocDefault));  // pinned
  HIP_CHECK(hipMalloc(&d, bytes));
  std::fill(h, h + n, 1.0f);
  int engines = 0;
  HIP_CHECK(hipDeviceGetAttribute(&engines, hipDeviceAttributeAsyncEngineCount, 0));

  auto timed = [&](auto body) {
    body();  // warm-up
    HIP_CHECK(hipDeviceSynchronize());
    std::vector<double> t;
    for (int r = 0; r < 5; ++r) {
      auto t0 = clk::now();
      body();
      HIP_CHECK(hipDeviceSynchronize());
      t.push_back(std::chrono::duration<double, std::milli>(clk::now() - t0).count());
    }
    std::sort(t.begin(), t.end());
    return t[2];
  };

  double one = timed([&] {
    HIP_CHECK(hipMemcpy(d, h, bytes, hipMemcpyHostToDevice));
    hipLaunchKernelGGL(crunch, dim3((n + 255) / 256), dim3(256), 0, 0, d, n);
    HIP_CHECK(hipMemcpy(h, d, bytes, hipMemcpyDeviceToHost));
  });

  const int chunks = 16, cn = n / chunks;
  const size_t cb = size_t(cn) * sizeof(float);
  auto chunked = [&](std::vector<hipStream_t>& s) {
    for (int c = 0; c < chunks; ++c) {
      hipStream_t st = s[c % s.size()];
      size_t off = size_t(c) * cn;
      HIP_CHECK(hipMemcpyAsync(d + off, h + off, cb, hipMemcpyHostToDevice, st));
      hipLaunchKernelGGL(crunch, dim3((cn + 255) / 256), dim3(256), 0, st, d + off, cn);
      HIP_CHECK(hipMemcpyAsync(h + off, d + off, cb, hipMemcpyDeviceToHost, st));
    }
  };

  std::printf("256 MiB in, kernel, out; GPU copy engines: %d\n", engines);
  std::printf("  1 stream, one chunk      : %8.1f ms\n", one);
  for (int ns : {1, 2, 4, 8}) {
    std::vector<hipStream_t> s(ns);
    for (auto& x : s) HIP_CHECK(hipStreamCreate(&x));
    double t = timed([&] { chunked(s); });
    std::printf("  %d stream%s, %d chunks     : %8.1f ms  (%.2fx)\n", ns, ns > 1 ? "s" : " ",
                chunks, t, one / t);
    for (auto& x : s) HIP_CHECK(hipStreamDestroy(x));
  }
  HIP_CHECK(hipHostFree(h));
  HIP_CHECK(hipFree(d));
  return 0;
}
