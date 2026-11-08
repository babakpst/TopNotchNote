// gpudirect_staging_cost.hip
// Companion example for: topnotchnote.com/gpu/gpudirect_rdma.html
//
// Without GPUDirect RDMA (Nvidia) or PeerDirect (AMD), a GPU buffer cannot be
// handed to the network card directly. The communication library first copies
// it into a pinned host buffer, the NIC sends from there, and the receiver
// does the reverse: NIC -> host buffer -> GPU. Each message pays one extra
// device-to-host copy on the way out and one host-to-device copy on the way
// in, on top of the network itself.
//
// No network card is needed to see that cost: this program times exactly
// those two staging copies for message sizes from 4 KiB to 256 MiB, which is
// the time GPUDirect RDMA removes from every send and receive.
//
// Compile:  hipcc -O2 -o main gpudirect_staging_cost.hip
// Nvidia:   hipcc also targets Nvidia (HIP_PLATFORM=nvidia), or swap the hip
//           prefix for cuda in the API calls and build with nvcc.

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

static float median_ms(void* dst, const void* src, size_t bytes, hipMemcpyKind kind) {
  hipEvent_t s, e;
  HIP_CHECK(hipEventCreate(&s));
  HIP_CHECK(hipEventCreate(&e));
  int reps = bytes < (1u << 20) ? 201 : 11;
  std::vector<float> t;
  for (int r = 0; r < reps; ++r) {
    HIP_CHECK(hipEventRecord(s));
    HIP_CHECK(hipMemcpyAsync(dst, src, bytes, kind, 0));
    HIP_CHECK(hipEventRecord(e));
    HIP_CHECK(hipEventSynchronize(e));
    float ms;
    HIP_CHECK(hipEventElapsedTime(&ms, s, e));
    if (r) t.push_back(ms);  // first copy is warm-up
  }
  std::sort(t.begin(), t.end());
  HIP_CHECK(hipEventDestroy(s));
  HIP_CHECK(hipEventDestroy(e));
  return t[t.size() / 2];
}

int main() {
  const size_t max_bytes = size_t(256) << 20;
  void *dev, *host;
  HIP_CHECK(hipMalloc(&dev, max_bytes));
  HIP_CHECK(hipHostMalloc(&host, max_bytes, hipHostMallocDefault));
  HIP_CHECK(hipMemset(dev, 1, max_bytes));

  std::printf("staging copies a non-GPUDirect send/receive adds per message\n\n");
  std::printf("%10s %12s %12s %12s %10s\n", "message", "GPU->host", "host->GPU",
              "both (us)", "GB/s");
  for (size_t b = 4096; b <= max_bytes; b *= 4) {
    float d2h = median_ms(host, dev, b, hipMemcpyDeviceToHost);
    float h2d = median_ms(dev, host, b, hipMemcpyHostToDevice);
    float both_us = (d2h + h2d) * 1e3f;
    const char* unit = b >= (1u << 20) ? "MiB" : "KiB";
    size_t shown = b >= (1u << 20) ? b >> 20 : b >> 10;
    std::printf("%6zu %-3s %10.1fus %10.1fus %12.1f %10.2f\n", shown, unit,
                d2h * 1e3f, h2d * 1e3f, both_us, 2.0 * b / (both_us * 1e-6) / 1e9);
  }
  HIP_CHECK(hipFree(dev));
  HIP_CHECK(hipHostFree(host));
  return 0;
}
