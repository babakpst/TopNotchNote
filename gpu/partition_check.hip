// partition_check.hip
// Companion example for: topnotchnote.com/gpu/mig.html
//
// When a GPU is partitioned (MIG on Nvidia, CPX/DPX/QPX compute partitions on
// AMD Instinct), each partition shows up to your program as its own device
// with fewer SMs/CUs, less memory and less bandwidth. A program that
// hardcodes "132 SMs" or "256 CUs" will size its grid wrongly inside one.
// This program prints what each visible device actually is, then measures
// its memory bandwidth with a simple copy kernel, so you can compare a whole
// GPU with one partition of it.
//
// Compile:  hipcc -O2 -o main partition_check.hip
// Nvidia:   hipcc also targets Nvidia (HIP_PLATFORM=nvidia), or swap the hip
//           prefix for cuda in the API calls and build with nvcc.
// Run:      ./main
//   Nvidia MIG: CUDA_VISIBLE_DEVICES=MIG-<uuid> ./main   (uuids: nvidia-smi -L)
//   AMD:        HIP_VISIBLE_DEVICES=<n> ./main           (one partition per n)

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

// Grid-stride copy: sized from the device's own CU count, never hardcoded.
__global__ void copy(const float4* __restrict__ in, float4* __restrict__ out,
                     size_t n) {
  for (size_t i = blockIdx.x * size_t(blockDim.x) + threadIdx.x; i < n;
       i += size_t(gridDim.x) * blockDim.x)
    out[i] = in[i];
}

int main() {
  int n = 0;
  HIP_CHECK(hipGetDeviceCount(&n));
  std::printf("visible devices: %d\n", n);
  for (int d = 0; d < n; ++d) {
    HIP_CHECK(hipSetDevice(d));
    hipDeviceProp_t p;
    HIP_CHECK(hipGetDeviceProperties(&p, d));
    size_t free_b = 0, total_b = 0;
    HIP_CHECK(hipMemGetInfo(&free_b, &total_b));
    std::printf("\n[%d] %s\n  uuid      : ", d, p.name);
    for (int i = 0; i < 16; ++i)
      std::printf("%02x%s", (unsigned char)p.uuid.bytes[i],
                  (i == 3 || i == 5 || i == 7 || i == 9) ? "-" : "");
    std::printf("\n  SMs/CUs   : %d\n", p.multiProcessorCount);
    std::printf("  memory    : %.1f GiB total, %.1f GiB free\n",
                total_b / 1073741824.0, free_b / 1073741824.0);
    std::printf("  L2        : %.1f MiB\n", p.l2CacheSize / 1048576.0);

    // Copy a quarter of free memory (capped at 1 GiB) in each direction.
    size_t bytes = std::min<size_t>(free_b / 4, size_t(1) << 30);
    bytes -= bytes % sizeof(float4);
    size_t elems = bytes / sizeof(float4);
    float4 *a, *b;
    HIP_CHECK(hipMalloc(&a, bytes));
    HIP_CHECK(hipMalloc(&b, bytes));
    HIP_CHECK(hipMemset(a, 0, bytes));
    int blocks = p.multiProcessorCount * 8;  // scales with the partition
    hipEvent_t start, stop;
    HIP_CHECK(hipEventCreate(&start));
    HIP_CHECK(hipEventCreate(&stop));
    std::vector<float> ms;
    for (int r = 0; r < 11; ++r) {
      HIP_CHECK(hipEventRecord(start));
      hipLaunchKernelGGL(copy, dim3(blocks), dim3(256), 0, 0, a, b, elems);
      HIP_CHECK(hipEventRecord(stop));
      HIP_CHECK(hipEventSynchronize(stop));
      float t;
      HIP_CHECK(hipEventElapsedTime(&t, start, stop));
      if (r) ms.push_back(t);
    }
    std::sort(ms.begin(), ms.end());
    double gbps = 2.0 * bytes / (ms[ms.size() / 2] * 1e-3) / 1e9;
    std::printf("  copy bandwidth (read + write): %.1f GB/s over %zu MiB\n", gbps,
                bytes >> 20);
    HIP_CHECK(hipFree(a));
    HIP_CHECK(hipFree(b));
  }
  return 0;
}
