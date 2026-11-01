// unified_memory_demo.hip
// Companion example for: topnotchnote.com/gpu/unified_memory.html
//
// Managed (unified) memory is one pointer valid on the CPU and the GPU.
// On GPUs with page-fault support the data moves on demand, one page at a
// time, the first time the other side touches it. This sample measures what
// that costs and what prefetching buys back:
//   1. which unified-memory features this device reports
//   2. a kernel scaling 256 MB in place (read + write) that the CPU just
//      wrote:
//        a. managed, first touch (page faults migrate it during the kernel)
//        b. managed, the same kernel again (already on the GPU)
//        c. managed, prefetched to the GPU first
//        d. hipMalloc + hipMemcpy, the explicit baseline
//   3. the CPU reading the data back after the GPU wrote it, with and
//      without a prefetch to the CPU
//   4. oversubscription: a managed array 1.25x larger than GPU memory,
//      swept twice by a kernel
//
// Compile:  hipcc -O2 -o main unified_memory_demo.hip
// AMD:      run with HSA_XNACK=1, or on-demand migration is disabled.
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

using Clock = std::chrono::steady_clock;
static double ms_since(Clock::time_point t0) {
  return std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
}

__global__ void scale(float* x, size_t n, float a) {
  for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n;
       i += (size_t)gridDim.x * blockDim.x)
    x[i] *= a;
}

static float kernel_ms(float* x, size_t n, float a) {
  hipEvent_t s, e;
  HIP_CHECK(hipEventCreate(&s));
  HIP_CHECK(hipEventCreate(&e));
  HIP_CHECK(hipEventRecord(s));
  scale<<<1024, 256>>>(x, n, a);
  HIP_CHECK(hipGetLastError());
  HIP_CHECK(hipEventRecord(e));
  HIP_CHECK(hipEventSynchronize(e));
  float ms;
  HIP_CHECK(hipEventElapsedTime(&ms, s, e));
  HIP_CHECK(hipEventDestroy(s));
  HIP_CHECK(hipEventDestroy(e));
  return ms;
}

static void cpu_fill(float* x, size_t n) {
  for (size_t i = 0; i < n; ++i) x[i] = 1.0f;
}
static double cpu_sum(const float* x, size_t n) {
  double s = 0;
  for (size_t i = 0; i < n; ++i) s += x[i];
  return s;
}

int main() {
  int dev = 0;
  hipDeviceProp_t prop;
  HIP_CHECK(hipGetDeviceProperties(&prop, dev));
  int managed = 0, concurrent = 0, pageable = 0;
  HIP_CHECK(hipDeviceGetAttribute(&managed, hipDeviceAttributeManagedMemory, dev));
  HIP_CHECK(hipDeviceGetAttribute(&concurrent,
                                  hipDeviceAttributeConcurrentManagedAccess, dev));
  HIP_CHECK(hipDeviceGetAttribute(&pageable,
                                  hipDeviceAttributePageableMemoryAccess, dev));
  std::printf("device: %s, %.1f GB\n", prop.name, prop.totalGlobalMem / 1e9);
  std::printf("1. managed memory %d, concurrent managed access %d, "
              "pageable memory access %d\n", managed, concurrent, pageable);
  if (!managed) return 0;

  // ---- 2 ----
  const size_t n = (256u << 20) / sizeof(float);
  const double mb = n * sizeof(float) / 1e6;
  float* m;
  HIP_CHECK(hipMallocManaged(&m, n * sizeof(float)));
  std::printf("\n2. kernel over %.0f MB the CPU just wrote\n", mb);

  cpu_fill(m, n);
  float t = kernel_ms(m, n, 2.f);
  std::printf("  a. managed, first touch on the GPU   %8.2f ms  (%5.1f GB/s)\n",
              t, 2 * mb / t);
  t = kernel_ms(m, n, 0.5f);
  std::printf("  b. managed, already on the GPU       %8.2f ms  (%5.1f GB/s)\n",
              t, 2 * mb / t);

  cpu_fill(m, n);  // move it back to the CPU
  auto t0 = Clock::now();
  if (concurrent) {
    HIP_CHECK(hipMemPrefetchAsync(m, n * sizeof(float), dev, 0));
    HIP_CHECK(hipDeviceSynchronize());
  }
  double tp = ms_since(t0);
  t = kernel_ms(m, n, 2.f);
  std::printf("  c. managed, prefetched first         %8.2f ms  "
              "+ prefetch %.2f ms (%5.1f GB/s)\n", t, tp, mb / tp);

  float *h, *d;
  HIP_CHECK(hipHostMalloc(&h, n * sizeof(float), hipHostMallocDefault));
  HIP_CHECK(hipMalloc(&d, n * sizeof(float)));
  cpu_fill(h, n);
  t0 = Clock::now();
  HIP_CHECK(hipMemcpy(d, h, n * sizeof(float), hipMemcpyHostToDevice));
  double tc = ms_since(t0);
  t = kernel_ms(d, n, 2.f);
  std::printf("  d. hipMalloc, after hipMemcpy        %8.2f ms  "
              "+ copy %.2f ms (%5.1f GB/s, pinned)\n", t, tc, mb / tc);

  // ---- 3 ----
  std::printf("\n3. the CPU reads %.0f MB the GPU just wrote\n", mb);
  kernel_ms(m, n, 1.f);
  t0 = Clock::now();
  double s = cpu_sum(m, n);
  std::printf("  page faults on the CPU      %8.2f ms  (sum %.0f)\n",
              ms_since(t0), s);
  kernel_ms(m, n, 1.f);
  t0 = Clock::now();
  if (concurrent) {
    HIP_CHECK(hipMemPrefetchAsync(m, n * sizeof(float), hipCpuDeviceId, 0));
    HIP_CHECK(hipDeviceSynchronize());
  }
  double tpc = ms_since(t0);
  t0 = Clock::now();
  s = cpu_sum(m, n);
  std::printf("  prefetched to the CPU first %8.2f ms  + prefetch %.2f ms "
              "(sum %.0f)\n", ms_since(t0), tpc, s);
  t0 = Clock::now();
  s = cpu_sum(m, n);
  std::printf("  read again, already on CPU  %8.2f ms  (sum %.0f)\n",
              ms_since(t0), s);

  HIP_CHECK(hipFree(m));
  HIP_CHECK(hipHostFree(h));
  HIP_CHECK(hipFree(d));

  // ---- 4 ----
  if (!concurrent) {
    std::printf("\n4. skipped: oversubscription needs concurrent managed "
                "access\n");
    return 0;
  }
  const size_t big = (size_t)(prop.totalGlobalMem * 1.25) / sizeof(float);
  const double gb = big * sizeof(float) / 1e9;
  float* o;
  HIP_CHECK(hipMallocManaged(&o, big * sizeof(float)));
  cpu_fill(o, big);
  std::printf("\n4. oversubscription: %.2f GB managed on a %.2f GB GPU\n", gb,
              prop.totalGlobalMem / 1e9);
  for (int pass = 1; pass <= 2; ++pass) {
    t = kernel_ms(o, big, 1.f);
    std::printf("  sweep %d                     %8.0f ms  (%5.2f GB/s)\n", pass,
                t, 2 * gb / t * 1e3);
  }
  t0 = Clock::now();
  s = cpu_sum(o, big);
  std::printf("  CPU check: sum %.0f, expected %zu\n", s, big);
  HIP_CHECK(hipFree(o));
  return 0;
}
