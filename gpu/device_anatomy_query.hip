// device_anatomy_query.hip
// Companion example for: topnotchnote.com/gpu/amd_instinct_anatomy.html
//
// Prints the hardware facts that shape kernel code, as the runtime reports
// them: architecture name, compute units, wavefront (warp) size, LDS (shared
// memory) per CU and per block, registers, L2, HBM capacity and bus width,
// and the peak clocks. Compare the output with the vendor's spec table: on an
// MI355X the ROCm hardware table lists gfx950, 256 CUs, wavefront 64, 160 KiB
// LDS and 288 GiB of HBM; on an MI300X gfx942, 304 CUs, 64 and 64 KiB.
//
// It also derives one number kernel authors use all the time:
//   * resident lanes = CUs x max threads per CU
// It does not turn the memory clock into a bandwidth: how many transfers per
// reported clock a memory type makes (GDDR vs HBM generations) differs, so
// take peak bandwidth from the vendor's spec sheet instead.
//
// Compile:  hipcc -O2 -o main device_anatomy_query.hip
// Nvidia:   hipcc also targets Nvidia (HIP_PLATFORM=nvidia), or swap the hip
//           prefix for cuda in the API calls and build with nvcc.

#include <hip/hip_runtime.h>

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

static int attr(hipDeviceAttribute_t a, int dev) {
  int v = 0;
  HIP_CHECK(hipDeviceGetAttribute(&v, a, dev));
  return v;
}

int main() {
  int n = 0;
  HIP_CHECK(hipGetDeviceCount(&n));
  for (int d = 0; d < n; ++d) {
    hipDeviceProp_t p;
    HIP_CHECK(hipGetDeviceProperties(&p, d));
    std::printf("device %d: %s\n", d, p.name);
#if defined(__HIP_PLATFORM_AMD__)
    std::printf("  architecture (gfx target)   : %s\n", p.gcnArchName);
#else
    std::printf("  compute capability          : %d.%d\n", p.major, p.minor);
#endif
    int cus = p.multiProcessorCount;
    int clock_khz = attr(hipDeviceAttributeClockRate, d);
    int mem_khz = attr(hipDeviceAttributeMemoryClockRate, d);
    int bus_bits = attr(hipDeviceAttributeMemoryBusWidth, d);
    std::printf("  compute units (CU / SM)     : %d\n", cus);
    std::printf("  wavefront / warp size       : %d\n", p.warpSize);
    std::printf("  max threads per CU          : %d (%d waves)\n",
                p.maxThreadsPerMultiProcessor,
                p.maxThreadsPerMultiProcessor / p.warpSize);
    std::printf("  resident lanes, whole GPU   : %d\n",
                cus * p.maxThreadsPerMultiProcessor);
    std::printf("  LDS / shared per CU         : %zu KiB\n",
                p.maxSharedMemoryPerMultiProcessor >> 10);
    std::printf("  LDS / shared per block      : %zu KiB\n",
                p.sharedMemPerBlock >> 10);
    std::printf("  32-bit registers per block  : %d\n", p.regsPerBlock);
    std::printf("  L2 cache                    : %.1f MiB\n",
                p.l2CacheSize / 1048576.0);
    std::printf("  device memory               : %.1f GiB\n",
                p.totalGlobalMem / 1073741824.0);
    std::printf("  memory bus width            : %d bits\n", bus_bits);
    std::printf("  peak engine clock           : %d MHz\n", clock_khz / 1000);
    std::printf("  peak memory clock           : %d MHz\n", mem_khz / 1000);
  }
  return 0;
}
