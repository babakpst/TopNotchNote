// memory_bandwidth_demo.hip
// Companion example for: topnotchnote.com/gpu/hbm_generations.html
//
// How fast can this GPU move bytes between its own memory and its cores?
//   1. Works out the paper number from two device attributes: the memory
//      bus width and the memory clock.
//   2. Measures three streaming kernels over a buffer far larger than the
//      L2 cache: read-only, write-only, and copy (read one buffer, write
//      another). Each thread moves 16 bytes per access, grid-stride.
//   3. Prints achieved bandwidth and the fraction of the paper number.
// The same program on an HBM card and a GDDR card shows the same shape at
// very different heights: HBM wins by width, not by clock.
//
// Compile:  hipcc -O2 -o main memory_bandwidth_demo.hip
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

__global__ void readKernel(const float4 *in, size_t n, float *sink) {
  float acc = 0.0f;
  for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n;
       i += (size_t)gridDim.x * blockDim.x) {
    float4 v = in[i];
    acc += v.x + v.y + v.z + v.w;
  }
  // Practically never true; stops the compiler deleting the loads.
  if (acc == -1.0f) *sink = acc;
}

__global__ void writeKernel(float4 *out, size_t n) {
  for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n;
       i += (size_t)gridDim.x * blockDim.x)
    out[i] = make_float4(1.0f, 2.0f, 3.0f, 4.0f);
}

__global__ void copyKernel(const float4 *in, float4 *out, size_t n) {
  for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n;
       i += (size_t)gridDim.x * blockDim.x)
    out[i] = in[i];
}

int main() {
  hipDeviceProp_t prop;
  HIP_CHECK(hipGetDeviceProperties(&prop, 0));
  int memClockKHz = 0, busWidthBits = 0;
  HIP_CHECK(hipDeviceGetAttribute(&memClockKHz,
                                  hipDeviceAttributeMemoryClockRate, 0));
  HIP_CHECK(hipDeviceGetAttribute(&busWidthBits,
                                  hipDeviceAttributeMemoryBusWidth, 0));

  // Paper bandwidth = transfers per second per pin x pins / 8.
  // The x2 assumes the reported clock moves data twice per cycle, which is
  // what Nvidia reports for GDDR5 (a GT 1030 says 3004 MHz, 64 bits, and
  // its spec sheet says 48 GB/s). How the reported clock relates to the
  // per-pin data rate differs between memory types and drivers, so check
  // the result against the spec sheet: an error of 2x or 4x here is the
  // most common mistake in this calculation.
  double paperGBs = 2.0 * memClockKHz * 1e3 * busWidthBits / 8.0 / 1e9;
  std::printf("Device            : %s\n", prop.name);
  std::printf("Memory bus width  : %d bits\n", busWidthBits);
  std::printf("Memory clock      : %d MHz (as reported)\n", memClockKHz / 1000);
  std::printf("Paper bandwidth   : %.1f GB/s\n", paperGBs);
  std::printf("L2 cache          : %d KB\n\n", prop.l2CacheSize >> 10);

  // Big enough that L2 cannot hide anything: at least 64x L2, at most a
  // quarter of free memory.
  size_t freeB = 0, totalB = 0;
  HIP_CHECK(hipMemGetInfo(&freeB, &totalB));
  size_t bytes = (size_t)prop.l2CacheSize * 64;
  if (bytes < (256u << 20)) bytes = 256u << 20;
  if (bytes > freeB / 4) bytes = freeB / 4;
  size_t n = bytes / sizeof(float4);
  bytes = n * sizeof(float4);

  float4 *a = nullptr, *b = nullptr;
  float *sink = nullptr;
  HIP_CHECK(hipMalloc(&a, bytes));
  HIP_CHECK(hipMalloc(&b, bytes));
  HIP_CHECK(hipMalloc(&sink, sizeof(float)));
  HIP_CHECK(hipMemset(a, 0, bytes));

  // Enough blocks to fill every SM/CU several times over.
  int block = 256;
  int grid = prop.multiProcessorCount * 8;

  hipEvent_t start, stop;
  HIP_CHECK(hipEventCreate(&start));
  HIP_CHECK(hipEventCreate(&stop));
  const int reps = 20;

  auto report = [&](const char *name, double bytesPerRep) {
    float ms = 0.0f;
    HIP_CHECK(hipEventElapsedTime(&ms, start, stop));
    double gbs = bytesPerRep * reps / (ms * 1e6);
    std::printf("%-6s %8.1f GB/s   %5.1f%% of paper\n", name, gbs,
                100.0 * gbs / paperGBs);
  };

  std::printf("Buffer: %zu MB, %d reps each\n", bytes >> 20, reps);

  readKernel<<<grid, block>>>(a, n, sink);  // warm-up
  HIP_CHECK(hipEventRecord(start, 0));
  for (int r = 0; r < reps; ++r) readKernel<<<grid, block>>>(a, n, sink);
  HIP_CHECK(hipEventRecord(stop, 0));
  HIP_CHECK(hipEventSynchronize(stop));
  report("read", (double)bytes);

  writeKernel<<<grid, block>>>(b, n);
  HIP_CHECK(hipEventRecord(start, 0));
  for (int r = 0; r < reps; ++r) writeKernel<<<grid, block>>>(b, n);
  HIP_CHECK(hipEventRecord(stop, 0));
  HIP_CHECK(hipEventSynchronize(stop));
  report("write", (double)bytes);

  copyKernel<<<grid, block>>>(a, b, n);
  HIP_CHECK(hipEventRecord(start, 0));
  for (int r = 0; r < reps; ++r) copyKernel<<<grid, block>>>(a, b, n);
  HIP_CHECK(hipEventRecord(stop, 0));
  HIP_CHECK(hipEventSynchronize(stop));
  report("copy", 2.0 * bytes);  // every byte is read once and written once

  HIP_CHECK(hipGetLastError());
  HIP_CHECK(hipEventDestroy(start));
  HIP_CHECK(hipEventDestroy(stop));
  HIP_CHECK(hipFree(a));
  HIP_CHECK(hipFree(b));
  HIP_CHECK(hipFree(sink));
  return 0;
}
