// flops_achieved_vs_peak.hip
// Companion example for: topnotchnote.com/gpu/flops_and_tflops.html
//
// Measures two things the spec sheet can only promise:
//   1. Achieved FP32 throughput. Every thread runs long chains of
//      independent fused multiply-adds (2 FLOPs each) with no memory traffic,
//      which is about as close to peak as general-purpose lanes get.
//   2. The clock the GPU actually held while doing it. The grid is exactly
//      one wave, so each block lives for the whole kernel and counts its
//      cycles; cycles divided by the kernel's time is the clock under this
//      load, to set against the boost clock. Both numbers come from the same
//      launches, so they describe the same moment.
// Pass the spec-sheet FP32 TFLOPS as an argument to get a percentage of peak.
// Compile:  hipcc -O3 -o main flops_achieved_vs_peak.hip
// Run:      ./main 163.4          (MI300X FP32 vector peak, for example)
// Nvidia:   hipcc also targets Nvidia (HIP_PLATFORM=nvidia), or swap the hip
//           prefix for cuda in the API calls and build with nvcc.
// Note:     on AMD, clock64() reads the shader clock counter, whose rate may
//           differ from the engine clock on some parts; treat the clock line
//           as indicative and confirm with rocm-smi or amd-smi while it runs.

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

constexpr int kChains = 8;       // independent FMA chains per thread, to hide FMA latency
constexpr int kIters = 65536;    // loop trips per launch: long enough to dwarf launch overhead

__global__ void fmaKernel(float *out, long long *cycles, float a, float b) {
  long long start = clock64();
  float x[kChains];
  for (int c = 0; c < kChains; ++c) x[c] = threadIdx.x + c;
  for (int i = 0; i < kIters; ++i) {
#pragma unroll
    for (int c = 0; c < kChains; ++c) x[c] = fmaf(x[c], a, b);
  }
  float sum = 0.0f;
  for (int c = 0; c < kChains; ++c) sum += x[c];
  out[blockIdx.x * blockDim.x + threadIdx.x] = sum;  // keeps the work from being optimized away
  if (threadIdx.x == 0) cycles[blockIdx.x] = clock64() - start;
}

int main(int argc, char **argv) {
  double specTflops = argc > 1 ? std::atof(argv[1]) : 0.0;

  hipDeviceProp_t p;
  HIP_CHECK(hipGetDeviceProperties(&p, 0));
  // Exactly one wave: as many blocks as the whole GPU holds at once, so every
  // block runs for the entire kernel and its cycle count spans the kernel.
  const int block = 256;
  int blocksPerUnit = 0;
  HIP_CHECK(hipOccupancyMaxActiveBlocksPerMultiprocessor(&blocksPerUnit, fmaKernel, block, 0));
  const int grid = blocksPerUnit * p.multiProcessorCount;
  const long long threads = static_cast<long long>(grid) * block;

  float *d_out = nullptr;
  long long *d_cycles = nullptr;
  HIP_CHECK(hipMalloc(&d_out, threads * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_cycles, grid * sizeof(long long)));

  for (int w = 0; w < 5; ++w) fmaKernel<<<grid, block>>>(d_out, d_cycles, 0.999f, 0.001f);  // ramp the clock
  HIP_CHECK(hipDeviceSynchronize());

  hipEvent_t start, stop;
  HIP_CHECK(hipEventCreate(&start));
  HIP_CHECK(hipEventCreate(&stop));
  const int reps = 20;
  HIP_CHECK(hipEventRecord(start));
  for (int r = 0; r < reps; ++r) fmaKernel<<<grid, block>>>(d_out, d_cycles, 0.999f, 0.001f);
  HIP_CHECK(hipEventRecord(stop));
  HIP_CHECK(hipEventSynchronize(stop));
  float ms = 0.0f;
  HIP_CHECK(hipEventElapsedTime(&ms, start, stop));

  double flops = 2.0 * kChains * kIters * static_cast<double>(threads) * reps;
  double tflops = flops / (ms * 1e-3) / 1e12;

  // Cycles counted by the last launch's blocks, over one launch's share of the time.
  long long *h_cycles = new long long[grid];
  HIP_CHECK(hipMemcpy(h_cycles, d_cycles, grid * sizeof(long long), hipMemcpyDeviceToHost));
  long long maxCycles = 0;
  for (int b = 0; b < grid; ++b) maxCycles = h_cycles[b] > maxCycles ? h_cycles[b] : maxCycles;
  double clockGHz = static_cast<double>(maxCycles) / (ms * 1e-3 / reps) / 1e9;

  std::printf("Device: %s, %d SMs/CUs\n", p.name, p.multiProcessorCount);
  std::printf("Achieved FP32:   %.2f TFLOPS  (%d blocks x %d threads, %d reps, %.2f ms)\n",
              tflops, grid, block, reps, ms);
  std::printf("Clock under load: about %.2f GHz\n", clockGHz);
  if (specTflops > 0.0) {
    std::printf("Spec-sheet peak: %.2f TFLOPS -> you reached %.0f%% of it\n", specTflops,
                100.0 * tflops / specTflops);
  } else {
    std::printf("Pass the spec-sheet FP32 TFLOPS as an argument to see the percentage.\n");
  }

  delete[] h_cycles;
  HIP_CHECK(hipEventDestroy(start));
  HIP_CHECK(hipEventDestroy(stop));
  HIP_CHECK(hipFree(d_cycles));
  HIP_CHECK(hipFree(d_out));
  return 0;
}
