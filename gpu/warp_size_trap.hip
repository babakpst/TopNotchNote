// warp_size_trap.hip
// Companion example for: topnotchnote.com/gpu/porting_cuda_to_hip.html
//
// The bug hipify cannot see. CUDA code often hardcodes 32 as the warp size,
// for example in a shuffle reduction that starts at offset 16. hipify
// translates the calls, the code compiles for AMD, and on a 64-wide
// wavefront (AMD Instinct CDNA 1-4) it silently sums only half the lanes.
//
// This program:
//   1. runs both reductions on your GPU, the hardcoded-32 one and one written
//      against warpSize, and prints the device's warp size;
//   2. emulates both on the CPU for a 64-lane wavefront, lane by lane, so the
//      wrong answer is visible even on an Nvidia card.
//
// Compile:  hipcc -O2 -o main warp_size_trap.hip
// Nvidia:   hipcc also targets Nvidia (HIP_PLATFORM=nvidia), or rename to .cu
//           and build with nvcc.

#include <hip/hip_runtime.h>

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

// One source for both vendors: CUDA needs the _sync form with a lane mask.
__device__ inline float shfl_down(float v, int off) {
#if defined(__HIP_PLATFORM_AMD__)
  return __shfl_down(v, off);
#else
  return __shfl_down_sync(0xffffffffu, v, off);
#endif
}

// Ported as-is from CUDA: assumes 32 lanes.
__global__ void reduce_hardcoded(const float* x, float* out) {
  float v = x[threadIdx.x];
  for (int off = 16; off > 0; off /= 2) v += shfl_down(v, off);
  if (threadIdx.x == 0) *out = v;
}

// Portable: starts from the device's real warp size.
__global__ void reduce_portable(const float* x, float* out) {
  float v = x[threadIdx.x];
  for (int off = warpSize / 2; off > 0; off /= 2) v += shfl_down(v, off);
  if (threadIdx.x == 0) *out = v;
}

// CPU emulation of a shuffle-down reduction across `lanes` lanes. Each step
// reads the value `off` lanes higher (out-of-range lanes keep their own value,
// as the hardware does); lane 0 ends with the result.
static float emulate(std::vector<float> v, int lanes, int start) {
  for (int off = start; off > 0; off /= 2) {
    std::vector<float> nv(v);
    for (int l = 0; l < lanes; ++l) nv[l] = v[l] + (l + off < lanes ? v[l + off] : v[l]);
    v = nv;
  }
  return v[0];
}

int main() {
  hipDeviceProp_t p;
  HIP_CHECK(hipGetDeviceProperties(&p, 0));
  const int W = p.warpSize;
  std::vector<float> h(W, 1.0f);  // every lane holds 1, so the answer is W
  float *x, *out, r1, r2;
  HIP_CHECK(hipMalloc(&x, W * sizeof(float)));
  HIP_CHECK(hipMalloc(&out, sizeof(float)));
  HIP_CHECK(hipMemcpy(x, h.data(), W * sizeof(float), hipMemcpyHostToDevice));
  hipLaunchKernelGGL(reduce_hardcoded, dim3(1), dim3(W), 0, 0, x, out);
  HIP_CHECK(hipMemcpy(&r1, out, sizeof(float), hipMemcpyDeviceToHost));
  hipLaunchKernelGGL(reduce_portable, dim3(1), dim3(W), 0, 0, x, out);
  HIP_CHECK(hipMemcpy(&r2, out, sizeof(float), hipMemcpyDeviceToHost));
  std::printf("%s: warp size %d, every lane = 1, correct sum = %d\n", p.name, W, W);
  std::printf("  on this GPU, hardcoded 32 : %g\n", r1);
  std::printf("  on this GPU, warpSize     : %g\n", r2);

  std::vector<float> ones(64, 1.0f);
  std::printf("emulated 64-lane wavefront (AMD CDNA 1-4), correct sum = 64\n");
  std::printf("  hardcoded 32 (offsets 16..1): %g\n", emulate(ones, 64, 16));
  std::printf("  warpSize     (offsets 32..1): %g\n", emulate(ones, 64, 32));
  return 0;
}
