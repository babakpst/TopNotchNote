// reading_asm_demo.hip
// Companion example for: topnotchnote.com/gpu/reading_gpu_assembly.html
//
// The smallest kernel worth disassembling. Build it, dump the machine code
// for your GPU, and match each line of the listing to a line of source:
//   * blockIdx/blockDim are the same for the whole warp or wavefront, so the
//     compiler keeps them (and the scalar `a`) in scalar registers (AMD s#)
//     or uniform registers (Nvidia UR#).
//   * threadIdx, i, x[i] and y[i] differ per lane, so they live in vector
//     registers (AMD v#, Nvidia R#).
//   * `if (i < n)` becomes a per-lane compare: an EXEC-mask update on AMD,
//     a predicate (@P0) on Nvidia.
//   * `a * x[i] + y[i]` becomes one fused multiply-add.
//
// Dump the AMD ISA (keeps a .s file per target next to the object):
//   hipcc -O3 --offload-arch=gfx942 --save-temps -c reading_asm_demo.hip
//   less reading_asm_demo-hip-amdgcn-amd-amdhsa-gfx942.s
// Dump Nvidia SASS (with nvcc, rename to .cu or pass -x cu):
//   nvcc -O3 -arch=sm_90 -cubin -x cu -o saxpy.cubin reading_asm_demo.hip
//   cuobjdump -sass saxpy.cubin
// Run it (checks the result, so you know the code you are reading works):
//   hipcc -O3 -o main reading_asm_demo.hip && ./main

#include <hip/hip_runtime.h>

#include <cmath>
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

__global__ void saxpy(int n, float a, const float* x, float* y) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) y[i] = a * x[i] + y[i];
}

int main() {
  const int n = 1 << 20;
  std::vector<float> hx(n), hy(n);
  for (int i = 0; i < n; ++i) { hx[i] = float(i % 97); hy[i] = 1.0f; }
  float *x, *y;
  HIP_CHECK(hipMalloc(&x, n * sizeof(float)));
  HIP_CHECK(hipMalloc(&y, n * sizeof(float)));
  HIP_CHECK(hipMemcpy(x, hx.data(), n * sizeof(float), hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(y, hy.data(), n * sizeof(float), hipMemcpyHostToDevice));
  hipLaunchKernelGGL(saxpy, dim3((n + 255) / 256), dim3(256), 0, 0, n, 2.0f, x, y);
  HIP_CHECK(hipGetLastError());
  HIP_CHECK(hipMemcpy(hy.data(), y, n * sizeof(float), hipMemcpyDeviceToHost));
  int bad = 0;
  for (int i = 0; i < n; ++i)
    if (std::fabs(hy[i] - (2.0f * float(i % 97) + 1.0f)) > 0) ++bad;
  std::printf("saxpy on %d elements: %s\n", n, bad ? "WRONG" : "correct");
  HIP_CHECK(hipFree(x));
  HIP_CHECK(hipFree(y));
  return bad != 0;
}
