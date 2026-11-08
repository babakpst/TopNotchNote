// first_launch_cost.hip
// Companion example for: topnotchnote.com/gpu/compiling_gpu_code.html
//
// Measures how long the FIRST launch of a kernel takes compared with later
// ones. Build it twice and compare:
//   * with machine code for your exact GPU embedded (SASS on Nvidia, a code
//     object for your gfx target on AMD), the first launch only pays for
//     loading the module;
//   * on Nvidia, with only PTX embedded for an older virtual architecture,
//     the driver must JIT-compile the PTX to SASS at the first launch.
// On AMD there is no PTX-style fallback: a binary without a code object for
// your gfx target fails to launch, which this program reports as an error.
//
// Build (Nvidia, adjust 61 to your GPU's compute capability):
//   nvcc -O2 -x cu -arch=sm_61 -o native first_launch_cost.hip
//   nvcc -O2 -x cu -gencode arch=compute_52,code=compute_52 -o ptx_only first_launch_cost.hip
//   CUDA_CACHE_DISABLE=1 ./ptx_only     (otherwise the JIT result is cached)
// Build (AMD):
//   hipcc -O2 --offload-arch=gfx942 -o native first_launch_cost.hip

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

// A few kernels with some code in them, so there is something to compile.
template <int K>
__global__ void work(float* y, const float* x, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  float v = x[i];
#pragma unroll
  for (int k = 0; k < 64; ++k) v = v * 1.0001f + float(K + k) * 1e-6f;
  y[i] = v;
}

using clk = std::chrono::steady_clock;

template <int K>
static double launch_ms(float* y, const float* x, int n) {
  auto t = clk::now();
  hipLaunchKernelGGL(work<K>, dim3((n + 255) / 256), dim3(256), 0, 0, y, x, n);
  HIP_CHECK(hipGetLastError());
  HIP_CHECK(hipDeviceSynchronize());
  return std::chrono::duration<double, std::milli>(clk::now() - t).count();
}

int main() {
  auto t0 = clk::now();
  HIP_CHECK(hipFree(nullptr));  // context creation, timed on its own
  double ctx = std::chrono::duration<double, std::milli>(clk::now() - t0).count();
  const int n = 1 << 20;
  float *x, *y;
  HIP_CHECK(hipMalloc(&x, n * sizeof(float)));
  HIP_CHECK(hipMalloc(&y, n * sizeof(float)));
  HIP_CHECK(hipMemset(x, 0, n * sizeof(float)));

  double first = launch_ms<0>(y, x, n) + launch_ms<1>(y, x, n) + launch_ms<2>(y, x, n) +
                 launch_ms<3>(y, x, n);
  double again = launch_ms<0>(y, x, n) + launch_ms<1>(y, x, n) + launch_ms<2>(y, x, n) +
                 launch_ms<3>(y, x, n);
  std::printf("context creation          : %8.2f ms\n", ctx);
  std::printf("first launch of 4 kernels : %8.2f ms\n", first);
  std::printf("same 4 kernels again      : %8.2f ms\n", again);
  return 0;
}
