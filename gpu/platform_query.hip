// platform_query.hip
// Companion example for: topnotchnote.com/gpu/cuda_hip.html
//
// One HIP source file, two possible backends. This program reports which one
// it was built for and what is underneath: the HIP platform (AMD/ROCm or
// Nvidia/CUDA), the runtime and driver versions HIP reports, and each GPU's
// name and architecture. It then runs a one-line kernel to prove the stack
// works end to end.
//
// Compile:  hipcc -O2 -o main platform_query.hip
// Nvidia:   with HIP_PLATFORM=nvidia, hipcc builds this through nvcc; or swap
//           the hip prefix for cuda in the API calls and build with nvcc.

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

__global__ void hello(int* out) { out[threadIdx.x] = threadIdx.x * 2; }

int main() {
#if defined(__HIP_PLATFORM_AMD__)
  std::printf("built for: HIP on AMD (ROCm)\n");
#elif defined(__HIP_PLATFORM_NVIDIA__)
  std::printf("built for: HIP on Nvidia (CUDA underneath)\n");
#else
  std::printf("built for: plain CUDA (nvcc, no HIP headers)\n");
#endif
  int rt = 0, drv = 0, n = 0;
  HIP_CHECK(hipRuntimeGetVersion(&rt));
  HIP_CHECK(hipDriverGetVersion(&drv));
  HIP_CHECK(hipGetDeviceCount(&n));
  std::printf("runtime version %d, driver version %d, %d device(s)\n", rt, drv, n);
  for (int d = 0; d < n; ++d) {
    hipDeviceProp_t p;
    HIP_CHECK(hipGetDeviceProperties(&p, d));
#if defined(__HIP_PLATFORM_AMD__)
    std::printf("  [%d] %s, %s, warp/wavefront %d\n", d, p.name, p.gcnArchName, p.warpSize);
#else
    std::printf("  [%d] %s, compute capability %d.%d, warp %d\n", d, p.name, p.major,
                p.minor, p.warpSize);
#endif
  }
  int* out;
  int h[4];
  HIP_CHECK(hipMalloc(&out, sizeof(h)));
  hipLaunchKernelGGL(hello, dim3(1), dim3(4), 0, 0, out);
  HIP_CHECK(hipMemcpy(h, out, sizeof(h), hipMemcpyDeviceToHost));
  std::printf("kernel ran: %d %d %d %d\n", h[0], h[1], h[2], h[3]);
  return 0;
}
