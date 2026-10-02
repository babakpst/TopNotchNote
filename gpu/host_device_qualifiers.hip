// host_device_qualifiers.hip
// Companion example for: topnotchnote.com/gpu/host_device_qualifiers.html
//
// One file, two compilers. hipcc (and nvcc) compile this source twice:
// once for the CPU and once for the GPU. The qualifier on each function
// decides which of the two passes keeps it.
//   1. __host__ __device__: the same function runs on both sides. The
//      program evaluates one polynomial on the CPU and on the GPU and
//      compares the results bit for bit.
//   2. The macro that tells you which pass you are in: compiledOnDevice()
//      returns a different answer depending on where it was compiled.
//   3. Why the two sides can disagree in the last bit: the device compiler
//      fuses a*b+c into one FMA instruction by default; the host compiler
//      usually does not. Rebuild with contraction off and they agree.
// Compile:  hipcc -O2 -o main host_device_qualifiers.hip
//           hipcc -O2 -ffp-contract=off -o main host_device_qualifiers.hip
// Nvidia:   hipcc also targets Nvidia (HIP_PLATFORM=nvidia), or swap the hip
//           prefix for cuda in the API calls and build with nvcc
//           (nvcc -O2 -x cu ..., and --fmad=false in place of -ffp-contract=off).

#include <hip/hip_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
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

// Compiled in BOTH passes. Horner's rule for a degree-7 polynomial: seven
// multiply-adds in a row, each one a chance for the compiler to fuse.
__host__ __device__ float poly(float x) {
  float r = 0.0078125f;
  r = r * x - 0.0416667f;
  r = r * x + 0.1666667f;
  r = r * x - 0.5000000f;
  r = r * x + 1.0000000f;
  r = r * x - 0.3333333f;
  r = r * x + 0.2500000f;
  r = r * x + 1.0000000f;
  return r;
}

// Compiled in both passes, but each pass sees a different body.
// __HIP_DEVICE_COMPILE__ is defined only during the device pass
// (on Nvidia the equivalent test is __CUDA_ARCH__).
__host__ __device__ int compiledOnDevice() {
#if defined(__HIP_DEVICE_COMPILE__) || defined(__CUDA_ARCH__)
  return 1;
#else
  return 0;
#endif
}

// Device pass only. It reads threadIdx and blockIdx, which have no meaning
// on the CPU. Callable from a kernel or another __device__ function, never
// from main(): uncomment the line marked (*) in main() to see the error.
__device__ int globalIndex() { return blockIdx.x * blockDim.x + threadIdx.x; }

// Device pass, plus a small host-side launch stub so main() can launch it.
__global__ void evalKernel(const float *x, float *y, int *where, int n) {
  int i = globalIndex();
  if (i < n) y[i] = poly(x[i]);
  if (i == 0) *where = compiledOnDevice();
}

// How many representable floats apart a and b are.
static int64_t ulpDistance(float a, float b) {
  int32_t ia, ib;
  std::memcpy(&ia, &a, sizeof ia);
  std::memcpy(&ib, &b, sizeof ib);
  if (ia < 0) ia = INT32_MIN - ia;  // map negative floats onto a monotonic scale
  if (ib < 0) ib = INT32_MIN - ib;
  return std::llabs(static_cast<int64_t>(ia) - static_cast<int64_t>(ib));
}

int main() {
  const int n = 1 << 20;
  std::vector<float> h_x(n), h_gpu(n), h_cpu(n);
  for (int i = 0; i < n; ++i) h_x[i] = -2.0f + 4.0f * i / (n - 1);  // x in [-2, 2]

  float *d_x = nullptr, *d_y = nullptr;
  int *d_where = nullptr;
  HIP_CHECK(hipMalloc(&d_x, n * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_y, n * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_where, sizeof(int)));
  HIP_CHECK(hipMemcpy(d_x, h_x.data(), n * sizeof(float), hipMemcpyHostToDevice));

  const int block = 256;
  evalKernel<<<(n + block - 1) / block, block>>>(d_x, d_y, d_where, n);
  HIP_CHECK(hipGetLastError());
  HIP_CHECK(hipMemcpy(h_gpu.data(), d_y, n * sizeof(float), hipMemcpyDeviceToHost));
  int whereGpu = 0;
  HIP_CHECK(hipMemcpy(&whereGpu, d_where, sizeof(int), hipMemcpyDeviceToHost));

  // int bad = globalIndex();  // (*) error: a __device__ function called from host code

  // The very same function, called from ordinary host code.
  for (int i = 0; i < n; ++i) h_cpu[i] = poly(h_x[i]);

  std::printf("compiledOnDevice() called from the CPU: %d\n", compiledOnDevice());
  std::printf("compiledOnDevice() called from the GPU: %d\n\n", whereGpu);

  int differ = 0;
  int64_t worst = 0;
  int worstAt = 0;
  for (int i = 0; i < n; ++i) {
    int64_t d = ulpDistance(h_cpu[i], h_gpu[i]);
    if (d != 0) ++differ;
    if (d > worst) { worst = d; worstAt = i; }
  }
  std::printf("poly(x) for %d inputs, CPU vs GPU:\n", n);
  std::printf("  bit-identical:  %d (%.1f%%)\n", n - differ, 100.0 * (n - differ) / n);
  std::printf("  differ:         %d (%.1f%%)\n", differ, 100.0 * differ / n);
  std::printf("  worst case:     %lld ulp at x = %.7f (CPU %.9g, GPU %.9g)\n",
              static_cast<long long>(worst), h_x[worstAt], h_cpu[worstAt], h_gpu[worstAt]);

  HIP_CHECK(hipFree(d_where));
  HIP_CHECK(hipFree(d_y));
  HIP_CHECK(hipFree(d_x));
  return 0;
}
