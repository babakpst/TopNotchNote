// hip_vector_add.hip
// Companion example for: topnotchnote.com/gpu/opencl_cuda_hip.html
//
// The HIP twin of opencl_vector_add.cpp: same kernel, same problem size,
// same timing method (events around the kernel, averaged over 20 runs).
// Compare the two files for how much host code each model needs, and
// compare their output for how fast the kernel runs.
// Compile:  hipcc -O2 -o main hip_vector_add.hip
// Nvidia:   hipcc also targets Nvidia (HIP_PLATFORM=nvidia), or swap the hip
//           prefix for cuda in the API calls and build with nvcc.

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

// Compiled ahead of time, in the same file as the host code.
__global__ void vecAdd(const float *a, const float *b, float *c, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) c[i] = a[i] + b[i];
}

int main() {
  const int n = 1 << 24;
  const size_t bytes = n * sizeof(float);
  std::vector<float> h_a(n, 1.0f), h_b(n, 2.0f), h_c(n);

  hipDeviceProp_t prop;
  HIP_CHECK(hipGetDeviceProperties(&prop, 0));
  std::printf("Device: %s\n", prop.name);

  float *d_a, *d_b, *d_c;
  HIP_CHECK(hipMalloc(&d_a, bytes));
  HIP_CHECK(hipMalloc(&d_b, bytes));
  HIP_CHECK(hipMalloc(&d_c, bytes));
  HIP_CHECK(hipMemcpy(d_a, h_a.data(), bytes, hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(d_b, h_b.data(), bytes, hipMemcpyHostToDevice));

  const int block = 256;
  const int grid = (n + block - 1) / block;
  vecAdd<<<grid, block>>>(d_a, d_b, d_c, n);  // warm-up
  HIP_CHECK(hipGetLastError());
  HIP_CHECK(hipDeviceSynchronize());

  hipEvent_t start, stop;
  HIP_CHECK(hipEventCreate(&start));
  HIP_CHECK(hipEventCreate(&stop));
  const int reps = 20;
  float totalMs = 0.0f;
  for (int r = 0; r < reps; ++r) {
    HIP_CHECK(hipEventRecord(start));
    vecAdd<<<grid, block>>>(d_a, d_b, d_c, n);
    HIP_CHECK(hipEventRecord(stop));
    HIP_CHECK(hipEventSynchronize(stop));
    float ms = 0.0f;
    HIP_CHECK(hipEventElapsedTime(&ms, start, stop));
    totalMs += ms;
  }

  HIP_CHECK(hipMemcpy(h_c.data(), d_c, bytes, hipMemcpyDeviceToHost));
  int wrong = 0;
  for (int i = 0; i < n; ++i) if (h_c[i] != 3.0f) ++wrong;

  const double ms = totalMs / reps;
  std::printf("HIP vecAdd, n = %d: %.3f ms, %.1f GB/s, %d wrong\n",
              n, ms, 3.0 * bytes / (ms * 1e-3) / 1e9, wrong);

  HIP_CHECK(hipEventDestroy(start));
  HIP_CHECK(hipEventDestroy(stop));
  HIP_CHECK(hipFree(d_c));
  HIP_CHECK(hipFree(d_b));
  HIP_CHECK(hipFree(d_a));
  return 0;
}
