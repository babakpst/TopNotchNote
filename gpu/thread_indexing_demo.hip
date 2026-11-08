// thread_indexing_demo.hip
// Companion example for: topnotchnote.com/gpu/thread_indexing.html
//
// Two questions every first kernel raises, measured on your GPU:
//   1. Block size: one thread per element, block sizes 32 to 1024.
//   2. Grid-stride loops: a fixed, small grid in which each thread walks the
//      array in steps of (gridDim * blockDim), against one thread per element.
// The kernel is a SAXPY (y = a*x + y) on 64M floats, so it is memory-bound and
// the numbers are effective bandwidth. It also checks that every element was
// touched exactly once, which is what the bounds guard and the stride are for.
//
// Compile:  hipcc -O2 -o main thread_indexing_demo.hip
// Nvidia:   hipcc also targets Nvidia (HIP_PLATFORM=nvidia), or swap the hip
//           prefix for cuda in the API calls and build with nvcc.

#include <hip/hip_runtime.h>

#include <algorithm>
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

__global__ void saxpy_one(int n, float a, const float* x, float* y) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;  // global index
  if (i < n) y[i] = a * x[i] + y[i];              // guard: last block overhangs
}

__global__ void saxpy_stride(int n, float a, const float* x, float* y) {
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
       i += gridDim.x * blockDim.x)               // grid-stride loop
    y[i] = a * x[i] + y[i];
}

static float run(bool stride, int grid, int block, int n, float* x, float* y) {
  hipEvent_t s, e;
  HIP_CHECK(hipEventCreate(&s));
  HIP_CHECK(hipEventCreate(&e));
  std::vector<float> t;
  for (int r = 0; r < 7; ++r) {
    HIP_CHECK(hipEventRecord(s));
    if (stride)
      hipLaunchKernelGGL(saxpy_stride, dim3(grid), dim3(block), 0, 0, n, 1.0f, x, y);
    else
      hipLaunchKernelGGL(saxpy_one, dim3(grid), dim3(block), 0, 0, n, 1.0f, x, y);
    HIP_CHECK(hipEventRecord(e));
    HIP_CHECK(hipEventSynchronize(e));
    float ms;
    HIP_CHECK(hipEventElapsedTime(&ms, s, e));
    if (r) t.push_back(ms);
  }
  std::sort(t.begin(), t.end());
  return t[t.size() / 2];
}

int main() {
  const int n = 64 << 20;
  const double bytes = 3.0 * n * sizeof(float);  // read x, read y, write y
  float *x, *y;
  HIP_CHECK(hipMalloc(&x, n * sizeof(float)));
  HIP_CHECK(hipMalloc(&y, n * sizeof(float)));
  HIP_CHECK(hipMemset(x, 0, n * sizeof(float)));
  HIP_CHECK(hipMemset(y, 0, n * sizeof(float)));
  int sms = 0;
  HIP_CHECK(hipDeviceGetAttribute(&sms, hipDeviceAttributeMultiprocessorCount, 0));

  std::printf("one thread per element, %d elements:\n", n);
  for (int block = 32; block <= 1024; block *= 2) {
    int grid = (n + block - 1) / block;
    float ms = run(false, grid, block, n, x, y);
    std::printf("  block %4d  grid %8d  %7.2f ms  %6.1f GB/s\n", block, grid, ms,
                bytes / (ms * 1e-3) / 1e9);
  }
  std::printf("\ngrid-stride loop, block 256, grid = SMs/CUs (%d) x k:\n", sms);
  for (int k : {1, 2, 4, 8, 16, 32}) {
    int grid = sms * k;
    float ms = run(true, grid, 256, n, x, y);
    std::printf("  k %3d  grid %6d  each thread does ~%5d elements  %7.2f ms  %6.1f GB/s\n",
                k, grid, (n + grid * 256 - 1) / (grid * 256), ms, bytes / (ms * 1e-3) / 1e9);
  }

  // Coverage check: with a = 1 and x = 1 everywhere, one pass adds exactly 1.
  std::vector<float> ones(n, 1.0f);
  HIP_CHECK(hipMemcpy(x, ones.data(), n * sizeof(float), hipMemcpyHostToDevice));
  HIP_CHECK(hipMemset(y, 0, n * sizeof(float)));
  hipLaunchKernelGGL(saxpy_stride, dim3(sms * 4), dim3(256), 0, 0, n, 1.0f, x, y);
  HIP_CHECK(hipMemcpy(ones.data(), y, n * sizeof(float), hipMemcpyDeviceToHost));
  long bad = 0;
  for (int i = 0; i < n; ++i) bad += ones[i] != 1.0f;
  std::printf("\ncoverage check (grid-stride, small grid): %s\n",
              bad ? "elements missed or repeated" : "every element exactly once");
  return bad != 0;
}
