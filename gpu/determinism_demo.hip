// determinism_demo.hip
// Companion example for: topnotchnote.com/gpu/numerical_determinism.html
//
// Sums the same 16M floats many times, three ways:
//   1. atomic: each block reduces its slice, then atomicAdd()s its partial sum
//      into one total. Blocks finish in a different order on each run, so the
//      partial sums are added in a different order.
//   2. two-pass: each block writes its partial sum to its own slot; a second
//      kernel adds the slots in a fixed order. Same order every run.
//   3. two-pass with a different block count: still repeatable, but a
//      different grouping of the additions, so possibly a different answer.
// For each, the program prints how many distinct results it saw in 20 runs
// and how far apart they were, next to a double-precision reference.
//
// Compile:  hipcc -O2 -o main determinism_demo.hip
// Nvidia:   hipcc also targets Nvidia (HIP_PLATFORM=nvidia), or swap the hip
//           prefix for cuda in the API calls and build with nvcc.

#include <hip/hip_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <set>
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

constexpr int kThreads = 256;

// Block-level tree sum in a fixed order; returns the block's sum in thread 0.
__device__ float block_sum(const float* x, size_t n) {
  __shared__ float s[kThreads];
  float acc = 0;
  for (size_t i = blockIdx.x * (size_t)kThreads + threadIdx.x; i < n;
       i += (size_t)gridDim.x * kThreads)
    acc += x[i];
  s[threadIdx.x] = acc;
  __syncthreads();
  for (int k = kThreads / 2; k > 0; k >>= 1) {
    if (threadIdx.x < k) s[threadIdx.x] += s[threadIdx.x + k];
    __syncthreads();
  }
  return s[0];
}

__global__ void sum_atomic(const float* x, size_t n, float* total) {
  float b = block_sum(x, n);
  if (threadIdx.x == 0) atomicAdd(total, b);   // order of arrival varies
}

__global__ void sum_partials(const float* x, size_t n, float* partial) {
  float b = block_sum(x, n);
  if (threadIdx.x == 0) partial[blockIdx.x] = b;
}

__global__ void sum_fixed_order(const float* partial, int m, float* total) {
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    float t = 0;
    for (int i = 0; i < m; ++i) t += partial[i];  // always the same order
    *total = t;
  }
}

static void report(const char* name, const std::vector<float>& r, double ref) {
  std::set<float> distinct(r.begin(), r.end());
  float lo = *distinct.begin(), hi = *distinct.rbegin();
  std::printf("%-34s %2zu distinct of %zu   min %.6f  max %.6f  spread %.2e  |max-ref| %.2e\n",
              name, distinct.size(), r.size(), lo, hi, (double)hi - lo,
              std::fmax(std::fabs(hi - ref), std::fabs(lo - ref)));
}

int main() {
  const size_t n = (size_t)16 << 20;
  std::vector<float> h(n);
  unsigned s = 12345;
  double ref = 0;
  for (size_t i = 0; i < n; ++i) {               // values with mixed magnitudes
    s = s * 1664525u + 1013904223u;
    float u = (s >> 8) * (1.0f / 16777216.0f);
    h[i] = (u - 0.5f) * ((i % 97 == 0) ? 1000.0f : 1.0f);
    ref += h[i];
  }
  float *x, *total, *partial;
  HIP_CHECK(hipMalloc(&x, n * sizeof(float)));
  HIP_CHECK(hipMalloc(&total, sizeof(float)));
  HIP_CHECK(hipMalloc(&partial, 4096 * sizeof(float)));
  HIP_CHECK(hipMemcpy(x, h.data(), n * sizeof(float), hipMemcpyHostToDevice));

  const int runs = 20;
  std::vector<float> a, b, c;
  for (int r = 0; r < runs; ++r) {
    float t;
    HIP_CHECK(hipMemset(total, 0, sizeof(float)));
    sum_atomic<<<1024, kThreads>>>(x, n, total);
    HIP_CHECK(hipMemcpy(&t, total, sizeof(float), hipMemcpyDeviceToHost));
    a.push_back(t);

    sum_partials<<<1024, kThreads>>>(x, n, partial);
    sum_fixed_order<<<1, 32>>>(partial, 1024, total);
    HIP_CHECK(hipMemcpy(&t, total, sizeof(float), hipMemcpyDeviceToHost));
    b.push_back(t);

    sum_partials<<<4096, kThreads>>>(x, n, partial);
    sum_fixed_order<<<1, 32>>>(partial, 4096, total);
    HIP_CHECK(hipMemcpy(&t, total, sizeof(float), hipMemcpyDeviceToHost));
    c.push_back(t);
  }
  std::printf("sum of %zu floats, double-precision reference %.6f\n", n, ref);
  report("1. atomicAdd of block sums", a, ref);
  report("2. fixed order, 1024 blocks", b, ref);
  report("3. fixed order, 4096 blocks", c, ref);
  HIP_CHECK(hipFree(x));
  HIP_CHECK(hipFree(total));
  HIP_CHECK(hipFree(partial));
  return 0;
}
