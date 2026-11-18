// loop_unrolling_demo.hip
// Companion example for: topnotchnote.com/gpu/loop_unrolling.html
//
// Two cases where #pragma unroll changes what the compiler produces.
//   1. A small per-thread array indexed by the loop counter. Fully unrolled,
//      every index is a constant and the array lives in registers. Not
//      unrolled (#pragma unroll 1), the index is a run-time value and the
//      array goes to local memory, which is backed by device memory.
//   2. A loop whose trip count is only known at run time: each thread sums a
//      column of a matrix. Compared: unrolling disabled, the compiler's
//      default, #pragma unroll 8, and four separate accumulators.
//
// Inspect what the compiler did:
//   AMD:    hipcc -O2 --save-temps ... ; look for scratch_ / buffer_ loads in
//           the .s file and ".private_segment_fixed_size" (local memory bytes)
//   Nvidia: nvcc -Xptxas -v ...  ("bytes stack frame"), and
//           cuobjdump -sass ./main | grep -E 'LDL|STL'  (local loads/stores)
//
// Compile:  hipcc -O2 -o main loop_unrolling_demo.hip
// Nvidia:   hipcc also targets Nvidia (HIP_PLATFORM=nvidia), or swap the hip
//           prefix for cuda in the API calls and build with nvcc.

#include <hip/hip_runtime.h>

#include <algorithm>
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

// ---- case 1: a per-thread array --------------------------------------------
constexpr int K = 16;

// Each thread keeps K running values and updates them for `steps` rounds.
template <bool kUnroll>
__global__ void array_kernel(const float* in, float* out, int n, int steps) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  float r[K];
  if (kUnroll) {
#pragma unroll
    for (int k = 0; k < K; ++k) r[k] = in[i] + k;
    for (int s = 0; s < steps; ++s) {
#pragma unroll
      for (int k = 0; k < K; ++k) r[k] = r[k] * 0.99f + r[(k + 1) % K] * 0.01f;
    }
    float sum = 0;
#pragma unroll
    for (int k = 0; k < K; ++k) sum += r[k];
    out[i] = sum;
  } else {
#pragma unroll 1
    for (int k = 0; k < K; ++k) r[k] = in[i] + k;
    for (int s = 0; s < steps; ++s) {
#pragma unroll 1
      for (int k = 0; k < K; ++k) r[k] = r[k] * 0.99f + r[(k + 1) % K] * 0.01f;
    }
    float sum = 0;
#pragma unroll 1
    for (int k = 0; k < K; ++k) sum += r[k];
    out[i] = sum;
  }
}

// ---- case 2: a run-time trip count -----------------------------------------
// Thread j sums column j of a rows x cols row-major matrix. Neighbouring
// threads read neighbouring addresses, so every variant is coalesced.
__global__ void colsum_nounroll(const float* m, float* out, int rows, int cols) {
  int j = blockIdx.x * blockDim.x + threadIdx.x;
  if (j >= cols) return;
  float s = 0;
#pragma unroll 1
  for (int r = 0; r < rows; ++r) s += m[(size_t)r * cols + j];
  out[j] = s;
}

__global__ void colsum_default(const float* m, float* out, int rows, int cols) {
  int j = blockIdx.x * blockDim.x + threadIdx.x;
  if (j >= cols) return;
  float s = 0;
  for (int r = 0; r < rows; ++r) s += m[(size_t)r * cols + j];
  out[j] = s;
}

__global__ void colsum_unroll8(const float* m, float* out, int rows, int cols) {
  int j = blockIdx.x * blockDim.x + threadIdx.x;
  if (j >= cols) return;
  float s = 0;
#pragma unroll 8
  for (int r = 0; r < rows; ++r) s += m[(size_t)r * cols + j];
  out[j] = s;
}

// Four accumulators: four independent add chains instead of one. This also
// changes the order of the additions, so the float result can differ in the
// last bits.
__global__ void colsum_4acc(const float* m, float* out, int rows, int cols) {
  int j = blockIdx.x * blockDim.x + threadIdx.x;
  if (j >= cols) return;
  float s0 = 0, s1 = 0, s2 = 0, s3 = 0;
  int r = 0;
  for (; r + 3 < rows; r += 4) {
    s0 += m[(size_t)(r + 0) * cols + j];
    s1 += m[(size_t)(r + 1) * cols + j];
    s2 += m[(size_t)(r + 2) * cols + j];
    s3 += m[(size_t)(r + 3) * cols + j];
  }
  for (; r < rows; ++r) s0 += m[(size_t)r * cols + j];
  out[j] = (s0 + s1) + (s2 + s3);
}

template <typename F>
static float median_ms(F launch, int reps = 9) {
  hipEvent_t a, b;
  HIP_CHECK(hipEventCreate(&a));
  HIP_CHECK(hipEventCreate(&b));
  launch();
  HIP_CHECK(hipGetLastError());
  std::vector<float> t;
  for (int i = 0; i < reps; ++i) {
    HIP_CHECK(hipEventRecord(a));
    launch();
    HIP_CHECK(hipEventRecord(b));
    HIP_CHECK(hipEventSynchronize(b));
    float ms;
    HIP_CHECK(hipEventElapsedTime(&ms, a, b));
    t.push_back(ms);
  }
  HIP_CHECK(hipEventDestroy(a));
  HIP_CHECK(hipEventDestroy(b));
  std::sort(t.begin(), t.end());
  return t[reps / 2];
}

int main(int argc, char** argv) {
  // Case 1
  const int n = 1 << 20, steps = 64;
  float *in, *out;
  HIP_CHECK(hipMalloc(&in, n * sizeof(float)));
  HIP_CHECK(hipMalloc(&out, n * sizeof(float)));
  std::vector<float> h(n);
  for (int i = 0; i < n; ++i) h[i] = (float)(i % 100) * 0.01f;
  HIP_CHECK(hipMemcpy(in, h.data(), n * sizeof(float), hipMemcpyHostToDevice));
  std::vector<float> o1(n), o2(n);
  float t_unrolled = median_ms([&] { array_kernel<true><<<n / 256, 256>>>(in, out, n, steps); });
  HIP_CHECK(hipMemcpy(o1.data(), out, n * sizeof(float), hipMemcpyDeviceToHost));
  float t_rolled = median_ms([&] { array_kernel<false><<<n / 256, 256>>>(in, out, n, steps); });
  HIP_CHECK(hipMemcpy(o2.data(), out, n * sizeof(float), hipMemcpyDeviceToHost));
  double d = 0;
  for (int i = 0; i < n; ++i) d = std::max(d, (double)std::fabs(o1[i] - o2[i]));
  std::printf("case 1: per-thread array of %d floats, %d update rounds, %d threads\n", K, steps, n);
  std::printf("  fully unrolled (array in registers)   %8.3f ms\n", t_unrolled);
  std::printf("  #pragma unroll 1 (array in local mem) %8.3f ms   %.1fx slower, max |diff| %.1e\n",
              t_rolled, t_rolled / t_unrolled, d);

  // Case 2
  int rows = argc > 1 ? std::atoi(argv[1]) : 16384;  // run-time trip count
  const int cols = argc > 2 ? std::atoi(argv[2]) : 512;   // few threads: latency-bound
  float *m, *cs;
  HIP_CHECK(hipMalloc(&m, (size_t)rows * cols * sizeof(float)));
  HIP_CHECK(hipMalloc(&cs, cols * sizeof(float)));
  HIP_CHECK(hipMemset(m, 0, (size_t)rows * cols * sizeof(float)));
  const double gb = (double)rows * cols * sizeof(float) / 1e9;
  std::printf("\ncase 2: column sums of a %d x %d matrix (%d threads, trip count %d)\n",
              rows, cols, cols, rows);
  auto row = [&](const char* name, float ms) {
    std::printf("  %-28s %8.3f ms  %6.1f GB/s\n", name, ms, gb / (ms / 1e3));
  };
  row("#pragma unroll 1", median_ms([&] { colsum_nounroll<<<cols / 256, 256>>>(m, cs, rows, cols); }));
  row("compiler default", median_ms([&] { colsum_default<<<cols / 256, 256>>>(m, cs, rows, cols); }));
  row("#pragma unroll 8", median_ms([&] { colsum_unroll8<<<cols / 256, 256>>>(m, cs, rows, cols); }));
  row("four accumulators", median_ms([&] { colsum_4acc<<<cols / 256, 256>>>(m, cs, rows, cols); }));

  HIP_CHECK(hipFree(in));
  HIP_CHECK(hipFree(out));
  HIP_CHECK(hipFree(m));
  HIP_CHECK(hipFree(cs));
  return 0;
}
