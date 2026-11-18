// amdahl_demo.hip
// Companion example for: topnotchnote.com/gpu/amdahls_law.html
//
// A small end-to-end pipeline, timed stage by stage:
//   1. prepare the input on the CPU
//   2. copy it to the GPU
//   3. run a kernel
//   4. copy the result back and check it on the CPU
// The kernel is then replaced by a faster version that computes the same
// result (coalesced reads instead of strided ones). The program prints the
// measured end-to-end speedup next to the one Amdahl's law predicts from the
// kernel's share of the original run time and the kernel's own speedup.
//
// Compile:  hipcc -O2 -o main amdahl_demo.hip
// Nvidia:   hipcc also targets Nvidia (HIP_PLATFORM=nvidia), or swap the hip
//           prefix for cuda in the API calls and build with nvcc.

#include <hip/hip_runtime.h>

#include <algorithm>
#include <chrono>
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

using Clock = std::chrono::steady_clock;
static double ms_since(Clock::time_point t0) {
  return std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
}

constexpr int kRows = 4096, kCols = 4096;      // 16M floats, 64 MB
constexpr int kWork = 32;                       // FMAs per element

__device__ float work(float x) {
  float y = x;
  for (int k = 0; k < kWork; ++k) y = y * 0.999f + 0.001f;
  return y;
}

// Slow: thread t handles row t. Neighbouring threads read addresses kCols
// floats apart, so a warp's reads are not coalesced.
__global__ void rows_per_thread(const float* in, float* out) {
  int r = blockIdx.x * blockDim.x + threadIdx.x;
  if (r >= kRows) return;
  for (int c = 0; c < kCols; ++c) out[r * kCols + c] = work(in[r * kCols + c]);
}

// Fast: one thread per element, neighbouring threads read neighbouring floats.
__global__ void element_per_thread(const float* in, float* out) {
  size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
  if (i < (size_t)kRows * kCols) out[i] = work(in[i]);
}

struct Stages { double prep, h2d, kernel, d2h, check; double total() const {
  return prep + h2d + kernel + d2h + check; } };

static Stages run(bool fast, std::vector<float>& h_in, std::vector<float>& h_out,
                  float* d_in, float* d_out) {
  const size_t n = (size_t)kRows * kCols;
  Stages s{};
  auto t = Clock::now();
  for (size_t i = 0; i < n; ++i) h_in[i] = std::sin(0.001f * (float)(i % 6283));
  s.prep = ms_since(t);

  t = Clock::now();
  HIP_CHECK(hipMemcpy(d_in, h_in.data(), n * sizeof(float), hipMemcpyHostToDevice));
  s.h2d = ms_since(t);

  t = Clock::now();
  if (fast) element_per_thread<<<(unsigned)((n + 255) / 256), 256>>>(d_in, d_out);
  else      rows_per_thread<<<(kRows + 127) / 128, 128>>>(d_in, d_out);
  HIP_CHECK(hipGetLastError());
  HIP_CHECK(hipDeviceSynchronize());
  s.kernel = ms_since(t);

  t = Clock::now();
  HIP_CHECK(hipMemcpy(h_out.data(), d_out, n * sizeof(float), hipMemcpyDeviceToHost));
  s.d2h = ms_since(t);

  t = Clock::now();
  double sum = 0;
  for (size_t i = 0; i < n; ++i) sum += h_out[i];
  s.check = ms_since(t);
  static double first_sum = 0;
  if (first_sum == 0) first_sum = sum;
  else if (std::fabs(sum - first_sum) > 1e-6 * std::fabs(first_sum)) {
    std::fprintf(stderr, "results differ: %f vs %f\n", sum, first_sum);
    std::exit(1);
  }
  return s;
}

static Stages median_of(int trials, bool fast, std::vector<float>& a,
                        std::vector<float>& b, float* d_in, float* d_out) {
  std::vector<Stages> v;
  for (int i = 0; i < trials; ++i) v.push_back(run(fast, a, b, d_in, d_out));
  std::sort(v.begin(), v.end(), [](const Stages& x, const Stages& y) {
    return x.total() < y.total(); });
  return v[trials / 2];
}

static void print(const char* name, const Stages& s) {
  std::printf("%-22s prep %7.1f  h2d %6.1f  kernel %7.1f  d2h %6.1f  check %6.1f"
              "  total %7.1f ms\n", name, s.prep, s.h2d, s.kernel, s.d2h, s.check,
              s.total());
}

int main() {
  const size_t n = (size_t)kRows * kCols;
  std::vector<float> h_in(n), h_out(n);
  float *d_in, *d_out;
  HIP_CHECK(hipMalloc(&d_in, n * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_out, n * sizeof(float)));

  run(false, h_in, h_out, d_in, d_out);          // warm-up, also sets the reference sum
  Stages slow = median_of(5, false, h_in, h_out, d_in, d_out);
  Stages fast = median_of(5, true, h_in, h_out, d_in, d_out);
  print("strided kernel", slow);
  print("coalesced kernel", fast);

  double P = slow.kernel / slow.total();         // kernel's share of the old run
  double s = slow.kernel / fast.kernel;           // kernel's own speedup
  double predicted = 1.0 / ((1.0 - P) + P / s);
  double measured = slow.total() / fast.total();
  std::printf("\nkernel share of the original run P = %.2f\n", P);
  std::printf("kernel speedup s                   = %.1fx\n", s);
  std::printf("Amdahl prediction 1/((1-P)+P/s)    = %.2fx\n", predicted);
  std::printf("measured end-to-end speedup        = %.2fx\n", measured);
  std::printf("ceiling if the kernel took zero time 1/(1-P) = %.2fx\n", 1.0 / (1.0 - P));

  HIP_CHECK(hipFree(d_in));
  HIP_CHECK(hipFree(d_out));
  return 0;
}
