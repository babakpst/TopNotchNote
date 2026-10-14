// fft_demo.hip
// Companion example for: topnotchnote.com/gpu/fft_libraries.html
//
// GPU Fourier transforms with hipFFT (which forwards to rocFFT on AMD and
// to cuFFT on Nvidia). Four experiments, single-precision complex data:
//   1. a hand-written O(n^2) DFT kernel against one hipfftExecC2C call
//   2. plan creation against execution: why you build the plan once
//   3. transform size: a power of two, a smooth size, and a prime
//   4. many small transforms: one batched plan against a loop of calls
// It also checks that forward + inverse returns the input, once you divide
// by n (neither library normalizes).
//
// Compile:  hipcc -O2 -o main fft_demo.hip -lhipfft
// Nvidia:   hipcc also targets Nvidia (HIP_PLATFORM=nvidia), or swap the
//           hipfft prefix for cufft and build with nvcc -O2 ... -lcufft

#include <hip/hip_runtime.h>
#include <hipfft/hipfft.h>

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

#define HIPFFT_CHECK(call)                                                \
  do {                                                                    \
    hipfftResult r_ = (call);                                             \
    if (r_ != HIPFFT_SUCCESS) {                                           \
      std::fprintf(stderr, "%s:%d: hipFFT error %d\n", __FILE__,          \
                   __LINE__, (int)r_);                                    \
      std::exit(1);                                                       \
    }                                                                     \
  } while (0)

// The definition, written out: each thread computes one output frequency
// by summing all n inputs times a twiddle factor. n^2 complex multiplies.
__global__ void naiveDft(const hipfftComplex *in, hipfftComplex *out, int n) {
  int k = blockIdx.x * blockDim.x + threadIdx.x;
  if (k >= n) return;
  float re = 0.0f, im = 0.0f;
  for (int j = 0; j < n; ++j) {
    // Reduce k*j mod n first so the angle stays small and accurate.
    int kj = (int)(((long long)k * j) % n);
    float s, c;
    sincospif(-2.0f * kj / n, &s, &c);
    re += in[j].x * c - in[j].y * s;
    im += in[j].x * s + in[j].y * c;
  }
  out[k].x = re;
  out[k].y = im;
}

static void fill(std::vector<hipfftComplex> &v) {
  for (size_t i = 0; i < v.size(); ++i) {
    v[i].x = (float)((i * 7) % 13) / 13.0f - 0.5f;
    v[i].y = (float)((i * 11) % 17) / 17.0f - 0.5f;
  }
}

static double msSince(std::chrono::steady_clock::time_point t0) {
  return std::chrono::duration<double, std::milli>(
             std::chrono::steady_clock::now() - t0).count();
}

// Average time of `reps` executions of an existing plan, after a warm-up.
static float timeExec(hipfftHandle plan, hipfftComplex *in, hipfftComplex *out,
                      int reps, hipEvent_t start, hipEvent_t stop) {
  HIPFFT_CHECK(hipfftExecC2C(plan, in, out, HIPFFT_FORWARD));
  HIP_CHECK(hipEventRecord(start, 0));
  for (int i = 0; i < reps; ++i)
    HIPFFT_CHECK(hipfftExecC2C(plan, in, out, HIPFFT_FORWARD));
  HIP_CHECK(hipEventRecord(stop, 0));
  HIP_CHECK(hipEventSynchronize(stop));
  float ms;
  HIP_CHECK(hipEventElapsedTime(&ms, start, stop));
  return ms / reps;
}

int main() {
  hipDeviceProp_t prop;
  HIP_CHECK(hipGetDeviceProperties(&prop, 0));
  std::printf("Device: %s\n\n", prop.name);
  hipEvent_t start, stop;
  HIP_CHECK(hipEventCreate(&start));
  HIP_CHECK(hipEventCreate(&stop));

  const int maxN = 1 << 22;
  hipfftComplex *in, *out, *back;
  HIP_CHECK(hipMalloc(&in, maxN * sizeof(hipfftComplex)));
  HIP_CHECK(hipMalloc(&out, maxN * sizeof(hipfftComplex)));
  HIP_CHECK(hipMalloc(&back, maxN * sizeof(hipfftComplex)));
  std::vector<hipfftComplex> h(maxN);
  fill(h);
  HIP_CHECK(hipMemcpy(in, h.data(), maxN * sizeof(hipfftComplex),
                      hipMemcpyHostToDevice));

  // 1. The definition against the library.
  std::printf("1. O(n^2) DFT kernel vs hipFFT\n");
  std::printf("%9s %12s %12s %9s %11s\n", "n", "naive ms", "hipFFT ms",
              "speedup", "max |diff|");
  for (int n = 1024; n <= 65536; n *= 4) {
    naiveDft<<<(n + 255) / 256, 256>>>(in, back, n);  // warm-up
    HIP_CHECK(hipEventRecord(start, 0));
    naiveDft<<<(n + 255) / 256, 256>>>(in, back, n);
    HIP_CHECK(hipEventRecord(stop, 0));
    HIP_CHECK(hipEventSynchronize(stop));
    HIP_CHECK(hipGetLastError());
    float naiveMs;
    HIP_CHECK(hipEventElapsedTime(&naiveMs, start, stop));

    hipfftHandle plan;
    HIPFFT_CHECK(hipfftPlan1d(&plan, n, HIPFFT_C2C, 1));
    float fftMs = timeExec(plan, in, out, 20, start, stop);
    HIPFFT_CHECK(hipfftDestroy(plan));

    std::vector<hipfftComplex> a(n), b(n);
    HIP_CHECK(hipMemcpy(a.data(), back, n * sizeof(hipfftComplex), hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(b.data(), out, n * sizeof(hipfftComplex), hipMemcpyDeviceToHost));
    double d = 0.0;
    for (int i = 0; i < n; ++i)
      d = std::fmax(d, std::fmax(std::fabs(a[i].x - b[i].x), std::fabs(a[i].y - b[i].y)));
    std::printf("%9d %12.3f %12.4f %8.0fx %11.2e\n", n, naiveMs, fftMs,
                naiveMs / fftMs, d);
  }

  // 2. Plan once, execute many times.
  {
    const int n = 1 << 22;
    std::printf("\n2. Plan vs execute, n = %d\n", n);
    auto t0 = std::chrono::steady_clock::now();
    hipfftHandle plan;
    HIPFFT_CHECK(hipfftPlan1d(&plan, n, HIPFFT_C2C, 1));
    HIP_CHECK(hipDeviceSynchronize());
    double planMs = msSince(t0);
    float execMs = timeExec(plan, in, out, 20, start, stop);
    // Bytes touched: one read and one write of n complex values per pass.
    double gbs = 2.0 * n * sizeof(hipfftComplex) / (execMs * 1e6);
    std::printf("   plan %.2f ms, execute %.3f ms (%.0f GB/s effective)\n",
                planMs, execMs, gbs);

    // Round trip: forward then inverse gives n * input, not input.
    HIPFFT_CHECK(hipfftExecC2C(plan, in, out, HIPFFT_FORWARD));
    HIPFFT_CHECK(hipfftExecC2C(plan, out, back, HIPFFT_BACKWARD));
    std::vector<hipfftComplex> r(n);
    HIP_CHECK(hipMemcpy(r.data(), back, n * sizeof(hipfftComplex), hipMemcpyDeviceToHost));
    double raw = 0.0, scaled = 0.0;
    for (int i = 0; i < n; ++i) {
      raw = std::fmax(raw, std::fabs(r[i].x - h[i].x));
      scaled = std::fmax(scaled, std::fabs(r[i].x / n - h[i].x));
    }
    std::printf("   inverse(forward(x)) vs x: max |diff| %.3g raw, %.2e after /n\n",
                raw, scaled);
    HIPFFT_CHECK(hipfftDestroy(plan));
  }

  // 3. Transform size matters. 2^20, a size made of small primes, and the
  // largest prime below 2^20.
  std::printf("\n3. Transform size\n");
  const int sizes[3] = {1 << 20, 1000000, 1048573};
  const char *what[3] = {"2^20", "2^6 * 5^6", "prime"};
  for (int i = 0; i < 3; ++i) {
    hipfftHandle plan;
    HIPFFT_CHECK(hipfftPlan1d(&plan, sizes[i], HIPFFT_C2C, 1));
    float ms = timeExec(plan, in, out, 20, start, stop);
    std::printf("   n = %8d (%-9s) %8.3f ms  %6.2f ns per point\n", sizes[i],
                what[i], ms, ms * 1e6 / sizes[i]);
    HIPFFT_CHECK(hipfftDestroy(plan));
  }

  // 4. Many small transforms: 4096 signals of 1024 points.
  {
    const int n = 1024, batch = 4096;
    std::printf("\n4. %d transforms of %d points\n", batch, n);
    hipfftHandle one, many;
    HIPFFT_CHECK(hipfftPlan1d(&one, n, HIPFFT_C2C, 1));
    HIPFFT_CHECK(hipfftPlan1d(&many, n, HIPFFT_C2C, batch));
    HIPFFT_CHECK(hipfftExecC2C(one, in, out, HIPFFT_FORWARD));  // warm-up
    HIP_CHECK(hipEventRecord(start, 0));
    for (int b = 0; b < batch; ++b)
      HIPFFT_CHECK(hipfftExecC2C(one, in + (size_t)b * n, out + (size_t)b * n,
                                 HIPFFT_FORWARD));
    HIP_CHECK(hipEventRecord(stop, 0));
    HIP_CHECK(hipEventSynchronize(stop));
    float loopMs;
    HIP_CHECK(hipEventElapsedTime(&loopMs, start, stop));
    float batchMs = timeExec(many, in, out, 20, start, stop);
    std::printf("   loop of %d calls %8.2f ms\n   one batched call  %8.3f ms  (%.0fx)\n",
                batch, loopMs, batchMs, loopMs / batchMs);
    HIPFFT_CHECK(hipfftDestroy(one));
    HIPFFT_CHECK(hipfftDestroy(many));
  }

  HIP_CHECK(hipFree(in));
  HIP_CHECK(hipFree(out));
  HIP_CHECK(hipFree(back));
  HIP_CHECK(hipEventDestroy(start));
  HIP_CHECK(hipEventDestroy(stop));
  return 0;
}
