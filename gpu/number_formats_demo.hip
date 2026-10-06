// number_formats_demo.hip
// Companion example for: topnotchnote.com/gpu/data_types.html
//
// Shows on the GPU what each number format does to a value:
//   1. Round-trips a handful of test values through FP16, BF16, the two OCP
//      FP8 formats, AMD's MI300 FP8 variant (FNUZ) and the FP6/FP4 element
//      types used by MX. One generic rounding function, parameterized by
//      exponent bits, mantissa bits and bias, handles every format, so you can
//      see that they are all the same idea with different bit budgets.
//      FP16 and BF16 are cross-checked against the native hardware types.
//   2. Quantizes one block of 32 values to MXFP4 exactly as the OCP MX spec
//      describes: one shared power-of-two scale (E8M0) for the block, then
//      each element rounded to FP4 (E2M1). Run it twice in your head: once
//      with the outlier and once without, and watch what one large value does
//      to its 31 neighbours.
// The rounding here saturates to the largest finite value, which is what ML
// conversion routines do. Plain IEEE FP16 overflows to infinity instead,
// which the native column shows.
// Compile:  hipcc -O2 -o main number_formats_demo.hip
// Run:      ./main
// Nvidia:   hipcc also targets Nvidia (HIP_PLATFORM=nvidia), or swap the hip
//           prefix for cuda (hip_fp16.h -> cuda_fp16.h, hip_bf16.h ->
//           cuda_bf16.h, __hip_bfloat16 -> __nv_bfloat16) and build with nvcc.

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <hip/hip_bf16.h>

#include <cmath>
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

struct Format {
  const char *name;
  int manBits;   // stored mantissa (fraction) bits
  int bias;      // exponent bias
  float maxVal;  // largest finite value, from the format's spec
};

// Every format below is (sign, exponent, mantissa) with a bias. The exponent
// width only shows up through bias and maxVal, so it is not needed here.
constexpr int kFormats = 9;
__constant__ Format dFormats[kFormats] = {
    {"FP16",      10, 15,  65504.0f},
    {"BF16",       7, 127, 3.38953139e38f},
    {"E4M3",       3, 7,   448.0f},     // OCP FP8, H100 and MI350
    {"E5M2",       2, 15,  57344.0f},   // OCP FP8
    {"E4M3FNUZ",   3, 8,   240.0f},     // MI300's FP8: one more bias, half the range
    {"E5M2FNUZ",   2, 16,  57344.0f},
    {"FP6 E2M3",   3, 1,   7.5f},       // MX element types
    {"FP6 E3M2",   2, 3,   28.0f},
    {"FP4 E2M1",   1, 1,   6.0f},
};

// Round x to the nearest value the format can hold (ties to even), saturating
// at maxVal. Below the smallest normal, the step size stops shrinking: that is
// the subnormal range, and anything under half a step becomes zero.
__device__ float roundToFormat(float x, const Format &f) {
  if (x == 0.0f || isnan(x)) return x;
  float a = fabsf(x);
  if (a >= f.maxVal) return copysignf(f.maxVal, x);
  int e;
  frexpf(a, &e);              // a = m * 2^e with m in [0.5, 1)
  int exponent = e - 1;       // so a is in [2^exponent, 2^(exponent+1))
  int minExponent = 1 - f.bias;
  if (exponent < minExponent) exponent = minExponent;     // subnormal range
  float step = ldexpf(1.0f, exponent - f.manBits);        // spacing between neighbours here
  float q = rintf(a / step) * step;                       // rintf rounds ties to even
  return copysignf(fminf(q, f.maxVal), x);
}

__global__ void roundTripKernel(const float *in, float *out, float *nativeHalf,
                                float *nativeBf16, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  for (int k = 0; k < kFormats; ++k) out[i * kFormats + k] = roundToFormat(in[i], dFormats[k]);
  nativeHalf[i] = __half2float(__float2half(in[i]));
  nativeBf16[i] = __bfloat162float(__float2bfloat16(in[i]));
}

// One MX block of 32 values. The spec's recipe: shared exponent =
// floor(log2(max |v|)) - (largest exponent the element type can hold), so the
// biggest value in the block lands at the top of FP4's range. FP4 E2M1 tops
// out at 1.5 * 2^2 = 6, so its largest exponent is 2.
constexpr int kBlock = 32;
__global__ void mxfp4Kernel(const float *in, float *out, int *sharedExp) {
  __shared__ float vals[kBlock];
  __shared__ float scale;
  int t = threadIdx.x;
  vals[t] = in[t];
  __syncthreads();
  if (t == 0) {
    float amax = 0.0f;
    for (int i = 0; i < kBlock; ++i) amax = fmaxf(amax, fabsf(vals[i]));
    int e = (amax > 0.0f) ? (int)floorf(log2f(amax)) - 2 : 0;
    *sharedExp = e;
    scale = ldexpf(1.0f, e);
  }
  __syncthreads();
  // dFormats[8] is FP4 E2M1. Store the element, then dequantize to compare.
  out[t] = roundToFormat(vals[t] / scale, dFormats[8]) * scale;
}

int main() {
  const float tests[] = {1.0f, 0.1f, 3.14159265f, 257.0f, 2049.0f,
                         500.0f, 70000.0f, 1e-6f, 1e-8f};
  const int n = sizeof(tests) / sizeof(tests[0]);
  const char *names[kFormats] = {"FP16", "BF16", "E4M3", "E5M2", "E4M3FNUZ",
                                 "E5M2FNUZ", "FP6 E2M3", "FP6 E3M2", "FP4 E2M1"};

  float *dIn, *dOut, *dHalf, *dBf16;
  HIP_CHECK(hipMalloc(&dIn, n * sizeof(float)));
  HIP_CHECK(hipMalloc(&dOut, n * kFormats * sizeof(float)));
  HIP_CHECK(hipMalloc(&dHalf, n * sizeof(float)));
  HIP_CHECK(hipMalloc(&dBf16, n * sizeof(float)));
  HIP_CHECK(hipMemcpy(dIn, tests, n * sizeof(float), hipMemcpyHostToDevice));
  roundTripKernel<<<1, 64>>>(dIn, dOut, dHalf, dBf16, n);
  HIP_CHECK(hipGetLastError());

  float out[n * kFormats], half[n], bf16[n];
  HIP_CHECK(hipMemcpy(out, dOut, sizeof(out), hipMemcpyDeviceToHost));
  HIP_CHECK(hipMemcpy(half, dHalf, sizeof(half), hipMemcpyDeviceToHost));
  HIP_CHECK(hipMemcpy(bf16, dBf16, sizeof(bf16), hipMemcpyDeviceToHost));

  std::printf("1. What each format stores for a value (saturating rounding)\n\n");
  std::printf("%-12s", "input");
  for (int k = 0; k < kFormats; ++k) std::printf("%-12s", names[k]);
  std::printf("%-12s%-12s\n", "native FP16", "native BF16");
  for (int i = 0; i < n; ++i) {
    std::printf("%-12g", tests[i]);
    for (int k = 0; k < kFormats; ++k) std::printf("%-12g", out[i * kFormats + k]);
    std::printf("%-12g%-12g\n", half[i], bf16[i]);
  }

  // A block of small activations: the first run has no outlier, the second
  // replaces one value with 40.
  float block[kBlock];
  for (int i = 0; i < kBlock; ++i) block[i] = 0.1f + 0.05f * (i % 8) - 0.2f * (i % 3);

  float *dBlockIn, *dBlockOut;
  int *dExp;
  HIP_CHECK(hipMalloc(&dBlockIn, sizeof(block)));
  HIP_CHECK(hipMalloc(&dBlockOut, sizeof(block)));
  HIP_CHECK(hipMalloc(&dExp, sizeof(int)));

  std::printf("\n2. One MXFP4 block: 32 FP4 values sharing one power-of-two scale\n");
  for (int run = 0; run < 2; ++run) {
    if (run == 1) block[5] = 40.0f;
    HIP_CHECK(hipMemcpy(dBlockIn, block, sizeof(block), hipMemcpyHostToDevice));
    mxfp4Kernel<<<1, kBlock>>>(dBlockIn, dBlockOut, dExp);
    HIP_CHECK(hipGetLastError());
    float deq[kBlock];
    int e;
    HIP_CHECK(hipMemcpy(deq, dBlockOut, sizeof(deq), hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(&e, dExp, sizeof(int), hipMemcpyDeviceToHost));

    int zeros = 0;
    double err = 0.0;
    for (int i = 0; i < kBlock; ++i) {
      if (i == 5 && run == 1) continue;  // judge the outlier's neighbours, not the outlier
      if (deq[i] == 0.0f && block[i] != 0.0f) ++zeros;
      err += std::fabs(deq[i] - block[i]);
    }
    int others = (run == 1) ? kBlock - 1 : kBlock;
    std::printf("\n%s: shared scale 2^%d = %g\n", run ? "With one outlier (40)" : "No outlier", e,
                std::ldexp(1.0, e));
    std::printf("  first 8 in : ");
    for (int i = 0; i < 8; ++i) std::printf("%7.3f", block[i]);
    std::printf("\n  first 8 out: ");
    for (int i = 0; i < 8; ++i) std::printf("%7.3f", deq[i] + 0.0f);  // + 0.0f prints -0 as 0
    std::printf("\n  %s %d values: %d flushed to zero, mean abs error %.4f\n",
                run ? "the other" : "all", others, zeros, err / others);
  }
  std::printf("\nStorage: 32 x 4 bits + one 8-bit scale = 136 bits, 4.25 bits per value.\n");

  HIP_CHECK(hipFree(dIn));
  HIP_CHECK(hipFree(dOut));
  HIP_CHECK(hipFree(dHalf));
  HIP_CHECK(hipFree(dBf16));
  HIP_CHECK(hipFree(dBlockIn));
  HIP_CHECK(hipFree(dBlockOut));
  HIP_CHECK(hipFree(dExp));
  return 0;
}
