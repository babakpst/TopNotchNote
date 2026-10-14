// dl_primitives_demo.hip
// Companion example for: topnotchnote.com/gpu/cudnn_miopen.html
//
// One 3x3 convolution layer (batch 16, 64 -> 64 channels, 56x56, FP32,
// NCHW), computed two ways:
//   1. a hand-written direct convolution, one thread per output value
//   2. MIOpen, AMD's deep learning primitives library (cuDNN's twin)
// MIOpen is used the way frameworks use it: describe the tensors, ask for
// the workspace size, run Find once to benchmark the candidate algorithms
// for this exact shape, then call the fastest one many times. The program
// prints what Find returned, what Find cost, and the steady-state time of
// both versions, plus the largest difference between their outputs.
//
// Compile:  hipcc -O2 -o main dl_primitives_demo.hip -lMIOpen
// Nvidia:   the cuDNN calls have the same shape (cudnnSetTensor4dDescriptor,
//           cudnnFindConvolutionForwardAlgorithmEx, cudnnConvolutionForward);
//           swap them in and build with nvcc -O2 ... -lcudnn

#include <hip/hip_runtime.h>
#include <miopen/miopen.h>

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

#define MIOPEN_CHECK(call)                                                \
  do {                                                                    \
    miopenStatus_t st_ = (call);                                          \
    if (st_ != miopenStatusSuccess) {                                     \
      std::fprintf(stderr, "%s:%d: MIOpen error %d\n", __FILE__,          \
                   __LINE__, (int)st_);                                   \
      std::exit(1);                                                       \
    }                                                                     \
  } while (0)

// Layer shape. Stride 1 and padding 1 keep the output at 56x56.
constexpr int N = 16, C = 64, H = 56, W = 56;  // input
constexpr int K = 64, R = 3, S = 3;            // filters
constexpr int P = 1;                           // padding

// The textbook version: each thread computes one output value by walking
// all C x R x S inputs that touch it, straight from global memory.
__global__ void directConv(const float *x, const float *w, float *y) {
  int ow = blockIdx.x * blockDim.x + threadIdx.x;
  int oh = blockIdx.y;
  int nk = blockIdx.z;  // batch and output channel folded together
  int n = nk / K, k = nk % K;
  if (ow >= W) return;
  float acc = 0.0f;
  for (int c = 0; c < C; ++c)
    for (int r = 0; r < R; ++r) {
      int ih = oh + r - P;
      if (ih < 0 || ih >= H) continue;
      for (int s = 0; s < S; ++s) {
        int iw = ow + s - P;
        if (iw < 0 || iw >= W) continue;
        acc += x[((n * C + c) * H + ih) * W + iw] *
               w[((k * C + c) * R + r) * S + s];
      }
    }
  y[((n * K + k) * H + oh) * W + ow] = acc;
}

static const char *algoName(miopenConvFwdAlgorithm_t a) {
  switch (a) {
    case miopenConvolutionFwdAlgoGEMM: return "GEMM (im2col)";
    case miopenConvolutionFwdAlgoDirect: return "Direct";
    case miopenConvolutionFwdAlgoFFT: return "FFT";
    case miopenConvolutionFwdAlgoWinograd: return "Winograd";
    case miopenConvolutionFwdAlgoImplicitGEMM: return "Implicit GEMM";
    default: return "other";
  }
}

int main() {
  hipDeviceProp_t prop;
  HIP_CHECK(hipGetDeviceProperties(&prop, 0));
  std::printf("Device: %s\n", prop.name);

  const size_t xElems = (size_t)N * C * H * W, wElems = (size_t)K * C * R * S,
               yElems = (size_t)N * K * H * W;
  const double flops = 2.0 * N * K * H * W * C * R * S;
  std::printf("Conv %dx%dx%dx%d * %dx%dx%dx%d, %.2f GFLOP per call\n\n", N, C,
              H, W, K, C, R, S, flops / 1e9);

  std::vector<float> hx(xElems), hw(wElems);
  for (size_t i = 0; i < xElems; ++i) hx[i] = (float)((i * 7) % 13) / 13.0f - 0.5f;
  for (size_t i = 0; i < wElems; ++i) hw[i] = (float)((i * 11) % 17) / 17.0f - 0.5f;

  float *x, *w, *yNaive, *yLib;
  HIP_CHECK(hipMalloc(&x, xElems * sizeof(float)));
  HIP_CHECK(hipMalloc(&w, wElems * sizeof(float)));
  HIP_CHECK(hipMalloc(&yNaive, yElems * sizeof(float)));
  HIP_CHECK(hipMalloc(&yLib, yElems * sizeof(float)));
  HIP_CHECK(hipMemcpy(x, hx.data(), xElems * sizeof(float), hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(w, hw.data(), wElems * sizeof(float), hipMemcpyHostToDevice));

  hipEvent_t start, stop;
  HIP_CHECK(hipEventCreate(&start));
  HIP_CHECK(hipEventCreate(&stop));
  const int reps = 10;

  // 1. Hand-written direct convolution.
  dim3 block(64), grid((W + 63) / 64, H, N * K);
  directConv<<<grid, block>>>(x, w, yNaive);  // warm-up
  HIP_CHECK(hipEventRecord(start, 0));
  for (int i = 0; i < reps; ++i) directConv<<<grid, block>>>(x, w, yNaive);
  HIP_CHECK(hipEventRecord(stop, 0));
  HIP_CHECK(hipEventSynchronize(stop));
  HIP_CHECK(hipGetLastError());
  float naiveMs;
  HIP_CHECK(hipEventElapsedTime(&naiveMs, start, stop));
  naiveMs /= reps;

  // 2. MIOpen. Describe the layer once.
  miopenHandle_t handle;
  MIOPEN_CHECK(miopenCreate(&handle));
  miopenTensorDescriptor_t xDesc, wDesc, yDesc;
  MIOPEN_CHECK(miopenCreateTensorDescriptor(&xDesc));
  MIOPEN_CHECK(miopenCreateTensorDescriptor(&wDesc));
  MIOPEN_CHECK(miopenCreateTensorDescriptor(&yDesc));
  MIOPEN_CHECK(miopenSet4dTensorDescriptor(xDesc, miopenFloat, N, C, H, W));
  MIOPEN_CHECK(miopenSet4dTensorDescriptor(wDesc, miopenFloat, K, C, R, S));
  miopenConvolutionDescriptor_t conv;
  MIOPEN_CHECK(miopenCreateConvolutionDescriptor(&conv));
  MIOPEN_CHECK(miopenInitConvolutionDescriptor(conv, miopenConvolution, P, P,
                                               1, 1, 1, 1));
  int on, oc, oh, ow;
  MIOPEN_CHECK(miopenGetConvolutionForwardOutputDim(conv, xDesc, wDesc, &on,
                                                    &oc, &oh, &ow));
  MIOPEN_CHECK(miopenSet4dTensorDescriptor(yDesc, miopenFloat, on, oc, oh, ow));

  // Scratch space: some algorithms (im2col GEMM, FFT, Winograd) need it.
  size_t wsBytes = 0;
  MIOPEN_CHECK(miopenConvolutionForwardGetWorkSpaceSize(handle, wDesc, xDesc,
                                                        conv, yDesc, &wsBytes));
  void *ws = nullptr;
  if (wsBytes) HIP_CHECK(hipMalloc(&ws, wsBytes));

  // Find: run every applicable algorithm on this shape and rank them. This
  // is the expensive first call that frameworks cache (MIOpen keeps the
  // answer in its find-db on disk; PyTorch's cudnn.benchmark does the same
  // in memory on Nvidia).
  miopenConvAlgoPerf_t perf[8];
  int found = 0;
  auto t0 = std::chrono::steady_clock::now();
  MIOPEN_CHECK(miopenFindConvolutionForwardAlgorithm(
      handle, xDesc, x, wDesc, w, conv, yDesc, yLib, 8, &found, perf, ws,
      wsBytes, /*exhaustiveSearch=*/true));
  HIP_CHECK(hipDeviceSynchronize());
  double findMs = std::chrono::duration<double, std::milli>(
                      std::chrono::steady_clock::now() - t0).count();

  std::printf("Find returned %d algorithms (workspace up to %.1f MB):\n", found,
              wsBytes / 1048576.0);
  for (int i = 0; i < found; ++i)
    std::printf("  %-14s %8.3f ms   %7.1f MB scratch\n",
                algoName(perf[i].fwd_algo), perf[i].time,
                perf[i].memory / 1048576.0);
  std::printf("Find itself took %.0f ms\n\n", findMs);
  if (found == 0) {
    std::fprintf(stderr, "no algorithm ran on this device\n");
    return 1;
  }

  // Steady state: call the winner the way a training loop would.
  const float alpha = 1.0f, beta = 0.0f;
  miopenConvFwdAlgorithm_t best = perf[0].fwd_algo;
  MIOPEN_CHECK(miopenConvolutionForward(handle, &alpha, xDesc, x, wDesc, w,
                                        conv, best, &beta, yDesc, yLib, ws,
                                        wsBytes));
  HIP_CHECK(hipEventRecord(start, 0));
  for (int i = 0; i < reps; ++i)
    MIOPEN_CHECK(miopenConvolutionForward(handle, &alpha, xDesc, x, wDesc, w,
                                          conv, best, &beta, yDesc, yLib, ws,
                                          wsBytes));
  HIP_CHECK(hipEventRecord(stop, 0));
  HIP_CHECK(hipEventSynchronize(stop));
  float libMs;
  HIP_CHECK(hipEventElapsedTime(&libMs, start, stop));
  libMs /= reps;

  // Same answer? Differences come from summation order (and, for FFT and
  // Winograd, from a different but equivalent arithmetic).
  std::vector<float> a(yElems), b(yElems);
  HIP_CHECK(hipMemcpy(a.data(), yNaive, yElems * sizeof(float), hipMemcpyDeviceToHost));
  HIP_CHECK(hipMemcpy(b.data(), yLib, yElems * sizeof(float), hipMemcpyDeviceToHost));
  double maxDiff = 0.0;
  for (size_t i = 0; i < yElems; ++i)
    maxDiff = std::fmax(maxDiff, std::fabs((double)a[i] - b[i]));

  // GFLOPS here counts the direct method's 2*N*K*H*W*C*R*S operations.
  // Winograd and FFT do fewer real multiplies, so their "effective" rate
  // can approach or pass the card's peak.
  std::printf("%-26s %9s %9s\n", "", "ms", "eff. GFLOPS");
  std::printf("%-26s %9.2f %9.1f\n", "hand-written direct conv", naiveMs,
              flops / (naiveMs * 1e6));
  std::printf("%-26s %9.2f %9.1f\n", "MIOpen, best from Find", libMs,
              flops / (libMs * 1e6));
  std::printf("speedup %.1fx, max |diff| %.2e\n", naiveMs / libMs, maxDiff);
  std::printf("Find cost as much as %.0f calls of the winner\n",
              findMs / libMs);

  if (ws) HIP_CHECK(hipFree(ws));
  MIOPEN_CHECK(miopenDestroyConvolutionDescriptor(conv));
  MIOPEN_CHECK(miopenDestroyTensorDescriptor(xDesc));
  MIOPEN_CHECK(miopenDestroyTensorDescriptor(wDesc));
  MIOPEN_CHECK(miopenDestroyTensorDescriptor(yDesc));
  MIOPEN_CHECK(miopenDestroy(handle));
  HIP_CHECK(hipFree(x));
  HIP_CHECK(hipFree(w));
  HIP_CHECK(hipFree(yNaive));
  HIP_CHECK(hipFree(yLib));
  return 0;
}
