// gpu_generation_query.hip
// Companion example for: topnotchnote.com/gpu/gpu_generations.html
//
// Two questions that sound the same and are not:
//   1. Which generation is this GPU?  Asked at run time, from the device
//      properties: compute capability on Nvidia, gfx target on AMD.
//   2. Which generation was this code compiled for?  Answered at compile
//      time by the device compiler: __CUDA_ARCH__ on Nvidia, a __gfxNNN__
//      macro on AMD. A tiny kernel reports it back to the host.
// When the two disagree on Nvidia, the driver JIT-compiled PTX for you. On
// AMD they cannot disagree: a binary without the right gfx target refuses
// to launch, and the sample prints that error instead.
//
// Compile:  hipcc -O2 -o main gpu_generation_query.hip
//           hipcc -O2 --offload-arch=gfx90a -o main gpu_generation_query.hip
// Nvidia:   hipcc also targets Nvidia (HIP_PLATFORM=nvidia), or swap the hip
//           prefix for cuda in the API calls and build with nvcc, e.g.
//           nvcc -O2 -arch=sm_61 ...   (exact match)
//           nvcc -O2 -gencode arch=compute_52,code=compute_52 ...  (PTX only,
//           so the driver JIT-compiles it for whatever card is installed)

#include <hip/hip_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>

#define HIP_CHECK(call)                                                   \
  do {                                                                    \
    hipError_t err_ = (call);                                             \
    if (err_ != hipSuccess) {                                             \
      std::fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__,             \
                   hipGetErrorString(err_));                              \
      std::exit(1);                                                       \
    }                                                                     \
  } while (0)

// Writes the architecture the *device code* was compiled for.
// Nvidia: __CUDA_ARCH__ is major*100 + minor*10, so sm_61 gives 610.
// AMD:    the compiler defines one __gfxNNN__ macro per target; we map a
//         few common ones to a number so the host can print them.
__global__ void compiledFor(int *out) {
#if defined(__CUDA_ARCH__)
  *out = __CUDA_ARCH__;
#elif defined(__gfx950__)
  *out = 950;
#elif defined(__gfx942__)
  *out = 942;
#elif defined(__gfx90a__)
  *out = 90010;  // gfx90a: "a" is not a digit, so tag it
#elif defined(__gfx908__)
  *out = 908;
#elif defined(__gfx1100__)
  *out = 1100;
#elif defined(__gfx1030__)
  *out = 1030;
#else
  *out = -1;     // some other target: still ran, just not in our list
#endif
}

// What a generation buys you, by compute capability. Not exhaustive; these
// are the features people most often trip over when a card is "too old".
static void printNvidiaFeatures(int major, int minor) {
  int cc = major * 10 + minor;
  std::printf("  FP16 arithmetic          : %s\n", cc >= 53 ? "yes" : "no");
  std::printf("  Tensor Cores (FP16)      : %s\n", cc >= 70 ? "yes" : "no");
  std::printf("  BF16 / TF32 Tensor Cores : %s\n", cc >= 80 ? "yes" : "no");
  std::printf("  FP8 Tensor Cores         : %s\n", cc >= 89 ? "yes" : "no");
  std::printf("  TMA, thread block clusters: %s\n", cc >= 90 ? "yes" : "no");
  std::printf("  FP4 Tensor Cores         : %s\n", cc >= 100 ? "yes" : "no");
}

int main() {
  hipDeviceProp_t prop;
  HIP_CHECK(hipGetDeviceProperties(&prop, 0));

  std::printf("Device                 : %s\n", prop.name);
#if defined(__HIP_PLATFORM_AMD__)
  // gcnArchName carries the gfx target plus feature flags,
  // e.g. "gfx90a:sramecc+:xnack-". The part before ':' is the target.
  std::printf("gfx target (run time)  : %s\n", prop.gcnArchName);
  std::printf("major.minor            : %d.%d  (HIP fills these in on AMD "
              "too, but the gfx name is the one that matters)\n",
              prop.major, prop.minor);
#else
  std::printf("Compute capability     : %d.%d  (sm_%d%d)\n", prop.major,
              prop.minor, prop.major, prop.minor);
  printNvidiaFeatures(prop.major, prop.minor);
#endif

  int *dOut = nullptr;
  int hOut = 0;
  HIP_CHECK(hipMalloc(&dOut, sizeof(int)));
  compiledFor<<<1, 1>>>(dOut);
  hipError_t launchErr = hipGetLastError();
  if (launchErr != hipSuccess) {
    // The usual failure when the binary has no code for this generation:
    // "no kernel image is available" (Nvidia) or "invalid device function"
    // (AMD).
    std::printf("Compiled for (code)    : launch failed: %s\n",
                hipGetErrorString(launchErr));
    std::printf("  -> rebuild with -arch / --offload-arch for this device\n");
    return 1;
  }
  HIP_CHECK(hipMemcpy(&hOut, dOut, sizeof(int), hipMemcpyDeviceToHost));
  HIP_CHECK(hipFree(dOut));

#if defined(__HIP_PLATFORM_AMD__)
  if (hOut == 90010)
    std::printf("Compiled for (code)    : gfx90a\n");
  else
    std::printf("Compiled for (code)    : gfx%d\n", hOut);
#else
  int devCC = prop.major * 100 + prop.minor * 10;
  std::printf("Compiled for (code)    : sm_%d  (__CUDA_ARCH__ = %d)\n",
              hOut / 10, hOut);
  if (hOut == devCC)
    std::printf("  -> exact match: SASS built for this generation ran.\n");
  else
    std::printf("  -> mismatch: the driver JIT-compiled PTX built for an "
                "older generation. It works, but it skipped every feature "
                "newer than sm_%d.\n", hOut / 10);
#endif
  return 0;
}
