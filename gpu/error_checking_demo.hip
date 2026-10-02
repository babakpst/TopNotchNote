// error_checking_demo.hip
// Companion example for: topnotchnote.com/gpu/error_checking.html
//
// Four ways a GPU program goes wrong, and which of them an error check can
// actually see. Run with an argument to pick one:
//   ./main 1   A bad launch, unchecked: the program "succeeds" and prints zeros.
//   ./main 2   The same launch, checked: hipGetLastError() names the problem.
//   ./main 3   A fault inside the kernel: the launch reports success and the
//              error appears at the next synchronizing call. It is sticky:
//              every later call on this device fails with the same error.
//              hipGetLastError() clears the code it reports, not the fault.
//   ./main 4   Writing one element past the end: no error at all, because
//              the allocation was rounded up and d[n] is padding the
//              program owns. Only a memory checker (compute-sanitizer on
//              Nvidia, AddressSanitizer on AMD) catches it.
// Compile:  hipcc -O2 -o main error_checking_demo.hip
// Nvidia:   hipcc also targets Nvidia (HIP_PLATFORM=nvidia), or swap the hip
//           prefix for cuda in the API calls and build with nvcc.
// Note:     on AMD GPUs, case 3 usually ends the process with a
//           "Memory access fault by GPU" message instead of returning an
//           error code, which is the same lesson delivered more bluntly.

#include <hip/hip_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <vector>

// Wrap every runtime call. On failure: say what, where, and stop.
#define HIP_CHECK(call)                                                   \
  do {                                                                    \
    hipError_t err_ = (call);                                             \
    if (err_ != hipSuccess) {                                             \
      std::fprintf(stderr, "%s:%d: %s failed: %s (%s)\n", __FILE__,       \
                   __LINE__, #call, hipGetErrorString(err_),              \
                   hipGetErrorName(err_));                                \
      std::exit(1);                                                       \
    }                                                                     \
  } while (0)

// A launch returns nothing, so check it in two steps: hipGetLastError()
// catches a bad launch configuration; in a debug build, synchronizing as
// well makes a fault inside the kernel appear here rather than later.
#ifdef DEBUG_SYNC
#define KERNEL_CHECK()                                                    \
  do { HIP_CHECK(hipGetLastError()); HIP_CHECK(hipDeviceSynchronize()); } while (0)
#else
#define KERNEL_CHECK() HIP_CHECK(hipGetLastError())
#endif

__global__ void fill(float *out, float value, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) out[i] = value;
}

// Deliberately wrong: the guard lets one thread too many through.
__global__ void fillOffByOne(float *out, float value, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i <= n) out[i] = value;
}

static void printFirst(const char *label, const float *d, int n) {
  std::vector<float> h(n);
  HIP_CHECK(hipMemcpy(h.data(), d, n * sizeof(float), hipMemcpyDeviceToHost));
  std::printf("%s first values: %.1f %.1f %.1f\n", label, h[0], h[1], h[2]);
}

int main(int argc, char **argv) {
  const int which = argc > 1 ? std::atoi(argv[1]) : 1;
  // Deliberately not a power of two: 4,000,000 bytes is rounded up to a
  // larger allocation, so the element just past the end is still mapped.
  const int n = 1000000;
  float *d = nullptr;
  HIP_CHECK(hipMalloc(&d, n * sizeof(float)));
  HIP_CHECK(hipMemset(d, 0, n * sizeof(float)));

  if (which == 1) {
    // 2048 threads per block is over the 1024 limit on every current GPU.
    fill<<<(n + 2047) / 2048, 2048>>>(d, 7.0f, n);
    printFirst("1. unchecked launch,", d, n);
    std::printf("   no error was printed, and the kernel never ran\n");
  } else if (which == 2) {
    fill<<<(n + 2047) / 2048, 2048>>>(d, 7.0f, n);
    KERNEL_CHECK();
    printFirst("2. never reached,", d, n);
  } else if (which == 3) {
    float *wild = reinterpret_cast<float *>(0x10);  // not a device allocation
    fill<<<(n + 255) / 256, 256>>>(wild, 7.0f, n);
    std::printf("3. launch returned:      %s\n", hipGetErrorString(hipGetLastError()));
    std::printf("   synchronize returned: %s\n", hipGetErrorString(hipDeviceSynchronize()));
    float *again = nullptr;
    std::printf("   a new hipMalloc:      %s\n", hipGetErrorString(hipMalloc(&again, 4)));
    std::printf("   hipGetLastError:      %s\n", hipGetErrorString(hipGetLastError()));
    std::printf("   and once more:        %s\n", hipGetErrorString(hipGetLastError()));
    std::printf("   but hipMalloc again:  %s\n", hipGetErrorString(hipMalloc(&again, 4)));
  } else if (which == 4) {
    fillOffByOne<<<(n + 1 + 255) / 256, 256>>>(d, 7.0f, n);  // writes d[n]
    KERNEL_CHECK();
    HIP_CHECK(hipDeviceSynchronize());
    printFirst("4. off-by-one write,", d, n);
    std::printf("   every check passed; d[n] was written anyway\n");
  }

  hipFree(d);
  return 0;
}
