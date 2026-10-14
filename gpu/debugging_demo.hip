// debugging_demo.hip
// Companion example for: topnotchnote.com/gpu/gpu_debugging.html
//
// Four small kernels, each with one classic GPU bug, chosen on the command
// line. Three of them run "successfully": no error code, no crash. That is
// the point. Run each one under the tools on the page to see them caught.
//
//   ./main oob     off-by-one: thread n writes one element past the end
//   ./main race    shared-memory sum missing a barrier: wrong, sometimes
//   ./main uninit  reads device memory that was allocated but never written
//   ./main crash   writes through a null pointer: the one bug that reports
//
// Compile:  hipcc -g -O0 -o main debugging_demo.hip
//           (-g keeps line numbers so the tools can point at source lines)
// Nvidia:   swap the hip prefix for cuda and build with
//           nvcc -g -G -o main debugging_demo.cu
//           (-G adds device-side debug info for cuda-gdb; it disables
//           optimization, so drop it for performance work)

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

constexpr int BLOCK = 256;

// Bug 1: `<=` instead of `<`. The thread with i == n writes one element
// past the end. The allocation is rounded up, so nothing faults.
__global__ void fillOffByOne(int *a, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i <= n) a[i] = i;
}

// Bug 2: a block-wide tree sum in shared memory with the barrier inside the
// loop missing. Lanes of different warps read partial sums that another
// warp has not finished writing.
__global__ void blockSumNoBarrier(const int *in, int *out) {
  __shared__ int s[BLOCK];
  int t = threadIdx.x;
  s[t] = in[blockIdx.x * BLOCK + t];
  __syncthreads();
  for (int stride = BLOCK / 2; stride > 0; stride /= 2) {
    if (t < stride) s[t] += s[t + stride];
    // __syncthreads();   <- the missing line
  }
  if (t == 0) out[blockIdx.x] = s[0];
}

// Bug 3: reads `scratch`, which was allocated but never initialized.
__global__ void useUninitialized(const int *scratch, int *out, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) out[i] = scratch[i] + 1;
}

// Bug 4: a pointer that was never set. This one does fault.
__global__ void writeNull(int *p) { p[threadIdx.x] = 42; }

int main(int argc, char **argv) {
  const char *mode = argc > 1 ? argv[1] : "";
  const int n = 1000;

  if (!std::strcmp(mode, "oob")) {
    int *a;
    HIP_CHECK(hipMalloc(&a, n * sizeof(int)));
    fillOffByOne<<<(n + BLOCK) / BLOCK, BLOCK>>>(a, n);
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipDeviceSynchronize());
    std::printf("oob: finished, no error reported\n");
    HIP_CHECK(hipFree(a));

  } else if (!std::strcmp(mode, "race")) {
    const int blocks = 4096, total = blocks * BLOCK;
    int *in, *out, h[blocks];
    HIP_CHECK(hipMalloc(&in, total * sizeof(int)));
    HIP_CHECK(hipMalloc(&out, blocks * sizeof(int)));
    int *ones = new int[total];
    for (int i = 0; i < total; ++i) ones[i] = 1;
    HIP_CHECK(hipMemcpy(in, ones, total * sizeof(int), hipMemcpyHostToDevice));
    int wrong = 0;
    for (int run = 0; run < 10; ++run) {
      blockSumNoBarrier<<<blocks, BLOCK>>>(in, out);
      HIP_CHECK(hipGetLastError());
      HIP_CHECK(hipMemcpy(h, out, sizeof(h), hipMemcpyDeviceToHost));
      for (int b = 0; b < blocks; ++b) wrong += (h[b] != BLOCK);
    }
    std::printf("race: %d of %d block sums wrong (expected %d each), no error reported\n",
                wrong, 10 * blocks, BLOCK);
    delete[] ones;
    HIP_CHECK(hipFree(in));
    HIP_CHECK(hipFree(out));

  } else if (!std::strcmp(mode, "uninit")) {
    int *scratch, *out;
    HIP_CHECK(hipMalloc(&scratch, n * sizeof(int)));  // never written
    HIP_CHECK(hipMalloc(&out, n * sizeof(int)));
    useUninitialized<<<(n + BLOCK - 1) / BLOCK, BLOCK>>>(scratch, out, n);
    HIP_CHECK(hipGetLastError());
    int first;
    HIP_CHECK(hipMemcpy(&first, out, sizeof(int), hipMemcpyDeviceToHost));
    std::printf("uninit: out[0] = %d, no error reported\n", first);
    HIP_CHECK(hipFree(scratch));
    HIP_CHECK(hipFree(out));

  } else if (!std::strcmp(mode, "crash")) {
    int *p = nullptr;
    writeNull<<<1, 32>>>(p);
    std::printf("crash: launch returned %s\n", hipGetErrorString(hipGetLastError()));
    std::fflush(stdout);
    // The fault is only reported when the host waits for the kernel.
    HIP_CHECK(hipDeviceSynchronize());

  } else {
    std::fprintf(stderr, "usage: %s oob|race|uninit|crash\n", argv[0]);
    return 2;
  }
  return 0;
}
