// memory_management_demo.hip
// Companion example for: topnotchnote.com/gpu/memory_management.html
//
// Four experiments on device and managed (unified) memory:
//   1. What allocation costs: 1,000 rounds of a 1 MB hipMalloc/hipFree,
//      against the stream-ordered hipMallocAsync/hipFreeAsync.
//   2. hipFree waits for the device: freeing an unrelated buffer while a
//      kernel runs, against hipFreeAsync.
//   3. Managed memory: a kernel over 256 MB that was written on the host,
//      run cold, warm, after the host touches it again, and after a
//      prefetch, against explicit hipMalloc + hipMemcpy.
//   4. Oversubscription: a managed buffer 1.5 times the size of device
//      memory, which hipMalloc refuses. Opt-in, because it thrashes on
//      purpose and needs that much free host RAM: run it as ./main 4
// Compile:  hipcc -O2 -o main memory_management_demo.hip
// Nvidia:   hipcc also targets Nvidia (HIP_PLATFORM=nvidia), or swap the hip
//           prefix for cuda in the API calls and build with nvcc.
// AMD:      managed memory migrates on page faults only when XNACK is enabled
//           (HSA_XNACK=1 on supported GPUs); otherwise experiments 3 and 4
//           behave differently. Prefetch signatures differ in CUDA 13.

#include <hip/hip_runtime.h>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <unistd.h>

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

static double msSince(Clock::time_point start) {
  return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
}

// Enough arithmetic per element that the kernel runs for a while.
__global__ void busyKernel(float *data, int n, int iters) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    float x = data[i];
    for (int k = 0; k < iters; ++k) {
      x = x * 0.999f + 0.001f;
    }
    data[i] = x;
  }
}

// One pass over the data, grid-stride so it works for any size.
__global__ void addOne(float *data, size_t n) {
  size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;
  for (size_t i = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x; i < n;
       i += stride) {
    data[i] += 1.0f;
  }
}

static double timedAddOne(float *data, size_t n) {
  auto t = Clock::now();
  addOne<<<1024, 256>>>(data, n);
  HIP_CHECK(hipGetLastError());
  HIP_CHECK(hipDeviceSynchronize());
  return msSince(t);
}

int main(int argc, char **argv) {
  const bool runOversubscription = argc > 1 && std::strcmp(argv[1], "4") == 0;

  hipDeviceProp_t prop;
  HIP_CHECK(hipGetDeviceProperties(&prop, 0));
  int device = 0;
  HIP_CHECK(hipGetDevice(&device));
  size_t freeBytes = 0, totalBytes = 0;
  HIP_CHECK(hipMemGetInfo(&freeBytes, &totalBytes));
  std::printf("Device: %s, %zu MB total, %zu MB free\n\n", prop.name, totalBytes >> 20,
              freeBytes >> 20);

  // 1. Allocation cost.
  const int rounds = 1000;
  const size_t small = 1u << 20;
  void *p = nullptr;
  HIP_CHECK(hipMalloc(&p, small));  // warmup
  HIP_CHECK(hipFree(p));
  auto t = Clock::now();
  for (int r = 0; r < rounds; ++r) {
    HIP_CHECK(hipMalloc(&p, small));
    HIP_CHECK(hipFree(p));
  }
  double syncUs = msSince(t) * 1000.0 / rounds;

  hipStream_t stream;
  HIP_CHECK(hipStreamCreate(&stream));
  HIP_CHECK(hipMallocAsync(&p, small, stream));  // warmup
  HIP_CHECK(hipFreeAsync(p, stream));
  HIP_CHECK(hipStreamSynchronize(stream));
  t = Clock::now();
  for (int r = 0; r < rounds; ++r) {
    HIP_CHECK(hipMallocAsync(&p, small, stream));
    HIP_CHECK(hipFreeAsync(p, stream));
  }
  HIP_CHECK(hipStreamSynchronize(stream));
  double asyncUs = msSince(t) * 1000.0 / rounds;
  std::printf("1. allocate and free 1 MB, %d rounds\n", rounds);
  std::printf("   hipMalloc + hipFree:           %6.1f us per round\n", syncUs);
  std::printf("   hipMallocAsync + hipFreeAsync: %6.1f us per round\n\n", asyncUs);

  // 2. hipFree waits for the device.
  const int n = 1 << 22;
  float *d_work = nullptr;
  HIP_CHECK(hipMalloc(&d_work, n * sizeof(float)));
  HIP_CHECK(hipMemset(d_work, 0, n * sizeof(float)));
  std::printf("2. free an unrelated 1 MB buffer while a ~20 ms kernel runs\n");
  for (int which = 0; which < 2; ++which) {
    void *unrelated = nullptr;
    if (which == 0) HIP_CHECK(hipMalloc(&unrelated, small));
    else HIP_CHECK(hipMallocAsync(&unrelated, small, stream));
    HIP_CHECK(hipDeviceSynchronize());
    busyKernel<<<(n + 255) / 256, 256, 0, stream>>>(d_work, n, 2000);
    t = Clock::now();
    if (which == 0) HIP_CHECK(hipFree(unrelated));
    else HIP_CHECK(hipFreeAsync(unrelated, stream));
    double returned = msSince(t);
    HIP_CHECK(hipDeviceSynchronize());
    std::printf("   %-13s returned after %7.3f ms\n", which == 0 ? "hipFree" : "hipFreeAsync",
                returned);
  }
  std::printf("\n");

  // 3. Managed memory.
  const size_t elems = (256u << 20) / sizeof(float);
  const size_t bytes = elems * sizeof(float);
  float *managed = nullptr;
  HIP_CHECK(hipMallocManaged(&managed, bytes, hipMemAttachGlobal));
  for (size_t i = 0; i < elems; ++i) managed[i] = 0.0f;  // pages now live on the host

  std::printf("3. one pass over 256 MB of managed memory, written on the host first\n");
  std::printf("   first kernel (pages fault over):  %7.1f ms\n", timedAddOne(managed, elems));
  std::printf("   second kernel (already resident): %7.1f ms\n", timedAddOne(managed, elems));
  for (size_t i = 0; i < elems; i += 1024) managed[i] += 1.0f;  // host touches every page
  std::printf("   after the host touched it again:  %7.1f ms\n", timedAddOne(managed, elems));
  for (size_t i = 0; i < elems; i += 1024) managed[i] += 1.0f;
  t = Clock::now();
  HIP_CHECK(hipMemPrefetchAsync(managed, bytes, device, 0));
  HIP_CHECK(hipDeviceSynchronize());
  double prefetchMs = msSince(t);
  std::printf("   prefetch, then kernel:             %7.1f ms + %.1f ms\n", prefetchMs,
              timedAddOne(managed, elems));

  float *h_plain = static_cast<float *>(std::malloc(bytes));
  for (size_t i = 0; i < elems; ++i) h_plain[i] = 0.0f;
  float *d_plain = nullptr;
  HIP_CHECK(hipMalloc(&d_plain, bytes));
  t = Clock::now();
  HIP_CHECK(hipMemcpy(d_plain, h_plain, bytes, hipMemcpyHostToDevice));
  double copyMs = msSince(t);
  std::printf("   hipMalloc + hipMemcpy, then kernel: %6.1f ms + %.1f ms\n\n", copyMs,
              timedAddOne(d_plain, elems));
  HIP_CHECK(hipFree(d_plain));
  std::free(h_plain);
  HIP_CHECK(hipFree(managed));

  // 4. Oversubscription.
  HIP_CHECK(hipMemGetInfo(&freeBytes, &totalBytes));
  const size_t over = totalBytes / 2 * 3;
  const size_t hostFree = static_cast<size_t>(sysconf(_SC_AVPHYS_PAGES)) *
                          static_cast<size_t>(sysconf(_SC_PAGESIZE));
  if (!runOversubscription) {
    std::printf("4. skipped: run ./main 4 to oversubscribe the device "
                "(slow, needs about %zu MB of free host RAM)\n", (over + (over >> 1)) >> 20);
  } else if (hostFree < over + (over >> 1)) {
    std::printf("4. skipped: needs about %zu MB of free host RAM, %zu MB free\n",
                (over + (over >> 1)) >> 20, hostFree >> 20);
  } else {
    std::printf("4. a buffer of %zu MB on a %zu MB device\n", over >> 20, totalBytes >> 20);
    void *tooBig = nullptr;
    hipError_t err = hipMalloc(&tooBig, over);
    std::printf("   hipMalloc:        %s\n", hipGetErrorString(err));
    (void)hipGetLastError();  // clear it: out of memory is not a sticky error
    float *overManaged = nullptr;
    HIP_CHECK(hipMallocManaged(&overManaged, over, hipMemAttachGlobal));
    const size_t overElems = over / sizeof(float);
    for (size_t i = 0; i < overElems; i += 1024) overManaged[i] = 0.0f;
    double first = timedAddOne(overManaged, overElems);
    double second = timedAddOne(overManaged, overElems);
    std::printf("   hipMallocManaged: works; one pass took %.0f ms, then %.0f ms\n", first,
                second);
    std::printf("   (%.2f GB/s and %.2f GB/s: every pass pages data in and out)\n",
                over / (first * 1e-3) / 1e9, over / (second * 1e-3) / 1e9);
    HIP_CHECK(hipFree(overManaged));
  }

  HIP_CHECK(hipFree(d_work));
  HIP_CHECK(hipStreamDestroy(stream));
  return 0;
}
