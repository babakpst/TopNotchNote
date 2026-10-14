// thrust_demo.hip
// Companion example for: topnotchnote.com/gpu/thrust.html
//
// STL-style algorithms on the GPU. The same source builds against Thrust
// (CUDA) and rocThrust (ROCm): both use the thrust:: namespace and the
// same headers. Timed on 64M elements:
//   1. reduce             : the sum the reduction page builds by hand
//   2. transform_reduce   : sum of squares, fused, with no temporary array
//   3. transform + reduce : the same thing in two steps, for comparison
//   4. inclusive_scan     : running totals (prefix sum)
//   5. sort               : 64M 32-bit keys
//   6. count_if           : how many elements pass a test
// Each line prints time and effective bandwidth.
//
// Compile:  hipcc -O2 -o main thrust_demo.hip        (needs rocThrust)
// Nvidia:   nvcc -O2 -x cu -o main thrust_demo.hip   (Thrust ships with
//           CUDA), after swapping the hip runtime calls (device properties,
//           events) for their cuda twins.

#include <hip/hip_runtime.h>
#include <thrust/count.h>
#include <thrust/device_vector.h>
#include <thrust/functional.h>
#include <thrust/reduce.h>
#include <thrust/scan.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>
#include <thrust/transform.h>
#include <thrust/transform_reduce.h>

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

struct Square {
  __host__ __device__ float operator()(float x) const { return x * x; }
};
struct Above {
  float t;
  __host__ __device__ bool operator()(float x) const { return x > t; }
};
// A cheap pseudo-random value per index, so the data is not sorted already.
struct Hash {
  __host__ __device__ unsigned operator()(unsigned i) const {
    i ^= i >> 16; i *= 0x7feb352du; i ^= i >> 15; i *= 0x846ca68bu;
    return i ^ (i >> 16);
  }
};
struct ToUnit {
  __host__ __device__ float operator()(unsigned i) const {
    return (float)(i % 1024u) / 1024.0f;
  }
};

int main() {
  hipDeviceProp_t prop;
  HIP_CHECK(hipGetDeviceProperties(&prop, 0));
  const size_t n = 64u << 20;
  std::printf("Device: %s, %zu elements\n\n", prop.name, n);

  // device_vector owns GPU memory; no hipMalloc, no hipFree.
  thrust::device_vector<unsigned> keys(n);
  thrust::device_vector<float> x(n), tmp(n);
  thrust::sequence(keys.begin(), keys.end());
  thrust::transform(keys.begin(), keys.end(), keys.begin(), Hash());
  thrust::transform(keys.begin(), keys.end(), x.begin(), ToUnit());

  hipEvent_t start, stop;
  HIP_CHECK(hipEventCreate(&start));
  HIP_CHECK(hipEventCreate(&stop));
  // Times one call of f (after one warm-up call); bytesMoved is the
  // minimum traffic the operation needs, for the GB/s column.
  auto timeIt = [&](const char *name, double bytesMoved, auto f) {
    f();
    HIP_CHECK(hipEventRecord(start, 0));
    f();
    HIP_CHECK(hipEventRecord(stop, 0));
    HIP_CHECK(hipEventSynchronize(stop));
    float ms = 0.0f;
    HIP_CHECK(hipEventElapsedTime(&ms, start, stop));
    std::printf("%-22s %9.3f ms %9.1f GB/s\n", name, ms,
                bytesMoved / (ms * 1e6));
  };
  const double fb = (double)n * sizeof(float);

  float sum = 0.0f, sumSq = 0.0f, sumSq2 = 0.0f;
  size_t above = 0;
  timeIt("reduce", fb, [&] { sum = thrust::reduce(x.begin(), x.end()); });
  timeIt("transform_reduce", fb, [&] {
    sumSq = thrust::transform_reduce(x.begin(), x.end(), Square(), 0.0f,
                                     thrust::plus<float>());
  });
  timeIt("transform + reduce", 3 * fb, [&] {
    thrust::transform(x.begin(), x.end(), tmp.begin(), Square());
    sumSq2 = thrust::reduce(tmp.begin(), tmp.end());
  });
  timeIt("inclusive_scan", 2 * fb, [&] {
    thrust::inclusive_scan(x.begin(), x.end(), tmp.begin());
  });
  timeIt("count_if", fb, [&] {
    above = thrust::count_if(x.begin(), x.end(), Above{0.5f});
  });
  // Sorting in place would hand the second call sorted data, so sort a
  // fresh copy each time and count the copy in the time.
  thrust::device_vector<unsigned> work(n);
  timeIt("sort (incl. copy)", 2.0 * n * sizeof(unsigned), [&] {
    work = keys;
    thrust::sort(work.begin(), work.end());
  });
  bool sorted = thrust::is_sorted(work.begin(), work.end());

  std::printf("\nsum = %.1f, sum of squares = %.1f (fused) / %.1f (two "
              "steps), above 0.5: %zu, sorted: %s\n", sum, sumSq, sumSq2,
              above, sorted ? "yes" : "NO");

  HIP_CHECK(hipEventDestroy(start));
  HIP_CHECK(hipEventDestroy(stop));
  return 0;
}
