// vectorized_loads_demo.hip
// Companion example for: topnotchnote.com/gpu/vectorized_loads.html
//
// Copies a buffer three ways: one float per thread, one float2 (8 bytes) per
// thread, and one float4 (16 bytes) per thread. Each version moves the same
// bytes; the wide versions do it with 1/2 and 1/4 as many load and store
// instructions. The program prints the copy bandwidth for each width over a
// range of sizes. It then repeats the copy with only a few threads in
// flight (one block per compute unit, grid-stride loop), where the width of
// each load decides how many bytes are on their way at once. Finally it shows
// what happens when a float4 pointer is not 16-byte aligned (run with the
// argument "misaligned").
//
// To see the instructions the compiler emitted:
//   AMD:    hipcc -O2 --save-temps -c vectorized_loads_demo.hip
//           and look for global_load_dword / _dwordx2 / _dwordx4 in the .s file
//   Nvidia: cuobjdump -sass ./main | grep -E 'LDG|STG'   (LDG.E, .64, .128)
//
// Compile:  hipcc -O2 -o main vectorized_loads_demo.hip
// Nvidia:   hipcc also targets Nvidia (HIP_PLATFORM=nvidia), or swap the hip
//           prefix for cuda in the API calls and build with nvcc.

#include <hip/hip_runtime.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
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

// T is float, float2 or float4. n_vec counts elements of T.
template <typename T>
__global__ void copy(const T* __restrict__ in, T* __restrict__ out, size_t n_vec) {
  size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
  if (i < n_vec) out[i] = in[i];
}

// Grid-stride version: a fixed, small number of threads walks the buffer.
template <typename T>
__global__ void copy_strided(const T* __restrict__ in, T* __restrict__ out, size_t n_vec) {
  for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n_vec;
       i += (size_t)gridDim.x * blockDim.x)
    out[i] = in[i];
}

template <typename T>
static float time_few(const float* in, float* out, size_t n_floats, unsigned blocks,
                      unsigned threads, int reps) {
  const size_t n_vec = n_floats / (sizeof(T) / sizeof(float));
  hipEvent_t a, b;
  HIP_CHECK(hipEventCreate(&a));
  HIP_CHECK(hipEventCreate(&b));
  copy_strided<T><<<blocks, threads>>>((const T*)in, (T*)out, n_vec);
  std::vector<float> ms;
  for (int r = 0; r < reps; ++r) {
    HIP_CHECK(hipEventRecord(a));
    copy_strided<T><<<blocks, threads>>>((const T*)in, (T*)out, n_vec);
    HIP_CHECK(hipEventRecord(b));
    HIP_CHECK(hipEventSynchronize(b));
    float t;
    HIP_CHECK(hipEventElapsedTime(&t, a, b));
    ms.push_back(t);
  }
  HIP_CHECK(hipGetLastError());
  HIP_CHECK(hipEventDestroy(a));
  HIP_CHECK(hipEventDestroy(b));
  std::sort(ms.begin(), ms.end());
  return ms[ms.size() / 2];
}

template <typename T>
static float time_copy(const float* in, float* out, size_t n_floats, int reps) {
  const size_t width = sizeof(T) / sizeof(float);
  const size_t n_vec = n_floats / width;      // sizes below are multiples of 4
  const unsigned threads = 256;
  const unsigned blocks = (unsigned)((n_vec + threads - 1) / threads);
  hipEvent_t a, b;
  HIP_CHECK(hipEventCreate(&a));
  HIP_CHECK(hipEventCreate(&b));
  copy<T><<<blocks, threads>>>((const T*)in, (T*)out, n_vec);   // warm-up
  std::vector<float> ms;
  for (int r = 0; r < reps; ++r) {
    HIP_CHECK(hipEventRecord(a));
    copy<T><<<blocks, threads>>>((const T*)in, (T*)out, n_vec);
    HIP_CHECK(hipEventRecord(b));
    HIP_CHECK(hipEventSynchronize(b));
    float t;
    HIP_CHECK(hipEventElapsedTime(&t, a, b));
    ms.push_back(t);
  }
  HIP_CHECK(hipGetLastError());
  HIP_CHECK(hipEventDestroy(a));
  HIP_CHECK(hipEventDestroy(b));
  std::sort(ms.begin(), ms.end());
  return ms[ms.size() / 2];
}

int main(int argc, char** argv) {
  const size_t max_floats = (size_t)64 << 20;   // 256 MB per buffer
  float *in, *out;
  HIP_CHECK(hipMalloc(&in, max_floats * sizeof(float)));
  HIP_CHECK(hipMalloc(&out, max_floats * sizeof(float)));
  HIP_CHECK(hipMemset(in, 0, max_floats * sizeof(float)));

  if (argc > 1 && std::strcmp(argv[1], "misaligned") == 0) {
    // in + 1 is 4-byte aligned but not 16-byte aligned. A float4 load from it
    // is not allowed; expect a misaligned-address error from the runtime.
    copy<float4><<<1, 64>>>((const float4*)(in + 1), (float4*)out, 64);
    hipError_t e = hipDeviceSynchronize();
    std::printf("float4 copy from in + 1: %s\n", hipGetErrorString(e));
    return 0;
  }

  std::printf("%10s %12s %12s %12s   (GB/s, read + write, median of 20)\n",
              "size", "float", "float2", "float4");
  for (size_t n = (size_t)1 << 16; n <= max_floats; n <<= 2) {
    const double bytes = 2.0 * n * sizeof(float);
    float t1 = time_copy<float>(in, out, n, 20);
    float t2 = time_copy<float2>(in, out, n, 20);
    float t4 = time_copy<float4>(in, out, n, 20);
    std::printf("%8zu KB %12.1f %12.1f %12.1f\n", n * sizeof(float) / 1024,
                bytes / t1 / 1e6, bytes / t2 / 1e6, bytes / t4 / 1e6);
  }

  // Few threads in flight: one block per compute unit, 64 MB copied.
  int cus = 0;
  HIP_CHECK(hipDeviceGetAttribute(&cus, hipDeviceAttributeMultiprocessorCount, 0));
  const size_t n = (size_t)16 << 20;
  const double bytes = 2.0 * n * sizeof(float);
  std::printf("\nfew threads in flight: %d blocks (one per compute unit), 64 MB\n", cus);
  std::printf("%10s %12s %12s %12s   (GB/s)\n", "threads", "float", "float2", "float4");
  for (unsigned t : {64u, 128u, 256u, 1024u}) {
    float t1 = time_few<float>(in, out, n, cus, t, 10);
    float t2 = time_few<float2>(in, out, n, cus, t, 10);
    float t4 = time_few<float4>(in, out, n, cus, t, 10);
    std::printf("%10u %12.1f %12.1f %12.1f\n", t, bytes / t1 / 1e6, bytes / t2 / 1e6,
                bytes / t4 / 1e6);
  }
  HIP_CHECK(hipFree(in));
  HIP_CHECK(hipFree(out));
  return 0;
}
