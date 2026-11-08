// atomic_scope_demo.hip
// Companion example for: topnotchnote.com/gpu/libcudacxx_libhipcxx.html
//
// libcu++ (Nvidia) and libhipcxx (AMD) give device code the C++ standard
// library's atomics, plus a "thread scope" saying which threads must see
// the update: a block, the whole device, or the whole system (GPU + CPU).
// The same #include <cuda/atomic> works with both libraries.
//
// Every thread adds 1 to its block's counter, many times, through
// cuda::atomic_ref with each scope and with relaxed or seq_cst ordering.
// The result is the same in every case; the cost is not.
//
// Compile (AMD):    hipcc -O2 -std=c++17 -o main atomic_scope_demo.hip   (libhipcxx headers on the include path)
// Compile (Nvidia): rename hip -> cuda in the runtime calls and build with
//                   nvcc -O2 -std=c++17 -x cu -o main atomic_scope_demo.hip  (libcu++ ships with the CUDA Toolkit)

#include <hip/hip_runtime.h>
#include <cuda/atomic>

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

template <cuda::thread_scope Scope>
__global__ void count(unsigned int* counters, int iters, cuda::std::memory_order order) {
  cuda::atomic_ref<unsigned int, Scope> c(counters[blockIdx.x]);
  for (int i = 0; i < iters; ++i) c.fetch_add(1u, order);
}

template <cuda::thread_scope Scope>
static void run(const char* name, cuda::std::memory_order order, const char* oname) {
  const int blocks = 240, threads = 256, iters = 256;
  unsigned int* d;
  HIP_CHECK(hipMalloc(&d, blocks * sizeof(unsigned int)));
  hipEvent_t a, b;
  HIP_CHECK(hipEventCreate(&a));
  HIP_CHECK(hipEventCreate(&b));
  float ms = 0;
  for (int rep = 0; rep < 3; ++rep) {  // keep the last of three runs
    HIP_CHECK(hipMemset(d, 0, blocks * sizeof(unsigned int)));
    HIP_CHECK(hipEventRecord(a));
    count<Scope><<<blocks, threads>>>(d, iters, order);
    HIP_CHECK(hipEventRecord(b));
    HIP_CHECK(hipEventSynchronize(b));
    HIP_CHECK(hipEventElapsedTime(&ms, a, b));
  }
  unsigned int h0;
  HIP_CHECK(hipMemcpy(&h0, d, sizeof h0, hipMemcpyDeviceToHost));
  double ops = (double)blocks * threads * iters;
  std::printf("  %-8s %-8s %8.2f ms  %7.1f M atomic adds/s   block 0 counted %u (expected %d)\n",
              name, oname, ms, ops / ms / 1e3, h0, threads * iters);
  HIP_CHECK(hipFree(d));
}

int main() {
  std::printf("240 blocks x 256 threads x 256 fetch_add on one counter per block\n");
  run<cuda::thread_scope_block>("block", cuda::std::memory_order_relaxed, "relaxed");
  run<cuda::thread_scope_device>("device", cuda::std::memory_order_relaxed, "relaxed");
  run<cuda::thread_scope_system>("system", cuda::std::memory_order_relaxed, "relaxed");
  run<cuda::thread_scope_block>("block", cuda::std::memory_order_seq_cst, "seq_cst");
  run<cuda::thread_scope_device>("device", cuda::std::memory_order_seq_cst, "seq_cst");
  run<cuda::thread_scope_system>("system", cuda::std::memory_order_seq_cst, "seq_cst");
  return 0;
}
