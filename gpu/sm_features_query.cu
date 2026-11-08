// sm_features_query.cu
// Companion example for: topnotchnote.com/gpu/nvidia_sm_generations.html
//
// What one SM holds changed a lot from Volta to Blackwell: shared memory,
// resident warps, matrix instructions, clusters. This program asks the runtime
// for the per-SM limits of every visible GPU, then prints which of the
// generation-specific features covered on the page its compute capability
// has, so you can see where your card sits.
//
// This one is Nvidia-only (CUDA): the features it lists are Nvidia's.
// Compile:  nvcc -O2 -o main sm_features_query.cu
// Run:      ./main

#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>

#define CUDA_CHECK(call)                                                  \
  do {                                                                    \
    cudaError_t err_ = (call);                                            \
    if (err_ != cudaSuccess) {                                            \
      std::fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__,             \
                   cudaGetErrorString(err_));                             \
      std::exit(1);                                                       \
    }                                                                     \
  } while (0)

static int attr(cudaDeviceAttr a, int dev) {
  int v = 0;
  CUDA_CHECK(cudaDeviceGetAttribute(&v, a, dev));
  return v;
}

static const char* yes(bool b) { return b ? "yes" : "no"; }

int main() {
  int n = 0;
  CUDA_CHECK(cudaGetDeviceCount(&n));
  for (int d = 0; d < n; ++d) {
    cudaDeviceProp p;
    CUDA_CHECK(cudaGetDeviceProperties(&p, d));
    int cc = p.major * 10 + p.minor;
    std::printf("device %d: %s, compute capability %d.%d, %d SMs\n", d, p.name,
                p.major, p.minor, p.multiProcessorCount);

    std::printf("  per SM, as the runtime reports it:\n");
    std::printf("    shared memory          : %d KiB\n",
                attr(cudaDevAttrMaxSharedMemoryPerMultiprocessor, d) >> 10);
    std::printf("    shared memory / block  : %d KiB (opt-in max)\n",
                attr(cudaDevAttrMaxSharedMemoryPerBlockOptin, d) >> 10);
    std::printf("    32-bit registers       : %d\n",
                attr(cudaDevAttrMaxRegistersPerMultiprocessor, d));
    std::printf("    resident threads       : %d (%d warps)\n",
                p.maxThreadsPerMultiProcessor, p.maxThreadsPerMultiProcessor / 32);
    std::printf("    resident blocks        : %d\n",
                attr(cudaDevAttrMaxBlocksPerMultiprocessor, d));
    std::printf("    L2 for the whole GPU   : %.1f MiB\n", p.l2CacheSize / 1048576.0);

    // Feature map by compute capability (see the page for sources).
    std::printf("  features by compute capability:\n");
    std::printf("    tensor cores (7.0+)                         : %s\n", yes(cc >= 70));
    std::printf("    independent thread scheduling (7.0+)        : %s\n", yes(cc >= 70));
    std::printf("    async copy global->shared, cp.async (8.0+)  : %s\n", yes(cc >= 80));
    std::printf("    2:4 sparse tensor cores (8.0+)              : %s\n", yes(cc >= 80));
    std::printf("    thread block clusters, TMA (9.0+)           : %s\n", yes(cc >= 90));
    std::printf("    wgmma warpgroup MMA (sm_90a only)           : %s\n", yes(cc == 90));
    std::printf("    tcgen05 MMA + tensor memory (sm_100 family) : %s\n",
                yes(cc >= 100 && cc < 120));
#if CUDART_VERSION >= 11080
    std::printf("  runtime says cluster launch supported        : %s\n",
                yes(attr(cudaDevAttrClusterLaunch, d) != 0));
#endif
  }
  return 0;
}
