// rand_demo.hip
// Companion example for: topnotchnote.com/gpu/random_libraries.html
//
// GPU random numbers with hipRAND, which forwards to rocRAND on AMD and to
// cuRAND on Nvidia. Four experiments:
//   1. host API: fill 64M floats in device memory with each generator type
//   2. one big generate call against many small ones
//   3. device API: per-thread generators inside a kernel (Monte Carlo pi),
//      timing the hiprand_init setup separately from the generation, for
//      "same seed, different subsequence" and "different seed, subsequence 0"
//   4. reproducibility: the same seed gives the same numbers
//
// Compile (AMD):    hipcc -O2 -o main rand_demo.hip -lhiprand
// Compile (Nvidia): HIP_PLATFORM=nvidia hipcc -O2 -o main rand_demo.hip -lhiprand
//                   or rename hiprand -> curand, hip -> cuda and build with
//                   nvcc -O2 -x cu -o main rand_demo.hip -lcurand

#include <hip/hip_runtime.h>
#include <hiprand/hiprand.h>
#include <hiprand/hiprand_kernel.h>

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

#define RAND_CHECK(call)                                                  \
  do {                                                                    \
    hiprandStatus_t s_ = (call);                                          \
    if (s_ != HIPRAND_STATUS_SUCCESS) {                                   \
      std::fprintf(stderr, "%s:%d: hipRAND status %d\n", __FILE__,        \
                   __LINE__, (int)s_);                                    \
      std::exit(1);                                                       \
    }                                                                     \
  } while (0)

struct Timer {
  hipEvent_t a, b;
  Timer() { HIP_CHECK(hipEventCreate(&a)); HIP_CHECK(hipEventCreate(&b)); }
  void start() { HIP_CHECK(hipEventRecord(a)); }
  float stop() {
    float ms;
    HIP_CHECK(hipEventRecord(b));
    HIP_CHECK(hipEventSynchronize(b));
    HIP_CHECK(hipEventElapsedTime(&ms, a, b));
    return ms;
  }
};

// Device API, setup: one generator state per thread.
// mode 0: same seed, subsequence = thread id (independent streams)
// mode 1: seed + thread id, subsequence 0 (the faster setup the docs mention)
template <typename State>
__global__ void init_states(State* s, unsigned long long seed, int mode, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  if (mode == 0) hiprand_init(seed, i, 0, &s[i]);
  else           hiprand_init(seed + i, 0, 0, &s[i]);
}

// Device API, use: each thread throws darts at the unit square.
template <typename State>
__global__ void pi_darts(State* s, int per_thread, unsigned long long* hits, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  State local = s[i];  // work on a register copy, write it back at the end
  unsigned int inside = 0;
  for (int k = 0; k < per_thread; ++k) {
    float x = hiprand_uniform(&local), y = hiprand_uniform(&local);
    inside += (x * x + y * y <= 1.0f);
  }
  s[i] = local;
  atomicAdd(hits, (unsigned long long)inside);
}

template <typename State>
static void device_api(const char* name, int mode) {
  const int threads = 1 << 16, per_thread = 1024;
  State* d;
  unsigned long long* dHits;
  HIP_CHECK(hipMalloc(&d, sizeof(State) * threads));
  HIP_CHECK(hipMalloc(&dHits, sizeof(unsigned long long)));
  HIP_CHECK(hipMemset(dHits, 0, sizeof(unsigned long long)));
  Timer t;
  init_states<<<threads / 256, 256>>>(d, 1234ULL, mode, threads);  // warm-up
  HIP_CHECK(hipDeviceSynchronize());
  t.start();
  init_states<<<threads / 256, 256>>>(d, 1234ULL, mode, threads);
  float init_ms = t.stop();
  t.start();
  pi_darts<<<threads / 256, 256>>>(d, per_thread, dHits, threads);
  float gen_ms = t.stop();
  unsigned long long hits;
  HIP_CHECK(hipMemcpy(&hits, dHits, sizeof(hits), hipMemcpyDeviceToHost));
  double darts = (double)threads * per_thread;
  std::printf("  %-34s init %8.3f ms  generate %6.2f ms (%.0f M darts)  pi ~ %.5f\n", name,
              init_ms, gen_ms, darts / 1e6, 4.0 * hits / darts);
  HIP_CHECK(hipFree(d));
  HIP_CHECK(hipFree(dHits));
}

int main() {
  const size_t n = size_t(1) << 26;  // 64M floats = 256 MB
  float* d;
  HIP_CHECK(hipMalloc(&d, n * sizeof(float)));
  Timer t;

  std::printf("1. Host API: %zu uniform floats into device memory\n", n);
  struct { const char* name; hiprandRngType_t type; } gens[] = {
      {"XORWOW", HIPRAND_RNG_PSEUDO_XORWOW},
      {"MRG32k3a", HIPRAND_RNG_PSEUDO_MRG32K3A},
      {"MTGP32", HIPRAND_RNG_PSEUDO_MTGP32},
      {"Philox4x32-10", HIPRAND_RNG_PSEUDO_PHILOX4_32_10},
      {"Sobol32 (quasi)", HIPRAND_RNG_QUASI_SOBOL32},
  };
  for (auto& g : gens) {
    hiprandGenerator_t gen;
    t.start();
    RAND_CHECK(hiprandCreateGenerator(&gen, g.type));
    if (g.type != HIPRAND_RNG_QUASI_SOBOL32) RAND_CHECK(hiprandSetPseudoRandomGeneratorSeed(gen, 1234ULL));
    RAND_CHECK(hiprandGenerateUniform(gen, d, n));
    float first = t.stop();
    t.start();
    RAND_CHECK(hiprandGenerateUniform(gen, d, n));
    float again = t.stop();
    std::printf("  %-16s create + first call %8.2f ms   next call %6.2f ms  (%5.1f GB/s written)\n",
                g.name, first, again, n * sizeof(float) / again / 1e6);
    RAND_CHECK(hiprandDestroyGenerator(gen));
  }

  std::printf("\n2. The same 64M floats as 1 call vs 1,024 calls of 64K (Philox)\n");
  {
    hiprandGenerator_t gen;
    RAND_CHECK(hiprandCreateGenerator(&gen, HIPRAND_RNG_PSEUDO_PHILOX4_32_10));
    RAND_CHECK(hiprandGenerateUniform(gen, d, n));  // warm-up
    t.start();
    RAND_CHECK(hiprandGenerateUniform(gen, d, n));
    float one = t.stop();
    t.start();
    for (int k = 0; k < 1024; ++k) RAND_CHECK(hiprandGenerateUniform(gen, d + k * (n / 1024), n / 1024));
    float many = t.stop();
    std::printf("  one call %6.2f ms, 1,024 calls %6.2f ms\n", one, many);
    RAND_CHECK(hiprandDestroyGenerator(gen));
  }

  std::printf("\n3. Device API: 65,536 threads, 1,024 darts each\n");
  device_api<hiprandState_t>("XORWOW, same seed, subsequence = i", 0);
  device_api<hiprandState_t>("XORWOW, seed + i, subsequence 0", 1);
  device_api<hiprandStatePhilox4_32_10_t>("Philox, same seed, subsequence = i", 0);

  std::printf("\n4. Reproducibility: Philox, seed 42, first three values, twice\n");
  for (int run = 0; run < 2; ++run) {
    hiprandGenerator_t gen;
    RAND_CHECK(hiprandCreateGenerator(&gen, HIPRAND_RNG_PSEUDO_PHILOX4_32_10));
    RAND_CHECK(hiprandSetPseudoRandomGeneratorSeed(gen, 42ULL));
    RAND_CHECK(hiprandGenerateUniform(gen, d, 1024));
    float h[3];
    HIP_CHECK(hipMemcpy(h, d, sizeof(h), hipMemcpyDeviceToHost));
    std::printf("  run %d: %.7f %.7f %.7f\n", run + 1, h[0], h[1], h[2]);
    RAND_CHECK(hiprandDestroyGenerator(gen));
  }
  HIP_CHECK(hipFree(d));
  return 0;
}
