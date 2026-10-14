// atomics_demo.hip
// Companion example for: topnotchnote.com/gpu/atomics.html
//
// What atomicAdd fixes, what it costs, and how to need fewer of them.
//   1. Counting 16M elements into one global counter:
//        a. plain `*count += 1`          (a data race: wrong answer)
//        b. one atomicAdd per element    (right, and faster than you would
//           expect: when every thread in a warp hits the same address, the
//           compiler merges them into one atomic per warp; look for VOTE,
//           POPC and a single RED in the SASS)
//        c. block-level sum in shared memory, one atomicAdd per block
//   Real contention needs addresses that come from data, which is what the
//   histogram in part 2 has.
//   2. A 256-bin histogram of 64M bytes, two kernels, two inputs:
//        kernels: atomics straight to global memory, or a private
//                 histogram per block in shared memory merged at the end
//        inputs:  uniform random bytes, or every byte the same value
//                 (the worst case: every atomic hits one address)
//   3. Float atomicAdd is not deterministic: the same float sum, run 20
//      times, can come back with more than one answer.
//
// Compile:  hipcc -O2 -o main atomics_demo.hip
// Nvidia:   hipcc also targets Nvidia (HIP_PLATFORM=nvidia), or swap the hip
//           prefix for cuda in the API calls and build with nvcc.

#include <hip/hip_runtime.h>

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

constexpr int BLOCK = 256;

// 1a. Read, add, write: two threads that read the same old value both
// write old + 1, and one increment is lost. `volatile` stops the compiler
// from merging the updates, so the race is visible rather than optimized.
__global__ void countRacy(volatile unsigned *count, size_t n) {
  size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
  if (i < n) *count = *count + 1;
}

// 1b. Correct: the read-add-write happens as one indivisible operation.
__global__ void countAtomic(unsigned *count, size_t n) {
  size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
  if (i < n) atomicAdd(count, 1u);
}

// 1c. Count inside the block first (shared-memory tree), then one atomic.
__global__ void countPerBlock(unsigned *count, size_t n) {
  __shared__ unsigned s[BLOCK];
  size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
  s[threadIdx.x] = i < n ? 1u : 0u;
  __syncthreads();
  for (unsigned stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) s[threadIdx.x] += s[threadIdx.x + stride];
    __syncthreads();
  }
  if (threadIdx.x == 0) atomicAdd(count, s[0]);
}

// 2a. Histogram, every update an atomic on global memory.
__global__ void histGlobal(const unsigned char *in, size_t n,
                           unsigned *bins) {
  for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n;
       i += (size_t)gridDim.x * blockDim.x)
    atomicAdd(&bins[in[i]], 1u);
}

// 2b. Histogram, privatized: each block counts into its own 256 bins in
// shared memory (fast atomics, contention only inside the block), then
// adds its bins to the global ones once.
__global__ void histShared(const unsigned char *in, size_t n,
                           unsigned *bins) {
  __shared__ unsigned local[256];
  for (int b = threadIdx.x; b < 256; b += blockDim.x) local[b] = 0;
  __syncthreads();
  for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n;
       i += (size_t)gridDim.x * blockDim.x)
    atomicAdd(&local[in[i]], 1u);
  __syncthreads();
  for (int b = threadIdx.x; b < 256; b += blockDim.x)
    if (local[b]) atomicAdd(&bins[b], local[b]);
}

// 3. Float sum through atomics: correct to rounding, order not fixed.
__global__ void sumFloatAtomic(const float *in, size_t n, float *sum) {
  __shared__ float s[BLOCK];
  size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
  s[threadIdx.x] = i < n ? in[i] : 0.0f;
  __syncthreads();
  for (unsigned stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) s[threadIdx.x] += s[threadIdx.x + stride];
    __syncthreads();
  }
  if (threadIdx.x == 0) atomicAdd(sum, s[0]);
}

static float elapsedMs(hipEvent_t a, hipEvent_t b) {
  float ms = 0.0f;
  HIP_CHECK(hipEventElapsedTime(&ms, a, b));
  return ms;
}

int main() {
  hipDeviceProp_t prop;
  HIP_CHECK(hipGetDeviceProperties(&prop, 0));
  std::printf("Device: %s\n\n", prop.name);
  hipEvent_t start, stop;
  HIP_CHECK(hipEventCreate(&start));
  HIP_CHECK(hipEventCreate(&stop));

  // ---- 1. One counter ----
  const size_t nCount = 16u << 20;
  const unsigned blocks = (unsigned)((nCount + BLOCK - 1) / BLOCK);
  unsigned *dCount;
  HIP_CHECK(hipMalloc(&dCount, sizeof(unsigned)));
  std::printf("1. Counting %zu elements into one counter\n", nCount);
  std::printf("   %-28s %10s %12s\n", "method", "time ms", "count");
  auto runCount = [&](const char *name, auto kernel, auto *arg) {
    HIP_CHECK(hipMemset(dCount, 0, sizeof(unsigned)));
    HIP_CHECK(hipEventRecord(start, 0));
    kernel<<<blocks, BLOCK>>>(arg, nCount);
    HIP_CHECK(hipEventRecord(stop, 0));
    HIP_CHECK(hipEventSynchronize(stop));
    HIP_CHECK(hipGetLastError());
    unsigned c = 0;
    HIP_CHECK(hipMemcpy(&c, dCount, sizeof(c), hipMemcpyDeviceToHost));
    std::printf("   %-28s %10.3f %12u%s\n", name, elapsedMs(start, stop), c,
                c == nCount ? "" : "   <- wrong");
  };
  countAtomic<<<blocks, BLOCK>>>(dCount, nCount);  // warm-up
  runCount("plain += (data race)", countRacy, (volatile unsigned *)dCount);
  runCount("atomicAdd(count, 1)", countAtomic, dCount);
  runCount("block sum + 1 atomic/block", countPerBlock, dCount);

  // ---- 2. Histogram ----
  const size_t nHist = 64u << 20;
  std::vector<unsigned char> hUniform(nHist), hSame(nHist, 42);
  unsigned x = 12345;
  for (size_t i = 0; i < nHist; ++i) {
    x = x * 1664525u + 1013904223u;
    hUniform[i] = (unsigned char)(x >> 24);
  }
  unsigned char *dIn;
  unsigned *dBins;
  HIP_CHECK(hipMalloc(&dIn, nHist));
  HIP_CHECK(hipMalloc(&dBins, 256 * sizeof(unsigned)));
  const unsigned histBlocks = prop.multiProcessorCount * 8;
  std::printf("\n2. 256-bin histogram of %zu MB\n", nHist >> 20);
  std::printf("   %-22s %-12s %10s %10s %8s\n", "input", "atomics in",
              "time ms", "GB/s", "check");
  for (int input = 0; input < 2; ++input) {
    const std::vector<unsigned char> &h = input ? hSame : hUniform;
    HIP_CHECK(hipMemcpy(dIn, h.data(), nHist, hipMemcpyHostToDevice));
    for (int shared = 0; shared < 2; ++shared) {
      auto launch = [&] {
        if (shared) histShared<<<histBlocks, BLOCK>>>(dIn, nHist, dBins);
        else histGlobal<<<histBlocks, BLOCK>>>(dIn, nHist, dBins);
      };
      HIP_CHECK(hipMemset(dBins, 0, 256 * sizeof(unsigned)));
      launch();  // warm-up
      HIP_CHECK(hipMemset(dBins, 0, 256 * sizeof(unsigned)));
      HIP_CHECK(hipEventRecord(start, 0));
      launch();
      HIP_CHECK(hipEventRecord(stop, 0));
      HIP_CHECK(hipEventSynchronize(stop));
      HIP_CHECK(hipGetLastError());
      unsigned bins[256];
      HIP_CHECK(hipMemcpy(bins, dBins, sizeof(bins), hipMemcpyDeviceToHost));
      size_t total = 0;
      for (unsigned b : bins) total += b;
      float ms = elapsedMs(start, stop);
      std::printf("   %-22s %-12s %10.3f %10.1f %8s\n",
                  input ? "every byte = 42" : "uniform random bytes",
                  shared ? "shared mem" : "global mem", ms,
                  nHist / (ms * 1e6), total == nHist ? "ok" : "WRONG");
    }
  }

  // ---- 3. Float atomics and order ----
  const size_t nF = 16u << 20;
  std::vector<float> hF(nF);
  for (size_t i = 0; i < nF; ++i) {
    x = x * 1664525u + 1013904223u;
    hF[i] = (float)(x >> 8) / 16777216.0f;  // [0, 1), full mantissa
  }
  float *dF, *dSum;
  HIP_CHECK(hipMalloc(&dF, nF * sizeof(float)));
  HIP_CHECK(hipMalloc(&dSum, sizeof(float)));
  HIP_CHECK(hipMemcpy(dF, hF.data(), nF * sizeof(float),
                      hipMemcpyHostToDevice));
  std::printf("\n3. The same float sum, 20 runs, block partials combined "
              "with atomicAdd\n");
  float seen[20];
  int counts[20], distinct = 0;
  for (int r = 0; r < 20; ++r) {
    HIP_CHECK(hipMemset(dSum, 0, sizeof(float)));
    sumFloatAtomic<<<(unsigned)((nF + BLOCK - 1) / BLOCK), BLOCK>>>(dF, nF,
                                                                   dSum);
    float s = 0.0f;
    HIP_CHECK(hipMemcpy(&s, dSum, sizeof(s), hipMemcpyDeviceToHost));
    int k = 0;
    while (k < distinct && seen[k] != s) ++k;
    if (k == distinct) { seen[distinct] = s; counts[distinct++] = 0; }
    ++counts[k];
  }
  for (int k = 0; k < distinct; ++k)
    std::printf("   %.1f  (%d of 20 runs)\n", seen[k], counts[k]);
  double exact = 0.0;
  for (float v : hF) exact += v;
  std::printf("   exact (double): %.3f\n", exact);
  if (distinct == 1)
    std::printf("   All runs agreed this time; the order is still not "
                "guaranteed. Run it again.\n");

  HIP_CHECK(hipFree(dCount));
  HIP_CHECK(hipFree(dIn));
  HIP_CHECK(hipFree(dBins));
  HIP_CHECK(hipFree(dF));
  HIP_CHECK(hipFree(dSum));
  HIP_CHECK(hipEventDestroy(start));
  HIP_CHECK(hipEventDestroy(stop));
  return 0;
}
