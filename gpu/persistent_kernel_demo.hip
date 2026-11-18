// persistent_kernel_demo.hip
// Companion example for: topnotchnote.com/gpu/persistent_kernels.html
//
// A queue of work items, each processed by one block of 128 threads. Every
// block first runs a prologue (building a lookup table in shared memory),
// then processes its item(s). Three ways to launch it:
//   1. one block per item: the grid size follows the problem size, and the
//      hardware scheduler hands out blocks as compute units free up
//   2. persistent, static: just enough blocks to fill the GPU once; block b
//      takes items b, b + G, b + 2G, ... (a grid-stride loop over items)
//   3. persistent, dynamic: same block count; each block takes the next item
//      from a global counter with atomicAdd until the queue is empty
// Run on two workloads: every item the same cost, and an irregular one where
// 1 item in 64 costs 64 times more. The prologue cost is a parameter.
//
// Compile:  hipcc -O2 -o main persistent_kernel_demo.hip
// Nvidia:   hipcc also targets Nvidia (HIP_PLATFORM=nvidia), or swap the hip
//           prefix for cuda in the API calls and build with nvcc.

#include <hip/hip_runtime.h>

#include <algorithm>
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

constexpr int kThreads = 128, kTable = 1024;

// Per-block setup that does not depend on which item is processed.
__device__ void prologue(float* table, int prologue_iters) {
  for (int i = threadIdx.x; i < kTable; i += kThreads) {
    float v = i * 0.001f;
    for (int k = 0; k < prologue_iters; ++k) v = __sinf(v) + 0.5f;
    table[i] = v;
  }
  __syncthreads();
}

// One item: `cost` rounds of table lookups and arithmetic per thread.
__device__ void process(int item, int cost, const float* table, float* out) {
  float acc = 0;
  for (int r = 0; r < cost; ++r)
    acc += table[(threadIdx.x * 7 + r * 13 + item) % kTable] * 1.0001f;
  if (threadIdx.x == 0) out[item] = acc;
}

__global__ void one_block_per_item(const int* cost, float* out, int prologue_iters) {
  __shared__ float table[kTable];
  prologue(table, prologue_iters);
  process(blockIdx.x, cost[blockIdx.x], table, out);
}

__global__ void persistent_static(const int* cost, float* out, int n, int prologue_iters) {
  __shared__ float table[kTable];
  prologue(table, prologue_iters);                 // once per block, not per item
  for (int item = blockIdx.x; item < n; item += gridDim.x)
    process(item, cost[item], table, out);
}

__global__ void persistent_dynamic(const int* cost, float* out, int n, int prologue_iters,
                                   int* next) {
  __shared__ float table[kTable];
  __shared__ int item;
  prologue(table, prologue_iters);
  while (true) {
    if (threadIdx.x == 0) item = atomicAdd(next, 1);   // take the next item
    __syncthreads();
    if (item >= n) break;
    process(item, cost[item], table, out);
    __syncthreads();                                  // everyone done with `item`
  }
}

template <typename F>
static float median_ms(F launch) {
  hipEvent_t a, b;
  HIP_CHECK(hipEventCreate(&a));
  HIP_CHECK(hipEventCreate(&b));
  launch();
  HIP_CHECK(hipGetLastError());
  std::vector<float> t;
  for (int r = 0; r < 7; ++r) {
    HIP_CHECK(hipEventRecord(a));
    launch();
    HIP_CHECK(hipEventRecord(b));
    HIP_CHECK(hipEventSynchronize(b));
    float ms;
    HIP_CHECK(hipEventElapsedTime(&ms, a, b));
    t.push_back(ms);
  }
  HIP_CHECK(hipEventDestroy(a));
  HIP_CHECK(hipEventDestroy(b));
  std::sort(t.begin(), t.end());
  return t[3];
}

int main() {
  const int n = 32768;
  int cus = 0, per_cu = 0;
  HIP_CHECK(hipDeviceGetAttribute(&cus, hipDeviceAttributeMultiprocessorCount, 0));
  HIP_CHECK(hipOccupancyMaxActiveBlocksPerMultiprocessor(&per_cu, persistent_dynamic, kThreads, 0));
  const int G = cus * per_cu;                       // blocks that fit at once
  std::printf("%d compute units x %d resident blocks = %d persistent blocks, %d items\n\n",
              cus, per_cu, G, n);

  int *cost, *next;
  float* out;
  HIP_CHECK(hipMalloc(&cost, n * sizeof(int)));
  HIP_CHECK(hipMalloc(&out, n * sizeof(float)));
  HIP_CHECK(hipMalloc(&next, sizeof(int)));

  for (int irregular = 0; irregular < 2; ++irregular) {
    std::vector<int> h(n);
    // Irregular: 1 item in 64 costs 64x; the rest cost 1x. Same average cost
    // would need 1x = 32 rounds; keep the base at 32 rounds either way.
    for (int i = 0; i < n; ++i) h[i] = (irregular && i % 64 == 0) ? 32 * 64 : 32;
    HIP_CHECK(hipMemcpy(cost, h.data(), n * sizeof(int), hipMemcpyHostToDevice));
    std::printf("%s items\n", irregular ? "irregular (1 in 64 costs 64x)" : "uniform");
    std::printf("  %-14s %18s %18s %18s\n", "prologue", "one block/item", "persistent static",
                "persistent dynamic");
    for (int p : {0, 16, 64}) {
      float t1 = median_ms([&] { one_block_per_item<<<n, kThreads>>>(cost, out, p); });
      float t2 = median_ms([&] { persistent_static<<<G, kThreads>>>(cost, out, n, p); });
      float t3 = median_ms([&] {
        HIP_CHECK(hipMemsetAsync(next, 0, sizeof(int)));
        persistent_dynamic<<<G, kThreads>>>(cost, out, n, p, next);
      });
      std::printf("  %-14d %15.2f ms %15.2f ms %15.2f ms\n", p, t1, t2, t3);
    }
    std::printf("\n");
  }
  HIP_CHECK(hipFree(cost));
  HIP_CHECK(hipFree(out));
  HIP_CHECK(hipFree(next));
  return 0;
}
