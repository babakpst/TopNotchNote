// synchronization_demo.hip
// Companion example for: topnotchnote.com/gpu/synchronization.html
//
// Synchronization on a GPU is about ordering and visibility: making sure
// one thread's writes are seen by another before it reads them.
//   1. __syncthreads(): a block reverses an array through shared memory,
//      with and without the barrier between the write and the read. Without
//      it, threads read slots other warps have not written yet.
//   2. Blocks cannot wait for each other inside a normal kernel, so a
//      grid-wide sum needs one of three ways out. All three are timed:
//        a. a kernel boundary: partial sums, then a second kernel
//        b. one kernel, "last block done": each block writes its partial,
//           __threadfence(), then bumps an atomic counter; the block that
//           sees gridDim.x - 1 sums the partials
//        c. a cooperative launch and grid.sync()
//   3. A cooperative launch with more blocks than can be resident at once
//      is refused up front instead of deadlocking.
//
// Compile:  hipcc -O2 -o main synchronization_demo.hip
// Nvidia:   hipcc also targets Nvidia (HIP_PLATFORM=nvidia), or swap the hip
//           prefix for cuda in the API calls and build with nvcc
//           (grid.sync() needs compute capability 6.0 or newer).

#include <hip/hip_runtime.h>
#include <hip/hip_cooperative_groups.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <vector>

namespace cg = cooperative_groups;

#define HIP_CHECK(call)                                                   \
  do {                                                                    \
    hipError_t err_ = (call);                                             \
    if (err_ != hipSuccess) {                                             \
      std::fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__,             \
                   hipGetErrorString(err_));                              \
      std::exit(1);                                                       \
    }                                                                     \
  } while (0)

constexpr int kThreads = 256;

// ---- 1. the block barrier ------------------------------------------------

// Each block reverses its 256-element slice through shared memory. The
// write and the read touch different slots, written by different warps.
template <bool kBarrier>
__global__ void reverse_block(const int* in, int* out) {
  __shared__ int s[kThreads];
  int t = threadIdx.x;
  int base = blockIdx.x * kThreads;
  s[t] = in[base + t];
  if (kBarrier) __syncthreads();
  out[base + t] = s[kThreads - 1 - t];
}

// ---- 2. block-level sum, shared by the three grid sums -------------------

__device__ float block_sum(float v) {
  __shared__ float s[kThreads];
  s[threadIdx.x] = v;
  __syncthreads();
  for (int stride = kThreads / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) s[threadIdx.x] += s[threadIdx.x + stride];
    __syncthreads();
  }
  float r = s[0];
  __syncthreads();  // s is reused if this is called again
  return r;
}

__device__ float grid_stride_load(const float* x, int n) {
  float v = 0.f;
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
       i += gridDim.x * blockDim.x)
    v += x[i];
  return v;
}

// a. kernel boundary
__global__ void partial_sums(const float* x, int n, float* partial) {
  float s = block_sum(grid_stride_load(x, n));
  if (threadIdx.x == 0) partial[blockIdx.x] = s;
}
__global__ void final_sum(const float* partial, int nblocks, float* out) {
  float v = 0.f;
  for (int i = threadIdx.x; i < nblocks; i += blockDim.x) v += partial[i];
  float s = block_sum(v);
  if (threadIdx.x == 0) *out = s;
}

// b. last block done (the pattern from the CUDA guide's fence section)
__device__ unsigned int g_count = 0;
__global__ void sum_last_block(const float* x, int n, float* partial,
                               float* out) {
  __shared__ bool am_last;
  float s = block_sum(grid_stride_load(x, n));
  if (threadIdx.x == 0) {
    partial[blockIdx.x] = s;
    __threadfence();  // the partial is ordered before the counter bump
    unsigned int ticket = atomicInc(&g_count, gridDim.x);
    am_last = (ticket == gridDim.x - 1);
  }
  __syncthreads();
  if (am_last) {
    float v = 0.f;
    for (int i = threadIdx.x; i < gridDim.x; i += blockDim.x)
      v += ((volatile float*)partial)[i];
    float t = block_sum(v);
    if (threadIdx.x == 0) {
      *out = t;
      g_count = 0;  // ready for the next launch
    }
  }
}

// c. cooperative launch and a grid-wide barrier
__global__ void sum_grid_sync(const float* x, int n, float* partial,
                              float* out) {
  cg::grid_group grid = cg::this_grid();
  float s = block_sum(grid_stride_load(x, n));
  if (threadIdx.x == 0) partial[blockIdx.x] = s;
  grid.sync();  // every block's partial is written and visible
  if (blockIdx.x == 0) {
    float v = 0.f;
    for (int i = threadIdx.x; i < gridDim.x; i += blockDim.x) v += partial[i];
    float t = block_sum(v);
    if (threadIdx.x == 0) *out = t;
  }
}

template <typename F>
float median_ms(F launch, int reps) {
  hipEvent_t a, b;
  HIP_CHECK(hipEventCreate(&a));
  HIP_CHECK(hipEventCreate(&b));
  for (int i = 0; i < 3; ++i) launch();  // warm-up
  HIP_CHECK(hipDeviceSynchronize());
  std::vector<float> t(reps);
  for (int i = 0; i < reps; ++i) {
    HIP_CHECK(hipEventRecord(a));
    launch();
    HIP_CHECK(hipEventRecord(b));
    HIP_CHECK(hipEventSynchronize(b));
    HIP_CHECK(hipEventElapsedTime(&t[i], a, b));
  }
  std::sort(t.begin(), t.end());
  HIP_CHECK(hipEventDestroy(a));
  HIP_CHECK(hipEventDestroy(b));
  return t[reps / 2];
}

int main() {
  int dev = 0;
  hipDeviceProp_t prop;
  HIP_CHECK(hipGetDeviceProperties(&prop, dev));
  std::printf("device: %s, %d SMs/CUs, warpSize %d\n", prop.name,
              prop.multiProcessorCount, prop.warpSize);

  // ---- 1 ----
  const int nblk = 4096, nr = nblk * kThreads;
  std::vector<int> h_in(nr), h_out(nr);
  for (int i = 0; i < nr; ++i) h_in[i] = i;
  int *d_in, *d_out;
  HIP_CHECK(hipMalloc(&d_in, nr * sizeof(int)));
  HIP_CHECK(hipMalloc(&d_out, nr * sizeof(int)));
  HIP_CHECK(hipMemcpy(d_in, h_in.data(), nr * sizeof(int),
                      hipMemcpyHostToDevice));
  for (int with = 0; with < 2; ++with) {
    long wrong = 0;
    for (int trial = 0; trial < 10; ++trial) {
      HIP_CHECK(hipMemset(d_out, 0xff, nr * sizeof(int)));
      if (with) reverse_block<true><<<nblk, kThreads>>>(d_in, d_out);
      else reverse_block<false><<<nblk, kThreads>>>(d_in, d_out);
      HIP_CHECK(hipGetLastError());
      HIP_CHECK(hipMemcpy(h_out.data(), d_out, nr * sizeof(int),
                          hipMemcpyDeviceToHost));
      for (int b = 0; b < nblk; ++b)
        for (int t = 0; t < kThreads; ++t)
          if (h_out[b * kThreads + t] != b * kThreads + kThreads - 1 - t)
            ++wrong;
    }
    std::printf("reverse %s __syncthreads(): %ld of %d values wrong "
                "(10 runs)\n", with ? "with   " : "without", wrong, nr * 10);
  }

  // ---- 2 ----
  int per_sm = 0;
  HIP_CHECK(hipOccupancyMaxActiveBlocksPerMultiprocessor(
      &per_sm, sum_grid_sync, kThreads, 0));
  int coop_max = per_sm * prop.multiProcessorCount;
  int grid = coop_max;  // the same grid for all three, so they compare
  int coop = 0;
  HIP_CHECK(hipDeviceGetAttribute(&coop, hipDeviceAttributeCooperativeLaunch,
                                  dev));
  const int nmax = 1 << 24;
  std::vector<float> hx(nmax);
  for (int i = 0; i < nmax; ++i) hx[i] = (float)(i % 4) * 0.25f;  // exact
  float *x, *partial, *out;
  HIP_CHECK(hipMalloc(&x, nmax * sizeof(float)));
  HIP_CHECK(hipMemcpy(x, hx.data(), nmax * sizeof(float),
                      hipMemcpyHostToDevice));
  HIP_CHECK(hipMalloc(&partial, coop_max * sizeof(float)));
  HIP_CHECK(hipMalloc(&out, sizeof(float)));

  for (int n : {1 << 16, 1 << 24}) {
    double ref = n * 0.375;
    std::printf("\nsum of %d floats, %d blocks of %d (%d resident per "
                "SM/CU)\n", n, grid, kThreads, per_sm);
    auto check = [&](const char* name, float ms) {
      float h;
      HIP_CHECK(hipMemcpy(&h, out, sizeof(float), hipMemcpyDeviceToHost));
      std::printf("  %-36s %8.1f us  sum %s\n", name, ms * 1000.f,
                  h == ref ? "correct" : "WRONG");
    };
    float ms = median_ms([&] {
      partial_sums<<<grid, kThreads>>>(x, n, partial);
      final_sum<<<1, kThreads>>>(partial, grid, out);
    }, 101);
    check("a. two kernels", ms);
    ms = median_ms([&] {
      sum_last_block<<<grid, kThreads>>>(x, n, partial, out);
    }, 101);
    check("b. one kernel, last block done", ms);
    if (coop) {
      void* args[] = {&x, &n, &partial, &out};
      ms = median_ms([&] {
        HIP_CHECK(hipLaunchCooperativeKernel((void*)sum_grid_sync,
                                             dim3(grid), dim3(kThreads), args,
                                             0, 0));
      }, 101);
      check("c. cooperative launch, grid.sync()", ms);
    }
  }

  // ---- 3 ----
  if (coop) {
    int n = 1 << 16;
    void* args[] = {&x, &n, &partial, &out};
    hipError_t e = hipLaunchCooperativeKernel(
        (void*)sum_grid_sync, dim3(coop_max + 1), dim3(kThreads), args, 0, 0);
    std::printf("\ncooperative launch of %d blocks (limit %d): %s\n",
                coop_max + 1, coop_max, hipGetErrorString(e));
    (void)hipGetLastError();
  } else {
    std::printf("\nthis device does not support cooperative launch\n");
  }

  HIP_CHECK(hipFree(d_in));
  HIP_CHECK(hipFree(d_out));
  HIP_CHECK(hipFree(x));
  HIP_CHECK(hipFree(partial));
  HIP_CHECK(hipFree(out));
  return 0;
}
