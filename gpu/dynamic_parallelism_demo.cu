// dynamic_parallelism_demo.cu
// Companion example for: topnotchnote.com/gpu/dynamic_parallelism.html
//
// Dynamic parallelism lets a kernel launch kernels. This sample is CUDA,
// not HIP, because HIP on ROCm does not support it (see the page).
//
// The workload is the one dynamic parallelism is meant for: work whose size
// is only known on the GPU. Sum each row of a ragged array where most rows
// are short and a few are very long.
//   1. flat:    one thread per row. Simple, but the long rows leave one
//               thread grinding while the rest of the GPU idles.
//   2. dynamic: one thread per row; a thread that finds a long row launches
//               a child grid for it, from device code.
//   3. host:    the "no dynamic parallelism" answer. The CPU already knows
//               the row lengths, so it splits the rows into short and long
//               and launches one kernel for each.
// Plus the cost of a launch itself: an empty child launched from the GPU
// against an empty kernel launched from the CPU.
//
// Compile:  nvcc -O2 -rdc=true -arch=sm_61 -o main dynamic_parallelism_demo.cu -lcudadevrt
//           (set -arch to your GPU; -rdc=true and -lcudadevrt are required)

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <vector>

#define CUDA_CHECK(call)                                                  \
  do {                                                                    \
    cudaError_t err_ = (call);                                            \
    if (err_ != cudaSuccess) {                                            \
      std::fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__,             \
                   cudaGetErrorString(err_));                             \
      std::exit(1);                                                       \
    }                                                                     \
  } while (0)

constexpr int kLong = 4096;  // rows longer than this get their own grid

__device__ float serial_sum(const float* v, int len) {
  float s = 0.f;
  for (int i = 0; i < len; ++i) s += v[i];
  return s;
}

// A block-per-row sum; used as the child grid and by the host version.
__global__ void long_row_sum(const float* v, int len, float* out) {
  __shared__ float s[256];
  float acc = 0.f;
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < len;
       i += gridDim.x * blockDim.x)
    acc += v[i];
  s[threadIdx.x] = acc;
  __syncthreads();
  for (int st = 128; st > 0; st >>= 1) {
    if (threadIdx.x < st) s[threadIdx.x] += s[threadIdx.x + st];
    __syncthreads();
  }
  if (threadIdx.x == 0) atomicAdd(out, s[0]);
}

// 1. flat
__global__ void flat(const float* v, const int* off, int rows, float* out) {
  int r = blockIdx.x * blockDim.x + threadIdx.x;
  if (r < rows) out[r] = serial_sum(v + off[r], off[r + 1] - off[r]);
}

// 2. dynamic
__global__ void dynamic(const float* v, const int* off, int rows, float* out) {
  int r = blockIdx.x * blockDim.x + threadIdx.x;
  if (r >= rows) return;
  int len = off[r + 1] - off[r];
  if (len <= kLong) {
    out[r] = serial_sum(v + off[r], len);
  } else {
    out[r] = 0.f;  // visible to the child: it is written before the launch
    long_row_sum<<<16, 256, 0, cudaStreamFireAndForget>>>(v + off[r], len,
                                                          out + r);
  }
}

// 3. host split: short rows by index list, long rows one launch each
__global__ void short_rows(const float* v, const int* off, const int* idx,
                           int n, float* out) {
  int k = blockIdx.x * blockDim.x + threadIdx.x;
  if (k < n) {
    int r = idx[k];
    out[r] = serial_sum(v + off[r], off[r + 1] - off[r]);
  }
}

// launch cost
__global__ void empty() {}
__global__ void launch_children(int n) {
  for (int i = 0; i < n; ++i) empty<<<1, 1, 0, cudaStreamFireAndForget>>>();
}

template <typename F>
float median_ms(F launch) {
  cudaEvent_t a, b;
  CUDA_CHECK(cudaEventCreate(&a));
  CUDA_CHECK(cudaEventCreate(&b));
  launch();
  CUDA_CHECK(cudaDeviceSynchronize());
  std::vector<float> t(11);
  for (auto& ti : t) {
    CUDA_CHECK(cudaEventRecord(a));
    launch();
    CUDA_CHECK(cudaEventRecord(b));
    CUDA_CHECK(cudaEventSynchronize(b));
    CUDA_CHECK(cudaEventElapsedTime(&ti, a, b));
  }
  std::sort(t.begin(), t.end());
  return t[t.size() / 2];
}

int main() {
  cudaDeviceProp prop;
  CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
  std::printf("device: %s (compute capability %d.%d)\n", prop.name,
              prop.major, prop.minor);

  // 100,000 rows of 32 values, plus 20 rows of 1M values spread through.
  const int rows = 100000;
  std::vector<int> off(rows + 1, 0);
  std::vector<int> short_idx, long_idx;
  for (int r = 0; r < rows; ++r) {
    int len = (r % 5000 == 17) ? (1 << 20) : 32;
    off[r + 1] = off[r] + len;
    (len > kLong ? long_idx : short_idx).push_back(r);
  }
  int total = off[rows];
  std::vector<float> hv(total);
  for (int i = 0; i < total; ++i) hv[i] = (float)(i % 3);  // exact sums
  std::printf("%d rows, %zu of them long, %d values (%.0f MB)\n", rows,
              long_idx.size(), total, total * 4.0 / 1e6);

  float *v, *out;
  int *doff, *didx;
  CUDA_CHECK(cudaMalloc(&v, total * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&out, rows * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&doff, (rows + 1) * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&didx, short_idx.size() * sizeof(int)));
  CUDA_CHECK(cudaMemcpy(v, hv.data(), total * sizeof(float),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(doff, off.data(), (rows + 1) * sizeof(int),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(didx, short_idx.data(), short_idx.size() * sizeof(int),
                        cudaMemcpyHostToDevice));

  std::vector<double> ref(rows);
  for (int r = 0; r < rows; ++r) {
    double s = 0;
    for (int i = off[r]; i < off[r + 1]; ++i) s += hv[i];
    ref[r] = s;
  }
  auto check = [&](const char* name, float ms) {
    std::vector<float> h(rows);
    CUDA_CHECK(cudaMemcpy(h.data(), out, rows * sizeof(float),
                          cudaMemcpyDeviceToHost));
    int bad = 0;
    for (int r = 0; r < rows; ++r) bad += h[r] != ref[r];
    std::printf("  %-44s %8.2f ms   %s\n", name, ms,
                bad ? "WRONG" : "all rows correct");
  };

  int tpb = 128, nb = (rows + tpb - 1) / tpb;
  check("1. flat, one thread per row",
        median_ms([&] { flat<<<nb, tpb>>>(v, doff, rows, out); }));
  check("2. dynamic, long rows launch a child grid",
        median_ms([&] { dynamic<<<nb, tpb>>>(v, doff, rows, out); }));
  int ns = (int)short_idx.size();
  check("3. host split, 1 + one launch per long row", median_ms([&] {
          short_rows<<<(ns + tpb - 1) / tpb, tpb>>>(v, doff, didx, ns, out);
          for (int r : long_idx) {
            cudaMemsetAsync(out + r, 0, sizeof(float));
            long_row_sum<<<16, 256>>>(v + off[r], off[r + 1] - off[r],
                                      out + r);
          }
        }));
  CUDA_CHECK(cudaGetLastError());

  // launch cost
  const int nl = 1000;
  float t_dev = median_ms([&] { launch_children<<<1, 1>>>(nl); });
  float t_host = median_ms([&] {
    for (int i = 0; i < nl; ++i) empty<<<1, 1>>>();
  });
  std::printf("\n%d empty kernels: launched from one GPU thread %.2f ms "
              "(%.1f us each), from the CPU %.2f ms (%.1f us each)\n", nl,
              t_dev, t_dev * 1000 / nl, t_host, t_host * 1000 / nl);

  CUDA_CHECK(cudaFree(v));
  CUDA_CHECK(cudaFree(out));
  CUDA_CHECK(cudaFree(doff));
  CUDA_CHECK(cudaFree(didx));
  return 0;
}
