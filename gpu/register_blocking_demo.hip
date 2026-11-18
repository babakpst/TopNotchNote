// register_blocking_demo.hip
// Companion example for: topnotchnote.com/gpu/register_blocking.html
//
// One shared-memory tiled SGEMM, C = A * B (row-major, N x N), with the
// per-thread output tile as a template parameter: each thread computes a
// TM x TN patch of C held in registers. Every block has 16 x 16 = 256
// threads, so the block tile is (16*TM) x (16*TN) and grows with the
// per-thread tile. The program prints GFLOP/s for 1x1, 2x2, 4x4 and 8x8, and
// checks every result against the 1x1 kernel.
//
// To see the register cost of each variant:
//   AMD:    hipcc -O2 --save-temps ... and read .vgpr_count in the .s file,
//           or run with rocprofv3 --kernel-trace and read the VGPR column
//   Nvidia: nvcc -Xptxas -v ...   prints "Used N registers" per kernel
//
// Compile:  hipcc -O2 -o main register_blocking_demo.hip
// Nvidia:   hipcc also targets Nvidia (HIP_PLATFORM=nvidia), or swap the hip
//           prefix for cuda in the API calls and build with nvcc.

#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
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

constexpr int kThreadsX = 16, kThreadsY = 16;   // 256 threads per block
constexpr int BK = 8;                           // depth of each shared tile

template <int TM, int TN>
__global__ void __launch_bounds__(256) sgemm(const float* A, const float* B, float* C, int N) {
  constexpr int BM = kThreadsY * TM, BN = kThreadsX * TN;
  __shared__ float As[BK][BM];   // stored transposed: As[k][m]
  __shared__ float Bs[BK][BN];

  const int tx = threadIdx.x, ty = threadIdx.y, tid = ty * kThreadsX + tx;
  const int row0 = blockIdx.y * BM, col0 = blockIdx.x * BN;

  float acc[TM][TN] = {};        // the register tile
  float a[TM], b[TN];

  for (int k0 = 0; k0 < N; k0 += BK) {
    // Cooperative loads: 256 threads fill BM*BK and BK*BN values.
    for (int i = tid; i < BM * BK; i += 256) {
      int m = i / BK, k = i % BK;
      As[k][m] = A[(row0 + m) * N + k0 + k];
    }
    for (int i = tid; i < BK * BN; i += 256) {
      int k = i / BN, n = i % BN;
      Bs[k][n] = B[(k0 + k) * N + col0 + n];
    }
    __syncthreads();

    for (int k = 0; k < BK; ++k) {
      // TM + TN shared-memory reads ...
#pragma unroll
      for (int i = 0; i < TM; ++i) a[i] = As[k][ty + i * kThreadsY];
#pragma unroll
      for (int j = 0; j < TN; ++j) b[j] = Bs[k][tx + j * kThreadsX];
      // ... feed TM * TN multiply-adds (an outer product).
#pragma unroll
      for (int i = 0; i < TM; ++i)
#pragma unroll
        for (int j = 0; j < TN; ++j) acc[i][j] += a[i] * b[j];
    }
    __syncthreads();
  }
#pragma unroll
  for (int i = 0; i < TM; ++i)
#pragma unroll
    for (int j = 0; j < TN; ++j)
      C[(row0 + ty + i * kThreadsY) * N + col0 + tx + j * kThreadsX] = acc[i][j];
}

static int g_blocks_per_cu = 0;   // filled by run(): resident blocks per CU/SM

template <int TM, int TN>
static double run(const float* A, const float* B, float* C, int N, float* ms_out) {
  dim3 block(kThreadsX, kThreadsY);
  dim3 grid(N / (kThreadsX * TN), N / (kThreadsY * TM));
  sgemm<TM, TN><<<grid, block>>>(A, B, C, N);   // warm-up
  HIP_CHECK(hipGetLastError());
  hipEvent_t s, e;
  HIP_CHECK(hipEventCreate(&s));
  HIP_CHECK(hipEventCreate(&e));
  std::vector<float> t;
  for (int r = 0; r < 7; ++r) {
    HIP_CHECK(hipEventRecord(s));
    sgemm<TM, TN><<<grid, block>>>(A, B, C, N);
    HIP_CHECK(hipEventRecord(e));
    HIP_CHECK(hipEventSynchronize(e));
    float ms;
    HIP_CHECK(hipEventElapsedTime(&ms, s, e));
    t.push_back(ms);
  }
  std::sort(t.begin(), t.end());
  *ms_out = t[3];
  int blocks_per_cu = 0;
  HIP_CHECK(hipOccupancyMaxActiveBlocksPerMultiprocessor(&blocks_per_cu, sgemm<TM, TN>, 256, 0));
  g_blocks_per_cu = blocks_per_cu;
  return 2.0 * N * N * (double)N / (t[3] * 1e6);
}

int main() {
  const int N = 2048;           // a multiple of every block tile (16 .. 128)
  const size_t bytes = (size_t)N * N * sizeof(float);
  std::vector<float> h(N * (size_t)N);
  for (size_t i = 0; i < h.size(); ++i) h[i] = (float)((i * 7) % 13) / 13.0f - 0.5f;
  float *A, *B, *C;
  HIP_CHECK(hipMalloc(&A, bytes));
  HIP_CHECK(hipMalloc(&B, bytes));
  HIP_CHECK(hipMalloc(&C, bytes));
  HIP_CHECK(hipMemcpy(A, h.data(), bytes, hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(B, h.data(), bytes, hipMemcpyHostToDevice));

  std::vector<float> ref(h.size()), out(h.size());
  float ms;
  std::printf("%-8s %-12s %10s %10s %14s %14s\n", "tile", "block tile", "ms", "GFLOP/s",
              "blocks per CU", "max |diff|");
  auto report = [&](const char* name, const char* bt, double gf, bool is_ref) {
    HIP_CHECK(hipMemcpy(is_ref ? ref.data() : out.data(), C, bytes, hipMemcpyDeviceToHost));
    double d = 0;
    if (!is_ref)
      for (size_t i = 0; i < ref.size(); ++i) d = std::max(d, (double)std::fabs(ref[i] - out[i]));
    std::printf("%-8s %-12s %10.2f %10.1f %14d %14.2e\n", name, bt, ms, gf, g_blocks_per_cu, d);
  };
  report("1x1", "16x16", run<1, 1>(A, B, C, N, &ms), true);
  report("2x2", "32x32", run<2, 2>(A, B, C, N, &ms), false);
  report("4x4", "64x64", run<4, 4>(A, B, C, N, &ms), false);
  report("8x8", "128x128", run<8, 8>(A, B, C, N, &ms), false);

  HIP_CHECK(hipFree(A));
  HIP_CHECK(hipFree(B));
  HIP_CHECK(hipFree(C));
  return 0;
}
