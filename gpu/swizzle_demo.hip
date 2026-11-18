// swizzle_demo.hip
// Companion example for: topnotchnote.com/gpu/swizzling.html
//
// A 32 x 32 float tile in shared memory (LDS), written row by row with
// 16-byte float4 stores and then read "down the columns": lane l of a warp
// reads one float4 from row l. This is the shape of a tile that is loaded
// row-major from global memory and consumed column-wise by the math.
// Three layouts:
//   1. plain:    chunk c of row r lives at  r*32 + c*4
//   2. padded:   rows are 36 floats long (one extra float4), so the start of
//                each row moves 4 banks. Costs 12.5% more shared memory.
//                (Padding by one float would break float4 alignment.)
//   3. swizzled: chunk c of row r lives at  r*32 + (c ^ (r % 8))*4
//                Same 4 KB as plain; the XOR moves each row's chunks to
//                different banks.
// The program times many column reads for each layout and checks that all
// three produce the same sums.
//
// Compile:  hipcc -O2 -o main swizzle_demo.hip
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

constexpr int kRows = 32, kChunks = 8;   // 8 float4 chunks = 32 floats per row

enum Layout { PLAIN = 0, PADDED = 1, SWIZZLED = 2 };

template <int L>
__device__ __forceinline__ int offset_floats(int r, int c) {
  if (L == PLAIN) return r * 32 + c * 4;
  if (L == PADDED) return r * 36 + c * 4;
  return r * 32 + ((c ^ (r % 8)) * 4);          // SWIZZLED
}

// 256 threads = 8 warps per block. Every warp reads the whole tile column by
// column, `reps` times.
template <int L>
__global__ void column_reads(const float* in, float* out, int reps) {
  __shared__ __align__(16) float tile[kRows * 36];   // big enough for padding
  const int t = threadIdx.x;
  // Row-major fill: thread t writes chunk t % 8 of row t / 8.
  {
    const int r = t / kChunks, c = t % kChunks;
    float4 v = reinterpret_cast<const float4*>(in)[blockIdx.x * 256 + t];
    *reinterpret_cast<float4*>(&tile[offset_floats<L>(r, c)]) = v;
  }
  __syncthreads();
  const int lane = t % 32, warp = t / 32;
  float acc = 0;
  for (int i = 0; i < reps; ++i) {
    const int c = (warp + i) % kChunks;              // the column chunk this warp reads
    float4 v = *reinterpret_cast<const float4*>(&tile[offset_floats<L>(lane, c)]);
    acc += v.x + v.y + v.z + v.w;
  }
  out[blockIdx.x * 256 + t] = acc;
}

template <int L>
static float time_it(const float* in, float* out, int blocks, int reps, std::vector<float>& h) {
  column_reads<L><<<blocks, 256>>>(in, out, reps);
  HIP_CHECK(hipGetLastError());
  hipEvent_t a, b;
  HIP_CHECK(hipEventCreate(&a));
  HIP_CHECK(hipEventCreate(&b));
  std::vector<float> t;
  for (int k = 0; k < 7; ++k) {
    HIP_CHECK(hipEventRecord(a));
    column_reads<L><<<blocks, 256>>>(in, out, reps);
    HIP_CHECK(hipEventRecord(b));
    HIP_CHECK(hipEventSynchronize(b));
    float ms;
    HIP_CHECK(hipEventElapsedTime(&ms, a, b));
    t.push_back(ms);
  }
  HIP_CHECK(hipMemcpy(h.data(), out, h.size() * sizeof(float), hipMemcpyDeviceToHost));
  std::sort(t.begin(), t.end());
  return t[3];
}

int main() {
  const int blocks = 1024, reps = 4096;
  const size_t n = (size_t)blocks * 256 * 4;      // one float4 per thread
  std::vector<float> h(n);
  for (size_t i = 0; i < n; ++i) h[i] = (float)(i % 7);
  float *in, *out;
  HIP_CHECK(hipMalloc(&in, n * sizeof(float)));
  HIP_CHECK(hipMalloc(&out, blocks * 256 * sizeof(float)));
  HIP_CHECK(hipMemcpy(in, h.data(), n * sizeof(float), hipMemcpyHostToDevice));

  std::vector<float> r0(blocks * 256), r1(blocks * 256), r2(blocks * 256);
  float t0 = time_it<PLAIN>(in, out, blocks, reps, r0);
  float t1 = time_it<PADDED>(in, out, blocks, reps, r1);
  float t2 = time_it<SWIZZLED>(in, out, blocks, reps, r2);
  bool same = r0 == r1 && r0 == r2;
  std::printf("%d blocks x 8 warps x %d float4 column reads each\n", blocks, reps);
  std::printf("  plain     (4096 B tile) %8.2f ms\n", t0);
  std::printf("  padded    (4608 B tile) %8.2f ms   %.2fx vs plain\n", t1, t0 / t1);
  std::printf("  swizzled  (4096 B tile) %8.2f ms   %.2fx vs plain\n", t2, t0 / t2);
  std::printf("  results identical: %s\n", same ? "yes" : "NO");
  HIP_CHECK(hipFree(in));
  HIP_CHECK(hipFree(out));
  return same ? 0 : 1;
}
