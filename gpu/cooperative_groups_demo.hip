// cooperative_groups_demo.hip
// Companion example for: topnotchnote.com/gpu/cooperative_groups.html
//
// Cooperative groups give a set of threads a name and a type, so code can
// say which threads it synchronizes, shuffles or reduces across.
//   1. One generic function, written once against "a group", called with
//      tiles of 8, 16 and 32 threads and with a whole block.
//   2. cg::reduce against a hand-written shuffle loop, on register values.
//   3. coalesced_threads() inside a branch: the group is exactly the lanes
//      that took the branch, and thread_rank() renumbers them from 0.
//   4. Stream compaction in a tile with exclusive_scan: each kept element
//      gets its output slot from the scan of the keep flags.
//
// Compile:  hipcc -O2 -o main cooperative_groups_demo.hip
// Nvidia:   hipcc also targets Nvidia (HIP_PLATFORM=nvidia), or swap the hip
//           prefix for cuda in the API calls and build with nvcc.

#include <hip/hip_runtime.h>
#include <hip/hip_cooperative_groups.h>
#if !defined(__HIP_PLATFORM_AMD__)
#include <cooperative_groups/reduce.h>
#include <cooperative_groups/scan.h>
#endif

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

// ---- 1. a function that takes any group ---------------------------------

// Tree sum through shared memory, valid for any group whose threads share
// the block's shared memory. `scratch` has one slot per thread of the block.
template <typename Group>
__device__ int group_sum(Group g, int* scratch, int v) {
  int r = g.thread_rank();
  int* s = scratch + (threadIdx.x - r);  // this group's slice
  s[r] = v;
  g.sync();
  for (int stride = g.size() / 2; stride > 0; stride >>= 1) {
    if (r < stride) s[r] += s[r + stride];
    g.sync();
  }
  int total = s[0];
  g.sync();
  return total;
}

template <int kTile>
__global__ void sums_by_group(const int* in, int* out) {
  __shared__ int scratch[kThreads];
  cg::thread_block block = cg::this_thread_block();
  int v = in[blockIdx.x * kThreads + threadIdx.x];
  int total;
  if (kTile == kThreads) {
    total = group_sum(block, scratch, v);
  } else {
    auto tile = cg::tiled_partition<kTile>(block);
    total = group_sum(tile, scratch, v);
  }
  out[blockIdx.x * kThreads + threadIdx.x] = total;
}

// ---- 2. cg::reduce vs a hand-written shuffle loop -----------------------

__global__ void tile_sum_shuffle(const float* in, float* out, int rounds) {
  auto tile = cg::tiled_partition<32>(cg::this_thread_block());
  float v = in[blockIdx.x * blockDim.x + threadIdx.x];
  float acc = 0.f;
  for (int i = 0; i < rounds; ++i) {
    float s = v + i;
    for (int o = 16; o > 0; o >>= 1) s += tile.shfl_xor(s, o);
    acc += s;
  }
  out[blockIdx.x * blockDim.x + threadIdx.x] = acc;
}

__global__ void tile_sum_cg_reduce(const float* in, float* out, int rounds) {
  auto tile = cg::tiled_partition<32>(cg::this_thread_block());
  float v = in[blockIdx.x * blockDim.x + threadIdx.x];
  float acc = 0.f;
  for (int i = 0; i < rounds; ++i) acc += cg::reduce(tile, v + i, cg::plus<float>());
  out[blockIdx.x * blockDim.x + threadIdx.x] = acc;
}

// ---- 3. coalesced_threads() in a branch ---------------------------------

__global__ void coalesced_demo(const int* in, int* group_size, int* rank) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  group_size[i] = -1;
  rank[i] = -1;
  if (in[i] % 3 == 0) {  // only some lanes of each warp get here
    cg::coalesced_group active = cg::coalesced_threads();
    group_size[i] = active.size();
    rank[i] = active.thread_rank();
  }
}

// ---- 4. compaction with exclusive_scan ----------------------------------

// Each tile of 32 writes its kept values, in order, to its own 32-slot
// output segment; count[] holds how many it kept.
__global__ void compact_tile(const int* in, int* out, int* count) {
  auto tile = cg::tiled_partition<32>(cg::this_thread_block());
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int x = in[i];
  int keep = (x % 3 == 0) ? 1 : 0;
  int slot = cg::exclusive_scan(tile, keep, cg::plus<int>());
  int seg = i / 32;
  if (keep) out[seg * 32 + slot] = x;
  if (tile.thread_rank() == 31) count[seg] = slot + keep;
}

template <typename F>
float median_ms(F launch) {
  hipEvent_t a, b;
  HIP_CHECK(hipEventCreate(&a));
  HIP_CHECK(hipEventCreate(&b));
  launch();
  HIP_CHECK(hipDeviceSynchronize());
  std::vector<float> t(11);
  for (auto& ti : t) {
    HIP_CHECK(hipEventRecord(a));
    launch();
    HIP_CHECK(hipEventRecord(b));
    HIP_CHECK(hipEventSynchronize(b));
    HIP_CHECK(hipEventElapsedTime(&ti, a, b));
  }
  std::sort(t.begin(), t.end());
  return t[t.size() / 2];
}

int main() {
  hipDeviceProp_t prop;
  HIP_CHECK(hipGetDeviceProperties(&prop, 0));
  std::printf("device: %s, warpSize %d\n", prop.name, prop.warpSize);

  const int nblk = 96, n = nblk * kThreads;
  std::vector<int> h(n), hout(n);
  for (int i = 0; i < n; ++i) h[i] = (i * 7919) % 1000;
  int *d_in, *d_out, *d_b, *d_cnt;
  HIP_CHECK(hipMalloc(&d_in, n * sizeof(int)));
  HIP_CHECK(hipMalloc(&d_out, n * sizeof(int)));
  HIP_CHECK(hipMalloc(&d_b, n * sizeof(int)));
  HIP_CHECK(hipMalloc(&d_cnt, n / 32 * sizeof(int)));
  HIP_CHECK(hipMemcpy(d_in, h.data(), n * sizeof(int), hipMemcpyHostToDevice));

  // ---- 1 ----
  auto check_sums = [&](int tile) {
    HIP_CHECK(hipMemcpy(hout.data(), d_out, n * sizeof(int),
                        hipMemcpyDeviceToHost));
    int bad = 0;
    for (int g = 0; g < n / tile; ++g) {
      int ref = 0;
      for (int j = 0; j < tile; ++j) ref += h[g * tile + j];
      for (int j = 0; j < tile; ++j) bad += hout[g * tile + j] != ref;
    }
    std::printf("  group_sum over groups of %3d: %s\n", tile,
                bad ? "WRONG" : "matches the CPU");
  };
  std::printf("\n1. one function, four group sizes\n");
  sums_by_group<8><<<nblk, kThreads>>>(d_in, d_out);    check_sums(8);
  sums_by_group<16><<<nblk, kThreads>>>(d_in, d_out);   check_sums(16);
  sums_by_group<32><<<nblk, kThreads>>>(d_in, d_out);   check_sums(32);
  sums_by_group<256><<<nblk, kThreads>>>(d_in, d_out);  check_sums(256);
  HIP_CHECK(hipGetLastError());

  // ---- 2 ----
  const int rblk = prop.multiProcessorCount * 8, rn = rblk * kThreads;
  const int rounds = 20000;
  float *fin, *f1, *f2;
  HIP_CHECK(hipMalloc(&fin, rn * sizeof(float)));
  HIP_CHECK(hipMalloc(&f1, rn * sizeof(float)));
  HIP_CHECK(hipMalloc(&f2, rn * sizeof(float)));
  std::vector<float> hf(rn);
  for (int i = 0; i < rn; ++i) hf[i] = (float)(i % 5);
  HIP_CHECK(hipMemcpy(fin, hf.data(), rn * sizeof(float),
                      hipMemcpyHostToDevice));
  float t_sh = median_ms([&] {
    tile_sum_shuffle<<<rblk, kThreads>>>(fin, f1, rounds);
  });
  float t_cg = median_ms([&] {
    tile_sum_cg_reduce<<<rblk, kThreads>>>(fin, f2, rounds);
  });
  std::vector<float> r1(rn), r2(rn);
  HIP_CHECK(hipMemcpy(r1.data(), f1, rn * sizeof(float), hipMemcpyDeviceToHost));
  HIP_CHECK(hipMemcpy(r2.data(), f2, rn * sizeof(float), hipMemcpyDeviceToHost));
  bool same = std::equal(r1.begin(), r1.end(), r2.begin());
  std::printf("\n2. %d warp sums in each of %d warps\n", rounds,
              rblk * kThreads / 32);
  std::printf("  hand-written shfl_xor loop: %7.2f ms\n", t_sh);
  std::printf("  cg::reduce(tile, v, plus):  %7.2f ms   results %s\n", t_cg,
              same ? "identical" : "DIFFER");

  // ---- 3 ----
  coalesced_demo<<<nblk, kThreads>>>(d_in, d_out, d_b);
  HIP_CHECK(hipGetLastError());
  std::vector<int> gs(n), rk(n);
  HIP_CHECK(hipMemcpy(gs.data(), d_out, n * sizeof(int), hipMemcpyDeviceToHost));
  HIP_CHECK(hipMemcpy(rk.data(), d_b, n * sizeof(int), hipMemcpyDeviceToHost));
  int ws = prop.warpSize, bad = 0;
  for (int w = 0; w < n / ws; ++w) {
    int taken = 0;
    for (int j = 0; j < ws; ++j) taken += h[w * ws + j] % 3 == 0;
    int next = 0;
    for (int j = 0; j < ws; ++j) {
      int i = w * ws + j;
      if (h[i] % 3 != 0) continue;
      bad += gs[i] != taken || rk[i] != next++;
    }
  }
  std::printf("\n3. coalesced_threads() in a branch: first warp has %d of %d "
              "lanes inside; sizes and ranks %s\n",
              gs[0] >= 0 ? gs[0] : gs[1], ws, bad ? "WRONG" : "all correct");

  // ---- 4 ----
  compact_tile<<<nblk, kThreads>>>(d_in, d_out, d_cnt);
  HIP_CHECK(hipGetLastError());
  std::vector<int> cnt(n / 32);
  HIP_CHECK(hipMemcpy(hout.data(), d_out, n * sizeof(int),
                      hipMemcpyDeviceToHost));
  HIP_CHECK(hipMemcpy(cnt.data(), d_cnt, n / 32 * sizeof(int),
                      hipMemcpyDeviceToHost));
  bad = 0;
  long kept = 0;
  for (int seg = 0; seg < n / 32; ++seg) {
    int k = 0;
    for (int j = 0; j < 32; ++j) {
      int x = h[seg * 32 + j];
      if (x % 3 == 0) bad += hout[seg * 32 + k++] != x;
    }
    bad += cnt[seg] != k;
    kept += k;
  }
  std::printf("\n4. compaction with exclusive_scan: kept %ld of %d, order "
              "and counts %s\n", kept, n, bad ? "WRONG" : "all correct");

  HIP_CHECK(hipFree(d_in));
  HIP_CHECK(hipFree(d_out));
  HIP_CHECK(hipFree(d_b));
  HIP_CHECK(hipFree(d_cnt));
  HIP_CHECK(hipFree(fin));
  HIP_CHECK(hipFree(f1));
  HIP_CHECK(hipFree(f2));
  return 0;
}
