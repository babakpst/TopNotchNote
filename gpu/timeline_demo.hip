// timeline_demo.hip
// Companion example for: topnotchnote.com/gpu/nsight_systems.html
//
// A training-style loop, written twice, to be looked at under a timeline
// profiler (Nsight Systems on NVIDIA, rocprofv3 / ROCm Systems Profiler on AMD).
// Each iteration: the CPU prepares a 16 MB batch, the batch is copied to the
// GPU, three kernels run, and a loss value comes back.
//   "slow": pageable host memory, synchronous copies, and the loss is read
//           every iteration, so the CPU and GPU take turns.
//   "fast": pinned memory, two buffers, async copies on a second stream, the
//           CPU prepares batch i+1 while the GPU works on batch i, and the
//           loss is read every 10th iteration.
// Each phase is wrapped in a named range (NVTX on NVIDIA, roctx on AMD) so it
// shows up by name on the timeline.
//
// Profile:  nsys profile -o timeline --trace=cuda,nvtx,osrt ./main
//           nsys stats --report cuda_gpu_kern_sum,cuda_gpu_mem_time_sum,nvtx_sum timeline.nsys-rep
// AMD:      rocprofv3 --kernel-trace --memory-copy-trace --marker-trace --stats -- ./main
//
// Compile:  hipcc -O2 -o main timeline_demo.hip -lroctx64
// Nvidia:   swap the hip prefix for cuda and build with nvcc ... -lnvToolsExt

#include <hip/hip_runtime.h>

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#if defined(__HIP_PLATFORM_AMD__)
#include <roctracer/roctx.h>
#define RANGE_PUSH(name) roctxRangePush(name)
#define RANGE_POP() roctxRangePop()
#else
#include <nvToolsExt.h>
#define RANGE_PUSH(name) nvtxRangePushA(name)
#define RANGE_POP() nvtxRangePop()
#endif

#define HIP_CHECK(call)                                                   \
  do {                                                                    \
    hipError_t err_ = (call);                                             \
    if (err_ != hipSuccess) {                                             \
      std::fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__,             \
                   hipGetErrorString(err_));                              \
      std::exit(1);                                                       \
    }                                                                     \
  } while (0)

constexpr int N = 4 << 20;          // 4M floats = 16 MB per batch
constexpr int kIters = 30;

__global__ void scale(float* x, float a) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < N) x[i] = x[i] * a + 0.5f;
}
__global__ void square(float* x) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < N) x[i] = x[i] * x[i];
}
__global__ void loss_kernel(const float* x, float* loss) {
  __shared__ float s[256];
  float acc = 0;
  for (int i = blockIdx.x * 256 + threadIdx.x; i < N; i += gridDim.x * 256) acc += x[i];
  s[threadIdx.x] = acc;
  __syncthreads();
  for (int k = 128; k > 0; k >>= 1) {
    if (threadIdx.x < k) s[threadIdx.x] += s[threadIdx.x + k];
    __syncthreads();
  }
  if (threadIdx.x == 0) atomicAdd(loss, s[0] * 1e-9f);
}

static void prepare(float* h, int iter) {          // CPU work: a few ms
  for (int i = 0; i < N; ++i) h[i] = std::sin(0.001f * ((i + iter) % 6283));
}

static void step_kernels(float* d, float* d_loss, hipStream_t s) {
  scale<<<N / 256, 256, 0, s>>>(d, 0.9f);
  square<<<N / 256, 256, 0, s>>>(d);
  loss_kernel<<<64, 256, 0, s>>>(d, d_loss);
}

using Clock = std::chrono::steady_clock;

int main() {
  float *d[2], *d_loss;
  HIP_CHECK(hipMalloc(&d[0], N * sizeof(float)));
  HIP_CHECK(hipMalloc(&d[1], N * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_loss, sizeof(float)));
  HIP_CHECK(hipMemset(d_loss, 0, sizeof(float)));
  float loss = 0;

  // ---- slow version ----
  std::vector<float> pageable(N);
  RANGE_PUSH("slow loop");
  auto t0 = Clock::now();
  for (int it = 0; it < kIters; ++it) {
    RANGE_PUSH("prepare batch (CPU)");
    prepare(pageable.data(), it);
    RANGE_POP();
    HIP_CHECK(hipMemcpy(d[0], pageable.data(), N * sizeof(float), hipMemcpyHostToDevice));
    step_kernels(d[0], d_loss, 0);
    HIP_CHECK(hipMemcpy(&loss, d_loss, sizeof(float), hipMemcpyDeviceToHost));  // waits
  }
  double slow_ms = std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
  RANGE_POP();

  // ---- fast version ----
  float* pinned[2];
  HIP_CHECK(hipHostMalloc(&pinned[0], N * sizeof(float)));
  HIP_CHECK(hipHostMalloc(&pinned[1], N * sizeof(float)));
  hipStream_t copy_s, comp_s;
  HIP_CHECK(hipStreamCreate(&copy_s));
  HIP_CHECK(hipStreamCreate(&comp_s));
  hipEvent_t copied[2], used[2];
  for (int b = 0; b < 2; ++b) {
    HIP_CHECK(hipEventCreateWithFlags(&copied[b], hipEventDisableTiming));
    HIP_CHECK(hipEventCreateWithFlags(&used[b], hipEventDisableTiming));
    HIP_CHECK(hipEventRecord(used[b], comp_s));
  }
  RANGE_PUSH("fast loop");
  t0 = Clock::now();
  for (int it = 0; it < kIters; ++it) {
    const int b = it % 2;
    HIP_CHECK(hipEventSynchronize(used[b]));        // buffer b free again?
    RANGE_PUSH("prepare batch (CPU)");
    prepare(pinned[b], it);
    RANGE_POP();
    HIP_CHECK(hipMemcpyAsync(d[b], pinned[b], N * sizeof(float), hipMemcpyHostToDevice, copy_s));
    HIP_CHECK(hipEventRecord(copied[b], copy_s));
    HIP_CHECK(hipStreamWaitEvent(comp_s, copied[b], 0));
    step_kernels(d[b], d_loss, comp_s);
    HIP_CHECK(hipEventRecord(used[b], comp_s));
    if (it % 10 == 9)
      HIP_CHECK(hipMemcpyAsync(&loss, d_loss, sizeof(float), hipMemcpyDeviceToHost, comp_s));
  }
  HIP_CHECK(hipStreamSynchronize(comp_s));
  double fast_ms = std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
  RANGE_POP();

  // CPU batch preparation alone, for comparison
  t0 = Clock::now();
  for (int it = 0; it < kIters; ++it) prepare(pageable.data(), it);
  double prep_ms = std::chrono::duration<double, std::milli>(Clock::now() - t0).count();

  std::printf("prepare only: %5.1f ms/iter (CPU work, no GPU)\n", prep_ms / kIters);
  std::printf("slow loop: %7.1f ms   (%5.1f ms/iter)\n", slow_ms, slow_ms / kIters);
  std::printf("fast loop: %7.1f ms   (%5.1f ms/iter)\n", fast_ms, fast_ms / kIters);
  for (int b = 0; b < 2; ++b) {
    HIP_CHECK(hipHostFree(pinned[b]));
    HIP_CHECK(hipFree(d[b]));
  }
  HIP_CHECK(hipFree(d_loss));
  return 0;
}
