// overlap_buckets_demo.hip
// Companion example for: topnotchnote.com/gpu/comm_compute_overlap.html
//
// The shape of data-parallel training's backward pass, on one GPU:
// L "layers" are processed last to first; each produces a gradient buffer
// that must then be sent somewhere. In real training the sending is an
// all-reduce over NVLink, Infinity Fabric or the network (NCCL / RCCL); here a
// device-to-host copy into pinned memory stands in for it, because it uses a
// separate engine from the compute units, just as the network transfer does.
//   1. compute only, and communication only (to know both costs)
//   2. no overlap: run every layer, then send every gradient
//   3. one bucket: send everything as one transfer after the last layer
//   4. bucketed: send each group of k layers' gradients on a second stream
//      as soon as that group is done, while the next layers compute
// Prints the time for each, next to max(compute, communication), the best
// possible with perfect overlap.
//
// Compile:  hipcc -O2 -o main overlap_buckets_demo.hip
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

constexpr int kLayers = 16;
constexpr size_t kGradFloats = 2 << 20;      // 8 MB of gradient per layer

// Stand-in for one layer's backward pass: some arithmetic that writes the
// layer's gradient buffer.
__global__ void layer_backward(float* grad, int work) {
  size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
  if (i >= kGradFloats) return;
  float g = grad[i] * 0.5f + 1.0f;
  for (int k = 0; k < work; ++k) g = g * 0.9999f + 0.0001f;
  grad[i] = g;
}

struct Ctx {
  float* grads;            // kLayers buffers on the device, back to back
  float* host;             // pinned host copy of all gradients
  hipStream_t compute, comm;
  hipEvent_t layer_done[kLayers];
  int work;
};

static void compute_layer(Ctx& c, int l) {
  layer_backward<<<(unsigned)(kGradFloats / 256), 256, 0, c.compute>>>(
      c.grads + l * kGradFloats, c.work);
}
static void send(Ctx& c, int first, int count, hipStream_t s) {
  HIP_CHECK(hipMemcpyAsync(c.host + first * kGradFloats, c.grads + first * kGradFloats,
                           count * kGradFloats * sizeof(float), hipMemcpyDeviceToHost, s));
}

// bucket_layers = 0 means "no overlap": everything on one stream, in order.
static float run(Ctx& c, int mode, int bucket_layers) {
  hipEvent_t a, b;
  HIP_CHECK(hipEventCreate(&a));
  HIP_CHECK(hipEventCreate(&b));
  HIP_CHECK(hipDeviceSynchronize());
  HIP_CHECK(hipEventRecord(a, c.compute));
  if (mode == 0) {                                         // compute only
    for (int l = kLayers - 1; l >= 0; --l) compute_layer(c, l);
  } else if (mode == 1) {                                  // communication only
    send(c, 0, kLayers, c.compute);
  } else if (mode == 2) {                                  // no overlap
    for (int l = kLayers - 1; l >= 0; --l) compute_layer(c, l);
    for (int l = kLayers - 1; l >= 0; --l) send(c, l, 1, c.compute);
  } else {                                                 // bucketed, overlapped
    int pending = 0;                                       // layers done, not yet sent
    for (int l = kLayers - 1; l >= 0; --l) {
      compute_layer(c, l);
      if (++pending == bucket_layers || l == 0) {
        // gradients for layers l .. l+pending-1 are ready: send them
        HIP_CHECK(hipEventRecord(c.layer_done[l], c.compute));
        HIP_CHECK(hipStreamWaitEvent(c.comm, c.layer_done[l], 0));
        send(c, l, pending, c.comm);
        pending = 0;
      }
    }
    HIP_CHECK(hipEventRecord(c.layer_done[0], c.comm));
    HIP_CHECK(hipStreamWaitEvent(c.compute, c.layer_done[0], 0));
  }
  HIP_CHECK(hipEventRecord(b, c.compute));
  HIP_CHECK(hipEventSynchronize(b));
  float ms;
  HIP_CHECK(hipEventElapsedTime(&ms, a, b));
  HIP_CHECK(hipEventDestroy(a));
  HIP_CHECK(hipEventDestroy(b));
  return ms;
}

static float median(Ctx& c, int mode, int bucket) {
  run(c, mode, bucket);
  std::vector<float> t;
  for (int i = 0; i < 7; ++i) t.push_back(run(c, mode, bucket));
  std::sort(t.begin(), t.end());
  return t[3];
}

int main(int argc, char** argv) {
  Ctx c;
  c.work = argc > 1 ? std::atoi(argv[1]) : 640;   // 640: compute about equals communication on a GT 1030
  HIP_CHECK(hipMalloc(&c.grads, kLayers * kGradFloats * sizeof(float)));
  HIP_CHECK(hipMemset(c.grads, 0, kLayers * kGradFloats * sizeof(float)));
  HIP_CHECK(hipHostMalloc(&c.host, kLayers * kGradFloats * sizeof(float)));
  HIP_CHECK(hipStreamCreate(&c.compute));
  HIP_CHECK(hipStreamCreate(&c.comm));
  for (auto& e : c.layer_done) HIP_CHECK(hipEventCreateWithFlags(&e, hipEventDisableTiming));

  float tc = median(c, 0, 0), tm = median(c, 1, 0);
  std::printf("%d layers, %zu MB of gradient each, work=%d\n", kLayers,
              kGradFloats * sizeof(float) >> 20, c.work);
  std::printf("  compute only                      %8.2f ms\n", tc);
  std::printf("  communication only                %8.2f ms\n", tm);
  std::printf("  best possible, max of the two     %8.2f ms\n", std::max(tc, tm));
  std::printf("  no overlap (compute, then send)   %8.2f ms\n", median(c, 2, 0));
  std::printf("  one bucket of %2d layers           %8.2f ms\n", kLayers, median(c, 3, kLayers));
  for (int k : {8, 4, 2, 1})
    std::printf("  buckets of %2d layers              %8.2f ms\n", k, median(c, 3, k));

  HIP_CHECK(hipFree(c.grads));
  HIP_CHECK(hipHostFree(c.host));
  return 0;
}
