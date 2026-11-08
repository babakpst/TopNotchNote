// peer_bandwidth_demo.hip
// Companion example for: topnotchnote.com/gpu/nvlink_infinity_fabric.html
//
// GPU-to-GPU traffic can take one of two roads:
//   * a direct peer link (NVLink/NVSwitch on Nvidia, Infinity Fabric/xGMI on
//     AMD, or PCIe peer-to-peer through a switch), used by hipMemcpyPeer once
//     peer access is enabled;
//   * no peer path at all, in which case the data is staged through host
//     memory: GPU A -> host over PCIe, then host -> GPU B over PCIe.
// This program measures both.
//   1. It prints the peer-access matrix (hipDeviceCanAccessPeer) for every
//      pair of visible GPUs.
//   2. For every pair, it times hipMemcpyPeer of 256 MiB, one direction at a
//      time and both directions at once, with peer access enabled when the
//      runtime allows it.
//   3. It times the staged route for the same buffer (device -> pinned host ->
//      device) and a copy inside one GPU, which is what a machine with a
//      single GPU can still show: the gap between on-card bandwidth and the
//      host link a missing peer link forces you onto.
//
// Compile:  hipcc -O2 -o main peer_bandwidth_demo.hip
// Nvidia:   hipcc also targets Nvidia (HIP_PLATFORM=nvidia), or swap the hip
//           prefix for cuda in the API calls and build with nvcc.
// Run:      ./main            (all visible GPUs)
//           HIP_VISIBLE_DEVICES=0,1 ./main   (or CUDA_VISIBLE_DEVICES on Nvidia)

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

constexpr size_t kBytes = size_t(256) << 20;  // 256 MiB per copy
constexpr int kReps = 10;

// Median of the per-repetition times, in GB/s (1 GB = 1e9 bytes).
static double median_gbps(std::vector<float>& ms, double bytes) {
  std::sort(ms.begin(), ms.end());
  return bytes / (ms[ms.size() / 2] * 1e-3) / 1e9;
}

// Times `body` kReps times on `stream` with events recorded on device `dev`.
template <class F>
static double time_gbps(int dev, hipStream_t stream, double bytes, F body) {
  HIP_CHECK(hipSetDevice(dev));
  hipEvent_t start, stop;
  HIP_CHECK(hipEventCreate(&start));
  HIP_CHECK(hipEventCreate(&stop));
  body();  // warm-up
  HIP_CHECK(hipStreamSynchronize(stream));
  std::vector<float> ms;
  for (int r = 0; r < kReps; ++r) {
    HIP_CHECK(hipEventRecord(start, stream));
    body();
    HIP_CHECK(hipEventRecord(stop, stream));
    HIP_CHECK(hipEventSynchronize(stop));
    float t = 0;
    HIP_CHECK(hipEventElapsedTime(&t, start, stop));
    ms.push_back(t);
  }
  HIP_CHECK(hipEventDestroy(start));
  HIP_CHECK(hipEventDestroy(stop));
  return median_gbps(ms, bytes);
}

int main() {
  int n = 0;
  HIP_CHECK(hipGetDeviceCount(&n));
  std::printf("visible GPUs: %d\n", n);
  for (int d = 0; d < n; ++d) {
    hipDeviceProp_t p;
    HIP_CHECK(hipGetDeviceProperties(&p, d));
    std::printf("  [%d] %s  PCI %02x:%02x\n", d, p.name, p.pciBusID,
                p.pciDeviceID);
  }

  // 1. Peer-access matrix.
  std::printf("\npeer access (row can read/write column's memory directly):\n     ");
  for (int j = 0; j < n; ++j) std::printf("%4d", j);
  std::printf("\n");
  for (int i = 0; i < n; ++i) {
    std::printf("  %2d ", i);
    for (int j = 0; j < n; ++j) {
      int ok = 0;
      if (i != j) HIP_CHECK(hipDeviceCanAccessPeer(&ok, i, j));
      std::printf("%4s", i == j ? "X" : (ok ? "yes" : "no"));
    }
    std::printf("\n");
  }

  // One buffer and one stream per GPU, plus one pinned host buffer.
  std::vector<void*> a(n), b(n);
  std::vector<hipStream_t> s(n);
  for (int d = 0; d < n; ++d) {
    HIP_CHECK(hipSetDevice(d));
    HIP_CHECK(hipMalloc(&a[d], kBytes));
    HIP_CHECK(hipMalloc(&b[d], kBytes));
    HIP_CHECK(hipMemset(a[d], 1, kBytes));
    HIP_CHECK(hipStreamCreate(&s[d]));
  }
  void* host = nullptr;
  HIP_CHECK(hipHostMalloc(&host, kBytes, hipHostMallocDefault));

  // 3a. Inside one GPU: the on-card memory bandwidth a peer copy competes with.
  // A device-to-device copy reads and writes every byte, so count 2x.
  std::printf("\nsame-GPU copy (read + write counted):\n");
  for (int d = 0; d < n; ++d) {
    double g = time_gbps(d, s[d], 2.0 * kBytes, [&] {
      HIP_CHECK(hipMemcpyAsync(b[d], a[d], kBytes, hipMemcpyDeviceToDevice, s[d]));
    });
    std::printf("  GPU %d: %7.1f GB/s\n", d, g);
  }

  // 3b. Staged through pinned host memory: what GPU-to-GPU costs with no peer
  // path. Each byte crosses the host link twice, one after the other.
  std::printf("\nstaged through host (GPU -> pinned host -> GPU), one direction:\n");
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < n; ++j) {
      if (n > 1 && i == j) continue;
      double g = time_gbps(i, s[i], kBytes, [&] {
        HIP_CHECK(hipMemcpyAsync(host, a[i], kBytes, hipMemcpyDeviceToHost, s[i]));
        HIP_CHECK(hipStreamSynchronize(s[i]));
        HIP_CHECK(hipSetDevice(j));
        HIP_CHECK(hipMemcpyAsync(b[j], host, kBytes, hipMemcpyHostToDevice, s[j]));
        HIP_CHECK(hipStreamSynchronize(s[j]));
        HIP_CHECK(hipSetDevice(i));
      });
      std::printf("  %d -> %d: %7.1f GB/s\n", i, j, g);
    }
  }

  // 2. Direct peer copies.
  if (n < 2) {
    std::printf("\nonly one GPU visible: no peer pairs to time.\n");
  } else {
    std::printf("\nhipMemcpyPeer, 256 MiB (unidirectional | both directions at once):\n");
    for (int i = 0; i < n; ++i) {
      for (int j = 0; j < n; ++j) {
        if (i == j) continue;
        int ij = 0, ji = 0;
        HIP_CHECK(hipDeviceCanAccessPeer(&ij, i, j));
        HIP_CHECK(hipDeviceCanAccessPeer(&ji, j, i));
        if (ij) {
          HIP_CHECK(hipSetDevice(i));
          hipError_t e = hipDeviceEnablePeerAccess(j, 0);
          if (e != hipSuccess && e != hipErrorPeerAccessAlreadyEnabled) HIP_CHECK(e);
          (void)hipGetLastError();  // clear "already enabled"
        }
        if (ji) {
          HIP_CHECK(hipSetDevice(j));
          hipError_t e = hipDeviceEnablePeerAccess(i, 0);
          if (e != hipSuccess && e != hipErrorPeerAccessAlreadyEnabled) HIP_CHECK(e);
          (void)hipGetLastError();
        }
        double uni = time_gbps(i, s[i], kBytes, [&] {
          HIP_CHECK(hipMemcpyPeerAsync(b[j], j, a[i], i, kBytes, s[i]));
        });
        // Bidirectional: i->j on i's stream and j->i on j's stream together.
        double bi = time_gbps(i, s[i], 2.0 * kBytes, [&] {
          HIP_CHECK(hipMemcpyPeerAsync(b[j], j, a[i], i, kBytes, s[i]));
          HIP_CHECK(hipSetDevice(j));
          HIP_CHECK(hipMemcpyPeerAsync(b[i], i, a[j], j, kBytes, s[j]));
          HIP_CHECK(hipStreamSynchronize(s[j]));
          HIP_CHECK(hipSetDevice(i));
        });
        std::printf("  %d -> %d: %7.1f GB/s | %7.1f GB/s  (peer access %s)\n", i, j,
                    uni, bi, ij ? "on" : "off: runtime stages the copy");
      }
    }
  }

  for (int d = 0; d < n; ++d) {
    HIP_CHECK(hipSetDevice(d));
    HIP_CHECK(hipFree(a[d]));
    HIP_CHECK(hipFree(b[d]));
    HIP_CHECK(hipStreamDestroy(s[d]));
  }
  HIP_CHECK(hipHostFree(host));
  return 0;
}
