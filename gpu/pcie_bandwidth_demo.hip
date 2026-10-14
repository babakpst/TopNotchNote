// pcie_bandwidth_demo.hip
// Companion example for: topnotchnote.com/gpu/pcie.html
//
// Measures the host-device link and compares it with what the link should
// carry on paper:
//   1. Reads the negotiated PCIe generation and lane count from Linux sysfs
//      (the runtime API does not expose them), at start and again mid-copy,
//      and works out the theoretical one-direction bandwidth of each.
//   2. Times pinned host-to-device and device-to-host copies (median of
//      five trials) from 4 KB to
//      256 MB, so you can see small copies drown in fixed overhead and large
//      ones level off at the link's real ceiling.
//   3. Times a device-to-device copy of the same size, for scale: that one
//      never touches PCIe.
// Compile:  hipcc -O2 -o main pcie_bandwidth_demo.hip
// Nvidia:   hipcc also targets Nvidia (HIP_PLATFORM=nvidia), or swap the hip
//           prefix for cuda in the API calls and build with nvcc.
// Linux only for step 1; the timings work anywhere.

#include <hip/hip_runtime.h>

#include <algorithm>
#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

#define HIP_CHECK(call)                                                   \
  do {                                                                    \
    hipError_t err_ = (call);                                             \
    if (err_ != hipSuccess) {                                             \
      std::fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__,             \
                   hipGetErrorString(err_));                              \
      std::exit(1);                                                       \
    }                                                                     \
  } while (0)

static std::string readSysfs(const std::string &path) {
  FILE *f = std::fopen(path.c_str(), "r");
  if (!f) return "";
  char buf[128] = {0};
  if (!std::fgets(buf, sizeof(buf), f)) buf[0] = 0;
  std::fclose(f);
  std::string s(buf);
  while (!s.empty() && std::isspace((unsigned char)s.back())) s.pop_back();
  return s;
}

// Per-lane, per-direction payload rate in GB/s after line encoding.
// Gen 1-2 use 8b/10b encoding, gen 3+ use 128b/130b; gen 6 uses PAM4 + FLIT,
// which lands close to double gen 5.
static double gbpsPerLane(double gtPerSec) {
  if (gtPerSec <= 5.0) return gtPerSec * 8.0 / 10.0 / 8.0;  // 8b/10b
  return gtPerSec * 128.0 / 130.0 / 8.0;                     // 128b/130b
}

// Average time of one copy over `reps` back-to-back copies.
static double timeCopyMs(void *dst, const void *src, size_t bytes,
                         hipMemcpyKind kind, int reps) {
  hipEvent_t start, stop;
  HIP_CHECK(hipEventCreate(&start));
  HIP_CHECK(hipEventCreate(&stop));
  HIP_CHECK(hipMemcpy(dst, src, bytes, kind));  // warm-up
  HIP_CHECK(hipEventRecord(start, 0));
  for (int r = 0; r < reps; ++r)
    HIP_CHECK(hipMemcpyAsync(dst, src, bytes, kind, 0));
  HIP_CHECK(hipEventRecord(stop, 0));
  HIP_CHECK(hipEventSynchronize(stop));
  float ms = 0.0f;
  HIP_CHECK(hipEventElapsedTime(&ms, start, stop));
  HIP_CHECK(hipEventDestroy(start));
  HIP_CHECK(hipEventDestroy(stop));
  return ms / reps;
}

// Median of five timings, so one disturbed run (another process using the
// GPU or the host's memory) does not decide the number.
static double medianCopyMs(void *dst, const void *src, size_t bytes,
                           hipMemcpyKind kind, int reps) {
  double t[5];
  for (double &x : t) x = timeCopyMs(dst, src, bytes, kind, reps);
  std::sort(t, t + 5);
  return t[2];
}

int main() {
  hipDeviceProp_t prop;
  HIP_CHECK(hipGetDeviceProperties(&prop, 0));
  std::printf("Device: %s\n\n", prop.name);

  // 1. What the link negotiated. An idle GPU drops its link to a lower PCIe
  // generation to save power, so read it again mid-copy below. Creating the
  // runtime context is usually enough to wake it, which is why the idle
  // downshift is easier to see from a shell (cat current_link_speed) while
  // nothing is running.
  char busId[32] = {0};
  HIP_CHECK(hipDeviceGetPCIBusId(busId, sizeof(busId), 0));
  for (char *c = busId; *c; ++c) *c = (char)std::tolower(*c);
  const std::string dir = std::string("/sys/bus/pci/devices/") + busId + "/";
  auto linkGBs = [&](const char *speedFile, const char *widthFile) {
    std::string speed = readSysfs(dir + speedFile);
    std::string width = readSysfs(dir + widthFile);
    if (speed.empty() || width.empty()) return 0.0;
    // "8.0 GT/s PCIe" -> 8.0
    return gbpsPerLane(std::atof(speed.c_str())) * std::atoi(width.c_str());
  };
  std::string startSpeed = readSysfs(dir + "current_link_speed");
  std::string startWidth = readSysfs(dir + "current_link_width");
  if (startSpeed.empty()) {
    std::printf("PCIe link: could not read %s (not Linux?)\n\n", dir.c_str());
  } else {
    std::printf("PCIe link (%s)\n", busId);
    std::printf("  maximum         : %s x%s = %.2f GB/s per direction\n",
                readSysfs(dir + "max_link_speed").c_str(),
                readSysfs(dir + "max_link_width").c_str(),
                linkGBs("max_link_speed", "max_link_width"));
    std::printf("  current, start  : %s x%s = %.2f GB/s per direction\n\n",
                startSpeed.c_str(), startWidth.c_str(),
                linkGBs("current_link_speed", "current_link_width"));
  }

  // 2. Pinned copies across sizes.
  const size_t maxBytes = 256u << 20;
  void *hBuf = nullptr, *dBuf = nullptr, *dBuf2 = nullptr;
  HIP_CHECK(hipHostMalloc(&hBuf, maxBytes, hipHostMallocDefault));
  HIP_CHECK(hipMalloc(&dBuf, maxBytes));
  HIP_CHECK(hipMalloc(&dBuf2, maxBytes));
  std::memset(hBuf, 1, maxBytes);
  HIP_CHECK(hipMemset(dBuf, 0, maxBytes));

  std::printf("%10s %12s %12s %10s\n", "size", "H2D GB/s", "D2H GB/s",
              "H2D us");
  std::string busySpeed, busyWidth;
  double busyGBs = 0.0, bestH2D = 0.0;
  for (size_t bytes = 4u << 10; bytes <= maxBytes; bytes *= 4) {
    int reps = bytes < (1u << 20) ? 200 : (bytes < (64u << 20) ? 20 : 4);
    double h2d = medianCopyMs(dBuf, hBuf, bytes, hipMemcpyHostToDevice, reps);
    double d2h = medianCopyMs(hBuf, dBuf, bytes, hipMemcpyDeviceToHost, reps);
    double h2dGBs = bytes / (h2d * 1e6);
    double d2hGBs = bytes / (d2h * 1e6);
    char label[32];
    if (bytes < (1u << 20))
      std::snprintf(label, sizeof(label), "%zu KB", bytes >> 10);
    else
      std::snprintf(label, sizeof(label), "%zu MB", bytes >> 20);
    std::printf("%10s %12.2f %12.2f %10.1f\n", label, h2dGBs, d2hGBs,
                h2d * 1000.0);
    if (bytes == (64u << 20)) {
      // Sample the link while it is busy, between two large copies.
      HIP_CHECK(hipMemcpyAsync(dBuf, hBuf, maxBytes, hipMemcpyHostToDevice, 0));
      busySpeed = readSysfs(dir + "current_link_speed");
      busyWidth = readSysfs(dir + "current_link_width");
      busyGBs = linkGBs("current_link_speed", "current_link_width");
      HIP_CHECK(hipDeviceSynchronize());
    }
    bestH2D = h2dGBs > bestH2D ? h2dGBs : bestH2D;
  }
  if (busyGBs > 0) {
    std::printf("\n  current, busy   : %s x%s = %.2f GB/s per direction\n",
                busySpeed.c_str(), busyWidth.c_str(), busyGBs);
    std::printf("  best H2D above is %.0f%% of that\n",
                100.0 * bestH2D / busyGBs);
  }

  // 3. The same 256 MB, device to device: the on-board memory, not PCIe.
  // A copy reads and writes every byte, so count both.
  double d2d = medianCopyMs(dBuf2, dBuf, maxBytes, hipMemcpyDeviceToDevice, 10);
  std::printf("\nDevice-to-device 256 MB: %.2f GB/s (read + write), "
              "never crosses PCIe\n", 2.0 * maxBytes / (d2d * 1e6));

  HIP_CHECK(hipHostFree(hBuf));
  HIP_CHECK(hipFree(dBuf));
  HIP_CHECK(hipFree(dBuf2));
  return 0;
}
