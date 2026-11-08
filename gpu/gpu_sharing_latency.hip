// gpu_sharing_latency.hip
// Companion example for: topnotchnote.com/gpu/virtual_gpus.html
//
// When two processes share one GPU without partitioning (plain sharing,
// Kubernetes time-slicing, a time-sliced vGPU), the GPU runs their work in
// turns. Nothing stops a heavy neighbour from delaying a light one.
// This program shows that on any GPU:
//   1. it measures the latency of a tiny kernel (launch + finish) alone;
//   2. it starts a second copy of itself in "hog" mode, a separate process
//      that keeps the GPU busy with long kernels;
//   3. it measures the tiny kernel's latency again while the hog runs.
// A hardware partition (MIG, an AMD CPX partition) would keep the two apart;
// plain sharing does not.
//
// Compile:  hipcc -O2 -o main gpu_sharing_latency.hip
// Nvidia:   hipcc also targets Nvidia (HIP_PLATFORM=nvidia), or swap the hip
//           prefix for cuda in the API calls and build with nvcc.
// Run:      ./main            (spawns its own hog process; Linux/POSIX only)

#include <hip/hip_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <signal.h>
#include <sys/wait.h>
#include <unistd.h>
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

__global__ void tiny(int* p) { if (threadIdx.x == 0) *p += 1; }

// Spin for roughly `cycles` GPU clock cycles in every thread.
__global__ void spin(long long cycles, int* p) {
  long long start = clock64();
  while (clock64() - start < cycles) {}
  if (threadIdx.x == 0 && blockIdx.x == 0) *p += 1;
}

static void hog(double seconds) {
  int* p;
  HIP_CHECK(hipMalloc(&p, sizeof(int)));
  int sms = 0;
  HIP_CHECK(hipDeviceGetAttribute(&sms, hipDeviceAttributeMultiprocessorCount, 0));
  auto end = std::chrono::steady_clock::now() + std::chrono::duration<double>(seconds);
  while (std::chrono::steady_clock::now() < end) {
    // ~50 M cycles: tens of milliseconds per kernel on most GPUs
    hipLaunchKernelGGL(spin, dim3(sms * 8), dim3(256), 0, 0, 50000000LL, p);
    HIP_CHECK(hipDeviceSynchronize());
  }
}

// Median and 99th percentile of tiny-kernel round trips, in microseconds.
static void measure(const char* label, int* p, int n) {
  std::vector<double> us;
  for (int i = 0; i < n; ++i) {
    auto t0 = std::chrono::steady_clock::now();
    hipLaunchKernelGGL(tiny, dim3(1), dim3(32), 0, 0, p);
    HIP_CHECK(hipDeviceSynchronize());
    auto t1 = std::chrono::steady_clock::now();
    us.push_back(std::chrono::duration<double, std::micro>(t1 - t0).count());
  }
  std::sort(us.begin(), us.end());
  std::printf("%-28s median %9.1f us   p99 %9.1f us   max %9.1f us\n", label,
              us[us.size() / 2], us[us.size() * 99 / 100], us.back());
}

int main(int argc, char** argv) {
  if (argc > 1 && std::strcmp(argv[1], "hog") == 0) {
    hog(argc > 2 ? std::atof(argv[2]) : 6.0);
    return 0;
  }
  int* p;
  HIP_CHECK(hipMalloc(&p, sizeof(int)));
  measure("alone:", p, 2000);

  pid_t child = fork();
  if (child == 0) {  // new process: exec a fresh copy in hog mode
    execl(argv[0], argv[0], "hog", "6", (char*)nullptr);
    std::perror("execl");
    _exit(1);
  }
  sleep(2);  // let the hog process start and fill the GPU
  measure("with a hog process:", p, 200);
  kill(child, SIGTERM);
  waitpid(child, nullptr, 0);
  measure("alone again:", p, 2000);
  return 0;
}
