// stack_probe.hip
// Companion example for: topnotchnote.com/gpu/cuda_rocm_stack.html
//
// Touches every layer of the GPU software stack once (runtime, a math library,
// a kernel), then asks Linux which pieces of the stack are actually in the
// process: the shared libraries mapped into memory (/proc/self/maps) and the
// GPU device files held open (/proc/self/fd).
//
// On AMD you should see the HIP runtime, the ROCr (HSA) runtime, the BLAS
// libraries and /dev/kfd plus a /dev/dri render node. On Nvidia (built through
// HIP's CUDA path) you see libcuda, cuBLAS and the /dev/nvidia* files instead.
//
// Compile (AMD):     hipcc -O2 -o stack_probe stack_probe.hip -lhipblas
// Compile (Nvidia):  HIP_PLATFORM=nvidia hipcc -O2 -o stack_probe stack_probe.hip -lhipblas
// Linux only (reads /proc).

#include <hip/hip_runtime.h>
#include <hipblas/hipblas.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dirent.h>
#include <set>
#include <string>
#include <unistd.h>

#define HIP_CHECK(call)                                                   \
  do {                                                                    \
    hipError_t err_ = (call);                                             \
    if (err_ != hipSuccess) {                                             \
      std::fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__,             \
                   hipGetErrorString(err_));                              \
      std::exit(1);                                                       \
    }                                                                     \
  } while (0)

__global__ void fill(float* x, float v, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) x[i] = v;
}

// True if a library path belongs to the GPU stack (AMD or Nvidia).
static bool is_gpu_lib(const std::string& path) {
  const char* keys[] = {"amdhip", "hsa-runtime", "hipblas", "rocblas",
                        "rocsolver", "libcuda", "cublas", "cudart", "nvidia"};
  for (const char* k : keys)
    if (path.find(k) != std::string::npos) return true;
  return false;
}

int main() {
  // 1. Runtime and driver versions, as HIP reports them.
  int rt = 0, drv = 0;
  HIP_CHECK(hipRuntimeGetVersion(&rt));
  HIP_CHECK(hipDriverGetVersion(&drv));
  std::printf("runtime version %d, driver version %d\n", rt, drv);

  // 2. A kernel, through the runtime.
  const int n = 1024;
  float* x = nullptr;
  HIP_CHECK(hipMalloc(&x, n * sizeof(float)));
  fill<<<(n + 255) / 256, 256>>>(x, 2.0f, n);
  HIP_CHECK(hipGetLastError());

  // 3. A math library call (BLAS sum of absolute values).
  hipblasHandle_t h;
  if (hipblasCreate(&h) != HIPBLAS_STATUS_SUCCESS) {
    std::fprintf(stderr, "hipblasCreate failed\n");
    return 1;
  }
  float sum = 0.0f;
  hipblasSasum(h, n, x, 1, &sum);
  hipblasDestroy(h);
  std::printf("kernel + BLAS ran: sum = %.1f (expected %.1f)\n", sum, 2.0f * n);

  // 4. Which stack libraries are mapped into this process?
  std::set<std::string> libs;
  if (FILE* f = std::fopen("/proc/self/maps", "r")) {
    char line[1024];
    while (std::fgets(line, sizeof line, f)) {
      const char* p = std::strchr(line, '/');
      if (!p) continue;
      std::string path(p);
      path.erase(path.find_last_not_of("\n") + 1);
      std::string base = path.substr(path.rfind('/') + 1);
      if (base.find(".so") != std::string::npos && is_gpu_lib(base)) libs.insert(path);
    }
    std::fclose(f);
  }
  std::printf("\nGPU-stack libraries loaded:\n");
  for (const auto& l : libs) std::printf("  %s\n", l.c_str());

  // 5. Which device files does the process hold open?
  std::set<std::string> devs;
  if (DIR* d = opendir("/proc/self/fd")) {
    while (dirent* e = readdir(d)) {
      char link[300], target[512];
      std::snprintf(link, sizeof link, "/proc/self/fd/%s", e->d_name);
      ssize_t len = readlink(link, target, sizeof target - 1);
      if (len <= 0) continue;
      target[len] = '\0';
      if (std::strncmp(target, "/dev/", 5) == 0 && std::strncmp(target, "/dev/pts", 8) != 0 &&
          std::strcmp(target, "/dev/null") != 0)
        devs.insert(target);
    }
    closedir(d);
  }
  std::printf("\nDevice files open:\n");
  for (const auto& dv : devs) std::printf("  %s\n", dv.c_str());

  HIP_CHECK(hipFree(x));
  return 0;
}
