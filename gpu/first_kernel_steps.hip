// first_kernel_steps.hip
// Companion example for: topnotchnote.com/gpu/first_kernel.html
//
// Vector addition, c = a + b, with every one of the five host-side steps
// timed separately: allocate, copy in, launch, copy out, free. The first
// kernel most people write spends almost all of its time outside the kernel,
// and this program shows by how much on your GPU.
//
// Compile:  hipcc -O2 -o main first_kernel_steps.hip
// Nvidia:   hipcc also targets Nvidia (HIP_PLATFORM=nvidia), or swap the hip
//           prefix for cuda in the API calls and build with nvcc.
// Run:      ./main [n]       (n = number of floats, default 16M)

#include <hip/hip_runtime.h>

#include <chrono>
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

__global__ void vector_add(const float* a, const float* b, float* c, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) c[i] = a[i] + b[i];
}

using clk = std::chrono::steady_clock;
static double ms_since(clk::time_point t) {
  return std::chrono::duration<double, std::milli>(clk::now() - t).count();
}

int main(int argc, char** argv) {
  int n = argc > 1 ? std::atoi(argv[1]) : (16 << 20);
  size_t bytes = size_t(n) * sizeof(float);
  std::vector<float> ha(n, 1.0f), hb(n, 2.0f), hc(n, 0.0f);
  HIP_CHECK(hipFree(nullptr));  // create the context up front, not in step 1

  float *da, *db, *dc;
  auto t = clk::now();
  HIP_CHECK(hipMalloc(&da, bytes));
  HIP_CHECK(hipMalloc(&db, bytes));
  HIP_CHECK(hipMalloc(&dc, bytes));
  double t_alloc = ms_since(t);

  t = clk::now();
  HIP_CHECK(hipMemcpy(da, ha.data(), bytes, hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(db, hb.data(), bytes, hipMemcpyHostToDevice));
  double t_in = ms_since(t);

  int block = 256, grid = (n + block - 1) / block;
  t = clk::now();
  hipLaunchKernelGGL(vector_add, dim3(grid), dim3(block), 0, 0, da, db, dc, n);
  HIP_CHECK(hipGetLastError());
  HIP_CHECK(hipDeviceSynchronize());
  double t_kernel = ms_since(t);

  t = clk::now();
  HIP_CHECK(hipMemcpy(hc.data(), dc, bytes, hipMemcpyDeviceToHost));
  double t_out = ms_since(t);

  t = clk::now();
  HIP_CHECK(hipFree(da));
  HIP_CHECK(hipFree(db));
  HIP_CHECK(hipFree(dc));
  double t_free = ms_since(t);

  int bad = 0;
  for (int i = 0; i < n; ++i) bad += std::fabs(hc[i] - 3.0f) > 0;
  double total = t_alloc + t_in + t_kernel + t_out + t_free;
  std::printf("%d floats (%.0f MiB per array): %s\n", n, bytes / 1048576.0,
              bad ? "WRONG" : "correct");
  const char* names[] = {"1 allocate", "2 copy in (a, b)", "3 kernel",
                         "4 copy out (c)", "5 free"};
  double times[] = {t_alloc, t_in, t_kernel, t_out, t_free};
  for (int s = 0; s < 5; ++s)
    std::printf("  %-18s %9.3f ms  %5.1f%%\n", names[s], times[s], 100 * times[s] / total);
  std::printf("  kernel bandwidth: %.1f GB/s (reads a, b, writes c)\n",
              3.0 * bytes / (t_kernel * 1e-3) / 1e9);
  return bad != 0;
}
