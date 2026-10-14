// sycl_saxpy.cpp
// Companion example for: topnotchnote.com/gpu/sycl_oneapi.html
//
// SAXPY (y = a*x + y) in standard SYCL 2020, the same source for Intel,
// Nvidia and AMD GPUs. It:
//   1. lists every SYCL device the runtime can see, and its backend
//   2. runs SAXPY on 64M floats with USM (unified shared memory) device
//      pointers, the style closest to CUDA and HIP
//   3. runs it again with buffers and accessors, where the runtime moves
//      the data for you
// Kernels are timed with SYCL's own event profiling, so the numbers can be
// compared with the HIP and CUDA versions on the site.
//
// Compile (oneAPI DPC++ or the open-source intel/llvm build):
//   Intel GPU:   icpx -fsycl -O2 -o main sycl_saxpy.cpp
//   Nvidia GPU:  clang++ -fsycl -fsycl-targets=nvptx64-nvidia-cuda -O2 -o main sycl_saxpy.cpp
//   AMD GPU:     clang++ -fsycl -fsycl-targets=amdgcn-amd-amdhsa \
//                  -Xsycl-target-backend --offload-arch=gfx942 -O2 -o main sycl_saxpy.cpp
//   AdaptiveCpp: acpp -O2 -o main sycl_saxpy.cpp
// Pick a device at run time with ONEAPI_DEVICE_SELECTOR, e.g. "cuda:0".

#include <sycl/sycl.hpp>

#include <cmath>
#include <cstdio>
#include <vector>

// Kernel time in milliseconds from a profiled event.
static double eventMs(const sycl::event &e) {
  auto t0 = e.get_profiling_info<sycl::info::event_profiling::command_start>();
  auto t1 = e.get_profiling_info<sycl::info::event_profiling::command_end>();
  return (t1 - t0) * 1e-6;
}

int main() {
  // 1. What can this runtime see?
  std::printf("SYCL devices:\n");
  for (const auto &p : sycl::platform::get_platforms())
    for (const auto &d : p.get_devices())
      std::printf("  [%s] %s\n",
                  p.get_info<sycl::info::platform::name>().c_str(),
                  d.get_info<sycl::info::device::name>().c_str());

  sycl::queue q{sycl::gpu_selector_v,
                sycl::property::queue::enable_profiling{}};
  std::printf("Running on: %s\n\n",
              q.get_device().get_info<sycl::info::device::name>().c_str());

  const size_t n = size_t(1) << 26;  // 64M floats, 256 MB per array
  const float a = 2.0f;
  const double gb = 3.0 * n * sizeof(float) / 1e9;  // read x, read y, write y
  const int reps = 20;
  std::vector<float> hx(n), hy(n);
  for (size_t i = 0; i < n; ++i) {
    hx[i] = float(i % 1000) * 0.001f;
    hy[i] = 1.0f;
  }

  // 2. USM: explicit device allocations and copies, like hipMalloc/hipMemcpy.
  float *x = sycl::malloc_device<float>(n, q);
  float *y = sycl::malloc_device<float>(n, q);
  q.memcpy(x, hx.data(), n * sizeof(float));
  q.memcpy(y, hy.data(), n * sizeof(float)).wait();

  auto saxpy = [=](sycl::id<1> i) { y[i] = a * x[i] + y[i]; };
  sycl::event first = q.parallel_for(sycl::range<1>(n), saxpy);
  first.wait();
  double total = 0.0;
  for (int r = 0; r < reps; ++r) {
    sycl::event e = q.parallel_for(sycl::range<1>(n), saxpy);
    e.wait();
    total += eventMs(e);
  }
  double usmMs = total / reps;
  std::printf("USM:      first launch %7.2f ms, then %6.2f ms per call (%.1f GB/s)\n",
              eventMs(first), usmMs, gb / (usmMs * 1e-3));

  std::vector<float> check(n);
  q.memcpy(check.data(), y, n * sizeof(float)).wait();
  // 1 warm-up + reps calls, each adding a*x to y.
  double maxErr = 0.0;
  for (size_t i = 0; i < n; i += 4099)
    maxErr = std::fmax(maxErr, std::fabs(check[i] - (1.0f + (reps + 1) * a * hx[i])));
  std::printf("          max error %.2e\n", maxErr);
  sycl::free(x, q);
  sycl::free(y, q);

  // 3. Buffers and accessors: the runtime tracks which kernel reads and
  // writes what, schedules the copies, and writes y back to hy when the
  // buffer is destroyed at the end of the scope.
  {
    sycl::buffer<float> bx(hx.data(), sycl::range<1>(n));
    sycl::buffer<float> by(hy.data(), sycl::range<1>(n));
    double bufTotal = 0.0;
    for (int r = 0; r < reps + 1; ++r) {
      sycl::event e = q.submit([&](sycl::handler &h) {
        sycl::accessor ax(bx, h, sycl::read_only);
        sycl::accessor ay(by, h, sycl::read_write);
        h.parallel_for(sycl::range<1>(n),
                       [=](sycl::id<1> i) { ay[i] = a * ax[i] + ay[i]; });
      });
      e.wait();
      if (r > 0) bufTotal += eventMs(e);  // skip the call that copies in
    }
    double bufMs = bufTotal / reps;
    std::printf("Buffers:  %6.2f ms per call (%.1f GB/s)\n", bufMs,
                gb / (bufMs * 1e-3));
  }
  std::printf("          hy[1] after write-back: %.3f (expected %.3f)\n", hy[1],
              1.0f + (reps + 1) * a * hx[1]);
  return 0;
}
