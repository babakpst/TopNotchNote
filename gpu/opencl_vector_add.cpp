// opencl_vector_add.cpp
// Companion example for: topnotchnote.com/gpu/opencl_cuda_hip.html
//
// The same vector addition as hip_vector_add.hip, written in OpenCL, so the
// two can be read side by side. The kernel is nearly identical. The host
// code is not: OpenCL has no single-source compiler, so the program must
// find a platform and a device, create a context and a queue, compile the
// kernel source at run time, and bind each argument by index.
// It then times the kernel with profiling events, the same way the HIP
// twin times it with hipEvents, and reports effective bandwidth.
// Compile:  g++ -O2 -o main opencl_vector_add.cpp -lOpenCL
//           Needs an OpenCL runtime (ICD) for your GPU: Nvidia's driver
//           ships one; on AMD it is part of ROCm (rocm-opencl-runtime).

#define CL_TARGET_OPENCL_VERSION 120
#include <CL/cl.h>

#include <cstdio>
#include <cstdlib>
#include <vector>

#define CL_CHECK(call)                                                    \
  do {                                                                    \
    cl_int err_ = (call);                                                 \
    if (err_ != CL_SUCCESS) {                                             \
      std::fprintf(stderr, "%s:%d: %s failed with %d\n", __FILE__,        \
                   __LINE__, #call, err_);                                \
      std::exit(1);                                                       \
    }                                                                     \
  } while (0)

// The kernel is a string. It is compiled when the program runs, by the
// driver, for whichever device was picked: that is OpenCL's portability.
static const char *kSource = R"CLC(
__kernel void vecAdd(__global const float *a, __global const float *b,
                     __global float *c, int n)
{
    int i = get_global_id(0);
    if (i < n) c[i] = a[i] + b[i];
}
)CLC";

int main() {
  const int n = 1 << 24;
  const size_t bytes = n * sizeof(float);
  std::vector<float> h_a(n, 1.0f), h_b(n, 2.0f), h_c(n);
  cl_int err;

  // 1. Platform and device: the runtime may expose several vendors at once.
  cl_platform_id platform;
  cl_device_id device;
  CL_CHECK(clGetPlatformIDs(1, &platform, nullptr));
  CL_CHECK(clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, nullptr));
  char name[256], version[256];
  CL_CHECK(clGetDeviceInfo(device, CL_DEVICE_NAME, sizeof name, name, nullptr));
  CL_CHECK(clGetDeviceInfo(device, CL_DEVICE_VERSION, sizeof version, version, nullptr));
  std::printf("Device: %s (%s)\n", name, version);

  // 2. Context and command queue (the queue is OpenCL's stream).
  cl_context ctx = clCreateContext(nullptr, 1, &device, nullptr, nullptr, &err);
  CL_CHECK(err);
  cl_command_queue queue =
      clCreateCommandQueue(ctx, device, CL_QUEUE_PROFILING_ENABLE, &err);
  CL_CHECK(err);

  // 3. Compile the kernel source now, at run time.
  cl_program prog = clCreateProgramWithSource(ctx, 1, &kSource, nullptr, &err);
  CL_CHECK(err);
  if (clBuildProgram(prog, 1, &device, "", nullptr, nullptr) != CL_SUCCESS) {
    char log[4096];
    clGetProgramBuildInfo(prog, device, CL_PROGRAM_BUILD_LOG, sizeof log, log, nullptr);
    std::fprintf(stderr, "build failed:\n%s\n", log);
    return 1;
  }
  cl_kernel kernel = clCreateKernel(prog, "vecAdd", &err);
  CL_CHECK(err);

  // 4. Buffers, created already holding the input data.
  cl_mem d_a = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, bytes, h_a.data(), &err);
  CL_CHECK(err);
  cl_mem d_b = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, bytes, h_b.data(), &err);
  CL_CHECK(err);
  cl_mem d_c = clCreateBuffer(ctx, CL_MEM_WRITE_ONLY, bytes, nullptr, &err);
  CL_CHECK(err);

  // 5. Arguments are bound one by one, by position.
  CL_CHECK(clSetKernelArg(kernel, 0, sizeof(cl_mem), &d_a));
  CL_CHECK(clSetKernelArg(kernel, 1, sizeof(cl_mem), &d_b));
  CL_CHECK(clSetKernelArg(kernel, 2, sizeof(cl_mem), &d_c));
  CL_CHECK(clSetKernelArg(kernel, 3, sizeof(int), &n));

  // 6. Launch. OpenCL takes the TOTAL work-item count, not a block count.
  const size_t local = 256;
  const size_t global = ((n + local - 1) / local) * local;
  CL_CHECK(clEnqueueNDRangeKernel(queue, kernel, 1, nullptr, &global, &local, 0, nullptr, nullptr));
  CL_CHECK(clFinish(queue));  // warm-up

  const int reps = 20;
  double totalNs = 0.0;
  for (int r = 0; r < reps; ++r) {
    cl_event ev;
    CL_CHECK(clEnqueueNDRangeKernel(queue, kernel, 1, nullptr, &global, &local, 0, nullptr, &ev));
    CL_CHECK(clWaitForEvents(1, &ev));
    cl_ulong t0, t1;
    CL_CHECK(clGetEventProfilingInfo(ev, CL_PROFILING_COMMAND_START, sizeof t0, &t0, nullptr));
    CL_CHECK(clGetEventProfilingInfo(ev, CL_PROFILING_COMMAND_END, sizeof t1, &t1, nullptr));
    totalNs += static_cast<double>(t1 - t0);
    CL_CHECK(clReleaseEvent(ev));
  }

  // 7. Copy back (blocking read) and check.
  CL_CHECK(clEnqueueReadBuffer(queue, d_c, CL_TRUE, 0, bytes, h_c.data(), 0, nullptr, nullptr));
  int wrong = 0;
  for (int i = 0; i < n; ++i) if (h_c[i] != 3.0f) ++wrong;

  const double ms = totalNs / reps / 1e6;
  std::printf("OpenCL vecAdd, n = %d: %.3f ms, %.1f GB/s, %d wrong\n",
              n, ms, 3.0 * bytes / (ms * 1e-3) / 1e9, wrong);

  clReleaseMemObject(d_c);
  clReleaseMemObject(d_b);
  clReleaseMemObject(d_a);
  clReleaseKernel(kernel);
  clReleaseProgram(prog);
  clReleaseCommandQueue(queue);
  clReleaseContext(ctx);
  return 0;
}
