// fused_op.cu
// Companion example for: topnotchnote.com/gpu/pytorch_extension.html
//
// A custom PyTorch operator with a CUDA kernel:
//   topnotch::scale_add_relu(x, y, s) = relu(x * s + y)
// In eager PyTorch this is three ops and three kernels (mul, add, relu),
// each reading and writing a full tensor. Here it is one kernel that reads
// x and y once and writes the result once.
//
// The operator is defined with TORCH_LIBRARY and its CUDA kernel registered
// with TORCH_LIBRARY_IMPL; autograd and torch.compile support are added from
// Python (see fused_ext/__init__.py). On a ROCm build of PyTorch,
// CUDAExtension hipifies this file automatically.
//
// Build:  pip install --no-build-isolation -e .   (or: python setup.py build_ext --inplace)

#include <Python.h>
#include <ATen/cuda/CUDAContext.h>
#include <torch/library.h>
#include <torch/extension.h>

namespace topnotch {

__global__ void scale_add_relu_kernel(const float* x, const float* y, float s,
                                      float* out, int64_t n) {
  int64_t i = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
  if (i < n) {
    float v = x[i] * s + y[i];
    out[i] = v > 0.f ? v : 0.f;
  }
}

at::Tensor scale_add_relu_cuda(const at::Tensor& x, const at::Tensor& y,
                               double s) {
  TORCH_CHECK(x.sizes() == y.sizes(), "x and y must have the same shape");
  TORCH_CHECK(x.dtype() == at::kFloat && y.dtype() == at::kFloat,
              "float32 only");
  TORCH_INTERNAL_ASSERT(x.device().type() == at::DeviceType::CUDA);
  TORCH_INTERNAL_ASSERT(y.device().type() == at::DeviceType::CUDA);
  at::Tensor xc = x.contiguous(), yc = y.contiguous();
  at::Tensor out = at::empty_like(xc);  // from the caching allocator
  int64_t n = xc.numel();
  if (n == 0) return out;
  int threads = 256;
  int64_t blocks = (n + threads - 1) / threads;
  // Launch on PyTorch's current stream, so the op is ordered with the rest.
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  scale_add_relu_kernel<<<blocks, threads, 0, stream>>>(
      xc.data_ptr<float>(), yc.data_ptr<float>(), (float)s,
      out.data_ptr<float>(), n);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}

TORCH_LIBRARY(topnotch, m) {
  m.def("scale_add_relu(Tensor x, Tensor y, float s) -> Tensor");
}

TORCH_LIBRARY_IMPL(topnotch, CUDA, m) {
  m.impl("scale_add_relu", &scale_add_relu_cuda);
}

}  // namespace topnotch

// An empty Python module, so `import fused_ext._C` loads the library and
// runs the registrations above.
extern "C" {
PyObject* PyInit__C(void) {
  static struct PyModuleDef module_def = {PyModuleDef_HEAD_INIT, "_C", NULL,
                                          -1, NULL};
  return PyModule_Create(&module_def);
}
}
