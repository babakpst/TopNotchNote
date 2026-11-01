# A custom PyTorch CUDA operator, end to end

Companion code for [topnotchnote.com/gpu/pytorch_extension.html](https://topnotchnote.com/gpu/pytorch_extension.html).

`topnotch::scale_add_relu(x, y, s) = relu(x * s + y)` as one CUDA kernel,
registered with `TORCH_LIBRARY`, with autograd and a `torch.compile` fake
kernel added from Python.

| File | What it is |
| --- | --- |
| `fused_op.cu` | the kernel, the C++ wrapper, the operator schema and its CUDA registration |
| `fused_ext/__init__.py` | loads the library; `register_fake` and `register_autograd` |
| `setup.py` | builds `fused_ext._C` with `CUDAExtension` |
| `test_fused_op.py` | `opcheck`, gradient check against eager PyTorch, and timing |

Needs PyTorch 2.4 or newer, a CUDA toolkit whose major version matches the
one PyTorch was built with, and a C++ compiler.

```bash
TORCH_CUDA_ARCH_LIST="6.1" python setup.py build_ext --inplace   # set your GPU's compute capability
python test_fused_op.py
```

On a ROCm build of PyTorch, `CUDAExtension` hipifies `fused_op.cu` automatically.
