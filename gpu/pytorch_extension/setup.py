# Builds fused_ext._C from fused_op.cu with PyTorch's CUDAExtension.
# Set TORCH_CUDA_ARCH_LIST (e.g. "6.1" or "9.0") to choose the GPUs to
# compile for; by default it targets the GPUs visible at build time.
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name="fused_ext",
    packages=["fused_ext"],
    ext_modules=[
        CUDAExtension(
            name="fused_ext._C",
            sources=["fused_op.cu"],
            extra_compile_args={"cxx": ["-O2"], "nvcc": ["-O2"]},
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
