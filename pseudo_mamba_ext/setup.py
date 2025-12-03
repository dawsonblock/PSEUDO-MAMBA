import torch
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

# Standard compilation flags
cxx_flags = ["-O3", "-std=c++17"]
nvcc_flags = ["-O3", "-std=c++17"]

if torch.cuda.is_available():
    # Add architecture flags if known
    # This helps avoid "no kernel image is available for execution" errors
    pass

setup(
    name="pseudo_mamba_ext",
    ext_modules=[
        CUDAExtension(
            name="pseudo_mamba_ext",
            sources=[
                "csrc/pseudo_mamba_ext.cpp",
                "csrc/pseudo_mamba_ext_kernel.cu",
            ],
            extra_compile_args={
                "cxx": cxx_flags,
                "nvcc": nvcc_flags,
            },
        )
    ],
    cmdclass={
        "build_ext": BuildExtension
    },
)
