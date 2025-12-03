from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name="pseudo_mamba_ext",
    ext_modules=[
        CUDAExtension(
            name="pseudo_mamba_ext",
            sources=[
                "csrc/pseudo_mamba_ext.cpp",
                "csrc/pseudo_mamba_ext_kernel.cu",
            ],
        )
    ],
    cmdclass={
        "build_ext": BuildExtension
    },
)
