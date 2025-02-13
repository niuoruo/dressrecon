from setuptools import setup, Extension
from torch.utils import cpp_extension

c_flags = [
    "-O3",
    "-std=c++17",
    "-ffast-math",
    "-mtune=native",
]
nvcc_flags = [
    "-O3",
    "-std=c++17",
    "-D__CUDA_NO_HALF_OPERATORS__",
    "-D__CUDA_NO_HALF_CONVERSIONS__",
    "-D__NO_CUDA_HALF2_OPERATORS__",
]

setup(
    name="dqtorch",  # package name, import this to use Python API
    ext_modules=[
        cpp_extension.CUDAExtension(
            name="_dqtorch_cuda",  # extension name, import this to use CUDA API
            sources=[
                "src/cuda/bindings.cpp",
                "src/cuda/dqtorch.cu",
            ],
            extra_compile_args={
                "cxx": c_flags,
                "nvcc": nvcc_flags,
            }
        ),
        cpp_extension.CppExtension(
            name="_dqtorch_cpu",  # extension name, import this to use CPU API
            sources=[
                "src/cpu/bindings.cpp",
                "src/cpu/dqtorch.cpp",
            ],
            extra_compile_args={
                "cxx": c_flags,
            }
        ),
    ],
    cmdclass={"build_ext": cpp_extension.BuildExtension}
)
