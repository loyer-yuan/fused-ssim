from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension

NVCC_FLAGS = [
    "-lineinfo",
]

CXX_FLAGS = [
    "-std=c++17",
]

setup(
    name="fused_ssim",
    packages=['fused_ssim'],
    ext_modules=[
        CUDAExtension(
            name="fused_ssim_cuda",
            extra_compile_args={
                "nvcc": NVCC_FLAGS,
            },
            sources=[
            "ssim.cu",
            "ext.cpp"]),
        CUDAExtension(
            name="fused_ssim_cuda_opt",
            extra_compile_args={
                "nvcc": NVCC_FLAGS,
                "cxx": CXX_FLAGS,
            },
            sources=[
            "ssim_opt.cu",
            "ext_opt.cpp"])
        ],
    cmdclass={
        'build_ext': BuildExtension
    }
)
