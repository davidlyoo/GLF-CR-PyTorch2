import os
import torch

from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

USE_CXX11_ABI = 0 if not hasattr(torch._C, '_GLIBCXX_USE_CXX11_ABI') else torch._C._GLIBCXX_USE_CXX11_ABI
ABI_FLAG = f'-D_GLIBCXX_USE_CXX11_ABI={int(USE_CXX11_ABI)}'

cxx_args = ['-std=c++17', ABI_FLAG]

nvcc_args = [
    '-gencode', 'arch=compute_50,code=sm_50',
    '-gencode', 'arch=compute_52,code=sm_52',
    '-gencode', 'arch=compute_60,code=sm_60',
    '-gencode', 'arch=compute_61,code=sm_61',
    '-gencode', 'arch=compute_70,code=sm_70',
    '-gencode', 'arch=compute_75,code=sm_75',
    '-gencode', 'arch=compute_86,code=sm_86',
    '-ccbin=/home/capstone_rs/miniconda3/envs/cuda113/bin/gcc'
]

setup(
    version='1.0.0',
    name='kernelconv2d_cuda',
    ext_modules=[
        CUDAExtension('kernelconv2d_cuda', [
            'KernelConv2D_cuda.cpp',
            'KernelConv2D_kernel.cu'
        ], extra_compile_args={
            'cxx': cxx_args, 
            'nvcc': nvcc_args,
            }
            )
    ],
    cmdclass={
        'build_ext': BuildExtension
    })
