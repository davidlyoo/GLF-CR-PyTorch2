# GLF-CR-PyTorch2 (PyTorch 2.x Compatible Version)

This repository provides a PyTorch 2.x compatible version of the original [GLF-CR](https://github.com/xxx/GLF-CR) project, a diffusion-based framework for cloud removal in satellite imagery. The primary motivation for this version is to resolve compatibility issues with modern GPU environments (e.g., CUDA 12.x, RTX A6000) while preserving the structure and methodology of the original implementation.

Key modifications include an upgrade to PyTorch 2.x, CUDA extension fixes, and dependency updates. This codebase has been tested with PyTorch 2.1+, CUDA 12.1, and modern GPUs like the NVIDIA RTX A6000.

This repository is intended for researchers and developers who wish to experiment with or build upon the GLF-CR framework in current hardware/software environments.

## What's Changed

- Upgraded from PyTorch 1.8.1 to PyTorch 2.x
- Updated `kernelconv2d_cuda` custom extension to compile with CUDA 12.x
- Fixed deprecated API usage
- Cleaned and refactored parts of the codebase for modern use

## Installation

```bash
pip install -r requirements.txt
cd cuda_ops/kernelconv2d_cuda
python setup.py install
