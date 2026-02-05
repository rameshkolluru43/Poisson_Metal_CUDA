# Poisson_Metal_CUDA
Solving Laplace or Poisson equation using Metal (Apple GPU) or CUDA (NVIDIA GPU).

## Overview
This repository contains a Metal-based Laplace/Poisson solver and a full CUDA port. The CUDA implementation mirrors the Metal kernels and provides a C++ solver interface, test suite, and build systems (CMake and Make).

## Features
- Metal GPU solver and kernels for Laplace/Poisson problems
- CUDA port with equivalent kernels and solver API
- Multigrid support (FMG/V-cycle), Jacobi, RBGS, SOR
- Residual computation, restriction, prolongation, and reduction kernels
- Build scripts and tests for CUDA

## Repository layout
- `metal/`: Metal kernels and solver implementation
- `cuda/`: CUDA kernels, wrappers, solver API, tests, and build files
- `opencl/`: OpenCL reference implementation
- `docs/`: Reports and plots

## Build (CUDA)
From the repository root:

```bash
cd cuda
./build.sh
```

Alternative builds:

```bash
cd cuda
mkdir -p build && cd build
cmake ..
cmake --build .
```

```bash
cd cuda
make
```

## Run tests (CUDA)
```bash
cd cuda
./build/test_cuda_solver
```

## Documentation
See the CUDA docs in:
- `cuda/README.md`
- `cuda/QUICKSTART.md`
- `cuda/METAL_TO_CUDA.md`

## Notes
- CUDA requires NVIDIA GPU and a recent CUDA toolkit.
- Metal requires macOS with a compatible GPU.
