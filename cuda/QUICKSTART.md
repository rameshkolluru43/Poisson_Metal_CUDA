# CUDA Laplace Solver - Quick Start Guide

This is a 5-minute quick start to get you up and running with the CUDA Laplace solver.

## Prerequisites Check

```bash
# Check if CUDA is installed
nvcc --version

# Check if you have an NVIDIA GPU
nvidia-smi

# Should show your GPU model and compute capability
```

If these commands fail, you need to install the CUDA toolkit from [NVIDIA's website](https://developer.nvidia.com/cuda-downloads).

## Build in 3 Steps

### Step 1: Navigate to the CUDA directory
```bash
cd cuda/
```

### Step 2: Build (choose one method)

**Option A: Automatic build script (recommended)**
```bash
./build.sh -t
```
This will detect your GPU, build, and run tests automatically.

**Option B: CMake**
```bash
mkdir build && cd build
cmake ..
make
cd ..
```

**Option C: Make**
```bash
make
```

### Step 3: Run
```bash
./build/test_cuda_solver
```

You should see output like:
```
CUDA Laplace Solver Test
========================

Grid size: 128 x 128
Grid spacing: dx = 0.0079, dy = 0.0079

Test 1: Jacobi Solver
---------------------
Iterations: 245
Final residual: 9.87234e-07

...
```

## Troubleshooting

**Problem: `nvcc: command not found`**
```bash
# Add CUDA to PATH (adjust path for your system)
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
```

**Problem: `No CUDA-capable device is detected`**
- Make sure you have an NVIDIA GPU
- Check drivers: `nvidia-smi`
- Reinstall NVIDIA drivers if needed

**Problem: Build fails with architecture error**
```bash
# Find your GPU's compute capability
nvidia-smi --query-gpu=compute_cap --format=csv

# Build for specific architecture (e.g., 7.5 -> sm_75)
./build.sh -a sm_75
```

**Problem: Out of memory**
- Reduce grid size in test_cuda_solver.cpp
- Check available GPU memory: `nvidia-smi`

## Next Steps

### Use in Your Code

```cpp
#include "LaplaceCUDASolver.hpp"

int main() {
    // Create solver for 256x256 grid
    LaplaceCUDASolver solver(256, 256, 0.01f, 0.01f);
    
    // Set boundary conditions
    std::vector<unsigned char> bcMask(256 * 256);
    std::vector<float> bcVals(256 * 256);
    // ... initialize BC arrays ...
    solver.setBoundaryConditions(bcMask, bcVals);
    
    // Set initial guess
    std::vector<float> u(256 * 256, 0.0f);
    solver.setInitialSolution(u);
    
    // Solve!
    int iters = solver.solveRBGS(1000, 1e-6f);
    
    // Get result
    solver.getSolution(u);
    
    return 0;
}
```

### Integrate with Your Project

1. Copy the `cuda/` folder to your project
2. Link against `liblaplace_cuda.a`
3. Include `LaplaceCUDASolver.hpp`

### Performance Tuning

- Use larger grids (512x512 or bigger) for better GPU utilization
- Try different solvers: RBGS is usually fastest
- Adjust omega parameter (0.8-1.2) for your problem
- Use FMG for very large problems

## Common Use Cases

### Solving Laplace's Equation
```cpp
solver.solveRBGS(1000, 1e-6f, 1.0f);
```

### Heat Equation (steady state)
```cpp
// Same as Laplace - already supported
solver.solveFMG(2, 2, 4, 1.0f);
```

### Custom Boundary Conditions
```cpp
// Dirichlet: bcMask[i] = 1, bcVals[i] = value
// Neumann: handle in pre/post-processing
```

## Getting Help

- Read [README.md](README.md) for full documentation
- Check [METAL_TO_CUDA.md](METAL_TO_CUDA.md) for implementation details
- Compare with Metal version in `../metal/`

## Performance Expectations

Typical performance on an RTX 3080 for 512×512 grid:

| Method | Iterations | Time | Residual |
|--------|------------|------|----------|
| Jacobi | ~1000 | 50ms | 1e-6 |
| RBGS | ~500 | 40ms | 1e-6 |
| FMG | ~10 V-cycles | 80ms | 1e-6 |

Your results will vary based on GPU model and problem.

## What's Included

All Metal shader kernels have been ported:

✅ Jacobi iteration  
✅ Red-Black Gauss-Seidel  
✅ Successive Over-Relaxation  
✅ Full Multigrid (restriction/prolongation)  
✅ Residual computation  
✅ Fused tiled kernels  
✅ Optimized reductions  

## Quick Commands Reference

```bash
# Build
./build.sh

# Clean build
./build.sh -c

# Build and test
./build.sh -t

# Debug build
./build.sh -d

# Specific architecture
./build.sh -a sm_80

# Show GPU info
./build.sh -i

# Help
./build.sh -h
```

## Success!

If you got here and tests passed, you're all set! 🎉

The CUDA Laplace solver is now ready to use in your projects.
