#!/bin/zsh
# Build and run LaplaceMetalSolver unit tests

set -e

# Change to the directory of this script so relative paths work regardless of invocation CWD
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# Compile the test file
clang++ -std=c++17 -fobjc-arc -ObjC++ -framework Metal -framework Foundation \
    metal/LaplaceMetalSolver_core.mm \
    metal/LaplaceMetalSolver_multigrid.mm \
    metal/LaplaceMetalSolver_solvers.mm \
    metal/LaplaceMetalSolver_io.mm \
    test_LaplaceMetalSolver.cpp -o test_laplace_solver

# Run the test executable
./test_laplace_solver
