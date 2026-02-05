#!/bin/bash
set -e

echo "Building RBGS Test Suite..."

# Clean previous builds
rm -f rbgs_test_suite

# Compile the test suite
clang++ -std=c++17 -O3 \
    -I. \
    main_rbgs_test_suite.cpp \
    metal/LaplaceMetalSolver_core.mm \
    metal/LaplaceMetalSolver_io.mm \
    metal/LaplaceMetalSolver_multigrid.mm \
    metal/LaplaceMetalSolver_solvers.mm \
    -framework Metal -framework Foundation \
    -o rbgs_test_suite

echo "Build complete. Run with: ./rbgs_test_suite"
echo ""
echo "Test Configuration:"
echo "  Grid Size: 8193x8193 (67M points)"
echo "  Target Tolerance: 1e-8"
echo "  Boundary Conditions: L=1, R=2, T=3, B=4"
echo ""
echo "Tests Included:"
echo "  1. Pure RBGS iterations (no multigrid)"
echo "  2. RBGS as smoother in multigrid"
echo "  3. Jacobi as smoother in multigrid (reference)"
echo ""