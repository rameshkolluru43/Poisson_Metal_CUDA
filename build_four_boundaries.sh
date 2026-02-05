#!/bin/bash

set -e  # Exit on any error

echo "Building Four Boundaries Test..."

# Compile the focused four boundaries test
clang++ -std=c++17 -O3 \
    -I. \
    -fobjc-arc \
    -framework Metal -framework Foundation -framework CoreGraphics \
    main_four_boundaries.cpp \
    metal/LaplaceMetalSolver_core.mm \
    metal/LaplaceMetalSolver_io.mm \
    metal/LaplaceMetalSolver_multigrid.mm \
    metal/LaplaceMetalSolver_solvers.mm \
    -o four_boundaries_test

echo "Build completed!"

# Run with provided arguments or default to 8193
if [ $# -eq 0 ]; then
    echo "Running with default 8193x8193 grid..."
    ./four_boundaries_test
else
    echo "Running with custom arguments: $*"
    ./four_boundaries_test "$@"
fi