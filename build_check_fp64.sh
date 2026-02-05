#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

OUT=check_fp64

echo "Building FP64 Metal support check..."
clang++ -std=c++17 -fobjc-arc \
  -framework Metal -framework Foundation \
  check_fp64.mm -o "$OUT"

echo "Running $OUT..."
./"$OUT"
