#!/bin/bash

# CUDA Laplace Solver Build Script
# This script provides convenient build options for the CUDA implementation

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Default values
BUILD_TYPE="Release"
USE_CMAKE=1
CUDA_ARCH=""
CLEAN_BUILD=0
RUN_TEST=0

# Function to print colored messages
print_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to detect GPU compute capability
detect_gpu() {
    if command -v nvidia-smi &> /dev/null; then
        COMPUTE_CAP=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -n 1 | tr -d '.')
        if [ ! -z "$COMPUTE_CAP" ]; then
            print_info "Detected GPU with compute capability: $COMPUTE_CAP"
            CUDA_ARCH="sm_$COMPUTE_CAP"
            return 0
        fi
    fi
    print_warning "Could not detect GPU compute capability"
    print_warning "Using default architectures: sm_75, sm_80, sm_86"
    CUDA_ARCH="sm_75 sm_80 sm_86"
    return 1
}

# Function to check CUDA installation
check_cuda() {
    if ! command -v nvcc &> /dev/null; then
        print_error "CUDA compiler (nvcc) not found!"
        print_error "Please install CUDA toolkit or add it to PATH"
        exit 1
    fi
    
    CUDA_VERSION=$(nvcc --version | grep "release" | sed 's/.*release //' | sed 's/,.*//')
    print_info "Found CUDA version: $CUDA_VERSION"
}

# Function to build with CMake
build_cmake() {
    print_info "Building with CMake..."
    
    mkdir -p build
    cd build
    
    if [ $CLEAN_BUILD -eq 1 ]; then
        print_info "Cleaning previous build..."
        rm -rf *
    fi
    
    # Prepare CMake arguments
    CMAKE_ARGS="-DCMAKE_BUILD_TYPE=$BUILD_TYPE"
    
    if [ ! -z "$CUDA_ARCH" ]; then
        # Convert sm_XX to just XX for CMake
        ARCH_LIST=$(echo $CUDA_ARCH | sed 's/sm_//g' | tr ' ' ';')
        CMAKE_ARGS="$CMAKE_ARGS -DCMAKE_CUDA_ARCHITECTURES=$ARCH_LIST"
    fi
    
    print_info "Running CMake with: $CMAKE_ARGS"
    cmake .. $CMAKE_ARGS
    
    print_info "Compiling..."
    cmake --build . --config $BUILD_TYPE -j$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)
    
    cd ..
    print_info "Build completed successfully!"
    print_info "Executable: build/test_cuda_solver"
}

# Function to build with Make
build_make() {
    print_info "Building with Make..."
    
    if [ $CLEAN_BUILD -eq 1 ]; then
        print_info "Cleaning previous build..."
        make clean
    fi
    
    MAKE_ARGS=""
    if [ ! -z "$CUDA_ARCH" ]; then
        MAKE_ARGS="SM_ARCH=\"$CUDA_ARCH\""
    fi
    
    print_info "Running Make with: $MAKE_ARGS"
    eval make $MAKE_ARGS
    
    print_info "Build completed successfully!"
    print_info "Executable: build/test_cuda_solver"
}

# Function to run tests
run_tests() {
    print_info "Running tests..."
    
    if [ -f "build/test_cuda_solver" ]; then
        ./build/test_cuda_solver
    else
        print_error "Test executable not found. Build first!"
        exit 1
    fi
}

# Function to show usage
show_usage() {
    cat << EOF
CUDA Laplace Solver Build Script

Usage: $0 [options]

Options:
    -h, --help              Show this help message
    -c, --clean             Clean build (remove previous build artifacts)
    -d, --debug             Build in Debug mode (default: Release)
    -m, --make              Use Make instead of CMake
    -a, --arch ARCH         Specify CUDA architecture (e.g., sm_75, "sm_75 sm_80")
    -t, --test              Run tests after building
    -i, --info              Show CUDA and GPU information only

Examples:
    $0                      # Build with CMake in Release mode
    $0 -c -t                # Clean build and run tests
    $0 -m -a sm_75          # Build with Make for specific architecture
    $0 -d                   # Build in Debug mode
    $0 -i                   # Show GPU info without building

EOF
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            show_usage
            exit 0
            ;;
        -c|--clean)
            CLEAN_BUILD=1
            shift
            ;;
        -d|--debug)
            BUILD_TYPE="Debug"
            shift
            ;;
        -m|--make)
            USE_CMAKE=0
            shift
            ;;
        -a|--arch)
            CUDA_ARCH="$2"
            shift 2
            ;;
        -t|--test)
            RUN_TEST=1
            shift
            ;;
        -i|--info)
            check_cuda
            detect_gpu
            nvidia-smi --query-gpu=name,compute_cap,driver_version,memory.total --format=csv
            exit 0
            ;;
        *)
            print_error "Unknown option: $1"
            show_usage
            exit 1
            ;;
    esac
done

# Main build process
print_info "======================================"
print_info "CUDA Laplace Solver Build Script"
print_info "======================================"

# Check prerequisites
check_cuda

# Detect GPU if architecture not specified
if [ -z "$CUDA_ARCH" ]; then
    detect_gpu
fi

# Build
if [ $USE_CMAKE -eq 1 ]; then
    build_cmake
else
    build_make
fi

# Run tests if requested
if [ $RUN_TEST -eq 1 ]; then
    echo ""
    run_tests
fi

print_info "======================================"
print_info "All done!"
print_info "======================================"
