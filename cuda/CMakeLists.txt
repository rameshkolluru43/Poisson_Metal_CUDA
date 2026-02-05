cmake_minimum_required(VERSION 3.18)
project(LaplaceCUDASolver LANGUAGES CXX CUDA)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Set CUDA standard
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Find CUDA
find_package(CUDA REQUIRED)

# Set CUDA architectures (adjust based on your GPU)
# Common values: 50, 60, 70, 75, 80, 86, 89, 90
set(CMAKE_CUDA_ARCHITECTURES 75 80 86)

# CUDA compilation flags
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr")
set(CMAKE_CUDA_FLAGS_RELEASE "-O3")
set(CMAKE_CUDA_FLAGS_DEBUG "-g -G")

# Include directories
include_directories(${CMAKE_CURRENT_SOURCE_DIR})
include_directories(${CUDA_INCLUDE_DIRS})

# Source files
set(CUDA_SOURCES
    laplace_kernels.cu
    kernel_wrappers.cu
)

set(CPP_SOURCES
    LaplaceCUDASolver.cpp
)

# Create library
add_library(laplace_cuda_lib STATIC
    ${CUDA_SOURCES}
    ${CPP_SOURCES}
)

# Set target properties
set_target_properties(laplace_cuda_lib PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE ON
)

# Link CUDA runtime
target_link_libraries(laplace_cuda_lib ${CUDA_LIBRARIES})

# Test executable
add_executable(test_cuda_solver test_cuda_solver.cpp)
target_link_libraries(test_cuda_solver laplace_cuda_lib)

# Set properties for the test executable
set_target_properties(test_cuda_solver PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

# Install targets
install(TARGETS laplace_cuda_lib test_cuda_solver
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    RUNTIME DESTINATION bin
)

install(FILES LaplaceCUDASolver.hpp
    DESTINATION include
)

# Print configuration info
message(STATUS "CUDA Version: ${CUDA_VERSION}")
message(STATUS "CUDA Architectures: ${CMAKE_CUDA_ARCHITECTURES}")
message(STATUS "CUDA Include Dirs: ${CUDA_INCLUDE_DIRS}")
