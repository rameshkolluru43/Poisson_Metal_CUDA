# CUDA Laplace Solver Makefile
# Adjust CUDA_PATH and SM_ARCH according to your system

# CUDA installation path
CUDA_PATH ?= /usr/local/cuda

# CUDA compiler
NVCC = $(CUDA_PATH)/bin/nvcc

# C++ compiler
CXX = g++

# Target GPU architecture (adjust for your GPU)
# Common values:
#   sm_50  - Maxwell (GTX 9xx)
#   sm_60  - Pascal (GTX 10xx, Quadro Pxxxx)
#   sm_70  - Volta (V100, Titan V)
#   sm_75  - Turing (RTX 20xx, T4)
#   sm_80  - Ampere (A100, RTX 30xx)
#   sm_86  - Ampere (RTX 30xx mobile)
#   sm_89  - Ada Lovelace (RTX 40xx)
#   sm_90  - Hopper (H100)
SM_ARCH = sm_75 sm_80 sm_86

# Compiler flags
NVCC_FLAGS = -std=c++14 $(foreach sm,$(SM_ARCH),-gencode arch=compute_$(subst sm_,,$(sm)),code=sm_$(subst sm_,,$(sm))) -O3 --expt-relaxed-constexpr
CXX_FLAGS = -std=c++17 -O3 -Wall -I$(CUDA_PATH)/include
LDFLAGS = -L$(CUDA_PATH)/lib64 -lcudart -lm

# Directories
BUILD_DIR = build
OBJ_DIR = $(BUILD_DIR)/obj

# Source files
CUDA_SOURCES = laplace_kernels.cu kernel_wrappers.cu
CPP_SOURCES = LaplaceCUDASolver.cpp
TEST_SOURCE = test_cuda_solver.cpp

# Object files
CUDA_OBJS = $(addprefix $(OBJ_DIR)/,$(CUDA_SOURCES:.cu=.o))
CPP_OBJS = $(addprefix $(OBJ_DIR)/,$(CPP_SOURCES:.cpp=.o))
TEST_OBJ = $(OBJ_DIR)/test_cuda_solver.o

# Targets
TARGET = $(BUILD_DIR)/test_cuda_solver
LIB_TARGET = $(BUILD_DIR)/liblaplace_cuda.a

# Default target
all: directories $(TARGET)

# Create directories
directories:
	@mkdir -p $(OBJ_DIR)

# Build static library
$(LIB_TARGET): $(CUDA_OBJS) $(CPP_OBJS)
	@echo "Creating static library: $@"
	ar rcs $@ $^

# Build test executable
$(TARGET): $(TEST_OBJ) $(LIB_TARGET)
	@echo "Linking test executable: $@"
	$(NVCC) $(NVCC_FLAGS) $< $(LIB_TARGET) $(LDFLAGS) -o $@

# Compile CUDA sources
$(OBJ_DIR)/%.o: %.cu
	@echo "Compiling CUDA: $<"
	$(NVCC) $(NVCC_FLAGS) -dc $< -o $@

# Compile C++ sources
$(OBJ_DIR)/%.o: %.cpp
	@echo "Compiling C++: $<"
	$(CXX) $(CXX_FLAGS) -c $< -o $@

# Clean build artifacts
clean:
	rm -rf $(BUILD_DIR)

# Run tests
test: $(TARGET)
	@echo "Running CUDA Laplace Solver tests..."
	$(TARGET)

# Show CUDA info
info:
	@echo "CUDA Path: $(CUDA_PATH)"
	@echo "NVCC: $(NVCC)"
	@echo "Target architectures: $(SM_ARCH)"
	@$(NVCC) --version

# Help
help:
	@echo "Available targets:"
	@echo "  all        - Build everything (default)"
	@echo "  clean      - Remove build artifacts"
	@echo "  test       - Build and run tests"
	@echo "  info       - Show CUDA configuration"
	@echo "  help       - Show this help message"
	@echo ""
	@echo "Configuration:"
	@echo "  Set SM_ARCH to specify target GPU architectures"
	@echo "  Set CUDA_PATH to specify CUDA installation directory"
	@echo ""
	@echo "Example: make SM_ARCH=\"sm_75\" CUDA_PATH=/opt/cuda"

.PHONY: all directories clean test info help
