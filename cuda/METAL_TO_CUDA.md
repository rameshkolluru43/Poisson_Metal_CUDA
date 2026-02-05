# Metal to CUDA Translation Reference

This document provides a quick reference for the differences between Metal and CUDA implementations in this project.

## Language Feature Mapping

| Feature | Metal Shading Language | CUDA C++ |
|---------|----------------------|----------|
| **Kernel declaration** | `kernel void` | `__global__ void` |
| **Device function** | `inline` or no qualifier | `__device__` or `__device__ __forceinline__` |
| **Thread index** | `thread_position_in_grid` | `blockIdx * blockDim + threadIdx` |
| **Thread group index** | `thread_position_in_threadgroup` | `threadIdx` |
| **Grid dimensions** | Passed as attribute | Computed from blocks/threads |
| **Shared memory** | `threadgroup` | `__shared__` |
| **Synchronization** | `threadgroup_barrier(mem_flags::mem_threadgroup)` | `__syncthreads()` |
| **Math min/max** | `min(a, b)`, `max(a, b)` | `fminf(a, b)`, `fmaxf(a, b)` |
| **Clamp** | `clamp(x, lo, hi)` | `fminf(fmaxf(x, lo), hi)` |
| **Linear interpolation** | `mix(a, b, t)` | `(1-t)*a + t*b` |

## Memory Management

### Metal
```cpp
// Host-side (Objective-C++)
id<MTLBuffer> buffer = [device newBufferWithLength:size options:MTLResourceStorageModeShared];
float* data = (float*)[buffer contents];
```

### CUDA
```cpp
// Host-side (C++)
float* d_buffer;
cudaMalloc(&d_buffer, size);
cudaMemcpy(d_buffer, h_data, size, cudaMemcpyHostToDevice);
```

## Kernel Launch

### Metal
```objc
id<MTLCommandBuffer> commandBuffer = [commandQueue commandBuffer];
id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
[encoder setComputePipelineState:pipeline];
[encoder setBuffer:buffer offset:0 atIndex:0];
[encoder dispatchThreadgroups:gridSize threadsPerThreadgroup:blockSize];
[encoder endEncoding];
[commandBuffer commit];
```

### CUDA
```cpp
dim3 blockSize(16, 16);
dim3 gridSize((nx + 15) / 16, (ny + 15) / 16);
kernel<<<gridSize, blockSize>>>(d_buffer, params);
cudaDeviceSynchronize();
```

## Thread Indexing

### Metal
```cpp
kernel void example(
    uint2 tid [[thread_position_in_grid]],
    uint2 tid_group [[thread_position_in_threadgroup]],
    uint2 tgrp_pos [[threadgroup_position_in_grid]]
) {
    uint i = tid.x;
    uint j = tid.y;
    // ...
}
```

### CUDA
```cpp
__global__ void example() {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int j = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int tx = threadIdx.x;
    unsigned int ty = threadIdx.y;
    // ...
}
```

## Shared Memory

### Metal
```cpp
kernel void example(/* ... */) {
    threadgroup float shared_data[256];
    
    // Load data
    shared_data[tid] = input[tid];
    
    // Synchronize
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // Use data
    float result = shared_data[tid] + shared_data[tid + 1];
}
```

### CUDA
```cpp
__global__ void example() {
    __shared__ float shared_data[256];
    
    unsigned int tid = threadIdx.x;
    
    // Load data
    shared_data[tid] = input[tid];
    
    // Synchronize
    __syncthreads();
    
    // Use data
    float result = shared_data[tid] + shared_data[tid + 1];
}
```

## Atomic Operations

### Metal
```cpp
#include <metal_atomic>
atomic_fetch_add_explicit(address, value, memory_order_relaxed);
```

### CUDA
```cpp
atomicAdd(address, value);
```

## Reduction Example

### Metal
```cpp
kernel void reduce(
    device const float* in [[buffer(0)]],
    device float* out [[buffer(1)]],
    uint tid [[thread_index_in_threadgroup]],
    uint tcount [[threads_per_threadgroup]]
) {
    threadgroup float scratch[256];
    
    scratch[tid] = in[tid];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    for (uint stride = tcount >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) {
            scratch[tid] += scratch[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    
    if (tid == 0) out[0] = scratch[0];
}
```

### CUDA
```cpp
__global__ void reduce(
    const float* in,
    float* out
) {
    __shared__ float scratch[256];
    
    unsigned int tid = threadIdx.x;
    
    scratch[tid] = in[tid];
    __syncthreads();
    
    for (unsigned int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) {
            scratch[tid] += scratch[tid + stride];
        }
        __syncthreads();
    }
    
    if (tid == 0) out[0] = scratch[0];
}
```

## Buffer/Pointer Access

### Metal
```cpp
kernel void example(
    device float* output [[buffer(0)]],
    device const float* input [[buffer(1)]],
    constant Uniforms& params [[buffer(2)]]
) {
    output[idx] = input[idx] * params.scale;
}
```

### CUDA
```cpp
__global__ void example(
    float* output,
    const float* input,
    Uniforms params  // Passed by value or use constant memory
) {
    output[idx] = input[idx] * params.scale;
}
```

## Constant Memory

### Metal
```cpp
constant float PI = 3.14159f;  // Compile-time constant

kernel void example(
    constant Uniforms& params [[buffer(0)]]  // Runtime constant
) {
    float result = params.value * PI;
}
```

### CUDA
```cpp
__constant__ float PI = 3.14159f;  // Device constant memory

__global__ void example(Uniforms params) {  // Passed by value
    float result = params.value * PI;
}
```

## Error Checking

### Metal
```objc
NSError* error = nil;
id<MTLLibrary> library = [device newLibraryWithSource:source 
                                              options:nil 
                                                error:&error];
if (!library) {
    NSLog(@"Error: %@", error);
}
```

### CUDA
```cpp
cudaError_t err = cudaMalloc(&ptr, size);
if (err != cudaSuccess) {
    fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(err));
}

// Or use macro:
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            /* handle error */ \
        } \
    } while (0)
```

## Performance Considerations

| Aspect | Metal | CUDA |
|--------|-------|------|
| **Warp/SIMD size** | 32 threads (Apple Silicon) | 32 threads (warp) |
| **Max threads/block** | 1024 | 1024 |
| **Shared memory** | 32-64 KB | 48-96 KB (configurable) |
| **Launch overhead** | Low (command encoding) | Very low (<<<>>>>) |
| **Memory coherency** | Unified on Apple Silicon | Separate host/device |

## Type Conversions

| Metal | CUDA |
|-------|------|
| `uint` | `unsigned int` |
| `ushort` | `unsigned short` |
| `uchar` | `unsigned char` |
| `float2` | `float2` |
| `float3` | `float3` |
| `float4` | `float4` |

## Vector Operations

### Metal
```cpp
float2 a = float2(1.0f, 2.0f);
float2 b = float2(3.0f, 4.0f);
float2 c = a + b;  // Element-wise
float d = dot(a, b);
```

### CUDA
```cpp
float2 a = make_float2(1.0f, 2.0f);
float2 b = make_float2(3.0f, 4.0f);
// No built-in element-wise ops, write manually:
float2 c;
c.x = a.x + b.x;
c.y = a.y + b.y;
// Or use utility functions
```

## Key Differences

1. **Threading Model**
   - Metal: Threadgroups → Threads
   - CUDA: Grids → Blocks → Threads

2. **Memory Model**
   - Metal: Unified memory on Apple Silicon (shared between CPU/GPU)
   - CUDA: Separate device memory (requires explicit transfers)

3. **Compilation**
   - Metal: Runtime compilation from source strings or precompiled libraries
   - CUDA: Compile-time with nvcc

4. **Platform**
   - Metal: macOS/iOS only
   - CUDA: NVIDIA GPUs on Windows/Linux/macOS

5. **Debugging**
   - Metal: Xcode GPU debugger, frame capture
   - CUDA: cuda-gdb, Nsight, printf in kernels

## Porting Checklist

When porting from Metal to CUDA:

- [ ] Replace `kernel` with `__global__`
- [ ] Change thread indexing from attributes to computed values
- [ ] Replace `threadgroup` with `__shared__`
- [ ] Change `threadgroup_barrier()` to `__syncthreads()`
- [ ] Update math functions (min/max/clamp)
- [ ] Convert buffer attributes to pointer parameters
- [ ] Add CUDA memory allocation/transfer code
- [ ] Update kernel launch syntax
- [ ] Add error checking
- [ ] Adjust build system (Makefile/CMake)

## Example: Complete Kernel Translation

### Metal
```cpp
kernel void add_arrays(
    device const float* a [[buffer(0)]],
    device const float* b [[buffer(1)]],
    device float* c [[buffer(2)]],
    constant uint& n [[buffer(3)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= n) return;
    c[tid] = a[tid] + b[tid];
}
```

### CUDA
```cpp
__global__ void add_arrays(
    const float* a,
    const float* b,
    float* c,
    unsigned int n
) {
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n) return;
    c[tid] = a[tid] + b[tid];
}

// Launch:
// dim3 block(256);
// dim3 grid((n + 255) / 256);
// add_arrays<<<grid, block>>>(d_a, d_b, d_c, n);
```

## Resources

- [Metal Shading Language Specification](https://developer.apple.com/metal/Metal-Shading-Language-Specification.pdf)
- [CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
- [CUDA C++ Best Practices](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/)
