// IO, BCs, and convenience methods matching public API
#import <Metal/Metal.h>
#include "LaplaceMetalSolver.hpp"
#include "LaplaceMetalSolverInternal.hpp"
#include <vector>
#include <limits>
#include <cstring>

/*
 @brief Initialize the level-0 device field with an optional host-provided scalar field.
 @param u0 Host-side field values in row-major order. If u0.size() >= nx_ * ny_, its first
           nx_ * ny_ elements are uploaded; otherwise the device field is zero-initialized.
 @details
 - Computes total grid points n = nx_ * ny_ and selects the source buffer:
   uses u0 if it has at least n elements; otherwise allocates an n-length zero-filled buffer.
 - Creates a shared-mode Metal staging buffer and copies the chosen host data into it.
 - Issues a blit copy from the staging buffer to the device-resident level-0 field buffer (levels_[0].uA).
 - Commits the command buffer and synchronously waits for completion to ensure the device buffer
   is fully initialized before returning.
 @note The staging buffer uses MTLResourceStorageModeShared to minimize CPU-GPU transfer overhead
       on Apple platforms. The final copy is performed via a MTLBlitCommandEncoder.
 @note This call blocks via waitUntilCompleted(); remove or defer waiting if higher throughput
       and explicit synchronization are desired elsewhere.
 @pre device_ and queue_ must reference valid Metal objects; levels_[0].uA must be a valid
      MTLBuffer with at least fieldBytes_ bytes. fieldBytes_ should be >= nx_ * ny_ * sizeof(float).
 @post The device buffer for level 0 contains either the provided initial field (truncated/used up to n)
       or zeros if u0 is too small.
 @warning No runtime validation ensures fieldBytes_ matches nx_ * ny_ * sizeof(float); a mismatch
          may cause incomplete copies or undefined behavior.
 @complexity Time: O(nx_ * ny_) for the data copy. Space: O(nx_ * ny_) temporary for the staging buffer.
 @thread_safety Not thread-safe with respect to concurrent calls on the same solver instance.
*/
void LaplaceMetalSolver::setInitial(const std::vector<float>& u0){
    id<MTLDevice> dev=(__bridge id<MTLDevice>)device_;
    id<MTLCommandQueue> queue=(__bridge id<MTLCommandQueue>)queue_;
    size_t n=size_t(nx_)*ny_;
    const float* src = u0.size()>=n ? u0.data() : nullptr;
    std::vector<float> zeros; if(!src){ zeros.assign(n,0.0f); src=zeros.data(); }
    id<MTLBuffer> staging=[dev newBufferWithBytes:src length:fieldBytes_ options:MTLResourceStorageModeShared];
    id<MTLCommandBuffer> cb=[queue commandBuffer]; id<MTLBlitCommandEncoder> blit=[cb blitCommandEncoder];
    [blit copyFromBuffer:staging sourceOffset:0 toBuffer:(__bridge id<MTLBuffer>)levels_[0].uA destinationOffset:0 size:fieldBytes_];
    [blit endEncoding]; [cb commit]; [cb waitUntilCompleted];
}

/*
 @brief Sets Dirichlet boundary conditions by uploading mask and values to GPU memory and computing their range.
 
 @details
 This method prepares and uploads per-cell Dirichlet boundary condition data for a 2D grid of size `nx_ * ny_` to
 Metal buffers, and derives the minimum and maximum boundary values present. The logic proceeds as follows:
 - Determine the expected element count `n = nx_ * ny_`.
 - Validate input vectors:
   - If `mask.size() < n`, internally substitute a zero-initialized mask of length `n` (i.e., no constrained cells).
   - If `values.size() < n`, internally substitute a zero-initialized values array of length `n`.
 - Create CPU-visible staging buffers (shared storage) and copy the selected host data (original or substituted).
 - Issue a Metal blit to copy staging data into the solver’s device-resident buffers (`levels_[0].bcMask` and
   `levels_[0].bcVals`), then commit and block until completion to ensure the GPU state is synchronized.
 - Scan the effective host-side arrays to compute `bcMin_` and `bcMax_` across cells where `mask[i] != 0`.
   If no cells are constrained, or if any reduction yields a non-finite result (e.g., NaN/Inf), both `bcMin_`
   and `bcMax_` are reset to `0.0f`.

 Semantics of inputs:
 - For each cell `i` in `[0, n)`, `mask[i] == 0` means unconstrained; any non-zero value means the corresponding
   Dirichlet value `values[i]` is applied at that cell.
 
 @param mask
 A vector of length at least `nx_ * ny_` specifying which cells are constrained (non-zero) versus unconstrained (zero).
 If shorter than required, a zero-filled mask is used instead (all cells unconstrained).

 @param values
 A vector of length at least `nx_ * ny_` containing Dirichlet values per cell. If shorter than required, a zero-filled
 value array is used instead. Only entries where `mask[i] != 0` are considered for the min/max reduction.

 @pre
 - `device_` and `queue_` are valid Metal objects.
 - `levels_[0].bcMask` and `levels_[0].bcVals` are valid `MTLBuffer`s with capacities of at least `n*sizeof(uint8_t)`
   and `n*sizeof(float)` respectively.
 - `nx_` and `ny_` have been initialized to positive dimensions.

 @post
 - `levels_[0].bcMask` and `levels_[0].bcVals` contain the uploaded mask and value data (from inputs or zero-fallbacks).
 - `bcMin_` and `bcMax_` reflect the finite min/max of all constrained values; if none are constrained or results are
   non-finite, both are set to `0.0f`.

 @note
 - The operation is synchronous with respect to the GPU blit: it waits for completion before returning.
 - Non-finite values in `values` (NaN/Inf) can cause the reduction to be treated as invalid, resetting both bounds to zero.
 - Fallback-zero behavior is intentional for robustness; callers that require strict validation should ensure `mask` and
   `values` are sized to at least `nx_*ny_`.

 @warning
 - Not thread-safe with respect to other operations that mutate or consume the same Metal resources concurrently.
 - Blocking on the command buffer may impact frame latency if called on timing-critical threads.

 @see bcMin_, bcMax_, levels_, nx_, ny_

 @complexity
 - Time: O(n) CPU for the reduction plus O(n) GPU copy for each buffer.
 - Space: O(n) additional temporary storage for staging buffers.
*/
void LaplaceMetalSolver::setDirichletBC(const std::vector<uint8_t>& mask, const std::vector<float>& values){
    id<MTLDevice> dev=(__bridge id<MTLDevice>)device_;
    id<MTLCommandQueue> queue=(__bridge id<MTLCommandQueue>)queue_;
    size_t n=size_t(nx_)*ny_;
    std::vector<uint8_t> maskLocal; std::vector<float> valsLocal;
    const uint8_t* maskSrc = (mask.size()>=n ? mask.data() : nullptr);
    const float* valsSrc = (values.size()>=n ? values.data() : nullptr);
    if(!maskSrc){ maskLocal.assign(n,0u); maskSrc=maskLocal.data(); }
    if(!valsSrc){ valsLocal.assign(n,0.0f); valsSrc=valsLocal.data(); }
    id<MTLBuffer> maskStaging=[dev newBufferWithBytes:maskSrc length:n*sizeof(uint8_t) options:MTLResourceStorageModeShared];
    id<MTLBuffer> valStaging=[dev newBufferWithBytes:valsSrc length:n*sizeof(float) options:MTLResourceStorageModeShared];
    id<MTLCommandBuffer> cb=[queue commandBuffer]; id<MTLBlitCommandEncoder> blit=[cb blitCommandEncoder];
    [blit copyFromBuffer:maskStaging sourceOffset:0 toBuffer:(__bridge id<MTLBuffer>)levels_[0].bcMask destinationOffset:0 size:n*sizeof(uint8_t)];
    [blit copyFromBuffer:valStaging sourceOffset:0 toBuffer:(__bridge id<MTLBuffer>)levels_[0].bcVals destinationOffset:0 size:n*sizeof(float)];
    [blit endEncoding]; [cb commit]; [cb waitUntilCompleted];
    bcMin_=std::numeric_limits<float>::infinity(); bcMax_=-std::numeric_limits<float>::infinity();
    for (size_t i=0;i<n;++i){ if(maskSrc[i]){ bcMin_=std::min(bcMin_, valsSrc[i]); bcMax_=std::max(bcMax_, valsSrc[i]); } }
    if(!std::isfinite(bcMin_)||!std::isfinite(bcMax_)){ bcMin_=0.0f; bcMax_=0.0f; }
}

/*
 @brief Uploads the right-hand side (RHS) vector for the Laplace solver to GPU memory.

 @details
 - Lazily ensures multigrid levels are available; if no levels are present, attempts to build them.
 - Computes n = nx_ * ny_ (number of grid points) and treats the input as a scalar field of n floats.
 - If rhs.size() >= n:
     - Uploads exactly the first n elements to the finest level's RHS buffer (levels_[0].rhs).
     - Any excess elements beyond n are ignored.
 - If rhs.size() < n:
     - Uploads a zero-filled RHS of length n (the input is not partially used).
 - Uses a staging MTLBuffer (shared storage) and a blit encoder to copy into the device-resident RHS buffer.
 - The command buffer is committed and synchronously waited upon to guarantee completion before return.

 @param rhs Host-side RHS samples to set for the solver; must contain at least nx_ * ny_ elements
            for a full upload. If larger, only the first nx_ * ny_ elements are used; if smaller,
            a zero field is uploaded instead.

 @pre
 - device_, queue_, nx_, ny_, and fieldBytes_ are initialized and consistent
   (fieldBytes_ == nx_ * ny_ * sizeof(float)).
 - If levels_ is empty, buildLevels_() must succeed for the upload to proceed.

 @post
 - levels_[0].rhs contains exactly n floats corresponding to the provided RHS (or zeros),
   and the data is visible to subsequent GPU work upon return.

 @note The operation blocks on GPU completion (waitUntilCompleted). For higher throughput,
       consider issuing uploads on a dedicated command buffer without waiting, or batching updates.

 @warning If rhs.size() < n, the entire uploaded RHS is zero-initialized; no partial copy occurs.

@see buildLevels_()
*/ 
void LaplaceMetalSolver::setRHS(const std::vector<float>& rhs){
    if (levels_.empty()) { if (!buildLevels_()) return; }
    id<MTLDevice> dev=(__bridge id<MTLDevice>)device_;
    id<MTLCommandQueue> q=(__bridge id<MTLCommandQueue>)queue_;
    size_t n=size_t(nx_)*ny_;
    const float* src = rhs.size()>=n ? rhs.data() : nullptr;
    std::vector<float> zeros; if(!src){ zeros.assign(n,0.0f); src=zeros.data(); }
    id<MTLBuffer> staging=[dev newBufferWithBytes:src length:fieldBytes_ options:MTLResourceStorageModeShared];
    id<MTLCommandBuffer> cb=[q commandBuffer]; id<MTLBlitCommandEncoder> blit=[cb blitCommandEncoder];
    [blit copyFromBuffer:staging sourceOffset:0 toBuffer:(__bridge id<MTLBuffer>)levels_[0].rhs destinationOffset:0 size:fieldBytes_];
    [blit endEncoding]; [cb commit]; [cb waitUntilCompleted];
}


/**
 * @brief Download the solver's level-0 field from GPU memory into a host-side vector.
 *
 * Performs a synchronous Metal readback of the solution field. Optionally runs a
 * clamping compute pass on the GPU immediately before the copy, then blits the
 * device buffer to a CPU-visible staging buffer, waits for completion, and
 * copies the bytes into a std::vector<float>.
 *
 * Logic:
 * 1) Allocate a shared (CPU-visible) staging MTLBuffer of size fieldBytes_.
 * 2) Create a command buffer on the associated MTLCommandQueue.
 * 3) If clampEnabled_ && psoClamp_ && !levels_.empty():
 *    - Bind level-0 field (uA), boundary mask (bcMask), and uniforms.
 *    - Upload clamping bounds {bcMin_, bcMax_} as small-bytes parameter.
 *    - Dispatch the clamp compute kernel over L.nx × L.ny using chooseThreadgroup2D.
 * 4) Encode a blit copy from levels_[0].uA to the staging buffer.
 * 5) Commit, then wait until the GPU finishes (synchronous fence).
 * 6) memcpy staging.contents into result.data() and return the vector.
 *
 * Return value:
 * - Row-major (y * nx_ + x) array of size nx_ * ny_, containing single-precision
 *   samples of the solver's level-0 field after optional clamping.
 *
 * Preconditions:
 * - device_ and queue_ are valid Metal objects (non-null).
 * - levels_ is non-empty and levels_[0].uA is a valid MTLBuffer.
 * - fieldBytes_ == nx_ * ny_ * sizeof(float).
 * - If clamping is enabled, psoClamp_ is a valid MTLComputePipelineState and
 *   levels_[0] has valid bcMask and uniforms buffers.
 *
 * Postconditions:
 * - The returned vector owns a copy of the field data; no GPU-side state is modified
 *   beyond the optional clamping pass.
 *
 * Thread-safety:
 * - Not thread-safe with respect to concurrent use of this LaplaceMetalSolver instance.
 *   External synchronization is required if accessed from multiple threads.
 *
 * Performance notes:
 * - This call blocks the caller until GPU work completes (waitUntilCompleted).
 *   Avoid calling on time-sensitive threads; consider an asynchronous path if needed.
 * - The dominant costs are the GPU->CPU blit of fieldBytes_ and a CPU memcpy of the same size.
 * - Using MTLResourceStorageModeShared eases readback but may trade off bandwidth on discrete GPUs.
 *
 * Data layout:
 * - index = y * nx_ + x, with 0 <= x < nx_ and 0 <= y < ny_.
 *
 * Exception safety:
 * - Strong guarantee for the host-side vector allocation; no C++ exceptions are thrown
 *   by Metal calls. Allocation failures may throw std::bad_alloc.
 *
 * See also:
 * - chooseThreadgroup2D
 * - MTLComputePipelineState
 * - MTLBlitCommandEncoder
 */
std::vector<float> LaplaceMetalSolver::downloadSolution() const {
    std::vector<float> result(nx_*ny_);
    id<MTLDevice> dev=(__bridge id<MTLDevice>)device_;
    id<MTLCommandQueue> queue=(__bridge id<MTLCommandQueue>)queue_;
    id<MTLBuffer> staging=[dev newBufferWithLength:fieldBytes_ options:MTLResourceStorageModeShared];
    id<MTLCommandBuffer> cb=[queue commandBuffer];
    if (clampEnabled_ && psoClamp_ && !levels_.empty()) {
        id<MTLComputePipelineState> pso=(__bridge id<MTLComputePipelineState>)psoClamp_;
        id<MTLComputeCommandEncoder> enc=[cb computeCommandEncoder];
        [enc setComputePipelineState:pso];
        auto &L=levels_[0];
        [enc setBuffer:(__bridge id<MTLBuffer>)L.uA offset:0 atIndex:0];
        [enc setBuffer:(__bridge id<MTLBuffer>)L.bcMask offset:0 atIndex:1];
        [enc setBuffer:(__bridge id<MTLBuffer>)L.uniforms offset:0 atIndex:2];
        struct { float x; float y; } bounds={bcMin_, bcMax_};
        [enc setBytes:&bounds length:sizeof(bounds) atIndex:3];
        [enc dispatchThreads:MTLSizeMake(L.nx,L.ny,1) threadsPerThreadgroup:chooseThreadgroup2D(pso,L.nx,L.ny)];
        [enc endEncoding];
    }
    id<MTLBlitCommandEncoder> blit=[cb blitCommandEncoder];
    [blit copyFromBuffer:(__bridge id<MTLBuffer>)levels_[0].uA sourceOffset:0 toBuffer:staging destinationOffset:0 size:fieldBytes_];
    [blit endEncoding]; [cb commit]; [cb waitUntilCompleted];
    memcpy(result.data(), staging.contents, fieldBytes_);
    return result;
}
