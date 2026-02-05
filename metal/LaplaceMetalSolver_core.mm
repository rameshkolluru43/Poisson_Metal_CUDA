// Core construction, pipelines, buffers, levels, and utilities
#import <Metal/Metal.h>
#import <Foundation/Foundation.h>

#include "LaplaceMetalSolver.hpp"
#include "LaplaceMetalSolverInternal.hpp"
#include <cstring>
#include <limits>
#include <stdexcept>



/**
 @brief Construct a LaplaceMetalSolver and initialize all required Metal resources.

 @param d Domain and solver descriptor containing grid resolution (nx, ny), cell spacing (dx, dy), and auxiliary configuration.
 @param metal_source_path Filesystem path to a .metal source file. This must be provided; embedded Metal kernels are not supported.

 @details
 This constructor fully prepares the solver for GPU execution by performing the following steps:
 1) Acquire default Metal device (MTLCreateSystemDefaultDevice). Throws if the platform lacks a Metal-capable GPU.
 2) Create a command queue on the device. On failure, releases the device and throws.
 3) Build the Metal shader library from the provided source file. Embedded Metal kernels are not supported; a valid path is required.
        On any compilation error, release previously created resources and throw with compiler diagnostics.
 4) Create compute pipelines required by the solver. If pipeline creation fails, release library, queue, and device, null out their handles, and throw with an error message.
 5) Compute the byte size for the primary field (nx * ny * sizeof(float)) and allocate all necessary Metal buffers. Throw if allocation fails.
 6) Construct the multigrid hierarchy (levels) used for efficient relaxation and restriction/prolongation. Throw if building the hierarchy fails.
 7) Initialize uniform/state buffers for the top level (grid geometry, solver parameters).

 The constructor also initializes default solver controls:
 - verbose_ = false
 - damping_ = 2/3 (relaxation damping)
 - smoother_ = Smoother::RBGS (red-black Gauss-Seidel)
 - relTol_ = -1.0 (disabled; enables absolute tolerance mode or requires explicit tolerance later)

 Memory management notes:
 - Objective-C objects (MTLDevice, MTLCommandQueue, MTLLibrary) are transferred into C++ ownership using __bridge_retained and are balanced with CFBridgingRelease on all error paths to prevent leaks.
 - On any initialization failure, partially created resources are released and internal pointers nulled before throwing.

 Preconditions:
 - d.nx > 0, d.ny > 0, d.dx > 0, d.dy > 0.
 - The process runs on a platform with Metal support.
 - metal_source_path must be a valid path to a Metal source file; embedded kernels are not supported.

 Postconditions:
 - On success, the solver holds a valid Metal device, command queue, shader library, compute pipelines, allocated field buffers sized nx*ny, and a fully built multigrid hierarchy; it is ready to encode and dispatch compute passes.
 - On failure, a std::runtime_error is thrown and no Metal resources are leaked.

 @throws std::runtime_error
     - "No Metal device available"
     - "Failed to create Metal command queue"
     - "Metal library compile error: <diagnostics>"
     - "Pipeline build error: <diagnostics>"
     - "Failed to allocate Metal buffers"
     - "Failed to build multigrid levels"

 @note The constructor favors safe fail-fast behavior with explicit diagnostics to simplify debugging of GPU availability, shader compilation, and pipeline creation issues.

 @complexity
 - Time: O(nx * ny) for buffer allocation and initial multigrid construction.
 - Space: O(nx * ny) floats for the main field plus per-level multigrid buffers.
*/

LaplaceMetalSolver::LaplaceMetalSolver(const Desc& d, const char* metal_source_path)
: nx_(d.nx), ny_(d.ny), dx_(d.dx), dy_(d.dy), verbose_(false), damping_(2.0f/3.0f), smoother_(Smoother::RBGS), relTol_(-1.0f)
{
        id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
        if (!dev) throw std::runtime_error("No Metal device available");
        device_ = (__bridge_retained void*)dev;
        id<MTLCommandQueue> q = [dev newCommandQueue];
        if (!q) { CFBridgingRelease(device_); device_=nullptr; throw std::runtime_error("Failed to create Metal command queue"); }
        queue_ = (__bridge_retained void*)q;

        std::string err; bool ok=false;
        if (metal_source_path && metal_source_path[0]) ok = compileLibraryFromFile_(metal_source_path, err);
        else { err = "Embedded Metal kernels are not supported; please provide a .metal source file path."; ok = false; }
        if (!ok) { CFBridgingRelease(queue_); CFBridgingRelease(device_); queue_=nullptr; device_=nullptr; throw std::runtime_error("Metal library compile error: "+err); }
        if (!buildPipelines_(err)) { CFBridgingRelease(lib_); CFBridgingRelease(queue_); CFBridgingRelease(device_); lib_=queue_=device_=nullptr; throw std::runtime_error("Pipeline build error: "+err); }

        fieldBytes_ = size_t(nx_) * size_t(ny_) * sizeof(float);
        if (!ensureBuffers_()) throw std::runtime_error("Failed to allocate Metal buffers");
        if (!buildLevels_()) throw std::runtime_error("Failed to build multigrid levels");
        updateTopLevelUniforms_();
}

/*
 @brief Destructor that releases all Metal and Core Foundation resources owned by the solver.

 @details
 Performs an orderly teardown of all allocated resources:
 - Iterates over all multigrid levels to release per-level resources (solution buffers, RHS,
   boundary condition masks/values, and level-specific uniforms).
 - Releases top-level buffers, compute pipeline states, the Metal library, command queue,
   and device.
 
 Ownership is balanced using CFBridgingRelease, which pairs with prior __bridge_retained /
 CFBridgingRetain transfers from ARC-managed Objective-C objects to Core Foundation.
 After releasing, each pointer is set to nullptr to avoid dangling references and to make
 double-destruction a no-op.

 The destruction order goes from most dependent (per-level resources) to least dependent
 (device), ensuring correct teardown of Metal resources.

 @pre No in-flight command buffers or encoders are using the resources managed by this solver.
 @post All owned CF/Objective-C/Metal objects are released and all member pointers are null.
 @exception None. The destructor must not throw.
 @complexity Linear in the number of levels (O(levels_.size())).
 @thread_safety Not thread-safe. Must not race with concurrent GPU work or any other thread
                accessing the solver's members.
 @warning CFBridgingRelease must only be used on objects previously retained for CF ownership
          (e.g., via __bridge_retained/CFBridgingRetain). Mismatched ownership will lead to
          over-release and crashes.
 @note Explicitly nulling pointers is defensive, preventing accidental reuse and simplifying
       diagnostics for use-after-free.
*/
LaplaceMetalSolver::~LaplaceMetalSolver(){
    for (auto &level : levels_) {
        if (level.uA) CFBridgingRelease(level.uA), level.uA=nullptr;
        if (level.uB) CFBridgingRelease(level.uB), level.uB=nullptr;
        if (level.rhs) CFBridgingRelease(level.rhs), level.rhs=nullptr;
        if (level.bcMask) CFBridgingRelease(level.bcMask), level.bcMask=nullptr;
        if (level.bcVals) CFBridgingRelease(level.bcVals), level.bcVals=nullptr;
        if (level.uniforms) CFBridgingRelease(level.uniforms), level.uniforms=nullptr;
    }
    if (bufUniforms_) CFBridgingRelease(bufUniforms_), bufUniforms_=nullptr;
    if (bufBCVals_) CFBridgingRelease(bufBCVals_), bufBCVals_=nullptr;
    if (bufBCMask_) CFBridgingRelease(bufBCMask_), bufBCMask_=nullptr;
    if (bufU1_) CFBridgingRelease(bufU1_), bufU1_=nullptr;
    if (bufU0_) CFBridgingRelease(bufU0_), bufU0_=nullptr;
    if (psoProlongAdd_) CFBridgingRelease(psoProlongAdd_), psoProlongAdd_=nullptr;
    if (psoRestrict_) CFBridgingRelease(psoRestrict_), psoRestrict_=nullptr;
    if (psoApplyBC_) CFBridgingRelease(psoApplyBC_), psoApplyBC_=nullptr;
    if (psoResidualRaw_) CFBridgingRelease(psoResidualRaw_), psoResidualRaw_=nullptr;
    if (psoJacobi_) CFBridgingRelease(psoJacobi_), psoJacobi_=nullptr;
    if (psoJacobiRHS_) CFBridgingRelease(psoJacobiRHS_), psoJacobiRHS_=nullptr;
    if (psoRBGS_) CFBridgingRelease(psoRBGS_), psoRBGS_=nullptr;
    if (psoRBGSRHS_) CFBridgingRelease(psoRBGSRHS_), psoRBGSRHS_=nullptr;
    if (psoClamp_) CFBridgingRelease(psoClamp_), psoClamp_=nullptr;
    if (psoJacobiFused_) CFBridgingRelease(psoJacobiFused_), psoJacobiFused_=nullptr;
    if (psoReduceSum_) CFBridgingRelease(psoReduceSum_), psoReduceSum_=nullptr;
    if (psoZeroFloat_) CFBridgingRelease(psoZeroFloat_), psoZeroFloat_=nullptr;
    if (psoSumSquaresPartial_) CFBridgingRelease(psoSumSquaresPartial_), psoSumSquaresPartial_=nullptr;
    if (psoSumSquaresDiffPartial_) CFBridgingRelease(psoSumSquaresDiffPartial_), psoSumSquaresDiffPartial_=nullptr;
    if (bufPartial0_) CFBridgingRelease(bufPartial0_), bufPartial0_=nullptr, bufPartial0Cap_=0;
    if (lib_) CFBridgingRelease(lib_), lib_=nullptr;
    if (queue_) CFBridgingRelease(queue_), queue_=nullptr;
    if (device_) CFBridgingRelease(device_), device_=nullptr;
}


/**
 @brief Compiles a Metal shader library from a source file and stores it in the solver.

 @details
 - Converts the provided UTF-8 C-string path into an NSString.
 - Loads the file contents as a UTF-8 NSString.
 - On file read failure, assigns the localized error description to `err` and returns false.
 - Bridges the internal `device_` to an `id<MTLDevice>` and compiles the library with `newLibraryWithSource:options:error:` using default options.
 - On compilation failure or if no library is produced, assigns the localized error description (or "Unknown error") to `err` and returns false.
 - On success, retains and stores the resulting `MTLLibrary` in `lib_` and returns true.

 @param path
 A UTF-8 encoded C-string path to a Metal source file (.metal). May be null or empty; in that case the file read will fail and `err` will be populated.
 @param[out] err
 On failure, receives a human-readable description of the error; left unchanged on success.

 @return
 true if the Metal library was successfully compiled and stored; false otherwise.

 @pre
 - `device_` must reference a valid `MTLDevice`.
 - The file at `path` must be accessible and UTF-8 encoded.

 @post
 - On success, `lib_` holds a retained `MTLLibrary` instance managed by the owning class lifecycle.

 @note
 - File reading uses UTF-8 encoding.
 - Compilation uses default `MTLLibrary` compile options (nil).

 @warning
 - This function performs synchronous I/O and compilation and may block the calling thread.
 - Not inherently thread-safe with respect to concurrent access/mutation of `lib_`.

 @see
 - `[MTLDevice newLibraryWithSource:options:error:]`
 */
bool LaplaceMetalSolver::compileLibraryFromFile_(const char* path, std::string& err){
    NSString* nsPath = [NSString stringWithUTF8String:path ? path : ""];
    NSError* error=nil;
    NSString* source = [NSString stringWithContentsOfFile:nsPath encoding:NSUTF8StringEncoding error:&error];
    if (error) { 
        err = std::string([[error localizedDescription] UTF8String]); 
        return false; 
        }
    id<MTLDevice> dev = (__bridge id<MTLDevice>)device_;
    id<MTLLibrary> library = [dev newLibraryWithSource:source options:nil error:&error];
    if (error || !library) { 
        err = error ? std::string([[error localizedDescription] UTF8String]) : "Unknown error"; 
        return false; 
    }
    lib_ = (__bridge_retained void*)library; 
    return true;
}

bool LaplaceMetalSolver::compileLibraryFromEmbedded_(std::string& err){ 
    err = "Embedded Metal kernels are not supported; please provide a .metal source file path."; 
    return false; 
}

bool LaplaceMetalSolver::buildPipelines_(std::string& err){
    id<MTLLibrary> lib = (__bridge id<MTLLibrary>)lib_;
    id<MTLDevice> dev = (__bridge id<MTLDevice>)device_;
    NSError* error=nil;
#define CREATE_PIPELINE(name, kernelName) do { \
    id<MTLFunction> func = [lib newFunctionWithName:@#kernelName]; \
    if (!func) { err = "Function '" #kernelName "' not found"; return false; } \
    id<MTLComputePipelineState> pso = [dev newComputePipelineStateWithFunction:func error:&error]; \
    if (error || !pso) { err = error ? std::string([[error localizedDescription] UTF8String]) : "Pipeline creation failed for " #kernelName; return false; } \
    name = (__bridge_retained void*)pso; \
} while(0)
    CREATE_PIPELINE(psoJacobi_, jacobi_step);
    CREATE_PIPELINE(psoJacobiRHS_, jacobi_step_rhs);
    CREATE_PIPELINE(psoRBGS_, rbgs_phase);
    CREATE_PIPELINE(psoRBGSRHS_, rbgs_phase_rhs);
    CREATE_PIPELINE(psoSOR_, sor_step);
    CREATE_PIPELINE(psoSORRHS_, sor_step_rhs);
    CREATE_PIPELINE(psoResidualRaw_, compute_residual_raw);
    CREATE_PIPELINE(psoApplyBC_, apply_dirichlet);
    CREATE_PIPELINE(psoRestrict_, restrict_full_weighting);
    CREATE_PIPELINE(psoProlongAdd_, prolong_bilinear_add);
    CREATE_PIPELINE(psoClamp_, clamp_to_bounds);
    CREATE_PIPELINE(psoJacobiFused_, jacobi_fused_residual_tiled);
    CREATE_PIPELINE(psoReduceSum_, reduce_sum);
    CREATE_PIPELINE(psoZeroFloat_, set_zero_float);
    CREATE_PIPELINE(psoResidualSmooth_, residual_smooth_step);
    // Optional helper for partial sums on arbitrary buffers
    CREATE_PIPELINE(psoSumSquaresPartial_, sum_squares_partial_tiled);
    CREATE_PIPELINE(psoSumSquaresDiffPartial_, sum_squares_diff_partial_tiled);
#undef CREATE_PIPELINE
    return true;
}

/**
 * @brief Ensure top-level Metal buffers exist, are sufficiently sized, and are initialized.
 *
 * Allocates the solver's root buffers if missing and grows them if the current capacity
 * is smaller than fieldBytes_. Existing larger buffers are retained and reused.
 *
 * For any newly created or resized private-storage buffers (u0, u1, bcMask, bcVals),
 * the method zero-initializes their contents via a single blit pass. The shared uniforms
 * buffer is created if needed and labeled; its contents are left to be populated by
 * updateTopLevelUniforms_().
 *
 * Buffer labels are set to aid GPU debugging in Xcode.
 *
 * Pre-conditions:
 * - device_ and queue_ are valid.
 * - fieldBytes_ == nx_ * ny_ * sizeof(float) and > 0.
 *
 * Post-conditions:
 * - All required buffers exist and have capacity >= required sizes.
 * - Newly allocated/resized private buffers are zero-filled.
 *
 * Returns:
 * - true on success, false on allocation failure or missing device/queue.
 *
 * Thread-safety:
 * - Not thread-safe. Must not race with GPU work using these buffers.
 */
bool LaplaceMetalSolver::ensureBuffers_(){
    if (!device_ || !queue_ || fieldBytes_ == 0) return false;

    id<MTLDevice> dev = (__bridge id<MTLDevice>)device_;
    const MTLResourceOptions privateOpts = MTLResourceStorageModePrivate;
    const MTLResourceOptions sharedOpts  = MTLResourceStorageModeShared;

    // Track which buffers were newly created or resized to decide zero-fill.
    bool newU0=false, newU1=false, newBCMask=false, newBCVals=false, newUniforms=false;

    auto ensureOne = [&](void*& slot,
                         size_t required,
                         MTLResourceOptions opts,
                         NSString* label,
                         bool& createdOrResized) -> bool
    {
        createdOrResized = false;

        if (slot) {
            id<MTLBuffer> existing = (__bridge id<MTLBuffer>)slot;
            if (existing.length >= required) {
                // Ensure label exists for easier debugging.
                if (!existing.label) existing.label = label;
                return true;
            }
        }

        // Allocate new buffer first; only release old on success.
        id<MTLBuffer> b = [dev newBufferWithLength:required options:opts];
        if (!b) return false;
        b.label = label;

        if (slot) { CFBridgingRelease(slot); slot = nullptr; }
        slot = (__bridge_retained void*)b;
        createdOrResized = true;
        return true;
    };

    if (!ensureOne(bufU0_, fieldBytes_, privateOpts, @"Laplace:u0", newU0)) return false;
    if (!ensureOne(bufU1_, fieldBytes_, privateOpts, @"Laplace:u1", newU1)) return false;
    if (!ensureOne(bufBCMask_, fieldBytes_, privateOpts, @"Laplace:bcMask", newBCMask)) return false;
    if (!ensureOne(bufBCVals_, fieldBytes_, privateOpts, @"Laplace:bcVals", newBCVals)) return false;
    if (!ensureOne(bufUniforms_, sizeof(UniformsCPU), sharedOpts, @"Laplace:uniforms", newUniforms)) return false;

    // Zero-fill only the private buffers that were created or resized.
    if (newU0 || newU1 || newBCMask || newBCVals) {
        id<MTLCommandQueue> q = (__bridge id<MTLCommandQueue>)queue_;
        if (!q) return false;

        id<MTLCommandBuffer> cb = [q commandBuffer];
        id<MTLBlitCommandEncoder> blit = [cb blitCommandEncoder];

        if (newU0)     [blit fillBuffer:(__bridge id<MTLBuffer>)bufU0_     range:NSMakeRange(0, fieldBytes_) value:0];
        if (newU1)     [blit fillBuffer:(__bridge id<MTLBuffer>)bufU1_     range:NSMakeRange(0, fieldBytes_) value:0];
        if (newBCMask) [blit fillBuffer:(__bridge id<MTLBuffer>)bufBCMask_ range:NSMakeRange(0, fieldBytes_) value:0];
        if (newBCVals) [blit fillBuffer:(__bridge id<MTLBuffer>)bufBCVals_ range:NSMakeRange(0, fieldBytes_) value:0];

        [blit endEncoding];
        [cb commit];
        [cb waitUntilCompleted];
    }

    // Uniforms are shared; leave contents to updateTopLevelUniforms_().
    return true;
}

/*
 @brief Builds the multigrid hierarchy and allocates/initializes Metal resources for each level.

 @details
 This routine constructs a vector of multigrid levels starting from the finest resolution
 (nx_, ny_), then repeatedly coarsens while both dimensions remain at least 3. For each level:
 - Allocates device-side field buffers (uA, uB, rhs, bcMask, bcVals) in MTLResourceStorageModePrivate.
 - Allocates a small uniforms buffer (UniformsCPU) in MTLResourceStorageModeShared for CPU writes.
 - Computes and stores per-level scaling of inverse spacing terms (inv_dx2, inv_dy2) based on level index,
   and precomputes the relaxation coefficient inv_coef = 1 / (2*inv_dx2 + 2*inv_dy2).
 - Initializes all field buffers to zero via a blit encoder, then synchronously waits for completion.

 Coarsening follows nx = (nx - 1)/2 + 1 and ny = (ny - 1)/2 + 1 to preserve boundary alignment under
 restriction. The loop terminates early if both nx and ny are <= 5 to avoid overly small grids,
 and it always enforces a minimum size of 3x3.

 @pre
 - device_ and queue_ are valid Metal objects created for the same device.
 - nx_ >= 1, ny_ >= 1, and dx_ > 0, dy_ > 0.
 - UniformsCPU layout matches the expectations of the corresponding Metal shader code.
 - damping_ is set to the desired relaxation damping factor.

 @post
 - On success, levels_ contains at least one level with all buffers allocated and initialized to zero,
   and each uniforms buffer is populated with per-level constants.
 - On failure, levels_ may be cleared and the method returns false.

 @return
 - true if all levels are built successfully and buffers are allocated/initialized;
 - false if allocation fails for any buffer or if no valid levels can be constructed.

 @note
 - Field buffers use private storage for optimal GPU access; uniforms use shared storage for CPU writes.
 - The method performs a synchronous wait (waitUntilCompleted) after zero-filling to ensure buffers are ready.
 - The per-level scaling uses scale = 2^lev, so inv_dx2 and inv_dy2 are divided by scale^2.

 @warning
 - Not thread-safe with respect to concurrent mutation of levels_, device_, or queue_.
 - __bridge_retained is used for buffer handles; corresponding releases must occur when tearing down levels
   to avoid leaks.

 @par Metal specifics
 - Uses MTLBlitCommandEncoder::fillBuffer to zero-initialize resources.
 - Requires that the device supports the chosen storage modes and blit operations.

 @par Complexity
 - Memory: O(sum over levels of nx_lev * ny_lev) floats across five field buffers + small uniforms per level.
 - Time: O(number_of_buffers) for zero-fill plus constant work per level to populate uniforms.
*/
bool LaplaceMetalSolver::buildLevels_(){
    levels_.clear();
    uint32_t nx = nx_, ny = ny_;
    id<MTLDevice> dev = (__bridge id<MTLDevice>)device_;
    const MTLResourceOptions privateOpts = MTLResourceStorageModePrivate;
    const MTLResourceOptions sharedOpts = MTLResourceStorageModeShared;
    while (nx >= 3 && ny >= 3) {
        MGLevel L; L.nx=nx; L.ny=ny; size_t bytes = size_t(nx) * size_t(ny) * sizeof(float);
        L.uA = (__bridge_retained void*)[dev newBufferWithLength:bytes options:privateOpts];
        L.uB = (__bridge_retained void*)[dev newBufferWithLength:bytes options:privateOpts];
        L.rhs = (__bridge_retained void*)[dev newBufferWithLength:bytes options:privateOpts];
        L.bcMask = (__bridge_retained void*)[dev newBufferWithLength:bytes options:privateOpts];
        L.bcVals = (__bridge_retained void*)[dev newBufferWithLength:bytes options:privateOpts];
        L.uniforms = (__bridge_retained void*)[dev newBufferWithLength:sizeof(UniformsCPU) options:sharedOpts];
        L.fieldBytes = bytes; if (!L.uA||!L.uB||!L.rhs||!L.bcMask||!L.bcVals||!L.uniforms) return false;
        levels_.push_back(L);
        if (nx <= 5 && ny <= 5) break; nx = (nx - 1) / 2 + 1; ny = (ny - 1) / 2 + 1;
    }
    if (levels_.empty()) return false;
    const float inv_dx2_0 = 1.0f / (dx_ * dx_); const float inv_dy2_0 = 1.0f / (dy_ * dy_);
    for (size_t lev = 0; lev < levels_.size(); ++lev) {
        auto &L = levels_[lev];
        float scale = float(1u << lev);
        UniformsCPU u{L.nx, L.ny, inv_dx2_0/(scale*scale), inv_dy2_0/(scale*scale), 0, damping_};
        u.inv_coef = 1.0f / (2.0f * u.inv_dx2 + 2.0f * u.inv_dy2);
        id<MTLBuffer> ub = (__bridge id<MTLBuffer>)L.uniforms; memcpy(ub.contents, &u, sizeof(u));
    }
    id<MTLCommandQueue> q = (__bridge id<MTLCommandQueue>)queue_;
    id<MTLCommandBuffer> cb = [q commandBuffer]; id<MTLBlitCommandEncoder> blit = [cb blitCommandEncoder];
    for (auto &L : levels_) {
        [blit fillBuffer:(__bridge id<MTLBuffer>)L.uA range:NSMakeRange(0, L.fieldBytes) value:0];
        [blit fillBuffer:(__bridge id<MTLBuffer>)L.uB range:NSMakeRange(0, L.fieldBytes) value:0];
        [blit fillBuffer:(__bridge id<MTLBuffer>)L.rhs range:NSMakeRange(0, L.fieldBytes) value:0];
        [blit fillBuffer:(__bridge id<MTLBuffer>)L.bcMask range:NSMakeRange(0, L.fieldBytes) value:0];
        [blit fillBuffer:(__bridge id<MTLBuffer>)L.bcVals range:NSMakeRange(0, L.fieldBytes) value:0];
    }
    [blit endEncoding]; [cb commit]; [cb waitUntilCompleted];

    // Enforce homogeneous Dirichlet BCs on all coarse levels (lev >= 1):
    // Mask = 1 on boundary nodes, 0 interior; Values = 0 everywhere.
    // Level 0 BCs are uploaded by setDirichletBC from the host, so skip modifying them here.
    if (!levels_.empty()) {
        id<MTLDevice> dev = (__bridge id<MTLDevice>)device_;
        for (size_t lev = 1; lev < levels_.size(); ++lev) {
            auto &L = levels_[lev];
            const size_t n = size_t(L.nx) * size_t(L.ny);
            // Build host-side mask and zero values
            std::vector<uint8_t> mask(n, 0);
            std::vector<float>   zeros(n, 0.0f);
            for (uint32_t j = 0; j < L.ny; ++j) {
                for (uint32_t i = 0; i < L.nx; ++i) {
                    if (i == 0 || j == 0 || i == L.nx - 1 || j == L.ny - 1) {
                        mask[size_t(j) * L.nx + i] = 1u;
                    }
                }
            }
            id<MTLBuffer> maskStaging = [dev newBufferWithBytes:mask.data() length:n * sizeof(uint8_t) options:MTLResourceStorageModeShared];
            id<MTLBuffer> valsStaging = [dev newBufferWithBytes:zeros.data() length:n * sizeof(float)   options:MTLResourceStorageModeShared];

            id<MTLCommandBuffer> cbb = [q commandBuffer];
            id<MTLBlitCommandEncoder> blb = [cbb blitCommandEncoder];
            [blb copyFromBuffer:maskStaging sourceOffset:0 toBuffer:(__bridge id<MTLBuffer>)L.bcMask destinationOffset:0 size:n * sizeof(uint8_t)];
            [blb copyFromBuffer:valsStaging sourceOffset:0 toBuffer:(__bridge id<MTLBuffer>)L.bcVals destinationOffset:0 size:n * sizeof(float)];
            [blb endEncoding];
            [cbb commit];
            [cbb waitUntilCompleted];
        }
    }
    return true;
}

/**
 * @brief Updates Metal uniform buffers with the current grid parameters and solver coefficients.
 *
 * Constructs a CPU-side Uniforms structure from the solver state (grid dimensions, inverse
 * squared grid spacings along x and y, damping value, and a zero-initialized control/iteration
 * field), computes the Jacobi inverse coefficient:
 *   inv_coef = 1 / (2 * inv_dx2 + 2 * inv_dy2),
 * and writes the result into available Metal uniform buffers.
 *
 * Specifically, the method:
 * - Computes inv_dx2 = 1 / (dx_ * dx_) and inv_dy2 = 1 / (dy_ * dy_).
 * - Computes the Jacobi normalization factor inv_coef used by the 5-point Laplacian stencil.
 * - If a level-0 uniforms buffer exists, copies the populated struct into it.
 * - If a global uniforms buffer exists, copies the populated struct into it as well.
 *
 * Safety and correctness:
 * - Pre-conditions:
 *   - dx_ > 0 and dy_ > 0 to avoid division by zero.
 *   - Any destination MTLBuffer must be allocated, CPU-visible, and at least sizeof(UniformsCPU) bytes.
 * - Post-conditions:
 *   - The level-0 and/or global uniforms buffers, if present, contain an up-to-date snapshot
 *     of the solver uniforms derived from the current state.
 * - Thread-safety:
 *   - Not thread-safe with concurrent mutations of solver state or buffer objects.
 *
 * Performance:
 * - Time complexity: O(1); Space complexity: O(1).
 *
 * Notes:
 * - Uses Objective-C bridging to obtain id<MTLBuffer> and memcpy into buffer.contents.
 * - For managed/private storage modes, callers must ensure proper synchronization semantics.
 */
void LaplaceMetalSolver::updateTopLevelUniforms_()
{
    // Update global uniforms for single-grid paths
    UniformsCPU top{nx_, ny_, 1.0f/(dx_*dx_), 1.0f/(dy_*dy_), 0, damping_};
    top.inv_coef = 1.0f / (2.0f * top.inv_dx2 + 2.0f * top.inv_dy2);
    if (bufUniforms_)
    {
        id<MTLBuffer> b = (__bridge id<MTLBuffer>)bufUniforms_;
        memcpy(b.contents, &top, sizeof(top));
    }

    // Refresh per-level uniforms (omega and inv_coef depend on level spacings)
    if (!levels_.empty())
    {
        const float inv_dx2_0 = 1.0f / (dx_ * dx_);
        const float inv_dy2_0 = 1.0f / (dy_ * dy_);
        for (size_t lev = 0; lev < levels_.size(); ++lev)
        {
            auto &L = levels_[lev];
            if (!L.uniforms) continue;
            float scale = float(1u << lev);
            UniformsCPU u{L.nx, L.ny, inv_dx2_0/(scale*scale), inv_dy2_0/(scale*scale), 0, damping_};
            u.inv_coef = 1.0f / (2.0f * u.inv_dx2 + 2.0f * u.inv_dy2);
            id<MTLBuffer> ub = (__bridge id<MTLBuffer>)L.uniforms;
            memcpy(ub.contents, &u, sizeof(u));
        }
    }
}

/**
 * @brief Releases all per-level Objective‑C/CF resources and clears the level hierarchy.
 *
 * @details
 * Iterates over each stored level and conditionally releases the following ARC/CF-bridged
 * resources using CFBridgingRelease to balance prior retains/bridges and prevent memory leaks:
 * - uA
 * - uB
 * - rhs
 * - bcMask
 * - bcVals
 * - uniforms
 *
 * Null checks are performed before releasing each resource. After releasing all per-level
 * objects, the levels_ container is cleared, removing any remaining references to level data.
 *
 * This helper is intended for teardown or when rebuilding the solver’s multilevel structures.
 *
 * @pre No in-flight GPU work or other consumers are using the per-level resources.
 * @post All per-level bridged objects are released and levels_ is empty.
 *
 * @note CFBridgingRelease safely handles null inputs and transfers ownership to ARC while
 * releasing the underlying object.
 *
 * @warning Not thread-safe. Ensure exclusive access to the solver state when invoking.
 *
 * @complexity Linear in the number of levels (O(N)).
 */
void LaplaceMetalSolver::destroyLevels_(){ 
    for (auto &L:levels_) { 
        if(L.uA) CFBridgingRelease(L.uA);
        if(L.uB) CFBridgingRelease(L.uB);
        if(L.rhs) CFBridgingRelease(L.rhs);
        if(L.bcMask) CFBridgingRelease(L.bcMask);
        if(L.bcVals) CFBridgingRelease(L.bcVals);
        if(L.uniforms) CFBridgingRelease(L.uniforms);
    }
    levels_.clear();
}

bool LaplaceMetalSolver::ensurePartialBuffer0_(uint32_t groupsX, uint32_t groupsY)
{
    uint32_t need = groupsX * groupsY;
    if (need == 0) need = 1;
    if (bufPartial0_ && bufPartial0Cap_ >= need) return true;
    id<MTLDevice> dev = (__bridge id<MTLDevice>)device_;
    if (!dev) return false;
    id<MTLBuffer> b = [dev newBufferWithLength:sizeof(float)*need options:MTLResourceStorageModeShared];
    if (!b) return false;
    if (bufPartial0_) { CFBridgingRelease(bufPartial0_); bufPartial0_=nullptr; }
    bufPartial0_ = (__bridge_retained void*)b;
    bufPartial0Cap_ = need;
    return true;
}

void LaplaceMetalSolver::setResidualSmoothingPasses(uint32_t passes) {
    residualSmoothPasses_ = passes;
}
