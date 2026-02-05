// Shared internal definitions for LaplaceMetalSolver implementation
#pragma once
#import <Metal/Metal.h>

/*
 * Internal structures and helpers for LaplaceMetalSolver.
 * Not part of the public API.
 */

// Uniforms structure matching Metal shader layout (16-byte aligned)
struct UniformsCPU
{
    uint32_t nx, ny;                         // grid dimensions (inc. BC)
    float inv_dx2, inv_dy2, inv_coef, omega; // precomputed constants
};
struct UniformsGPU
{
    uint32_t nx, ny;
    float inv_dx2;
    float inv_dy2;
    float inv_coef;
    float omega;
    float padding[2]; // pad to 32 bytes (multiple of 16)
};

struct GridDims
{
    uint32_t nx_f, ny_f, nx_c, ny_c;
};

/**
 * @brief Chooses a 2D Metal threadgroup size (bx, by, 1) for a given pipeline and domain.
 *
 * Selects a threadgroup shape that aligns the X dimension to the pipeline's SIMD width,
 * stays within the device's maximum threads-per-threadgroup, and clamps to the problem size.
 *
 * Logic:
 * - If the pipeline state is null, return a safe default of 16 x 16 x 1.
 * - Initialize bx to pso->threadExecutionWidth to align X with SIMD lanes for utilization.
 * - Compute by as min(maxTotalThreadsPerThreadgroup / bx, 16) to respect hardware limits
 *   and avoid overly tall groups.
 * - Clamp bx to nx and by to ny so the group never exceeds the remaining problem tile.
 * - Set z to 1 for 2D kernels.
 *
 * Rationale:
 * - Aligning the X dimension with the SIMD width improves occupancy and scheduling.
 * - Respecting maxTotalThreadsPerThreadgroup prevents invalid dispatch sizes.
 * - Capping Y at 16 is a conservative heuristic that keeps groups reasonably square
 *   across devices and kernels.
 * - Clamping to (nx, ny) handles small domains and edge tiles.
 *
 * @param pso Compute pipeline used to query threadExecutionWidth and maxTotalThreadsPerThreadgroup.
 * @param nx Problem size (or tile size) in the X dimension.
 * @param ny Problem size (or tile size) in the Y dimension.
 * @return MTLSize representing the chosen threadgroup: width = bx, height = by, depth = 1.
 *
 * @pre nx >= 1 and ny >= 1. If either is 0, the returned size may contain 0, which is invalid for dispatch.
 * @note This is a heuristic; optimal sizes are kernel- and device-dependent. Profile if performance is critical.
 */
static inline MTLSize chooseThreadgroup2D(id<MTLComputePipelineState> pso, uint32_t nx, uint32_t ny)
{
    if (!pso)
        return MTLSizeMake(16, 16, 1);
    const NSUInteger simdWidth = pso.threadExecutionWidth;
    const NSUInteger maxThreads = pso.maxTotalThreadsPerThreadgroup;
    NSUInteger bx = simdWidth;
    NSUInteger by = std::min(maxThreads / bx, NSUInteger(16));
    bx = std::min(bx, NSUInteger(nx));
    by = std::min(by, NSUInteger(ny));
    return MTLSizeMake(bx, by, 1);
}
