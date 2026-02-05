// Multigrid V-cycle solver implementation split out from solvers
#import <Metal/Metal.h>
#include "LaplaceMetalSolver.hpp"
#include "LaplaceMetalSolverInternal.hpp"
#include <cmath>
#include <vector>
#include <iostream>
#include <fstream>

bool LaplaceMetalSolver::solveMultigrid(uint32_t max_vcycles, float tol, uint32_t nu1, uint32_t nu2, uint32_t coarse_iters)
{
    // Build levels if not already built
    if (levels_.empty())
    {
         if (!buildLevels_())
             return false;
    }
    // Report number of levels
    if (verbose_) {
        std::cout << "solveMultigrid: built " << levels_.size() << " levels\n";
    }

    // Validate parameters
    if (max_vcycles == 0) {
        std::cerr << "[ERROR] solveMultigrid: max_vcycles must be > 0.\n";
        return false;
    }
    // Note: tol <= 0 means absolute tolerance is disabled; rely on relative tolerance if set.
    if (nu1 == 0 && nu2 == 0 && coarse_iters == 0) {
        std::cerr << "[ERROR] solveMultigrid: at least one of nu1, nu2, coarse_iters must be > 0.\n";
        return false;
    }   
    if (smoother_ == Smoother::Jacobi && (nu1 == 0 || nu2 == 0)) {
        std::cerr << "[ERROR] solveMultigrid: Jacobi smoother requires both nu1 and nu2 to be > 0.\n";
        return false;
    }
    if (smoother_ == Smoother::RBGS && coarse_iters == 0) {
        std::cerr << "[ERROR] solveMultigrid: Red-Black Gauss-Seidel smoother requires coarse_iters to be > 0.\n";
        return false;
    }
    if (levels_.size() < 2) {
        std::cerr << "[ERROR] solveMultigrid: at least two levels are required for multigrid.\n";
        return false;
    }
    // Ensure all levels have their buffers allocated
    for (size_t lev = 0; lev < levels_.size(); ++lev) {
        // TODO: Implement level buffer allocation check if needed
        // if (!ensureLevelBuffers_(lev)) {
        //     std::cerr << "[ERROR] solveMultigrid: failed to allocate buffers for level " << lev << ".\n";
        //     return false;
        // }
    }

    id<MTLCommandQueue> q = (__bridge id<MTLCommandQueue>)queue_;
    id<MTLComputePipelineState> psoApply = (__bridge id<MTLComputePipelineState>)psoApplyBC_;
    id<MTLComputePipelineState> psoResRaw = (__bridge id<MTLComputePipelineState>)psoResidualRaw_;
    id<MTLComputePipelineState> psoRestr = (__bridge id<MTLComputePipelineState>)psoRestrict_;
    id<MTLComputePipelineState> psoProl  = (__bridge id<MTLComputePipelineState>)psoProlongAdd_;
    id<MTLComputePipelineState> psoJac   = (__bridge id<MTLComputePipelineState>)psoJacobi_;
    id<MTLComputePipelineState> psoJacR  = (__bridge id<MTLComputePipelineState>)psoJacobiRHS_;
    id<MTLComputePipelineState> psoRB    = (__bridge id<MTLComputePipelineState>)psoRBGS_;
    id<MTLComputePipelineState> psoRBRHS = (__bridge id<MTLComputePipelineState>)psoRBGSRHS_;
    id<MTLComputePipelineState> psoSOR   = (__bridge id<MTLComputePipelineState>)psoSOR_;
    id<MTLComputePipelineState> psoSORRHS = (__bridge id<MTLComputePipelineState>)psoSORRHS_;
    id<MTLComputePipelineState> psoZero  = (__bridge id<MTLComputePipelineState>)psoZeroFloat_;
   
    if (!psoApply || !psoResRaw || !psoRestr || !psoProl || !psoJac || !psoJacR || !psoRB || !psoRBRHS || !psoSOR || !psoSORRHS) {
        std::cerr << "[ERROR] solveMultigrid: one or more required compute pipeline states are not initialized.\n";
        return false;
    }

    // Lambda functions for various operations

    // encApplyBC: apply boundary conditions to the current iterate uA on level 'lev'
    auto encApplyBC = [&](id<MTLCommandBuffer> cb, size_t lev)
    {
        auto &L = levels_[lev];
        // Launch BC kernel: uses bcMask/bcVals and uniforms to enforce BCs over the grid
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        [enc setComputePipelineState:psoApply];                                 // in: apply BCs
        [enc setBuffer:(__bridge id<MTLBuffer>)L.uA       offset:0 atIndex:0]; // in/out: solution
        [enc setBuffer:(__bridge id<MTLBuffer>)L.bcMask   offset:0 atIndex:1]; // in: mask of BC nodes
        [enc setBuffer:(__bridge id<MTLBuffer>)L.bcVals   offset:0 atIndex:2]; // in: BC values
        [enc setBuffer:(__bridge id<MTLBuffer>)L.uniforms offset:0 atIndex:3]; // in: grid params
        [enc dispatchThreads:MTLSizeMake(L.nx, L.ny, 1)
             threadsPerThreadgroup:chooseThreadgroup2D(psoApply, L.nx, L.ny)];
        [enc endEncoding];
    };

    auto encResidual = [&](id<MTLCommandBuffer> cb, size_t lev)
    {
         auto &L=levels_[lev]; 
         id<MTLComputeCommandEncoder> enc=[cb computeCommandEncoder];
         [enc setComputePipelineState:psoResRaw];
         [enc setBuffer:(__bridge id<MTLBuffer>)L.uA offset:0 atIndex:0];
         [enc setBuffer:(__bridge id<MTLBuffer>)L.bcMask offset:0 atIndex:1];
         [enc setBuffer:(__bridge id<MTLBuffer>)L.rhs offset:0 atIndex:2];
         [enc setBuffer:(__bridge id<MTLBuffer>)L.uniforms offset:0 atIndex:3];
         [enc dispatchThreads:MTLSizeMake(L.nx,L.ny,1) threadsPerThreadgroup:chooseThreadgroup2D(psoResRaw,L.nx,L.ny)];
         [enc endEncoding];
    };

    // encRestrict: restrict residual from fine level 'lev_from' to coarse level 'lev_to = lev_from + 1'
    auto encRestrict = [&](id<MTLCommandBuffer> cb, size_t lev_from)
    {
        size_t lev_to = lev_from + 1;
        auto &Lf = levels_[lev_from];
        auto &Lc = levels_[lev_to];
        struct { uint32_t nx_f, ny_f, nx_c, ny_c; } gd = { Lf.nx, Lf.ny, Lc.nx, Lc.ny };
        id<MTLDevice> dev = (__bridge id<MTLDevice>)device_;
        id<MTLBuffer> gdBuf = [dev newBufferWithBytes:&gd length:sizeof(gd) options:MTLResourceStorageModeShared];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        [enc setComputePipelineState:psoRestr];
        [enc setBuffer:(__bridge id<MTLBuffer>)Lf.rhs offset:0 atIndex:0];
        [enc setBuffer:(__bridge id<MTLBuffer>)Lc.rhs offset:0 atIndex:1];
        [enc setBuffer:gdBuf offset:0 atIndex:2];
        [enc dispatchThreads:MTLSizeMake(Lc.nx, Lc.ny, 1)
           threadsPerThreadgroup:chooseThreadgroup2D(psoRestr, Lc.nx, Lc.ny)];
        [enc endEncoding];
    };



    auto encProlong = [&](id<MTLCommandBuffer> cb, size_t lev_to_fine)
    {
        auto &Lf = levels_[lev_to_fine];
        auto &Lc = levels_[lev_to_fine + 1];

        struct { uint32_t nx_f, ny_f, nx_c, ny_c; } gd = { Lf.nx, Lf.ny, Lc.nx, Lc.ny };

        id<MTLDevice> dev = (__bridge id<MTLDevice>)device_;
        id<MTLBuffer> gdBuf = [dev newBufferWithBytes:&gd length:sizeof(gd) options:MTLResourceStorageModeShared];

        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        [enc setComputePipelineState:psoProl];
        [enc setBuffer:(__bridge id<MTLBuffer>)Lc.uA offset:0 atIndex:0];
        [enc setBuffer:(__bridge id<MTLBuffer>)Lf.uA offset:0 atIndex:1];
        [enc setBuffer:gdBuf offset:0 atIndex:2];
        [enc dispatchThreads:MTLSizeMake(Lf.nx, Lf.ny, 1)
           threadsPerThreadgroup:chooseThreadgroup2D(psoProl, Lf.nx, Lf.ny)];
        [enc endEncoding];
    };


    auto encSmooth = [&](id<MTLCommandBuffer> cb, size_t lev, uint32_t iters, bool use_rhs)
    {
        if (iters == 0) return;

        auto &L = levels_[lev];

        for (uint32_t k = 0; k < iters; ++k)
        {
            if (smoother_ == Smoother::Jacobi)
            {
                id<MTLComputePipelineState> pso = use_rhs ? psoJacR : psoJac;
                id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
                [enc setComputePipelineState:pso];
                [enc setBuffer:(__bridge id<MTLBuffer>)L.uA       offset:0 atIndex:0];
                [enc setBuffer:(__bridge id<MTLBuffer>)L.uB       offset:0 atIndex:1];
                [enc setBuffer:(__bridge id<MTLBuffer>)L.bcMask   offset:0 atIndex:2];
                [enc setBuffer:(__bridge id<MTLBuffer>)L.bcVals   offset:0 atIndex:3];
                [enc setBuffer:(__bridge id<MTLBuffer>)L.uniforms offset:0 atIndex:4];
                if (use_rhs)
                    [enc setBuffer:(__bridge id<MTLBuffer>)L.rhs offset:0 atIndex:5];
                [enc dispatchThreads:MTLSizeMake(L.nx, L.ny, 1)
                   threadsPerThreadgroup:chooseThreadgroup2D(pso, L.nx, L.ny)];
                [enc endEncoding];
            }
            else if (smoother_ == Smoother::SOR)
            {
                id<MTLComputePipelineState> pso = use_rhs ? psoSORRHS : psoSOR;
                id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
                [enc setComputePipelineState:pso];
                [enc setBuffer:(__bridge id<MTLBuffer>)L.uA       offset:0 atIndex:0];
                [enc setBuffer:(__bridge id<MTLBuffer>)L.bcMask   offset:0 atIndex:1];
                [enc setBuffer:(__bridge id<MTLBuffer>)L.bcVals   offset:0 atIndex:2];
                [enc setBuffer:(__bridge id<MTLBuffer>)L.uniforms offset:0 atIndex:3];
                if (use_rhs)
                    [enc setBuffer:(__bridge id<MTLBuffer>)L.rhs offset:0 atIndex:4];
                [enc dispatchThreads:MTLSizeMake(L.nx, L.ny, 1)
                   threadsPerThreadgroup:chooseThreadgroup2D(pso, L.nx, L.ny)];
                [enc endEncoding];
            }
            else
            {
                for (uint32_t parity = 0; parity < 2; ++parity)
                {
                    id<MTLComputePipelineState> pso = use_rhs ? psoRBRHS : psoRB;
                    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
                    [enc setComputePipelineState:pso];
                    [enc setBuffer:(__bridge id<MTLBuffer>)L.uA       offset:0 atIndex:0];
                    [enc setBuffer:(__bridge id<MTLBuffer>)L.bcMask   offset:0 atIndex:1];
                    [enc setBuffer:(__bridge id<MTLBuffer>)L.bcVals   offset:0 atIndex:2];
                    [enc setBuffer:(__bridge id<MTLBuffer>)L.uniforms offset:0 atIndex:3];
                    if (use_rhs)
                        [enc setBuffer:(__bridge id<MTLBuffer>)L.rhs offset:0 atIndex:4];
                    [enc setBytes:&parity length:sizeof(uint32_t) atIndex:(use_rhs ? 5 : 4)];
                    [enc dispatchThreads:MTLSizeMake(L.nx, L.ny, 1)
                       threadsPerThreadgroup:chooseThreadgroup2D(pso, L.nx, L.ny)];
                    [enc endEncoding];
                }
            }

            void* tmp = L.uA;
            L.uA = L.uB;
            L.uB = tmp;
        }

        encApplyBC(cb, lev);
    };


    auto compute_r2_norm = [&]() {
        id<MTLDevice> dev = (__bridge id<MTLDevice>)device_;
        id<MTLCommandQueue> qloc = (__bridge id<MTLCommandQueue>)queue_;
        id<MTLBuffer> src = (__bridge id<MTLBuffer>)levels_[0].rhs;
        if (!src) return NAN;

        size_t nbytes = size_t(levels_[0].nx) * size_t(levels_[0].ny) * sizeof(float);
        id<MTLBuffer> staging = [dev newBufferWithLength:nbytes options:MTLResourceStorageModeShared];

        id<MTLCommandBuffer> cb = [qloc commandBuffer];
        id<MTLBlitCommandEncoder> blit = [cb blitCommandEncoder];
        [blit copyFromBuffer:src sourceOffset:0 toBuffer:staging destinationOffset:0 size:nbytes];
        [blit endEncoding];
        [cb commit];
        [cb waitUntilCompleted];

        float* r = (float*)staging.contents;
        size_t n = nbytes / sizeof(float);
        double sum = 0.0;
        for (size_t i = 0; i < n; ++i) {
            double v = r[i];
            sum += v * v;
        }
        return (float)std::sqrt(sum);
    };

    {
        id<MTLCommandBuffer> cb = [q commandBuffer];
        encApplyBC(cb, 0);
        encResidual(cb, 0);
        [cb commit];
        [cb waitUntilCompleted];
    }

    float res = compute_r2_norm();
    float res0 = (res > 0.f) ? res : 1.f;
    if (std::isnan(res)) return false;
    bool absTolEnabled = (tol > 0.f);
    if (absTolEnabled && res <= tol) return true;

    std::ofstream rh("residual_history.csv");
    if (rh.is_open()) rh << "cycle,residual_L2,rel_residual\n";

    const float oscIncreaseEps = 1e-6f, oscBigJump = 0.10f;
    const uint32_t oscConsecUpLimit = 3;
    float prev_res = res;
    uint32_t consecUp = 0;
    const bool tightTol = ((absTolEnabled && tol <= 1e-9f) || (relTol_ > 0.0f && relTol_ <= 1e-9f));
    const uint32_t residualStrideCycles = (strictConvergence_ || tightTol) ? 1u : 25u;
    const size_t nlevels = levels_.size();

    // Optional CSV for per-level stats
    std::ofstream vcsv;
    if (statsEnabled_) {
        if (!statsCsvPath_.empty()) vcsv.open(statsCsvPath_);
        else vcsv.open("vcycle_stats.csv");
        if (vcsv.is_open()) vcsv << "cycle,level,nx,ny,nu1,nu2,coarse_iters,damping,smoother,res_init,res_final\n";
    }

    // Main V-cycle loop (modularized version)
    for (uint32_t cycle = 0; cycle < max_vcycles; ++cycle)
    {
        // Start a new command buffer for this V-cycle
        id<MTLCommandBuffer> cb = [q commandBuffer];

        // Pre-measure residuals at all levels (before cycle)
        std::vector<float> res_init_levels;
        if (statsEnabled_) {
            [cb commit]; [cb waitUntilCompleted];
            computeResidualsAllLevels_(res_init_levels);
            cb = [q commandBuffer];
        }

        // 1. Downward sweep (fine to coarse)
        if (cycleType_ == CycleType::V_Cycle) {
            downwardSweep_((__bridge void*)cb, nu1, nlevels);
        } else { // W_Cycle
            wCycleDownward_((__bridge void*)cb, nu1, nlevels, 0);
        }
        
        // 2. Coarse grid solve
        coarseGridSolve_((__bridge void*)cb, coarse_iters, nlevels);
        
        // 3. Upward sweep (coarse to fine)
        if (cycleType_ == CycleType::V_Cycle) {
            upwardSweep_((__bridge void*)cb, nu2, nlevels);
        } else { // W_Cycle
            wCycleUpward_((__bridge void*)cb, nu2, nlevels, 0);
        }

        encApplyBC(cb, 0);
        [cb commit];

        // Post-measure per-level residuals and log stats
        if (statsEnabled_) {
            [cb waitUntilCompleted];
            std::vector<float> res_final_levels;
            computeResidualsAllLevels_(res_final_levels);
            if (res_init_levels.size() == nlevels && res_final_levels.size() == nlevels) {
                for (size_t lev = 0; lev < nlevels; ++lev) {
                    VLevelStat s{};
                    s.cycle = cycle + 1;
                    s.level = static_cast<uint32_t>(lev);
                    s.nx = levels_[lev].nx; s.ny = levels_[lev].ny;
                    s.nu1 = nu1; s.nu2 = nu2; s.coarse_iters = coarse_iters;
                    s.damping = damping_; s.smoother = smoother_;
                    s.res_init = res_init_levels[lev];
                    s.res_final = res_final_levels[lev];
                    vstats_.push_back(s);
                    if (vcsv.is_open()) {
                        vcsv << s.cycle << "," << s.level << "," << s.nx << "," << s.ny << ","
                             << s.nu1 << "," << s.nu2 << "," << s.coarse_iters << ","
                             << s.damping << "," << (s.smoother==Smoother::SOR?"SOR":(s.smoother==Smoother::RBGS?"RBGS":"Jacobi")) << ","
                             << s.res_init << "," << s.res_final << "\n";
                    }
                }
            }
        }

        bool doMeasure = ((cycle + 1) % residualStrideCycles) == 0;
        if (doMeasure) {
            [cb waitUntilCompleted];
            
            float new_res = computeResidualAndCheck_(cycle, res0, prev_res, consecUp, tightTol, doMeasure, rh);
            if (std::isnan(new_res)) return false;
            
            if (absTolEnabled && new_res <= tol) return true;
            if (relTol_ > 0.0f && (new_res / res0) <= relTol_) return true;
            
            // Check for divergence patterns (early stopping logic)
            if (!strictConvergence_ && new_res > 0.f && prev_res > 0.f) {
                float relInc = (new_res - prev_res) / prev_res;
                
                if (relInc > oscBigJump) {
                    if (verbose_)
                        std::cout << "Stopping early: residual increased sharply by "
                                  << (relInc * 100.0f) << "% at cycle " << (cycle + 1) << "\n";
                    return true;
                }
                
                if (relInc > oscIncreaseEps) {
                    consecUp++;
                    if (consecUp >= oscConsecUpLimit) {
                        if (verbose_)
                            std::cout << "Stopping early: residual increased in " << consecUp
                                      << " consecutive cycles (oscillation detected) at cycle "
                                      << (cycle + 1) << "\n";
                        return true;
                    }
                } else {
                    consecUp = 0;
                }
            }
            prev_res = new_res;
        }
    }
    // If we exit due to reaching max_vcycles without early return, do one final residual measurement and log it.
    {
        id<MTLDevice> dev = (__bridge id<MTLDevice>)device_;
        id<MTLCommandQueue> qloc = (__bridge id<MTLCommandQueue>)queue_;
        auto &L0 = levels_[0];
        id<MTLCommandBuffer> rcb = [qloc commandBuffer];
        encResidual(rcb, 0);
        [rcb commit]; [rcb waitUntilCompleted];

        id<MTLComputePipelineState> psoSum = (__bridge id<MTLComputePipelineState>)psoSumSquaresPartial_;
        id<MTLComputePipelineState> psoRed = (__bridge id<MTLComputePipelineState>)psoReduceSum_;
        float finalRes = NAN;
        if (psoSum && psoRed) {
            const uint32_t tileX = 16, tileY = 16;
            uint32_t groupsX = (L0.nx + tileX - 1) / tileX;
            uint32_t groupsY = (L0.ny + tileY - 1) / tileY;
            uint32_t M = groupsX * groupsY;
            if (!ensurePartialBuffer0_(groupsX, groupsY)) return false;
            id<MTLBuffer> partial = (__bridge id<MTLBuffer>)bufPartial0_;

            id<MTLCommandBuffer> pcb = [qloc commandBuffer]; id<MTLComputeCommandEncoder> penc = [pcb computeCommandEncoder];
            [penc setComputePipelineState:psoSum];
            [penc setBuffer:(__bridge id<MTLBuffer>)L0.rhs offset:0 atIndex:0];
            [penc setBuffer:(__bridge id<MTLBuffer>)L0.uniforms offset:0 atIndex:1];
            [penc setBuffer:partial offset:0 atIndex:2];
            [penc dispatchThreadgroups:MTLSizeMake(groupsX, groupsY, 1) threadsPerThreadgroup:MTLSizeMake(tileX, tileY, 1)];
            [penc endEncoding]; [pcb commit]; [pcb waitUntilCompleted];

            id<MTLBuffer> out = [dev newBufferWithLength:sizeof(float) options:MTLResourceStorageModeShared];
            id<MTLCommandBuffer> r2cb = [qloc commandBuffer]; id<MTLComputeCommandEncoder> renc = [r2cb computeCommandEncoder];
            [renc setComputePipelineState:psoRed];
            [renc setBuffer:partial offset:0 atIndex:0];
            [renc setBuffer:out offset:0 atIndex:1];
            [renc setBytes:&M length:sizeof(uint32_t) atIndex:2];
            NSUInteger tcount = std::min<NSUInteger>(256, psoRed.maxTotalThreadsPerThreadgroup);
            [renc dispatchThreadgroups:MTLSizeMake(1,1,1) threadsPerThreadgroup:MTLSizeMake(tcount,1,1)];
            [renc endEncoding]; [r2cb commit]; [r2cb waitUntilCompleted];
            float sumsq = *(float*)out.contents; finalRes = std::sqrt(sumsq);
        } else {
            size_t nbytes = size_t(L0.nx) * size_t(L0.ny) * sizeof(float);
            id<MTLBuffer> staging = [dev newBufferWithLength:nbytes options:MTLResourceStorageModeShared];
            id<MTLCommandBuffer> blcb = [qloc commandBuffer]; id<MTLBlitCommandEncoder> blit = [blcb blitCommandEncoder];
            [blit copyFromBuffer:(__bridge id<MTLBuffer>)L0.rhs sourceOffset:0 toBuffer:staging destinationOffset:0 size:nbytes];
            [blit endEncoding]; [blcb commit]; [blcb waitUntilCompleted];
            float* r = (float*)staging.contents; size_t n = nbytes/sizeof(float); double sum = 0.0; for (size_t i=0;i<n;++i){ double v=r[i]; sum+=v*v; }
            finalRes = (float)std::sqrt(sum);
        }
        float finalRel = finalRes / res0;
        if (std::isnan(finalRes)) {
            std::cerr << "[FATAL] NaN residual at final measurement" << "\n";
            return false;
        }
        if (rh.is_open()) rh << max_vcycles << "," << finalRes << "," << finalRel << "\n";
        // Report whether any tolerance was met
        if (relTol_ > 0.0f) {
            return finalRel <= relTol_ || (absTolEnabled && finalRes <= tol);
        } else if (absTolEnabled) {
            return finalRes <= tol;
        } else {
            // No tolerance specified; treat as not converged
            return false;
        }
    }
    // Unreachable
    return false;
}

// ==================== MODULAR V-CYCLE HELPER FUNCTIONS ====================

void LaplaceMetalSolver::downwardSweep_(void* commandBuffer, uint32_t nu1, const size_t nlevels)
{
    id<MTLCommandBuffer> cb = (__bridge id<MTLCommandBuffer>)commandBuffer;
    
    // Recreate lambda functions for this sweep
    id<MTLComputePipelineState> psoApply = (__bridge id<MTLComputePipelineState>)psoApplyBC_;
    id<MTLComputePipelineState> psoResRaw = (__bridge id<MTLComputePipelineState>)psoResidualRaw_;
    id<MTLComputePipelineState> psoRestr = (__bridge id<MTLComputePipelineState>)psoRestrict_;
    id<MTLComputePipelineState> psoJac = (__bridge id<MTLComputePipelineState>)psoJacobi_;
    id<MTLComputePipelineState> psoJacR = (__bridge id<MTLComputePipelineState>)psoJacobiRHS_;
    id<MTLComputePipelineState> psoRB = (__bridge id<MTLComputePipelineState>)psoRBGS_;
    id<MTLComputePipelineState> psoRBRHS = (__bridge id<MTLComputePipelineState>)psoRBGSRHS_;
    id<MTLComputePipelineState> psoSOR = (__bridge id<MTLComputePipelineState>)psoSOR_;
    id<MTLComputePipelineState> psoSORRHS = (__bridge id<MTLComputePipelineState>)psoSORRHS_;
    id<MTLComputePipelineState> psoResSmooth = (__bridge id<MTLComputePipelineState>)psoResidualSmooth_;

    auto encApplyBC = [&](id<MTLCommandBuffer> cb, size_t lev) {
        auto &L = levels_[lev];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        [enc setComputePipelineState:psoApply];
        [enc setBuffer:(__bridge id<MTLBuffer>)L.uA offset:0 atIndex:0];
        [enc setBuffer:(__bridge id<MTLBuffer>)L.bcMask offset:0 atIndex:1];
        [enc setBuffer:(__bridge id<MTLBuffer>)L.bcVals offset:0 atIndex:2];
        [enc setBuffer:(__bridge id<MTLBuffer>)L.uniforms offset:0 atIndex:3];
        [enc dispatchThreads:MTLSizeMake(L.nx, L.ny, 1)
             threadsPerThreadgroup:chooseThreadgroup2D(psoApply, L.nx, L.ny)];
        [enc endEncoding];
    };

    auto encResidual = [&](id<MTLCommandBuffer> cb, size_t lev) {
        auto &L = levels_[lev]; 
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        [enc setComputePipelineState:psoResRaw];
        [enc setBuffer:(__bridge id<MTLBuffer>)L.uA offset:0 atIndex:0];
        [enc setBuffer:(__bridge id<MTLBuffer>)L.bcMask offset:0 atIndex:1];
        [enc setBuffer:(__bridge id<MTLBuffer>)L.rhs offset:0 atIndex:2];
        [enc setBuffer:(__bridge id<MTLBuffer>)L.uniforms offset:0 atIndex:3];
        [enc dispatchThreads:MTLSizeMake(L.nx, L.ny, 1) threadsPerThreadgroup:chooseThreadgroup2D(psoResRaw, L.nx, L.ny)];
        [enc endEncoding];
    };

    auto encRestrict = [&](id<MTLCommandBuffer> cb, size_t lev_from) {
        size_t lev_to = lev_from + 1;
        auto &Lf = levels_[lev_from];
        auto &Lc = levels_[lev_to];
        struct { uint32_t nx_f, ny_f, nx_c, ny_c; } gd = { Lf.nx, Lf.ny, Lc.nx, Lc.ny };
        id<MTLDevice> dev = (__bridge id<MTLDevice>)device_;
        id<MTLBuffer> gdBuf = [dev newBufferWithBytes:&gd length:sizeof(gd) options:MTLResourceStorageModeShared];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        [enc setComputePipelineState:psoRestr];
        [enc setBuffer:(__bridge id<MTLBuffer>)Lf.rhs offset:0 atIndex:0];
        [enc setBuffer:(__bridge id<MTLBuffer>)Lc.rhs offset:0 atIndex:1];
        [enc setBuffer:gdBuf offset:0 atIndex:2];
        [enc dispatchThreads:MTLSizeMake(Lc.nx, Lc.ny, 1)
           threadsPerThreadgroup:chooseThreadgroup2D(psoRestr, Lc.nx, Lc.ny)];
        [enc endEncoding];
    };

    auto encSmooth = [&](id<MTLCommandBuffer> cb, size_t lev, uint32_t iters, bool use_rhs) {
        if (iters == 0) return;
        auto &L = levels_[lev];

        for (uint32_t k = 0; k < iters; ++k) {
            if (smoother_ == Smoother::Jacobi) {
                id<MTLComputePipelineState> pso = use_rhs ? psoJacR : psoJac;
                id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
                [enc setComputePipelineState:pso];
                [enc setBuffer:(__bridge id<MTLBuffer>)L.uA offset:0 atIndex:0];
                [enc setBuffer:(__bridge id<MTLBuffer>)L.uB offset:0 atIndex:1];
                [enc setBuffer:(__bridge id<MTLBuffer>)L.bcMask offset:0 atIndex:2];
                [enc setBuffer:(__bridge id<MTLBuffer>)L.bcVals offset:0 atIndex:3];
                [enc setBuffer:(__bridge id<MTLBuffer>)L.uniforms offset:0 atIndex:4];
                if (use_rhs)
                    [enc setBuffer:(__bridge id<MTLBuffer>)L.rhs offset:0 atIndex:5];
                [enc dispatchThreads:MTLSizeMake(L.nx, L.ny, 1)
                   threadsPerThreadgroup:chooseThreadgroup2D(pso, L.nx, L.ny)];
                [enc endEncoding];
            } else if (smoother_ == Smoother::SOR) {
                id<MTLComputePipelineState> pso = use_rhs ? psoSORRHS : psoSOR;
                id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
                [enc setComputePipelineState:pso];
                [enc setBuffer:(__bridge id<MTLBuffer>)L.uA offset:0 atIndex:0];
                [enc setBuffer:(__bridge id<MTLBuffer>)L.bcMask offset:0 atIndex:1];
                [enc setBuffer:(__bridge id<MTLBuffer>)L.bcVals offset:0 atIndex:2];
                [enc setBuffer:(__bridge id<MTLBuffer>)L.uniforms offset:0 atIndex:3];
                if (use_rhs)
                    [enc setBuffer:(__bridge id<MTLBuffer>)L.rhs offset:0 atIndex:4];
                [enc dispatchThreads:MTLSizeMake(L.nx, L.ny, 1)
                   threadsPerThreadgroup:chooseThreadgroup2D(pso, L.nx, L.ny)];
                [enc endEncoding];
            } else {
                for (uint32_t parity = 0; parity < 2; ++parity) {
                    id<MTLComputePipelineState> pso = use_rhs ? psoRBRHS : psoRB;
                    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
                    [enc setComputePipelineState:pso];
                    [enc setBuffer:(__bridge id<MTLBuffer>)L.uA offset:0 atIndex:0];
                    [enc setBuffer:(__bridge id<MTLBuffer>)L.bcMask offset:0 atIndex:1];
                    [enc setBuffer:(__bridge id<MTLBuffer>)L.bcVals offset:0 atIndex:2];
                    [enc setBuffer:(__bridge id<MTLBuffer>)L.uniforms offset:0 atIndex:3];
                    if (use_rhs)
                        [enc setBuffer:(__bridge id<MTLBuffer>)L.rhs offset:0 atIndex:4];
                    [enc setBytes:&parity length:sizeof(uint32_t) atIndex:(use_rhs ? 5 : 4)];
                    [enc dispatchThreads:MTLSizeMake(L.nx, L.ny, 1)
                       threadsPerThreadgroup:chooseThreadgroup2D(pso, L.nx, L.ny)];
                    [enc endEncoding];
                }
            }

            void* tmp = L.uA;
            L.uA = L.uB;
            L.uB = tmp;
        }
        encApplyBC(cb, lev);
    };

    // Pre-smoothing, residual computation, optional residual smoothing, restriction, and BC application for each level
    for (size_t lev = 0; lev + 1 < nlevels; ++lev) {
        bool use_rhs = (lev > 0);  // First level uses RHS, others use restricted residual
        encSmooth(cb, lev, nu1, use_rhs);     // Pre-smoothing
        encResidual(cb, lev);                 // Compute raw Laplacian residual into rhs

        // Optionally smooth residual multiple times to reduce aliasing before restriction
        if (psoResSmooth && residualSmoothPasses_ > 0) {
            auto &L = levels_[lev];
            id<MTLDevice> dev = (__bridge id<MTLDevice>)device_;
            id<MTLBuffer> rNew = [dev newBufferWithLength:L.fieldBytes options:MTLResourceStorageModePrivate];
            if (!rNew) { /* fallthrough without smoothing if alloc fails */ }
            else {
                for (uint32_t pass = 0; pass < residualSmoothPasses_; ++pass) {
                    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
                    [enc setComputePipelineState:psoResSmooth];
                    [enc setBuffer:(__bridge id<MTLBuffer>)L.rhs offset:0 atIndex:0]; // r_old
                    [enc setBuffer:rNew offset:0 atIndex:1]; // r_new temp buffer
                    [enc setBuffer:(__bridge id<MTLBuffer>)L.bcMask offset:0 atIndex:2];
                    [enc setBuffer:(__bridge id<MTLBuffer>)L.uniforms offset:0 atIndex:3];
                    [enc dispatchThreads:MTLSizeMake(L.nx, L.ny, 1)
                       threadsPerThreadgroup:chooseThreadgroup2D(psoResSmooth, L.nx, L.ny)];
                    [enc endEncoding];
                    // Copy smoothed residual back into rhs for next pass
                    id<MTLBlitCommandEncoder> blit = [cb blitCommandEncoder];
                    [blit copyFromBuffer:rNew sourceOffset:0 toBuffer:(__bridge id<MTLBuffer>)L.rhs destinationOffset:0 size:L.fieldBytes];
                    [blit endEncoding];
                }
            }
        }
        encRestrict(cb, lev);                 // Restrict residual to coarser level
        encApplyBC(cb, lev + 1);             // Apply boundary conditions on coarser level
    }
}

void LaplaceMetalSolver::coarseGridSolve_(void* commandBuffer, uint32_t coarse_iters, const size_t nlevels)
{
    id<MTLCommandBuffer> cb = (__bridge id<MTLCommandBuffer>)commandBuffer;
    
    // Recreate smoother lambda for coarse grid
    id<MTLComputePipelineState> psoJac = (__bridge id<MTLComputePipelineState>)psoJacobi_;
    id<MTLComputePipelineState> psoJacR = (__bridge id<MTLComputePipelineState>)psoJacobiRHS_;
    id<MTLComputePipelineState> psoRB = (__bridge id<MTLComputePipelineState>)psoRBGS_;
    id<MTLComputePipelineState> psoRBRHS = (__bridge id<MTLComputePipelineState>)psoRBGSRHS_;
    id<MTLComputePipelineState> psoSOR = (__bridge id<MTLComputePipelineState>)psoSOR_;
    id<MTLComputePipelineState> psoSORRHS = (__bridge id<MTLComputePipelineState>)psoSORRHS_;
    id<MTLComputePipelineState> psoApply = (__bridge id<MTLComputePipelineState>)psoApplyBC_;

    auto encApplyBC = [&](id<MTLCommandBuffer> cb, size_t lev) {
        auto &L = levels_[lev];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        [enc setComputePipelineState:psoApply];
        [enc setBuffer:(__bridge id<MTLBuffer>)L.uA offset:0 atIndex:0];
        [enc setBuffer:(__bridge id<MTLBuffer>)L.bcMask offset:0 atIndex:1];
        [enc setBuffer:(__bridge id<MTLBuffer>)L.bcVals offset:0 atIndex:2];
        [enc setBuffer:(__bridge id<MTLBuffer>)L.uniforms offset:0 atIndex:3];
        [enc dispatchThreads:MTLSizeMake(L.nx, L.ny, 1)
             threadsPerThreadgroup:chooseThreadgroup2D(psoApply, L.nx, L.ny)];
        [enc endEncoding];
    };

    auto encSmooth = [&](id<MTLCommandBuffer> cb, size_t lev, uint32_t iters, bool use_rhs) {
        if (iters == 0) return;
        auto &L = levels_[lev];

        for (uint32_t k = 0; k < iters; ++k) {
            if (smoother_ == Smoother::Jacobi) {
                id<MTLComputePipelineState> pso = use_rhs ? psoJacR : psoJac;
                id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
                [enc setComputePipelineState:pso];
                [enc setBuffer:(__bridge id<MTLBuffer>)L.uA offset:0 atIndex:0];
                [enc setBuffer:(__bridge id<MTLBuffer>)L.uB offset:0 atIndex:1];
                [enc setBuffer:(__bridge id<MTLBuffer>)L.bcMask offset:0 atIndex:2];
                [enc setBuffer:(__bridge id<MTLBuffer>)L.bcVals offset:0 atIndex:3];
                [enc setBuffer:(__bridge id<MTLBuffer>)L.uniforms offset:0 atIndex:4];
                if (use_rhs)
                    [enc setBuffer:(__bridge id<MTLBuffer>)L.rhs offset:0 atIndex:5];
                [enc dispatchThreads:MTLSizeMake(L.nx, L.ny, 1)
                   threadsPerThreadgroup:chooseThreadgroup2D(pso, L.nx, L.ny)];
                [enc endEncoding];
            } else if (smoother_ == Smoother::SOR) {
                id<MTLComputePipelineState> pso = use_rhs ? psoSORRHS : psoSOR;
                id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
                [enc setComputePipelineState:pso];
                [enc setBuffer:(__bridge id<MTLBuffer>)L.uA offset:0 atIndex:0];
                [enc setBuffer:(__bridge id<MTLBuffer>)L.bcMask offset:0 atIndex:1];
                [enc setBuffer:(__bridge id<MTLBuffer>)L.bcVals offset:0 atIndex:2];
                [enc setBuffer:(__bridge id<MTLBuffer>)L.uniforms offset:0 atIndex:3];
                if (use_rhs)
                    [enc setBuffer:(__bridge id<MTLBuffer>)L.rhs offset:0 atIndex:4];
                [enc dispatchThreads:MTLSizeMake(L.nx, L.ny, 1)
                   threadsPerThreadgroup:chooseThreadgroup2D(pso, L.nx, L.ny)];
                [enc endEncoding];
            } else {
                for (uint32_t parity = 0; parity < 2; ++parity) {
                    id<MTLComputePipelineState> pso = use_rhs ? psoRBRHS : psoRB;
                    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
                    [enc setComputePipelineState:pso];
                    [enc setBuffer:(__bridge id<MTLBuffer>)L.uA offset:0 atIndex:0];
                    [enc setBuffer:(__bridge id<MTLBuffer>)L.bcMask offset:0 atIndex:1];
                    [enc setBuffer:(__bridge id<MTLBuffer>)L.bcVals offset:0 atIndex:2];
                    [enc setBuffer:(__bridge id<MTLBuffer>)L.uniforms offset:0 atIndex:3];
                    if (use_rhs)
                        [enc setBuffer:(__bridge id<MTLBuffer>)L.rhs offset:0 atIndex:4];
                    [enc setBytes:&parity length:sizeof(uint32_t) atIndex:(use_rhs ? 5 : 4)];
                    [enc dispatchThreads:MTLSizeMake(L.nx, L.ny, 1)
                       threadsPerThreadgroup:chooseThreadgroup2D(pso, L.nx, L.ny)];
                    [enc endEncoding];
                }
            }

            void* tmp = L.uA;
            L.uA = L.uB;
            L.uB = tmp;
        }
        encApplyBC(cb, lev);
    };
    
    // Zero the coarse-level correction (uA) to solve for error freshly each cycle
    id<MTLComputePipelineState> psoZero = (__bridge id<MTLComputePipelineState>)psoZeroFloat_;
    if (psoZero) {
        auto &Lc = levels_[nlevels - 1];
        id<MTLComputeCommandEncoder> zenc = [cb computeCommandEncoder];
        [zenc setComputePipelineState:psoZero];
        [zenc setBuffer:(__bridge id<MTLBuffer>)Lc.uA offset:0 atIndex:0];
        [zenc setBuffer:(__bridge id<MTLBuffer>)Lc.uniforms offset:0 atIndex:1];
        [zenc dispatchThreads:MTLSizeMake(Lc.nx, Lc.ny, 1)
           threadsPerThreadgroup:chooseThreadgroup2D(psoZero, Lc.nx, Lc.ny)];
        [zenc endEncoding];
    }
    
    // Solve on the coarsest grid: A*e = r (where e is the correction)
    encSmooth(cb, nlevels - 1, coarse_iters, true);  // use_rhs=true (solve with restricted residual)
}

void LaplaceMetalSolver::upwardSweep_(void* commandBuffer, uint32_t nu2, const size_t nlevels)
{
    id<MTLCommandBuffer> cb = (__bridge id<MTLCommandBuffer>)commandBuffer;
    
    // Recreate lambda functions for upward sweep
    id<MTLComputePipelineState> psoProl = (__bridge id<MTLComputePipelineState>)psoProlongAdd_;
    id<MTLComputePipelineState> psoJac = (__bridge id<MTLComputePipelineState>)psoJacobi_;
    id<MTLComputePipelineState> psoJacR = (__bridge id<MTLComputePipelineState>)psoJacobiRHS_;
    id<MTLComputePipelineState> psoRB = (__bridge id<MTLComputePipelineState>)psoRBGS_;
    id<MTLComputePipelineState> psoRBRHS = (__bridge id<MTLComputePipelineState>)psoRBGSRHS_;
    id<MTLComputePipelineState> psoSOR = (__bridge id<MTLComputePipelineState>)psoSOR_;
    id<MTLComputePipelineState> psoSORRHS = (__bridge id<MTLComputePipelineState>)psoSORRHS_;
    id<MTLComputePipelineState> psoApply = (__bridge id<MTLComputePipelineState>)psoApplyBC_;

    auto encApplyBC = [&](id<MTLCommandBuffer> cb, size_t lev) {
        auto &L = levels_[lev];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        [enc setComputePipelineState:psoApply];
        [enc setBuffer:(__bridge id<MTLBuffer>)L.uA offset:0 atIndex:0];
        [enc setBuffer:(__bridge id<MTLBuffer>)L.bcMask offset:0 atIndex:1];
        [enc setBuffer:(__bridge id<MTLBuffer>)L.bcVals offset:0 atIndex:2];
        [enc setBuffer:(__bridge id<MTLBuffer>)L.uniforms offset:0 atIndex:3];
        [enc dispatchThreads:MTLSizeMake(L.nx, L.ny, 1)
             threadsPerThreadgroup:chooseThreadgroup2D(psoApply, L.nx, L.ny)];
        [enc endEncoding];
    };

    auto encProlong = [&](id<MTLCommandBuffer> cb, size_t lev_to_fine) {
        auto &Lf = levels_[lev_to_fine];
        auto &Lc = levels_[lev_to_fine + 1];
        struct { uint32_t nx_f, ny_f, nx_c, ny_c; } gd = { Lf.nx, Lf.ny, Lc.nx, Lc.ny };
        id<MTLDevice> dev = (__bridge id<MTLDevice>)device_;
        id<MTLBuffer> gdBuf = [dev newBufferWithBytes:&gd length:sizeof(gd) options:MTLResourceStorageModeShared];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        [enc setComputePipelineState:psoProl];
        [enc setBuffer:(__bridge id<MTLBuffer>)Lc.uA offset:0 atIndex:0];
        [enc setBuffer:(__bridge id<MTLBuffer>)Lf.uA offset:0 atIndex:1];
        [enc setBuffer:gdBuf offset:0 atIndex:2];
        [enc dispatchThreads:MTLSizeMake(Lf.nx, Lf.ny, 1)
           threadsPerThreadgroup:chooseThreadgroup2D(psoProl, Lf.nx, Lf.ny)];
        [enc endEncoding];
    };

    auto encSmooth = [&](id<MTLCommandBuffer> cb, size_t lev, uint32_t iters, bool use_rhs) {
        if (iters == 0) return;
        auto &L = levels_[lev];

        for (uint32_t k = 0; k < iters; ++k) {
            if (smoother_ == Smoother::Jacobi) {
                id<MTLComputePipelineState> pso = use_rhs ? psoJacR : psoJac;
                id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
                [enc setComputePipelineState:pso];
                [enc setBuffer:(__bridge id<MTLBuffer>)L.uA offset:0 atIndex:0];
                [enc setBuffer:(__bridge id<MTLBuffer>)L.uB offset:0 atIndex:1];
                [enc setBuffer:(__bridge id<MTLBuffer>)L.bcMask offset:0 atIndex:2];
                [enc setBuffer:(__bridge id<MTLBuffer>)L.bcVals offset:0 atIndex:3];
                [enc setBuffer:(__bridge id<MTLBuffer>)L.uniforms offset:0 atIndex:4];
                if (use_rhs)
                    [enc setBuffer:(__bridge id<MTLBuffer>)L.rhs offset:0 atIndex:5];
                [enc dispatchThreads:MTLSizeMake(L.nx, L.ny, 1)
                   threadsPerThreadgroup:chooseThreadgroup2D(pso, L.nx, L.ny)];
                [enc endEncoding];
            } else if (smoother_ == Smoother::SOR) {
                id<MTLComputePipelineState> pso = use_rhs ? psoSORRHS : psoSOR;
                id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
                [enc setComputePipelineState:pso];
                [enc setBuffer:(__bridge id<MTLBuffer>)L.uA offset:0 atIndex:0];
                [enc setBuffer:(__bridge id<MTLBuffer>)L.bcMask offset:0 atIndex:1];
                [enc setBuffer:(__bridge id<MTLBuffer>)L.bcVals offset:0 atIndex:2];
                [enc setBuffer:(__bridge id<MTLBuffer>)L.uniforms offset:0 atIndex:3];
                if (use_rhs)
                    [enc setBuffer:(__bridge id<MTLBuffer>)L.rhs offset:0 atIndex:4];
                [enc dispatchThreads:MTLSizeMake(L.nx, L.ny, 1)
                   threadsPerThreadgroup:chooseThreadgroup2D(pso, L.nx, L.ny)];
                [enc endEncoding];
            } else {
                for (uint32_t parity = 0; parity < 2; ++parity) {
                    id<MTLComputePipelineState> pso = use_rhs ? psoRBRHS : psoRB;
                    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
                    [enc setComputePipelineState:pso];
                    [enc setBuffer:(__bridge id<MTLBuffer>)L.uA offset:0 atIndex:0];
                    [enc setBuffer:(__bridge id<MTLBuffer>)L.bcMask offset:0 atIndex:1];
                    [enc setBuffer:(__bridge id<MTLBuffer>)L.bcVals offset:0 atIndex:2];
                    [enc setBuffer:(__bridge id<MTLBuffer>)L.uniforms offset:0 atIndex:3];
                    if (use_rhs)
                        [enc setBuffer:(__bridge id<MTLBuffer>)L.rhs offset:0 atIndex:4];
                    [enc setBytes:&parity length:sizeof(uint32_t) atIndex:(use_rhs ? 5 : 4)];
                    [enc dispatchThreads:MTLSizeMake(L.nx, L.ny, 1)
                       threadsPerThreadgroup:chooseThreadgroup2D(pso, L.nx, L.ny)];
                    [enc endEncoding];
                }
            }

            void* tmp = L.uA;
            L.uA = L.uB;
            L.uB = tmp;
        }
        encApplyBC(cb, lev);
    };

    // Prolongation and post-smoothing from coarse to fine levels
    for (size_t lev = nlevels - 2;; --lev) {
        encProlong(cb, lev);                  // Prolong correction: u += P*e
        encSmooth(cb, lev, nu2, false);       // Post-smoothing (use_rhs=false)
        if (lev == 0) break;                  // Stop at finest level
    }
}

float LaplaceMetalSolver::computeResidualAndCheck_(uint32_t cycle, float res0, float prev_res, uint32_t& consecUp,
                                                  bool tightTol, bool doMeasure, std::ofstream& rh)
{
    id<MTLDevice> dev = (__bridge id<MTLDevice>)device_;
    id<MTLCommandQueue> q = (__bridge id<MTLCommandQueue>)queue_;
    id<MTLComputePipelineState> psoSum = (__bridge id<MTLComputePipelineState>)psoSumSquaresPartial_;
    id<MTLComputePipelineState> psoRed = (__bridge id<MTLComputePipelineState>)psoReduceSum_;
    
    float res = NAN;
    
    // Choose residual computation method based on tolerance requirements
    if (tightTol) {
        // CPU reduction in double precision for tight tolerances
        id<MTLCommandBuffer> rcb = [q commandBuffer];
        auto encResidual = [&](id<MTLCommandBuffer> cb, size_t lev) {
            auto &L = levels_[lev]; 
            id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
            id<MTLComputePipelineState> psoResRaw = (__bridge id<MTLComputePipelineState>)psoResidualRaw_;
            [enc setComputePipelineState:psoResRaw];
            [enc setBuffer:(__bridge id<MTLBuffer>)L.uA offset:0 atIndex:0];
            [enc setBuffer:(__bridge id<MTLBuffer>)L.bcMask offset:0 atIndex:1];
            [enc setBuffer:(__bridge id<MTLBuffer>)L.rhs offset:0 atIndex:2];
            [enc setBuffer:(__bridge id<MTLBuffer>)L.uniforms offset:0 atIndex:3];
            [enc dispatchThreads:MTLSizeMake(L.nx, L.ny, 1) threadsPerThreadgroup:chooseThreadgroup2D(psoResRaw, L.nx, L.ny)];
            [enc endEncoding];
        };
        encResidual(rcb, 0);
        [rcb commit]; [rcb waitUntilCompleted];
        
        size_t nbytes = size_t(levels_[0].nx) * size_t(levels_[0].ny) * sizeof(float);
        id<MTLBuffer> staging = [dev newBufferWithLength:nbytes options:MTLResourceStorageModeShared];
        id<MTLCommandBuffer> blcb = [q commandBuffer]; 
        id<MTLBlitCommandEncoder> blit = [blcb blitCommandEncoder];
        [blit copyFromBuffer:(__bridge id<MTLBuffer>)levels_[0].rhs sourceOffset:0 toBuffer:staging destinationOffset:0 size:nbytes];
        [blit endEncoding]; [blcb commit]; [blcb waitUntilCompleted];
        
        float* r = (float*)staging.contents; 
        size_t n = nbytes/sizeof(float); 
        long double sum = 0.0; 
        for (size_t i = 0; i < n; ++i) { 
            long double v = r[i]; 
            sum += v*v; 
        }
        res = (float)std::sqrt((double)sum);
        
    } else if (!psoSum || !psoRed) {
        // Fallback to CPU path if GPU reduction pipelines unavailable
        id<MTLCommandBuffer> rcb = [q commandBuffer];
        auto encResidual = [&](id<MTLCommandBuffer> cb, size_t lev) {
            auto &L = levels_[lev]; 
            id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
            id<MTLComputePipelineState> psoResRaw = (__bridge id<MTLComputePipelineState>)psoResidualRaw_;
            [enc setComputePipelineState:psoResRaw];
            [enc setBuffer:(__bridge id<MTLBuffer>)L.uA offset:0 atIndex:0];
            [enc setBuffer:(__bridge id<MTLBuffer>)L.bcMask offset:0 atIndex:1];
            [enc setBuffer:(__bridge id<MTLBuffer>)L.rhs offset:0 atIndex:2];
            [enc setBuffer:(__bridge id<MTLBuffer>)L.uniforms offset:0 atIndex:3];
            [enc dispatchThreads:MTLSizeMake(L.nx, L.ny, 1) threadsPerThreadgroup:chooseThreadgroup2D(psoResRaw, L.nx, L.ny)];
            [enc endEncoding];
        };
        encResidual(rcb, 0);
        [rcb commit]; [rcb waitUntilCompleted];
        
        size_t nbytes = size_t(levels_[0].nx) * size_t(levels_[0].ny) * sizeof(float);
        id<MTLBuffer> staging = [dev newBufferWithLength:nbytes options:MTLResourceStorageModeShared];
        id<MTLCommandBuffer> blcb = [q commandBuffer]; 
        id<MTLBlitCommandEncoder> blit = [blcb blitCommandEncoder];
        [blit copyFromBuffer:(__bridge id<MTLBuffer>)levels_[0].rhs sourceOffset:0 toBuffer:staging destinationOffset:0 size:nbytes];
        [blit endEncoding]; [blcb commit]; [blcb waitUntilCompleted];
        
        float* r = (float*)staging.contents; 
        size_t n = nbytes/sizeof(float); 
        double sum = 0.0; 
        for (size_t i = 0; i < n; ++i) { 
            double v = r[i]; 
            sum += v*v; 
        }
        res = (float)std::sqrt(sum);
        
    } else {
        // GPU-accelerated residual computation and reduction
        auto &L0 = levels_[0];
        
        // Compute residual into rhs buffer
        id<MTLCommandBuffer> rcb = [q commandBuffer];
        auto encResidual = [&](id<MTLCommandBuffer> cb, size_t lev) {
            auto &L = levels_[lev]; 
            id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
            id<MTLComputePipelineState> psoResRaw = (__bridge id<MTLComputePipelineState>)psoResidualRaw_;
            [enc setComputePipelineState:psoResRaw];
            [enc setBuffer:(__bridge id<MTLBuffer>)L.uA offset:0 atIndex:0];
            [enc setBuffer:(__bridge id<MTLBuffer>)L.bcMask offset:0 atIndex:1];
            [enc setBuffer:(__bridge id<MTLBuffer>)L.rhs offset:0 atIndex:2];
            [enc setBuffer:(__bridge id<MTLBuffer>)L.uniforms offset:0 atIndex:3];
            [enc dispatchThreads:MTLSizeMake(L.nx, L.ny, 1) threadsPerThreadgroup:chooseThreadgroup2D(psoResRaw, L.nx, L.ny)];
            [enc endEncoding];
        };
        encResidual(rcb, 0); 
        [rcb commit]; [rcb waitUntilCompleted];

        // Setup for partial sums reduction
        const uint32_t tileX = 16, tileY = 16;
        uint32_t groupsX = (L0.nx + tileX - 1) / tileX;
        uint32_t groupsY = (L0.ny + tileY - 1) / tileY;
        uint32_t M = groupsX * groupsY;
        if (!ensurePartialBuffer0_(groupsX, groupsY)) return NAN;
        id<MTLBuffer> partial = (__bridge id<MTLBuffer>)bufPartial0_;

        // Launch partial sum computation
        id<MTLCommandBuffer> pcb = [q commandBuffer]; 
        id<MTLComputeCommandEncoder> penc = [pcb computeCommandEncoder];
        [penc setComputePipelineState:psoSum];
        [penc setBuffer:(__bridge id<MTLBuffer>)L0.rhs offset:0 atIndex:0];
        [penc setBuffer:(__bridge id<MTLBuffer>)L0.uniforms offset:0 atIndex:1];
        [penc setBuffer:partial offset:0 atIndex:2];
        [penc dispatchThreadgroups:MTLSizeMake(groupsX, groupsY, 1) threadsPerThreadgroup:MTLSizeMake(tileX, tileY, 1)];
        [penc endEncoding]; [pcb commit]; [pcb waitUntilCompleted];

        // Final reduction to single sum
        id<MTLBuffer> out = [dev newBufferWithLength:sizeof(float) options:MTLResourceStorageModeShared];
        id<MTLCommandBuffer> r2cb = [q commandBuffer]; 
        id<MTLComputeCommandEncoder> renc = [r2cb computeCommandEncoder];
        [renc setComputePipelineState:psoRed];
        [renc setBuffer:partial offset:0 atIndex:0];
        [renc setBuffer:out offset:0 atIndex:1];
        [renc setBytes:&M length:sizeof(uint32_t) atIndex:2];
        NSUInteger tcount = std::min<NSUInteger>(256, psoRed.maxTotalThreadsPerThreadgroup);
        [renc dispatchThreadgroups:MTLSizeMake(1,1,1) threadsPerThreadgroup:MTLSizeMake(tcount,1,1)];
        [renc endEncoding]; [r2cb commit]; [r2cb waitUntilCompleted];
        
        float sumsq = *(float*)out.contents; 
        res = std::sqrt(sumsq);
    }

    // Safety boundary clamping
    clampToBoundaryRange_();

    // Log and report residual
    float rel = res / res0;
    if (verbose_) {
        std::cout << "MG cycle " << (cycle + 1)
                  << ", residual_L2=" << res
                  << ", rel=" << rel << std::endl;
    }
    if (rh.is_open()) {
        rh << (cycle + 1) << "," << res << "," << rel << "\n";
    }

    return res;
}

// ==================== W-CYCLE HELPER FUNCTIONS ====================

void LaplaceMetalSolver::wCycleDownward_(void* commandBuffer, uint32_t nu1, const size_t nlevels, size_t current_level)
{
    id<MTLCommandBuffer> cb = (__bridge id<MTLCommandBuffer>)commandBuffer;
    
    // Recreate lambda functions for this sweep
    id<MTLComputePipelineState> psoApply = (__bridge id<MTLComputePipelineState>)psoApplyBC_;
    id<MTLComputePipelineState> psoResRaw = (__bridge id<MTLComputePipelineState>)psoResidualRaw_;
    id<MTLComputePipelineState> psoRestr = (__bridge id<MTLComputePipelineState>)psoRestrict_;
    id<MTLComputePipelineState> psoJac = (__bridge id<MTLComputePipelineState>)psoJacobi_;
    id<MTLComputePipelineState> psoJacR = (__bridge id<MTLComputePipelineState>)psoJacobiRHS_;
    id<MTLComputePipelineState> psoRB = (__bridge id<MTLComputePipelineState>)psoRBGS_;
    id<MTLComputePipelineState> psoRBRHS = (__bridge id<MTLComputePipelineState>)psoRBGSRHS_;
    id<MTLComputePipelineState> psoSOR = (__bridge id<MTLComputePipelineState>)psoSOR_;
    id<MTLComputePipelineState> psoSORRHS = (__bridge id<MTLComputePipelineState>)psoSORRHS_;
    id<MTLComputePipelineState> psoResSmooth = (__bridge id<MTLComputePipelineState>)psoResidualSmooth_;

    auto encApplyBC = [&](id<MTLCommandBuffer> cb, size_t lev) {
        auto &L = levels_[lev];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        [enc setComputePipelineState:psoApply];
        [enc setBuffer:(__bridge id<MTLBuffer>)L.uA offset:0 atIndex:0];
        [enc setBuffer:(__bridge id<MTLBuffer>)L.bcMask offset:0 atIndex:1];
        [enc setBuffer:(__bridge id<MTLBuffer>)L.bcVals offset:0 atIndex:2];
        [enc setBuffer:(__bridge id<MTLBuffer>)L.uniforms offset:0 atIndex:3];
        [enc dispatchThreads:MTLSizeMake(L.nx, L.ny, 1)
             threadsPerThreadgroup:chooseThreadgroup2D(psoApply, L.nx, L.ny)];
        [enc endEncoding];
    };

    auto encResidual = [&](id<MTLCommandBuffer> cb, size_t lev) {
        auto &L = levels_[lev]; 
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        [enc setComputePipelineState:psoResRaw];
        [enc setBuffer:(__bridge id<MTLBuffer>)L.uA offset:0 atIndex:0];
        [enc setBuffer:(__bridge id<MTLBuffer>)L.bcMask offset:0 atIndex:1];
        [enc setBuffer:(__bridge id<MTLBuffer>)L.rhs offset:0 atIndex:2];
        [enc setBuffer:(__bridge id<MTLBuffer>)L.uniforms offset:0 atIndex:3];
        [enc dispatchThreads:MTLSizeMake(L.nx, L.ny, 1) threadsPerThreadgroup:chooseThreadgroup2D(psoResRaw, L.nx, L.ny)];
        [enc endEncoding];
    };

    auto encRestrict = [&](id<MTLCommandBuffer> cb, size_t lev_from) {
        size_t lev_to = lev_from + 1;
        auto &Lf = levels_[lev_from];
        auto &Lc = levels_[lev_to];
        struct { uint32_t nx_f, ny_f, nx_c, ny_c; } gd = { Lf.nx, Lf.ny, Lc.nx, Lc.ny };
        id<MTLDevice> dev = (__bridge id<MTLDevice>)device_;
        id<MTLBuffer> gdBuf = [dev newBufferWithBytes:&gd length:sizeof(gd) options:MTLResourceStorageModeShared];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        [enc setComputePipelineState:psoRestr];
        [enc setBuffer:(__bridge id<MTLBuffer>)Lf.rhs offset:0 atIndex:0];
        [enc setBuffer:(__bridge id<MTLBuffer>)Lc.rhs offset:0 atIndex:1];
        [enc setBuffer:gdBuf offset:0 atIndex:2];
        [enc dispatchThreads:MTLSizeMake(Lc.nx, Lc.ny, 1)
           threadsPerThreadgroup:chooseThreadgroup2D(psoRestr, Lc.nx, Lc.ny)];
        [enc endEncoding];
    };

    auto encSmooth = [&](id<MTLCommandBuffer> cb, size_t lev, uint32_t iters, bool use_rhs) {
        if (iters == 0) return;
        auto &L = levels_[lev];

        for (uint32_t k = 0; k < iters; ++k) {
            if (smoother_ == Smoother::Jacobi) {
                id<MTLComputePipelineState> pso = use_rhs ? psoJacR : psoJac;
                id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
                [enc setComputePipelineState:pso];
                [enc setBuffer:(__bridge id<MTLBuffer>)L.uA offset:0 atIndex:0];
                [enc setBuffer:(__bridge id<MTLBuffer>)L.uB offset:0 atIndex:1];
                [enc setBuffer:(__bridge id<MTLBuffer>)L.bcMask offset:0 atIndex:2];
                [enc setBuffer:(__bridge id<MTLBuffer>)L.bcVals offset:0 atIndex:3];
                [enc setBuffer:(__bridge id<MTLBuffer>)L.uniforms offset:0 atIndex:4];
                if (use_rhs)
                    [enc setBuffer:(__bridge id<MTLBuffer>)L.rhs offset:0 atIndex:5];
                [enc dispatchThreads:MTLSizeMake(L.nx, L.ny, 1)
                   threadsPerThreadgroup:chooseThreadgroup2D(pso, L.nx, L.ny)];
                [enc endEncoding];
            } else if (smoother_ == Smoother::SOR) {
                id<MTLComputePipelineState> pso = use_rhs ? psoSORRHS : psoSOR;
                id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
                [enc setComputePipelineState:pso];
                [enc setBuffer:(__bridge id<MTLBuffer>)L.uA offset:0 atIndex:0];
                [enc setBuffer:(__bridge id<MTLBuffer>)L.bcMask offset:0 atIndex:1];
                [enc setBuffer:(__bridge id<MTLBuffer>)L.bcVals offset:0 atIndex:2];
                [enc setBuffer:(__bridge id<MTLBuffer>)L.uniforms offset:0 atIndex:3];
                if (use_rhs)
                    [enc setBuffer:(__bridge id<MTLBuffer>)L.rhs offset:0 atIndex:4];
                [enc dispatchThreads:MTLSizeMake(L.nx, L.ny, 1)
                   threadsPerThreadgroup:chooseThreadgroup2D(pso, L.nx, L.ny)];
                [enc endEncoding];
            } else {
                for (uint32_t parity = 0; parity < 2; ++parity) {
                    id<MTLComputePipelineState> pso = use_rhs ? psoRBRHS : psoRB;
                    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
                    [enc setComputePipelineState:pso];
                    [enc setBuffer:(__bridge id<MTLBuffer>)L.uA offset:0 atIndex:0];
                    [enc setBuffer:(__bridge id<MTLBuffer>)L.bcMask offset:0 atIndex:1];
                    [enc setBuffer:(__bridge id<MTLBuffer>)L.bcVals offset:0 atIndex:2];
                    [enc setBuffer:(__bridge id<MTLBuffer>)L.uniforms offset:0 atIndex:3];
                    if (use_rhs)
                        [enc setBuffer:(__bridge id<MTLBuffer>)L.rhs offset:0 atIndex:4];
                    [enc setBytes:&parity length:sizeof(uint32_t) atIndex:(use_rhs ? 5 : 4)];
                    [enc dispatchThreads:MTLSizeMake(L.nx, L.ny, 1)
                       threadsPerThreadgroup:chooseThreadgroup2D(pso, L.nx, L.ny)];
                    [enc endEncoding];
                }
            }

            void* tmp = L.uA;
            L.uA = L.uB;
            L.uB = tmp;
        }
        encApplyBC(cb, lev);
    };

    // Modified W-cycle: extra smoothing on intermediate levels
    if (current_level + 1 < nlevels) {
        // Pre-smoothing on current level
        bool use_rhs = (current_level > 0);
        encSmooth(cb, current_level, nu1, use_rhs);
        encResidual(cb, current_level);

        // Optional residual smoothing
        if (psoResSmooth && residualSmoothPasses_ > 0) {
            auto &L = levels_[current_level];
            id<MTLDevice> dev = (__bridge id<MTLDevice>)device_;
            id<MTLBuffer> rNew = [dev newBufferWithLength:L.fieldBytes options:MTLResourceStorageModePrivate];
            if (!rNew) { /* fallthrough without smoothing if alloc fails */ }
            else {
                for (uint32_t pass = 0; pass < residualSmoothPasses_; ++pass) {
                    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
                    [enc setComputePipelineState:psoResSmooth];
                    [enc setBuffer:(__bridge id<MTLBuffer>)L.rhs offset:0 atIndex:0];
                    [enc setBuffer:rNew offset:0 atIndex:1];
                    [enc setBuffer:(__bridge id<MTLBuffer>)L.bcMask offset:0 atIndex:2];
                    [enc setBuffer:(__bridge id<MTLBuffer>)L.uniforms offset:0 atIndex:3];
                    [enc dispatchThreads:MTLSizeMake(L.nx, L.ny, 1)
                       threadsPerThreadgroup:chooseThreadgroup2D(psoResSmooth, L.nx, L.ny)];
                    [enc endEncoding];
                    id<MTLBlitCommandEncoder> blit = [cb blitCommandEncoder];
                    [blit copyFromBuffer:rNew sourceOffset:0 toBuffer:(__bridge id<MTLBuffer>)L.rhs destinationOffset:0 size:L.fieldBytes];
                    [blit endEncoding];
                }
            }
        }

        // Restrict to next level
        encRestrict(cb, current_level);
        encApplyBC(cb, current_level + 1);

        // Recursively go to next level
        wCycleDownward_(commandBuffer, nu1, nlevels, current_level + 1);

        // For W-cycle effect: do extra smoothing on intermediate levels
        if (current_level > 0 && current_level < nlevels - 2) {
            // Extra smoothing on intermediate levels for W-cycle effect
            encSmooth(cb, current_level, nu1, false);
        }
    }
}

void LaplaceMetalSolver::wCycleUpward_(void* commandBuffer, uint32_t nu2, const size_t nlevels, size_t current_level)
{
    id<MTLCommandBuffer> cb = (__bridge id<MTLCommandBuffer>)commandBuffer;
    
    // Recreate lambda functions for upward sweep
    id<MTLComputePipelineState> psoProl = (__bridge id<MTLComputePipelineState>)psoProlongAdd_;
    id<MTLComputePipelineState> psoJac = (__bridge id<MTLComputePipelineState>)psoJacobi_;
    id<MTLComputePipelineState> psoJacR = (__bridge id<MTLComputePipelineState>)psoJacobiRHS_;
    id<MTLComputePipelineState> psoRB = (__bridge id<MTLComputePipelineState>)psoRBGS_;
    id<MTLComputePipelineState> psoRBRHS = (__bridge id<MTLComputePipelineState>)psoRBGSRHS_;
    id<MTLComputePipelineState> psoSOR = (__bridge id<MTLComputePipelineState>)psoSOR_;
    id<MTLComputePipelineState> psoSORRHS = (__bridge id<MTLComputePipelineState>)psoSORRHS_;
    id<MTLComputePipelineState> psoApply = (__bridge id<MTLComputePipelineState>)psoApplyBC_;

    auto encApplyBC = [&](id<MTLCommandBuffer> cb, size_t lev) {
        auto &L = levels_[lev];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        [enc setComputePipelineState:psoApply];
        [enc setBuffer:(__bridge id<MTLBuffer>)L.uA offset:0 atIndex:0];
        [enc setBuffer:(__bridge id<MTLBuffer>)L.bcMask offset:0 atIndex:1];
        [enc setBuffer:(__bridge id<MTLBuffer>)L.bcVals offset:0 atIndex:2];
        [enc setBuffer:(__bridge id<MTLBuffer>)L.uniforms offset:0 atIndex:3];
        [enc dispatchThreads:MTLSizeMake(L.nx, L.ny, 1)
             threadsPerThreadgroup:chooseThreadgroup2D(psoApply, L.nx, L.ny)];
        [enc endEncoding];
    };

    auto encProlong = [&](id<MTLCommandBuffer> cb, size_t lev_to_fine) {
        auto &Lf = levels_[lev_to_fine];
        auto &Lc = levels_[lev_to_fine + 1];
        struct { uint32_t nx_f, ny_f, nx_c, ny_c; } gd = { Lf.nx, Lf.ny, Lc.nx, Lc.ny };
        id<MTLDevice> dev = (__bridge id<MTLDevice>)device_;
        id<MTLBuffer> gdBuf = [dev newBufferWithBytes:&gd length:sizeof(gd) options:MTLResourceStorageModeShared];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        [enc setComputePipelineState:psoProl];
        [enc setBuffer:(__bridge id<MTLBuffer>)Lc.uA offset:0 atIndex:0];
        [enc setBuffer:(__bridge id<MTLBuffer>)Lf.uA offset:0 atIndex:1];
        [enc setBuffer:gdBuf offset:0 atIndex:2];
        [enc dispatchThreads:MTLSizeMake(Lf.nx, Lf.ny, 1)
           threadsPerThreadgroup:chooseThreadgroup2D(psoProl, Lf.nx, Lf.ny)];
        [enc endEncoding];
    };

    auto encSmooth = [&](id<MTLCommandBuffer> cb, size_t lev, uint32_t iters, bool use_rhs) {
        if (iters == 0) return;
        auto &L = levels_[lev];

        for (uint32_t k = 0; k < iters; ++k) {
            if (smoother_ == Smoother::Jacobi) {
                id<MTLComputePipelineState> pso = use_rhs ? psoJacR : psoJac;
                id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
                [enc setComputePipelineState:pso];
                [enc setBuffer:(__bridge id<MTLBuffer>)L.uA offset:0 atIndex:0];
                [enc setBuffer:(__bridge id<MTLBuffer>)L.uB offset:0 atIndex:1];
                [enc setBuffer:(__bridge id<MTLBuffer>)L.bcMask offset:0 atIndex:2];
                [enc setBuffer:(__bridge id<MTLBuffer>)L.bcVals offset:0 atIndex:3];
                [enc setBuffer:(__bridge id<MTLBuffer>)L.uniforms offset:0 atIndex:4];
                if (use_rhs)
                    [enc setBuffer:(__bridge id<MTLBuffer>)L.rhs offset:0 atIndex:5];
                [enc dispatchThreads:MTLSizeMake(L.nx, L.ny, 1)
                   threadsPerThreadgroup:chooseThreadgroup2D(pso, L.nx, L.ny)];
                [enc endEncoding];
            } else if (smoother_ == Smoother::SOR) {
                id<MTLComputePipelineState> pso = use_rhs ? psoSORRHS : psoSOR;
                id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
                [enc setComputePipelineState:pso];
                [enc setBuffer:(__bridge id<MTLBuffer>)L.uA offset:0 atIndex:0];
                [enc setBuffer:(__bridge id<MTLBuffer>)L.bcMask offset:0 atIndex:1];
                [enc setBuffer:(__bridge id<MTLBuffer>)L.bcVals offset:0 atIndex:2];
                [enc setBuffer:(__bridge id<MTLBuffer>)L.uniforms offset:0 atIndex:3];
                if (use_rhs)
                    [enc setBuffer:(__bridge id<MTLBuffer>)L.rhs offset:0 atIndex:4];
                [enc dispatchThreads:MTLSizeMake(L.nx, L.ny, 1)
                   threadsPerThreadgroup:chooseThreadgroup2D(pso, L.nx, L.ny)];
                [enc endEncoding];
            } else {
                for (uint32_t parity = 0; parity < 2; ++parity) {
                    id<MTLComputePipelineState> pso = use_rhs ? psoRBRHS : psoRB;
                    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
                    [enc setComputePipelineState:pso];
                    [enc setBuffer:(__bridge id<MTLBuffer>)L.uA offset:0 atIndex:0];
                    [enc setBuffer:(__bridge id<MTLBuffer>)L.bcMask offset:0 atIndex:1];
                    [enc setBuffer:(__bridge id<MTLBuffer>)L.bcVals offset:0 atIndex:2];
                    [enc setBuffer:(__bridge id<MTLBuffer>)L.uniforms offset:0 atIndex:3];
                    if (use_rhs)
                        [enc setBuffer:(__bridge id<MTLBuffer>)L.rhs offset:0 atIndex:4];
                    [enc setBytes:&parity length:sizeof(uint32_t) atIndex:(use_rhs ? 5 : 4)];
                    [enc dispatchThreads:MTLSizeMake(L.nx, L.ny, 1)
                       threadsPerThreadgroup:chooseThreadgroup2D(pso, L.nx, L.ny)];
                    [enc endEncoding];
                }
            }

            void* tmp = L.uA;
            L.uA = L.uB;
            L.uB = tmp;
        }
        encApplyBC(cb, lev);
    };

    // Modified W-cycle upward sweep: extra smoothing on intermediate levels
    if (current_level + 1 < nlevels) {
        // Recursively go to next level first
        wCycleUpward_(commandBuffer, nu2, nlevels, current_level + 1);

        // Prolongate and post-smooth on current level
        encProlong(cb, current_level);
        encSmooth(cb, current_level, nu2, false);

        // For W-cycle effect: do extra smoothing on intermediate levels
        if (current_level > 0 && current_level < nlevels - 2) {
            // Extra smoothing on intermediate levels for W-cycle effect
            encSmooth(cb, current_level, nu2, false);
        }
    }
}

// Compute L2 residual at a specific level: r = b - A u, returns ||r||_2
float LaplaceMetalSolver::computeResidualL2Level_(size_t lev)
{
    if (lev >= levels_.size()) return NAN;
    id<MTLCommandQueue> qloc = (__bridge id<MTLCommandQueue>)queue_;
    id<MTLDevice> dev = (__bridge id<MTLDevice>)device_;
    id<MTLComputePipelineState> psoResRaw = (__bridge id<MTLComputePipelineState>)psoResidualRaw_;
    if (!psoResRaw) return NAN;
    auto &L = levels_[lev];
    id<MTLCommandBuffer> cb = [qloc commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
    [enc setComputePipelineState:psoResRaw];
    [enc setBuffer:(__bridge id<MTLBuffer>)L.uA offset:0 atIndex:0];
    [enc setBuffer:(__bridge id<MTLBuffer>)L.bcMask offset:0 atIndex:1];
    [enc setBuffer:(__bridge id<MTLBuffer>)L.rhs offset:0 atIndex:2];
    [enc setBuffer:(__bridge id<MTLBuffer>)L.uniforms offset:0 atIndex:3];
    [enc dispatchThreads:MTLSizeMake(L.nx, L.ny, 1)
       threadsPerThreadgroup:chooseThreadgroup2D(psoResRaw, L.nx, L.ny)];
    [enc endEncoding];
    [cb commit]; [cb waitUntilCompleted];

    size_t nbytes = size_t(L.nx) * size_t(L.ny) * sizeof(float);
    id<MTLBuffer> staging = [dev newBufferWithLength:nbytes options:MTLResourceStorageModeShared];
    id<MTLCommandBuffer> blcb = [qloc commandBuffer]; id<MTLBlitCommandEncoder> blit = [blcb blitCommandEncoder];
    [blit copyFromBuffer:(__bridge id<MTLBuffer>)L.rhs sourceOffset:0 toBuffer:staging destinationOffset:0 size:nbytes];
    [blit endEncoding]; [blcb commit]; [blcb waitUntilCompleted];
    float* r = (float*)staging.contents; size_t n = nbytes/sizeof(float);
    double sum = 0.0; for (size_t i=0;i<n;++i){ double v=r[i]; sum+=v*v; }
    return (float)std::sqrt(sum);
}

void LaplaceMetalSolver::computeResidualsAllLevels_(std::vector<float>& out)
{
    out.clear(); out.reserve(levels_.size());
    for (size_t lev = 0; lev < levels_.size(); ++lev) {
        out.push_back(computeResidualL2Level_(lev));
    }
}
