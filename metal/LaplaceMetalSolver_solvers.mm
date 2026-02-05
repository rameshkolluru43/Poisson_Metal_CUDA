// Solver routines and metrics pulled from the monolithic implementation
#import <Metal/Metal.h>
#include "LaplaceMetalSolver.hpp"
#include "LaplaceMetalSolverInternal.hpp"
#include <cmath>
#include <limits>
#include <vector>
#include <iostream>
#include <fstream>

bool LaplaceMetalSolver::applyBC_(){
    if (levels_.empty()) return false;
    id<MTLCommandQueue> q=(__bridge id<MTLCommandQueue>)queue_;
    id<MTLCommandBuffer> cb=[q commandBuffer];
    id<MTLComputePipelineState> pso=(__bridge id<MTLComputePipelineState>)psoApplyBC_;
    id<MTLComputeCommandEncoder> enc=[cb computeCommandEncoder];
    const auto &L=levels_[0];
    [enc setComputePipelineState:pso];
    [enc setBuffer:(__bridge id<MTLBuffer>)L.uA offset:0 atIndex:0];
    [enc setBuffer:(__bridge id<MTLBuffer>)L.bcMask offset:0 atIndex:1];
    [enc setBuffer:(__bridge id<MTLBuffer>)L.bcVals offset:0 atIndex:2];
    [enc setBuffer:(__bridge id<MTLBuffer>)L.uniforms offset:0 atIndex:3];
    [enc dispatchThreads:MTLSizeMake(L.nx,L.ny,1) threadsPerThreadgroup:chooseThreadgroup2D(pso,L.nx,L.ny)];
    [enc endEncoding]; [cb commit]; [cb waitUntilCompleted];
    return true;
}

float LaplaceMetalSolver::computeResidualL2_(){
    if (levels_.empty()) return NAN;
    id<MTLCommandQueue> q=(__bridge id<MTLCommandQueue>)queue_;
    id<MTLCommandBuffer> cb=[q commandBuffer];
    id<MTLComputePipelineState> pso=(__bridge id<MTLComputePipelineState>)psoResidualRaw_;
    auto &L=levels_[0];
    id<MTLComputeCommandEncoder> enc=[cb computeCommandEncoder];
    [enc setComputePipelineState:pso];
    [enc setBuffer:(__bridge id<MTLBuffer>)L.uA offset:0 atIndex:0];
    [enc setBuffer:(__bridge id<MTLBuffer>)L.bcMask offset:0 atIndex:1];
    [enc setBuffer:(__bridge id<MTLBuffer>)L.rhs offset:0 atIndex:2];
    [enc setBuffer:(__bridge id<MTLBuffer>)L.uniforms offset:0 atIndex:3];
    [enc dispatchThreads:MTLSizeMake(L.nx,L.ny,1) threadsPerThreadgroup:chooseThreadgroup2D(pso,L.nx,L.ny)];
    [enc endEncoding]; [cb commit]; [cb waitUntilCompleted];
    id<MTLDevice> dev=(__bridge id<MTLDevice>)device_;
    size_t nbytes=size_t(L.nx)*size_t(L.ny)*sizeof(float);
    id<MTLBuffer> staging=[dev newBufferWithLength:nbytes options:MTLResourceStorageModeShared];
    id<MTLCommandBuffer> blcb=[q commandBuffer]; id<MTLBlitCommandEncoder> blit=[blcb blitCommandEncoder];
    [blit copyFromBuffer:(__bridge id<MTLBuffer>)L.rhs sourceOffset:0 toBuffer:staging destinationOffset:0 size:nbytes];
    [blit endEncoding]; [blcb commit]; [blcb waitUntilCompleted];
    float* r=(float*)staging.contents; size_t n=nbytes/sizeof(float); double sum=0.0; for(size_t i=0;i<n;++i){ double v=r[i]; sum+=v*v; }
    return (float)std::sqrt(sum);
}

float LaplaceMetalSolver::computeResidualL2Fused_(uint32_t groupsX, uint32_t groupsY){
    id<MTLDevice> dev=(__bridge id<MTLDevice>)device_;
    id<MTLCommandQueue> q=(__bridge id<MTLCommandQueue>)queue_;
    if (!psoReduceSum_) return NAN;
    uint32_t M = groupsX * groupsY;
    id<MTLBuffer> partial = [dev newBufferWithLength:sizeof(float)*M options:MTLResourceStorageModeShared];
    id<MTLBuffer> out = [dev newBufferWithLength:sizeof(float) options:MTLResourceStorageModeShared];
    if (!partial || !out) return NAN;
    id<MTLComputePipelineState> psoRed=(__bridge id<MTLComputePipelineState>)psoReduceSum_;
    id<MTLCommandBuffer> cb=[q commandBuffer]; id<MTLComputeCommandEncoder> enc=[cb computeCommandEncoder];
    [enc setComputePipelineState:psoRed];
    [enc setBuffer:partial offset:0 atIndex:0];
    [enc setBuffer:out offset:0 atIndex:1];
    [enc setBytes:&M length:sizeof(uint32_t) atIndex:2];
    NSUInteger tcount = std::min<NSUInteger>(256, psoRed.maxTotalThreadsPerThreadgroup);
    [enc dispatchThreadgroups:MTLSizeMake(1,1,1) threadsPerThreadgroup:MTLSizeMake(tcount,1,1)];
    [enc endEncoding]; [cb commit]; [cb waitUntilCompleted];
    float sumsq = *(float*)out.contents; return std::sqrt(sumsq);
}

void LaplaceMetalSolver::swapFields_(){ if (levels_.empty()) return; std::swap(levels_[0].uA, levels_[0].uB); }

void LaplaceMetalSolver::clampToBoundaryRange_(){
    if (!clampEnabled_ || !psoClamp_ || levels_.empty()) return;
    id<MTLCommandQueue> q=(__bridge id<MTLCommandQueue>)queue_;
    id<MTLCommandBuffer> cb=[q commandBuffer];
    id<MTLComputePipelineState> pso=(__bridge id<MTLComputePipelineState>)psoClamp_;
    id<MTLComputeCommandEncoder> enc=[cb computeCommandEncoder];
    auto &L=levels_[0];
    [enc setComputePipelineState:pso];
    [enc setBuffer:(__bridge id<MTLBuffer>)L.uA offset:0 atIndex:0];
    [enc setBuffer:(__bridge id<MTLBuffer>)L.bcMask offset:0 atIndex:1];
    [enc setBuffer:(__bridge id<MTLBuffer>)L.uniforms offset:0 atIndex:2];
    struct { float x; float y; } bounds={bcMin_, bcMax_};
    [enc setBytes:&bounds length:sizeof(bounds) atIndex:3];
    [enc dispatchThreads:MTLSizeMake(L.nx,L.ny,1) threadsPerThreadgroup:chooseThreadgroup2D(pso,L.nx,L.ny)];
    [enc endEncoding]; [cb commit]; [cb waitUntilCompleted];
}

bool LaplaceMetalSolver::solveJacobi(uint32_t max_iters, float tol, uint32_t *out_iters, float *residual_l2){
    if (levels_.empty()) { if (!buildLevels_()) return false; }
    auto &L=levels_[0];
    id<MTLCommandQueue> q=(__bridge id<MTLCommandQueue>)queue_;
    id<MTLComputePipelineState> pso=(__bridge id<MTLComputePipelineState>)psoJacobi_;
    applyBC_();
    float res=computeResidualL2_(); if (residual_l2) *residual_l2 = res; if (res<=tol){ if(out_iters) *out_iters=0; return true; }
    const uint32_t stride=50; uint32_t it=0;
    for (it=0; it<max_iters; ++it){
        id<MTLCommandBuffer> cb=[q commandBuffer]; id<MTLComputeCommandEncoder> enc=[cb computeCommandEncoder];
        [enc setComputePipelineState:pso];
        [enc setBuffer:(__bridge id<MTLBuffer>)L.uA offset:0 atIndex:0];
        [enc setBuffer:(__bridge id<MTLBuffer>)L.uB offset:0 atIndex:1];
        [enc setBuffer:(__bridge id<MTLBuffer>)L.bcMask offset:0 atIndex:2];
        [enc setBuffer:(__bridge id<MTLBuffer>)L.bcVals offset:0 atIndex:3];
        [enc setBuffer:(__bridge id<MTLBuffer>)L.uniforms offset:0 atIndex:4];
        [enc dispatchThreads:MTLSizeMake(L.nx,L.ny,1) threadsPerThreadgroup:chooseThreadgroup2D(pso,L.nx,L.ny)];
        [enc endEncoding]; [cb commit]; [cb waitUntilCompleted];
        swapFields_(); applyBC_();
        if ((it+1)%stride==0 || it+1==max_iters){ res=computeResidualL2_(); if (residual_l2) *residual_l2=res; if (res<=tol){ ++it; break; } }
    }
    clampToBoundaryRange_(); if (out_iters) *out_iters=it; return res<=tol;
}

bool LaplaceMetalSolver::solveJacobiFused(uint32_t max_iters, float tol, uint32_t flushStride, uint32_t *out_iters, float *residual_l2){
    if (levels_.empty()) { if (!buildLevels_()) return false; }
    auto &L=levels_[0];
    id<MTLDevice> dev=(__bridge id<MTLDevice>)device_;
    id<MTLCommandQueue> q=(__bridge id<MTLCommandQueue>)queue_;
    id<MTLComputePipelineState> psoF=(__bridge id<MTLComputePipelineState>)psoJacobiFused_;
    if (!psoF) return false;

    const uint32_t tileX = 16, tileY = 16;
    uint32_t groupsX = (L.nx + tileX - 1) / tileX;
    uint32_t groupsY = (L.ny + tileY - 1) / tileY;
    uint32_t M = groupsX * groupsY;
    if (!ensurePartialBuffer0_(groupsX, groupsY)) return false;
    id<MTLBuffer> partial = (__bridge id<MTLBuffer>)bufPartial0_;

    // Initial BC and residual measurement
    applyBC_();
    float res = computeResidualL2_();
    if (residual_l2) *residual_l2 = res;
    if (res <= tol) { if (out_iters) *out_iters = 0; return true; }

    uint32_t it = 0;
    for (it = 0; it < max_iters; ++it){
        id<MTLCommandBuffer> cb=[q commandBuffer]; id<MTLComputeCommandEncoder> enc=[cb computeCommandEncoder];
        [enc setComputePipelineState:psoF];
        [enc setBuffer:(__bridge id<MTLBuffer>)L.uA offset:0 atIndex:0];
        [enc setBuffer:(__bridge id<MTLBuffer>)L.uB offset:0 atIndex:1];
        [enc setBuffer:(__bridge id<MTLBuffer>)L.bcMask offset:0 atIndex:2];
        [enc setBuffer:(__bridge id<MTLBuffer>)L.bcVals offset:0 atIndex:3];
        [enc setBuffer:(__bridge id<MTLBuffer>)L.uniforms offset:0 atIndex:4];
        [enc setBuffer:partial offset:0 atIndex:5];
        [enc dispatchThreadgroups:MTLSizeMake(groupsX, groupsY, 1) threadsPerThreadgroup:MTLSizeMake(tileX, tileY, 1)];
        [enc endEncoding]; [cb commit];

        // swap buffers on host tracking without extra copy
        std::swap(L.uA, L.uB);

        bool doMeasure = ((it + 1) % flushStride) == 0 || (it + 1) == max_iters;
        if (doMeasure){
            [cb waitUntilCompleted];
            // Reduce partial sums to one value
            id<MTLComputePipelineState> psoRed=(__bridge id<MTLComputePipelineState>)psoReduceSum_;
            if (!psoRed) return false;
            id<MTLBuffer> out=[dev newBufferWithLength:sizeof(float) options:MTLResourceStorageModeShared];
            id<MTLCommandBuffer> rcb=[q commandBuffer]; id<MTLComputeCommandEncoder> renc=[rcb computeCommandEncoder];
            [renc setComputePipelineState:psoRed];
            [renc setBuffer:partial offset:0 atIndex:0];
            [renc setBuffer:out offset:0 atIndex:1];
            [renc setBytes:&M length:sizeof(uint32_t) atIndex:2];
            NSUInteger tcount = std::min<NSUInteger>(256, psoRed.maxTotalThreadsPerThreadgroup);
            [renc dispatchThreadgroups:MTLSizeMake(1,1,1) threadsPerThreadgroup:MTLSizeMake(tcount,1,1)];
            [renc endEncoding]; [rcb commit]; [rcb waitUntilCompleted];
            float sumsq = *(float*)out.contents; res = std::sqrt(sumsq);
            if (residual_l2) *residual_l2 = res;
            if (res <= tol){ ++it; break; }
        }
    }
    clampToBoundaryRange_(); if (out_iters) *out_iters = it; return res <= tol;
}

bool LaplaceMetalSolver::solveRBGS(uint32_t max_iters, float tol, uint32_t *out_iters, float *residual_l2){
    if (levels_.empty()) { if (!buildLevels_()) return false; }
    auto &L=levels_[0];
    id<MTLCommandQueue> q=(__bridge id<MTLCommandQueue>)queue_;
    id<MTLComputePipelineState> pso=(__bridge id<MTLComputePipelineState>)psoRBGS_;
    applyBC_();
    float res=computeResidualL2_(); if (residual_l2) *residual_l2 = res; if (res<=tol){ if(out_iters) *out_iters=0; return true; }
    const uint32_t stride=50; uint32_t it=0;
    for (it=0; it<max_iters; ++it){
        for(uint32_t parity=0; parity<2; ++parity){
            id<MTLCommandBuffer> cb=[q commandBuffer]; id<MTLComputeCommandEncoder> enc=[cb computeCommandEncoder];
            [enc setComputePipelineState:pso];
            [enc setBuffer:(__bridge id<MTLBuffer>)L.uA offset:0 atIndex:0];
            [enc setBuffer:(__bridge id<MTLBuffer>)L.bcMask offset:0 atIndex:1];
            [enc setBuffer:(__bridge id<MTLBuffer>)L.bcVals offset:0 atIndex:2];
            [enc setBuffer:(__bridge id<MTLBuffer>)L.uniforms offset:0 atIndex:3];
            [enc setBytes:&parity length:sizeof(uint32_t) atIndex:4];
            [enc dispatchThreads:MTLSizeMake(L.nx,L.ny,1) threadsPerThreadgroup:chooseThreadgroup2D(pso,L.nx,L.ny)];
            [enc endEncoding]; [cb commit]; [cb waitUntilCompleted];
        }
        applyBC_();
        if ((it+1)%stride==0 || it+1==max_iters){ res=computeResidualL2_(); if (residual_l2) *residual_l2=res; if (res<=tol){ ++it; break; } }
    }
    clampToBoundaryRange_(); if (out_iters) *out_iters=it; return res<=tol;
}

float LaplaceMetalSolver::computeResidualL2WithRHS_(){
        if (levels_.empty()) return NAN;
        auto &L = levels_[0];
        id<MTLCommandQueue> q=(__bridge id<MTLCommandQueue>)queue_;
        id<MTLDevice> dev=(__bridge id<MTLDevice>)device_;
        id<MTLComputePipelineState> psoRes=(__bridge id<MTLComputePipelineState>)psoResidualRaw_;
        id<MTLComputePipelineState> psoDiff=(__bridge id<MTLComputePipelineState>)psoSumSquaresDiffPartial_;
        id<MTLComputePipelineState> psoRed=(__bridge id<MTLComputePipelineState>)psoReduceSum_;
        size_t nbytes = size_t(L.nx) * size_t(L.ny) * sizeof(float);
        // Copy f from L.rhs to a temporary shared buffer before overwriting rhs with Au
        id<MTLBuffer> fcopy=[dev newBufferWithLength:nbytes options:MTLResourceStorageModeShared];
        if (!fcopy) return NAN;
        { id<MTLCommandBuffer> cbb=[q commandBuffer]; id<MTLBlitCommandEncoder> bl=[cbb blitCommandEncoder];
            [bl copyFromBuffer:(__bridge id<MTLBuffer>)L.rhs sourceOffset:0 toBuffer:fcopy destinationOffset:0 size:nbytes];
            [bl endEncoding]; [cbb commit]; [cbb waitUntilCompleted]; }
        // Compute Au into L.rhs
        { id<MTLCommandBuffer> cb=[q commandBuffer]; id<MTLComputeCommandEncoder> enc=[cb computeCommandEncoder];
            [enc setComputePipelineState:psoRes];
            [enc setBuffer:(__bridge id<MTLBuffer>)L.uA offset:0 atIndex:0];
            [enc setBuffer:(__bridge id<MTLBuffer>)L.bcMask offset:0 atIndex:1];
            [enc setBuffer:(__bridge id<MTLBuffer>)L.rhs offset:0 atIndex:2];
            [enc setBuffer:(__bridge id<MTLBuffer>)L.uniforms offset:0 atIndex:3];
            [enc dispatchThreads:MTLSizeMake(L.nx,L.ny,1) threadsPerThreadgroup:chooseThreadgroup2D(psoRes,L.nx,L.ny)];
            [enc endEncoding]; [cb commit]; [cb waitUntilCompleted]; }
        if (!psoDiff || !psoRed) {
                // Fallback to CPU diff if helper pipelines unavailable
                id<MTLBuffer> staging=[dev newBufferWithLength:nbytes options:MTLResourceStorageModeShared];
                { id<MTLCommandBuffer> cbb=[q commandBuffer]; id<MTLBlitCommandEncoder> bl=[cbb blitCommandEncoder];
                    [bl copyFromBuffer:(__bridge id<MTLBuffer>)L.rhs sourceOffset:0 toBuffer:staging destinationOffset:0 size:nbytes];
                    [bl endEncoding]; [cbb commit]; [cbb waitUntilCompleted]; }
                float* Au=(float*)staging.contents; float* f=(float*)fcopy.contents; size_t n=nbytes/sizeof(float); double sum=0.0;
                for (size_t i=0;i<n;++i) { double r = (double)f[i] - (double)Au[i]; sum += r*r; }
                return (float)std::sqrt(sum);
        }
        const uint32_t tileX = 16, tileY = 16;
        uint32_t groupsX = (L.nx + tileX - 1) / tileX;
        uint32_t groupsY = (L.ny + tileY - 1) / tileY;
        if (!ensurePartialBuffer0_(groupsX, groupsY)) return NAN;
        uint32_t M = groupsX * groupsY;
        id<MTLBuffer> partial = (__bridge id<MTLBuffer>)bufPartial0_;
        { id<MTLCommandBuffer> pcb=[q commandBuffer]; id<MTLComputeCommandEncoder> penc=[pcb computeCommandEncoder];
            [penc setComputePipelineState:psoDiff];
            [penc setBuffer:fcopy offset:0 atIndex:0];           // a = f
            [penc setBuffer:(__bridge id<MTLBuffer>)L.rhs offset:0 atIndex:1]; // b = Au
            [penc setBuffer:(__bridge id<MTLBuffer>)L.uniforms offset:0 atIndex:2];
            [penc setBuffer:partial offset:0 atIndex:3];
            [penc dispatchThreadgroups:MTLSizeMake(groupsX, groupsY, 1) threadsPerThreadgroup:MTLSizeMake(tileX, tileY, 1)];
            [penc endEncoding]; [pcb commit]; [pcb waitUntilCompleted]; }
        id<MTLBuffer> out=[dev newBufferWithLength:sizeof(float) options:MTLResourceStorageModeShared];
        { id<MTLCommandBuffer> rcb=[q commandBuffer]; id<MTLComputeCommandEncoder> renc=[rcb computeCommandEncoder];
            [renc setComputePipelineState:psoRed];
            [renc setBuffer:partial offset:0 atIndex:0];
            [renc setBuffer:out offset:0 atIndex:1];
            [renc setBytes:&M length:sizeof(uint32_t) atIndex:2];
            NSUInteger tcount = std::min<NSUInteger>(256, psoRed.maxTotalThreadsPerThreadgroup);
            [renc dispatchThreadgroups:MTLSizeMake(1,1,1) threadsPerThreadgroup:MTLSizeMake(tcount,1,1)];
            [renc endEncoding]; [rcb commit]; [rcb waitUntilCompleted]; }
        float sumsq = *(float*)out.contents; return std::sqrt(sumsq);
}

bool LaplaceMetalSolver::solveRBGSWithRHS(uint32_t max_iters, float tol, uint32_t *out_iters, float *residual_l2){
    if (levels_.empty()) { if (!buildLevels_()) return false; }
    auto &L=levels_[0];
    id<MTLCommandQueue> q=(__bridge id<MTLCommandQueue>)queue_;
    id<MTLComputePipelineState> pso=(__bridge id<MTLComputePipelineState>)psoRBGSRHS_;
    applyBC_();
    float res=computeResidualL2WithRHS_(); if (residual_l2) *residual_l2=res; if (res<=tol){ if(out_iters) *out_iters=0; return true; }
    const uint32_t stride=50; uint32_t it=0;
    for (it=0; it<max_iters; ++it){
        for(uint32_t parity=0; parity<2; ++parity){
            id<MTLCommandBuffer> cb=[q commandBuffer]; id<MTLComputeCommandEncoder> enc=[cb computeCommandEncoder];
            [enc setComputePipelineState:pso];
            [enc setBuffer:(__bridge id<MTLBuffer>)L.uA offset:0 atIndex:0];
            [enc setBuffer:(__bridge id<MTLBuffer>)L.bcMask offset:0 atIndex:1];
            [enc setBuffer:(__bridge id<MTLBuffer>)L.bcVals offset:0 atIndex:2];
            [enc setBuffer:(__bridge id<MTLBuffer>)L.uniforms offset:0 atIndex:3];
            [enc setBuffer:(__bridge id<MTLBuffer>)L.rhs offset:0 atIndex:4];
            [enc setBytes:&parity length:sizeof(uint32_t) atIndex:5];
            [enc dispatchThreads:MTLSizeMake(L.nx,L.ny,1) threadsPerThreadgroup:chooseThreadgroup2D(pso,L.nx,L.ny)];
            [enc endEncoding]; [cb commit]; [cb waitUntilCompleted];
        }
        applyBC_();
        if ((it+1)%stride==0 || it+1==max_iters){ res=computeResidualL2WithRHS_(); if (residual_l2) *residual_l2=res; if (res<=tol){ ++it; break; } }
    }
    clampToBoundaryRange_(); if (out_iters) *out_iters=it; return res<=tol;
}

// solveMultigrid moved to LaplaceMetalSolver_multigrid.mm
