#include <metal_stdlib>
using namespace metal;

struct Uniforms {
    uint nx;
    uint ny;
    float inv_dx2;
    float inv_dy2;
    float inv_coef; // 1.0 / (2/dx^2 + 2/dy^2)
    float omega;    // damping factor in (0,1]
};

struct GridDims {
    uint nx_f, ny_f, nx_c, ny_c;
};

inline uint idx2d(uint i, uint j, uint nx) { return j * nx + i; }

kernel void apply_dirichlet(
    device float*       u        [[ buffer(0) ]],
    device const uchar* bcMask   [[ buffer(1) ]],
    device const float* bcVals   [[ buffer(2) ]],
    device const Uniforms& uni   [[ buffer(3) ]],
    uint2 tid                    [[ thread_position_in_grid ]]
) {
    if (tid.x >= uni.nx || tid.y >= uni.ny) return;
    uint idx = idx2d(tid.x, tid.y, uni.nx);
    if (bcMask[idx]) {
        u[idx] = bcVals[idx];
    }
}

kernel void jacobi_step(
    device const float* u_old    [[ buffer(0) ]],
    device float*       u_new    [[ buffer(1) ]],
    device const uchar* bcMask   [[ buffer(2) ]],
    device const float* bcVals   [[ buffer(3) ]],
    device const Uniforms& uni   [[ buffer(4) ]],
    uint2 tid                    [[ thread_position_in_grid ]]
) {
    if (tid.x >= uni.nx || tid.y >= uni.ny) return;
    uint i = tid.x, j = tid.y;
    uint idx = idx2d(i, j, uni.nx);

    if (bcMask[idx]) { u_new[idx] = bcVals[idx]; return; }
    if (i == 0 || j == 0 || i + 1 == uni.nx || j + 1 == uni.ny) { u_new[idx] = u_old[idx]; return; }

    uint L = idx - 1, R = idx + 1;
    uint B = idx - uni.nx, T = idx + uni.nx;
    float num = uni.inv_dx2 * (u_old[L] + u_old[R]) + uni.inv_dy2 * (u_old[B] + u_old[T]);
    float u_j = num * uni.inv_coef;
    // damped update
    u_new[idx] = mix(u_old[idx], u_j, uni.omega);
}

kernel void jacobi_step_rhs(
    device const float* u_old    [[ buffer(0) ]],
    device float*       u_new    [[ buffer(1) ]],
    device const uchar* bcMask   [[ buffer(2) ]],
    device const float* bcVals   [[ buffer(3) ]],
    device const Uniforms& uni   [[ buffer(4) ]],
    device const float* rhs      [[ buffer(5) ]],
    uint2 tid                    [[ thread_position_in_grid ]]
) {
    if (tid.x >= uni.nx || tid.y >= uni.ny) return;
    uint i = tid.x, j = tid.y;
    uint idx = idx2d(i, j, uni.nx);

    if (bcMask[idx]) { u_new[idx] = bcVals[idx]; return; }
    if (i == 0 || j == 0 || i + 1 == uni.nx || j + 1 == uni.ny) { u_new[idx] = u_old[idx]; return; }

    uint L = idx - 1, R = idx + 1;
    uint B = idx - uni.nx, T = idx + uni.nx;
    float num = uni.inv_dx2 * (u_old[L] + u_old[R]) + uni.inv_dy2 * (u_old[B] + u_old[T]) + rhs[idx];
    float u_j = num * uni.inv_coef;
    u_new[idx] = mix(u_old[idx], u_j, uni.omega);
}

// Red-Black Gauss-Seidel without RHS (in-place)
kernel void rbgs_phase(
    device float*       u        [[ buffer(0) ]],
    device const uchar* bcMask   [[ buffer(1) ]],
    device const float* bcVals   [[ buffer(2) ]],
    device const Uniforms& uni   [[ buffer(3) ]],
    constant uint&      parity   [[ buffer(4) ]],
    uint2 tid                    [[ thread_position_in_grid ]]
) {
    if (tid.x >= uni.nx || tid.y >= uni.ny) return;
    uint i = tid.x, j = tid.y;
    if (((i + j) & 1u) != (parity & 1u)) return;
    uint idx = idx2d(i, j, uni.nx);
    if (bcMask[idx]) { u[idx] = bcVals[idx]; return; }
    if (i == 0 || j == 0 || i + 1 == uni.nx || j + 1 == uni.ny) return;

    uint L = idx - 1, R = idx + 1;
    uint B = idx - uni.nx, T = idx + uni.nx;
    float num = uni.inv_dx2 * (u[L] + u[R]) + uni.inv_dy2 * (u[B] + u[T]);
    float u_gs = num * uni.inv_coef;
    u[idx] = mix(u[idx], u_gs, uni.omega);
}

// Red-Black Gauss-Seidel with RHS (in-place)
kernel void rbgs_phase_rhs(
    device float*       u        [[ buffer(0) ]],
    device const uchar* bcMask   [[ buffer(1) ]],
    device const float* bcVals   [[ buffer(2) ]],
    device const Uniforms& uni   [[ buffer(3) ]],
    device const float* rhs      [[ buffer(4) ]],
    constant uint&      parity   [[ buffer(5) ]],
    uint2 tid                    [[ thread_position_in_grid ]]
) {
    if (tid.x >= uni.nx || tid.y >= uni.ny) return;
    uint i = tid.x, j = tid.y;
    if (((i + j) & 1u) != (parity & 1u)) return;
    uint idx = idx2d(i, j, uni.nx);
    if (bcMask[idx]) { u[idx] = bcVals[idx]; return; }
    if (i == 0 || j == 0 || i + 1 == uni.nx || j + 1 == uni.ny) return;

    uint L = idx - 1, R = idx + 1;
    uint B = idx - uni.nx, T = idx + uni.nx;
    float num = uni.inv_dx2 * (u[L] + u[R]) + uni.inv_dy2 * (u[B] + u[T]) + rhs[idx];
    float u_gs = num * uni.inv_coef;
    u[idx] = mix(u[idx], u_gs, uni.omega);
}

// Compute raw residual r = (u_xx + u_yy) with Dirichlet boundaries -> 0
kernel void compute_residual_raw(
    device const float* u        [[ buffer(0) ]],
    device const uchar* bcMask   [[ buffer(1) ]],
    device float*       r        [[ buffer(2) ]],
    device const Uniforms& uni   [[ buffer(3) ]],
    uint2 tid                    [[ thread_position_in_grid ]]
) {
    if (tid.x >= uni.nx || tid.y >= uni.ny) return;
    uint i = tid.x, j = tid.y;
    uint idx = idx2d(i, j, uni.nx);
    if (bcMask[idx] || i == 0 || j == 0 || i + 1 == uni.nx || j + 1 == uni.ny) { r[idx] = 0.0f; return; }
    uint L = idx - 1, R = idx + 1;
    uint B = idx - uni.nx, T = idx + uni.nx;
    float uxx = (u[L] - 2.0f * u[idx] + u[R]) * uni.inv_dx2;
    float uyy = (u[B] - 2.0f * u[idx] + u[T]) * uni.inv_dy2;
    r[idx] = uxx + uyy; // note: raw Laplacian
}

// One Jacobi-like smoothing step on the residual field (homogeneous Dirichlet)
// r_new = (inv_dx2*(r[L]+r[R]) + inv_dy2*(r[B]+r[T])) * inv_coef, damped by omega
kernel void residual_smooth_step(
    device const float* r_old   [[ buffer(0) ]],
    device float*       r_new   [[ buffer(1) ]],
    device const uchar* bcMask  [[ buffer(2) ]],
    device const Uniforms& uni  [[ buffer(3) ]],
    uint2 tid                   [[ thread_position_in_grid ]]
){
    if (tid.x >= uni.nx || tid.y >= uni.ny) return;
    uint i = tid.x, j = tid.y;
    uint idx = idx2d(i, j, uni.nx);
    // On BCs and edges, keep residual zero (homogeneous)
    if (bcMask[idx] || i == 0 || j == 0 || i + 1 == uni.nx || j + 1 == uni.ny) {
        r_new[idx] = 0.0f; return;
    }
    uint L = idx - 1, R = idx + 1;
    uint B = idx - uni.nx, T = idx + uni.nx;
    float num = uni.inv_dx2 * (r_old[L] + r_old[R]) + uni.inv_dy2 * (r_old[B] + r_old[T]);
    float rj = num * uni.inv_coef;
    r_new[idx] = mix(r_old[idx], rj, uni.omega);
}

// Full-weighting restriction from fine residual to coarse rhs
kernel void restrict_full_weighting(
    device const float* rf       [[ buffer(0) ]],
    device float*       rc       [[ buffer(1) ]],
    constant GridDims&  gd       [[ buffer(2) ]],
    uint2 tid                    [[ thread_position_in_grid ]]
) {
    if (tid.x >= gd.nx_c || tid.y >= gd.ny_c) return;
    uint I = tid.x, J = tid.y;
    uint idxc = idx2d(I, J, gd.nx_c);
    if (I == 0 || J == 0 || I + 1 == gd.nx_c || J + 1 == gd.ny_c) { rc[idxc] = 0.0f; return; }
    uint i = 2u * I;
    uint j = 2u * J;
    uint idx = idx2d(i, j, gd.nx_f);
    uint L = idx - 1, R = idx + 1;
    uint B = idx - gd.nx_f, T = idx + gd.nx_f;
    uint BL = B - 1, BR = B + 1, TL = T - 1, TR = T + 1;
    float sum = rf[TL] + 2.0f * rf[T] + rf[TR]
              + 2.0f * rf[L] + 4.0f * rf[idx] + 2.0f * rf[R]
              + rf[BL] + 2.0f * rf[B] + rf[BR];
    rc[idxc] = sum * (1.0f / 16.0f);
}

// Bilinear prolongation-add: uf += P ec
kernel void prolong_bilinear_add(
    device const float* ec       [[ buffer(0) ]],
    device float*       uf       [[ buffer(1) ]],
    constant GridDims&  gd       [[ buffer(2) ]],
    uint2 tid                    [[ thread_position_in_grid ]]
) {
    if (tid.x >= gd.nx_f || tid.y >= gd.ny_f) return;
    uint i = tid.x, j = tid.y;
    uint idxf = idx2d(i, j, gd.nx_f);
    if (i == 0 || j == 0 || i + 1 == gd.nx_f || j + 1 == gd.ny_f) return;
    uint I = i / 2, J = j / 2;
    uint I1 = min(I + 1, gd.nx_c - 1);
    uint J1 = min(J + 1, gd.ny_c - 1);
    float fi = (float)(i & 1u);
    float fj = (float)(j & 1u);
    float wx0 = (fi == 0.0f) ? 1.0f : 0.5f;
    float wx1 = (fi == 0.0f) ? 0.0f : 0.5f;
    float wy0 = (fj == 0.0f) ? 1.0f : 0.5f;
    float wy1 = (fj == 0.0f) ? 0.0f : 0.5f;
    float e00 = ec[idx2d(I,  J,  gd.nx_c)];
    float e10 = ec[idx2d(I1, J,  gd.nx_c)];
    float e01 = ec[idx2d(I,  J1, gd.nx_c)];
    float e11 = ec[idx2d(I1, J1, gd.nx_c)];
    float add = wx0 * wy0 * e00 + wx1 * wy0 * e10 + wx0 * wy1 * e01 + wx1 * wy1 * e11;
    uf[idxf] += add;
}

kernel void set_zero_float(
    device float*             buf [[ buffer(0) ]],
    device const Uniforms&    uni [[ buffer(1) ]],
    uint2 tid                      [[ thread_position_in_grid ]]
) {
    if (tid.x >= uni.nx || tid.y >= uni.ny) return;
    buf[idx2d(tid.x, tid.y, uni.nx)] = 0.0f;
}

// Clamp solution to [min,max] on non-BC nodes
kernel void clamp_to_bounds(
    device float*             u        [[ buffer(0) ]],
    device const uchar*       bcMask   [[ buffer(1) ]],
    device const Uniforms&    uni      [[ buffer(2) ]],
    constant float2&          bounds   [[ buffer(3) ]],
    uint2 tid                          [[ thread_position_in_grid ]]
) {
    if (tid.x >= uni.nx || tid.y >= uni.ny) return;
    uint idx = idx2d(tid.x, tid.y, uni.nx);
    if (bcMask[idx]) return;
    u[idx] = clamp(u[idx], bounds.x, bounds.y);
}

// -----------------------------
// Fused tiled Jacobi + residual with GPU-side partial reduction
// Tile size (must match host dispatch)
constant ushort TILE_X = 16;
constant ushort TILE_Y = 16;

kernel void jacobi_fused_residual_tiled(
    device const float* u_old           [[ buffer(0) ]],
    device float*       u_new           [[ buffer(1) ]],
    device const uchar* bcMask          [[ buffer(2) ]],
    device const float* bcVals          [[ buffer(3) ]],
    device const Uniforms& uni          [[ buffer(4) ]],
    device float*       partial_sums    [[ buffer(5) ]],
    uint2 tid_grid                      [[ thread_position_in_grid ]],
    uint2 tid_group                     [[ thread_position_in_threadgroup ]],
    uint2 tgrp_pos                      [[ threadgroup_position_in_grid ]],
    uint2 tgrp_size                     [[ threads_per_threadgroup ]]
)
{
    // Guard threads outside domain
    bool inBounds = (tid_grid.x < uni.nx) && (tid_grid.y < uni.ny);

    // Threadgroup-shared tiles with 1-cell halo on each side
    threadgroup float tileOld[(TILE_X + 2) * (TILE_Y + 2)];
    threadgroup float tileNew[(TILE_X + 2) * (TILE_Y + 2)];
    threadgroup float redScratch[TILE_X * TILE_Y];

    auto tIndex = [&](ushort x, ushort y) -> ushort { return (ushort)((y) * (TILE_X + 2) + (x)); };
    auto sIndex = [&](ushort x, ushort y) -> ushort { return (ushort)((y) * TILE_X + (x)); };

    ushort tx = tid_group.x;
    ushort ty = tid_group.y;
    ushort lx = tx + 1;
    ushort ly = ty + 1;

    uint i = tid_grid.x;
    uint j = tid_grid.y;
    uint idx = inBounds ? idx2d(i, j, uni.nx) : 0u;

    // Load center
    float u_c = inBounds ? u_old[idx] : 0.0f;
    tileOld[tIndex(lx, ly)] = u_c;

    // Load halos along tile borders (conditionally)
    // Left/right neighbors
    if (tx == 0) {
        uint iL = (i > 0) ? i - 1 : i;
        uint idxL = idx2d(iL, j, uni.nx);
        tileOld[tIndex(0, ly)] = (inBounds && i > 0) ? u_old[idxL] : u_c;
    }
    if (tx + 1 == TILE_X) {
        uint iR = (i + 1 < uni.nx) ? i + 1 : i;
        uint idxR = idx2d(iR, j, uni.nx);
        tileOld[tIndex(lx + 1, ly)] = (inBounds && (i + 1 < uni.nx)) ? u_old[idxR] : u_c;
    }
    // Bottom/top neighbors
    if (ty == 0) {
        uint jB = (j > 0) ? j - 1 : j;
        uint idxB = idx2d(i, jB, uni.nx);
        tileOld[tIndex(lx, 0)] = (inBounds && j > 0) ? u_old[idxB] : u_c;
    }
    if (ty + 1 == TILE_Y) {
        uint jT = (j + 1 < uni.ny) ? j + 1 : j;
        uint idxT = idx2d(i, jT, uni.nx);
        tileOld[tIndex(lx, ly + 1)] = (inBounds && (j + 1 < uni.ny)) ? u_old[idxT] : u_c;
    }
    // Corners (only needed for residual later)
    if (tx == 0 && ty == 0) {
        uint iL = (i > 0) ? i - 1 : i;
        uint jB = (j > 0) ? j - 1 : j;
        tileOld[tIndex(0, 0)] = (inBounds && i > 0 && j > 0) ? u_old[idx2d(iL, jB, uni.nx)] : u_c;
    }
    if (tx + 1 == TILE_X && ty == 0) {
        uint iR = (i + 1 < uni.nx) ? i + 1 : i;
        uint jB = (j > 0) ? j - 1 : j;
        tileOld[tIndex(lx + 1, 0)] = (inBounds && (i + 1 < uni.nx) && j > 0) ? u_old[idx2d(iR, jB, uni.nx)] : u_c;
    }
    if (tx == 0 && ty + 1 == TILE_Y) {
        uint iL = (i > 0) ? i - 1 : i;
        uint jT = (j + 1 < uni.ny) ? j + 1 : j;
        tileOld[tIndex(0, ly + 1)] = (inBounds && i > 0 && (j + 1 < uni.ny)) ? u_old[idx2d(iL, jT, uni.nx)] : u_c;
    }
    if (tx + 1 == TILE_X && ty + 1 == TILE_Y) {
        uint iR = (i + 1 < uni.nx) ? i + 1 : i;
        uint jT = (j + 1 < uni.ny) ? j + 1 : j;
        tileOld[tIndex(lx + 1, ly + 1)] = (inBounds && (i + 1 < uni.nx) && (j + 1 < uni.ny)) ? u_old[idx2d(iR, jT, uni.nx)] : u_c;
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Compute Jacobi update with BC fusion
    float u_next = u_c;
    bool isBC = false;
    if (inBounds) {
        uint idxBC = idx;
        isBC = bcMask[idxBC] != 0;
        bool onEdge = (i == 0 || j == 0 || i + 1 == uni.nx || j + 1 == uni.ny);
        if (isBC) {
            u_next = bcVals[idxBC];
        } else if (onEdge) {
            u_next = u_c;
        } else {
            float L = tileOld[tIndex(lx - 1, ly)];
            float R = tileOld[tIndex(lx + 1, ly)];
            float B = tileOld[tIndex(lx, ly - 1)];
            float T = tileOld[tIndex(lx, ly + 1)];
            float num = uni.inv_dx2 * (L + R) + uni.inv_dy2 * (B + T);
            float u_j = num * uni.inv_coef;
            u_next = mix(u_c, u_j, uni.omega);
        }
    }

    tileNew[tIndex(lx, ly)] = u_next;
    if (inBounds) u_new[idx] = u_next;

    // Ensure all centers written before copying halos
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Halos for residual based on updated values
    if (tx == 0)                  tileNew[tIndex(0, ly)]      = tileNew[tIndex(1, ly)];
    if (tx + 1 == TILE_X)         tileNew[tIndex(lx + 1, ly)] = tileNew[tIndex(lx, ly)];
    if (ty == 0)                  tileNew[tIndex(lx, 0)]      = tileNew[tIndex(lx, 1)];
    if (ty + 1 == TILE_Y)         tileNew[tIndex(lx, ly + 1)] = tileNew[tIndex(lx, ly)];
    if (tx == 0 && ty == 0)       tileNew[tIndex(0, 0)]       = tileNew[tIndex(1, 1)];
    if (tx + 1 == TILE_X && ty==0)tileNew[tIndex(lx + 1, 0)]  = tileNew[tIndex(lx, 1)];
    if (tx == 0 && ty + 1==TILE_Y)tileNew[tIndex(0, ly + 1)]  = tileNew[tIndex(1, ly)];
    if (tx + 1==TILE_X && ty+1==TILE_Y) tileNew[tIndex(lx + 1, ly + 1)] = tileNew[tIndex(lx, ly)];

    // Ensure halos visible before residual computation
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Compute residual r = uxx + uyy on updated field (exclude BC and domain edges)
    float r = 0.0f;
    if (inBounds) {
        bool onEdge = (i == 0 || j == 0 || i + 1 == uni.nx || j + 1 == uni.ny);
        if (!isBC && !onEdge) {
            float L = tileNew[tIndex(lx - 1, ly)];
            float R = tileNew[tIndex(lx + 1, ly)];
            float B = tileNew[tIndex(lx, ly - 1)];
            float T = tileNew[tIndex(lx, ly + 1)];
            float ucc = tileNew[tIndex(lx, ly)];
            float uxx = (L - 2.0f * ucc + R) * uni.inv_dx2;
            float uyy = (B - 2.0f * ucc + T) * uni.inv_dy2;
            r = uxx + uyy; // Laplacian (f=0)
        }
    }

    // Reduction within threadgroup (sum of squares)
    uint lin = sIndex(tx, ty);
    redScratch[lin] = r * r;

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Parallel reduction in-place with bounds checks for non-power-of-two sizes
    uint n = TILE_X * TILE_Y;
    // Round up to next power-of-two for loop control (without reading out of range)
    uint p2 = 1; while (p2 < n) p2 <<= 1;
    for (uint stride = p2 >> 1; stride > 0; stride >>= 1) {
        if (lin < stride) {
            uint other = lin + stride;
            if (other < n) redScratch[lin] += redScratch[other];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (lin == 0) {
        // Compute global threadgroup index
        uint groups_x = (uni.nx + TILE_X - 1) / TILE_X;
        uint tg_index = tgrp_pos.y * groups_x + tgrp_pos.x;
        partial_sums[tg_index] = redScratch[0];
    }
}

// Reduce an array of M partial sums to one value. Launch with one threadgroup, e.g., 256 threads.
kernel void reduce_sum(
    device const float* in      [[ buffer(0) ]],
    device float*       out     [[ buffer(1) ]],
    constant uint&      M       [[ buffer(2) ]],
    uint tid                    [[ thread_index_in_threadgroup ]],
    uint tcount                 [[ threads_per_threadgroup ]]
)
{
    threadgroup float scratch[256];
    float s = 0.0f;
    for (uint i = tid; i < M; i += tcount) s += in[i];
    scratch[tid] = s;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = tcount >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) scratch[tid] += scratch[tid + stride];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid == 0) out[0] = scratch[0];
}

// Compute per-tile partial sums of squares over a 2D array
// Launch with threadgroups covering ceil(nx/TILE_X) x ceil(ny/TILE_Y)
// and threadsPerThreadgroup = (TILE_X, TILE_Y, 1)
kernel void sum_squares_partial_tiled(
    device const float* in           [[ buffer(0) ]],
    device const Uniforms& uni       [[ buffer(1) ]],
    device float*       partial_sums [[ buffer(2) ]],
    uint2 tid_grid                   [[ thread_position_in_grid ]],
    uint2 tid_group                  [[ thread_position_in_threadgroup ]],
    uint2 tgrp_pos                   [[ threadgroup_position_in_grid ]])
{
    threadgroup float scratch[TILE_X * TILE_Y];

    auto sIndex = [&](ushort x, ushort y) -> ushort { return (ushort)(y * TILE_X + x); };

    ushort tx = tid_group.x;
    ushort ty = tid_group.y;
    uint i = tid_grid.x;
    uint j = tid_grid.y;

    bool inBounds = (i < uni.nx) && (j < uni.ny);
    uint idx = inBounds ? idx2d(i, j, uni.nx) : 0u;
    float v = inBounds ? in[idx] : 0.0f;
    scratch[sIndex(tx, ty)] = v * v;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Reduce within tile to one value (thread 0)
    uint lin = sIndex(tx, ty);
    uint n = TILE_X * TILE_Y;
    uint p2 = 1; while (p2 < n) p2 <<= 1;
    for (uint stride = p2 >> 1; stride > 0; stride >>= 1) {
        if (lin < stride) {
            uint other = lin + stride;
            if (other < n) scratch[lin] += scratch[other];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (lin == 0) {
        uint groups_x = (uni.nx + TILE_X - 1) / TILE_X;
        uint tg_index = tgrp_pos.y * groups_x + tgrp_pos.x;
        partial_sums[tg_index] = scratch[0];
    }
}

// Compute per-tile partial sums of squares of the difference between two 2D arrays
// partial_sums[tg] = sum_{tile} (a - b)^2
// Launch with threadgroups covering ceil(nx/TILE_X) x ceil(ny/TILE_Y)
// and threadsPerThreadgroup = (TILE_X, TILE_Y, 1)
kernel void sum_squares_diff_partial_tiled(
    device const float* a            [[ buffer(0) ]],
    device const float* b            [[ buffer(1) ]],
    device const Uniforms& uni       [[ buffer(2) ]],
    device float*       partial_sums [[ buffer(3) ]],
    uint2 tid_grid                   [[ thread_position_in_grid ]],
    uint2 tid_group                  [[ thread_position_in_threadgroup ]],
    uint2 tgrp_pos                   [[ threadgroup_position_in_grid ]])
{
    threadgroup float scratch[TILE_X * TILE_Y];

    auto sIndex = [&](ushort x, ushort y) -> ushort { return (ushort)(y * TILE_X + x); };

    ushort tx = tid_group.x;
    ushort ty = tid_group.y;
    uint i = tid_grid.x;
    uint j = tid_grid.y;

    bool inBounds = (i < uni.nx) && (j < uni.ny);
    uint idx = inBounds ? idx2d(i, j, uni.nx) : 0u;
    float va = inBounds ? a[idx] : 0.0f;
    float vb = inBounds ? b[idx] : 0.0f;
    float d = va - vb;
    scratch[sIndex(tx, ty)] = d * d;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Reduce within tile to one value (thread 0)
    uint lin = sIndex(tx, ty);
    uint n = TILE_X * TILE_Y;
    uint p2 = 1; while (p2 < n) p2 <<= 1;
    for (uint stride = p2 >> 1; stride > 0; stride >>= 1) {
        if (lin < stride) {
            uint other = lin + stride;
            if (other < n) scratch[lin] += scratch[other];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (lin == 0) {
        uint groups_x = (uni.nx + TILE_X - 1) / TILE_X;
        uint tg_index = tgrp_pos.y * groups_x + tgrp_pos.x;
        partial_sums[tg_index] = scratch[0];
    }
}

// Successive Over-Relaxation (SOR) without RHS (in-place)
// SOR is more aggressive than damped Jacobi: u_new = u_old + ω * (Au - rhs) / A_ii
kernel void sor_step(
    device float*       u        [[ buffer(0) ]],
    device const uchar* bcMask   [[ buffer(1) ]],
    device const float* bcVals   [[ buffer(2) ]],
    device const Uniforms& uni   [[ buffer(3) ]],
    uint2 tid                    [[ thread_position_in_grid ]]
) {
    if (tid.x >= uni.nx || tid.y >= uni.ny) return;
    uint i = tid.x, j = tid.y;
    uint idx = idx2d(i, j, uni.nx);
    if (bcMask[idx]) { u[idx] = bcVals[idx]; return; }
    if (i == 0 || j == 0 || i + 1 == uni.nx || j + 1 == uni.ny) return;

    uint L = idx - 1, R = idx + 1;
    uint B = idx - uni.nx, T = idx + uni.nx;

    // Compute residual: r = Au - rhs = (u_xx + u_yy) - 0 = u_xx + u_yy
    float uxx = (u[L] - 2.0f * u[idx] + u[R]) * uni.inv_dx2;
    float uyy = (u[B] - 2.0f * u[idx] + u[T]) * uni.inv_dy2;
    float residual = uxx + uyy;

    // SOR update: u_new = u_old + ω * residual / A_ii
    // For 5-point Laplace: A_ii = -4*(inv_dx2 + inv_dy2) = -4*inv_coef
    // So: u_new = u_old + ω * residual / (-4*inv_coef) = u_old - ω * residual / (4*inv_coef)
    float correction = uni.omega * residual * uni.inv_coef * 0.25f; // 1/(4*inv_coef) = inv_coef/4
    u[idx] -= correction; // Note: minus because residual = Au, and we want u_new = u_old - ω*(Au)/A_ii
}

// Successive Over-Relaxation (SOR) with RHS (in-place)
kernel void sor_step_rhs(
    device float*       u        [[ buffer(0) ]],
    device const uchar* bcMask   [[ buffer(1) ]],
    device const float* bcVals   [[ buffer(2) ]],
    device const Uniforms& uni   [[ buffer(3) ]],
    device const float* rhs      [[ buffer(4) ]],
    uint2 tid                    [[ thread_position_in_grid ]]
) {
    if (tid.x >= uni.nx || tid.y >= uni.ny) return;
    uint i = tid.x, j = tid.y;
    uint idx = idx2d(i, j, uni.nx);
    if (bcMask[idx]) { u[idx] = bcVals[idx]; return; }
    if (i == 0 || j == 0 || i + 1 == uni.nx || j + 1 == uni.ny) return;

    uint L = idx - 1, R = idx + 1;
    uint B = idx - uni.nx, T = idx + uni.nx;

    // Compute residual: r = Au - rhs
    float uxx = (u[L] - 2.0f * u[idx] + u[R]) * uni.inv_dx2;
    float uyy = (u[B] - 2.0f * u[idx] + u[T]) * uni.inv_dy2;
    float residual = uxx + uyy - rhs[idx];

    // SOR update: u_new = u_old - ω * residual / A_ii
    // For 5-point Laplace: A_ii = -4*(inv_dx2 + inv_dy2) = -4*inv_coef
    // So: u_new = u_old - ω * residual / (-4*inv_coef) = u_old + ω * residual / (4*inv_coef)
    float correction = uni.omega * residual * uni.inv_coef * 0.25f; // 1/(4*inv_coef) = inv_coef/4
    u[idx] += correction;
}
