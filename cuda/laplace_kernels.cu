#include <cuda_runtime.h>
#include <device_launch_parameters.h>

// Uniform parameters structure
struct Uniforms {
    unsigned int nx;
    unsigned int ny;
    float inv_dx2;
    float inv_dy2;
    float inv_coef; // 1.0 / (2/dx^2 + 2/dy^2)
    float omega;    // damping factor in (0,1]
};

struct GridDims {
    unsigned int nx_f, ny_f, nx_c, ny_c;
};

// Inline device function for 2D indexing
__device__ __forceinline__ unsigned int idx2d(unsigned int i, unsigned int j, unsigned int nx) {
    return j * nx + i;
}

// Apply Dirichlet boundary conditions
__global__ void apply_dirichlet(
    float* u,
    const unsigned char* bcMask,
    const float* bcVals,
    Uniforms uni
) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int j = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (i >= uni.nx || j >= uni.ny) return;
    
    unsigned int idx = idx2d(i, j, uni.nx);
    if (bcMask[idx]) {
        u[idx] = bcVals[idx];
    }
}

// Jacobi iteration step
__global__ void jacobi_step(
    const float* u_old,
    float* u_new,
    const unsigned char* bcMask,
    const float* bcVals,
    Uniforms uni
) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int j = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (i >= uni.nx || j >= uni.ny) return;
    
    unsigned int idx = idx2d(i, j, uni.nx);
    
    // Handle boundary conditions
    if (bcMask[idx]) {
        u_new[idx] = bcVals[idx];
        return;
    }
    
    // Handle domain edges
    if (i == 0 || j == 0 || i + 1 == uni.nx || j + 1 == uni.ny) {
        u_new[idx] = u_old[idx];
        return;
    }
    
    // Interior points: 5-point stencil
    unsigned int L = idx - 1;
    unsigned int R = idx + 1;
    unsigned int B = idx - uni.nx;
    unsigned int T = idx + uni.nx;
    
    float num = uni.inv_dx2 * (u_old[L] + u_old[R]) + uni.inv_dy2 * (u_old[B] + u_old[T]);
    float u_j = num * uni.inv_coef;
    
    // Damped update
    u_new[idx] = (1.0f - uni.omega) * u_old[idx] + uni.omega * u_j;
}

// Jacobi iteration step with RHS
__global__ void jacobi_step_rhs(
    const float* u_old,
    float* u_new,
    const unsigned char* bcMask,
    const float* bcVals,
    Uniforms uni,
    const float* rhs
) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int j = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (i >= uni.nx || j >= uni.ny) return;
    
    unsigned int idx = idx2d(i, j, uni.nx);
    
    if (bcMask[idx]) {
        u_new[idx] = bcVals[idx];
        return;
    }
    
    if (i == 0 || j == 0 || i + 1 == uni.nx || j + 1 == uni.ny) {
        u_new[idx] = u_old[idx];
        return;
    }
    
    unsigned int L = idx - 1, R = idx + 1;
    unsigned int B = idx - uni.nx, T = idx + uni.nx;
    
    float num = uni.inv_dx2 * (u_old[L] + u_old[R]) + uni.inv_dy2 * (u_old[B] + u_old[T]) + rhs[idx];
    float u_j = num * uni.inv_coef;
    u_new[idx] = (1.0f - uni.omega) * u_old[idx] + uni.omega * u_j;
}

// Red-Black Gauss-Seidel phase (in-place)
__global__ void rbgs_phase(
    float* u,
    const unsigned char* bcMask,
    const float* bcVals,
    Uniforms uni,
    unsigned int parity
) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int j = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (i >= uni.nx || j >= uni.ny) return;
    
    // Check parity
    if (((i + j) & 1u) != (parity & 1u)) return;
    
    unsigned int idx = idx2d(i, j, uni.nx);
    
    if (bcMask[idx]) {
        u[idx] = bcVals[idx];
        return;
    }
    
    if (i == 0 || j == 0 || i + 1 == uni.nx || j + 1 == uni.ny) return;
    
    unsigned int L = idx - 1, R = idx + 1;
    unsigned int B = idx - uni.nx, T = idx + uni.nx;
    
    float num = uni.inv_dx2 * (u[L] + u[R]) + uni.inv_dy2 * (u[B] + u[T]);
    float u_gs = num * uni.inv_coef;
    u[idx] = (1.0f - uni.omega) * u[idx] + uni.omega * u_gs;
}

// Red-Black Gauss-Seidel phase with RHS (in-place)
__global__ void rbgs_phase_rhs(
    float* u,
    const unsigned char* bcMask,
    const float* bcVals,
    Uniforms uni,
    const float* rhs,
    unsigned int parity
) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int j = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (i >= uni.nx || j >= uni.ny) return;
    
    if (((i + j) & 1u) != (parity & 1u)) return;
    
    unsigned int idx = idx2d(i, j, uni.nx);
    
    if (bcMask[idx]) {
        u[idx] = bcVals[idx];
        return;
    }
    
    if (i == 0 || j == 0 || i + 1 == uni.nx || j + 1 == uni.ny) return;
    
    unsigned int L = idx - 1, R = idx + 1;
    unsigned int B = idx - uni.nx, T = idx + uni.nx;
    
    float num = uni.inv_dx2 * (u[L] + u[R]) + uni.inv_dy2 * (u[B] + u[T]) + rhs[idx];
    float u_gs = num * uni.inv_coef;
    u[idx] = (1.0f - uni.omega) * u[idx] + uni.omega * u_gs;
}

// Compute raw residual r = (u_xx + u_yy)
__global__ void compute_residual_raw(
    const float* u,
    const unsigned char* bcMask,
    float* r,
    Uniforms uni
) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int j = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (i >= uni.nx || j >= uni.ny) return;
    
    unsigned int idx = idx2d(i, j, uni.nx);
    
    if (bcMask[idx] || i == 0 || j == 0 || i + 1 == uni.nx || j + 1 == uni.ny) {
        r[idx] = 0.0f;
        return;
    }
    
    unsigned int L = idx - 1, R = idx + 1;
    unsigned int B = idx - uni.nx, T = idx + uni.nx;
    
    float uxx = (u[L] - 2.0f * u[idx] + u[R]) * uni.inv_dx2;
    float uyy = (u[B] - 2.0f * u[idx] + u[T]) * uni.inv_dy2;
    r[idx] = uxx + uyy;
}

// One Jacobi-like smoothing step on the residual field
__global__ void residual_smooth_step(
    const float* r_old,
    float* r_new,
    const unsigned char* bcMask,
    Uniforms uni
) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int j = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (i >= uni.nx || j >= uni.ny) return;
    
    unsigned int idx = idx2d(i, j, uni.nx);
    
    // On BCs and edges, keep residual zero (homogeneous)
    if (bcMask[idx] || i == 0 || j == 0 || i + 1 == uni.nx || j + 1 == uni.ny) {
        r_new[idx] = 0.0f;
        return;
    }
    
    unsigned int L = idx - 1, R = idx + 1;
    unsigned int B = idx - uni.nx, T = idx + uni.nx;
    
    float num = uni.inv_dx2 * (r_old[L] + r_old[R]) + uni.inv_dy2 * (r_old[B] + r_old[T]);
    float rj = num * uni.inv_coef;
    r_new[idx] = (1.0f - uni.omega) * r_old[idx] + uni.omega * rj;
}

// Full-weighting restriction from fine residual to coarse rhs
__global__ void restrict_full_weighting(
    const float* rf,
    float* rc,
    GridDims gd
) {
    unsigned int I = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int J = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (I >= gd.nx_c || J >= gd.ny_c) return;
    
    unsigned int idxc = idx2d(I, J, gd.nx_c);
    
    if (I == 0 || J == 0 || I + 1 == gd.nx_c || J + 1 == gd.ny_c) {
        rc[idxc] = 0.0f;
        return;
    }
    
    unsigned int i = 2u * I;
    unsigned int j = 2u * J;
    unsigned int idx = idx2d(i, j, gd.nx_f);
    
    unsigned int L = idx - 1, R = idx + 1;
    unsigned int B = idx - gd.nx_f, T = idx + gd.nx_f;
    unsigned int BL = B - 1, BR = B + 1, TL = T - 1, TR = T + 1;
    
    float sum = rf[TL] + 2.0f * rf[T] + rf[TR]
              + 2.0f * rf[L] + 4.0f * rf[idx] + 2.0f * rf[R]
              + rf[BL] + 2.0f * rf[B] + rf[BR];
    rc[idxc] = sum * (1.0f / 16.0f);
}

// Bilinear prolongation-add: uf += P ec
__global__ void prolong_bilinear_add(
    const float* ec,
    float* uf,
    GridDims gd
) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int j = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (i >= gd.nx_f || j >= gd.ny_f) return;
    
    unsigned int idxf = idx2d(i, j, gd.nx_f);
    
    if (i == 0 || j == 0 || i + 1 == gd.nx_f || j + 1 == gd.ny_f) return;
    
    unsigned int I = i / 2;
    unsigned int J = j / 2;
    unsigned int I1 = min(I + 1, gd.nx_c - 1);
    unsigned int J1 = min(J + 1, gd.ny_c - 1);
    
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

// Set buffer to zero
__global__ void set_zero_float(
    float* buf,
    Uniforms uni
) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int j = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (i >= uni.nx || j >= uni.ny) return;
    
    buf[idx2d(i, j, uni.nx)] = 0.0f;
}

// Clamp solution to [min,max] on non-BC nodes
__global__ void clamp_to_bounds(
    float* u,
    const unsigned char* bcMask,
    Uniforms uni,
    float2 bounds
) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int j = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (i >= uni.nx || j >= uni.ny) return;
    
    unsigned int idx = idx2d(i, j, uni.nx);
    
    if (bcMask[idx]) return;
    
    u[idx] = fminf(fmaxf(u[idx], bounds.x), bounds.y);
}

// -----------------------------
// Fused tiled Jacobi + residual with GPU-side partial reduction
// Tile size (must match host dispatch)
#define TILE_X 16
#define TILE_Y 16

__global__ void jacobi_fused_residual_tiled(
    const float* u_old,
    float* u_new,
    const unsigned char* bcMask,
    const float* bcVals,
    Uniforms uni,
    float* partial_sums
) {
    __shared__ float tileOld[(TILE_X + 2) * (TILE_Y + 2)];
    __shared__ float tileNew[(TILE_X + 2) * (TILE_Y + 2)];
    __shared__ float redScratch[TILE_X * TILE_Y];
    
    // Thread and block indices
    unsigned int tx = threadIdx.x;
    unsigned int ty = threadIdx.y;
    unsigned int lx = tx + 1;
    unsigned int ly = ty + 1;
    
    unsigned int i = blockIdx.x * blockDim.x + tx;
    unsigned int j = blockIdx.y * blockDim.y + ty;
    
    bool inBounds = (i < uni.nx) && (j < uni.ny);
    unsigned int idx = inBounds ? idx2d(i, j, uni.nx) : 0u;
    
    // Load center
    float u_c = inBounds ? u_old[idx] : 0.0f;
    tileOld[ly * (TILE_X + 2) + lx] = u_c;
    
    // Load halos
    if (tx == 0) {
        unsigned int iL = (i > 0) ? i - 1 : i;
        unsigned int idxL = idx2d(iL, j, uni.nx);
        tileOld[ly * (TILE_X + 2) + 0] = (inBounds && i > 0) ? u_old[idxL] : u_c;
    }
    if (tx + 1 == TILE_X) {
        unsigned int iR = (i + 1 < uni.nx) ? i + 1 : i;
        unsigned int idxR = idx2d(iR, j, uni.nx);
        tileOld[ly * (TILE_X + 2) + lx + 1] = (inBounds && (i + 1 < uni.nx)) ? u_old[idxR] : u_c;
    }
    if (ty == 0) {
        unsigned int jB = (j > 0) ? j - 1 : j;
        unsigned int idxB = idx2d(i, jB, uni.nx);
        tileOld[0 * (TILE_X + 2) + lx] = (inBounds && j > 0) ? u_old[idxB] : u_c;
    }
    if (ty + 1 == TILE_Y) {
        unsigned int jT = (j + 1 < uni.ny) ? j + 1 : j;
        unsigned int idxT = idx2d(i, jT, uni.nx);
        tileOld[(ly + 1) * (TILE_X + 2) + lx] = (inBounds && (j + 1 < uni.ny)) ? u_old[idxT] : u_c;
    }
    
    __syncthreads();
    
    // Compute Jacobi update
    float u_next = u_c;
    bool isBC = false;
    
    if (inBounds) {
        isBC = bcMask[idx] != 0;
        bool onEdge = (i == 0 || j == 0 || i + 1 == uni.nx || j + 1 == uni.ny);
        
        if (isBC) {
            u_next = bcVals[idx];
        } else if (onEdge) {
            u_next = u_c;
        } else {
            float L = tileOld[ly * (TILE_X + 2) + lx - 1];
            float R = tileOld[ly * (TILE_X + 2) + lx + 1];
            float B = tileOld[(ly - 1) * (TILE_X + 2) + lx];
            float T = tileOld[(ly + 1) * (TILE_X + 2) + lx];
            float num = uni.inv_dx2 * (L + R) + uni.inv_dy2 * (B + T);
            float u_j = num * uni.inv_coef;
            u_next = (1.0f - uni.omega) * u_c + uni.omega * u_j;
        }
    }
    
    tileNew[ly * (TILE_X + 2) + lx] = u_next;
    if (inBounds) u_new[idx] = u_next;
    
    __syncthreads();
    
    // Copy halos for residual
    if (tx == 0)                  tileNew[ly * (TILE_X + 2) + 0] = tileNew[ly * (TILE_X + 2) + 1];
    if (tx + 1 == TILE_X)         tileNew[ly * (TILE_X + 2) + lx + 1] = tileNew[ly * (TILE_X + 2) + lx];
    if (ty == 0)                  tileNew[0 * (TILE_X + 2) + lx] = tileNew[1 * (TILE_X + 2) + lx];
    if (ty + 1 == TILE_Y)         tileNew[(ly + 1) * (TILE_X + 2) + lx] = tileNew[ly * (TILE_X + 2) + lx];
    
    __syncthreads();
    
    // Compute residual
    float r = 0.0f;
    if (inBounds) {
        bool onEdge = (i == 0 || j == 0 || i + 1 == uni.nx || j + 1 == uni.ny);
        if (!isBC && !onEdge) {
            float L = tileNew[ly * (TILE_X + 2) + lx - 1];
            float R = tileNew[ly * (TILE_X + 2) + lx + 1];
            float B = tileNew[(ly - 1) * (TILE_X + 2) + lx];
            float T = tileNew[(ly + 1) * (TILE_X + 2) + lx];
            float ucc = tileNew[ly * (TILE_X + 2) + lx];
            float uxx = (L - 2.0f * ucc + R) * uni.inv_dx2;
            float uyy = (B - 2.0f * ucc + T) * uni.inv_dy2;
            r = uxx + uyy;
        }
    }
    
    // Reduction within block
    unsigned int lin = ty * TILE_X + tx;
    redScratch[lin] = r * r;
    
    __syncthreads();
    
    // Parallel reduction
    for (unsigned int stride = (TILE_X * TILE_Y) / 2; stride > 0; stride >>= 1) {
        if (lin < stride) {
            redScratch[lin] += redScratch[lin + stride];
        }
        __syncthreads();
    }
    
    if (lin == 0) {
        unsigned int groups_x = (uni.nx + TILE_X - 1) / TILE_X;
        unsigned int tg_index = blockIdx.y * groups_x + blockIdx.x;
        partial_sums[tg_index] = redScratch[0];
    }
}

// Reduce an array of M partial sums to one value
__global__ void reduce_sum(
    const float* in,
    float* out,
    unsigned int M
) {
    __shared__ float scratch[256];
    
    unsigned int tid = threadIdx.x;
    unsigned int tcount = blockDim.x;
    
    float s = 0.0f;
    for (unsigned int i = tid; i < M; i += tcount) {
        s += in[i];
    }
    scratch[tid] = s;
    
    __syncthreads();
    
    for (unsigned int stride = tcount >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) {
            scratch[tid] += scratch[tid + stride];
        }
        __syncthreads();
    }
    
    if (tid == 0) out[0] = scratch[0];
}

// Compute per-tile partial sums of squares over a 2D array
__global__ void sum_squares_partial_tiled(
    const float* in,
    Uniforms uni,
    float* partial_sums
) {
    __shared__ float scratch[TILE_X * TILE_Y];
    
    unsigned int tx = threadIdx.x;
    unsigned int ty = threadIdx.y;
    unsigned int i = blockIdx.x * blockDim.x + tx;
    unsigned int j = blockIdx.y * blockDim.y + ty;
    
    bool inBounds = (i < uni.nx) && (j < uni.ny);
    unsigned int idx = inBounds ? idx2d(i, j, uni.nx) : 0u;
    float v = inBounds ? in[idx] : 0.0f;
    
    scratch[ty * TILE_X + tx] = v * v;
    
    __syncthreads();
    
    // Reduce within tile
    unsigned int lin = ty * TILE_X + tx;
    for (unsigned int stride = (TILE_X * TILE_Y) / 2; stride > 0; stride >>= 1) {
        if (lin < stride) {
            scratch[lin] += scratch[lin + stride];
        }
        __syncthreads();
    }
    
    if (lin == 0) {
        unsigned int groups_x = (uni.nx + TILE_X - 1) / TILE_X;
        unsigned int tg_index = blockIdx.y * groups_x + blockIdx.x;
        partial_sums[tg_index] = scratch[0];
    }
}

// Compute per-tile partial sums of squares of the difference between two 2D arrays
__global__ void sum_squares_diff_partial_tiled(
    const float* a,
    const float* b,
    Uniforms uni,
    float* partial_sums
) {
    __shared__ float scratch[TILE_X * TILE_Y];
    
    unsigned int tx = threadIdx.x;
    unsigned int ty = threadIdx.y;
    unsigned int i = blockIdx.x * blockDim.x + tx;
    unsigned int j = blockIdx.y * blockDim.y + ty;
    
    bool inBounds = (i < uni.nx) && (j < uni.ny);
    unsigned int idx = inBounds ? idx2d(i, j, uni.nx) : 0u;
    float va = inBounds ? a[idx] : 0.0f;
    float vb = inBounds ? b[idx] : 0.0f;
    float d = va - vb;
    
    scratch[ty * TILE_X + tx] = d * d;
    
    __syncthreads();
    
    // Reduce within tile
    unsigned int lin = ty * TILE_X + tx;
    for (unsigned int stride = (TILE_X * TILE_Y) / 2; stride > 0; stride >>= 1) {
        if (lin < stride) {
            scratch[lin] += scratch[lin + stride];
        }
        __syncthreads();
    }
    
    if (lin == 0) {
        unsigned int groups_x = (uni.nx + TILE_X - 1) / TILE_X;
        unsigned int tg_index = blockIdx.y * groups_x + blockIdx.x;
        partial_sums[tg_index] = scratch[0];
    }
}

// Successive Over-Relaxation (SOR) without RHS (in-place)
__global__ void sor_step(
    float* u,
    const unsigned char* bcMask,
    const float* bcVals,
    Uniforms uni
) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int j = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (i >= uni.nx || j >= uni.ny) return;
    
    unsigned int idx = idx2d(i, j, uni.nx);
    
    if (bcMask[idx]) {
        u[idx] = bcVals[idx];
        return;
    }
    
    if (i == 0 || j == 0 || i + 1 == uni.nx || j + 1 == uni.ny) return;
    
    unsigned int L = idx - 1, R = idx + 1;
    unsigned int B = idx - uni.nx, T = idx + uni.nx;
    
    float uxx = (u[L] - 2.0f * u[idx] + u[R]) * uni.inv_dx2;
    float uyy = (u[B] - 2.0f * u[idx] + u[T]) * uni.inv_dy2;
    float residual = uxx + uyy;
    
    float correction = uni.omega * residual * uni.inv_coef * 0.25f;
    u[idx] -= correction;
}

// Successive Over-Relaxation (SOR) with RHS (in-place)
__global__ void sor_step_rhs(
    float* u,
    const unsigned char* bcMask,
    const float* bcVals,
    Uniforms uni,
    const float* rhs
) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int j = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (i >= uni.nx || j >= uni.ny) return;
    
    unsigned int idx = idx2d(i, j, uni.nx);
    
    if (bcMask[idx]) {
        u[idx] = bcVals[idx];
        return;
    }
    
    if (i == 0 || j == 0 || i + 1 == uni.nx || j + 1 == uni.ny) return;
    
    unsigned int L = idx - 1, R = idx + 1;
    unsigned int B = idx - uni.nx, T = idx + uni.nx;
    
    float uxx = (u[L] - 2.0f * u[idx] + u[R]) * uni.inv_dx2;
    float uyy = (u[B] - 2.0f * u[idx] + u[T]) * uni.inv_dy2;
    float residual = uxx + uyy - rhs[idx];
    
    float correction = uni.omega * residual * uni.inv_coef * 0.25f;
    u[idx] += correction;
}
