#include <cuda_runtime.h>
#include <device_launch_parameters.h>

// Uniform parameters structure for 3D
struct Uniforms3D {
    unsigned int nx;
    unsigned int ny;
    unsigned int nz;
    float inv_dx2;
    float inv_dy2;
    float inv_dz2;
    float inv_coef; // 1.0 / (2/dx^2 + 2/dy^2 + 2/dz^2)
    float omega;    // damping factor in (0,1]
};

// Inline device function for 3D indexing
__device__ __forceinline__ unsigned int idx3d(unsigned int i, unsigned int j, unsigned int k,
                                               unsigned int nx, unsigned int ny) {
    return (k * ny + j) * nx + i;
}

// Apply Dirichlet boundary conditions (3D)
__global__ void apply_dirichlet_3d(
    float* u,
    const unsigned char* bcMask,
    const float* bcVals,
    Uniforms3D uni
) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int j = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int k = blockIdx.z * blockDim.z + threadIdx.z;

    if (i >= uni.nx || j >= uni.ny || k >= uni.nz) return;

    unsigned int idx = idx3d(i, j, k, uni.nx, uni.ny);
    if (bcMask[idx]) {
        u[idx] = bcVals[idx];
    }
}

// Jacobi iteration step (3D)
__global__ void jacobi_step_3d(
    const float* u_old,
    float* u_new,
    const unsigned char* bcMask,
    const float* bcVals,
    Uniforms3D uni
) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int j = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int k = blockIdx.z * blockDim.z + threadIdx.z;

    if (i >= uni.nx || j >= uni.ny || k >= uni.nz) return;

    unsigned int idx = idx3d(i, j, k, uni.nx, uni.ny);

    if (bcMask[idx]) {
        u_new[idx] = bcVals[idx];
        return;
    }

    if (i == 0 || j == 0 || k == 0 || i + 1 == uni.nx || j + 1 == uni.ny || k + 1 == uni.nz) {
        u_new[idx] = u_old[idx];
        return;
    }

    unsigned int stride_z = uni.nx * uni.ny;
    unsigned int L = idx - 1;
    unsigned int R = idx + 1;
    unsigned int B = idx - uni.nx;
    unsigned int T = idx + uni.nx;
    unsigned int D = idx - stride_z;
    unsigned int U = idx + stride_z;

    float num = uni.inv_dx2 * (u_old[L] + u_old[R])
              + uni.inv_dy2 * (u_old[B] + u_old[T])
              + uni.inv_dz2 * (u_old[D] + u_old[U]);
    float u_j = num * uni.inv_coef;
    u_new[idx] = (1.0f - uni.omega) * u_old[idx] + uni.omega * u_j;
}

// Jacobi iteration step with RHS (3D)
__global__ void jacobi_step_rhs_3d(
    const float* u_old,
    float* u_new,
    const unsigned char* bcMask,
    const float* bcVals,
    Uniforms3D uni,
    const float* rhs
) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int j = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int k = blockIdx.z * blockDim.z + threadIdx.z;

    if (i >= uni.nx || j >= uni.ny || k >= uni.nz) return;

    unsigned int idx = idx3d(i, j, k, uni.nx, uni.ny);

    if (bcMask[idx]) {
        u_new[idx] = bcVals[idx];
        return;
    }

    if (i == 0 || j == 0 || k == 0 || i + 1 == uni.nx || j + 1 == uni.ny || k + 1 == uni.nz) {
        u_new[idx] = u_old[idx];
        return;
    }

    unsigned int stride_z = uni.nx * uni.ny;
    unsigned int L = idx - 1;
    unsigned int R = idx + 1;
    unsigned int B = idx - uni.nx;
    unsigned int T = idx + uni.nx;
    unsigned int D = idx - stride_z;
    unsigned int U = idx + stride_z;

    float num = uni.inv_dx2 * (u_old[L] + u_old[R])
              + uni.inv_dy2 * (u_old[B] + u_old[T])
              + uni.inv_dz2 * (u_old[D] + u_old[U])
              + rhs[idx];
    float u_j = num * uni.inv_coef;
    u_new[idx] = (1.0f - uni.omega) * u_old[idx] + uni.omega * u_j;
}

// Red-Black Gauss-Seidel phase (3D, in-place)
__global__ void rbgs_phase_3d(
    float* u,
    const unsigned char* bcMask,
    const float* bcVals,
    Uniforms3D uni,
    unsigned int parity
) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int j = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int k = blockIdx.z * blockDim.z + threadIdx.z;

    if (i >= uni.nx || j >= uni.ny || k >= uni.nz) return;

    if (((i + j + k) & 1u) != (parity & 1u)) return;

    unsigned int idx = idx3d(i, j, k, uni.nx, uni.ny);

    if (bcMask[idx]) {
        u[idx] = bcVals[idx];
        return;
    }

    if (i == 0 || j == 0 || k == 0 || i + 1 == uni.nx || j + 1 == uni.ny || k + 1 == uni.nz) return;

    unsigned int stride_z = uni.nx * uni.ny;
    unsigned int L = idx - 1;
    unsigned int R = idx + 1;
    unsigned int B = idx - uni.nx;
    unsigned int T = idx + uni.nx;
    unsigned int D = idx - stride_z;
    unsigned int U = idx + stride_z;

    float num = uni.inv_dx2 * (u[L] + u[R])
              + uni.inv_dy2 * (u[B] + u[T])
              + uni.inv_dz2 * (u[D] + u[U]);
    float u_gs = num * uni.inv_coef;
    u[idx] = (1.0f - uni.omega) * u[idx] + uni.omega * u_gs;
}

// Red-Black Gauss-Seidel phase with RHS (3D, in-place)
__global__ void rbgs_phase_rhs_3d(
    float* u,
    const unsigned char* bcMask,
    const float* bcVals,
    Uniforms3D uni,
    const float* rhs,
    unsigned int parity
) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int j = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int k = blockIdx.z * blockDim.z + threadIdx.z;

    if (i >= uni.nx || j >= uni.ny || k >= uni.nz) return;

    if (((i + j + k) & 1u) != (parity & 1u)) return;

    unsigned int idx = idx3d(i, j, k, uni.nx, uni.ny);

    if (bcMask[idx]) {
        u[idx] = bcVals[idx];
        return;
    }

    if (i == 0 || j == 0 || k == 0 || i + 1 == uni.nx || j + 1 == uni.ny || k + 1 == uni.nz) return;

    unsigned int stride_z = uni.nx * uni.ny;
    unsigned int L = idx - 1;
    unsigned int R = idx + 1;
    unsigned int B = idx - uni.nx;
    unsigned int T = idx + uni.nx;
    unsigned int D = idx - stride_z;
    unsigned int U = idx + stride_z;

    float num = uni.inv_dx2 * (u[L] + u[R])
              + uni.inv_dy2 * (u[B] + u[T])
              + uni.inv_dz2 * (u[D] + u[U])
              + rhs[idx];
    float u_gs = num * uni.inv_coef;
    u[idx] = (1.0f - uni.omega) * u[idx] + uni.omega * u_gs;
}

// Compute raw residual r = (u_xx + u_yy + u_zz)
__global__ void compute_residual_raw_3d(
    const float* u,
    const unsigned char* bcMask,
    float* r,
    Uniforms3D uni
) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int j = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int k = blockIdx.z * blockDim.z + threadIdx.z;

    if (i >= uni.nx || j >= uni.ny || k >= uni.nz) return;

    unsigned int idx = idx3d(i, j, k, uni.nx, uni.ny);

    if (bcMask[idx] || i == 0 || j == 0 || k == 0 || i + 1 == uni.nx || j + 1 == uni.ny || k + 1 == uni.nz) {
        r[idx] = 0.0f;
        return;
    }

    unsigned int stride_z = uni.nx * uni.ny;
    unsigned int L = idx - 1;
    unsigned int R = idx + 1;
    unsigned int B = idx - uni.nx;
    unsigned int T = idx + uni.nx;
    unsigned int D = idx - stride_z;
    unsigned int U = idx + stride_z;

    float uxx = (u[L] - 2.0f * u[idx] + u[R]) * uni.inv_dx2;
    float uyy = (u[B] - 2.0f * u[idx] + u[T]) * uni.inv_dy2;
    float uzz = (u[D] - 2.0f * u[idx] + u[U]) * uni.inv_dz2;
    r[idx] = -(uxx + uyy + uzz);
}

// One Jacobi-like smoothing step on residual field (3D)
__global__ void residual_smooth_step_3d(
    const float* r_old,
    float* r_new,
    const unsigned char* bcMask,
    Uniforms3D uni
) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int j = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int k = blockIdx.z * blockDim.z + threadIdx.z;

    if (i >= uni.nx || j >= uni.ny || k >= uni.nz) return;

    unsigned int idx = idx3d(i, j, k, uni.nx, uni.ny);
    if (bcMask[idx] || i == 0 || j == 0 || k == 0 || i + 1 == uni.nx || j + 1 == uni.ny || k + 1 == uni.nz) {
        r_new[idx] = 0.0f;
        return;
    }

    unsigned int stride_z = uni.nx * uni.ny;
    unsigned int L = idx - 1;
    unsigned int R = idx + 1;
    unsigned int B = idx - uni.nx;
    unsigned int T = idx + uni.nx;
    unsigned int D = idx - stride_z;
    unsigned int U = idx + stride_z;

    float num = uni.inv_dx2 * (r_old[L] + r_old[R])
              + uni.inv_dy2 * (r_old[B] + r_old[T])
              + uni.inv_dz2 * (r_old[D] + r_old[U]);
    float rj = num * uni.inv_coef;
    r_new[idx] = (1.0f - uni.omega) * r_old[idx] + uni.omega * rj;
}

// Set a 3D buffer to zero
__global__ void set_zero_float_3d(
    float* buf,
    Uniforms3D uni
) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int j = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int k = blockIdx.z * blockDim.z + threadIdx.z;

    if (i >= uni.nx || j >= uni.ny || k >= uni.nz) return;
    unsigned int idx = idx3d(i, j, k, uni.nx, uni.ny);
    buf[idx] = 0.0f;
}

// Clamp solution to [min,max] on non-BC nodes (3D)
__global__ void clamp_to_bounds_3d(
    float* u,
    const unsigned char* bcMask,
    Uniforms3D uni,
    float2 bounds
) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int j = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int k = blockIdx.z * blockDim.z + threadIdx.z;

    if (i >= uni.nx || j >= uni.ny || k >= uni.nz) return;
    unsigned int idx = idx3d(i, j, k, uni.nx, uni.ny);
    if (bcMask[idx]) return;
    float v = u[idx];
    if (v < bounds.x) v = bounds.x;
    if (v > bounds.y) v = bounds.y;
    u[idx] = v;
}
