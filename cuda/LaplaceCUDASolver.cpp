#include "LaplaceCUDASolver.hpp"
#include <cuda_runtime.h>
#include <cmath>
#include <stdexcept>
#include <iostream>

// Include kernel declarations
struct Uniforms;
struct GridDims;

// Kernel function declarations
extern "C"
{
    void launch_apply_dirichlet(float *u, const unsigned char *bcMask, const float *bcVals,
                                Uniforms uni, dim3 grid, dim3 block);
    void launch_jacobi_step(const float *u_old, float *u_new, const unsigned char *bcMask,
                            const float *bcVals, Uniforms uni, dim3 grid, dim3 block);
    void launch_rbgs_phase(float *u, const unsigned char *bcMask, const float *bcVals,
                           Uniforms uni, unsigned int parity, dim3 grid, dim3 block);
    void launch_compute_residual_raw(const float *u, const unsigned char *bcMask, float *r,
                                     Uniforms uni, dim3 grid, dim3 block);
    void launch_restrict_full_weighting(const float *rf, float *rc, GridDims gd,
                                        dim3 grid, dim3 block);
    void launch_prolong_bilinear_add(const float *ec, float *uf, GridDims gd,
                                     dim3 grid, dim3 block);
    void launch_set_zero_float(float *buf, Uniforms uni, dim3 grid, dim3 block);
}

// Error checking macro
#define CUDA_CHECK(call)                                                                     \
    do                                                                                       \
    {                                                                                        \
        cudaError_t err = call;                                                              \
        if (err != cudaSuccess)                                                              \
        {                                                                                    \
            throw std::runtime_error(std::string("CUDA error: ") + cudaGetErrorString(err)); \
        }                                                                                    \
    } while (0)

LaplaceCUDASolver::LaplaceCUDASolver(size_t nx, size_t ny, float dx, float dy)
    : nx_(nx), ny_(ny), dx_(dx), dy_(dy),
      d_u_(nullptr), d_u_temp_(nullptr), d_bcMask_(nullptr),
      d_bcVals_(nullptr), d_residual_(nullptr)
{
    allocateDeviceMemory();
}

LaplaceCUDASolver::~LaplaceCUDASolver()
{
    freeDeviceMemory();
}

void LaplaceCUDASolver::allocateDeviceMemory()
{
    size_t size = nx_ * ny_ * sizeof(float);
    size_t bcSize = nx_ * ny_ * sizeof(unsigned char);

    CUDA_CHECK(cudaMalloc(&d_u_, size));
    CUDA_CHECK(cudaMalloc(&d_u_temp_, size));
    CUDA_CHECK(cudaMalloc(&d_bcMask_, bcSize));
    CUDA_CHECK(cudaMalloc(&d_bcVals_, size));
    CUDA_CHECK(cudaMalloc(&d_residual_, size));

    // Initialize to zero
    CUDA_CHECK(cudaMemset(d_u_, 0, size));
    CUDA_CHECK(cudaMemset(d_u_temp_, 0, size));
    CUDA_CHECK(cudaMemset(d_bcMask_, 0, bcSize));
    CUDA_CHECK(cudaMemset(d_bcVals_, 0, size));
    CUDA_CHECK(cudaMemset(d_residual_, 0, size));
}

void LaplaceCUDASolver::freeDeviceMemory()
{
    if (d_u_)
        cudaFree(d_u_);
    if (d_u_temp_)
        cudaFree(d_u_temp_);
    if (d_bcMask_)
        cudaFree(d_bcMask_);
    if (d_bcVals_)
        cudaFree(d_bcVals_);
    if (d_residual_)
        cudaFree(d_residual_);

    // Free multigrid levels
    for (auto &level : gridLevels_)
    {
        if (level.d_u)
            cudaFree(level.d_u);
        if (level.d_rhs)
            cudaFree(level.d_rhs);
        if (level.d_residual)
            cudaFree(level.d_residual);
    }
}

void LaplaceCUDASolver::setBoundaryConditions(const std::vector<unsigned char> &bcMask,
                                              const std::vector<float> &bcVals)
{
    if (bcMask.size() != nx_ * ny_ || bcVals.size() != nx_ * ny_)
    {
        throw std::invalid_argument("BC arrays size mismatch");
    }

    CUDA_CHECK(cudaMemcpy(d_bcMask_, bcMask.data(), nx_ * ny_ * sizeof(unsigned char),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_bcVals_, bcVals.data(), nx_ * ny_ * sizeof(float),
                          cudaMemcpyHostToDevice));
}

void LaplaceCUDASolver::setInitialSolution(const std::vector<float> &u)
{
    if (u.size() != nx_ * ny_)
    {
        throw std::invalid_argument("Solution array size mismatch");
    }

    CUDA_CHECK(cudaMemcpy(d_u_, u.data(), nx_ * ny_ * sizeof(float),
                          cudaMemcpyHostToDevice));
}

void LaplaceCUDASolver::getSolution(std::vector<float> &u) const
{
    u.resize(nx_ * ny_);
    CUDA_CHECK(cudaMemcpy(u.data(), d_u_, nx_ * ny_ * sizeof(float),
                          cudaMemcpyDeviceToHost));
}

void LaplaceCUDASolver::applyBoundaryConditions()
{
    // Set up grid and block dimensions
    dim3 block(16, 16);
    dim3 grid((nx_ + block.x - 1) / block.x, (ny_ + block.y - 1) / block.y);

    Uniforms uni;
    uni.nx = nx_;
    uni.ny = ny_;
    uni.inv_dx2 = 1.0f / (dx_ * dx_);
    uni.inv_dy2 = 1.0f / (dy_ * dy_);
    uni.inv_coef = 1.0f / (2.0f * uni.inv_dx2 + 2.0f * uni.inv_dy2);
    uni.omega = 1.0f;

    launch_apply_dirichlet(d_u_, d_bcMask_, d_bcVals_, uni, grid, block);
    CUDA_CHECK(cudaGetLastError());
}

int LaplaceCUDASolver::solveJacobi(int maxIters, float tol, float omega)
{
    residualHistory_.clear();

    Uniforms uni;
    uni.nx = nx_;
    uni.ny = ny_;
    uni.inv_dx2 = 1.0f / (dx_ * dx_);
    uni.inv_dy2 = 1.0f / (dy_ * dy_);
    uni.inv_coef = 1.0f / (2.0f * uni.inv_dx2 + 2.0f * uni.inv_dy2);
    uni.omega = omega;

    dim3 block(16, 16);
    dim3 grid((nx_ + block.x - 1) / block.x, (ny_ + block.y - 1) / block.y);

    int iter;
    for (iter = 0; iter < maxIters; ++iter)
    {
        // Jacobi step
        launch_jacobi_step(d_u_, d_u_temp_, d_bcMask_, d_bcVals_, uni, grid, block);
        CUDA_CHECK(cudaGetLastError());

        // Swap buffers
        std::swap(d_u_, d_u_temp_);

        // Check convergence every 10 iterations
        if (iter % 10 == 0)
        {
            float residual = computeResidualNorm();
            residualHistory_.push_back(residual);

            if (residual < tol)
            {
                break;
            }
        }
    }

    return iter;
}

int LaplaceCUDASolver::solveRBGS(int maxIters, float tol, float omega)
{
    residualHistory_.clear();

    Uniforms uni;
    uni.nx = nx_;
    uni.ny = ny_;
    uni.inv_dx2 = 1.0f / (dx_ * dx_);
    uni.inv_dy2 = 1.0f / (dy_ * dy_);
    uni.inv_coef = 1.0f / (2.0f * uni.inv_dx2 + 2.0f * uni.inv_dy2);
    uni.omega = omega;

    dim3 block(16, 16);
    dim3 grid((nx_ + block.x - 1) / block.x, (ny_ + block.y - 1) / block.y);

    int iter;
    for (iter = 0; iter < maxIters; ++iter)
    {
        // Red phase
        launch_rbgs_phase(d_u_, d_bcMask_, d_bcVals_, uni, 0, grid, block);
        CUDA_CHECK(cudaGetLastError());

        // Black phase
        launch_rbgs_phase(d_u_, d_bcMask_, d_bcVals_, uni, 1, grid, block);
        CUDA_CHECK(cudaGetLastError());

        // Check convergence every 10 iterations
        if (iter % 10 == 0)
        {
            float residual = computeResidualNorm();
            residualHistory_.push_back(residual);

            if (residual < tol)
            {
                break;
            }
        }
    }

    return iter;
}

float LaplaceCUDASolver::computeResidualNorm() const
{
    Uniforms uni;
    uni.nx = nx_;
    uni.ny = ny_;
    uni.inv_dx2 = 1.0f / (dx_ * dx_);
    uni.inv_dy2 = 1.0f / (dy_ * dy_);
    uni.inv_coef = 1.0f / (2.0f * uni.inv_dx2 + 2.0f * uni.inv_dy2);
    uni.omega = 1.0f;

    dim3 block(16, 16);
    dim3 grid((nx_ + block.x - 1) / block.x, (ny_ + block.y - 1) / block.y);

    // Compute residual
    launch_compute_residual_raw(d_u_, d_bcMask_, d_residual_, uni, grid, block);
    CUDA_CHECK(cudaGetLastError());

    // Compute L2 norm
    return computeL2Norm(d_residual_, nx_ * ny_);
}

float LaplaceCUDASolver::computeL2Norm(const float *d_array, size_t size) const
{
    // Simple implementation: copy to host and compute (can be optimized with reduction kernel)
    std::vector<float> h_array(size);
    CUDA_CHECK(cudaMemcpy(h_array.data(), d_array, size * sizeof(float),
                          cudaMemcpyDeviceToHost));

    float sum = 0.0f;
    for (float val : h_array)
    {
        sum += val * val;
    }

    return std::sqrt(sum);
}

int LaplaceCUDASolver::solveFMG(int v1, int v2, int maxLevels, float omega)
{
    // Build multigrid hierarchy
    gridLevels_.clear();

    size_t curr_nx = nx_;
    size_t curr_ny = ny_;

    for (int level = 0; level < maxLevels; ++level)
    {
        if (curr_nx <= 8 || curr_ny <= 8)
            break;

        GridLevel glevel;
        glevel.nx = curr_nx;
        glevel.ny = curr_ny;

        size_t size = curr_nx * curr_ny * sizeof(float);
        CUDA_CHECK(cudaMalloc(&glevel.d_u, size));
        CUDA_CHECK(cudaMalloc(&glevel.d_rhs, size));
        CUDA_CHECK(cudaMalloc(&glevel.d_residual, size));
        CUDA_CHECK(cudaMemset(glevel.d_u, 0, size));
        CUDA_CHECK(cudaMemset(glevel.d_rhs, 0, size));

        gridLevels_.push_back(glevel);

        curr_nx = (curr_nx + 1) / 2;
        curr_ny = (curr_ny + 1) / 2;
    }

    // Perform V-cycles
    residualHistory_.clear();
    int numCycles = 10; // Could be parameter

    for (int cycle = 0; cycle < numCycles; ++cycle)
    {
        vCycle(0, v1, v2, omega);

        float residual = computeResidualNorm();
        residualHistory_.push_back(residual);

        if (residual < 1e-6f)
            break;
    }

    return numCycles;
}

void LaplaceCUDASolver::vCycle(int level, int v1, int v2, float omega)
{
    if (level >= gridLevels_.size())
        return;

    GridLevel &glevel = gridLevels_[level];

    // Pre-smoothing
    smooth(glevel.d_u, glevel.d_rhs, glevel.nx, glevel.ny, v1, omega);

    // Compute residual and restrict
    if (level + 1 < gridLevels_.size())
    {
        Uniforms uni;
        uni.nx = glevel.nx;
        uni.ny = glevel.ny;
        uni.inv_dx2 = 1.0f / (dx_ * dx_);
        uni.inv_dy2 = 1.0f / (dy_ * dy_);
        uni.inv_coef = 1.0f / (2.0f * uni.inv_dx2 + 2.0f * uni.inv_dy2);
        uni.omega = omega;

        dim3 block(16, 16);
        dim3 grid((glevel.nx + block.x - 1) / block.x, (glevel.ny + block.y - 1) / block.y);

        launch_compute_residual_raw(glevel.d_u, d_bcMask_, glevel.d_residual, uni, grid, block);

        // Restrict to coarse grid
        GridLevel &coarse = gridLevels_[level + 1];
        restrict(glevel.d_residual, coarse.d_rhs, glevel.nx, glevel.ny);

        // Recursive call
        vCycle(level + 1, v1, v2, omega);

        // Prolongate and correct
        prolongate(coarse.d_u, glevel.d_u, coarse.nx, coarse.ny, glevel.nx, glevel.ny);
    }

    // Post-smoothing
    smooth(glevel.d_u, glevel.d_rhs, glevel.nx, glevel.ny, v2, omega);
}

void LaplaceCUDASolver::smooth(float *d_u, const float *d_rhs, size_t nx, size_t ny,
                               int iters, float omega)
{
    Uniforms uni;
    uni.nx = nx;
    uni.ny = ny;
    uni.inv_dx2 = 1.0f / (dx_ * dx_);
    uni.inv_dy2 = 1.0f / (dy_ * dy_);
    uni.inv_coef = 1.0f / (2.0f * uni.inv_dx2 + 2.0f * uni.inv_dy2);
    uni.omega = omega;

    dim3 block(16, 16);
    dim3 grid((nx + block.x - 1) / block.x, (ny + block.y - 1) / block.y);

    for (int i = 0; i < iters; ++i)
    {
        launch_rbgs_phase(d_u, d_bcMask_, d_bcVals_, uni, 0, grid, block);
        launch_rbgs_phase(d_u, d_bcMask_, d_bcVals_, uni, 1, grid, block);
    }
}

void LaplaceCUDASolver::restrict(const float *d_fine, float *d_coarse,
                                 size_t nx_f, size_t ny_f)
{
    size_t nx_c = (nx_f + 1) / 2;
    size_t ny_c = (ny_f + 1) / 2;

    GridDims gd;
    gd.nx_f = nx_f;
    gd.ny_f = ny_f;
    gd.nx_c = nx_c;
    gd.ny_c = ny_c;

    dim3 block(16, 16);
    dim3 grid((nx_c + block.x - 1) / block.x, (ny_c + block.y - 1) / block.y);

    launch_restrict_full_weighting(d_fine, d_coarse, gd, grid, block);
}

void LaplaceCUDASolver::prolongate(const float *d_coarse, float *d_fine,
                                   size_t nx_c, size_t ny_c, size_t nx_f, size_t ny_f)
{
    GridDims gd;
    gd.nx_f = nx_f;
    gd.ny_f = ny_f;
    gd.nx_c = nx_c;
    gd.ny_c = ny_c;

    dim3 block(16, 16);
    dim3 grid((nx_f + block.x - 1) / block.x, (ny_f + block.y - 1) / block.y);

    launch_prolong_bilinear_add(d_coarse, d_fine, gd, grid, block);
}
