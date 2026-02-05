#pragma once
#include <OpenCL/opencl.h>
#include <cstdint>
#include <string>
#include <vector>

// Simple OpenCL Multigrid Diffusion/Poisson solver (2D/3D) with Helmholtz support.
// Can solve steady Laplace/Poisson:  -Lap(u) = f
// and implicit unsteady diffusion via backward Euler: (alpha) u - Lap(u) = b, where alpha = 1/dt.

class OpenCLMGSolver
{
public:
    enum class Dim
    {
        D2 = 2,
        D3 = 3
    };
    struct Desc
    {
        Dim dim{Dim::D2};
        uint32_t nx{129}, ny{129}, nz{1};
        float dx{1.0f}, dy{1.0f}, dz{1.0f};
        uint32_t maxLevels{0}; // 0=auto
    };

    explicit OpenCLMGSolver(const Desc &d, const std::string &kernelPath);
    ~OpenCLMGSolver();

    // Problem setup
    void setInitial(const std::vector<float> &u0);
    void setRHS(const std::vector<float> &f);
    void setDirichletBC(const std::vector<uint8_t> &mask, const std::vector<float> &vals);

    // Steady solve: -Lap(u)=f
    bool solveSteady(uint32_t maxVCycles, float tol, uint32_t nu1 = 4, uint32_t nu2 = 4, uint32_t coarseIters = 200);

    // Helmholtz solve: (alpha) u - Lap(u) = b, used for implicit time steps (alpha=1/dt)
    bool solveHelmholtz(float alpha, uint32_t maxVCycles, float tol, uint32_t nu1 = 4, uint32_t nu2 = 4, uint32_t coarseIters = 200);

    // Unsteady diffusion step: backward Euler
    // (I - dt*Lap) u^{n+1} = u^n + dt*f^n
    bool stepBackwardEuler(float dt, uint32_t vcyclesPerStep, float tol, uint32_t nu1 = 3, uint32_t nu2 = 3, uint32_t coarseIters = 120);

    // Download solution from finest level
    std::vector<float> downloadSolution() const;

    // Utilities
    void setVerbose(bool v) { verbose_ = v; }

private:
    struct Level
    {
        uint32_t nx{}, ny{}, nz{};
        size_t N{};                                   // nx*ny*nz
        cl_mem uA{}, uB{}, rhs{}, bcMask{}, bcVals{}; // device buffers
    };

    Desc desc_;
    bool verbose_{false};

    // OpenCL objects
    cl_platform_id platform_{};
    cl_device_id device_{};
    cl_context context_{};
    cl_command_queue queue_{};
    cl_program program_{};

    // Kernels
    cl_kernel k_apply_dirichlet_{};
    cl_kernel k_residual_{};      // for -Lap(u) case
    cl_kernel k_residual_helm_{}; // for alpha*u - Lap(u)
    cl_kernel k_jacobi_{};        // Jacobi relax for -Lap(u)
    cl_kernel k_jacobi_helm_{};   // Jacobi relax for Helmholtz
    cl_kernel k_restrict_fw_{};
    cl_kernel k_prolong_add_{};

    std::vector<Level> levels_;
    uint32_t Lf_{}; // finest level index 0

    // Helpers
    void initOpenCL_(const std::string &kernelPath);
    void buildLevels_();
    void destroyLevels_();
    void zeroBuffer_(cl_mem buf, size_t bytes);
    void swap_(cl_mem &a, cl_mem &b)
    {
        cl_mem t = a;
        a = b;
        b = t;
    }

    // Launch helpers
    void applyDirichlet_(Level &lev);
    void jacobiRelax_(Level &lev, bool helmholtz, float alpha, uint32_t iters);
    void computeResidual_(Level &lev, bool helmholtz, float alpha);
    void restrictFullWeighting_(Level &fine, Level &coarse);
    void prolongAdd_(Level &coarse, Level &fine);
    float residualL2_(Level &lev, bool helmholtz, float alpha);
};
