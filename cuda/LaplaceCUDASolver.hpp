#ifndef LAPLACE_CUDA_SOLVER_HPP
#define LAPLACE_CUDA_SOLVER_HPP

#include <vector>
#include <string>
#include <cstddef>

/**
 * @brief CUDA-based Laplace equation solver with multigrid support
 *
 * This class provides GPU-accelerated solvers for the 2D Laplace equation
 * using CUDA. It supports various iterative methods including Jacobi,
 * Red-Black Gauss-Seidel (RBGS), and Full Multigrid (FMG).
 */
class LaplaceCUDASolver
{
public:
    /**
     * @brief Constructor
     * @param nx Number of grid points in x-direction
     * @param ny Number of grid points in y-direction
     * @param dx Grid spacing in x-direction
     * @param dy Grid spacing in y-direction
     */
    LaplaceCUDASolver(size_t nx, size_t ny, float dx, float dy);

    /**
     * @brief Destructor - cleans up CUDA resources
     */
    ~LaplaceCUDASolver();

    // Delete copy constructor and assignment operator
    LaplaceCUDASolver(const LaplaceCUDASolver &) = delete;
    LaplaceCUDASolver &operator=(const LaplaceCUDASolver &) = delete;

    /**
     * @brief Set boundary conditions
     * @param bcMask Array indicating boundary condition locations (1=BC, 0=interior)
     * @param bcVals Array of boundary condition values
     */
    void setBoundaryConditions(const std::vector<unsigned char> &bcMask,
                               const std::vector<float> &bcVals);

    /**
     * @brief Set initial solution
     * @param u Initial solution values
     */
    void setInitialSolution(const std::vector<float> &u);

    /**
     * @brief Get current solution
     * @param u Output array for solution values
     */
    void getSolution(std::vector<float> &u) const;

    /**
     * @brief Solve using Jacobi iteration
     * @param maxIters Maximum number of iterations
     * @param tol Convergence tolerance
     * @param omega Damping factor (default 1.0)
     * @return Number of iterations performed
     */
    int solveJacobi(int maxIters, float tol, float omega = 1.0f);

    /**
     * @brief Solve using Red-Black Gauss-Seidel iteration
     * @param maxIters Maximum number of iterations
     * @param tol Convergence tolerance
     * @param omega Damping factor (default 1.0)
     * @return Number of iterations performed
     */
    int solveRBGS(int maxIters, float tol, float omega = 1.0f);

    /**
     * @brief Solve using Full Multigrid method
     * @param v1 Number of pre-smoothing iterations
     * @param v2 Number of post-smoothing iterations
     * @param maxLevels Maximum number of multigrid levels
     * @param omega Damping factor (default 1.0)
     * @return Number of V-cycles performed
     */
    int solveFMG(int v1, int v2, int maxLevels, float omega = 1.0f);

    /**
     * @brief Compute L2 norm of residual
     * @return L2 norm of residual
     */
    float computeResidualNorm() const;

    /**
     * @brief Get grid dimensions
     * @param nx_out Output for nx
     * @param ny_out Output for ny
     */
    void getGridDimensions(size_t &nx_out, size_t &ny_out) const
    {
        nx_out = nx_;
        ny_out = ny_;
    }

    /**
     * @brief Get convergence history
     * @return Vector of residual norms from iterations
     */
    const std::vector<float> &getResidualHistory() const
    {
        return residualHistory_;
    }

private:
    // Grid dimensions
    size_t nx_, ny_;
    float dx_, dy_;

    // Device pointers
    float *d_u_;              // Current solution
    float *d_u_temp_;         // Temporary solution buffer
    unsigned char *d_bcMask_; // Boundary condition mask
    float *d_bcVals_;         // Boundary condition values
    float *d_residual_;       // Residual array

    // Multigrid hierarchy (device pointers)
    struct GridLevel
    {
        size_t nx, ny;
        float *d_u;
        float *d_rhs;
        float *d_residual;
    };
    std::vector<GridLevel> gridLevels_;

    // Convergence history
    std::vector<float> residualHistory_;

    // Helper functions
    void allocateDeviceMemory();
    void freeDeviceMemory();
    void applyBoundaryConditions();
    float computeL2Norm(const float *d_array, size_t size) const;

    // Multigrid operations
    void restrict(const float *d_fine, float *d_coarse, size_t nx_f, size_t ny_f);
    void prolongate(const float *d_coarse, float *d_fine, size_t nx_c, size_t ny_c, size_t nx_f, size_t ny_f);
    void vCycle(int level, int v1, int v2, float omega);
    void smooth(float *d_u, const float *d_rhs, size_t nx, size_t ny, int iters, float omega);
};

#endif // LAPLACE_CUDA_SOLVER_HPP
