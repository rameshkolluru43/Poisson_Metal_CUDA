#include "LaplaceCUDASolver.hpp"
#include <iostream>
#include <vector>
#include <cmath>
#include <iomanip>

int main()
{
    std::cout << "CUDA Laplace Solver Test\n";
    std::cout << "========================\n\n";

    // Grid parameters
    const size_t nx = 128;
    const size_t ny = 128;
    const float dx = 1.0f / (nx - 1);
    const float dy = 1.0f / (ny - 1);

    std::cout << "Grid size: " << nx << " x " << ny << "\n";
    std::cout << "Grid spacing: dx = " << dx << ", dy = " << dy << "\n\n";

    // Create solver
    LaplaceCUDASolver solver(nx, ny, dx, dy);

    // Set up boundary conditions (Dirichlet on all boundaries)
    std::vector<unsigned char> bcMask(nx * ny, 0);
    std::vector<float> bcVals(nx * ny, 0.0f);

    // Set boundaries: top = 1.0, others = 0.0
    for (size_t i = 0; i < nx; ++i)
    {
        // Bottom boundary
        bcMask[i] = 1;
        bcVals[i] = 0.0f;

        // Top boundary
        bcMask[(ny - 1) * nx + i] = 1;
        bcVals[(ny - 1) * nx + i] = 1.0f;
    }

    for (size_t j = 0; j < ny; ++j)
    {
        // Left boundary
        bcMask[j * nx] = 1;
        bcVals[j * nx] = 0.0f;

        // Right boundary
        bcMask[j * nx + (nx - 1)] = 1;
        bcVals[j * nx + (nx - 1)] = 0.0f;
    }

    solver.setBoundaryConditions(bcMask, bcVals);

    // Initialize solution to zero
    std::vector<float> u(nx * ny, 0.0f);
    solver.setInitialSolution(u);

    // Test 1: Jacobi solver
    std::cout << "Test 1: Jacobi Solver\n";
    std::cout << "---------------------\n";
    int jacobiIters = solver.solveJacobi(1000, 1e-6f, 0.8f);
    std::cout << "Iterations: " << jacobiIters << "\n";
    std::cout << "Final residual: " << solver.computeResidualNorm() << "\n\n";

    // Get solution
    solver.getSolution(u);

    // Print middle row values
    std::cout << "Solution at middle row (y = 0.5):\n";
    size_t mid_j = ny / 2;
    for (size_t i = 0; i < nx; i += nx / 10)
    {
        std::cout << "  u(" << i * dx << ", 0.5) = "
                  << std::fixed << std::setprecision(4) << u[mid_j * nx + i] << "\n";
    }
    std::cout << "\n";

    // Reset for next test
    std::fill(u.begin(), u.end(), 0.0f);
    solver.setInitialSolution(u);

    // Test 2: Red-Black Gauss-Seidel solver
    std::cout << "Test 2: Red-Black Gauss-Seidel Solver\n";
    std::cout << "-------------------------------------\n";
    int rbgsIters = solver.solveRBGS(1000, 1e-6f, 1.0f);
    std::cout << "Iterations: " << rbgsIters << "\n";
    std::cout << "Final residual: " << solver.computeResidualNorm() << "\n\n";

    // Get convergence history
    const auto &history = solver.getResidualHistory();
    std::cout << "Convergence history (first 10 steps):\n";
    for (size_t i = 0; i < std::min(history.size(), size_t(10)); ++i)
    {
        std::cout << "  Step " << std::setw(3) << i * 10 << ": residual = "
                  << std::scientific << history[i] << "\n";
    }
    std::cout << "\n";

    // Reset for next test
    std::fill(u.begin(), u.end(), 0.0f);
    solver.setInitialSolution(u);

    // Test 3: Full Multigrid solver
    std::cout << "Test 3: Full Multigrid Solver\n";
    std::cout << "-----------------------------\n";
    int fmgCycles = solver.solveFMG(2, 2, 4, 1.0f);
    std::cout << "V-cycles: " << fmgCycles << "\n";
    std::cout << "Final residual: " << solver.computeResidualNorm() << "\n\n";

    std::cout << "All tests completed successfully!\n";

    return 0;
}
