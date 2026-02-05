#include "metal/LaplaceMetalSolver.hpp"
#include <iostream>
#include <vector>
#include <chrono>
#include <fstream>

int main()
{
    std::cout << "[Simple FMG + W-Cycle Test - 1e-8 Target]" << std::endl;

    // Problem setup: Four boundaries (L=1, R=2, T=3, B=4)
    const float L = 1.0f, R = 2.0f, T = 3.0f, B = 4.0f;
    const uint32_t nx = 8193, ny = 8193;

    // Advanced multigrid parameters for ultra-high precision
    const uint32_t max_vcycles = 3000;
    const float tol = 1e-8f; // Target: 1e-8 precision
    const uint32_t nu1 = 8, nu2 = 8;
    const uint32_t coarse_iters = 1500;

    std::cout << "\n=== Ultra-High Precision Configuration ===" << std::endl;
    std::cout << "Grid: " << nx << "x" << ny << " (" << (nx * ny / 1e6) << "M points)" << std::endl;
    std::cout << "Target tolerance: " << tol << std::endl;
    std::cout << "Max V-cycles: " << max_vcycles << std::endl;
    std::cout << "Smoothing: nu1=" << nu1 << ", nu2=" << nu2 << std::endl;
    std::cout << "Coarse iterations: " << coarse_iters << std::endl;

    // Create main solver
    LaplaceMetalSolver::Desc desc{nx, ny, 1.f / (nx - 1), 1.f / (ny - 1)};
    LaplaceMetalSolver solver(desc, "metal/laplace_kernels.metal");
    solver.setSmoother(LaplaceMetalSolver::Smoother::RBGS);
    solver.setStrictConvergence(true);
    solver.setVerbose(true);

    // Set boundary conditions
    std::vector<uint8_t> mask(size_t(nx) * ny, 0);
    std::vector<float> vals(size_t(nx) * ny, 0.0f);

    // Apply four-boundary pattern
    for (uint32_t j = 0; j < ny; ++j)
    {
        mask[j * nx] = 1;
        vals[j * nx] = L; // Left boundary
        mask[j * nx + (nx - 1)] = 1;
        vals[j * nx + (nx - 1)] = R; // Right boundary
    }
    for (uint32_t i = 0; i < nx; ++i)
    {
        mask[i] = 1;
        vals[i] = T; // Top boundary
        mask[(ny - 1) * nx + i] = 1;
        vals[(ny - 1) * nx + i] = B; // Bottom boundary
    }

    // Handle corner averaging
    vals[0] = 0.5f * (T + L);                        // Top-left
    vals[nx - 1] = 0.5f * (T + R);                   // Top-right
    vals[(ny - 1) * nx] = 0.5f * (B + L);            // Bottom-left
    vals[(ny - 1) * nx + (nx - 1)] = 0.5f * (B + R); // Bottom-right

    solver.setDirichletBC(mask, vals);

    // Enhanced bilinear initial guess for better convergence
    std::vector<float> u0(size_t(nx) * ny);
    for (uint32_t j = 0; j < ny; ++j)
    {
        for (uint32_t i = 0; i < nx; ++i)
        {
            float x = float(i) / (nx - 1);
            float y = float(j) / (ny - 1);
            // Better bilinear interpolation that respects boundary gradients
            u0[j * nx + i] = L * (1 - x) + R * x + (T - B) * y + B;
        }
    }
    solver.setInitial(u0);

    bool converged = false;
    double total_ms = 0.0;

    std::cout << "\n=== Phase 1: Standard V-Cycle Approach ===" << std::endl;

    auto t0 = std::chrono::high_resolution_clock::now();
    converged = solver.solveMultigrid(max_vcycles, tol, nu1, nu2, coarse_iters);
    auto t1 = std::chrono::high_resolution_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    total_ms += ms;

    std::cout << "Standard V-Cycle: " << (converged ? "CONVERGED" : "PARTIAL") << " in " << ms << " ms" << std::endl;

    // W-Cycle simulation: Multiple aggressive passes if needed
    if (!converged)
    {
        std::cout << "\n=== Phase 2: W-Cycle Simulation (Enhanced Multi-Pass) ===" << std::endl;

        // Progressive W-cycle strategies with increasing aggressiveness
        std::vector<std::tuple<uint32_t, uint32_t, uint32_t, const char *>> strategies = {
            {nu1 + 2, nu2 + 2, coarse_iters, "Enhanced Smoothing"},
            {nu1 + 4, nu2 + 4, coarse_iters * 2, "Aggressive Smoothing"},
            {nu1 + 6, nu2 + 6, coarse_iters * 3, "Ultra-Aggressive"},
            {nu1 + 8, nu2 + 8, coarse_iters * 4, "Maximum Effort"}};

        for (size_t s = 0; s < strategies.size() && !converged; ++s)
        {
            auto [w_nu1, w_nu2, w_coarse, label] = strategies[s];

            std::cout << "\n--- W-Strategy " << (s + 1) << ": " << label
                      << " (nu1=" << w_nu1 << ", nu2=" << w_nu2 << ", coarse=" << w_coarse << ") ---" << std::endl;

            auto t0_w = std::chrono::high_resolution_clock::now();
            converged = solver.solveMultigrid(max_vcycles / 2, tol, w_nu1, w_nu2, w_coarse);
            auto t1_w = std::chrono::high_resolution_clock::now();
            double ms_w = std::chrono::duration<double, std::milli>(t1_w - t0_w).count();
            total_ms += ms_w;

            std::cout << "Strategy " << (s + 1) << ": " << (converged ? "CONVERGED" : "PARTIAL") << " in " << ms_w << " ms" << std::endl;

            if (converged)
            {
                std::cout << "🎉 SUCCESS: Achieved 1e-8 precision using " << label << "!" << std::endl;
                break;
            }
        }
    }
    else
    {
        std::cout << "🎉 SUCCESS: Standard V-cycles achieved 1e-8 precision!" << std::endl;
    }

    // Final summary
    std::cout << "\n=== FINAL RESULTS ===" << std::endl;
    std::cout << "Convergence: " << (converged ? "✅ ACHIEVED 1e-8 PRECISION" : "❌ PARTIAL CONVERGENCE") << std::endl;
    std::cout << "Total compute time: " << total_ms << " ms (" << (total_ms / 1000.0) << " seconds)" << std::endl;
    std::cout << "Performance: " << (nx * ny * 1e-6) << "M points in " << (total_ms / 1000.0) << "s = " << (nx * ny / 1e6 / (total_ms / 1000.0)) << " Mpoints/sec" << std::endl;

    // Check residual history for detailed analysis
    std::ifstream residual_file("residual_history.csv");
    if (residual_file.good())
    {
        std::cout << "\nResidual progression saved to residual_history.csv" << std::endl;

        std::string line;
        std::vector<std::string> last_lines;
        while (std::getline(residual_file, line) && last_lines.size() < 5)
        {
            last_lines.push_back(line);
        }

        if (!last_lines.empty())
        {
            std::cout << "Final residuals:" << std::endl;
            for (const auto &l : last_lines)
            {
                std::cout << "  " << l << std::endl;
            }
        }
    }

    return converged ? 0 : 1;
}