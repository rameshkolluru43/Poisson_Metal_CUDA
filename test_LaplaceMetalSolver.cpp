#include "metal/LaplaceMetalSolver.hpp"
#include <iostream>
#include <vector>
#include <cassert>
#include <cmath>

// Helper for exception tracing
#define TRACE_EXCEPTION(msg) std::cerr << "[EXCEPTION] " << msg << std::endl;

void test_constructor()
{
    try
    {
        LaplaceMetalSolver::Desc desc{8, 8, 1.0f, 1.0f};
        LaplaceMetalSolver solver(desc, "metal/laplace_kernels.metal");
        std::cout << "Constructor test passed." << std::endl;
    }
    catch (const std::exception &e)
    {
        TRACE_EXCEPTION("Constructor: " << e.what());
        assert(false);
    }
}

void test_setInitial()
{
    try
    {
        LaplaceMetalSolver::Desc desc{8, 8, 1.0f, 1.0f};
        LaplaceMetalSolver solver(desc, "metal/laplace_kernels.metal");
        std::vector<float> u0(64, 1.0f);
        solver.setInitial(u0);
        std::cout << "setInitial test passed." << std::endl;
    }
    catch (const std::exception &e)
    {
        TRACE_EXCEPTION("setInitial: " << e.what());
        assert(false);
    }
}

void test_setDirichletBC()
{
    try
    {
        LaplaceMetalSolver::Desc desc{8, 8, 1.0f, 1.0f};
        LaplaceMetalSolver solver(desc, "metal/laplace_kernels.metal");
        std::vector<uint8_t> mask(64, 0);
        std::vector<float> vals(64, 0.0f);
        for (int i = 0; i < 8; ++i)
        {
            mask[i] = 1;
            mask[56 + i] = 1;
            mask[i * 8] = 1;
            mask[i * 8 + 7] = 1;
        }
        solver.setDirichletBC(mask, vals);
        std::cout << "setDirichletBC test passed." << std::endl;
    }
    catch (const std::exception &e)
    {
        TRACE_EXCEPTION("setDirichletBC: " << e.what());
        assert(false);
    }
}

void test_solveJacobi()
{
    try
    {
        LaplaceMetalSolver::Desc desc{8, 8, 1.0f, 1.0f};
        LaplaceMetalSolver solver(desc, "metal/laplace_kernels.metal");
        std::vector<float> u0(64, 1.0f);
        solver.setInitial(u0);
        std::vector<uint8_t> mask(64, 0);
        std::vector<float> vals(64, 0.0f);
        for (int i = 0; i < 8; ++i)
        {
            mask[i] = 1;
            mask[56 + i] = 1;
            mask[i * 8] = 1;
            mask[i * 8 + 7] = 1;
        }
        solver.setDirichletBC(mask, vals);
        uint32_t iters = 0;
        float res = 0.0f;
        // Jacobi converges slowly; allow more iterations and a reasonable tolerance
        bool ok = solver.solveJacobi(1000, 1e-3f, &iters, &res);
        assert(ok);
        std::cout << "solveJacobi test passed. Iters: " << iters << ", Residual: " << res << std::endl;
    }
    catch (const std::exception &e)
    {
        TRACE_EXCEPTION("solveJacobi: " << e.what());
        assert(false);
    }
}

void test_downloadSolution()
{
    try
    {
        LaplaceMetalSolver::Desc desc{8, 8, 1.0f, 1.0f};
        LaplaceMetalSolver solver(desc, "metal/laplace_kernels.metal");
        solver.setClampEnabled(false);
        std::vector<float> u0(64, 2.0f);
        solver.setInitial(u0);
        // No BC applied; directly read back and check the values
        auto sol = solver.downloadSolution();
        assert(sol.size() == 64);
        for (float v : sol)
            assert(std::fabs(v - 2.0f) < 1e-6f);
        std::cout << "downloadSolution test passed." << std::endl;
    }
    catch (const std::exception &e)
    {
        TRACE_EXCEPTION("downloadSolution: " << e.what());
        assert(false);
    }
}

void test_solveMultigrid()
{
    try
    {
        LaplaceMetalSolver::Desc desc{17, 17, 1.0f, 1.0f};
        LaplaceMetalSolver solver(desc, "metal/laplace_kernels.metal");
        std::vector<float> u0(17 * 17, 1.0f);
        solver.setInitial(u0);
        std::vector<uint8_t> mask(17 * 17, 0);
        std::vector<float> vals(17 * 17, 0.0f);
        for (int i = 0; i < 17; ++i)
        {
            mask[i] = 1;
            mask[16 * 17 + i] = 1;
            mask[i * 17] = 1;
            mask[i * 17 + 16] = 1;
        }
        solver.setDirichletBC(mask, vals);
        bool ok = solver.solveMultigrid(2, 1e-4f, 2, 2, 10);
        assert(ok);
        std::cout << "solveMultigrid test passed." << std::endl;
    }
    catch (const std::exception &e)
    {
        TRACE_EXCEPTION("solveMultigrid: " << e.what());
        assert(false);
    }
}

int main()
{
    test_constructor();
    test_setInitial();
    test_setDirichletBC();
    test_solveJacobi();
    test_downloadSolution();
    test_solveMultigrid();
    std::cout << "All LaplaceMetalSolver unit tests passed." << std::endl;
    return 0;
}
