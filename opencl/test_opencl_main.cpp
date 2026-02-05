#include "OpenCLMGSolver.hpp"
#include <iostream>
#include <fstream>
#include <vector>
#include <chrono>

static std::string resolve_kernels_path()
{
    // Try common locations relative to likely working directories
    const char *candidates[] = {
        "opencl/kernels.cl",   // when running from project root
        "kernels.cl",          // when running from opencl/ directory
        "../opencl/kernels.cl" // when running from a subdir of opencl/
    };
    for (const char *p : candidates)
    {
        std::ifstream f(p);
        if (f.good())
            return std::string(p);
    }
    // Fallback to default path; constructor will throw with a clear message
    return std::string("opencl/kernels.cl");
}

static void make_dirichlet_bc_avg_corners(uint32_t nx, uint32_t ny,
                                          float L, float R, float T, float B,
                                          std::vector<uint8_t> &mask,
                                          std::vector<float> &vals)
{
    mask.assign(size_t(nx) * ny, 0);
    vals.assign(size_t(nx) * ny, 0.0f);
    for (uint32_t j = 0; j < ny; ++j)
    {
        mask[j * nx] = 1;
        vals[j * nx] = L; // left
        mask[j * nx + (nx - 1)] = 1;
        vals[j * nx + (nx - 1)] = R; // right
    }
    for (uint32_t i = 0; i < nx; ++i)
    {
        mask[i] = 1;
        vals[i] = T; // top
        mask[(ny - 1) * nx + i] = 1;
        vals[(ny - 1) * nx + i] = B; // bottom
    }
    // corner averaging
    vals[0] = 0.5f * (T + L);
    vals[nx - 1] = 0.5f * (T + R);
    vals[(ny - 1) * nx] = 0.5f * (B + L);
    vals[(ny - 1) * nx + (nx - 1)] = 0.5f * (B + R);
}

int main(int argc, char **argv)
{
    try
    {
        // Simple 2D Laplace test: Left=1, others=0
        uint32_t nx = 129, ny = 129;
        OpenCLMGSolver::Desc d;
        d.dim = OpenCLMGSolver::Dim::D2;
        d.nx = nx;
        d.ny = ny;
        d.nz = 1;
        d.dx = 1.0f / float(nx - 1);
        d.dy = 1.0f / float(ny - 1);
        d.dz = 1.0f;
        d.maxLevels = 0; // auto

        std::string kpath = resolve_kernels_path();
        OpenCLMGSolver solver(d, kpath);
        // Optional: enable to inspect convergence progress
        // solver.setVerbose(true);
        std::vector<float> u0(size_t(nx) * ny, 0.0f);
        solver.setInitial(u0);

        std::vector<float> rhs(size_t(nx) * ny, 0.0f);
        solver.setRHS(rhs);

        std::vector<uint8_t> mask;
        std::vector<float> vals;
        make_dirichlet_bc_avg_corners(nx, ny, /*L=*/1.0f, /*R=*/0.0f, /*T=*/0.0f, /*B=*/0.0f, mask, vals);
        solver.setDirichletBC(mask, vals);

        auto t0 = std::chrono::high_resolution_clock::now();
        bool ok = solver.solveSteady(/*maxVCycles=*/2000, /*tol=*/1e-3f, /*nu1=*/5, /*nu2=*/5, /*coarseIters=*/400);
        auto t1 = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        std::cerr << "[OpenCL-MG] Laplace 2D " << nx << "x" << ny << ": ok=" << ok << ", time_ms=" << ms << "\n";

        auto U = solver.downloadSolution();
        std::ofstream f("opencl_solution_laplace2d.csv");
        for (uint32_t j = 0; j < ny; ++j)
        {
            for (uint32_t i = 0; i < nx; ++i)
            {
                f << U[size_t(j) * nx + i];
                if (i + 1 < nx)
                    f << ",";
            }
            f << "\n";
        }
        return 0;
    }
    catch (const std::exception &e)
    {
        std::cerr << "OpenCL test failed: " << e.what() << "\n";
        return 1;
    }
}
