#include "OpenCLMGSolver.hpp"
#include <stdexcept>
#include <fstream>
#include <sstream>
#include <cmath>
#include <cstring>

static std::string readTextFile(const std::string &path)
{
    std::ifstream f(path);
    if (!f)
        throw std::runtime_error("Failed to read " + path);
    std::stringstream ss;
    ss << f.rdbuf();
    return ss.str();
}

OpenCLMGSolver::OpenCLMGSolver(const Desc &d, const std::string &kernelPath) : desc_(d)
{
    initOpenCL_(kernelPath);
    buildLevels_();
}

OpenCLMGSolver::~OpenCLMGSolver()
{
    destroyLevels_();
    if (k_apply_dirichlet_)
        clReleaseKernel(k_apply_dirichlet_);
    if (k_residual_)
        clReleaseKernel(k_residual_);
    if (k_residual_helm_)
        clReleaseKernel(k_residual_helm_);
    if (k_jacobi_)
        clReleaseKernel(k_jacobi_);
    if (k_jacobi_helm_)
        clReleaseKernel(k_jacobi_helm_);
    if (k_restrict_fw_)
        clReleaseKernel(k_restrict_fw_);
    if (k_prolong_add_)
        clReleaseKernel(k_prolong_add_);
    if (program_)
        clReleaseProgram(program_);
    if (queue_)
        clReleaseCommandQueue(queue_);
    if (context_)
        clReleaseContext(context_);
}

void OpenCLMGSolver::initOpenCL_(const std::string &kernelPath)
{
    cl_int err;
    cl_uint numPlatforms = 0;
    clGetPlatformIDs(0, nullptr, &numPlatforms);
    if (numPlatforms == 0)
        throw std::runtime_error("No OpenCL platforms");
    std::vector<cl_platform_id> plats(numPlatforms);
    clGetPlatformIDs(numPlatforms, plats.data(), nullptr);
    platform_ = plats[0];
    cl_uint numDevices = 0;
    clGetDeviceIDs(platform_, CL_DEVICE_TYPE_GPU, 0, nullptr, &numDevices);
    if (numDevices == 0)
        clGetDeviceIDs(platform_, CL_DEVICE_TYPE_DEFAULT, 0, nullptr, &numDevices);
    if (numDevices == 0)
        throw std::runtime_error("No OpenCL devices");
    std::vector<cl_device_id> devs(numDevices);
    clGetDeviceIDs(platform_, CL_DEVICE_TYPE_ALL, numDevices, devs.data(), nullptr);
    device_ = devs[0];
    context_ = clCreateContext(nullptr, 1, &device_, nullptr, nullptr, &err);
    if (err)
        throw std::runtime_error("clCreateContext failed");
#ifdef CL_VERSION_2_0
    queue_ = clCreateCommandQueueWithProperties(context_, device_, nullptr, &err);
#else
    queue_ = clCreateCommandQueue(context_, device_, 0, &err);
#endif
    if (err)
        throw std::runtime_error("clCreateCommandQueue failed");

    auto src = readTextFile(kernelPath);
    const char *csrc = src.c_str();
    size_t len = src.size();
    program_ = clCreateProgramWithSource(context_, 1, &csrc, &len, &err);
    if (err)
        throw std::runtime_error("clCreateProgramWithSource failed");
    err = clBuildProgram(program_, 1, &device_, "-cl-fast-relaxed-math", nullptr, nullptr);
    if (err)
    {
        size_t logSize = 0;
        clGetProgramBuildInfo(program_, device_, CL_PROGRAM_BUILD_LOG, 0, nullptr, &logSize);
        std::string log(logSize, '\0');
        clGetProgramBuildInfo(program_, device_, CL_PROGRAM_BUILD_LOG, logSize, log.data(), nullptr);
        throw std::runtime_error("clBuildProgram failed:\n" + log);
    }

    // Create kernels
    k_apply_dirichlet_ = clCreateKernel(program_, "apply_dirichlet", &err);
    k_residual_ = (desc_.dim == Dim::D2) ? clCreateKernel(program_, "residual2d", &err)
                                         : clCreateKernel(program_, "residual3d", &err);
    k_residual_helm_ = (desc_.dim == Dim::D2) ? clCreateKernel(program_, "residual2d_helm", &err)
                                              : clCreateKernel(program_, "residual3d_helm", &err);
    k_jacobi_ = (desc_.dim == Dim::D2) ? clCreateKernel(program_, "jacobi2d", &err)
                                       : clCreateKernel(program_, "jacobi3d", &err);
    k_jacobi_helm_ = (desc_.dim == Dim::D2) ? clCreateKernel(program_, "jacobi2d_helm", &err)
                                            : clCreateKernel(program_, "jacobi3d_helm", &err);
    k_restrict_fw_ = (desc_.dim == Dim::D2) ? clCreateKernel(program_, "restrict2d", &err)
                                            : clCreateKernel(program_, "restrict3d", &err);
    k_prolong_add_ = (desc_.dim == Dim::D2) ? clCreateKernel(program_, "prolong2d_add", &err)
                                            : clCreateKernel(program_, "prolong3d_add", &err);
}

void OpenCLMGSolver::buildLevels_()
{
    // build levels down to >=3 in each dimension
    uint32_t nx = desc_.nx, ny = desc_.ny, nz = (desc_.dim == Dim::D3 ? desc_.nz : 1);
    uint32_t L = 0;
    std::vector<Level> tmp;
    while (nx >= 3 && ny >= 3 && nz >= 1)
    {
        Level lev{};
        lev.nx = nx;
        lev.ny = ny;
        lev.nz = nz;
        lev.N = size_t(nx) * ny * nz;
        tmp.push_back(lev);
        if (desc_.maxLevels && tmp.size() >= desc_.maxLevels)
            break;
        if (nx == 3 || ny == 3 || (desc_.dim == Dim::D3 && nz == 3))
            break;
        nx = (nx - 1) / 2 + 1;
        ny = (ny - 1) / 2 + 1;
        if (desc_.dim == Dim::D3)
            nz = (nz - 1) / 2 + 1;
        L++;
    }
    levels_.assign(tmp.begin(), tmp.end());
    Lf_ = 0;

    cl_int err;
    for (auto &lev : levels_)
    {
        lev.uA = clCreateBuffer(context_, CL_MEM_READ_WRITE, lev.N * sizeof(float), nullptr, &err);
        lev.uB = clCreateBuffer(context_, CL_MEM_READ_WRITE, lev.N * sizeof(float), nullptr, &err);
        lev.rhs = clCreateBuffer(context_, CL_MEM_READ_WRITE, lev.N * sizeof(float), nullptr, &err);
        lev.bcMask = clCreateBuffer(context_, CL_MEM_READ_WRITE, lev.N * sizeof(cl_uchar), nullptr, &err);
        lev.bcVals = clCreateBuffer(context_, CL_MEM_READ_WRITE, lev.N * sizeof(float), nullptr, &err);
        zeroBuffer_(lev.uA, lev.N * sizeof(float));
        zeroBuffer_(lev.uB, lev.N * sizeof(float));
        zeroBuffer_(lev.rhs, lev.N * sizeof(float));
        // default no BC
        std::vector<cl_uchar> zmask(lev.N, 0);
        clEnqueueWriteBuffer(queue_, lev.bcMask, CL_TRUE, 0, lev.N, zmask.data(), 0, nullptr, nullptr);
        std::vector<float> zvals(lev.N, 0.0f);
        clEnqueueWriteBuffer(queue_, lev.bcVals, CL_TRUE, 0, lev.N * sizeof(float), zvals.data(), 0, nullptr, nullptr);
    }
}

void OpenCLMGSolver::destroyLevels_()
{
    for (auto &lev : levels_)
    {
        if (lev.uA)
            clReleaseMemObject(lev.uA);
        if (lev.uB)
            clReleaseMemObject(lev.uB);
        if (lev.rhs)
            clReleaseMemObject(lev.rhs);
        if (lev.bcMask)
            clReleaseMemObject(lev.bcMask);
        if (lev.bcVals)
            clReleaseMemObject(lev.bcVals);
    }
    levels_.clear();
}

void OpenCLMGSolver::zeroBuffer_(cl_mem buf, size_t bytes)
{
    std::vector<char> zero(bytes, 0);
    clEnqueueWriteBuffer(queue_, buf, CL_TRUE, 0, bytes, zero.data(), 0, nullptr, nullptr);
}

void OpenCLMGSolver::setInitial(const std::vector<float> &u0)
{
    auto &lev = levels_[0];
    if (u0.size() != lev.N)
        throw std::runtime_error("u0 size mismatch");
    clEnqueueWriteBuffer(queue_, lev.uA, CL_TRUE, 0, lev.N * sizeof(float), u0.data(), 0, nullptr, nullptr);
}

void OpenCLMGSolver::setRHS(const std::vector<float> &f)
{
    auto &lev = levels_[0];
    if (f.size() != lev.N)
        throw std::runtime_error("rhs size mismatch");
    clEnqueueWriteBuffer(queue_, lev.rhs, CL_TRUE, 0, lev.N * sizeof(float), f.data(), 0, nullptr, nullptr);
}

void OpenCLMGSolver::setDirichletBC(const std::vector<uint8_t> &mask, const std::vector<float> &vals)
{
    // Write finest level BCs directly
    auto &lev0 = levels_[0];
    if (mask.size() != lev0.N || vals.size() != lev0.N)
        throw std::runtime_error("bc size mismatch");
    clEnqueueWriteBuffer(queue_, lev0.bcMask, CL_TRUE, 0, lev0.N * sizeof(cl_uchar), mask.data(), 0, nullptr, nullptr);
    clEnqueueWriteBuffer(queue_, lev0.bcVals, CL_TRUE, 0, lev0.N * sizeof(float), vals.data(), 0, nullptr, nullptr);

    // Downsample BCs to all coarser levels so boundaries are enforced at every level
    const int nx0 = (int)lev0.nx, ny0 = (int)lev0.ny, nz0 = (int)lev0.nz;
    auto idx2 = [&](int i, int j, int nx)
    { return j * nx + i; };
    auto idx3 = [&](int i, int j, int k, int nx, int ny)
    { return (k * ny + j) * nx + i; };
    for (size_t l = 1; l < levels_.size(); ++l)
    {
        auto &lev = levels_[l];
        std::vector<cl_uchar> lmask(lev.N, 0);
        std::vector<float> lvals(lev.N, 0.0f);
        const int nlx = (int)lev.nx, nly = (int)lev.ny, nlz = (int)lev.nz;
        const float sx = (nx0 - 1) / float(nlx - 1);
        const float sy = (ny0 - 1) / float(nly - 1);
        const float sz = (nz0 - 1) / float(std::max(1, nlz - 1));

        if (desc_.dim == Dim::D2)
        {
            for (int j = 0; j < nly; ++j)
            {
                for (int i = 0; i < nlx; ++i)
                {
                    bool isBnd = (i == 0 || j == 0 || i == nlx - 1 || j == nly - 1);
                    if (!isBnd)
                        continue;
                    int fi = (int)std::lround(i * sx);
                    int fj = (int)std::lround(j * sy);
                    fi = std::min(std::max(fi, 0), nx0 - 1);
                    fj = std::min(std::max(fj, 0), ny0 - 1);
                    size_t fid = (size_t)idx2(fi, fj, nx0);
                    size_t lid = (size_t)idx2(i, j, nlx);
                    lmask[lid] = 1;
                    // For coarse levels we solve the error equation with homogeneous Dirichlet BC
                    lvals[lid] = 0.0f;
                }
            }
        }
        else
        {
            for (int k = 0; k < nlz; ++k)
            {
                for (int j = 0; j < nly; ++j)
                {
                    for (int i = 0; i < nlx; ++i)
                    {
                        bool isBnd = (i == 0 || j == 0 || k == 0 || i == nlx - 1 || j == nly - 1 || k == nlz - 1);
                        if (!isBnd)
                            continue;
                        int fi = (int)std::lround(i * sx);
                        int fj = (int)std::lround(j * sy);
                        int fk = (int)std::lround(k * sz);
                        fi = std::min(std::max(fi, 0), nx0 - 1);
                        fj = std::min(std::max(fj, 0), ny0 - 1);
                        fk = std::min(std::max(fk, 0), nz0 - 1);
                        size_t fid = (size_t)idx3(fi, fj, fk, nx0, ny0);
                        size_t lid = (size_t)idx3(i, j, k, nlx, nly);
                        lmask[lid] = 1;
                        lvals[lid] = 0.0f;
                    }
                }
            }
        }
        clEnqueueWriteBuffer(queue_, lev.bcMask, CL_TRUE, 0, lev.N * sizeof(cl_uchar), lmask.data(), 0, nullptr, nullptr);
        clEnqueueWriteBuffer(queue_, lev.bcVals, CL_TRUE, 0, lev.N * sizeof(float), lvals.data(), 0, nullptr, nullptr);
    }
}

void OpenCLMGSolver::applyDirichlet_(Level &lev)
{
    cl_int err = 0;
    int N = (int)lev.N;
    err |= clSetKernelArg(k_apply_dirichlet_, 0, sizeof(cl_mem), &lev.uA);
    err |= clSetKernelArg(k_apply_dirichlet_, 1, sizeof(cl_mem), &lev.bcMask);
    err |= clSetKernelArg(k_apply_dirichlet_, 2, sizeof(cl_mem), &lev.bcVals);
    err |= clSetKernelArg(k_apply_dirichlet_, 3, sizeof(int), &N);
    size_t g = ((N + 255) / 256) * 256;
    clEnqueueNDRangeKernel(queue_, k_apply_dirichlet_, 1, nullptr, &g, nullptr, 0, nullptr, nullptr);
}

void OpenCLMGSolver::jacobiRelax_(Level &lev, bool helmholtz, float alpha, uint32_t iters)
{
    const int nx = lev.nx, ny = lev.ny, nz = lev.nz;
    cl_int err = 0;
    size_t g2[2] = {(size_t)nx, (size_t)ny};
    size_t g3[3] = {(size_t)nx, (size_t)ny, (size_t)nz};
    for (uint32_t k = 0; k < iters; ++k)
    {
        if (desc_.dim == Dim::D2)
        {
            if (helmholtz)
            {
                err |= clSetKernelArg(k_jacobi_helm_, 0, sizeof(cl_mem), &lev.uA);
                err |= clSetKernelArg(k_jacobi_helm_, 1, sizeof(cl_mem), &lev.rhs);
                err |= clSetKernelArg(k_jacobi_helm_, 2, sizeof(cl_mem), &lev.uB);
                float idx2 = 1.0f / (desc_.dx * desc_.dx), idy2 = 1.0f / (desc_.dy * desc_.dy);
                err |= clSetKernelArg(k_jacobi_helm_, 3, sizeof(int), &lev.nx);
                err |= clSetKernelArg(k_jacobi_helm_, 4, sizeof(int), &lev.ny);
                err |= clSetKernelArg(k_jacobi_helm_, 5, sizeof(float), &idx2);
                err |= clSetKernelArg(k_jacobi_helm_, 6, sizeof(float), &idy2);
                err |= clSetKernelArg(k_jacobi_helm_, 7, sizeof(float), &alpha);
                clEnqueueNDRangeKernel(queue_, k_jacobi_helm_, 2, nullptr, g2, nullptr, 0, nullptr, nullptr);
            }
            else
            {
                err |= clSetKernelArg(k_jacobi_, 0, sizeof(cl_mem), &lev.uA);
                err |= clSetKernelArg(k_jacobi_, 1, sizeof(cl_mem), &lev.rhs);
                err |= clSetKernelArg(k_jacobi_, 2, sizeof(cl_mem), &lev.uB);
                float idx2 = 1.0f / (desc_.dx * desc_.dx), idy2 = 1.0f / (desc_.dy * desc_.dy);
                err |= clSetKernelArg(k_jacobi_, 3, sizeof(int), &lev.nx);
                err |= clSetKernelArg(k_jacobi_, 4, sizeof(int), &lev.ny);
                err |= clSetKernelArg(k_jacobi_, 5, sizeof(float), &idx2);
                err |= clSetKernelArg(k_jacobi_, 6, sizeof(float), &idy2);
                clEnqueueNDRangeKernel(queue_, k_jacobi_, 2, nullptr, g2, nullptr, 0, nullptr, nullptr);
            }
        }
        else
        {
            if (helmholtz)
            {
                err |= clSetKernelArg(k_jacobi_helm_, 0, sizeof(cl_mem), &lev.uA);
                err |= clSetKernelArg(k_jacobi_helm_, 1, sizeof(cl_mem), &lev.rhs);
                err |= clSetKernelArg(k_jacobi_helm_, 2, sizeof(cl_mem), &lev.uB);
                float idx2 = 1.0f / (desc_.dx * desc_.dx), idy2 = 1.0f / (desc_.dy * desc_.dy), idz2 = 1.0f / (desc_.dz * desc_.dz);
                err |= clSetKernelArg(k_jacobi_helm_, 3, sizeof(int), &lev.nx);
                err |= clSetKernelArg(k_jacobi_helm_, 4, sizeof(int), &lev.ny);
                err |= clSetKernelArg(k_jacobi_helm_, 5, sizeof(int), &lev.nz);
                err |= clSetKernelArg(k_jacobi_helm_, 6, sizeof(float), &idx2);
                err |= clSetKernelArg(k_jacobi_helm_, 7, sizeof(float), &idy2);
                err |= clSetKernelArg(k_jacobi_helm_, 8, sizeof(float), &idz2);
                err |= clSetKernelArg(k_jacobi_helm_, 9, sizeof(float), &alpha);
                clEnqueueNDRangeKernel(queue_, k_jacobi_helm_, 3, nullptr, g3, nullptr, 0, nullptr, nullptr);
            }
            else
            {
                err |= clSetKernelArg(k_jacobi_, 0, sizeof(cl_mem), &lev.uA);
                err |= clSetKernelArg(k_jacobi_, 1, sizeof(cl_mem), &lev.rhs);
                err |= clSetKernelArg(k_jacobi_, 2, sizeof(cl_mem), &lev.uB);
                float idx2 = 1.0f / (desc_.dx * desc_.dx), idy2 = 1.0f / (desc_.dy * desc_.dy), idz2 = 1.0f / (desc_.dz * desc_.dz);
                err |= clSetKernelArg(k_jacobi_, 3, sizeof(int), &lev.nx);
                err |= clSetKernelArg(k_jacobi_, 4, sizeof(int), &lev.ny);
                err |= clSetKernelArg(k_jacobi_, 5, sizeof(int), &lev.nz);
                err |= clSetKernelArg(k_jacobi_, 6, sizeof(float), &idx2);
                err |= clSetKernelArg(k_jacobi_, 7, sizeof(float), &idy2);
                err |= clSetKernelArg(k_jacobi_, 8, sizeof(float), &idz2);
                clEnqueueNDRangeKernel(queue_, k_jacobi_, 3, nullptr, g3, nullptr, 0, nullptr, nullptr);
            }
        }
        applyDirichlet_(lev);
        swap_(lev.uA, lev.uB);
    }
    if (err)
        throw std::runtime_error("jacobiRelax_ setarg failed");
}

void OpenCLMGSolver::computeResidual_(Level &lev, bool helmholtz, float alpha)
{
    cl_int err = 0;
    const int nx = lev.nx, ny = lev.ny, nz = lev.nz;
    if (desc_.dim == Dim::D2)
    {
        float idx2 = 1.0f / (desc_.dx * desc_.dx), idy2 = 1.0f / (desc_.dy * desc_.dy);
        if (helmholtz)
        {
            err |= clSetKernelArg(k_residual_helm_, 0, sizeof(cl_mem), &lev.uA);
            err |= clSetKernelArg(k_residual_helm_, 1, sizeof(cl_mem), &lev.rhs);
            err |= clSetKernelArg(k_residual_helm_, 2, sizeof(cl_mem), &lev.uB);
            err |= clSetKernelArg(k_residual_helm_, 3, sizeof(int), &lev.nx);
            err |= clSetKernelArg(k_residual_helm_, 4, sizeof(int), &lev.ny);
            err |= clSetKernelArg(k_residual_helm_, 5, sizeof(float), &idx2);
            err |= clSetKernelArg(k_residual_helm_, 6, sizeof(float), &idy2);
            err |= clSetKernelArg(k_residual_helm_, 7, sizeof(float), &alpha);
            size_t g[2] = {(size_t)nx, (size_t)ny};
            clEnqueueNDRangeKernel(queue_, k_residual_helm_, 2, nullptr, g, nullptr, 0, nullptr, nullptr);
        }
        else
        {
            err |= clSetKernelArg(k_residual_, 0, sizeof(cl_mem), &lev.uA);
            err |= clSetKernelArg(k_residual_, 1, sizeof(cl_mem), &lev.rhs);
            err |= clSetKernelArg(k_residual_, 2, sizeof(cl_mem), &lev.uB);
            err |= clSetKernelArg(k_residual_, 3, sizeof(int), &lev.nx);
            err |= clSetKernelArg(k_residual_, 4, sizeof(int), &lev.ny);
            err |= clSetKernelArg(k_residual_, 5, sizeof(float), &idx2);
            err |= clSetKernelArg(k_residual_, 6, sizeof(float), &idy2);
            size_t g[2] = {(size_t)nx, (size_t)ny};
            clEnqueueNDRangeKernel(queue_, k_residual_, 2, nullptr, g, nullptr, 0, nullptr, nullptr);
        }
    }
    else
    {
        float idx2 = 1.0f / (desc_.dx * desc_.dx), idy2 = 1.0f / (desc_.dy * desc_.dy), idz2 = 1.0f / (desc_.dz * desc_.dz);
        if (helmholtz)
        {
            err |= clSetKernelArg(k_residual_helm_, 0, sizeof(cl_mem), &lev.uA);
            err |= clSetKernelArg(k_residual_helm_, 1, sizeof(cl_mem), &lev.rhs);
            err |= clSetKernelArg(k_residual_helm_, 2, sizeof(cl_mem), &lev.uB);
            err |= clSetKernelArg(k_residual_helm_, 3, sizeof(int), &lev.nx);
            err |= clSetKernelArg(k_residual_helm_, 4, sizeof(int), &lev.ny);
            err |= clSetKernelArg(k_residual_helm_, 5, sizeof(int), &lev.nz);
            err |= clSetKernelArg(k_residual_helm_, 6, sizeof(float), &idx2);
            err |= clSetKernelArg(k_residual_helm_, 7, sizeof(float), &idy2);
            err |= clSetKernelArg(k_residual_helm_, 8, sizeof(float), &idz2);
            err |= clSetKernelArg(k_residual_helm_, 9, sizeof(float), &alpha);
            size_t g[3] = {(size_t)nx, (size_t)ny, (size_t)nz};
            clEnqueueNDRangeKernel(queue_, k_residual_helm_, 3, nullptr, g, nullptr, 0, nullptr, nullptr);
        }
        else
        {
            err |= clSetKernelArg(k_residual_, 0, sizeof(cl_mem), &lev.uA);
            err |= clSetKernelArg(k_residual_, 1, sizeof(cl_mem), &lev.rhs);
            err |= clSetKernelArg(k_residual_, 2, sizeof(cl_mem), &lev.uB);
            err |= clSetKernelArg(k_residual_, 3, sizeof(int), &lev.nx);
            err |= clSetKernelArg(k_residual_, 4, sizeof(int), &lev.ny);
            err |= clSetKernelArg(k_residual_, 5, sizeof(int), &lev.nz);
            err |= clSetKernelArg(k_residual_, 6, sizeof(float), &idx2);
            err |= clSetKernelArg(k_residual_, 7, sizeof(float), &idy2);
            err |= clSetKernelArg(k_residual_, 8, sizeof(float), &idz2);
            size_t g[3] = {(size_t)nx, (size_t)ny, (size_t)nz};
            clEnqueueNDRangeKernel(queue_, k_residual_, 3, nullptr, g, nullptr, 0, nullptr, nullptr);
        }
    }
    if (err)
        throw std::runtime_error("computeResidual_ setarg failed");
}

void OpenCLMGSolver::restrictFullWeighting_(Level &fine, Level &coarse)
{
    cl_int err = 0;
    if (desc_.dim == Dim::D2)
    {
        err |= clSetKernelArg(k_restrict_fw_, 0, sizeof(cl_mem), &fine.uB);
        err |= clSetKernelArg(k_restrict_fw_, 1, sizeof(cl_mem), &coarse.rhs);
        err |= clSetKernelArg(k_restrict_fw_, 2, sizeof(int), &fine.nx);
        err |= clSetKernelArg(k_restrict_fw_, 3, sizeof(int), &fine.ny);
        err |= clSetKernelArg(k_restrict_fw_, 4, sizeof(int), &coarse.nx);
        err |= clSetKernelArg(k_restrict_fw_, 5, sizeof(int), &coarse.ny);
        size_t g[2] = {(size_t)coarse.nx, (size_t)coarse.ny};
        clEnqueueNDRangeKernel(queue_, k_restrict_fw_, 2, nullptr, g, nullptr, 0, nullptr, nullptr);
    }
    else
    {
        err |= clSetKernelArg(k_restrict_fw_, 0, sizeof(cl_mem), &fine.uB);
        err |= clSetKernelArg(k_restrict_fw_, 1, sizeof(cl_mem), &coarse.rhs);
        err |= clSetKernelArg(k_restrict_fw_, 2, sizeof(int), &fine.nx);
        err |= clSetKernelArg(k_restrict_fw_, 3, sizeof(int), &fine.ny);
        err |= clSetKernelArg(k_restrict_fw_, 4, sizeof(int), &fine.nz);
        err |= clSetKernelArg(k_restrict_fw_, 5, sizeof(int), &coarse.nx);
        err |= clSetKernelArg(k_restrict_fw_, 6, sizeof(int), &coarse.ny);
        err |= clSetKernelArg(k_restrict_fw_, 7, sizeof(int), &coarse.nz);
        size_t g[3] = {(size_t)coarse.nx, (size_t)coarse.ny, (size_t)coarse.nz};
        clEnqueueNDRangeKernel(queue_, k_restrict_fw_, 3, nullptr, g, nullptr, 0, nullptr, nullptr);
    }
    if (err)
        throw std::runtime_error("restrictFullWeighting_ setarg failed");
}

void OpenCLMGSolver::prolongAdd_(Level &coarse, Level &fine)
{
    cl_int err = 0;
    if (desc_.dim == Dim::D2)
    {
        err |= clSetKernelArg(k_prolong_add_, 0, sizeof(cl_mem), &coarse.uA);
        err |= clSetKernelArg(k_prolong_add_, 1, sizeof(cl_mem), &fine.uA);
        err |= clSetKernelArg(k_prolong_add_, 2, sizeof(int), &fine.nx);
        err |= clSetKernelArg(k_prolong_add_, 3, sizeof(int), &fine.ny);
        err |= clSetKernelArg(k_prolong_add_, 4, sizeof(int), &coarse.nx);
        err |= clSetKernelArg(k_prolong_add_, 5, sizeof(int), &coarse.ny);
        size_t g[2] = {(size_t)fine.nx, (size_t)fine.ny};
        clEnqueueNDRangeKernel(queue_, k_prolong_add_, 2, nullptr, g, nullptr, 0, nullptr, nullptr);
    }
    else
    {
        err |= clSetKernelArg(k_prolong_add_, 0, sizeof(cl_mem), &coarse.uA);
        err |= clSetKernelArg(k_prolong_add_, 1, sizeof(cl_mem), &fine.uA);
        err |= clSetKernelArg(k_prolong_add_, 2, sizeof(int), &fine.nx);
        err |= clSetKernelArg(k_prolong_add_, 3, sizeof(int), &fine.ny);
        err |= clSetKernelArg(k_prolong_add_, 4, sizeof(int), &fine.nz);
        err |= clSetKernelArg(k_prolong_add_, 5, sizeof(int), &coarse.nx);
        err |= clSetKernelArg(k_prolong_add_, 6, sizeof(int), &coarse.ny);
        err |= clSetKernelArg(k_prolong_add_, 7, sizeof(int), &coarse.nz);
        size_t g[3] = {(size_t)fine.nx, (size_t)fine.ny, (size_t)fine.nz};
        clEnqueueNDRangeKernel(queue_, k_prolong_add_, 3, nullptr, g, nullptr, 0, nullptr, nullptr);
    }
    if (err)
        throw std::runtime_error("prolongAdd_ setarg failed");
}

float OpenCLMGSolver::residualL2_(Level &lev, bool helmholtz, float alpha)
{
    // compute residual into lev.uB, reduce on host (simple)
    computeResidual_(lev, helmholtz, alpha);
    clFinish(queue_);
    std::vector<float> r(lev.N);
    clEnqueueReadBuffer(queue_, lev.uB, CL_TRUE, 0, lev.N * sizeof(float), r.data(), 0, nullptr, nullptr);
    double s = 0.0;
    for (size_t i = 0; i < r.size(); ++i)
        s += (double)r[i] * (double)r[i];
    return (float)std::sqrt(s);
}

bool OpenCLMGSolver::solveSteady(uint32_t maxVCycles, float tol, uint32_t nu1, uint32_t nu2, uint32_t coarseIters)
{
    return solveHelmholtz(0.0f, maxVCycles, tol, nu1, nu2, coarseIters);
}

bool OpenCLMGSolver::solveHelmholtz(float alpha, uint32_t maxVCycles, float tol, uint32_t nu1, uint32_t nu2, uint32_t coarseIters)
{
    const bool helm = (alpha > 0.0f);
    float r0 = residualL2_(levels_[0], helm, alpha);
    if (verbose_)
        fprintf(stderr, "[CL-MG] r0=%g\n", r0);
    // Combined absolute/relative tolerance (robust across grid sizes)
    const float target = std::max(tol, tol * r0);
    for (uint32_t v = 0; v < maxVCycles; ++v)
    {
        // Pre-smooth
        jacobiRelax_(levels_[0], helm, alpha, nu1);
        applyDirichlet_(levels_[0]);

        // Descend
        for (size_t l = 0; l + 1 < levels_.size(); ++l)
        {
            // Compute fine-level residual and use it as RHS on the next coarser level
            computeResidual_(levels_[l], helm, alpha);
            zeroBuffer_(levels_[l + 1].rhs, levels_[l + 1].N * sizeof(float));
            restrictFullWeighting_(levels_[l], levels_[l + 1]);
            zeroBuffer_(levels_[l + 1].uA, levels_[l + 1].N * sizeof(float));
            // smooth on coarse
            jacobiRelax_(levels_[l + 1], helm, alpha, (l + 1 == levels_.size() - 1) ? coarseIters : nu1);
            applyDirichlet_(levels_[l + 1]);
        }
        // Ascend
        for (int l = (int)levels_.size() - 2; l >= 0; --l)
        {
            prolongAdd_(levels_[l + 1], levels_[l]);
            jacobiRelax_(levels_[l], helm, alpha, nu2);
            applyDirichlet_(levels_[l]);
        }
        float r = residualL2_(levels_[0], helm, alpha);
        if (verbose_)
            fprintf(stderr, "[CL-MG] v=%u r=%g\n", v, r);
        if (r <= target)
            return true;
    }
    return false;
}

bool OpenCLMGSolver::stepBackwardEuler(float dt, uint32_t vcyclesPerStep, float tol, uint32_t nu1, uint32_t nu2, uint32_t coarseIters)
{
    // (I - dt*Lap) u^{n+1} = u^n + dt*f^n  => alpha=1/dt, b = alpha * u^n + f^n
    auto &lev = levels_[0];
    std::vector<float> u0(lev.N);
    clEnqueueReadBuffer(queue_, lev.uA, CL_TRUE, 0, lev.N * sizeof(float), u0.data(), 0, nullptr, nullptr);
    std::vector<float> f(lev.N);
    clEnqueueReadBuffer(queue_, lev.rhs, CL_TRUE, 0, lev.N * sizeof(float), f.data(), 0, nullptr, nullptr);
    float alpha = 1.0f / dt;
    for (size_t i = 0; i < lev.N; ++i)
        f[i] += alpha * u0[i];
    clEnqueueWriteBuffer(queue_, lev.rhs, CL_TRUE, 0, lev.N * sizeof(float), f.data(), 0, nullptr, nullptr);
    return solveHelmholtz(alpha, vcyclesPerStep, tol, nu1, nu2, coarseIters);
}

std::vector<float> OpenCLMGSolver::downloadSolution() const
{
    auto &lev = const_cast<OpenCLMGSolver *>(this)->levels_[0];
    std::vector<float> u(lev.N);
    clEnqueueReadBuffer(queue_, lev.uA, CL_TRUE, 0, lev.N * sizeof(float), u.data(), 0, nullptr, nullptr);
    return u;
}
