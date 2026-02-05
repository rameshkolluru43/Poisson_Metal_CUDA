#pragma once
#include <cstdint>
#include <vector>
#include <string>
#include <functional>
#include <fstream>
#include <string>

// Simple forward declarations - avoid Objective-C++ issues in header
struct MTLCommandBuffer;
typedef void *id;

class LaplaceMetalSolver
{
public:
    enum class Smoother
    {
        Jacobi,
        RBGS,
        SOR
    };

    enum class CycleType
    {
        V_Cycle,
        W_Cycle
    };
    struct Desc
    {
        uint32_t nx = 0, ny = 0;
        float dx = 1.f, dy = 1.f;
    };

    struct MGLevel
    {
        uint32_t nx = 0, ny = 0;
        float dx = 1.f, dy = 1.f;
        // Solution/correction ping-pong
        void *uA = nullptr; // current
        void *uB = nullptr; // next
        // RHS / residual on this level
        void *rhs = nullptr;
        // Boundary data (mask=1 at domain boundary; values = nonhom on L0, zeros otherwise)
        void *bcMask = nullptr; // uint8_t
        void *bcVals = nullptr; // float
        // Uniforms for this level
        void *uniforms = nullptr;
        size_t fieldBytes = 0;
    };
    std::vector<MGLevel> levels_;

    /*
     * Constructor for LaplaceMetalSolver.
     * Initializes the solver with the given descriptor and the Metal shader path.
     * @param d Descriptor containing grid dimensions and spacing.
     * @param metal_path Filesystem path to the Metal shader source (.metal). Must be provided.
     * @throws std::runtime_error if initialization fails (e.g., invalid parameters,
     *                            missing/invalid shader path, Metal device or queue creation failure,
     *                            or shader compilation/pipeline creation errors).
     */
    LaplaceMetalSolver(const Desc &d, const char *metal_path = nullptr);

    ~LaplaceMetalSolver();

    void setInitial(const std::vector<float> &u0);

    void setDirichletBC(const std::vector<uint8_t> &mask,
                        const std::vector<float> &values);

    bool solveJacobi(uint32_t max_iters, float tol,
                     uint32_t *out_iters = nullptr,
                     float *residual_l2 = nullptr);

    // Single-level Red-Black Gauss–Seidel (in-place) solver.
    // Runs alternating red/black updates until residual L2 <= tol or max_iters reached.
    bool solveRBGS(uint32_t max_iters, float tol,
                   uint32_t *out_iters = nullptr,
                   float *residual_l2 = nullptr);

    std::vector<float> downloadSolution() const;

    // Optional: provide a right-hand side f for Poisson: -∇² u = f
    // Copies data into the finest level's rhs buffer.
    void setRHS(const std::vector<float> &rhs);

    // Single-level Red-Black Gauss–Seidel solve with RHS (Poisson form).
    // Uses rbgs_phase_rhs kernel and stops when L2 residual (rhs - Au) <= tol.
    bool solveRBGSWithRHS(uint32_t max_iters, float tol,
                          uint32_t *out_iters = nullptr,
                          float *residual_l2 = nullptr);

    // Fused single-grid Jacobi with GPU-side residual reduction. Optional drop-in fast path.
    bool solveJacobiFused(uint32_t max_iters, float tol,
                          uint32_t flushStride = 50,
                          uint32_t *out_iters = nullptr,
                          float *residual_l2 = nullptr);

    // Toggle safety clamp to boundary range. Enable for Laplace (f=0). Disable for Poisson (f!=0).
    void setClampEnabled(bool e) { clampEnabled_ = e; }
    bool clampEnabled() const { return clampEnabled_; }

    LaplaceMetalSolver(const LaplaceMetalSolver &) = delete;
    LaplaceMetalSolver &operator=(const LaplaceMetalSolver &) = delete;

    // Optional: control verbosity of internal logging (default: false)
    void setVerbose(bool v) { verbose_ = v; }
    bool verbose() const { return verbose_; }

    // Optional: control Jacobi damping factor omega in (0,1]. Default: 0.7f
    void setDamping(float w)
    {
        damping_ = w;
        updateTopLevelUniforms_();
    }
    float damping() const { return damping_; }

    // Optional: choose the smoother used in multigrid (default: RBGS)
    void setSmoother(Smoother s) { smoother_ = s; }
    Smoother smoother() const { return smoother_; }

    // Optional: choose the cycle type used in multigrid (default: V_Cycle)
    void setCycleType(CycleType c) { cycleType_ = c; }
    CycleType cycleType() const { return cycleType_; }

    // Optional: relative residual stopping criterion for multigrid (res/res0 <= relTol)
    void setRelativeTolerance(float r) { relTol_ = r; }
    float relativeTolerance() const { return relTol_; }

    // Optional: disable early-stop heuristics and force strict convergence behavior
    // (e.g., when targeting very small residuals like 1e-9).
    void setStrictConvergence(bool s) { strictConvergence_ = s; }

    // Enable/disable per-level V-cycle statistics collection.
    // When enabled, each V-cycle captures initial and final residual L2 at every level
    // and writes rows to a CSV if a non-empty path is provided.
    void enableVcycleStats(bool enable, const std::string &csvPath = "")
    {
        statsEnabled_ = enable;
        statsCsvPath_ = csvPath;
    }

    struct VLevelStat
    {
        uint32_t cycle;
        uint32_t level;
        uint32_t nx, ny;
        uint32_t nu1, nu2;
        uint32_t coarse_iters;
        float damping;
        Smoother smoother;
        float res_init;
        float res_final;
    };
    const std::vector<VLevelStat> &getVcycleStats() const { return vstats_; }

    // Geometric Multigrid V-cycle solver.
    // max_vcycles: number of V-cycles
    // tol: stop if L2 residual on finest level <= tol
    // nu1/nu2: pre/post relaxations per level
    // coarse_iters: smoothing iterations on coarsest grid
    // Note: residual measurements during cycles use GPU-side reductions (no full readbacks).
    bool solveMultigrid(uint32_t max_vcycles, float tol,
                        uint32_t nu1 = 3, uint32_t nu2 = 3,
                        uint32_t coarse_iters = 30);

    // Debug: run a GPU vs CPU full-weighting restriction test
    // Fills the fine residual with a known pattern, runs restriction,
    // and reports element-wise differences (max abs diff printed).
    void debugRestrictTest();

    // Debug: run a GPU vs CPU bilinear prolongation test
    // Fills a coarse correction with a deterministic pattern, runs prolongation
    // (add) into fine grid and compares GPU result against CPU bilinear interpolation.
    void debugProlongTest();

    // Debug: run a GPU vs CPU residual test
    // Fills fine-level solution with a deterministic pattern, runs compute_residual_raw,
    // and compares GPU residuals to CPU finite-difference r = -(uxx+uyy).
    void debugResidualTest();

    // Debug: run a GPU vs CPU single Jacobi-with-RHS step
    // Fills u_old and rhs with deterministic patterns, runs jacobi_step_rhs once
    // and compares GPU u_new against CPU update formula.
    void debugJacobiRhsTest();

    // Debug: print per-level Uniforms and verify coarsening scaling
    void debugCheckUniforms();

    // Optional: set number of residual smoothing passes before restriction (default: 0)
    void setResidualSmoothingPasses(uint32_t passes);
    uint32_t residualSmoothingPasses() const { return residualSmoothPasses_; }

private:
    void *device_ = nullptr;
    void *queue_ = nullptr;
    void *lib_ = nullptr;
    void *psoJacobi_ = nullptr;
    void *psoApplyBC_ = nullptr;
    // Optional smoother pipelines
    void *psoRBGS_ = nullptr;    // rbgs_phase (homogeneous)
    void *psoRBGSRHS_ = nullptr; // rbgs_phase_rhs (with RHS)
    void *psoSOR_ = nullptr;     // sor_step (homogeneous)
    void *psoSORRHS_ = nullptr;  // sor_step_rhs (with RHS)
    void *bufU0_ = nullptr;
    void *bufU1_ = nullptr;
    void *bufBCMask_ = nullptr;
    void *bufBCVals_ = nullptr;

    void *bufUniforms_ = nullptr;
    uint32_t nx_ = 0, ny_ = 0;
    float dx_ = 1, dy_ = 1;
    size_t fieldBytes_ = 0;
    // Extra pipelines for MG
    void *psoResidualRaw_ = nullptr;    // compute_residual_raw
    void *psoRestrict_ = nullptr;       // restrict_full_weighting
    void *psoProlongAdd_ = nullptr;     // prolong_bilinear_add
    void *psoJacobiRHS_ = nullptr;      // jacobi_step_rhs
    void *psoZeroFloat_ = nullptr;      // set_zero_float
    void *psoClamp_ = nullptr;          // clamp_to_bounds
    void *psoResidualSmooth_ = nullptr; // residual_smooth_step
    // Fused tiled kernels
    void *psoJacobiFused_ = nullptr;           // jacobi_fused_residual_tiled
    void *psoReduceSum_ = nullptr;             // reduce_sum
    void *psoSumSquaresPartial_ = nullptr;     // sum_squares_partial_tiled
    void *psoSumSquaresDiffPartial_ = nullptr; // sum_squares_diff_partial_tiled

    // Reusable partial sum buffer for finest level reductions
    void *bufPartial0_ = nullptr; // float[M] where M = groupsX*groupsY
    uint32_t bufPartial0Cap_ = 0; // capacity in elements

    bool compileLibraryFromFile_(const char *, std::string &);
    bool compileLibraryFromEmbedded_(std::string &);
    bool buildPipelines_(std::string &);
    bool ensureBuffers_();
    float computeResidualL2_();
    bool applyBC_();
    void swapFields_();
    void clampToBoundaryRange_();

    bool buildLevels_();   // allocate/build level hierarchy
    void destroyLevels_(); // free level buffers
    bool vcycle_(uint32_t nu1, uint32_t nu2, uint32_t coarse_iters);

    // Modular V-cycle components
    void downwardSweep_(void *cb, uint32_t nu1, const size_t nlevels);
    void coarseGridSolve_(void *cb, uint32_t coarse_iters, const size_t nlevels);
    void upwardSweep_(void *cb, uint32_t nu2, const size_t nlevels);
    float computeResidualAndCheck_(uint32_t cycle, float res0, float prev_res, uint32_t &consecUp,
                                   bool tightTol, bool doMeasure, std::ofstream &rh);

    // W-cycle components
    void wCycleDownward_(void *cb, uint32_t nu1, const size_t nlevels, size_t current_level);
    void wCycleUpward_(void *cb, uint32_t nu2, const size_t nlevels, size_t current_level);

    bool smoothJacobi_(size_t lev, uint32_t iters, bool use_rhs);
    bool smoothRBGS_(size_t lev, uint32_t iters, bool use_rhs);
    bool computeResidual_(size_t lev);       // writes levels_[lev].rhs = residual
    bool restrictDown_(size_t lev_from);     // rhs_{l+1} = R rhs_l
    bool prolongateAdd_(size_t lev_to_fine); // u_l += P e_{l+1}
    bool applyBCLevel_(size_t lev);          // enforce BCs on levels_[lev].uA
    bool zeroFloat_(void *buf, size_t lev);  // GPU zero
    float computeResidualL2WithRHS_();       // CPU reduction of ||rhs - A u|| on finest level
    float computeResidualL2Fused_(uint32_t groupsX, uint32_t groupsY);
    bool ensurePartialBuffer0_(uint32_t groupsX, uint32_t groupsY);

    // Stats helpers
    float computeResidualL2Level_(size_t lev);
    void computeResidualsAllLevels_(std::vector<float> &out);

    // Config flags
    bool verbose_ = false;
    float damping_ = 2.0f / 3.0f;
    Smoother smoother_ = Smoother::Jacobi;
    CycleType cycleType_ = CycleType::V_Cycle;
    float relTol_ = -1.0f;
    bool strictConvergence_ = false;

    // V-cycle stats state
    bool statsEnabled_ = false;
    std::string statsCsvPath_;
    std::vector<VLevelStat> vstats_;

    // Residual smoothing state
    uint32_t residualSmoothPasses_ = 0;

    // Boundary extrema on finest level (for safety clamp)
    float bcMin_ = 0.0f;
    float bcMax_ = 0.0f;
    bool clampEnabled_ = true;

    // Helper to refresh omega in the top-level uniforms buffer
    void updateTopLevelUniforms_();

    // Residual metrics on finest level (level 0)
    struct ResidualMetrics
    {
        float l2;
        float l2_h;
        float linf;
        size_t n;
    };
    bool computeResidualMetrics0_(ResidualMetrics &out);
};
