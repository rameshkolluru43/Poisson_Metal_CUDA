#include "laplace_kernels.cu"
#include <cuda_runtime.h>

// Kernel wrapper functions that can be called from C++

extern "C" void launch_apply_dirichlet(
    float* u,
    const unsigned char* bcMask,
    const float* bcVals,
    Uniforms uni,
    dim3 grid,
    dim3 block
) {
    apply_dirichlet<<<grid, block>>>(u, bcMask, bcVals, uni);
}

extern "C" void launch_jacobi_step(
    const float* u_old,
    float* u_new,
    const unsigned char* bcMask,
    const float* bcVals,
    Uniforms uni,
    dim3 grid,
    dim3 block
) {
    jacobi_step<<<grid, block>>>(u_old, u_new, bcMask, bcVals, uni);
}

extern "C" void launch_jacobi_step_rhs(
    const float* u_old,
    float* u_new,
    const unsigned char* bcMask,
    const float* bcVals,
    Uniforms uni,
    const float* rhs,
    dim3 grid,
    dim3 block
) {
    jacobi_step_rhs<<<grid, block>>>(u_old, u_new, bcMask, bcVals, uni, rhs);
}

extern "C" void launch_rbgs_phase(
    float* u,
    const unsigned char* bcMask,
    const float* bcVals,
    Uniforms uni,
    unsigned int parity,
    dim3 grid,
    dim3 block
) {
    rbgs_phase<<<grid, block>>>(u, bcMask, bcVals, uni, parity);
}

extern "C" void launch_rbgs_phase_rhs(
    float* u,
    const unsigned char* bcMask,
    const float* bcVals,
    Uniforms uni,
    const float* rhs,
    unsigned int parity,
    dim3 grid,
    dim3 block
) {
    rbgs_phase_rhs<<<grid, block>>>(u, bcMask, bcVals, uni, rhs, parity);
}

extern "C" void launch_compute_residual_raw(
    const float* u,
    const unsigned char* bcMask,
    float* r,
    Uniforms uni,
    dim3 grid,
    dim3 block
) {
    compute_residual_raw<<<grid, block>>>(u, bcMask, r, uni);
}

extern "C" void launch_residual_smooth_step(
    const float* r_old,
    float* r_new,
    const unsigned char* bcMask,
    Uniforms uni,
    dim3 grid,
    dim3 block
) {
    residual_smooth_step<<<grid, block>>>(r_old, r_new, bcMask, uni);
}

extern "C" void launch_restrict_full_weighting(
    const float* rf,
    float* rc,
    GridDims gd,
    dim3 grid,
    dim3 block
) {
    restrict_full_weighting<<<grid, block>>>(rf, rc, gd);
}

extern "C" void launch_prolong_bilinear_add(
    const float* ec,
    float* uf,
    GridDims gd,
    dim3 grid,
    dim3 block
) {
    prolong_bilinear_add<<<grid, block>>>(ec, uf, gd);
}

extern "C" void launch_set_zero_float(
    float* buf,
    Uniforms uni,
    dim3 grid,
    dim3 block
) {
    set_zero_float<<<grid, block>>>(buf, uni);
}

extern "C" void launch_clamp_to_bounds(
    float* u,
    const unsigned char* bcMask,
    Uniforms uni,
    float2 bounds,
    dim3 grid,
    dim3 block
) {
    clamp_to_bounds<<<grid, block>>>(u, bcMask, uni, bounds);
}

extern "C" void launch_jacobi_fused_residual_tiled(
    const float* u_old,
    float* u_new,
    const unsigned char* bcMask,
    const float* bcVals,
    Uniforms uni,
    float* partial_sums,
    dim3 grid,
    dim3 block
) {
    jacobi_fused_residual_tiled<<<grid, block>>>(u_old, u_new, bcMask, bcVals, uni, partial_sums);
}

extern "C" void launch_reduce_sum(
    const float* in,
    float* out,
    unsigned int M,
    dim3 grid,
    dim3 block
) {
    reduce_sum<<<grid, block>>>(in, out, M);
}

extern "C" void launch_sum_squares_partial_tiled(
    const float* in,
    Uniforms uni,
    float* partial_sums,
    dim3 grid,
    dim3 block
) {
    sum_squares_partial_tiled<<<grid, block>>>(in, uni, partial_sums);
}

extern "C" void launch_sum_squares_diff_partial_tiled(
    const float* a,
    const float* b,
    Uniforms uni,
    float* partial_sums,
    dim3 grid,
    dim3 block
) {
    sum_squares_diff_partial_tiled<<<grid, block>>>(a, b, uni, partial_sums);
}

extern "C" void launch_sor_step(
    float* u,
    const unsigned char* bcMask,
    const float* bcVals,
    Uniforms uni,
    dim3 grid,
    dim3 block
) {
    sor_step<<<grid, block>>>(u, bcMask, bcVals, uni);
}

extern "C" void launch_sor_step_rhs(
    float* u,
    const unsigned char* bcMask,
    const float* bcVals,
    Uniforms uni,
    const float* rhs,
    dim3 grid,
    dim3 block
) {
    sor_step_rhs<<<grid, block>>>(u, bcMask, bcVals, uni, rhs);
}
