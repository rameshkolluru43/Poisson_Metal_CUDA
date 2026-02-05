// OpenCL kernels for multigrid diffusion/Poisson in 2D/3D
// Coordinate indexing is row-major for 2D and z-major for 3D: id = (k*ny + j)*nx + i

inline int idx2(int i,int j,int nx){ return j*nx + i; }
inline int idx3(int i,int j,int k,int nx,int ny){ return (k*ny + j)*nx + i; }

__kernel void apply_dirichlet(__global float* u,
                              __global const uchar* bcMask,
                              __global const float* bcVals,
                              int N)
{
    int id = get_global_id(0);
    if(id>=N) return;
    if(bcMask[id]) u[id] = bcVals[id];
}

__kernel void residual2d(__global const float* u,
                         __global const float* rhs,
                         __global float* r,
                         int nx,int ny,
                         float idx2,float idy2)
{
    int i = get_global_id(0);
    int j = get_global_id(1);
    if(i<=0||j<=0||i>=nx-1||j>=ny-1){ r[j*nx+i]=0.0f; return; }
    int id = j*nx + i;
    float lap = (u[id+1]-2.0f*u[id]+u[id-1])*idx2 + (u[id+nx]-2.0f*u[id]+u[id-nx])*idy2;
    // For equation -Lap(u) = rhs, the operator applied to u is (-Lap(u))
    // Here 'lap' computed above equals +Lap(u), so -Lap(u) = -(lap)
    r[id] = rhs[id] + lap; // rhs - ( -Lap(u) )
}

__kernel void residual2d_helm(__global const float* u,
                              __global const float* rhs,
                              __global float* r,
                              int nx,int ny,
                              float idx2,float idy2,
                              float alpha)
{
    int i = get_global_id(0);
    int j = get_global_id(1);
    if(i<=0||j<=0||i>=nx-1||j>=ny-1){ r[j*nx+i]=0.0f; return; }
    int id = j*nx + i;
    float lap = (u[id+1]-2.0f*u[id]+u[id-1])*idx2 + (u[id+nx]-2.0f*u[id]+u[id-nx])*idy2;
    // For Helmholtz: alpha*u - Lap(u)
    r[id] = rhs[id] - (alpha*u[id] - lap);
}

__kernel void residual3d(__global const float* u,
                         __global const float* rhs,
                         __global float* r,
                         int nx,int ny,int nz,
                         float idx2,float idy2,float idz2)
{
    int i = get_global_id(0);
    int j = get_global_id(1);
    int k = get_global_id(2);
    if(i<=0||j<=0||k<=0||i>=nx-1||j>=ny-1||k>=nz-1){ r[(k*ny + j)*nx + i]=0.0f; return; }
    int id = (k*ny + j)*nx + i;
    float lap = (u[id+1]-2.0f*u[id]+u[id-1])*idx2
              + (u[id+nx]-2.0f*u[id]+u[id-nx])*idy2
              + (u[id+nx*ny]-2.0f*u[id]+u[id-nx*ny])*idz2;
    r[id] = rhs[id] + lap;
}

__kernel void residual3d_helm(__global const float* u,
                              __global const float* rhs,
                              __global float* r,
                              int nx,int ny,int nz,
                              float idx2,float idy2,float idz2,
                              float alpha)
{
    int i = get_global_id(0);
    int j = get_global_id(1);
    int k = get_global_id(2);
    if(i<=0||j<=0||k<=0||i>=nx-1||j>=ny-1||k>=nz-1){ r[(k*ny + j)*nx + i]=0.0f; return; }
    int id = (k*ny + j)*nx + i;
    float lap = (u[id+1]-2.0f*u[id]+u[id-1])*idx2
              + (u[id+nx]-2.0f*u[id]+u[id-nx])*idy2
              + (u[id+nx*ny]-2.0f*u[id]+u[id-nx*ny])*idz2;
    r[id] = rhs[id] - (alpha*u[id] - lap);
}

__kernel void jacobi2d(__global const float* u,
                       __global const float* rhs,
                       __global float* unew,
                       int nx,int ny,
                       float idx2,float idy2)
{
    int i = get_global_id(0);
    int j = get_global_id(1);
    if(i<=0||j<=0||i>=nx-1||j>=ny-1) return;
    int id = j*nx + i;
    // Solve -Lap(u)=rhs => (2*(idx2+idy2))*u - sum = rhs => u = (sum - rhs)/denom
    float sum = (u[id+1]+u[id-1])*idx2 + (u[id+nx]+u[id-nx])*idy2;
    float denom = 2.0f*(idx2+idy2);
    unew[id] = (sum - rhs[id]) / denom;
}

__kernel void jacobi2d_helm(__global const float* u,
                            __global const float* rhs,
                            __global float* unew,
                            int nx,int ny,
                            float idx2,float idy2,
                            float alpha)
{
    int i = get_global_id(0);
    int j = get_global_id(1);
    if(i<=0||j<=0||i>=nx-1||j>=ny-1) return;
    int id = j*nx + i;
    float sum = (u[id+1]+u[id-1])*idx2 + (u[id+nx]+u[id-nx])*idy2;
    float denom = alpha + 2.0f*(idx2+idy2);
    unew[id] = (sum - rhs[id]) / denom;
}

__kernel void jacobi3d(__global const float* u,
                       __global const float* rhs,
                       __global float* unew,
                       int nx,int ny,int nz,
                       float idx2,float idy2,float idz2)
{
    int i = get_global_id(0);
    int j = get_global_id(1);
    int k = get_global_id(2);
    if(i<=0||j<=0||k<=0||i>=nx-1||j>=ny-1||k>=nz-1) return;
    int id = (k*ny + j)*nx + i;
    float sum = (u[id+1]+u[id-1])*idx2 + (u[id+nx]+u[id-nx])*idy2 + (u[id+nx*ny]+u[id-nx*ny])*idz2;
    float denom = 2.0f*(idx2+idy2+idz2);
    unew[id] = (sum - rhs[id]) / denom;
}

__kernel void jacobi3d_helm(__global const float* u,
                            __global const float* rhs,
                            __global float* unew,
                            int nx,int ny,int nz,
                            float idx2,float idy2,float idz2,
                            float alpha)
{
    int i = get_global_id(0);
    int j = get_global_id(1);
    int k = get_global_id(2);
    if(i<=0||j<=0||k<=0||i>=nx-1||j>=ny-1||k>=nz-1) return;
    int id = (k*ny + j)*nx + i;
    float sum = (u[id+1]+u[id-1])*idx2 + (u[id+nx]+u[id-nx])*idy2 + (u[id+nx*ny]+u[id-nx*ny])*idz2;
    float denom = alpha + 2.0f*(idx2+idy2+idz2);
    unew[id] = (sum - rhs[id]) / denom;
}

// full-weighting restriction (2h <- h) for 2D
__kernel void restrict2d(__global const float* fine,
                         __global float* coarse,
                         int nx_f,int ny_f,
                         int nx_c,int ny_c)
{
    int I = get_global_id(0);
    int J = get_global_id(1);
    if(I<=0||J<=0||I>=nx_c-1||J>=ny_c-1) return;
    int i = 2*I; int j = 2*J;
    // 9-point full weighting
    float s = 0.0f;
    s += 0.25f * fine[j*nx_f + i];
    s += 0.125f * (fine[j*nx_f + i-1] + fine[j*nx_f + i+1] + fine[(j-1)*nx_f + i] + fine[(j+1)*nx_f + i]);
    s += 0.0625f * (fine[(j-1)*nx_f + i-1] + fine[(j-1)*nx_f + i+1] + fine[(j+1)*nx_f + i-1] + fine[(j+1)*nx_f + i+1]);
    coarse[J*nx_c + I] = s;
}

// trilinear restriction (3D) using 3x3x3 weights (simplified equal weights for neighbors)
__kernel void restrict3d(__global const float* fine,
                         __global float* coarse,
                         int nx_f,int ny_f,int nz_f,
                         int nx_c,int ny_c,int nz_c)
{
    int I = get_global_id(0);
    int J = get_global_id(1);
    int K = get_global_id(2);
    if(I<=0||J<=0||K<=0||I>=nx_c-1||J>=ny_c-1||K>=nz_c-1) return;
    int i = 2*I; int j = 2*J; int k = 2*K;
    float s = 0.0f; int cnt=0;
    for(int dk=-1;dk<=1;++dk) for(int dj=-1;dj<=1;++dj) for(int di=-1;di<=1;++di){
        int ii=i+di, jj=j+dj, kk=k+dk;
        s += fine[(kk*ny_f + jj)*nx_f + ii]; cnt++; }
    coarse[(K*ny_c + J)*nx_c + I] = s / (float)cnt;
}

// prolongation + add (bilinear/trilinear)
__kernel void prolong2d_add(__global const float* coarse,
                            __global float* fine,
                            int nx_f,int ny_f,
                            int nx_c,int ny_c)
{
    int i = get_global_id(0);
    int j = get_global_id(1);
    if(i<=0||j<=0||i>=nx_f-1||j>=ny_f-1) return;
    float I = 0.5f*i; float J = 0.5f*j;
    int I0 = (int)floor(I); int J0 = (int)floor(J);
    float a = I - I0; float b = J - J0;
    // bilinear weights
    float c00 = coarse[J0*nx_c + I0];
    float c10 = coarse[J0*nx_c + (I0+1)];
    float c01 = coarse[(J0+1)*nx_c + I0];
    float c11 = coarse[(J0+1)*nx_c + (I0+1)];
    float v = (1-a)*(1-b)*c00 + a*(1-b)*c10 + (1-a)*b*c01 + a*b*c11;
    fine[j*nx_f + i] += v;
}

__kernel void prolong3d_add(__global const float* coarse,
                            __global float* fine,
                            int nx_f,int ny_f,int nz_f,
                            int nx_c,int ny_c,int nz_c)
{
    int i = get_global_id(0);
    int j = get_global_id(1);
    int k = get_global_id(2);
    if(i<=0||j<=0||k<=0||i>=nx_f-1||j>=ny_f-1||k>=nz_f-1) return;
    float I = 0.5f*i, J = 0.5f*j, K = 0.5f*k;
    int I0=(int)floor(I), J0=(int)floor(J), K0=(int)floor(K);
    float a=I-I0, b=J-J0, c=K-K0;
    int sxy = nx_c; int sxz = nx_c*ny_c;
    float c000 = coarse[(K0*ny_c + J0)*nx_c + I0];
    float c100 = coarse[(K0*ny_c + J0)*nx_c + (I0+1)];
    float c010 = coarse[(K0*ny_c + (J0+1))*nx_c + I0];
    float c110 = coarse[(K0*ny_c + (J0+1))*nx_c + (I0+1)];
    float c001 = coarse[((K0+1)*ny_c + J0)*nx_c + I0];
    float c101 = coarse[((K0+1)*ny_c + J0)*nx_c + (I0+1)];
    float c011 = coarse[((K0+1)*ny_c + (J0+1))*nx_c + I0];
    float c111 = coarse[((K0+1)*ny_c + (J0+1))*nx_c + (I0+1)];
    float v =
        (1-a)*(1-b)*(1-c)*c000 + a*(1-b)*(1-c)*c100 + (1-a)*b*(1-c)*c010 + a*b*(1-c)*c110 +
        (1-a)*(1-b)*c*c001 + a*(1-b)*c*c101 + (1-a)*b*c*c011 + a*b*c*c111;
    fine[(k*ny_f + j)*nx_f + i] += v;
}
