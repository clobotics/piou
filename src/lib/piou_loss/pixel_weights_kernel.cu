#include <ATen/ATen.h>
#include <cfloat>
#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>
#include <torch/extension.h>
using namespace at;

/* ------------------------------begin of the forward--------------------------- */
#define CUDA_1D_KERNEL_LOOP(i, n)                              \
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; \
         i += blockDim.x * gridDim.x)

#define maxn 51
const double eps = 1E-8;

__device__ inline int sig(const float &d)
{
    return (d > eps) - (d < -eps);
}

__device__ inline int point_eq(const float2 &a, const float2 &b)
{
    return sig(a.x - b.x) == 0 && sig(a.y - b.y) == 0;
}

__device__ inline void point_swap(float2 *a, float2 *b)
{
    float2 temp = *a;
    *a = *b;
    *b = temp;
}

__device__ inline void point_reverse(float2 *first, float2 *last)
{
    while ((first != last) && (first != --last))
    {
        point_swap(first, last);
        ++first;
    }
}

__device__ inline float cross(const float2 &o, const float2 &a, const float2 &b)
{ //叉积
    return (a.x - o.x) * (b.y - o.y) - (b.x - o.x) * (a.y - o.y);
}
__device__ inline float area(float2 *ps, const int &n)
{
    ps[n] = ps[0];
    float res = 0;
    for (int i = 0; i < n; i++)
    {
        res += ps[i].x * ps[i + 1].y - ps[i].y * ps[i + 1].x;
    }
    return res / 2.0;
}
__device__ inline int lineCross(const float2 &a, const float2 &b, const float2 &c, const float2 &d, float2 &p)
{
    float s1, s2;
    s1 = cross(a, b, c);
    s2 = cross(a, b, d);
    if (sig(s1) == 0 && sig(s2) == 0)
        return 2;
    if (sig(s2 - s1) == 0)
        return 0;
    p.x = (c.x * s2 - d.x * s1) / (s2 - s1);
    p.y = (c.y * s2 - d.y * s1) / (s2 - s1);
    return 1;
}
__device__ inline void polygon_cut(float2 *p, int &n, const float2 &a, const float2 &b, float2 *pp)
{

    int m = 0;
    p[n] = p[0];
    for (int i = 0; i < n; i++)
    {
        if (sig(cross(a, b, p[i])) > 0)
            pp[m++] = p[i];
        if (sig(cross(a, b, p[i])) != sig(cross(a, b, p[i + 1])))
            lineCross(a, b, p[i], p[i + 1], pp[m++]);
    }
    n = 0;
    for (int i = 0; i < m; i++)
        if (!i || !(point_eq(pp[i], pp[i - 1])))
            p[n++] = pp[i];
    // while(n>1&&p[n-1]==p[0])n--;
    while (n > 1 && point_eq(p[n - 1], p[0]))
        n--;
}

//---------------华丽的分隔线-----------------//
//返回三角形oab和三角形ocd的有向交面积,o是原点//
__device__ inline float intersectArea(float2 a, float2 b, float2 c, float2 d)
{
    float2 o = make_float2(0, 0);
    int s1 = sig(cross(o, a, b));
    int s2 = sig(cross(o, c, d));
    if (s1 == 0 || s2 == 0)
        return 0.0; //退化，面积为0
    // if (s1 == -1)
    //     point_swap(&a, &b);
    if (s2 == -1)
        point_swap(&c, &d);
    float2 p[10] = {o, a, b};

    if (s1 == -1)
    {
        p[1] = b;
        p[2] = a;
    }
    int n = 3;
    float2 pp[maxn];
    polygon_cut(p, n, o, c, pp);
    polygon_cut(p, n, c, d, pp);
    polygon_cut(p, n, d, o, pp);

    float res = fabs(area(p, n));
    if (s1 * s2 == -1)
        res = -res;
    return res;
}
//求两多边形的交面积
__device__ inline float intersectArea(float2 *ps1, const int &n1, float2 *ps2, const int &n2)
{
    if (area(ps1, n1) < 0)
        point_reverse(ps1, ps1 + n1);
    if (area(ps2, n2) < 0)
        point_reverse(ps2, ps2 + n2);
    ps1[n1] = ps1[0];
    ps2[n2] = ps2[0];
    float res = 0;
    for (int i = 0; i < n1; i++)
    {
        for (int j = 0; j < n2; j++)
        {
            res += intersectArea(ps1[i], ps1[i + 1], ps2[j], ps2[j + 1]);
        }
    }
    return res; //assumeresispositive!
}

__device__ inline float devPolyIoU(float const *const p, float const *const q)
{
    float2 ps1[maxn], ps2[maxn];
    int n1 = 4;
    int n2 = 4;
    for (int i = 0; i < 4; i++)
    {
        ps1[i].x = p[i * 2];
        ps1[i].y = p[i * 2 + 1];

        ps2[i].x = q[i * 2];
        ps2[i].y = q[i * 2 + 1];
    }
    float inter_area = intersectArea(ps1, n1, ps2, n2);
    float union_area = fabs(area(ps1, n1)) + fabs(area(ps2, n2)) - inter_area;
    return inter_area / (union_area + 1e-9);
}

__device__ void get_pixel_area(const float &cx, const float &cy, const float &w, const float &h, const float &angle, const float &x, const float &y, const int &k, float &pixel_weights, float *grad_loc=NULL)
{
    const float dx = cx - x;
    const float dy = cy - y;
    const float xx = dx * dx;
    const float yy = dy * dy;
    const float dis = sqrt(xx + yy) + 1e-9;
    const float rate = dx / dis;
    const float cos_v = acos(rate);
    const float a1 = angle + cos_v;
    const float a2 = angle - cos_v;
    float a = (cy > y) ? a1 : a2;
    const float cos_a = cos(a);
    const float sin_a = sin(a);
    const float dis_w = dis * fabs(cos_a);
    const float dis_h = dis * fabs(sin_a);
    float factor_h = -k * (dis_h - h / 2.);
    factor_h = (factor_h > 50) ? 50 : (factor_h < -50) ? -50 : factor_h;
    float factor_w = -k * (dis_w - w / 2.);
    factor_w = (factor_w > 50) ? 50 : (factor_w < -50) ? -50 : factor_w;
    const float kerner_h = 1.0 / (exp(factor_h) + 1.0);
    const float kerner_w = 1.0 / (exp(factor_w) + 1.0);
    const float pixel_area = (1 - kerner_h) * (1 - kerner_w);
    pixel_weights = pixel_area;
    if (grad_loc == NULL) return;
    if (pixel_area < 1e-9) return;
    if (kerner_w < 1e-9 && kerner_h < 1e-9) return;
    const float dx_sin_a = dx * sin_a;
    const float dy_sin_a = dy * sin_a;
    const float dx_cos_a = dx * cos_a;
    const float dy_cos_a = dy * cos_a;

    float dis_w_dcx = (dy_sin_a + dx_cos_a) / dis;
    float dis_w_dcy = (dy_cos_a - dx_sin_a) / dis;
    float dis_w_da = -dis * sin_a;

    float dis_h_dcx = (dx_sin_a - dy_cos_a) / dis;
    float dis_h_dcy = (dy_sin_a + dx_cos_a) / dis;
    float dis_h_da = dis * cos_a;

    if (cos_a < 0.0)
    {
        dis_w_dcx = -dis_w_dcx;
        dis_w_dcy = -dis_w_dcy;
        dis_w_da = -dis_w_da;
    }
    if (sin_a < 0.0)
    {
        dis_h_dcx = -dis_h_dcx;
        dis_h_dcy = -dis_h_dcy;
        dis_h_da = -dis_h_da;
    }

    const float fw_dcx = - k * dis_w_dcx;
    const float fw_dcy = - k * dis_w_dcy;
    const float fw_da = - k * dis_w_da;
    const float fw_dw = k / 2.;

    const float fh_dcx = - k * dis_h_dcx;
    const float fh_dcy = - k * dis_h_dcy;
    const float fh_da = - k * dis_h_da;
    const float fh_dh = k / 2.;
    const float kwp = kerner_w * pixel_area;
    const float khp = kerner_h * pixel_area;
    const float p_dcx = kwp * fw_dcx + khp * fh_dcx;
    const float p_dcy = kwp * fw_dcy + khp * fh_dcy;
    const float p_dw = kwp * fw_dw;
    const float p_dh = khp * fh_dh;
    const float p_da = kwp * fw_da + khp * fh_da;
    grad_loc[0] = p_dcx;
    grad_loc[1] = p_dcy;
    grad_loc[2] = p_dw;
    grad_loc[3] = p_dh;
    grad_loc[4] = p_da;
}

__device__ inline float get_pixel_area_fast(const float &cx, const float &cy, const float &w, const float &h, const float &angle, const float &x, const float &y)
{
    const float dx = cx - x;
    const float dy = cy - y;
    const float xx = dx * dx;
    const float yy = dy * dy;
    const float dis = sqrt(xx + yy) + 1e-9;
    const float rate = dx / dis;
    const float cos_v = acos(rate);
    const float a1 = angle + cos_v;
    const float a2 = angle - cos_v;
    float a = (cy > y) ? a1 : a2;
    const float cos_a = cos(a);
    const float sin_a = sin(a);
    const float dis_w = dis * fabs(cos_a);
    const float dis_h = dis * fabs(sin_a);
    if ((dis_h < h / 2.) && (dis_w < w / 2.)) return 1.0;
    else return 0.0;
}

__device__ void rbox2corners_y(const float &cx, const float &cy, const float &w, const float &h, const float &angle, float *grid_data)
{
    const float w_sin = 0.5 * w * sin(angle);
    const float h_cos = 0.5 * h * cos(angle);
    const float y0 = cy - w_sin + h_cos;
    const float y1 = cy + w_sin + h_cos;
    const float y2 = cy + w_sin - h_cos;
    const float y3 = cy - w_sin - h_cos;
    const float ymin = min(y0, min(y1, min(y2, y3)));
    const float ymax = max(y0, max(y1, max(y2, y3)));
    grid_data[0] = ymin;
    grid_data[1] = ymax;
}
// loc_p_data: Nx5
// loc_t_data: Nx5
__global__ void get_grid_forward_kernel(
    const int nthreads,
    const float *loc_p_data,
    const float *loc_t_data,
    float *grid_data)
{
    CUDA_1D_KERNEL_LOOP(index, nthreads)
    {
        loc_p_data += index * 5;
        loc_t_data += index * 5;
        grid_data += index * 2;
        float grid_y_p[2] = {0};
        float grid_y_t[2] = {0};
        rbox2corners_y(loc_p_data[0], loc_p_data[1], loc_p_data[2], loc_p_data[3], loc_p_data[4], grid_y_p);
        rbox2corners_y(loc_t_data[0], loc_t_data[1], loc_t_data[2], loc_t_data[3], loc_t_data[4], grid_y_t);
        grid_data[0] = (float)min(grid_y_p[0], grid_y_t[0]);
        grid_data[1] = (float)max(grid_y_p[1], grid_y_t[1]);
    }
}

// loc_p_data: Nx3
// loc_t_data: Nx5
__global__ void get_grid_share_center_forward_kernel(
    const int nthreads,
    const float *loc_p_data,
    const float *loc_t_data,
    float *grid_data)
{
    CUDA_1D_KERNEL_LOOP(index, nthreads)
    {
        loc_p_data += index * 3;
        loc_t_data += index * 5;
        grid_data += index * 2;
        float grid_y_p[2] = {0};
        float grid_y_t[2] = {0};
        rbox2corners_y(loc_t_data[0], loc_t_data[1], loc_p_data[0], loc_p_data[1], loc_p_data[2], grid_y_p);
        rbox2corners_y(loc_t_data[0], loc_t_data[1], loc_t_data[2], loc_t_data[3], loc_t_data[4], grid_y_t);
        grid_data[0] = (float)min(grid_y_p[0], grid_y_t[0]);
        grid_data[1] = (float)max(grid_y_p[1], grid_y_t[1]);
    }
}
// loc_p_data: N x 5
// loc_t_data: N x 5
// grid_x_data: N x 1
// grid_y_data: N x 2
__global__ void pixel_weights_forward_kernel(
    const int nthreads,
    const float *loc_p_data,
    const float *loc_t_data,
    const float *grid_x_data,
    const float *grid_y_data,
    const int k, const int num,
    const int dim,
    float *inter_union_data,
    float *grad_pixel_weights_data)
{
    CUDA_1D_KERNEL_LOOP(index, nthreads)
    {
        // locate batch
        const int n = index / dim;
        const int d = index % dim;
        loc_p_data += 5 * n;
        loc_t_data += 5 * n;
        grid_y_data += 2 * n;
        inter_union_data += 2 * (n * dim + d);
        grad_pixel_weights_data += 10 * (n * dim + d);
        int ymin = grid_y_data[0] - 5;
        int ymax = grid_y_data[1] + 5;
        ymin = (ymin < -100) ? -100 : ymin;
        ymax = (ymax > 1000) ? 1000 : ymax;
        float grad_cx_1 = 0.0;
        float grad_cy_1 = 0.0;
        float grad_w_1 = 0.0;
        float grad_h_1 = 0.0;
        float grad_angle_1 = 0.0;
        float grad_cx_2 = 0.0;
        float grad_cy_2 = 0.0;
        float grad_w_2 = 0.0;
        float grad_h_2 = 0.0;
        float grad_angle_2 = 0.0;
        // sum grad
        // sum area
        float grad_loc[5] = {0};
        for (int i = ymin; i <= ymax; i++)
        {
            const float y = i + 0.5;
            float pixel_weight_p = 0.0;
            float pixel_weight_t = 0.0;
            get_pixel_area(loc_p_data[0], loc_p_data[1], loc_p_data[2], loc_p_data[3], loc_p_data[4], grid_x_data[d], y, k, pixel_weight_p, grad_loc);
            get_pixel_area(loc_t_data[0], loc_t_data[1], loc_t_data[2], loc_t_data[3], loc_t_data[4], grid_x_data[d], y, k, pixel_weight_t);
            if (pixel_weight_p < 1e-9 && pixel_weight_t < 1e-9)
                continue;
            const float inter_pixel_area = pixel_weight_p * pixel_weight_t;
            const float union_pixel_area = pixel_weight_p + pixel_weight_t - inter_pixel_area;
            grad_cx_1 += pixel_weight_t * grad_loc[0];
            grad_cy_1 += pixel_weight_t * grad_loc[1];
            grad_w_1 += pixel_weight_t * grad_loc[2];
            grad_h_1 += pixel_weight_t * grad_loc[3];
            grad_angle_1 += pixel_weight_t * grad_loc[4];
            grad_cx_2 += grad_loc[0];
            grad_cy_2 += grad_loc[1];
            grad_w_2 += grad_loc[2];
            grad_h_2 += grad_loc[3];
            grad_angle_2 += grad_loc[4];
            inter_union_data[0] += inter_pixel_area;
            inter_union_data[1] += union_pixel_area;
        }
        grad_pixel_weights_data[0] = grad_cx_1;
        grad_pixel_weights_data[1] = grad_cy_1;
        grad_pixel_weights_data[2] = grad_w_1;
        grad_pixel_weights_data[3] = grad_h_1;
        grad_pixel_weights_data[4] = grad_angle_1;
        grad_pixel_weights_data[5] = grad_cx_2;
        grad_pixel_weights_data[6] = grad_cy_2;
        grad_pixel_weights_data[7] = grad_w_2;
        grad_pixel_weights_data[8] = grad_h_2;
        grad_pixel_weights_data[9] = grad_angle_2;
    }
}

// loc_p_data: N x 3
// loc_t_data: N x 5
// grid_x_data: N x 1
// grid_y_data: N x 2
__global__ void pixel_weights_share_center_forward_kernel(
    const int nthreads,
    const float *loc_p_data,
    const float *loc_t_data,
    const float *grid_x_data,
    const float *grid_y_data,
    const int k, const int num,
    const int dim,
    float *inter_union_data,
    float *grad_pixel_weights_data)
{
    CUDA_1D_KERNEL_LOOP(index, nthreads)
    {
        // locate batch
        const int n = index / dim;
        const int d = index % dim;
        loc_p_data += 3 * n;
        loc_t_data += 5 * n;
        grid_y_data += 2 * n;
        inter_union_data += 2 * (n * dim + d);
        grad_pixel_weights_data += 10 * (n * dim + d);
        int ymin = grid_y_data[0] - 5;
        int ymax = grid_y_data[1] + 5;
        ymin = (ymin < -100) ? -100 : ymin;
        ymax = (ymax > 1000) ? 1000 : ymax;
        float grad_cx_1 = 0.0;
        float grad_cy_1 = 0.0;
        float grad_w_1 = 0.0;
        float grad_h_1 = 0.0;
        float grad_angle_1 = 0.0;
        float grad_cx_2 = 0.0;
        float grad_cy_2 = 0.0;
        float grad_w_2 = 0.0;
        float grad_h_2 = 0.0;
        float grad_angle_2 = 0.0;
        // sum grad
        // sum area
        float grad_loc[5] = {0};
        for (int i = ymin; i <= ymax; i++)
        {
            const float y = i + 0.5;
            float pixel_weight_p = 0.0;
            float pixel_weight_t = 0.0;
            get_pixel_area(loc_t_data[0], loc_t_data[1], loc_p_data[0], loc_p_data[1], loc_p_data[2], grid_x_data[d], y, k, pixel_weight_p, grad_loc);
            get_pixel_area(loc_t_data[0], loc_t_data[1], loc_t_data[2], loc_t_data[3], loc_t_data[4], grid_x_data[d], y, k, pixel_weight_t);
            if (pixel_weight_p < 1e-9 && pixel_weight_t < 1e-9)
                continue;
            const float inter_pixel_area = pixel_weight_p * pixel_weight_t;
            const float union_pixel_area = pixel_weight_p + pixel_weight_t - inter_pixel_area;
            grad_cx_1 += pixel_weight_t * grad_loc[0];
            grad_cy_1 += pixel_weight_t * grad_loc[1];
            grad_w_1 += pixel_weight_t * grad_loc[2];
            grad_h_1 += pixel_weight_t * grad_loc[3];
            grad_angle_1 += pixel_weight_t * grad_loc[4];
            grad_cx_2 += grad_loc[0];
            grad_cy_2 += grad_loc[1];
            grad_w_2 += grad_loc[2];
            grad_h_2 += grad_loc[3];
            grad_angle_2 += grad_loc[4];
            inter_union_data[0] += inter_pixel_area;
            inter_union_data[1] += union_pixel_area;
        }
        grad_pixel_weights_data[0] = grad_cx_1;
        grad_pixel_weights_data[1] = grad_cy_1;
        grad_pixel_weights_data[2] = grad_w_1;
        grad_pixel_weights_data[3] = grad_h_1;
        grad_pixel_weights_data[4] = grad_angle_1;
        grad_pixel_weights_data[5] = grad_cx_2;
        grad_pixel_weights_data[6] = grad_cy_2;
        grad_pixel_weights_data[7] = grad_w_2;
        grad_pixel_weights_data[8] = grad_h_2;
        grad_pixel_weights_data[9] = grad_angle_2;
    }
}

// loc_p_data: N x 5
// loc_t_data: N x 5
// grid_x_data: N x 1
// grid_y_data: N x 2
__global__ void hpixel_weights_forward_kernel(
    const int nthreads,
    const float *loc_p_data,
    const float *loc_t_data,
    const float *grid_x_data,
    const float *grid_y_data,
    const int k, const int num,
    const int dim,
    float *inter_data,
    float *grad_pixel_weights_data)
{
    CUDA_1D_KERNEL_LOOP(index, nthreads)
    {
        // locate batch
        const int n = index / dim;
        const int d = index % dim;
        loc_p_data += 5 * n;
        loc_t_data += 5 * n;
        grid_y_data += 2 * n;
        inter_data += (n * dim + d);
        grad_pixel_weights_data += 5 * (n * dim + d);
        int ymin = grid_y_data[0] - 5;
        int ymax = grid_y_data[1] + 5;
        ymin = (ymin < -100) ? -100 : ymin;
        ymax = (ymax > 1000) ? 1000 : ymax;
        float grad_cx = 0.0;
        float grad_cy = 0.0;
        float grad_w = 0.0;
        float grad_h = 0.0;
        float grad_angle = 0.0;
        // sum grad
        // sum area
        float grad_loc[5] = {0};
        for (int i = ymin; i <= ymax; i++)
        {
            const float y = i + 0.5;
            float pixel_weight_p = 0.0;
            float pixel_weight_t = 0.0;
            get_pixel_area(loc_p_data[0], loc_p_data[1], loc_p_data[2], loc_p_data[3], loc_p_data[4], grid_x_data[d], y, k, pixel_weight_p, grad_loc);
            get_pixel_area(loc_t_data[0], loc_t_data[1], loc_t_data[2], loc_t_data[3], loc_t_data[4], grid_x_data[d], y, k, pixel_weight_t);
            if (pixel_weight_p < 1e-9 && pixel_weight_t < 1e-9)
                continue;
            const float inter_pixel_area = pixel_weight_p * pixel_weight_t;
            grad_cx += pixel_weight_t * grad_loc[0];
            grad_cy += pixel_weight_t * grad_loc[1];
            grad_w += pixel_weight_t * grad_loc[2];
            grad_h += pixel_weight_t * grad_loc[3];
            grad_angle += pixel_weight_t * grad_loc[4];
            inter_data[0] += inter_pixel_area;
        }
        grad_pixel_weights_data[0] = grad_cx;
        grad_pixel_weights_data[1] = grad_cy;
        grad_pixel_weights_data[2] = grad_w;
        grad_pixel_weights_data[3] = grad_h;
        grad_pixel_weights_data[4] = grad_angle;
    }
}
// // loc_p_data: N x 8
// // loc_t_data: M x 8
// __global__ void overlap_r_forward_kernel(
//     const int nthreads,
//     const float *loc_p_data,
//     const float *loc_t_data,
//     const int dim,
//     float *pious_data)
// {
//     CUDA_1D_KERNEL_LOOP(index, nthreads)
//     {
//         // locate batch
//         const int n = index / dim;
//         const int d = index % dim;
//         loc_p_data += 8 * n;
//         loc_t_data += 8 * d;
//         pious_data += (n * dim + d);
//         const float xmin_p = min(loc_p_data[0], min(loc_p_data[2], min(loc_p_data[4], loc_p_data[6])));
//         const float ymin_p = min(loc_p_data[1], min(loc_p_data[3], min(loc_p_data[5], loc_p_data[7])));
//         const float xmax_p = max(loc_p_data[0], max(loc_p_data[2], max(loc_p_data[4], loc_p_data[6])));
//         const float ymax_p = max(loc_p_data[1], max(loc_p_data[3], max(loc_p_data[5], loc_p_data[7])));

//         const float xmin_t = min(loc_t_data[0], min(loc_t_data[2], min(loc_t_data[4], loc_t_data[6])));
//         const float ymin_t = min(loc_t_data[1], min(loc_t_data[3], min(loc_t_data[5], loc_t_data[7])));
//         const float xmax_t = max(loc_t_data[0], max(loc_t_data[2], max(loc_t_data[4], loc_t_data[6])));
//         const float ymax_t = max(loc_t_data[1], max(loc_t_data[3], max(loc_t_data[5], loc_t_data[7])));
//         float iou = 0.0;
//         if (xmin_p > xmax_t || xmax_p < xmin_t || ymin_p > ymax_t || ymax_p < ymin_t)
//         {
//             iou = 0.0;
//         } else {
//             const float inter_xmin = max(xmin_p, xmin_t);
//             const float inter_ymin = max(ymin_p, ymin_t);
//             const float inter_xmax = min(xmax_p, xmax_t);
//             const float inter_ymax = min(ymax_p, ymax_t);
//             const float inter_area = (inter_ymax - inter_ymin) * (inter_xmax - inter_xmin);
//             const float area_p = (ymax_p - ymin_p) * (xmax_p - xmin_p);
//             const float area_t = (ymax_t - ymin_t) * (xmax_t - xmin_t);
//             iou = inter_area / (area_p + area_t - inter_area);
//         }
//         // pious_data[0] = iou;
//         if (iou < 0.25)
//         {
//             pious_data[0] = iou;
//         } else {
//             const float union_xmin = min(xmin_p, xmin_t);
//             const float union_ymin = min(ymin_p, ymin_t);
//             const float union_xmax = max(xmax_p, xmax_t);
//             const float union_ymax = max(ymax_p, ymax_t);

//             int ymin = grid_y_data[0] - 5;
//             int ymax = grid_y_data[1] + 5;
//             ymin = (ymin < -100) ? -100 : ymin;
//             ymax = (ymax > 1000) ? 1000 : ymax;

//             pious_data[0] = devPolyIoU(loc_p_data, loc_t_data);
//         }
        
//     }
// }
// loc_p_data: N x 5
// loc_t_data: M x 5
// __global__ void overlap_r_fast_forward_kernel(
//     const int nthreads,
//     const float *loc_p_data,
//     const float *loc_t_data,
//     const int dim,
//     float *pious_data)
// {
//     CUDA_1D_KERNEL_LOOP(index, nthreads)
//     {
//         // locate batch
//         const int n = index / dim;
//         const int d = index % dim;
//         loc_p_data += 5 * n;
//         loc_t_data += 5 * d;
//         pious_data += (n * dim + d);
//         // pre_processing
//         const float dis = sqrt(xx * xx + yy * yy);
//         const float max_wh_p = max(loc_p_data[2], loc_p_data[3]);
//         const float max_wh_t = max(loc_t_data[2], loc_t_data[3]);
//         if (dis > 0.707107 * (max_wh_p + max_wh_t))
//         {
//             pious_data[0] = 0.0;
//             return;
//         } else {
//             const float xmin_p = loc_p_data[0] - loc_p_data[2];
//             const float ymin_p = loc_p_data[1] - loc_p_data[3];
//             const float xmax_p = loc_p_data[0] + loc_p_data[2];
//             const float ymax_p = loc_p_data[1] + loc_p_data[3];
//             const float xmin_t = loc_t_data[0] - loc_t_data[2];
//             const float ymin_t = loc_t_data[1] - loc_t_data[3];
//             const float xmax_t = loc_t_data[0] + loc_t_data[2];
//             const float ymax_t = loc_t_data[1] + loc_t_data[3];

//             float iou = 0.0;
//             if (xmin_p > xmax_t || xmax_p < xmin_t || ymin_p > ymax_t || ymax_p < ymin_t)
//             {
//                 iou = 0.0;
//             } else {
//                 const float inter_xmin = max(xmin_p, xmin_t);
//                 const float inter_ymin = max(ymin_p, ymin_t);
//                 const float inter_xmax = min(xmax_p, xmax_t);
//                 const float inter_ymax = min(ymax_p, ymax_t);
//                 const float inter_area = (inter_ymax - inter_ymin) * (inter_xmax - inter_xmin);
//                 const float area_p = (ymax_p - ymin_p) * (xmax_p - xmin_p);
//                 const float area_t = (ymax_t - ymin_t) * (xmax_t - xmin_t);
//                 iou = inter_area / (area_p + area_t - inter_area);
//             }
//             const float ar_p = loc_p_data[2] / loc_p_data[3];
//             const float ar_t = loc_t_data[2] / loc_t_data[3];
//             if (ar_p < 1.2 && ar_t < 1.2)
//             {
                
//             } else {

//             }
            
//         }
//     }
// }

// __device__ float OverlapArea(const float &xcenter1, const float &ycenter1, const float &width1, const float &height1, const float &angle1, const float &xcenter2, const float &ycenter2, const float &width2, const float &height2, const float &angle2)
// {
//     float angle1_ = -angle1;
//     float angle2_ = -angle2;
//     float angled = angle2_ - angle1_;
//     angled *= (float)3.14159265 / 180;
//     angle1_ *= (float)3.14159265 / 180;
//     float area = 0;
//     float hw1 = width1 / 2;
//     float hh1 = height1 / 2;
//     float hw2 = width2 / 2;
//     float hh2 = height2 / 2;
//     float xcenterd = xcenter2 - xcenter1;
//     float ycenterd = ycenter2 - ycenter1;
//     float tmp = xcenterd * cosf(angle1_) + ycenterd * sinf(angle1_);
//     ycenterd = -xcenterd * sinf(angle1_) + ycenterd * cosf(angle1_);
//     xcenterd = tmp;
//     float max_width_height1 = width1 > height1 ? width1 : height1;
//     float max_width_height2 = width2 > height2 ? width2 : height2;
//     if (sqrt(xcenterd * xcenterd + ycenterd * ycenterd) >
//         (max_width_height1 + max_width_height2) * 0.707107)
//     {
//         area = 0;
//         return (area);
//     }
//     if (fabs(sin(angled)) < 1e-3)
//     {
//         if (fabs(xcenterd) > (hw1 + hw2) || fabs(ycenterd) > (hh1 + hh2))
//         {
//             area = 0;
//             return (area);
//         }
//         else
//         {
//             float x_min_inter = -hw1 > (xcenterd - hw2) ? -hw1 : (xcenterd - hw2);
//             float x_max_inter = hw1 < (xcenterd + hw2) ? hw1 : (xcenterd + hw2);
//             float y_min_inter = -hh1 > (ycenterd - hh2) ? -hh1 : (ycenterd - hh2);
//             float y_max_inter = hh1 < (ycenterd + hh2) ? hh1 : (ycenterd + hh2);
//             const float inter_width = x_max_inter - x_min_inter;
//             const float inter_height = y_max_inter - y_min_inter;
//             const float inter_size = inter_width * inter_height;
//             area = inter_size;
//             return (area);
//         }
//     }
//     if (fabs(cos(angled)) < 1e-3)
//     {
//         float x_min_inter = -hw1 > (xcenterd - hh2) ? -hw1 : (xcenterd - hh2);
//         float x_max_inter = hw1 < (xcenterd + hh2) ? hw1 : (xcenterd + hh2);
//         float y_min_inter = -hh1 > (ycenterd - hw2) ? -hh1 : (ycenterd - hw2);
//         float y_max_inter = hh1 < (ycenterd + hw2) ? hh1 : (ycenterd + hw2);
//         const float inter_width = x_max_inter - x_min_inter;
//         const float inter_height = y_max_inter - y_min_inter;
//         const float inter_size = inter_width * inter_height;
//         area = inter_size;
//         return (area);
//     }
// }
//     __global__ void overlap_r_fast_forward_kernel(
//         const int nthreads,
//         const float *loc_p_data,
//         const float *loc_t_data,
//         const int dim,
//         float *pious_data)
//     {
//         CUDA_1D_KERNEL_LOOP(index, nthreads)
//         {
//             // locate batch
//             const int n = index / dim;
//             const int d = index % dim;
//             loc_p_data += 5 * n;
//             loc_t_data += 5 * d;
//             pious_data += (n * dim + d);
//             const float xmin = grid_data[0];
//             const float ymin = grid_data[1];
//             const float xmax = grid_data[2];
//             const float ymax = grid_data[3];
//             const float overlap_h = grid_data[4];
//             const float area_p = loc_p_data[2] * loc_p_data[3];
//             const float area_t = loc_t_data[2] * loc_t_data[3];
//             const float xx = loc_p_data[0] - loc_t_data[0];
//             const float yy = loc_p_data[1] - loc_t_data[1];

//             // pre_processing
//             const float dis = sqrt(xx * xx + yy * yy);
//             const float max_wh_p = max(loc_p_data[2], loc_p_data[3]);
//             const float max_wh_t = max(loc_t_data[2], loc_t_data[3]);
//             if (dis > 0.707107 * (max_wh_p + max_wh_t))
//             {
//                 pious_data[0] = 0.0;
//                 return;
//             }
//             pious_data[0] = overlap_h;
//             return;
//             if (overlap_h < 0.1)
//             {
//                 pious_data[0] = overlap_h;
//                 return;
//             }
//             else
//             {
//             }
//         }
//     }

    // inter_union_data: Nxdimx2
    // grad_pixel_weights_data: Nxdimx10
    // pious_data:Nx1
    // grad_loc_memory_data:Nx5
    __global__ void pious_forward_kernel(
        const int nthreads,
        const float *inter_union_data,
        const float *grad_pixel_weights_data,
        const int num,
        const int dim,
        float *pious_data,
        float *grad_loc_memory_data)
    {
        CUDA_1D_KERNEL_LOOP(index, nthreads)
        {
            // locate batch
            inter_union_data += index * dim * 2;
            grad_pixel_weights_data += index * dim * 10;
            grad_loc_memory_data += index * 5;
            float inter_area = 0.0;
            float union_area = 0.0;
            float grad_cx_1 = 0.0;
            float grad_cy_1 = 0.0;
            float grad_w_1 = 0.0;
            float grad_h_1 = 0.0;
            float grad_angle_1 = 0.0;
            float grad_cx_2 = 0.0;
            float grad_cy_2 = 0.0;
            float grad_w_2 = 0.0;
            float grad_h_2 = 0.0;
            float grad_angle_2 = 0.0;
            for (int d = 0; d < dim; d++)
            {
                const int offset = 2 * d;
                const int offset_ = 10 * d;
                grad_cx_1 += grad_pixel_weights_data[offset_];
                grad_cy_1 += grad_pixel_weights_data[offset_ + 1];
                grad_w_1 += grad_pixel_weights_data[offset_ + 2];
                grad_h_1 += grad_pixel_weights_data[offset_ + 3];
                grad_angle_1 += grad_pixel_weights_data[offset_ + 4];
                grad_cx_2 += grad_pixel_weights_data[offset_ + 5];
                grad_cy_2 += grad_pixel_weights_data[offset_ + 6];
                grad_w_2 += grad_pixel_weights_data[offset_ + 7];
                grad_h_2 += grad_pixel_weights_data[offset_ + 8];
                grad_angle_2 += grad_pixel_weights_data[offset_ + 9];
                inter_area += inter_union_data[offset];
                union_area += inter_union_data[offset + 1];
            }
            pious_data[index] = inter_area / (union_area + 1e-9);
            const float k = inter_area + union_area;
            const float b = union_area * union_area + 1e-9;
            grad_loc_memory_data[0] = (k * grad_cx_1 - inter_area * grad_cx_2) / b;
            grad_loc_memory_data[1] = (k * grad_cy_1 - inter_area * grad_cy_2) / b;
            grad_loc_memory_data[2] = (k * grad_w_1 - inter_area * grad_w_2) / b;
            grad_loc_memory_data[3] = (k * grad_h_1 - inter_area * grad_h_2) / b;
            grad_loc_memory_data[4] = (k * grad_angle_1 - inter_area * grad_angle_2) / b;
        }
    }

    // inter_union_data: Nxdimx2
    // grad_pixel_weights_data: Nxdimx10
    // pious_data:Nx1
    // grad_loc_memory_data:Nx5
    __global__ void hpious_forward_kernel(
        const int nthreads,
        const float *loc_p_data,
        const float *loc_t_data,
        const float *inter_data,
        const float *grad_pixel_weights_data,
        const int num,
        const int dim,
        float *pious_data,
        float *grad_loc_memory_data)
    {
        CUDA_1D_KERNEL_LOOP(index, nthreads)
        {
            // locate batch
            loc_p_data += 5 * index;
            loc_t_data += 5 * index;
            inter_data += index * dim;
            grad_pixel_weights_data += index * dim * 5;
            grad_loc_memory_data += index * 5;
            float inter_area = 0.0;
            float grad_cx = 0.0;
            float grad_cy = 0.0;
            float grad_w = 0.0;
            float grad_h = 0.0;
            float grad_angle = 0.0;
            for (int d = 0; d < dim; d++)
            {
                const int offset = 5 * d;
                grad_cx += grad_pixel_weights_data[offset];
                grad_cy += grad_pixel_weights_data[offset + 1];
                grad_w += grad_pixel_weights_data[offset + 2];
                grad_h += grad_pixel_weights_data[offset + 3];
                grad_angle += grad_pixel_weights_data[offset + 4];
                inter_area += inter_data[d];
            }
            const float union_area = loc_p_data[2] * loc_p_data[3] + loc_t_data[2] * loc_t_data[3] - inter_area;
            pious_data[index] = inter_area / (union_area + 1e-9);
            const float k = inter_area + union_area;
            const float b = union_area * union_area + 1e-9;
            grad_loc_memory_data[0] = k * grad_cx / b;
            grad_loc_memory_data[1] = k * grad_cy / b;
            grad_loc_memory_data[2] = (k * grad_w - loc_p_data[3] * inter_area) / b;
            grad_loc_memory_data[3] = (k * grad_h - loc_p_data[2] * inter_area) / b;
            grad_loc_memory_data[4] = k * grad_angle / b;
        }
    }

    // #loc-- > num x 5
    // #grid_xy-- > dim x 2
    // output-- > num x 1

    at::Tensor pixel_weights_forward_cuda(
        const at::Tensor &loc_p,  //
        const at::Tensor &loc_t,  //
        const at::Tensor &grid_x, //
        const int k,
        const bool is_hard,
        at::Tensor &grad_loc_memory)
    {
        const int num = loc_p.size(0);
        const int dim = grid_x.size(0);
        const int tmp_total_count = num * dim;
        const int total_count = num;
        const int thread_per_block = 1024;
        const int block_count = (total_count + thread_per_block - 1) / thread_per_block;
        const int tmp_block_count = (tmp_total_count + thread_per_block - 1) / thread_per_block;
        // final output
        auto options = torch::TensorOptions().dtype(torch::kFloat32).requires_grad(true).device(torch::kCUDA, loc_p.device().index());
        auto output = torch::zeros({num}, options).to(torch::kCUDA);
        auto pixel_weights = torch::zeros({num, dim}, options).to(torch::kCUDA);
        auto grid_y = torch::zeros({num, 2}, options).to(torch::kCUDA);

        AT_CHECK(loc_p.device().index() == loc_t.device().index(), "loc_p & loc_t must be same device");
        AT_CHECK(loc_p.device().index() == grid_x.device().index(), "loc_p & grid_x must be same device");

        // for paralell
        if (block_count <= 0)
            return output;
        // get grid y-dir
        get_grid_forward_kernel<<<block_count, thread_per_block>>>(
            total_count, loc_p.data<float>(), loc_t.data<float>(), grid_y.data<float>());

        // get_grid_share_center_forward_kernel<<<block_count, thread_per_block>>>(
        //     total_count, loc_p.data<float>(), loc_t.data<float>(), grid_y.data<float>());

        // kernel function for sum
        if (is_hard)
        {
            auto inter = torch::zeros({num, dim}, options).to(torch::kCUDA);
            auto grad_pixel_weights = torch::zeros({num, dim, 5}, options).to(torch::kCUDA);
            hpixel_weights_forward_kernel<<<tmp_block_count, thread_per_block>>>(
                tmp_total_count, loc_p.data<float>(), loc_t.data<float>(), grid_x.data<float>(), grid_y.data<float>(), k, num, dim, inter.data<float>(), grad_pixel_weights.data<float>());
            hpious_forward_kernel<<<block_count, thread_per_block>>>(
                total_count, loc_p.data<float>(), loc_t.data<float>(), inter.data<float>(), grad_pixel_weights.data<float>(), num, dim, output.data<float>(), grad_loc_memory.data<float>());
        }
        else
        {
            auto inter_union = torch::zeros({num, dim, 2}, options).to(torch::kCUDA);
            auto grad_pixel_weights = torch::zeros({num, dim, 10}, options).to(torch::kCUDA);
            // kernel function for pixels
            pixel_weights_forward_kernel<<<tmp_block_count, thread_per_block>>>(
                tmp_total_count, loc_p.data<float>(), loc_t.data<float>(), grid_x.data<float>(), grid_y.data<float>(), k, num, dim, inter_union.data<float>(), grad_pixel_weights.data<float>());
            // pixel_weights_share_center_forward_kernel<<<tmp_block_count, thread_per_block>>>(
            //     tmp_total_count, loc_p.data<float>(), loc_t.data<float>(), grid_x.data<float>(), grid_y.data<float>(), k, num, dim, inter_union.data<float>(), grad_pixel_weights.data<float>());
            pious_forward_kernel<<<block_count, thread_per_block>>>(
                total_count, inter_union.data<float>(), grad_pixel_weights.data<float>(), num, dim, output.data<float>(), grad_loc_memory.data<float>());
        }

        AT_CHECK(cudaGetLastError() == cudaSuccess, "pious_forward_kernel failed");
        return output;
    }

    at::Tensor overlap_r_forward_cuda(
        const at::Tensor &loc_p, //
        const at::Tensor &loc_t,
        const at::Tensor &grid)
    {
        const int num_p = loc_p.size(0);
        const int num_t = loc_t.size(0);
        const int total_count = num_p * num_t;
        const int thread_per_block = 1024;
        const int block_count = (total_count + thread_per_block - 1) / thread_per_block;
        // final output
        auto options = torch::TensorOptions().dtype(torch::kFloat32).requires_grad(true).device(torch::kCUDA, loc_p.device().index());
        auto output = torch::zeros({num_p, num_t}, options).to(torch::kCUDA);
        // for paralell
        if (block_count <= 0)
            return output;
        // // kernel function for sum
        // overlap_r_forward_kernel<<<block_count, thread_per_block>>>(
        //     total_count, loc_p.data<float>(), loc_t.data<float>(), grid.data<float>(), num_t, output.data<float>());G56
        // AT_CHECK(cudaGetLastError() == cudaSuccess, "overlap_r_forward_kernel failed");
        return output;
    }
    /* ------------------------------end of the forward--------------------------- */
    __global__ void pious_backward_kernel(
        const int nthreads,
        const float *grad_pious_data,
        const float *grad_loc_memory_data,
        float *grad_loc_p_data)
    {
        CUDA_1D_KERNEL_LOOP(index, nthreads)
        {
            // locate batch
            const int n = index / 5;
            const int d = index % 5;
            grad_loc_memory_data += n * 5;
            grad_loc_p_data += n * 5;
            grad_loc_p_data[d] = grad_pious_data[n] * grad_loc_memory_data[d];
        }
    }

    at::Tensor pixel_weights_backward_cuda(const at::Tensor &grads_pious, const at::Tensor &grad_loc_memory)
    {
        const int num = grads_pious.size(0);
        const int total_count = 5 * num;
        const int thread_per_block = 1024;
        const int block_count = (total_count + thread_per_block - 1) / thread_per_block;
        auto options = torch::TensorOptions().dtype(torch::kFloat32).requires_grad(true).device(torch::kCUDA, grads_pious.device().index());
        auto grad_loc_p = torch::zeros({num, 5}, options).to(torch::kCUDA);
        pious_backward_kernel<<<block_count, thread_per_block>>>(
            total_count, grads_pious.data<float>(), grad_loc_memory.data<float>(), grad_loc_p.data<float>());
        AT_CHECK(cudaGetLastError() == cudaSuccess, "pious_backward_kernel failed");
        return grad_loc_p;
    }
