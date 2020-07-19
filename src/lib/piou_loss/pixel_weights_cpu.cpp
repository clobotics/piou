#include <ATen/ATen.h>
#include <cfloat>
#include <torch/extension.h>
using std::vector;
using std::max;
using std::min;


/* -----------------------------begin of the forward---------------------------------  */
// #loc-- > num x 5
// #grid_xy-- > dim x 2
// output-- > num x 1
inline float kernel_function(const float factor) 
{
    return 1.0 / (exp(factor) + 1.0);
}

float get_pixel_area(const float &cx, const float &cy, const float &w, const float &h, const float &angle, const float &x, const float &y, const int &k)
{
    const float dx = cx - x;
    const float dy = cy - y;
    const float xx = dx * dx;
    const float yy = dy * dy;
    float dis = sqrt(xx + yy);
    dis = (dis < 1e-9) ? 1e-9 : dis;
    const float cos_v = acos(dx / dis);

    const float a1 = angle + cos_v;
    const float a2 = angle - cos_v;
    float a = (cy > y) ? a1 : a2;

    const float dis_w = dis * fabs(cos(a));
    const float dis_h = dis * fabs(sin(a));
    float factor_h = -k * (dis_h - h / 2.);
    factor_h = (factor_h > 50) ? 50 : (factor_h < -50) ? -50 : factor_h;
    float factor_w = -k * (dis_w - w / 2.);
    factor_w = (factor_w > 50) ? 50 : (factor_w < -50) ? -50 : factor_w;
    const float kerner_h = kernel_function(factor_h);
    const float kerner_w = kernel_function(factor_w);
    const float pixel_area = (1 - kerner_h) * (1 - kerner_w);
    return pixel_area;
}

at::Tensor pixel_weights_forward_cpu(const at::Tensor &loc_p,   //
                                    const at::Tensor &loc_t,   //
                                    const at::Tensor &grid_xy, //
                                    const int k)
{
    AT_CHECK(grid_xy.ndimension() == 2, "Feature should be NxC forms");
    AT_CHECK(loc_t.ndimension() == 2, "Feature should be NxC forms");
    AT_CHECK(loc_p.ndimension() == 2, "Feature should be NxC forms");
    AT_CHECK(grid_xy.is_contiguous(), "Feature should be contiguous");
    AT_CHECK(loc_p.is_contiguous(), "Feature should be contiguous");
    AT_CHECK(loc_t.is_contiguous(), "Feature should be contiguous");
    const int num = loc_p.size(0);
    const int dim = grid_xy.size(0);
    const float *loc_p_data = loc_p.data<float>();
    const float *loc_t_data = loc_t.data<float>();
    const float *grid_xy_data = grid_xy.data<float>();
    auto options = torch::TensorOptions().dtype(torch::kFloat32).requires_grad(false);
    auto pious = torch::zeros({num}, options);
    float *pious_data = pious.data<float>();

    for (int n = 0; n < num; n++)
    {
        float inter_area = 0.0;
        float union_area = 0.0;
        // float area_p = 0.0;
        // float area_t = 0.0;
        for (int d = 0; d < dim; d++)
        {
            const int offset_x = 2 * d;
            const int offset_y = offset_x + 1;
            const float pixel_weight_p = get_pixel_area(loc_p_data[0], loc_p_data[1], loc_p_data[2], loc_p_data[3], loc_p_data[4], grid_xy_data[offset_x], grid_xy_data[offset_y], k);

            const float pixel_weight_t = get_pixel_area(loc_t_data[0], loc_t_data[1], loc_t_data[2], loc_t_data[3], loc_t_data[4], grid_xy_data[offset_x], grid_xy_data[offset_y], k);

            // area_p += pixel_weight_p;
            // area_t += pixel_weight_t;
            if (pixel_weight_p < 1e-9 && pixel_weight_t < 1e-9)
                continue;
            const float inter_pixel_area = pixel_weight_p * pixel_weight_t;
            const float union_pixel_area = pixel_weight_p + pixel_weight_t - inter_pixel_area;
            inter_area += inter_pixel_area;
            union_area += union_pixel_area;
        }
        pious_data[n] = inter_area / (union_area + 1e-9);
        loc_p_data += 5;
        loc_t_data += 5;
    }
    return pious;
}


at::Tensor pixel_weights_backward_cpu(const at::Tensor &grads_pious)

{
    const int num = grads_pious.size(0);
    auto options = torch::TensorOptions().dtype(torch::kFloat32).requires_grad(true);
    auto grads_loc_p = torch::zeros({num, 5}, options);
    return grads_loc_p;
}
