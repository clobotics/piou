#include<iostream>
using namespace std;
void get_pixel_area(const float &cx, const float &cy, const float &w, const float &h, const float &angle, const float &x, const float &y, const int &k, float *pixel_weights, float *grad_loc = NULL)
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
    pixel_weights[0] = pixel_area;
    if (grad_loc == NULL)
        return;
    if (pixel_area < 1e-9)
        return;
    if (kerner_w < 1e-9 && kerner_h < 1e-9)
        return;
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

    const float fw_dcx = -k * dis_w_dcx;
    const float fw_dcy = -k * dis_w_dcy;
    const float fw_da = -k * dis_w_da;
    const float fw_dw = k / 2.;

    const float fh_dcx = -k * dis_h_dcx;
    const float fh_dcy = -k * dis_h_dcy;
    const float fh_da = -k * dis_h_da;
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

int main(int argc, char const *argv[])
{
    float aa = 1e-9;
    std::cout << "aa: " << aa << std::endl;
    return 0;
}
