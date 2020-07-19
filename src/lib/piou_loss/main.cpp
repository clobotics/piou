#include <iostream>
#include <ATen/ATen.h>
#include "temp.h"

using namespace std;
using namespace at;

int main() {
    auto feat = CPU(kFloat).arange(64).view({1, 1, 8, 8});
    cout << feat << endl;
    float roi_data[] = {0, 1.6, 1.6, 9.2, 11.0};
    auto roi = CPU(kFloat).tensorFromBlob(roi_data, {1, 5});
    auto memory = CPU(kInt).zeros({0});
    int64_t pool_h = 2, pool_w = 2;
    double scale = 0.5;
    // auto output = roi_pool_forward_cpu(feat, roi, pool_h, pool_w, scale, memory);
    cout << output << endl;
}