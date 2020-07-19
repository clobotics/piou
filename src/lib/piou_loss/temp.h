#ifndef PIXEL_WEIGHTS_TEMP_H
#define PIXEL_WEIGHTS_TEMP_H

#include <ATen/ATen.h>
#include <vector>

at::Tensor pixel_weights_forward_cpu(const at::Tensor &loc_p,   //
                                                const at::Tensor &loc_t,   //
                                                const at::Tensor &grid_xy, //
                                                const int k);

at::Tensor pixel_weights_backward_cpu(const at::Tensor &grads_pious);
#endif //SAS_TEMP_H
