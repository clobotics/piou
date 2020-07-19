#include <torch/torch.h>
#include <torch/extension.h>
// CUDA forward declarations
at::Tensor pixel_weights_forward_cuda(
    const at::Tensor &loc_p,   //
    const at::Tensor &loc_t,   //
    const at::Tensor &grid, //
    const int k,
    const bool is_hard,
    at::Tensor &grad_loc_memory);

at::Tensor overlap_r_forward_cuda(
    const at::Tensor &loc_p, //
    const at::Tensor &loc_t,
    const at::Tensor &grid);

at::Tensor pixel_weights_backward_cuda(const at::Tensor &grads_pious, const at::Tensor &grad_loc_memory);

// C++ interface
#define CHECK_CUDA(x) AT_CHECK(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

at::Tensor pixel_weights_forward(
    const at::Tensor &loc_p,   //
    const at::Tensor &loc_t,   //
    const at::Tensor &grid, //
    const int k,
    const bool is_hard,
    at::Tensor &grad_loc_memory)
{
    CHECK_INPUT(grid);
    CHECK_INPUT(loc_p);
    CHECK_INPUT(loc_t);
    return pixel_weights_forward_cuda(loc_p, loc_t, grid, k, is_hard, grad_loc_memory);
}

at::Tensor overlap_r_forward(
    const at::Tensor &loc_p, //
    const at::Tensor &loc_t,
    const at::Tensor &grid)
{
    CHECK_INPUT(grid);
    CHECK_INPUT(loc_p);
    CHECK_INPUT(loc_t);
    return overlap_r_forward_cuda(loc_p, loc_t, grid);
}

at::Tensor pixel_weights_backward(const at::Tensor &grads_pious, const at::Tensor &grad_loc_memory)

{
    CHECK_INPUT(grads_pious);
    return pixel_weights_backward_cuda(grads_pious, grad_loc_memory);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward_cuda", &pixel_weights_forward, "pixel_weights_forward_cuda");
    m.def("overlap_r_cuda", &overlap_r_forward, "overlap_r_forward_cuda");
    m.def("backward_cuda", &pixel_weights_backward, "pixel_weights_backward_cuda");
}