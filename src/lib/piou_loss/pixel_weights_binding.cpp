#include <torch/torch.h>
#include "pixel_weights_cpu.cpp"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward_cpu", &pixel_weights_forward_cpu, "pixel_weights_forward_cpu");
    m.def("backward_cpu", &pixel_weights_backward_cpu, "pixel_weights_backward_cpu");
}