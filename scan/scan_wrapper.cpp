#include <torch/extension.h>

void parallel_scan(torch::Tensor d_out, torch::Tensor d_in, int B, int D, int L);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("parallel_scan", &parallel_scan, "Parallel Prefix Scan");
}

