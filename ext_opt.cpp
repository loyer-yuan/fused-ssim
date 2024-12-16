#include <torch/extension.h>
#include "ssim_opt.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("fusedssim_opt", &fusedssim_opt);
  m.def("fusedssim_opt_backward", &fusedssim_opt_backward);
}
