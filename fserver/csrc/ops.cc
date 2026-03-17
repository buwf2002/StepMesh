/* Copyright (c) 2025, StepFun Authors. All rights reserved. */


#include "./util.hpp"

#include "./public.hpp"
#ifdef DMLC_USE_CUDA
  #include "./private.hpp"
  #include "./kernel.hpp"
#endif

#ifdef DMLC_USE_HIP
  #include "./private_hip.hpp"
  #include "./kernel_hip.hpp"
#endif

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  pybind_public(m);
#if defined(DMLC_USE_CUDA) || defined(DMLC_USE_HIP)
  pybind_private(m);
  pybind_kernel(m);
#endif
}
