//===----RTLs/cuda/src/rtl.h   - Target RTLs Implementation ------- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.txt for details.
//
//===----------------------------------------------------------------------===//
//
// RTL for CUDA machine
//
//===----------------------------------------------------------------------===//

// To be shared with device, i.e. nvptx deviceRTL

struct omptarget_device_environmentTy {
  int32_t num_devices;
  int32_t device_num;
  int32_t debug_mode;
};

