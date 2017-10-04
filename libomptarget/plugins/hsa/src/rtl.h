//===----RTLs/hsa/src/rtl.h   - Target RTLs Implementation ------- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.txt for details.
//
//===----------------------------------------------------------------------===//
//
// RTL for HSA machine
//
//===----------------------------------------------------------------------===//

// To be shared with device, i.e. amdgcn deviceRTL

struct omptarget_device_environmentTy {
  int32_t num_devices;
  int32_t device_num;
  int32_t debug_mode;
};

