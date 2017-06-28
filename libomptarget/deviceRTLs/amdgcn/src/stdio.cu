//===------------- stdio.cu - NVPTX OpenMP Std I/O --------------- CUDA -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.txt for details.
//
//===----------------------------------------------------------------------===//
//
// Implements standard IO functions. Note that varargs are not supported in
// CUDA, therefore the compiler needs to analyze the arguments passed to
// printf and generate a call to one of the functions defined here.
//
//===----------------------------------------------------------------------===//

#include "omptarget-nvptx.h"

EXTERN int __kmpc_printf(const char *str) { return printf("%s", str); }
