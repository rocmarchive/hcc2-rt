##===----------------------------------------------------------------------===##
# 
#                     The LLVM Compiler Infrastructure
#
# This file is dual licensed under the MIT and the University of Illinois Open
# Source Licenses. See LICENSE.txt for details.
# 
##===----------------------------------------------------------------------===##
#
# Build the NVPTX (CUDA) Device RTL if the CUDA tools are available
#
##===----------------------------------------------------------------------===##

set(LIBOMPTARGET_NVPTX_ALTERNATE_HOST_COMPILER "" CACHE STRING
  "Path to alternate NVCC host compiler to be used by the NVPTX device RTL.")
  
if(LIBOMPTARGET_NVPTX_ALTERNATE_HOST_COMPILER)
  find_program(CUDA_HOST_COMPILER NAMES ${LIBOMPTARGET_NVPTX_ALTERNATE_HOST_COMPILER})
  if(NOT CUDA_HOST_COMPILER)
    libomptarget_say("Not building CUDA offloading device RTL: invalid NVPTX alternate host compiler.")
  endif()
endif()

# We can't use clang as nvcc host processor, so we attempt to replace it with
# gcc.
if(CUDA_HOST_COMPILER MATCHES "(clang)|(.*/clang)$")

  find_program(LIBOMPTARGET_NVPTX_ALTERNATE_GCC_HOST_COMPILER NAMES gcc)
  
  if(NOT LIBOMPTARGET_NVPTX_ALTERNATE_GCC_HOST_COMPILER)
    libomptarget_say("Not building CUDA offloading device RTL: clang is not supported as NVCC host compiler.")
    libomptarget_say("Please include gcc in your path or set LIBOMPTARGET_NVPTX_ALTERNATE_HOST_COMPILER to the full path of of valid compiler.")
    return()
  endif()
  set(CUDA_HOST_COMPILER "${LIBOMPTARGET_NVPTX_ALTERNATE_GCC_HOST_COMPILER}")
endif()

if(LIBOMPTARGET_DEP_CUDA_FOUND)
  libomptarget_say("Building CUDA offloading device RTL.")
  
  # We really don't have any host code, so we don't need to care about 
  # propagating host flags.
  set(CUDA_PROPAGATE_HOST_FLAGS OFF)
  
  set(cuda_src_files
      src/cancel.cu
      src/critical.cu
      src/debug.cu
      src/libcall.cu
      src/loop.cu
      src/omptarget-nvptx.cu
      src/parallel.cu
      src/reduction.cu
      src/stdio.cu
      src/sync.cu
      src/task.cu
  )

  include_directories(
      src/
  )

  # Get all the compute capabilities the user requested or use SM_35 by default.
  if(LIBOMPTARGET_NVPTX_COMPUTE_CAPABILITY)
    string(REPLACE "," ";" nvptx_sm_list ${LIBOMPTARGET_NVPTX_COMPUTE_CAPABILITY})
    foreach(sm ${nvptx_sm_list})
      set(CUDA_ARCH ${CUDA_ARCH} -gencode arch=compute_${sm},code=sm_${sm})
    endforeach()
  else()
    set(CUDA_ARCH -arch sm_35)
  endif()
  
  # Activate RTL message dumps if requested by the user.
  if(LIBOMPTARGET_NVPTX_DEBUG)
    set(CUDA_DEBUG -DOMPTARGET_NVPTX_DEBUG=-1 -g --ptxas-options=-v)
  endif()

  # NVPTX runtime library has to be statically linked. Dynamic linking is not
  # yet supported by the CUDA toolchain on the device.
  set(BUILD_SHARED_LIBS OFF)
  set(CUDA_SEPARABLE_COMPILATION ON)

  cuda_add_library(omptarget-nvptx STATIC ${cuda_src_files} 
      OPTIONS ${CUDA_ARCH} ${CUDA_DEBUG})
      
  # Install device RTL under the lib destination folder.
  install(TARGETS omptarget-nvptx ARCHIVE DESTINATION "lib")

  target_link_libraries(omptarget-nvptx ${CUDA_LIBRARIES})

else()
  libomptarget_say("Not building CUDA offloading device RTL: CUDA tools not found in the system.")
endif()
