##===----------------------------------------------------------------------===##
#
#                     The LLVM Compiler Infrastructure
#
# This file is dual licensed under the MIT and the University of Illinois Open
# Source Licenses. See LICENSE.txt for details.
##===----------------------------------------------------------------------===##
#
# amdgcn/CMakeLists.txt
# Written by Greg Rodgers (Gregory.Rodgers@amd.com)
#
##===----------------------------------------------------------------------===##

#set(amdgcn_mcpus fiji kaveri carrizo hawaii)
set(amdgcn_mcpus gfx803 gfx700 gfx801 gfx701 gfx800 gfx900 gfx901)

################################################################################
# Add check for required libraries
if(NOT AMDLLVM_DIR_FOUND)
  libomptarget_say("Not building AMDGCN device RTL: AMDLLVM not found")
  return()
endif()

if(NOT LIBOMPTARGET_DEP_LIBPTHREAD_FOUND)
  libomptarget_say("Not building AMDGCN device RTL: No libpthread found.")
  return()
endif()

if(NOT LIBOMPTARGET_DEP_LIBTINFO_FOUND)
  libomptarget_say("Not building AMDGCN device RTL: No libtinfo found.")
  return()
endif()

# Pass check
libomptarget_say("Building AMDGCN device RTL. LLVM_COMPILER_PATH=${AMDLLVM_BINDIR} ")

# AMDLLVM clang has been patched to relax diagnosis of extern shared variable without incompletearray type
# The final solution is to use coral-clang instead of AMDLLVM clang to compile OMP Runtime since it is
# totally customized for OMP Runtime.
#
# TODO:
#   (1) Sync coral-clang with latest trunk
#   (2) Sync coral-clang with AMDLLVM clang and support CUDA 8.0
#


project(omptarget-amdgcn)
add_custom_target(omptarget-amdgcn ALL)

add_dependencies(omptarget-amdgcn utilities)

###############################################################################
# Create bc library from different sources
###############################################################################
include (bc.cmake)

#optimization level
set(optimization_level 2)

# Activate RTL message dumps if requested by the user.
if(LIBOMPTARGET_AMDGCN_DEBUG)
  set(CUDA_DEBUG -DOMPTARGET_NVPTX_DEBUG=-1)
endif()

file(GLOB sources
  ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cu
  ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cl
  ${CMAKE_CURRENT_SOURCE_DIR}/src/*.ll)

# for both in-tree and out-of-tree build
if (NOT CMAKE_ARCHIVE_OUTPUT_DIRECTORY)
  set(OUTPUTDIR ${CMAKE_CURRENT_BINARY_DIR})
else()
  set(OUTPUTDIR ${CMAKE_ARCHIVE_OUTPUT_DIRECTORY})
endif()

# create libraries
foreach(mcpu ${amdgcn_mcpus})
  add_bc_library(omptarget-amdgcn ${CMAKE_CURRENT_SOURCE_DIR} ${sources})

  if(CMAKE_BUILD_TYPE MATCHES Debug)
    install(FILES ${OUTPUTDIR}/libomptarget-amdgcn-${mcpu}.bc DESTINATION "lib-debug/libdevice")
  else()
    install(FILES ${OUTPUTDIR}/libomptarget-amdgcn-${mcpu}.bc DESTINATION "lib/libdevice")
  endif()
endforeach()
