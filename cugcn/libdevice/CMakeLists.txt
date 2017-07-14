##===----------------------------------------------------------------------===##
#
#                     The LLVM Compiler Infrastructure
#
# This file is dual licensed under the MIT and the University of Illinois Open
# Source Licenses. See LICENSE.txt for details.
##===----------------------------------------------------------------------===##
#
# cugcn/libdevice/CMakeLists.txt
# Written by Greg Rodgers (Gregory.Rodgers@amd.com)
#
##===----------------------------------------------------------------------===##

#set(amdgcn_mcpus fiji kaveri carrizo hawaii)
set(amdgcn_mcpus gfx803 gfx700 gfx801 gfx701 gfx800 gfx900 gfx901)

###############################################################################
# find llvm
###############################################################################
find_package(LLVM 5.0.0 QUIET CONFIG
  PATHS
  $ENV{HCC2}
  /opt/rocm/hcc2
  )

if (LLVM_DIR)
  #message(STATUS "Found LLVM ${LLVM_PACKAGE_VERSION}. Configure: ${LLVM_DIR}/LLVMConfig.cmake")
  #message(STATUS "      LLVM LLVM_INSTALL_PREFIX: ${LLVM_INSTALL_PREFIX}")
  #message(STATUS "      LLVM HCC2_MAIN_INCDIR: ${HCC2_MAIN_INCDIR}")
endif()

set(HCC2_DIR_FOUND ${LLVM_DIR})
set(HCC2_INSTALL_PREFIX ${LLVM_INSTALL_PREFIX})
set(HCC2_MAIN_INCDIR ${LLVM_BUILD_MAIN_INCLUDE_DIR})

if (HCC2_INSTALL_PREFIX)
  set(HCC2_BINDIR ${HCC2_INSTALL_PREFIX}/bin)
  set(HCC2_INCDIR ${HCC2_INSTALL_PREFIX}/include)
  set(HCC2_LIBDIR ${HCC2_INSTALL_PREFIX}/lib)
else()
  set(HCC2_BINDIR ${LLVM_BUILD_BINARY_DIR}/bin)
  set(HCC2_INCDIR ${LLVM_BUILD_BINARY_DIR}/include)
  set(HCC2_LIBDIR ${LLVM_LIBRARY_DIRS})
endif()

mark_as_advanced(
  HCC2_DIR_FOUND
  HCC2_INSTALL_PREFIX
  HCC2_BINDIR
  HCC2_INCDIR
  HCC2_LIBDIR
  HCC2_MAIN_INCDIR)

message("---> Building libicuda2gcn LLVM_COMPILER_PATH=${HCC2_BINDIR} for ${amdgcn_mcpus} ")

################################################################################
# Looking for ROCm-Device-Lib ...
################################################################################
find_path (
  LIBAMDGCN_ROOT
  NAMES
  libamdgcn
  PATHS
  $ENV{LIBAMDGCN}/..
  /opt/rocm
  )

if (LIBAMDGCN_ROOT)
  set(LIBAMDGCN_ROOTDIR ${LIBAMDGCN_ROOT}/libamdgcn)
endif()

mark_as_advanced(
  LIBAMDGCN_ROOTDIR)

message("---> Building libicuda2gcn LIBAMDGCN_ROOTDIR=${LIBAMDGCN_ROOTDIR}")

################################################################################
# Looking for ATMI...
################################################################################
find_path (
  LIBOMPTARGET_DEP_ATMI_INCLUDE_DIRS_1
  NAMES
  atmi.h
  PATHS
  $ENV{ATMI_RUNTIME_PATH}/include
  ${HCC2_INCDIR}
  /opt/rocm/atmi/include
  /opt/rocm/libatmi/include
  /usr/local/include
  )

find_path (
  LIBOMPTARGET_DEP_ATMI_INCLUDE_DIRS_2
  NAMES
  atmi_runtime.h
  PATHS
  $ENV{ATMI_RUNTIME_PATH}/include
  ${HCC2_INCDIR}
  /opt/rocm/atmi/include
  /opt/rocm/libatmi/include
  /usr/local/include
  )

if(${LIBOMPTARGET_DEP_ATMI_INCLUDE_DIRS_1} STREQUAL ${LIBOMPTARGET_DEP_ATMI_INCLUDE_DIRS_2})
  set(LIBOMPTARGET_DEP_ATMI_INCLUDE_DIRS ${LIBOMPTARGET_DEP_ATMI_INCLUDE_DIRS_1})
endif()

find_path (
  LIBOMPTARGET_DEP_ATMI_LIBRARIES
  NAMES
  libatmi_runtime.so
  PATHS
  $ENV{ATMI_RUNTIME_PATH}/lib
  ${HCC2_LIBDIR}
  /opt/rocm/atmi/lib
  /opt/rocm/libatmi/lib
  /usr/local/lib
  )

find_package_handle_standard_args(
  LIBOMPTARGET_DEP_ATMI
  DEFAULT_MSG
  LIBOMPTARGET_DEP_ATMI_INCLUDE_DIRS
  LIBOMPTARGET_DEP_ATMI_LIBRARIES)

mark_as_advanced(
  LIBOMPTARGET_DEP_ATMI_INCLUDE_DIRS
  LIBOMPTARGET_DEP_ATMI_LIBRARIES)

message("---> Building libicuda2gcn LIBOMPTARGET_DEP_ATMI_INCLUDE_DIRS=${LIBOMPTARGET_DEP_ATMI_INCLUDE_DIRS}")
message("---> Building libicuda2gcn LIBOMPTARGET_DEP_ATMI_LIBRARIES=${LIBOMPTARGET_DEP_ATMI_LIBRARIES}")

###############################################################################
# Main project
###############################################################################

project(icuda2gcn)

add_custom_target(icuda2gcn ALL)

add_dependencies(icuda2gcn utilities)

###############################################################################
# Create bc library from different sources
###############################################################################
include (bc.cmake)

#optimization level
set(optimization_level 2)

# for both in-tree and out-of-tree build
if (CMAKE_ARCHIVE_OUTPUT_DIRECTORY)
  set(OUTPUTDIR ${CMAKE_ARCHIVE_OUTPUT_DIRECTORY})
else()
  set(OUTPUTDIR ${CMAKE_CURRENT_BINARY_DIR})
endif()

file(GLOB sources
  ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cu
  ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cl
  ${CMAKE_CURRENT_SOURCE_DIR}/src/*.ll)

# create libraries
foreach(mcpu ${amdgcn_mcpus})
  set(final_bc_filename libicuda2gcn-${mcpu}.bc)

  add_bc_library(icuda2gcn ${CMAKE_CURRENT_SOURCE_DIR} ${sources})

  if(CMAKE_BUILD_TYPE MATCHES Debug)
    install(FILES ${OUTPUTDIR}/${final_bc_filename} DESTINATION "lib-debug/libdevice")
  else()
    install(FILES ${OUTPUTDIR}/${final_bc_filename} DESTINATION "lib/libdevice")
  endif()

endforeach()

