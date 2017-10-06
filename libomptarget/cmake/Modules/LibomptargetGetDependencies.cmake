#
#//===----------------------------------------------------------------------===//
#//
#//                     The LLVM Compiler Infrastructure
#//
#// This file is dual licensed under the MIT and the University of Illinois Open
#// Source Licenses. See LICENSE.txt for details.
#//
#//===----------------------------------------------------------------------===//
#

# Try to detect in the system several dependencies required by the different
# components of libomptarget. These are the dependencies we have:
#
# libelf : required by some targets to handle the ELF files at runtime.
# libffi : required to launch target kernels given function and argument 
#          pointers.
# CUDA : required to control offloading to NVIDIA GPUs.

include (FindPackageHandleStandardArgs)

find_package(PkgConfig)

################################################################################
# Looking for libelf...
################################################################################
pkg_check_modules(LIBOMPTARGET_SEARCH_LIBELF QUIET libelf)

find_path (
  LIBOMPTARGET_DEP_LIBELF_INCLUDE_DIR
  NAMES
    libelf.h
  PATHS
    /usr/include
    /usr/local/include
    /opt/local/include
    /sw/include
    ENV CPATH
  PATH_SUFFIXES
    libelf)

find_library (
  LIBOMPTARGET_DEP_LIBELF_LIBRARIES
  NAMES
    elf
  PATHS
    /usr/lib
    /usr/local/lib
    /opt/local/lib
    /sw/lib
    ENV LIBRARY_PATH
    ENV LD_LIBRARY_PATH)
    
set(LIBOMPTARGET_DEP_LIBELF_INCLUDE_DIRS ${LIBOMPTARGET_DEP_LIBELF_INCLUDE_DIR})
find_package_handle_standard_args(
  LIBOMPTARGET_DEP_LIBELF 
  DEFAULT_MSG
  LIBOMPTARGET_DEP_LIBELF_LIBRARIES
  LIBOMPTARGET_DEP_LIBELF_INCLUDE_DIRS)

mark_as_advanced(
  LIBOMPTARGET_DEP_LIBELF_INCLUDE_DIRS 
  LIBOMPTARGET_DEP_LIBELF_LIBRARIES)
  
################################################################################
# Looking for libffi...
################################################################################
pkg_check_modules(LIBOMPTARGET_SEARCH_LIBFFI QUIET libffi)

find_path (
  LIBOMPTARGET_DEP_LIBFFI_INCLUDE_DIR
  NAMES
    ffi.h
  HINTS
    ${LIBOMPTARGET_SEARCH_LIBFFI_INCLUDEDIR}
    ${LIBOMPTARGET_SEARCH_LIBFFI_INCLUDE_DIRS}
  PATHS
    /usr/include
    /usr/local/include
    /opt/local/include
    /sw/include
    ENV CPATH)

# Don't bother look for the library if the header files were not found.
if (LIBOMPTARGET_DEP_LIBFFI_INCLUDE_DIR)
  find_library (
      LIBOMPTARGET_DEP_LIBFFI_LIBRARIES
    NAMES
      ffi
    HINTS
      ${LIBOMPTARGET_SEARCH_LIBFFI_LIBDIR}
      ${LIBOMPTARGET_SEARCH_LIBFFI_LIBRARY_DIRS}
    PATHS
      /usr/lib
      /usr/local/lib
      /opt/local/lib
      /sw/lib
      ENV LIBRARY_PATH
      ENV LD_LIBRARY_PATH)
endif()

set(LIBOMPTARGET_DEP_LIBFFI_INCLUDE_DIRS ${LIBOMPTARGET_DEP_LIBFFI_INCLUDE_DIR})
find_package_handle_standard_args(
  LIBOMPTARGET_DEP_LIBFFI 
  DEFAULT_MSG
  LIBOMPTARGET_DEP_LIBFFI_LIBRARIES
  LIBOMPTARGET_DEP_LIBFFI_INCLUDE_DIRS)

mark_as_advanced(
  LIBOMPTARGET_DEP_LIBFFI_INCLUDE_DIRS 
  LIBOMPTARGET_DEP_LIBFFI_LIBRARIES)
  
################################################################################
# Looking for CUDA...
################################################################################
find_package(CUDA QUIET)

set(LIBOMPTARGET_DEP_CUDA_FOUND ${CUDA_FOUND})
set(LIBOMPTARGET_DEP_CUDA_LIBRARIES ${CUDA_LIBRARIES})
set(LIBOMPTARGET_DEP_CUDA_INCLUDE_DIRS ${CUDA_INCLUDE_DIRS})

mark_as_advanced(
  LIBOMPTARGET_DEP_CUDA_FOUND 
  LIBOMPTARGET_DEP_CUDA_INCLUDE_DIRS
  LIBOMPTARGET_DEP_CUDA_LIBRARIES)

################################################################################
# Looking for AMDGCN devce compiler
################################################################################

find_package(LLVM 6.0.0 QUIET CONFIG
  PATHS
  $ENV{HCC2}
  /opt/rocm/hcc2
  NO_DEFAULT_PATH
  )

if (LLVM_DIR)
  message(STATUS "Found LLVM ${LLVM_PACKAGE_VERSION}. Configure: ${LLVM_DIR}/LLVMConfig.cmake")
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

find_package(Clang QUIET CONFIG
  PATHS
  $ENV{HCC2}
  /opt/rocm/hcc2
  NO_DEFAULT_PATH
  )

if (CLANG_CMAKE_DIR)
  message(STATUS "Found Clang ${LLVM_PACKAGE_VERSION}. Configure: ${CLANG_CMAKE_DIR}/ClangConfig.cmake")
endif()

mark_as_advanced(
  HCC2_DIR_FOUND
  HCC2_INSTALL_PREFIX
  HCC2_BINDIR
  HCC2_INCDIR
  HCC2_LIBDIR
  HCC2_MAIN_INCDIR)

################################################################################
# Looking for ROCM...
################################################################################
pkg_check_modules(LIBOMPTARGET_SEARCH_LIBHSA QUIET libhsa-runtime64)

find_path (
  LIBOMPTARGET_DEP_LIBHSA_INCLUDE_DIRS
  NAMES
  hsa.h
  PATHS
  $ENV{HSA_RUNTIME_PATH}/include
  /opt/rocm/include/hsa
  /usr/local/include/hsa
  )

find_path (
  LIBOMPTARGET_DEP_LIBHSA_LIBRARIES_DIRS
  NAMES
  libhsa-runtime64.so
  PATHS
  $ENV{HSA_RUNTIME_PATH}/lib
  /opt/rocm/lib/
  /usr/local/lib
  )

find_package_handle_standard_args(
  LIBOMPTARGET_DEP_LIBHSA
  DEFAULT_MSG
  LIBOMPTARGET_DEP_LIBHSA_LIBRARIES_DIRS
  LIBOMPTARGET_DEP_LIBHSA_INCLUDE_DIRS)

mark_as_advanced(
  LIBOMPTARGET_DEP_LIBHSA_INCLUDE_DIRS
  LIBOMPTARGET_DEP_LIBHSA_LIBRARIES_DIRS)

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
  /opt/rocm/libatmi/lib/
  /usr/local/lib
  )

find_package_handle_standard_args(
  LIBOMPTARGET_DEP_ATMI
  DEFAULT_MSG
  LIBOMPTARGET_DEP_ATMI_LIBRARIES
  LIBOMPTARGET_DEP_ATMI_INCLUDE_DIRS)

mark_as_advanced(
  LIBOMPTARGET_DEP_ATMI_INCLUDE_DIRS
  LIBOMPTARGET_DEP_ATMI_LIBRARIES)

################################################################################
# Looking for PTHREAD ...
################################################################################
pkg_check_modules(LIBOMPTARGET_SEARCH_LIBPTHREAD QUIET libpthread)
find_library (
  LIBOMPTARGET_DEP_LIBPTHREAD_LIBRARIES
  NAMES
  pthread
  PATHS
  /usr/lib
  /usr/local/lib
  /opt/local/lib
  /sw/lib
  ENV LIBRARY_PATH
  ENV LD_LIBRARY_PATH)

find_package_handle_standard_args(
  LIBOMPTARGET_DEP_LIBPTHREAD
  DEFAULT_MSG
  LIBOMPTARGET_DEP_LIBPTHREAD_LIBRARIES)

mark_as_advanced(
  LIBOMPTARGET_DEP_LIBPTHREAD_LIBRARIES_DIRS)

################################################################################
# Looking for TINFO...
################################################################################
pkg_check_modules(LIBOMPTARGET_SEARCH_LIBTINFO QUIET libtinfo)
find_library (
  LIBOMPTARGET_DEP_LIBTINFO_LIBRARIES
  NAMES
  tinfo
  PATHS
  /usr/lib
  /usr/local/lib
  /opt/local/lib
  /sw/lib
  ENV LIBRARY_PATH
  ENV LD_LIBRARY_PATH)

find_package_handle_standard_args(
  LIBOMPTARGET_DEP_LIBTINFO
  DEFAULT_MSG
  LIBOMPTARGET_DEP_LIBTINFO_LIBRARIES)

mark_as_advanced(
  LIBOMPTARGET_DEP_LIBTINFO_LIBRARIES_DIRS)

