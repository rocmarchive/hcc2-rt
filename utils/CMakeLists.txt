##===----------------------------------------------------------------------===##
#
#                     The LLVM Compiler Infrastructure
#
# This file is dual licensed under the MIT and the University of Illinois Open
# Source Licenses. See LICENSE.txt for details.
##===----------------------------------------------------------------------===##
#
# utils/CMakeLists.txt
# Written by Greg Rodgers (Gregory.Rodgers@amd.com)
#
##===----------------------------------------------------------------------===##

find_package(LLVM QUIET CONFIG
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

set(HCC2_LIBS
  ${HCC2_LIBDIR}/libLLVMBitWriter.a
  ${HCC2_LIBDIR}/libLLVMBitReader.a
  ${HCC2_LIBDIR}/libLLVMObject.a
  ${HCC2_LIBDIR}/libLLVMCore.a
  ${HCC2_LIBDIR}/libLLVMDemangle.a
  ${HCC2_LIBDIR}/libLLVMMC.a
  ${HCC2_LIBDIR}/libLLVMMCParser.a
  ${HCC2_LIBDIR}/libLLVMSupport.a
  ${HCC2_LIBDIR}/libLLVMBinaryFormat.a )

mark_as_advanced(
  HCC2_DIR_FOUND
  HCC2_INSTALL_PREFIX
  HCC2_BINDIR
  HCC2_INCDIR
  HCC2_LIBDIR
  HCC2_MAIN_INCDIR)

message("Building omprt utils using ${HCC2_BINDIR}/clang++ ")

project(omprt-utils)
add_custom_target(omprt-utils ALL)

# We want set-linkage in the distribution for library builders to use
# Also install shell utilities that are nice to have for debugging
install(PROGRAMS
   ${CMAKE_CURRENT_SOURCE_DIR}/bin/bundle.sh
   ${CMAKE_CURRENT_SOURCE_DIR}/bin/unbundle.sh
   ${CMAKE_CURRENT_SOURCE_DIR}/bin/cloc.sh
   ${CMAKE_CURRENT_SOURCE_DIR}/bin/mymcpu
   ${CMAKE_CURRENT_SOURCE_DIR}/bin/mygpu
   DESTINATION "bin")

