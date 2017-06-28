##===----------------------------------------------------------------------===##
#
#                     The LLVM Compiler Infrastructure
#
# This file is dual licensed under the MIT and the University of Illinois Open
# Source Licenses. See LICENSE.txt for details.
#
##===----------------------------------------------------------------------===##
#
# Build a plugin for an HSA machine if available.
#
##===----------------------------------------------------------------------===##

################################################################################
# Add check for required libraries
if(NOT LIBOMPTARGET_DEP_LIBELF_FOUND)
  libomptarget_say("Not building HSA plugin: LIBELF not found")
  return()
endif()

if(NOT LIBOMPTARGET_DEP_LIBHSA_FOUND)
  libomptarget_say("Not building HSA plugin: LIBHSA not found")
  return()
endif()


if(NOT LIBOMPTARGET_DEP_ATMI_INCLUDE_DIRS)
  libomptarget_say("Not building HSA plugin: LIBHSA found, but ATMI not installed")
  return()
endif()

if(NOT LIBOMPTARGET_DEP_LIBPTHREAD_FOUND)
  libomptarget_say("Not building HSA plugin: No libpthread found.")
  return()
endif()

if(NOT LIBOMPTARGET_DEP_LIBTINFO_FOUND)
  libomptarget_say("Not building HSA plugin: No libtinfo found.")
  return()
endif()

if(NOT CMAKE_SYSTEM_PROCESSOR MATCHES "(x86_64)|(ppc64le)$" AND CMAKE_SYSTEM_NAME MATCHES "Linux")
  libomptarget_say("Not building HSA plugin: only support HSA in Linux x86_64 or ppc64le hosts.")
  return()
endif()

if(NOT AMDLLVM_DIR_FOUND)
  libomptarget_say("Not building HSA plugin: No AMDLLVM package found.")
  return()
endif()

# Pass check
#libomptarget_say("Building HSA offloading plugin.  ELF_LIBRARIES=${LIBOMPTARGET_DEP_LIBELF_LIBRARIES} ")
#libomptarget_say("Building HSA offloading plugin.  TINFO_LIBRARIES=${LIBOMPTARGET_DEP_LIBTINFO_LIBRARIES} ")
libomptarget_say("Building HSA offloading plugin. ATMI_LIBRARIES_PATH=${LIBOMPTARGET_DEP_ATMI_LIBRARIES} ")
#libomptarget_say("Building HSA offloading plugin.  AMDLLVM=${AMDLLVM_DIR_FOUND} ")

################################################################################
# Define the suffix for the runtime messaging dumps.
add_definitions(-DTARGET_NAME=HSA)

# cancel z,defs
#set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -Wl,--unresolved-symbols=ignore-all")

if(CMAKE_BUILD_TYPE MATCHES Debug)
  add_definitions(-DHSA_ERROR_REPORT)
endif()

include_directories(
  ${LIBOMPTARGET_DEP_LIBHSA_INCLUDE_DIRS}
  ${LIBOMPTARGET_DEP_ATMI_INCLUDE_DIRS}
)

#  ${AMDLLVM_MAIN_INCDIR}/../lib/Target/AMDGPU
#  ${AMDLLVM_MAIN_INCDIR}/../lib/Target/AMDGPU/MCTargetDesc)

add_library(omptarget.rtl.hsa SHARED
  src/rtl.cpp src/elf_utils.c )

# Install plugin under the lib destination folder.
if(CMAKE_BUILD_TYPE MATCHES Debug)
  install(TARGETS omptarget.rtl.hsa LIBRARY DESTINATION "lib-debug")
else()
  install(TARGETS omptarget.rtl.hsa LIBRARY DESTINATION "lib")
endif()

target_link_libraries(
  omptarget.rtl.hsa
  -L${LIBOMPTARGET_DEP_ATMI_LIBRARIES} -latmi_runtime
  -L${AMDLLVM_LIBDIR} -lLLVMAMDGPUDesc -lLLVMAMDGPUUtils -lLLVMMC -lLLVMCore -lLLVMSupport
  ${LIBOMPTARGET_DEP_LIBTINFO_LIBRARIES}
  ${LIBOMPTARGET_DEP_LIBPTHREAD_LIBRARIES}
  ${LIBOMPTARGET_DEP_LIBELF_LIBRARIES}
  "-Wl,--version-script=${CMAKE_CURRENT_SOURCE_DIR}/../exports"
  )

# Report to the parent scope that we are building a plugin for hsa
set(LIBOMPTARGET_SYSTEM_TARGETS "${LIBOMPTARGET_SYSTEM_TARGETS} amdgcn-amd-hsa" PARENT_SCOPE)

