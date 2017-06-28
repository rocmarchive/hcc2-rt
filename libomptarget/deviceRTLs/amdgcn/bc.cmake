##===----------------------------------------------------------------------===##
#
#                     The LLVM Compiler Infrastructure
#
# This file is dual licensed under the MIT and the University of Illinois Open
# Source Licenses. See LICENSE.txt for details.
##===----------------------------------------------------------------------===##
#
# amdgcn/bc.cmake
#
##===----------------------------------------------------------------------===##

macro(collect_sources name dir)
  set(cuda_sources)
  set(ocl_sources)
  set(llvm_sources)

  foreach(file ${ARGN})
    file(RELATIVE_PATH rfile ${dir} ${file})
    get_filename_component(rdir ${rfile} DIRECTORY)
    get_filename_component(fname ${rfile} NAME_WE)
    get_filename_component(fext ${rfile} EXT)
    #file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/${rdir})
    if (fext STREQUAL ".cu")
      set(cfile ${CMAKE_CURRENT_BINARY_DIR}/${rdir}/${fname}.cu)
      list(APPEND cuda_sources ${cfile})
    endif()

    if (fext STREQUAL ".cl")
      set(cfile ${CMAKE_CURRENT_BINARY_DIR}/${rdir}/${fname}.cl)
      list(APPEND ocl_sources ${cfile})
    endif()
    if (fext STREQUAL ".ll")
      list(APPEND csources ${file})
      set(cfile ${CMAKE_CURRENT_BINARY_DIR}/${rdir}/${fname}.ll)
      list(APPEND llvm_sources ${cfile})
    endif()
  endforeach()

  #set (MY_LIST a b c d)
  #list (APPEND MY_LIST e)
  #list (GET MY_LIST 0 HEAD)
  #list (LENGTH MY_LIST LISTCOUNT)
  #message ("HEAD = ${HEAD}, ${LISTCOUNT}")
endmacro()

macro(add_llvm_bc_library name dir)
  set(ll_files)

  foreach(file ${ARGN})
    file(RELATIVE_PATH rfile ${dir} ${file})
    get_filename_component(rdir ${rfile} DIRECTORY)
    get_filename_component(fname ${rfile} NAME_WE)
    get_filename_component(fext ${rfile} EXT)

    list(APPEND ll_files ${CMAKE_CURRENT_SOURCE_DIR}/src/${fname}.ll)
  endforeach()

  add_custom_command(
    OUTPUT linkout.llvm.${mcpu}.bc
    COMMAND ${AMDLLVM_BINDIR}/llvm-link ${ll_files} -o linkout.llvm.${mcpu}.bc
    DEPENDS ${ll_files}
    )

  list(APPEND bc_files linkout.llvm.${mcpu}.bc)
endmacro()

macro(add_ocl_bc_library name dir)
  set(cl_cmd ${AMDLLVM_BINDIR}/clang
    -S -emit-llvm
    -DCL_VERSION_2_0=200 -D__OPENCL_C_VERSION__=200
    -Dcl_khr_fp64 -Dcl_khr_fp16
    -Dcl_khr_subgroups -Dcl_khr_int64_base_atomics -Dcl_khr_int64_extended_atomics
    -x cl -Xclang -cl-std=CL2.0 -Xclang -finclude-default-header
    -target amdgcn--cuda
    -I${CMAKE_CURRENT_SOURCE_DIR}/src)

  set(ll_files)

  foreach(file ${ARGN})
    file(RELATIVE_PATH rfile ${dir} ${file})
    get_filename_component(rdir ${rfile} DIRECTORY)
    get_filename_component(fname ${rfile} NAME_WE)
    get_filename_component(fext ${rfile} EXT)

    set(ll_filename ${fname}.${mcpu}.ll)

    file(GLOB h_files "${CMAKE_CURRENT_SOURCE_DIR}/src/*.h")

    add_custom_command(
      OUTPUT ${ll_filename}
      COMMAND ${cl_cmd} ${CMAKE_CURRENT_SOURCE_DIR}/src/${fname}.cl -o ${ll_filename}
      DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/src/${fname}.cl" ${h_files}
      )

    list(APPEND ll_files ${ll_filename})
  endforeach()

  add_custom_command(
    OUTPUT linkout.ocl.${mcpu}.bc
    COMMAND ${AMDLLVM_BINDIR}/llvm-link ${ll_files} -o linkout.ocl.${mcpu}.bc
    DEPENDS ${ll_files}
    )

  list(APPEND bc_files linkout.ocl.${mcpu}.bc)
endmacro()

macro(add_cuda_bc_library name dir)
  set(cu_cmd ${AMDLLVM_BINDIR}/clang++
    -S -emit-llvm
    --cuda-device-only
    -nocudalib
    -DGPUCC_AMDGCN
    --cuda-gpu-arch=${mcpu}
    ${CUDA_DEBUG}
    -I${CMAKE_CURRENT_SOURCE_DIR}/src)

  set(ll_files)
  set(fixed_files)

  foreach(file ${ARGN})
    file(RELATIVE_PATH rfile ${dir} ${file})
    get_filename_component(rdir ${rfile} DIRECTORY)
    get_filename_component(fname ${rfile} NAME_WE)
    get_filename_component(fext ${rfile} EXT)

    set(ll_filename ${fname}.${mcpu}.ll)
    set(fixed_filename fixed.${fname}.${mcpu}.ll)

    file(GLOB h_files "${CMAKE_CURRENT_SOURCE_DIR}/src/*.h")

    add_custom_command(
      OUTPUT ${ll_filename}
      COMMAND ${cu_cmd} ${CMAKE_CURRENT_SOURCE_DIR}/src/${fname}.cu -o ${ll_filename}
      DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/src/${fname}.cu" ${h_files}
      )

    add_custom_command(
      OUTPUT ${fixed_filename}
      COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/../../../utils/fixup/ll_gcn_fixup.sh ${ll_filename} ${fixed_filename} ${AMDLLVM_BINDIR}
      DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/../../../utils/fixup/ll_gcn_fixup.sh" ${ll_filename}
      )

    list(APPEND ll_files ${ll_filename})
    list(APPEND fixed_files ${fixed_filename})
  endforeach()

  add_custom_command(
    OUTPUT linkout.cuda.${mcpu}.bc
    COMMAND ${AMDLLVM_BINDIR}/llvm-link ${fixed_files} -o linkout.cuda.${mcpu}.bc
    DEPENDS ${fixed_files}
    )

  list(APPEND bc_files linkout.cuda.${mcpu}.bc)
endmacro()

macro(add_bc_library name dir)
  set(bc_files)

  collect_sources(${name} ${dir} ${sources})

  if (llvm_sources)
    add_llvm_bc_library(${name} ${dir} ${llvm_sources})
  else()
    #message(STATUS "No LLVM IR source.")
  endif()
  if (ocl_sources)
    add_ocl_bc_library(${name} ${dir} ${ocl_sources})
  else()
    #message(STATUS "No OpenCL source.")
  endif()
  if (cuda_sources)
    add_cuda_bc_library(${name} ${dir} ${cuda_sources})
  else()
    #message(STATUS "No CUDA source.")
  endif()

  add_custom_command(
    OUTPUT linkout.${mcpu}.bc
    #FIXME: remove the warning suppress when the address space strategy are unified
    #COMMAND ${AMDLLVM_BINDIR}/llvm-link ${bc_files} -o linkout.${mcpu}.bc
    COMMAND ${AMDLLVM_BINDIR}/llvm-link -suppress-warnings ${bc_files} -o linkout.${mcpu}.bc
    DEPENDS ${bc_files}
    )
  add_custom_command(
    OUTPUT optout.${mcpu}.bc
    COMMAND ${AMDLLVM_BINDIR}/opt -O${optimization_level} linkout.${mcpu}.bc -o optout.${mcpu}.bc
    DEPENDS linkout.${mcpu}.bc
    )
  add_custom_command(
    OUTPUT lib${name}-${mcpu}.bc
    COMMAND ${CMAKE_CURRENT_BINARY_DIR}/../../../utils/set-linkage optout.${mcpu}.bc -o ${OUTPUTDIR}/lib${name}-${mcpu}.bc
    DEPENDS optout.${mcpu}.bc utilities
    )

  add_custom_target(lib${name}-${mcpu} ALL DEPENDS lib${name}-${mcpu}.bc)
endmacro()

