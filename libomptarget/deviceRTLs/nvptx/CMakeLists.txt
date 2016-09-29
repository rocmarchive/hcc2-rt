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
      src/data_sharing.cu
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

  set(omp_data_objects src/omp_data.cu)

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

  cuda_add_library(omptarget-nvptx STATIC ${cuda_src_files} ${omp_data_objects}
      OPTIONS ${CUDA_ARCH} ${CUDA_DEBUG})
      
  # Install device RTL under the lib destination folder.
  install(TARGETS omptarget-nvptx ARCHIVE DESTINATION "lib")

  target_link_libraries(omptarget-nvptx ${CUDA_LIBRARIES})

else()
  libomptarget_say("Not building CUDA offloading device RTL: CUDA tools not found in the system.")
endif()

set(LIBOMPTARGET_NVPTX_ENABLE_BCLIB FALSE CACHE BOOL
    "Enable CUDA LLVM bitcode offloading device RTL.")
if (${LIBOMPTARGET_NVPTX_ENABLE_BCLIB})
  libomptarget_say("Building CUDA LLVM bitcode offloading device RTL.")

  # Find a clang compiler capable of compiling cuda files to LLVM bitcode.
  set(LIBOMPTARGET_NVPTX_CUDA_COMPILER "" CACHE STRING
    "Location of a CUDA compiler capable of emitting LLVM bitcode.")
  if (DEFINED LIBOMPTARGET_NVPTX_CUDA_COMPILER)
    set(CUDA_COMPILER ${LIBOMPTARGET_NVPTX_CUDA_COMPILER})
  else()
    find_program(CUDA_COMPILER clang++)
  endif()
  if(NOT CUDA_COMPILER)
    libomptarget_say("Cannot find a CUDA compiler capable of emitting LLVM bitcode.")
    libomptarget_say("Please configure with flag -DLIBOMPTARGET_NVPTX_CUDA_COMPILER")
    return()
  endif()

  # Find a linker capable of linking bitcode objects.
  set(LIBOMPTARGET_NVPTX_BC_LINKER "" CACHE STRING
    "Location of a linker capable of linking LLVM bitcode objects.")
  if (DEFINED LIBOMPTARGET_NVPTX_BC_LINKER)
    set(BC_LINKER ${LIBOMPTARGET_NVPTX_BC_LINKER})
  else()
    find_program(BC_LINKER llvm-link)
  endif()
  if(NOT BC_LINKER)
    libomptarget_say("Cannot find a linker capable of linking LLVM bitcode objects.")
    libomptarget_say("Please configure with flag -DLIBOMPTARGET_NVPTX_BC_LINKER")
    return()
  endif()

  # Set flags for Clang cuda compilation.  Only Clang is supported because there is
  # no other compiler capable of generating bitcode from cuda sources.
  set(CUDA_FLAGS -emit-llvm -O0 -Xclang -target-feature -Xclang +ptx42 --cuda-device-only -DOMPTARGET_NVPTX_TEST=0 -DOMPTARGET_NVPTX_DEBUG=0)

  # This is currently broken
  set(CUDA_INCLUDES -I/usr/include/powerpc64le-linux-gnu)

  # Get all the compute capabilities the user requested or use SM_35 by default.
  set(CUDA_ARCH "")
  if(LIBOMPTARGET_NVPTX_COMPUTE_CAPABILITY)
    string(REPLACE "," ";" nvptx_sm_list ${LIBOMPTARGET_NVPTX_COMPUTE_CAPABILITY})
    foreach(sm ${nvptx_sm_list})
      set(CUDA_ARCH ${CUDA_ARCH} --cuda-gpu-arch=sm_${sm})
    endforeach()
  else()
    set(CUDA_ARCH --cuda-gpu-arch=sm_35)
  endif()

  # Compile cuda files to bitcode.
  set(bc_files "")
  foreach(src ${cuda_src_files})
    get_filename_component(infile ${src} ABSOLUTE)
    get_filename_component(outfile ${src} NAME)

    add_custom_command(OUTPUT ${outfile}.bc
      COMMAND ${CUDA_COMPILER} ${CUDA_FLAGS} ${CUDA_ARCH} ${CUDA_INCLUDES}
        -c ${infile} -o ${outfile}.bc
      DEPENDS ${infile}
      IMPLICIT_DEPENDS CXX ${infile}
      COMMENT "Building LLVM bitcode ${outfile}.bc"
      VERBATIM
    )
    set_property(DIRECTORY APPEND PROPERTY ADDITIONAL_MAKE_CLEAN_FILES ${outfile}.bc)

    list(APPEND bc_files ${outfile}.bc)
  endforeach()

  # Link to a bitcode library.
  add_custom_command(OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/libomptarget-nvptx.bc
      COMMAND ${BC_LINKER} -o ${CMAKE_CURRENT_BINARY_DIR}/libomptarget-nvptx.bc ${bc_files}
      DEPENDS ${bc_files}
      COMMENT "Linking LLVM bitcode libomptarget-nvptx.bc"
  )
  set_property(DIRECTORY APPEND PROPERTY ADDITIONAL_MAKE_CLEAN_FILES libomptarget-nvptx.bc)

  add_custom_target(omptarget-nvptx-bc ALL DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/libomptarget-nvptx.bc)

  # Copy library to destination.
  add_custom_command(TARGET omptarget-nvptx-bc POST_BUILD
                     COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_BINARY_DIR}/libomptarget-nvptx.bc
                     $<TARGET_FILE_DIR:omptarget-nvptx>)

  # Install device RTL under the lib destination folder.
  install(FILES ${CMAKE_CURRENT_BINARY_DIR}/libomptarget-nvptx.bc DESTINATION "lib")

endif()

