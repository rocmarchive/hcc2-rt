//===----------------------------------------------------------------------===//
//   vprintfl.cl: amdgcn device routine for vprintfl 
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.txt for details.
//
//===----------------------------------------------------------------------===//

#include "atmi.h"

#ifndef NULL
#define NULL 0
#endif

#define INLINE __attribute__((always_inline))
#define OFFSET 8

//From irif.h
extern uint __llvm_ld_atomic_a1_x_dev_i32(__global uint *);
extern uint __llvm_cmpxchg_a1_x_x_dev_i32(__global uint *, uint, uint);
extern __attribute__((const)) __constant void*
  __llvm_amdgcn_implicitarg_ptr(void) __asm("llvm.amdgcn.implicitarg.ptr");

// atl_Malloc:  Allocate device global memory atl_Malloc
//              Eventually, atl_Malloc will be part of ATMI Services 
INLINE __global char * atl_Malloc(uint n) {
    __global char *ptr = (__global char *)
      (((__constant size_t *)__llvm_amdgcn_implicitarg_ptr())[3]);
    uint size = ((__global uint *)ptr)[1];
    uint offset = __llvm_ld_atomic_a1_x_dev_i32((__global uint*) ptr);
    for (;;) {
        if (OFFSET + offset + n > size) return NULL;
        if (__llvm_cmpxchg_a1_x_x_dev_i32((__global uint *)ptr, offset, 
           offset+n)) break;
    }
    return ptr + OFFSET + offset;
}

// gen2dev_memcpy:  Generic to global memcpy for character string
INLINE void gen2dev_memcpy(__global char*dst, char*src, uint len) {
    for (int i=0 ; i< len ; i++) dst[i]=src[i];
}

// vprintfl: allocate device mem, create header, copy fmtstr, return data ptr
INLINE char* vprintfl(char*fmtstr, uint fmtlen, uint datalen) {
    // Allocate device global memory
    size_t headsize = sizeof(atl_service_header_t);
    uint buffsize   = (uint) headsize + fmtlen + datalen ;
    __global char* buffer = atl_Malloc(buffsize);
    if (buffer) {
        __global atl_service_header_t* header = 
           (__global atl_service_header_t*) buffer;
        header->size           = buffsize;
        header->service_id     = (atl_service_id_t) ATMI_SERVICE_PRINTF;
        header->device_atmi_id = (atmi_id_t) ATMI_ID ;
        gen2dev_memcpy((__global char*) (buffer+headsize), fmtstr, fmtlen);
        return (buffer + headsize + (size_t)fmtlen);
    } else 
        return NULL;
}
