//===---- reduction.cu - NVPTX OpenMP reduction implementation ---- CUDA
//-*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.txt for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains the implementation of reduction with KMPC interface.
//
//===----------------------------------------------------------------------===//

#include <stdio.h>
#include <complex.h>

#include "omptarget-nvptx.h"

// cannot implement atomic_start and atomic_end for GPU. Report runtime error
EXTERN void __kmpc_atomic_start() {
  printf("__kmpc_atomic_start not supported\n");
  asm("trap;");
  return;
}

EXTERN void __kmpc_atomic_end() {
  printf("__kmpc_atomic_end not supported\n");
  asm("trap;");
  return;
}

//may eventually remove this
EXTERN
int32_t __gpu_block_reduce() {
  int tid = GetLogicalThreadIdInBlock();
  int nt = GetNumberOfOmpThreads(tid);
  if (nt != blockDim.x)
    return 0;
  unsigned tnum = __ballot(1);
  if (tnum != (~0x0)) { // assume swapSize is 32
    return 0;
  }
  return 1;
}

EXTERN
int32_t __kmpc_reduce_gpu(kmp_Indent *loc, int32_t global_tid, int32_t num_vars,
                          size_t reduce_size, void *reduce_data,
                          void *reduce_array_size, kmp_ReductFctPtr *reductFct,
                          kmp_CriticalName *lck) {
  int threadId = GetLogicalThreadIdInBlock();
  omptarget_nvptx_TaskDescr *currTaskDescr =
      getMyTopTaskDescriptor(threadId);
  int numthread;
  if (currTaskDescr->IsParallelConstruct()) {
    numthread = GetNumberOfOmpThreads(threadId);
  } else {
    numthread = GetNumberOfOmpTeams();
  }

  if (numthread == 1)
    return 1;
  else if (!__gpu_block_reduce())
    return 2;
  else {
    if (threadIdx.x == 0)
      return 1;
    else
      return 0;
  }
}

EXTERN
int32_t __kmpc_reduce_combined(kmp_Indent *loc) {
  if (threadIdx.x == 0) {
    return 2;
  } else {
    return 0;
  }
}

EXTERN
int32_t __kmpc_reduce_simd(kmp_Indent *loc) {
  if (threadIdx.x % 32 == 0) {
    return 1;
  } else {
    return 0;
  }
}

EXTERN
int32_t __kmpc_reduce41(kmp_Indent *loc, int32_t global_tid, int32_t num_vars,
                        size_t reduce_size, void *reduce_data,
                        void *reduce_array_size, kmp_ReductFctPtr *reductFct,
                        kmp_CriticalName *lck) {
  return __kmpc_reduce_gpu(loc, global_tid, num_vars, reduce_size, reduce_data,
                           reduce_array_size, reductFct, lck);
}

EXTERN
void __kmpc_nvptx_end_reduce(int32_t global_tid) {}

EXTERN
void __kmpc_nvptx_end_reduce_nowait(int32_t global_tid) {}

// implement different data type or operations  with atomicCAS
#define omptarget_nvptx_add(x, y) ((x) + (y))
#define omptarget_nvptx_sub(x, y) ((x) - (y))
#define omptarget_nvptx_sub_rev(y, x) ((x) - (y))
#define omptarget_nvptx_mul(x, y) ((x) * (y))
#define omptarget_nvptx_div(x, y) ((x) / (y))
#define omptarget_nvptx_div_rev(y, x) ((x) / (y))
#define omptarget_nvptx_min(x, y) ((x) > (y) ? (y) : (x))
#define omptarget_nvptx_max(x, y) ((x) < (y) ? (y) : (x))
#define omptarget_nvptx_andb(x, y) ((x) & (y))
#define omptarget_nvptx_orb(x, y) ((x) | (y))
#define omptarget_nvptx_xor(x, y) ((x) ^ (y))
#define omptarget_nvptx_shl(x, y) ((x) << (y))
#define omptarget_nvptx_shr(x, y) ((x) >> (y))
#define omptarget_nvptx_andl(x, y) ((x) && (y))
#define omptarget_nvptx_orl(x, y) ((x) || (y))
#define omptarget_nvptx_eqv(x, y) ((x) == (y))
#define omptarget_nvptx_neqv(x, y) ((x) != (y))

INLINE __device__ float atomicCAS(float *_addr, float _compare, float _val) {
  int *addr = (int *)_addr;
  int compare = __float_as_int(_compare);
  int val = __float_as_int(_val);
  return __int_as_float(atomicCAS(addr, compare, val));
}

INLINE __device__ double atomicCAS(double *_addr, double _compare,
                                   double _val) {
  unsigned long long int *addr = (unsigned long long int *)_addr;
  unsigned long long int compare = __double_as_longlong(_compare);
  unsigned long long int val = __double_as_longlong(_val);
  return __longlong_as_double(atomicCAS(addr, compare, val));
}

INLINE __device__ long long int
atomicCAS(long long int *_addr, long long int _compare, long long int _val) {
  unsigned long long int *addr = (unsigned long long int *)_addr;
  unsigned long long int compare = (unsigned long long int)(_compare);
  unsigned long long int val = (unsigned long long int)(_val);
  return (long long int)(atomicCAS(addr, compare, val));
}

INLINE __device__ int64_t atomicCAS(int64_t *_addr, int64_t _compare,
                                    int64_t _val) {
  unsigned long long int *addr = (unsigned long long int *)_addr;
  unsigned long long int compare = (unsigned long long int)(_compare);
  unsigned long long int val = (unsigned long long int)(_val);
  return (int64_t)(atomicCAS(addr, compare, val));
}

INLINE __device__ uint64_t atomicCAS(uint64_t *_addr, uint64_t _compare,
                                     uint64_t _val) {
  unsigned long long int *addr = (unsigned long long int *)_addr;
  unsigned long long int compare = (unsigned long long int)(_compare);
  unsigned long long int val = (unsigned long long int)(_val);
  return (uint64_t)(atomicCAS(addr, compare, val));
}

INLINE __device__ float complex atomicCAS(float complex *_addr,
                                          float complex _compare,
                                          float complex _val) {
  double *addr = (double *)_addr;
  double compare = (double)(_compare);
  double val = (double)(_val);
  return (float complex)(atomicCAS(addr, compare, val));
}

#define ATOMIC_GENOP_NATIVE(_name, _dtype, _op, _cudaop)                       \
  EXTERN void __kmpc_atomic_##_name##_##_op(kmp_Indent *id_ref, int32_t gtid,  \
                                            _dtype *lhs, _dtype rhs) {         \
    PRINT(LD_LOOP, "Reduction: thead %d\n", gtid);                             \
    atomic##_cudaop(lhs, rhs);                                                 \
  }                                                                            \
                                                                               \
  EXTERN _dtype __kmpc_atomic_##_name##_##_op##_cpt(                           \
      kmp_Indent *id_ref, int32_t gtid, _dtype *lhs, _dtype rhs, int flag) {   \
    _dtype old = atomic##_cudaop(lhs, rhs);                                    \
    if (flag) {                                                                \
      return omptarget_nvptx_##_op(old, rhs);                                  \
    } else {                                                                   \
      return old;                                                              \
    }                                                                          \
  }

// for types that are supported directly by atomicCAS
#define ATOMIC_GENOP_DIRECT(_name, _dtype, _op)                                \
  EXTERN void __kmpc_atomic_##_name##_##_op(kmp_Indent *id_ref, int32_t gtid,  \
                                            _dtype *lhs, _dtype rhs) {         \
    PRINT(LD_LOOP, "Reduction: thead %d\n", gtid);                             \
    _dtype *temp_lhs = lhs;                                                    \
    _dtype oldvalue = *temp_lhs;                                               \
    _dtype saved;                                                              \
    _dtype newvalue;                                                           \
    do {                                                                       \
      saved = oldvalue;                                                        \
      newvalue = (_dtype)omptarget_nvptx_##_op(saved, rhs);                    \
      oldvalue = atomicCAS(temp_lhs, saved, newvalue);                         \
    } while (saved != oldvalue);                                               \
  }                                                                            \
                                                                               \
  EXTERN _dtype __kmpc_atomic_##_name##_##_op##_cpt(                           \
      kmp_Indent *id_ref, int32_t gtid, _dtype *lhs, _dtype rhs, int flag) {   \
    _dtype *temp_lhs = lhs;                                                    \
    _dtype oldvalue = *temp_lhs;                                               \
    _dtype saved;                                                              \
    _dtype newvalue;                                                           \
    do {                                                                       \
      saved = oldvalue;                                                        \
      newvalue = (_dtype)omptarget_nvptx_##_op(saved, rhs);                    \
      oldvalue = atomicCAS(temp_lhs, saved, newvalue);                         \
    } while (saved != oldvalue);                                               \
    if (flag)                                                                  \
      return newvalue;                                                         \
    else                                                                       \
      return oldvalue;                                                         \
  }

#define ATOMIC_GENOP_DIRECT_REV(_name, _dtype, _op)                            \
  EXTERN void __kmpc_atomic_##_name##_##_op##_rev(                             \
      kmp_Indent *id_ref, int32_t gtid, _dtype *lhs, _dtype rhs) {             \
    _dtype *temp_lhs = lhs;                                                    \
    _dtype oldvalue = *temp_lhs;                                               \
    _dtype saved;                                                              \
    _dtype newvalue;                                                           \
    do {                                                                       \
      saved = oldvalue;                                                        \
      newvalue = (_dtype)omptarget_nvptx_##_op(rhs, saved);                    \
      oldvalue = atomicCAS(temp_lhs, saved, newvalue);                         \
    } while (saved != oldvalue);                                               \
  }                                                                            \
                                                                               \
  EXTERN _dtype __kmpc_atomic_##_name##_##_op##_cpt##_rev(                     \
      kmp_Indent *id_ref, int32_t gtid, _dtype *lhs, _dtype rhs, int flag) {   \
    _dtype *temp_lhs = lhs;                                                    \
    _dtype oldvalue = *temp_lhs;                                               \
    _dtype saved;                                                              \
    _dtype newvalue;                                                           \
    do {                                                                       \
      saved = oldvalue;                                                        \
      newvalue = (_dtype)omptarget_nvptx_##_op(rhs, saved);                    \
      oldvalue = atomicCAS(temp_lhs, saved, newvalue);                         \
    } while (saved != oldvalue);                                               \
    if (flag)                                                                  \
      return newvalue;                                                         \
    else                                                                       \
      return oldvalue;                                                         \
  }

INLINE __device__ void dc_add(double complex *lhs, double complex rhs) {
  double *ptrl = (double *)lhs;
  double *ptrr = (double *)&rhs;
  ptrl[0] += ptrr[0];
  ptrl[1] += ptrr[1];
}

INLINE __device__ void dc_sub(double complex *lhs, double complex rhs) {
  double *ptrl = (double *)lhs;
  double *ptrr = (double *)&rhs;
  ptrl[0] -= ptrr[0];
  ptrl[1] -= ptrr[1];
}

INLINE __device__ void dc_mul(double complex *lhs, double complex rhs) {
  double *ptrl = (double *)lhs;
  double *ptrr = (double *)&rhs;
  double r1 = ptrl[0], r2 = ptrr[0];
  double i1 = ptrl[1], i2 = ptrr[1];
  ptrl[0] = r1 * r2 - i1 * i2;
  ptrl[1] = r1 * i2 + r2 * i1;
}

INLINE __device__ void dc_div(double complex *lhs, double complex rhs) {
  double *ptrl = (double *)lhs;
  double *ptrr = (double *)&rhs;
  double r1 = ptrl[0], r2 = ptrr[0];
  double i1 = ptrl[1], i2 = ptrr[1];
  ptrl[0] = (r1 * r2 + i1 * i2) / (r2 * r2 + i2 * i2);
  ptrl[1] = (i1 * r2 - r1 * i2) / (r2 * r2 + i2 * i2);
}

#define ATOMIC_GENOP_DC(_op)                                                   \
  EXTERN void __kmpc_atomic_cmplx8_##_op(kmp_Indent *id_ref, int32_t gtid,     \
                                         double _Complex *lhs,                 \
                                         double _Complex rhs) {                \
    printf("Double complex atomic operation not supported\n");                 \
    asm("trap;");                                                              \
    return;                                                                    \
  }                                                                            \
  EXTERN double _Complex __gpu_warpBlockRedu_cmplx8_##_op(                     \
      double _Complex rhs) {                                                   \
    __shared__ double _Complex lhs;                                            \
    if (threadIdx.x == 0)                                                      \
      lhs = rhs;                                                               \
    __syncthreads();                                                           \
    for (int i = 1; i < blockDim.x; i++) {                                     \
      if (threadIdx.x == i) {                                                  \
        dc_##_op(&lhs, rhs);                                                   \
      }                                                                        \
      __syncthreads();                                                         \
    }                                                                          \
    return lhs;                                                                \
  }

// implementation with shared
#define ATOMIC_GENOP_DC_obsolete(_op)                                          \
  EXTERN void __kmpc_atomic_cmplx16_##_op(kmp_Indent *id_ref, int32_t gtid,    \
                                          double _Complex *lhs,                \
                                          double _Complex rhs) {               \
    __shared__ unsigned int stepinblock;                                       \
    unsigned tnum = __ballot(1);                                               \
    if (tnum != (~0x0)) {                                                      \
      return;                                                                  \
    }                                                                          \
    if (threadIdx.x == 0)                                                      \
      stepinblock = 0;                                                         \
    __syncthreads();                                                           \
    while (stepinblock < blockDim.x) {                                         \
      if (threadIdx.x == stepinblock) {                                        \
        dc_##_op(lhs, rhs);                                                    \
        stepinblock++;                                                         \
      }                                                                        \
      __syncthreads();                                                         \
    }                                                                          \
  }

ATOMIC_GENOP_DC(add);
ATOMIC_GENOP_DC(sub);
ATOMIC_GENOP_DC(mul);
ATOMIC_GENOP_DC(div);

INLINE __device__ uint64_t fc_add(float r1, float i1, float r2, float i2) {
  uint64_t result;
  float *rr = (float *)&result;
  float *ri = rr + 1;
  *rr = r1 + r2;
  *ri = i1 + i2;
  return result;
}

INLINE __device__ uint64_t fc_sub(float r1, float i1, float r2, float i2) {
  uint64_t result;
  float *rr = (float *)&result;
  float *ri = rr + 1;
  *rr = r1 - r2;
  *ri = i1 - i2;
  return result;
}

INLINE __device__ uint64_t fc_mul(float r1, float i1, float r2, float i2) {
  uint64_t result;
  float *rr = (float *)&result;
  float *ri = rr + 1;
  *rr = r1 * r2 - i1 * i2;
  *ri = r1 * i2 + r2 * i1;
  return result;
}

INLINE __device__ uint64_t fc_div(float r1, float i1, float r2, float i2) {
  uint64_t result;
  float *rr = (float *)&result;
  float *ri = rr + 1;
  *rr = (r1 * r2 + i1 * i2) / (r2 * r2 + i2 * i2);
  *ri = (i1 * r2 - r1 * i2) / (r2 * r2 + i2 * i2);
  return result;
}

#define ATOMIC_GENOP_FC(_op)                                                   \
  EXTERN void __kmpc_atomic_cmplx4_##_op(kmp_Indent *id_ref, int32_t gtid,     \
                                         float complex *lhs,                   \
                                         float complex rhs) {                  \
    uint64_t *temp_lhs = (uint64_t *)lhs;                                      \
    uint64_t oldvalue = *temp_lhs;                                             \
    uint64_t saved;                                                            \
    float *pr1 = (float *)&rhs;                                                \
    float *pi1 = pr1 + 1;                                                      \
    float r1 = *pr1;                                                           \
    float i1 = *pi1;                                                           \
    uint64_t newvalue;                                                         \
    do {                                                                       \
      saved = oldvalue;                                                        \
      float *pr2 = (float *)&saved;                                            \
      float *pi2 = pr2 + 1;                                                    \
      newvalue = fc_##_op(*pr2, *pi2, r1, i1);                                 \
      oldvalue = atomicCAS(temp_lhs, saved, newvalue);                         \
    } while (saved != oldvalue);                                               \
  }                                                                            \
                                                                               \
  EXTERN void __kmpc_atomic_cmplx4_##_op##_cpt(                                \
      kmp_Indent *id_ref, int32_t gtid, float complex *lhs, float complex rhs, \
      float complex *outp, int flag) {                                         \
    uint64_t *temp_lhs = (uint64_t *)lhs;                                      \
    uint64_t oldvalue = *temp_lhs;                                             \
    uint64_t saved;                                                            \
    float *pr1 = (float *)&rhs;                                                \
    float *pi1 = pr1 + 1;                                                      \
    float r1 = *pr1;                                                           \
    float i1 = *pi1;                                                           \
    uint64_t newvalue;                                                         \
    do {                                                                       \
      saved = oldvalue;                                                        \
      float *pr2 = (float *)&saved;                                            \
      float *pi2 = pr2 + 1;                                                    \
      newvalue = fc_##_op(*pr2, *pi2, r1, i1);                                 \
      oldvalue = atomicCAS(temp_lhs, saved, newvalue);                         \
    } while (saved != oldvalue);                                               \
    if (flag) {                                                                \
      float complex *temp = (float complex *)&newvalue;                        \
      *outp = *temp;                                                           \
    } else {                                                                   \
      float complex *temp = (float complex *)&saved;                           \
      *outp = *temp;                                                           \
    }                                                                          \
  }

#define ATOMIC_GENOP_FC_REV(_op)                                               \
  EXTERN void __kmpc_atomic_cmplx4_##_op##_rev(                                \
      kmp_Indent *id_ref, int32_t gtid, float complex *lhs,                    \
      float complex rhs) {                                                     \
    uint64_t *temp_lhs = (uint64_t *)lhs;                                      \
    uint64_t oldvalue = *temp_lhs;                                             \
    uint64_t saved;                                                            \
    float *pr1 = (float *)&rhs;                                                \
    float *pi1 = pr1 + 1;                                                      \
    float r1 = *pr1;                                                           \
    float i1 = *pi1;                                                           \
    uint64_t newvalue;                                                         \
    do {                                                                       \
      saved = oldvalue;                                                        \
      float *pr2 = (float *)&saved;                                            \
      float *pi2 = pr2 + 1;                                                    \
      newvalue = fc_##_op(r1, i1, *pr2, *pi2);                                 \
      oldvalue = atomicCAS(temp_lhs, saved, newvalue);                         \
    } while (saved != oldvalue);                                               \
  }                                                                            \
                                                                               \
  EXTERN void __kmpc_atomic_cmplx4_##_op##_cpt##_rev(                          \
      kmp_Indent *id_ref, int32_t gtid, float complex *lhs, float complex rhs, \
      float complex *outp, int flag) {                                         \
    uint64_t *temp_lhs = (uint64_t *)lhs;                                      \
    uint64_t oldvalue = *temp_lhs;                                             \
    uint64_t saved;                                                            \
    float *pr1 = (float *)&rhs;                                                \
    float *pi1 = pr1 + 1;                                                      \
    float r1 = *pr1;                                                           \
    float i1 = *pi1;                                                           \
    uint64_t newvalue;                                                         \
    do {                                                                       \
      saved = oldvalue;                                                        \
      float *pr2 = (float *)&saved;                                            \
      float *pi2 = pr2 + 1;                                                    \
      newvalue = fc_##_op(r1, i1, *pr2, *pi2);                                 \
      oldvalue = atomicCAS(temp_lhs, saved, newvalue);                         \
    } while (saved != oldvalue);                                               \
    if (flag) {                                                                \
      float complex *temp = (float complex *)&newvalue;                        \
      *outp = *temp;                                                           \
    } else {                                                                   \
      float complex *temp = (float complex *)&saved;                           \
      *outp = *temp;                                                           \
    }                                                                          \
  }

ATOMIC_GENOP_FC(add);
ATOMIC_GENOP_FC(sub);
ATOMIC_GENOP_FC_REV(sub);
ATOMIC_GENOP_FC(mul);
ATOMIC_GENOP_FC(div);
ATOMIC_GENOP_FC_REV(div);

// for int and uint
#define ATOMIC_GENOP_ALL_MIXED(_name, _dirname, _tname, _optype)               \
  _dirname(_tname, _optype, add, Add);                                         \
  _dirname(_tname, _optype, sub, Sub);                                         \
  _name##_REV(_tname, _optype, sub);                                           \
  _name(_tname, _optype, mul);                                                 \
  _name(_tname, _optype, div);                                                 \
  _name##_REV(_tname, _optype, div);                                           \
  _dirname(_tname, _optype, min, Min);                                         \
  _dirname(_tname, _optype, max, Max);                                         \
  _dirname(_tname, _optype, andb, And);                                        \
  _dirname(_tname, _optype, orb, Or);                                          \
  _dirname(_tname, _optype, xor, Xor);                                         \
  _name(_tname, _optype, shl);                                                 \
  _name(_tname, _optype, shr);                                                 \
  _name(_tname, _optype, andl);                                                \
  _name(_tname, _optype, orl);                                                 \
  _name(_tname, _optype, eqv);                                                 \
  _name(_tname, _optype, neqv);

#define ATOMIC_GENOP_ALL(_name, _tname, _optype)                               \
  _name(_tname, _optype, add);                                                 \
  _name(_tname, _optype, sub);                                                 \
  _name##_REV(_tname, _optype, sub);                                           \
  _name(_tname, _optype, mul);                                                 \
  _name(_tname, _optype, div);                                                 \
  _name##_REV(_tname, _optype, div);                                           \
  _name(_tname, _optype, min);                                                 \
  _name(_tname, _optype, max);                                                 \
  _name(_tname, _optype, andb);                                                \
  _name(_tname, _optype, orb);                                                 \
  _name(_tname, _optype, xor);                                                 \
  _name(_tname, _optype, shl);                                                 \
  _name(_tname, _optype, shr);                                                 \
  _name(_tname, _optype, andl);                                                \
  _name(_tname, _optype, orl);                                                 \
  _name(_tname, _optype, eqv);                                                 \
  _name(_tname, _optype, neqv);

#define ATOMIC_GENOP_FLOAT(_name, _tname, _optype)                             \
  _name(_tname, _optype, add);                                                 \
  _name(_tname, _optype, sub);                                                 \
  _name##_REV(_tname, _optype, sub);                                           \
  _name(_tname, _optype, mul);                                                 \
  _name(_tname, _optype, div);                                                 \
  _name##_REV(_tname, _optype, div);                                           \
  _name(_tname, _optype, min);                                                 \
  _name(_tname, _optype, max);

ATOMIC_GENOP_ALL_MIXED(ATOMIC_GENOP_DIRECT, ATOMIC_GENOP_NATIVE, fixed4,
                       int32_t);
ATOMIC_GENOP_ALL_MIXED(ATOMIC_GENOP_DIRECT, ATOMIC_GENOP_NATIVE, fixed4u,
                       uint32_t);
ATOMIC_GENOP_ALL(ATOMIC_GENOP_DIRECT, fixed8, int64_t);
ATOMIC_GENOP_ALL(ATOMIC_GENOP_DIRECT, fixed8u, uint64_t);
ATOMIC_GENOP_FLOAT(ATOMIC_GENOP_DIRECT, float4, float);
ATOMIC_GENOP_FLOAT(ATOMIC_GENOP_DIRECT, float8, double);

//
// data type of size not 32 nor 64
//

typedef enum {
  omptarget_nvptx_inc,
  omptarget_nvptx_dec,
  omptarget_nvptx_add,
  omptarget_nvptx_sub,
  omptarget_nvptx_sub_rev,
  omptarget_nvptx_mul,
  omptarget_nvptx_div,
  omptarget_nvptx_div_rev,
  omptarget_nvptx_min,
  omptarget_nvptx_max,
  omptarget_nvptx_rd,
  omptarget_nvptx_wr,
  omptarget_nvptx_swp,
  omptarget_nvptx_andb,
  omptarget_nvptx_orb,
  omptarget_nvptx_xor,
  omptarget_nvptx_andl,
  omptarget_nvptx_orl,
  omptarget_nvptx_eqv,
  omptarget_nvptx_neqv,
  omptarget_nvptx_shl,
  omptarget_nvptx_shl_rev,
  omptarget_nvptx_shr,
  omptarget_nvptx_shr_rev,
} omptarget_nvptx_BINOP_t;

template <typename OpType,              // type of the operation performed
          omptarget_nvptx_BINOP_t binop // enum describing the operation
          >
INLINE __device__ OpType Compute(OpType a,
                                 OpType b) // a is old value, b is new value
{
  OpType res = 0;
  if (binop == omptarget_nvptx_inc)
    res = a + b;
  if (binop == omptarget_nvptx_dec)
    res = a - b;
  if (binop == omptarget_nvptx_add)
    res = a + b;
  if (binop == omptarget_nvptx_sub)
    res = a - b;
  if (binop == omptarget_nvptx_sub_rev)
    res = b - a;
  if (binop == omptarget_nvptx_mul)
    res = a * b;
  if (binop == omptarget_nvptx_div)
    res = a / b;
  if (binop == omptarget_nvptx_div_rev)
    res = b / a;
  if (binop == omptarget_nvptx_min)
    res = a < b ? a : b;
  if (binop == omptarget_nvptx_max)
    res = a > b ? a : b;
  if (binop == omptarget_nvptx_rd)
    res = a; // read
  if (binop == omptarget_nvptx_wr)
    res = b; // write and swap are the same
  if (binop == omptarget_nvptx_swp)
    res = b; // write and swap are the same
  if (binop == omptarget_nvptx_andb)
    res = a & b;
  if (binop == omptarget_nvptx_orb)
    res = a | b;
  if (binop == omptarget_nvptx_xor)
    res = a ^ b;
  if (binop == omptarget_nvptx_andl)
    res = a && b;
  if (binop == omptarget_nvptx_orl)
    res = a || b;
  if (binop == omptarget_nvptx_eqv)
    res = a == b;
  if (binop == omptarget_nvptx_neqv)
    res = a != b;
  if (binop == omptarget_nvptx_shl)
    res = a << b;
  if (binop == omptarget_nvptx_shl_rev)
    res = b << a;
  if (binop == omptarget_nvptx_shr)
    res = a >> b;
  if (binop == omptarget_nvptx_shr_rev)
    res = b >> a;

  return res;
}

template <>
INLINE __device__ float Compute<float, omptarget_nvptx_add>(float a, float b) {
  return a + b;
}

template <>
INLINE __device__ float Compute<float, omptarget_nvptx_sub>(float a, float b) {
  return a - b;
}

template <>
INLINE __device__ float Compute<float, omptarget_nvptx_mul>(float a, float b) {
  return a * b;
}

template <>
INLINE __device__ float Compute<float, omptarget_nvptx_div>(float a, float b) {
  return a / b;
}

template <>
INLINE __device__ float Compute<float, omptarget_nvptx_min>(float a, float b) {
  return a < b ? a : b;
}

template <>
INLINE __device__ float Compute<float, omptarget_nvptx_max>(float a, float b) {
  return a > b ? a : b;
}

template <>
INLINE __device__ double Compute<double, omptarget_nvptx_add>(double a,
                                                              double b) {
  return a + b;
}

template <>
INLINE __device__ double Compute<double, omptarget_nvptx_sub>(double a,
                                                              double b) {
  return a - b;
}

template <>
INLINE __device__ double Compute<double, omptarget_nvptx_mul>(double a,
                                                              double b) {
  return a * b;
}

template <>
INLINE __device__ double Compute<double, omptarget_nvptx_div>(double a,
                                                              double b) {
  return a / b;
}

template <>
INLINE __device__ double Compute<double, omptarget_nvptx_min>(double a,
                                                              double b) {
  return a < b ? a : b;
}

template <>
INLINE __device__ double Compute<double, omptarget_nvptx_max>(double a,
                                                              double b) {
  return a > b ? a : b;
}

////////////////////////////////////////////////////////////////////////////////
// common atomic slicing functions (modifying only a part of a word)
////////////////////////////////////////////////////////////////////////////////

template <typename MemType, // type of the underlying atomic memory operation
          typename OpType   // type of the operation performed
          >
INLINE __device__ void ComputeAtomic_PrepareSlice(
    OpType *addr,         // original address
    MemType **memAddrPtr, // truncated address to MemType boundary
    MemType
        *memBitShiftRightPtr, // bits to shift to move val to rightmost position
    MemType *memValMaskInPlacePtr) // mask of val in proper position
{
  // compute the mask that corresponds to the natural alignment of memType
  // int -> 0x3; long long -> 0x7
  unsigned long memAddrMask = sizeof(MemType) - 1;
  // compute the addr of the atomic variable truncated to alignment of memType
  *memAddrPtr = (MemType *)((unsigned long)addr & ~memAddrMask);
  // compute the number of bit shift to move the target atomic value in
  // the rightmost position
  unsigned long byteOffsetInMem = (unsigned long)addr & memAddrMask;

  // assumes little-endian
  unsigned long byteShiftRight = byteOffsetInMem;
  *memBitShiftRightPtr = (MemType)(byteShiftRight << 3); // 3: byte to bits

  // mask to isolate target atomic value located in rightmost position
  MemType memValMask = ((MemType)1 << (sizeof(OpType) << 3)) - 1;
  // mask to isolate target atomic value located in place
  *memValMaskInPlacePtr = memValMask << *memBitShiftRightPtr;
}

template <typename MemType, // type of the underlying atomic memory operation
          typename OpType,  // type of the operation performed
          omptarget_nvptx_BINOP_t binop // enum describing the operation
          >
INLINE __device__ MemType ComputeAtomic_ComputeSlice(
    MemType oldMemVal,        // old value
    OpType val,               // value to compute with
    MemType memBitShiftRight, // bits to shift to move val to rightmost position
    MemType memValMaskInPlace // mask of val in proper position
    ) {
  OpType oldValtmp;
  OpType newValtmp;
  // select target atomic val
  MemType oldMemVal_targetVal = oldMemVal & memValMaskInPlace;
  MemType oldMemVal_otherVal = oldMemVal & ~memValMaskInPlace;
  // shift target atomic val to rightmost place: this is the old value

  // type conversion??
  oldValtmp = (OpType)(oldMemVal_targetVal >> memBitShiftRight);
  // perform op

  newValtmp = Compute<OpType, binop>(oldValtmp, val);

  // insert new value in old world mem

  // type conversion??
  MemType newMemVal_targetVal = ((MemType)newValtmp) << memBitShiftRight;
  newMemVal_targetVal &= memValMaskInPlace;
  MemType newMemVal = oldMemVal_otherVal | newMemVal_targetVal;
  return newMemVal;
}

#define ATOMIC_GENOP_PARTIAL(_name, _dtype, _op, _memType)                     \
  EXTERN void __kmpc_atomic_##_name##_##_op(kmp_Indent *id_ref, int32_t gtid,  \
                                            _dtype *lhs, _dtype rhs) {         \
    _memType *memAddr;                                                         \
    _memType memBitShiftRightPtr;                                              \
    _memType memValMaskInPlacePtr;                                             \
    ComputeAtomic_PrepareSlice<_memType, _dtype>(                              \
        lhs, &memAddr, &memBitShiftRightPtr, &memValMaskInPlacePtr);           \
    _memType oldMemVal, newMemVal;                                             \
    oldMemVal = *memAddr;                                                      \
    _memType savedMemVal;                                                      \
    do {                                                                       \
      savedMemVal = oldMemVal;                                                 \
      newMemVal =                                                              \
          ComputeAtomic_ComputeSlice<_memType, _dtype, omptarget_nvptx_##_op>( \
              oldMemVal, rhs, memBitShiftRightPtr, memValMaskInPlacePtr);      \
      oldMemVal = atomicCAS(memAddr, savedMemVal, newMemVal);                  \
    } while (savedMemVal != oldMemVal);                                        \
  }                                                                            \
                                                                               \
  EXTERN _dtype __kmpc_atomic_##_name##_##_op##_cpt(                           \
      kmp_Indent *id_ref, int32_t gtid, _dtype *lhs, _dtype rhs, int flag) {   \
    _memType *memAddr;                                                         \
    _memType memBitShiftRightPtr;                                              \
    _memType memValMaskInPlacePtr;                                             \
    ComputeAtomic_PrepareSlice<_memType, _dtype>(                              \
        lhs, &memAddr, &memBitShiftRightPtr, &memValMaskInPlacePtr);           \
    _memType oldMemVal, newMemVal;                                             \
    oldMemVal = *memAddr;                                                      \
    _memType savedMemVal;                                                      \
    do {                                                                       \
      savedMemVal = oldMemVal;                                                 \
      newMemVal =                                                              \
          ComputeAtomic_ComputeSlice<_memType, _dtype, omptarget_nvptx_##_op>( \
              oldMemVal, rhs, memBitShiftRightPtr, memValMaskInPlacePtr);      \
      oldMemVal = atomicCAS(memAddr, savedMemVal, newMemVal);                  \
    } while (savedMemVal != oldMemVal);                                        \
    if (flag)                                                                  \
      return (_dtype)((newMemVal & memValMaskInPlacePtr) >>                    \
                      memBitShiftRightPtr);                                    \
    else                                                                       \
      return (_dtype)((oldMemVal & memValMaskInPlacePtr) >>                    \
                      memBitShiftRightPtr);                                    \
  }

#define ATOMIC_GENOP_PARTIAL_REV(_name, _dtype, _op, _memType)                 \
  EXTERN void __kmpc_atomic_##_name##_##_op##_rev(                             \
      kmp_Indent *id_ref, int32_t gtid, _dtype *lhs, _dtype rhs) {             \
    _memType *memAddr;                                                         \
    _memType memBitShiftRightPtr;                                              \
    _memType memValMaskInPlacePtr;                                             \
    ComputeAtomic_PrepareSlice<_memType, _dtype>(                              \
        lhs, &memAddr, &memBitShiftRightPtr, &memValMaskInPlacePtr);           \
    _memType oldMemVal, newMemVal;                                             \
    oldMemVal = *memAddr;                                                      \
    _memType savedMemVal;                                                      \
    do {                                                                       \
      savedMemVal = oldMemVal;                                                 \
      newMemVal =                                                              \
          ComputeAtomic_ComputeSlice<_memType, _dtype, omptarget_nvptx_##_op>( \
              oldMemVal, rhs, memBitShiftRightPtr, memValMaskInPlacePtr);      \
      oldMemVal = atomicCAS(memAddr, savedMemVal, newMemVal);                  \
    } while (savedMemVal != oldMemVal);                                        \
  }                                                                            \
                                                                               \
  EXTERN _dtype __kmpc_atomic_##_name##_##_op##_cpt_rev(                       \
      kmp_Indent *id_ref, int32_t gtid, _dtype *lhs, _dtype rhs, int flag) {   \
    _memType *memAddr;                                                         \
    _memType memBitShiftRightPtr;                                              \
    _memType memValMaskInPlacePtr;                                             \
    ComputeAtomic_PrepareSlice<_memType, _dtype>(                              \
        lhs, &memAddr, &memBitShiftRightPtr, &memValMaskInPlacePtr);           \
    _memType oldMemVal, newMemVal;                                             \
    oldMemVal = *memAddr;                                                      \
    _memType savedMemVal;                                                      \
    do {                                                                       \
      savedMemVal = oldMemVal;                                                 \
      newMemVal =                                                              \
          ComputeAtomic_ComputeSlice<_memType, _dtype, omptarget_nvptx_##_op>( \
              oldMemVal, rhs, memBitShiftRightPtr, memValMaskInPlacePtr);      \
      oldMemVal = atomicCAS(memAddr, savedMemVal, newMemVal);                  \
    } while (savedMemVal != oldMemVal);                                        \
    if (flag)                                                                  \
      return (_dtype)((newMemVal & memValMaskInPlacePtr) >>                    \
                      memBitShiftRightPtr);                                    \
    else                                                                       \
      return (_dtype)((oldMemVal & memValMaskInPlacePtr) >>                    \
                      memBitShiftRightPtr);                                    \
  }

#define ATOMIC_GENOP_ALL4(_name, _tname, _optype, _memtype)                    \
  _name(_tname, _optype, add, _memtype);                                       \
  _name(_tname, _optype, sub, _memtype);                                       \
  _name##_REV(_tname, _optype, sub_rev, _memtype);                             \
  _name(_tname, _optype, mul, _memtype);                                       \
  _name(_tname, _optype, div, _memtype);                                       \
  _name##_REV(_tname, _optype, div_rev, _memtype);                             \
  _name(_tname, _optype, min, _memtype);                                       \
  _name(_tname, _optype, max, _memtype);                                       \
  _name(_tname, _optype, andb, _memtype);                                      \
  _name(_tname, _optype, orb, _memtype);                                       \
  _name(_tname, _optype, xor, _memtype);                                       \
  _name(_tname, _optype, andl, _memtype);                                      \
  _name(_tname, _optype, orl, _memtype);                                       \
  _name(_tname, _optype, eqv, _memtype);                                       \
  _name(_tname, _optype, neqv, _memtype);                                      \
  _name(_tname, _optype, shl, _memtype);                                       \
  _name(_tname, _optype, shr, _memtype);

ATOMIC_GENOP_ALL4(ATOMIC_GENOP_PARTIAL, fixed1, int8_t, int32_t);
ATOMIC_GENOP_ALL4(ATOMIC_GENOP_PARTIAL, fixed1u, uint8_t, int32_t);
ATOMIC_GENOP_ALL4(ATOMIC_GENOP_PARTIAL, fixed2u, uint16_t, int32_t);
ATOMIC_GENOP_ALL4(ATOMIC_GENOP_PARTIAL, fixed2, int16_t, int32_t);

EXTERN int32_t __kmpc_shuffle_int32(int32_t val, int16_t delta, int16_t size) {
  return __shfl_down(val, delta, size);
}

EXTERN int64_t __kmpc_shuffle_int64(int64_t val, int16_t delta, int16_t size) {
  int lo, hi;
  asm volatile("mov.b64 {%0,%1}, %2;" : "=r"(lo), "=r"(hi) : "l"(val));
  hi = __shfl_down(hi, delta, size);
  lo = __shfl_down(lo, delta, size);
  asm volatile("mov.b64 %0, {%1,%2};" : "=l"(val) : "r"(lo), "r"(hi));
  return val;
}

template <typename T, omptarget_nvptx_BINOP_t binop>
__inline__ __device__ T reduInitVal() {
  switch (binop) {
  case omptarget_nvptx_inc:
  case omptarget_nvptx_dec:
  case omptarget_nvptx_add:
  case omptarget_nvptx_sub:
  case omptarget_nvptx_sub_rev:
    return (T)0;
  case omptarget_nvptx_mul:
  case omptarget_nvptx_div:
    return (T)1;
  default:
    return (T)0;
  }
}

#define MYGSIZE 32

static INLINE void gpu_regular_warp_reduce(void *reduce_data, kmp_ShuffleReductFctPtr shflFct) {
  for (uint32_t mask = WARPSIZE/2; mask > 0; mask /= 2) {
    shflFct(reduce_data, /*LaneId - not used= */0,
            /*Offset = */mask, /*AlgoVersion=*/0);
  }
}

static INLINE void gpu_irregular_warp_reduce(void *reduce_data, kmp_ShuffleReductFctPtr shflFct,
                                             uint32_t size, uint32_t tid) {
  uint32_t curr_size;
  uint32_t mask;
  curr_size = size; 
  mask = curr_size/2;
  while (mask>0) {
    shflFct(reduce_data, /*LaneId = */tid, /*Offset=*/mask, /*AlgoVersion=*/1);
    curr_size = (curr_size+1)/2;
    mask = curr_size/2;
  }
}

static INLINE uint32_t gpu_irregular_simd_reduce(void *reduce_data, kmp_ShuffleReductFctPtr shflFct) {
  uint32_t lanemask_lt;
  uint32_t lanemask_gt;
  uint32_t size, remote_id, physical_lane_id;
  physical_lane_id = GetThreadIdInBlock() % WARPSIZE;
  asm("mov.u32 %0, %%lanemask_lt;" : "=r"(lanemask_lt));
  uint32_t Liveness = __ballot(true);
  uint32_t logical_lane_id = __popc(Liveness & lanemask_lt) * 2;
  asm("mov.u32 %0, %%lanemask_gt;" : "=r"(lanemask_gt));
  do {
    Liveness = __ballot(true);
    remote_id = __ffs(Liveness & lanemask_gt);
    size = __popc(Liveness);
    logical_lane_id /= 2;
    shflFct(reduce_data, /*LaneId =*/logical_lane_id,
            /*Offset=*/remote_id-1-physical_lane_id, /*AlgoVersion=*/2);
  } while (logical_lane_id % 2 == 0 && size > 1);
  return (logical_lane_id == 0);
}

//
// runtime support for array reduction
//

#define ARRAYATOMIC_GENOP(_name, _dtype, _op)                                  \
  EXTERN void __array_atomic_##_name##_##_op(                                  \
      kmp_Indent *id_ref, int32_t gtid, _dtype *lhs, _dtype *rhs, int64_t n) { \
    PRINT(LD_LOOP, "Reduction: thead %d\n", gtid);                             \
    for (int i = 0; i < n / sizeof(_dtype); i++) {                             \
      __kmpc_atomic_##_name##_##_op(id_ref, gtid, lhs + i, rhs[i]);            \
    }                                                                          \
  }                                                                            \
  EXTERN void __gpu_array_warpBlockRedu_##_name##_##_op(_dtype *ldata,         \
                                                        int64_t n) {           \
    for (int i = 0; i < n / sizeof(_dtype); i++) {                             \
      ldata[i] = __gpu_warpBlockRedu_##_name##_##_op(ldata[i]);                \
    }                                                                          \
  }

#define ARRAY_GEN_ALLOP_INTEGER(_name, _tname, _optype)                        \
  _name(_tname, _optype, add);                                                 \
  _name(_tname, _optype, sub);                                                 \
  _name(_tname, _optype, mul);                                                 \
  _name(_tname, _optype, div);                                                 \
  _name(_tname, _optype, min);                                                 \
  _name(_tname, _optype, max);                                                 \
  _name(_tname, _optype, andb);                                                \
  _name(_tname, _optype, orb);                                                 \
  _name(_tname, _optype, xor);                                                 \
  _name(_tname, _optype, shl);                                                 \
  _name(_tname, _optype, shr);                                                 \
  _name(_tname, _optype, andl);                                                \
  _name(_tname, _optype, orl);                                                 \
  _name(_tname, _optype, eqv);                                                 \
  _name(_tname, _optype, neqv);

#define ARRAY_GEN_ALLOP_FLOAT(_name, _tname, _optype)                          \
  _name(_tname, _optype, add);                                                 \
  _name(_tname, _optype, sub);                                                 \
  _name(_tname, _optype, mul);                                                 \
  _name(_tname, _optype, div);                                                 \
  _name(_tname, _optype, min);                                                 \
  _name(_tname, _optype, max);

EXTERN
int32_t __kmpc_nvptx_simd_reduce_nowait(int32_t global_tid,
                                        int32_t num_vars, size_t reduce_size,
                                        void *reduce_data,
                                        kmp_ShuffleReductFctPtr shflFct,
                                        kmp_InterWarpCopyFctPtr cpyFct) {
  uint32_t Liveness = __ballot(true);
  if (Liveness == 0xffffffff) {
    gpu_regular_warp_reduce(reduce_data, shflFct);
    return GetThreadIdInBlock() % WARPSIZE == 0; // Result on lane 0 of the simd warp.
  } else {
    return gpu_irregular_simd_reduce(reduce_data, shflFct); // Result on the first active lane.
  }
}

EXTERN
int32_t __kmpc_nvptx_parallel_reduce_nowait(int32_t global_tid,
                                            int32_t num_vars, size_t reduce_size,
                                            void *reduce_data, 
                                            kmp_ShuffleReductFctPtr shflFct,
                                            kmp_InterWarpCopyFctPtr cpyFct) {
  /*
   * This reduce function handles reduction within a team. It handles
   * parallel regions in both L1 and L2 parallelism levels. It also
   * supports Generic, SPMD, and NoOMP modes.
   *
   * 1. Reduce within a warp.
   * 2. Warp master copies value to warp 0 via shared memory.
   * 3. Warp 0 reduces to a single value.
   * 4. The reduced value is available in the thread that returns 1.
   */
  uint32_t Liveness = __ballot(true);
  if (Liveness == 0xffffffff) // Full warp
    gpu_regular_warp_reduce(reduce_data, shflFct);
  else if (!(Liveness & (Liveness + 1))) // Partial warp but contiguous lanes
    gpu_irregular_warp_reduce(reduce_data, shflFct,
        /*LaneCount=*/__popc(Liveness), /*LaneId=*/GetThreadIdInBlock() % WARPSIZE);
  else // Dispersed lanes. Only threads in L2 parallel region may enter here; return early.
    return gpu_irregular_simd_reduce(reduce_data, shflFct);

  uint32_t BlockThreadId = GetLogicalThreadIdInBlock();
  uint32_t NumThreads = isNoOMPMode() ?
      GetNumberOfThreadsInBlock() : GetNumberOfOmpThreads(BlockThreadId);

  // When we have more than [warpsize] number of threads
  // a block reduction is performed here.
  //
  // Only L1 parallel region can enter this if condition.
  if (NumThreads > WARPSIZE) {
    uint32_t WarpsNeeded = (NumThreads+WARPSIZE-1)/WARPSIZE;
    // Gather all the reduced values from each warp
    // to the first warp.
    cpyFct(reduce_data, WarpsNeeded);

    uint32_t WarpId = BlockThreadId/WARPSIZE;
    if (WarpId == 0)
      gpu_irregular_warp_reduce(reduce_data, shflFct, WarpsNeeded, BlockThreadId);

    return BlockThreadId == 0;
  }

  // Get the OMP thread Id. This is different from BlockThreadId in the case of
  // an L2 parallel region.
  return (isNoOMPMode() ? BlockThreadId : GetOmpThreadId(BlockThreadId)) == 0;
}

// FIXME
__device__ int scratchpad[65536];
__device__ static unsigned timestamp = 0;

EXTERN
int32_t __kmpc_nvptx_teams_reduce_nowait(int32_t global_tid,
                             int32_t num_vars, size_t reduce_size,
                             void *reduce_data,
                             kmp_ShuffleReductFctPtr shflFct,
                             kmp_InterWarpCopyFctPtr cpyFct,
                             kmp_CopyToScratchpadFctPtr scratchFct,
                             kmp_LoadReduceFctPtr ldFct) {
  uint32_t ThreadId = GetLogicalThreadIdInBlock();
  uint32_t NumThreads = isNoOMPMode() ?
     GetNumberOfThreadsInBlock() : GetNumberOfOmpThreads(ThreadId);
  uint32_t TeamId = GetBlockIdInKernel();
  uint32_t NumTeams = GetNumberOfBlocksInKernel();
  __shared__ volatile bool IsLastTeam;

  // Team masters of all teams write to the scratchpad.
  if (ThreadId == 0) {
    scratchFct(reduce_data, scratchpad, TeamId, NumTeams);
    __threadfence();

    // atomicInc increments 'timestamp' and has a range [0, NumTeams-1].
    // It resets 'timestamp' back to 0 once the last team increments
    // this counter.
    unsigned val = atomicInc(&timestamp, NumTeams-1);
    IsLastTeam = val == NumTeams - 1;
  }

  // In non-generic mode all workers participate in the teams reduction.
  // In generic mode only the team master participates in the teams
  // reduction because the workers are waiting for parallel work.
  //
  // We have to wait on L1 barrier because in GENERIC mode the workers
  // are waiting on barrier 0 for work.
  //
  // If we guard this barrier as follows it leads to deadlock, probably
  // because of a compiler bug: if (!IsGenericMode()) __syncthreads();
  uint16_t SyncWarps = (NumThreads+WARPSIZE-1)/WARPSIZE;
  named_sync(L1_BARRIER, SyncWarps*WARPSIZE);

  // If this team is not the last, quit.
  if (/* Volatile read by all threads */ !IsLastTeam)
    return 0;

  //
  // Last team processing.
  //

  // Threads in excess of #teams do not participate in reduction of the
  // scratchpad values.
  if (ThreadId >= NumTeams)
    return 0;

  // Load from scratchpad and reduce.
  ldFct(reduce_data, scratchpad, ThreadId, NumTeams, /*Load only*/0);
  for (uint32_t i = NumThreads + ThreadId; i < NumTeams; i += NumThreads)
    ldFct(reduce_data, scratchpad, i, NumTeams, /*Load and reduce*/1);

  // Reduce across warps to the warp master.
  uint32_t Liveness = __ballot(true);
  if (Liveness == 0xffffffff) // Full warp
    gpu_regular_warp_reduce(reduce_data, shflFct);
  else // Partial warp but contiguous lanes
    gpu_irregular_warp_reduce(reduce_data, shflFct,
        /*LaneCount=*/__popc(Liveness), /*LaneId=*/ThreadId % WARPSIZE);

  // When we have more than [warpsize] number of threads
  // a block reduction is performed here.
  uint32_t ActiveThreads = NumTeams < NumThreads ? NumTeams : NumThreads;
  if (ActiveThreads > WARPSIZE) {
    uint32_t WarpsNeeded = (ActiveThreads+WARPSIZE-1)/WARPSIZE;
    // Gather all the reduced values from each warp
    // to the first warp.
    cpyFct(reduce_data, WarpsNeeded);

    uint32_t WarpId = ThreadId/WARPSIZE;
    if (WarpId == 0)
      gpu_irregular_warp_reduce(reduce_data, shflFct, WarpsNeeded, ThreadId);
  }

  return ThreadId == 0;
}

