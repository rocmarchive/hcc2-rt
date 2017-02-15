//===------------ option.h - NVPTX OpenMP GPU options ------------ CUDA -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.txt for details.
//
//===----------------------------------------------------------------------===//
//
// GPU default options
//
//===----------------------------------------------------------------------===//
#ifndef _OPTION_H_
#define _OPTION_H_

////////////////////////////////////////////////////////////////////////////////
// Kernel options
////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////
// following two defs must match absolute limit hardwired in the host RTL
#define TEAMS_ABSOLUTE_LIMIT                                                   \
  512 /* omptx limit (must match teamsAbsoluteLimit) */
#define THREAD_ABSOLUTE_LIMIT                                                  \
  1024 /* omptx limit (must match threadAbsoluteLimit) */

// max number of thread per team
#define MAX_THREADS_PER_TEAM THREAD_ABSOLUTE_LIMIT

// max number of blocks depend on the kernel we are executing - pick default
// here
#define MAX_NUM_TEAMS TEAMS_ABSOLUTE_LIMIT
#define WARPSIZE 32
#define MAX_NUM_WARPS (MAX_NUM_TEAMS * THREAD_ABSOLUTE_LIMIT / WARPSIZE)
#define MAX_NUM_THREADS MAX_NUM_WARPS *WARPSIZE

#ifdef OMPTHREAD_IS_WARP
// assume here one OpenMP thread per CUDA warp
#define MAX_NUM_OMP_THREADS MAX_NUM_WARPS
#else
// assume here one OpenMP thread per CUDA thread
#define MAX_NUM_OMP_THREADS MAX_NUM_THREADS
#endif

// The named barrier for active parallel threads of a team in an L1 parallel region
// to synchronize with each other.
#define L1_BARRIER (1)

// Maximum number of omp state objects per SM allocated statically in global memory.
#if __CUDA_ARCH__ >= 600
#define OMP_STATE_COUNT 32
#define MAX_SM 56
#else
#define OMP_STATE_COUNT 16
#define MAX_SM 16
#endif

////////////////////////////////////////////////////////////////////////////////
// algo options
////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////
// data options
////////////////////////////////////////////////////////////////////////////////

// decide if counters are 32 or 64 bit
#define Counter unsigned long long

// TODO: KMP defines kmp_int to be 32 or 64 bits depending on the target.
// think we don't need it here (meaning we can be always 64 bit compatible)
//
// #ifdef KMP_I8
//   typedef kmp_int64		kmp_int;
// #else
//   typedef kmp_int32		kmp_int;
// #endif

////////////////////////////////////////////////////////////////////////////////
// misc options (by def everythig here is device)
////////////////////////////////////////////////////////////////////////////////

#define EXTERN extern "C" __device__
#define INLINE __inline__ __device__
#define NOINLINE __noinline__ __device__
#ifndef TRUE
#define TRUE 1
#endif
#ifndef FALSE
#define FALSE 0
#endif

#endif
