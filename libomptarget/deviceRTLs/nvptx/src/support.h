//===--------- support.h - NVPTX OpenMP support functions -------- CUDA -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.txt for details.
//
//===----------------------------------------------------------------------===//
//
// Wrapper to some functions natively supported by the GPU.
//
//===----------------------------------------------------------------------===//

////////////////////////////////////////////////////////////////////////////////
// Mode of Operation
////////////////////////////////////////////////////////////////////////////////
INLINE void setGenericMode();
INLINE bool isGenericMode();
INLINE void setSPMDMode();
INLINE bool isSPMDMode();
INLINE void setNoOMPMode();
INLINE bool isNoOMPMode();

////////////////////////////////////////////////////////////////////////////////
// get info from machine
////////////////////////////////////////////////////////////////////////////////

// get low level ids of resources
INLINE int GetThreadIdInBlock();
INLINE int GetBlockIdInKernel();
INLINE int GetNumberOfBlocksInKernel();
INLINE int GetNumberOfThreadsInBlock();

// get global ids to locate tread/team info (constant regardless of OMP)
INLINE int GetLogicalThreadIdInBlock();
INLINE int GetMasterThreadID();
INLINE int GetNumberOfWorkersInTeam();

// get OpenMP thread and team ids
INLINE int GetOmpThreadId(int threadId); // omp_thread_num
INLINE int GetOmpTeamId();               // omp_team_num

// get OpenMP number of threads and team
INLINE int GetNumberOfOmpThreads(int threadId); // omp_num_threads
INLINE int GetNumberOfOmpTeams();               // omp_num_teams

// get OpenMP number of procs
INLINE int GetNumberOfProcsInTeam();
INLINE int GetNumberOfProcsInDevice();

// masters
INLINE int IsTeamMaster(int ompThreadId);

////////////////////////////////////////////////////////////////////////////////
// Memory
////////////////////////////////////////////////////////////////////////////////

// safe alloc and free
INLINE void *SafeMalloc(size_t size, const char *msg); // check if success
INLINE void *SafeFree(void *ptr, const char *msg);
// pad to a alignment (power of 2 only)
INLINE unsigned long PadBytes(unsigned long size, unsigned long alignment);
#define ADD_BYTES(_addr, _bytes)                                               \
  ((void *)((char *)((void *)(_addr)) + (_bytes)))
#define SUB_BYTES(_addr, _bytes)                                               \
  ((void *)((char *)((void *)(_addr)) - (_bytes)))

////////////////////////////////////////////////////////////////////////////////
// Named Barrier Routines
////////////////////////////////////////////////////////////////////////////////
INLINE void named_sync(const int barrier, const int num_threads);
