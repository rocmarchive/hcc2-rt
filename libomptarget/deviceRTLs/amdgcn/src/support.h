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

#ifdef GPUCC_AMDGCN
// function defined in cuda header
EXTERN void __device_trap();

// time function
EXTERN int32_t __clock();
EXTERN int64_t __clock64();

// CU id
EXTERN unsigned int __smid() ;

// named sync
EXTERN void __named_sync(const int barrier, const int num_threads);

// warp vote function
EXTERN uint64_t __ballot64(int predicate);
// initialized with a 64-bit mask with bits set in positions less than the thread's lane number in the warp
EXTERN uint64_t __lanemask_lt();
// initialized with a 64-bit mask with bits set in positions greater than the thread's lane number in the warp
EXTERN uint64_t __lanemask_gt();

// memory allocation
EXTERN void* __malloc(size_t);
EXTERN void __free(void*);
#endif

#ifdef GPUCC_AMDGCN
#ifdef clock
#undef clock
#endif
#define clock __clock

#ifdef clock64
#undef clock64
#endif
#define clock64 __clock64

#ifdef malloc
#undef malloc
#endif
#define malloc __malloc

#ifdef free
#undef free
#endif
#define free __free
#endif

////////////////////////////////////////////////////////////////////////////////
// Execution Parameters
////////////////////////////////////////////////////////////////////////////////
enum ExecutionMode {
  Generic = 0x00u,
  Spmd = 0x01u,
  ModeMask = 0x01u,
};

enum RuntimeMode {
  RuntimeInitialized = 0x00u,
  RuntimeUninitialized = 0x02u,
  RuntimeMask = 0x02u,
};

INLINE void setExecutionParameters(ExecutionMode EMode, RuntimeMode RMode);
INLINE bool isGenericMode();
INLINE bool isSPMDMode();

//INLINE void setNoOMPMode();
//INLINE bool isNoOMPMode();

INLINE bool isRuntimeUninitialized();
INLINE bool isRuntimeInitialized();

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
INLINE int GetOmpThreadId(int threadId, bool isSPMDExecutionMode,
                          bool isRuntimeUninitialized); // omp_thread_num
INLINE int GetOmpTeamId();               // omp_team_num

// get OpenMP number of threads and team
INLINE int GetNumberOfOmpThreads(int threadId, bool isSPMDExecutionMode,
                                 bool isRuntimeUninitialized); // omp_num_threads
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
