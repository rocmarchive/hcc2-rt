//===--------- supporti.h - NVPTX OpenMP support functions ------- CUDA -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.txt for details.
//
//===----------------------------------------------------------------------===//
//
// Wrapper implementation to some functions natively supported by the GPU.
//
//===----------------------------------------------------------------------===//

////////////////////////////////////////////////////////////////////////////////
// Mode of Operation
////////////////////////////////////////////////////////////////////////////////

namespace {
enum EXECUTION_MODE {
  GENERIC = 0,
  SPMD = 1,
  NO_OMP = 2,
};
};

INLINE void setGenericMode() {
  execution_mode = GENERIC;
}

INLINE bool isGenericMode() {
  return execution_mode == GENERIC;
}

INLINE void setSPMDMode() {
  execution_mode = SPMD;
}

INLINE bool isSPMDMode() {
  return execution_mode == SPMD;
}

INLINE void setNoOMPMode() {
  // Minimal OMP mode.  Will not have OMP state.
  execution_mode = NO_OMP;
}

INLINE bool isNoOMPMode() {
  return execution_mode == NO_OMP;
}

////////////////////////////////////////////////////////////////////////////////
// support: get info from machine
////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////
//
// Calls to the NVPTX layer  (assuming 1D layout)
//
////////////////////////////////////////////////////////////////////////////////

INLINE int GetThreadIdInBlock() { return threadIdx.x; }

INLINE int GetBlockIdInKernel() { return blockIdx.x; }

INLINE int GetNumberOfBlocksInKernel() { return gridDim.x; }

INLINE int GetNumberOfThreadsInBlock() { return blockDim.x; }

////////////////////////////////////////////////////////////////////////////////
//
// Calls to the Generic Scheme Implementation Layer (assuming 1D layout)
//
////////////////////////////////////////////////////////////////////////////////

// The master thread id is the first thread (lane) of the last warp.
// Thread id is 0 indexed.
// E.g: If NumThreads is 33, master id is 32.
//      If NumThreads is 64, master id is 32.
//      If NumThreads is 97, master id is 96.
//      If NumThreads is 1024, master id is 992.
//
// Called in Generic Execution Mode only.
INLINE int GetMasterThreadID() { return (blockDim.x - 1) & ~(warpSize - 1); }

// The last warp is reserved for the master; other warps are workers.
// Called in Generic Execution Mode only.
INLINE int GetNumberOfWorkersInTeam() { return GetMasterThreadID(); }

////////////////////////////////////////////////////////////////////////////////
// get thread id in team

// This function may be called in a parallel region by the workers
// or a serial region by the master.  If the master (whose CUDA thread
// id is GetMasterThreadID()) calls this routine, we return 0 because
// it is a shadow for the first worker.
INLINE int GetLogicalThreadIdInBlock() {
//  return GetThreadIdInBlock() % GetMasterThreadID();

  // Implemented using control flow (predication) instead of with a modulo
  // operation.
  int tid = GetThreadIdInBlock();
  if (isGenericMode() && tid >= GetMasterThreadID())
    return 0;
  else
    return tid;
}

////////////////////////////////////////////////////////////////////////////////
//
// OpenMP Thread Support Layer
//
////////////////////////////////////////////////////////////////////////////////

INLINE int GetOmpThreadId(int threadId) {
  // omp_thread_num
  omptarget_nvptx_TaskDescr *currTaskDescr =
      omptarget_nvptx_threadPrivateContext->GetTopLevelTaskDescr(
          threadId);
  int rc = currTaskDescr->ThreadId();
  return rc;
}

INLINE int GetNumberOfOmpThreads(int threadId) {
  // omp_num_threads
  omptarget_nvptx_TaskDescr *currTaskDescr =
      omptarget_nvptx_threadPrivateContext->GetTopLevelTaskDescr(
          threadId);

  ASSERT0(LT_FUSSY, currTaskDescr, "expected a top task descr");

  int rc = currTaskDescr->ThreadsInTeam();
  return rc;
}

////////////////////////////////////////////////////////////////////////////////
// Team id linked to OpenMP

INLINE int GetOmpTeamId() {
  // omp_team_num
  return GetBlockIdInKernel(); // assume 1 block per team
}

INLINE int GetNumberOfOmpTeams() {
  // omp_num_teams
  return GetNumberOfBlocksInKernel(); // assume 1 block per team
}

////////////////////////////////////////////////////////////////////////////////
// Masters

INLINE int IsTeamMaster(int ompThreadId) { return (ompThreadId == 0); }

////////////////////////////////////////////////////////////////////////////////
// get OpenMP number of procs

// Get the number of processors in the device.
INLINE int GetNumberOfProcsInDevice() {
  if (isGenericMode())
    return GetNumberOfWorkersInTeam();
  else
    return GetNumberOfThreadsInBlock();
}

INLINE int GetNumberOfProcsInTeam() {
  return GetNumberOfProcsInDevice();
}

////////////////////////////////////////////////////////////////////////////////
// Memory
////////////////////////////////////////////////////////////////////////////////

INLINE unsigned long PadBytes(unsigned long size,
                              unsigned long alignment) // must be a power of 2
{
  // compute the necessary padding to satify alignment constraint
  ASSERT(LT_FUSSY, (alignment & (alignment - 1)) == 0,
         "alignment %ld is not a power of 2\n", alignment);
  return (~(unsigned long)size + 1) & (alignment - 1);
}

INLINE void *SafeMalloc(size_t size, const char *msg) // check if success
{
  void *ptr = malloc(size);
  PRINT(LD_MEM, "malloc data of size %d for %s: 0x%llx\n", size, msg, P64(ptr));
  ASSERT(LT_SAFETY, ptr, "failed to allocate %d bytes for %s\n", size, msg);
  return ptr;
}

INLINE void *SafeFree(void *ptr, const char *msg) {
  PRINT(LD_MEM, "free data ptr 0x%llx for %s\n", P64(ptr), msg);
  free(ptr);
  return NULL;
}

////////////////////////////////////////////////////////////////////////////////
// Named Barrier Routines
////////////////////////////////////////////////////////////////////////////////

INLINE void named_sync(const int barrier, const int num_threads) {
  asm volatile("bar.sync %0, %1;" : : "r"(barrier), "r"(num_threads) : "memory" );
}
