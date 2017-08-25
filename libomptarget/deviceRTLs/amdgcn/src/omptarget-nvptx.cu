//===--- omptarget-nvptx.cu - NVPTX OpenMP GPU initialization ---- CUDA -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.txt for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains the initialization code for the GPU
//
//===----------------------------------------------------------------------===//

#include "omptarget-nvptx.h"

////////////////////////////////////////////////////////////////////////////////
// init entry points
////////////////////////////////////////////////////////////////////////////////

#ifdef GPUCC_AMDGCN
INLINE unsigned smid() {
  // return __smid();
  // For amdgcn, use a virtual smid based on global thread number
  unsigned id = ((blockIdx.x*blockDim.x) + threadIdx.x) / warpSize;
  PRINT(LD_IO, "smid() returns value %d with MAX_SM %d\n",id,MAX_SM);
  return id;
}
#else
INLINE unsigned smid() {
  unsigned id;
  asm("mov.u32 %0, %%smid;" : "=r"(id));
  return id;
}

INLINE unsigned n_sm() {
  unsigned n_sm;
  asm("mov.u32 %0, %%nsmid;" : "=r"(n_sm));
  return n_sm;
}
#endif

EXTERN void __kmpc_kernel_init(int ThreadLimit,
                               int16_t RequiresOMPRuntime) {
  PRINT(LD_IO, "call to __kmpc_kernel_init with version %f, threadlimit %d\n",
      OMPTARGET_NVPTX_VERSION, ThreadLimit);

  if (!RequiresOMPRuntime) {
    PRINT0(LD_IO, "OMP runtime not required\n");
    // If OMP runtime is not required don't initialize OMP state.
    setExecutionParameters(Generic, RuntimeUninitialized);
    return;
  }
  setExecutionParameters(Generic, RuntimeInitialized);

  int threadIdInBlock = GetThreadIdInBlock();
  ASSERT0(LT_FUSSY, threadIdInBlock == GetMasterThreadID(),
          "__kmpc_kernel_init() must be called by team master warp only!");
  PRINT0(LD_IO, "call to __kmpc_kernel_init for master\n");

  // Get a state object from the queue.
  int slot = smid() % MAX_SM;
  omptarget_nvptx_threadPrivateContext = omptarget_nvptx_device_State[slot].Dequeue();

  // init thread private
  int threadId = GetLogicalThreadIdInBlock();
  omptarget_nvptx_threadPrivateContext->InitThreadPrivateContext(
      threadId);

  // init team context
  omptarget_nvptx_TeamDescr &currTeamDescr = getMyTeamDescriptor();
  currTeamDescr.InitTeamDescr();
  // this thread will start execution... has to update its task ICV
  // to point to the level zero task ICV. That ICV was init in
  // InitTeamDescr()
  omptarget_nvptx_threadPrivateContext->SetTopLevelTaskDescr(
      threadId, currTeamDescr.LevelZeroTaskDescr());

  // set number of threads and thread limit in team to started value
  omptarget_nvptx_TaskDescr *currTaskDescr =
      omptarget_nvptx_threadPrivateContext->GetTopLevelTaskDescr(
          threadId);
  currTaskDescr->NThreads() = GetNumberOfWorkersInTeam();
  currTaskDescr->ThreadLimit() = ThreadLimit;
}

EXTERN void __kmpc_kernel_deinit(int16_t IsOMPRuntimeInitialized) {
  PRINT(LD_IO, "call to __kmpc_kernel_deinit, IsOMPRuntimeInitialized %d\n",
      IsOMPRuntimeInitialized);

  if (IsOMPRuntimeInitialized) {
    // Enqueue omp state object for use by another team.
    int slot = smid() % MAX_SM;
    omptarget_nvptx_device_State[slot].Enqueue(omptarget_nvptx_threadPrivateContext);
  }
  // Done with work.  Kill the workers.
  omptarget_nvptx_workFn = 0;
}

EXTERN void __kmpc_spmd_kernel_init(int ThreadLimit,
                                    int16_t RequiresOMPRuntime,
                                    int16_t RequiresDataSharing) {
  PRINT(LD_IO, "call to __kmpc_spmd_kernel_init with version %f, threadlimit %d\n",
      OMPTARGET_NVPTX_VERSION, ThreadLimit);

  if (!RequiresOMPRuntime) {
    PRINT0(LD_IO, "OMP runtime not required\n");
    // If OMP runtime is not required don't initialize OMP state.
    setExecutionParameters(Spmd, RuntimeUninitialized);
    return;
  }
  setExecutionParameters(Spmd, RuntimeInitialized);

  //
  // Team Context Initialization.
  //
  // In SPMD mode there is no master thread so use any cuda thread for team
  // context initialization.
  int threadId = GetThreadIdInBlock();
  if (threadId == 0) {
    // Get a state object from the queue.
    int slot = smid() % MAX_SM;
    omptarget_nvptx_threadPrivateContext = omptarget_nvptx_device_State[slot].Dequeue();

    omptarget_nvptx_TeamDescr &currTeamDescr = getMyTeamDescriptor();
    omptarget_nvptx_WorkDescr &workDescr = getMyWorkDescriptor();
    // init team context
    currTeamDescr.InitTeamDescr();
    // init counters (copy start to init)
    workDescr.CounterGroup().Reset();
  }
  __syncthreads();

  omptarget_nvptx_TeamDescr &currTeamDescr = getMyTeamDescriptor();
  omptarget_nvptx_WorkDescr &workDescr = getMyWorkDescriptor();

  //
  // Initialize task descr for each thread.
  //
  omptarget_nvptx_TaskDescr *newTaskDescr =
      omptarget_nvptx_threadPrivateContext->Level1TaskDescr(threadId);
  ASSERT0(LT_FUSSY, newTaskDescr, "expected a task descr");
  newTaskDescr->InitLevelOneTaskDescr(
    ThreadLimit, currTeamDescr.LevelZeroTaskDescr());
  // install new top descriptor
  omptarget_nvptx_threadPrivateContext->SetTopLevelTaskDescr(threadId,
                                                             newTaskDescr);

  // init thread private from init value
  workDescr.CounterGroup().Init(
      omptarget_nvptx_threadPrivateContext->Priv(threadId));
  PRINT(LD_PAR, "thread will execute parallel region with id %d in a team of "
                "%d threads\n",
                newTaskDescr->ThreadId(), newTaskDescr->NThreads());

  if (RequiresDataSharing && threadId % WARPSIZE == 0) {
    // Warp master innitializes data sharing environment.
    unsigned WID = threadId >> DS_Max_Worker_Warp_Size_Log2;
    __kmpc_data_sharing_slot *RootS = currTeamDescr.RootS(WID);
    DataSharingState.SlotPtr[WID] = RootS;
    DataSharingState.StackPtr[WID] = (void*)&RootS->Data[0];
  }
}

EXTERN void __kmpc_spmd_kernel_deinit() {
  PRINT0(LD_IO, "call to __kmpc_spmd_kernel_deinit\n");

  // We're not going to pop the task descr stack of each thread since
  // there are no more parallel regions in SPMD mode.
  __syncthreads();
  int threadId = GetThreadIdInBlock();
  if (threadId == 0) {
    // Enqueue omp state object for use by another team.
    int slot = smid() % MAX_SM;
    omptarget_nvptx_device_State[slot].Enqueue(omptarget_nvptx_threadPrivateContext);
  }
}
