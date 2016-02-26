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

#include "../../../deviceRTLs/nvptx/src/omptarget-nvptx.h"

////////////////////////////////////////////////////////////////////////////////
// global data tables
////////////////////////////////////////////////////////////////////////////////

__device__ omptarget_nvptx_TeamDescr
    omptarget_nvptx_device_teamContexts[MAX_INSTANCES][MAX_NUM_TEAMS];
__device__ omptarget_nvptx_ThreadPrivateContext
    omptarget_nvptx_device_threadPrivateContext[MAX_INSTANCES];
__device__ omptarget_nvptx_GlobalICV
    omptarget_nvptx_device_globalICV[MAX_INSTANCES];

__shared__ omptarget_nvptx_ThreadPrivateContext
    *omptarget_nvptx_threadPrivateContext;

////////////////////////////////////////////////////////////////////////////////
// init entry points
////////////////////////////////////////////////////////////////////////////////

EXTERN void __kmpc_kernel_init(int OmpHandle, int ThreadLimit) {
  PRINT(LD_IO, "call to __kmpc_kernel_init with version %f\n",
        OMPTARGET_NVPTX_VERSION);
  ASSERT0(LT_FUSSY, OmpHandle >= 0 && OmpHandle < MAX_INSTANCES,
          "omp handle out of bounds");
  omptarget_nvptx_threadPrivateContext =
      &omptarget_nvptx_device_threadPrivateContext[OmpHandle];
  omptarget_nvptx_threadPrivateContext->SetTeamContext(
      &omptarget_nvptx_device_teamContexts[OmpHandle][0]);
  omptarget_nvptx_threadPrivateContext->SetGlobalICV(
      &omptarget_nvptx_device_globalICV[OmpHandle]);

  // init thread private
  int globalThreadId = GetGlobalThreadId();
  omptarget_nvptx_threadPrivateContext->InitThreadPrivateContext(
      globalThreadId);

  int threadIdInBlock = GetThreadIdInBlock();
  if (threadIdInBlock == TEAM_MASTER) {
    PRINT0(LD_IO, "call to __kmpc_kernel_init for master\n");
    // init global icv
    omptarget_nvptx_threadPrivateContext->GlobalICV()->gpuCycleTime =
        1.0 / 745000000.0; // host reports 745 mHz
    omptarget_nvptx_threadPrivateContext->GlobalICV()->cancelPolicy =
        FALSE; // currently false only
    // init team context
    omptarget_nvptx_TeamDescr &currTeamDescr = getMyTeamDescriptor();
    currTeamDescr.InitTeamDescr();
    // this thread will start execution... has to update its task ICV
    // to point to the level zero task ICV. That ICV was init in
    // InitTeamDescr()
    omptarget_nvptx_threadPrivateContext->SetTopLevelTaskDescr(
        globalThreadId, currTeamDescr.LevelZeroTaskDescr());

    // set number of threads and thread limit in team to started value
    int globalThreadId = GetGlobalThreadId();
    omptarget_nvptx_TaskDescr *currTaskDescr =
        omptarget_nvptx_threadPrivateContext->GetTopLevelTaskDescr(
            globalThreadId);
    currTaskDescr->NThreads() = GetNumberOfThreadsInBlock();
    currTaskDescr->ThreadLimit() = ThreadLimit;
  }
}
