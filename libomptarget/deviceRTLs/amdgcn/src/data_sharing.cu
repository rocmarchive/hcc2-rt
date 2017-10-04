//===----- data_sharing.cu - NVPTX OpenMP debug utilities -------- CUDA -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.txt for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains the implementation of data sharing environments/
//
//===----------------------------------------------------------------------===//
#include "omptarget-nvptx.h"
#include <stdio.h>

#define DSFLAG 1
#define DSFLAG_INIT 1

#define DSPRINT0 PRINT0
#define DSPRINT  PRINT

// Number of threads in the CUDA block.
__device__ static unsigned getNumThreads() {
  return blockDim.x;
}
// Thread ID in the CUDA block
__device__ static unsigned getThreadId() {
  return threadIdx.x;
}
// Warp ID in the CUDA block
__device__ static unsigned getWarpId() {
  return threadIdx.x >> DS_Max_Worker_Warp_Size_Log2;
}
//// Team ID in the CUDA grid
//__device__ static unsigned getTeamId() {
//  return blockIdx.x;
//}
// The CUDA thread ID of the master thread.
__device__ static unsigned getMasterThreadId() {
  unsigned Mask = DS_Max_Worker_Warp_Size - 1;
  return (getNumThreads() - 1) & (~Mask);
}
// The lowest ID among the active threads in the warp.
EXTERN int32_t __kmpc_warp_master_active_thread_id() {
#ifdef __AMDGCN__
  int64_t Mask = __kmpc_warp_active_thread_mask64();
  unsigned tid  = threadIdx.x;
  unsigned laneid = (tid & 0X3F);
  int64_t Sh;
  unsigned Shnum;
  if (laneid < 32) {
   Mask = Mask << 32 ;
   Shnum = 32-laneid;
  } else  {
   Shnum = 64-laneid;
  }
  Sh = Mask << Shnum;
  unsigned popret = __popcll(Sh);
  //PRINT(LD_IO,"__kmpc_warp_master_active_thread_id: tid=%d laneid=%d Shnum=%d RETURN VAL=%d\n", tid, laneid, Shnum, popret);
  return popret;
#else
  //unsigned long long Mask = __ballot(true);
  unsigned long long Mask = __kmpc_warp_active_thread_mask();
  unsigned long long ShNum = 32 - (getThreadId() & DS_Max_Worker_Warp_Size_Log2_Mask);
  unsigned long long Sh = Mask << ShNum;
  return __popc(Sh);
#endif
}

// Return true if this is the master thread.
__device__ static bool IsMasterThread() {
  return getMasterThreadId() == getThreadId();
}
// Return true if this is the first thread in the warp.
//static bool IsWarpMasterThread() {
//  return (getThreadId() & DS_Max_Worker_Warp_Size_Log2_Mask) == 0u;
//}
// Return true if this is the first active thread in the warp.
__device__ static bool IsWarpMasterActiveThread() {
  return (__kmpc_warp_master_active_thread_id() == 0u);
}
/// Return the provided size aligned to the size of a pointer.
__device__ static size_t AlignVal(size_t Val) {
  const size_t Align = (size_t)sizeof(void*);
  if (Val & (Align-1)) {
    Val += Align;
    Val &= ~(Align-1);
  }
  return Val;
}

#if 0
// Turn on Datasharing debug with other debugging
#if OMPTARGET_NVPTX_DEBUG
#define DSFLAG 1
#define DSFLAG_INIT 1
#define DSPRINT(_flag, _str, _args...)                                         \
  {                                                                            \
    if (_flag) {                                                               \
      printf("(%d,%d) -> " _str, blockIdx.x, threadIdx.x, _args);          \
    }                                                                          \
  }
#define DSPRINT0(_flag, _str)                                                  \
  {                                                                            \
    if (_flag) {                                                               \
      printf("(%d,%d) -> " _str, blockIdx.x, threadIdx.x);                 \
    }                                                                          \
  }
#else 
#define DSPRINT0(flag, str)
#define DSPRINT(flag, str, _args...)
#endif
#endif

// Initialize the shared data structures. This is expected to be called for the master thread and warp masters.
// \param RootS: A pointer to the root of the data sharing stack.
// \param InitialDataSize: The initial size of the data in the slot.
EXTERN void __kmpc_initialize_data_sharing_environment(
    __kmpc_data_sharing_slot *rootS,
    size_t InitialDataSize){

  DSPRINT0(DSFLAG_INIT,"Entering __kmpc_initialize_data_sharing_environment\n");

  unsigned WID = getWarpId();
  DSPRINT(DSFLAG_INIT,"Warp ID: %d\n", WID);

  omptarget_nvptx_TeamDescr *teamDescr = &omptarget_nvptx_threadPrivateContext->TeamContext();
  __kmpc_data_sharing_slot *RootS = teamDescr->RootS(WID);

  DataSharingState.SlotPtr[WID] = RootS;
  DataSharingState.StackPtr[WID] = (void*)&RootS->Data[0];
  if (InitialDataSize == 256) { // Only zero for team master
    DataSharingState.FramePtr[WID] = 0 ;
    DataSharingState.ActiveThreads[WID] = 0 ;
  }

  // We don't need to initialize the frame and active threads.

  DSPRINT(DSFLAG_INIT,"Initial data size: %08lx \n", InitialDataSize);
  DSPRINT(DSFLAG_INIT,"Root slot at: %016llx \n", (long long)RootS);
  DSPRINT(DSFLAG_INIT,"Root slot data-end at: %016llx \n", (long long)RootS->DataEnd);
  DSPRINT(DSFLAG_INIT,"Root slot next at: %016llx \n", (long long)RootS->Next);
  DSPRINT(DSFLAG_INIT,"Shared slot ptr at: %016llx \n", (long long)DataSharingState.SlotPtr[WID]);
  DSPRINT(DSFLAG_INIT,"Shared stack ptr at: %016llx \n", (long long)DataSharingState.StackPtr[WID]);

  DSPRINT0(DSFLAG_INIT,"Exiting __kmpc_initialize_data_sharing_environment\n");
}

EXTERN void* __kmpc_data_sharing_environment_begin(
    __kmpc_data_sharing_slot **SavedSharedSlot,
    void **SavedSharedStack,
    void **SavedSharedFrame,
#ifdef __AMDGCN__
    int64_t *SavedActiveThreads,
#else
    int32_t *SavedActiveThreads,
#endif
    size_t SharingDataSize,
    size_t SharingDefaultDataSize,
    int16_t IsOMPRuntimeInitialized
    ){

  DSPRINT0(DSFLAG,"Entering __kmpc_data_sharing_environment_begin\n");

  // If the runtime has been elided, used __shared__ memory for master-worker
  // data sharing.
  if (!IsOMPRuntimeInitialized) return (void *) &DataSharingState;

  DSPRINT(DSFLAG,"Data Size %016lx\n", SharingDataSize);
  DSPRINT(DSFLAG,"Default Data Size %016lx\n", SharingDefaultDataSize);

  unsigned WID = getWarpId();
#ifdef __AMDGCN__
  unsigned long long CurActiveThreads = __ballot64(true);
#else
  unsigned CurActiveThreads = __ballot(true);
#endif

  __kmpc_data_sharing_slot *&SlotP = DataSharingState.SlotPtr[WID];
  void *&StackP = DataSharingState.StackPtr[WID];
  void *&FrameP = DataSharingState.FramePtr[WID];
#ifdef __AMDGCN__
  int64_t &ActiveT = DataSharingState.ActiveThreads[WID];
#else
  int32_t &ActiveT = DataSharingState.ActiveThreads[WID];
#endif

  DSPRINT0(DSFLAG,"Save current slot/stack values.\n");
  // Save the current values.
  *SavedSharedSlot = SlotP;
  *SavedSharedStack = StackP;
  *SavedSharedFrame = FrameP;
  *SavedActiveThreads = ActiveT;

  DSPRINT(DSFLAG,"Warp ID: %d\n", WID);
  DSPRINT(DSFLAG,"Saved slot ptr at: %016llx \n", (long long)SlotP);
  DSPRINT(DSFLAG,"Saved stack ptr at: %016llx \n", (long long)StackP);
  DSPRINT(DSFLAG,"Saved frame ptr at: %016llx \n", (long long)FrameP);
#ifdef __AMDGCN__
  DSPRINT(DSFLAG,"Active threads: %08lx \n", ActiveT);
#else
  DSPRINT(DSFLAG,"Active threads: %08x \n", ActiveT);
#endif

  // Only the warp active master needs to grow the stack.
  if (IsWarpMasterActiveThread()) {
    // Save the current active threads.
    ActiveT = CurActiveThreads;

    // Make sure we use aligned sizes to avoid rematerialization of data.
    SharingDataSize = AlignVal(SharingDataSize);
    // FIXME: The default data size can be assumed to be aligned?
    SharingDefaultDataSize = AlignVal(SharingDefaultDataSize);

    // Check if we have room for the data in the current slot.
    const uintptr_t CurrentStartAddress = (uintptr_t)StackP;
    const uintptr_t CurrentEndAddress = (uintptr_t)SlotP->DataEnd;
    const uintptr_t RequiredEndAddress =CurrentStartAddress + (uintptr_t)SharingDataSize;

    DSPRINT(DSFLAG,"Data Size %016lx\n", SharingDataSize);
    DSPRINT(DSFLAG,"Default Data Size %016lx\n", SharingDefaultDataSize);
    DSPRINT(DSFLAG,"Current Start Address %016lx\n", CurrentStartAddress);
    DSPRINT(DSFLAG,"Current End Address %016lx\n", CurrentEndAddress);
    DSPRINT(DSFLAG,"Required End Address %016lx\n", RequiredEndAddress);
#ifdef __AMDGCN__
    DSPRINT(DSFLAG,"Active Threads %08lx \n", ActiveT);
#else
    DSPRINT(DSFLAG,"Active Threads %08x \n", ActiveT);
#endif

    // If we require a new slot, allocate it and initialize it (or attempt to reuse one). Also, set the shared stack and slot pointers to the new place. If we do not need to grow the stack, just adapt the stack and frame pointers.
    if (CurrentEndAddress < RequiredEndAddress) {
      size_t NewSize = (SharingDataSize > SharingDefaultDataSize) ? SharingDataSize : SharingDefaultDataSize;
      __kmpc_data_sharing_slot *NewSlot = 0;

      // Attempt to reuse an existing slot.
      if (__kmpc_data_sharing_slot *ExistingSlot = SlotP->Next) {
        uintptr_t ExistingSlotSize = (uintptr_t)ExistingSlot->DataEnd - (uintptr_t)(&ExistingSlot->Data[0]);
        if (ExistingSlotSize >= NewSize) {
          DSPRINT(DSFLAG,"Reusing stack slot %016llx\n", (long long)ExistingSlot);
          NewSlot = ExistingSlot;
        } else {
          DSPRINT(DSFLAG,"Cleaning up -failed reuse - %016llx\n", (long long)SlotP->Next);
          free(ExistingSlot);
        }
      }

      if (!NewSlot) {
        NewSlot = ( __kmpc_data_sharing_slot *)malloc(sizeof(__kmpc_data_sharing_slot) + NewSize);
        DSPRINT(DSFLAG,"New slot allocated %016llx (data size=%016lx)\n", (long long)NewSlot, NewSize);
      }

      NewSlot->Next = 0;
      NewSlot->DataEnd = &NewSlot->Data[NewSize];

      SlotP->Next = NewSlot;
      SlotP = NewSlot;
      StackP = &NewSlot->Data[SharingDataSize];
      FrameP = &NewSlot->Data[0];
    } else {

      // Clean up any old slot that we may still have. The slot producers, do not eliminate them because that may be used to return data.
      if (SlotP->Next) {
        DSPRINT(DSFLAG,"Cleaning up - old not required - %016llx\n", (long long)SlotP->Next);
        free(SlotP->Next);
        SlotP->Next = 0;
      }

      FrameP = StackP;
      StackP = (void*)RequiredEndAddress;
    }
  }

  // FIXME: Need to see the impact of doing it here.
  __threadfence_block();

  DSPRINT(DSFLAG,"Exiting __kmpc_data_sharing_environment_begin with FrameP %p\n",FrameP);

  // All the threads in this warp get the frame they should work with.
  return FrameP;
}

EXTERN void __kmpc_data_sharing_environment_end(
    __kmpc_data_sharing_slot **SavedSharedSlot,
    void **SavedSharedStack,
    void **SavedSharedFrame,
#ifdef __AMDGCN__
    int64_t *SavedActiveThreads,
#else
    int32_t *SavedActiveThreads,
#endif
    int32_t IsEntryPoint
    ){

  DSPRINT0(DSFLAG,"Entering __kmpc_data_sharing_environment_end\n");

  unsigned WID = getWarpId();

  if (IsEntryPoint){
    if (IsWarpMasterActiveThread()) {
      DSPRINT0(DSFLAG,"Doing clean up\n");

      // The master thread cleans the saved slot, because this is an environment only for the master.
      __kmpc_data_sharing_slot *S = IsMasterThread() ? *SavedSharedSlot : DataSharingState.SlotPtr[WID];

      if (S->Next) {
        free(S->Next);
        S->Next = 0;
      }
    }

    DSPRINT0(DSFLAG,"Exiting Exiting __kmpc_data_sharing_environment_end\n");
    return;
  }
#ifdef __AMDGCN__
  int64_t CurActive = __ballot64(true);
#else
  int32_t CurActive = __ballot(true);
#endif

  // Only the warp master can restore the stack and frame information, and only if there are no other threads left behind in this environment (i.e. the warp diverged and returns in different places). This only works if we assume that threads will converge right after the call site that started the environment.
  if (IsWarpMasterActiveThread()) {
#ifdef __AMDGCN__
    int64_t &ActiveT = DataSharingState.ActiveThreads[WID];
#else
    int32_t &ActiveT = DataSharingState.ActiveThreads[WID];
#endif

    DSPRINT0(DSFLAG,"Before restoring the stack\n");
    // Zero the bits in the mask. If it is still different from zero, then we have other threads that will return after the current ones.
    ActiveT &= ~CurActive;
#ifdef __AMDGCN__
    DSPRINT(DSFLAG,"Active threads: %08lx; New mask: %08lx \n", CurActive, ActiveT);
#else
    DSPRINT(DSFLAG,"Active threads: %08x; New mask: %08x\n", CurActive, ActiveT);
#endif

    if (!ActiveT) {
      // No other active threads? Great, lets restore the stack.

      __kmpc_data_sharing_slot *&SlotP = DataSharingState.SlotPtr[WID];
      void *&StackP = DataSharingState.StackPtr[WID];
      void *&FrameP = DataSharingState.FramePtr[WID];

      SlotP = *SavedSharedSlot;
      StackP = *SavedSharedStack;
      FrameP = *SavedSharedFrame;
      ActiveT = *SavedActiveThreads;

      DSPRINT(DSFLAG,"Restored slot ptr at: %016llx \n",(long long)SlotP);
      DSPRINT(DSFLAG,"Restored stack ptr at: %016llx \n",(long long)StackP);
      DSPRINT(DSFLAG,"Restored frame ptr at: %016llx \n", (long long)FrameP);
#ifdef __AMDGCN__
      DSPRINT(DSFLAG,"Active threads: %08lx \n", ActiveT);
#else
      DSPRINT(DSFLAG,"Active threads: %08x \n", ActiveT);
#endif

    }
  }

  // FIXME: Need to see the impact of doing it here.
  __threadfence_block();

  DSPRINT0(DSFLAG,"Exiting __kmpc_data_sharing_environment_end\n");
  return;
}

EXTERN void* __kmpc_get_data_sharing_environment_frame(int32_t SourceThreadID,
                                                       int16_t IsOMPRuntimeInitialized){
  unsigned tid  = getThreadId() ;
  DSPRINT(DSFLAG,"Entering __kmpc_get_data_sharing_environment_frame id:%d this_tid:%d\n",
    SourceThreadID,tid);

  // If the runtime has been elided, use __shared__ memory for master-worker
  // data sharing.  We're reusing the statically allocated data structure
  // that is used for standard data sharing.
  if (!IsOMPRuntimeInitialized) return (void *) &DataSharingState;

  // Get the frame used by the requested thread.

  unsigned SourceWID = SourceThreadID >> DS_Max_Worker_Warp_Size_Log2;

  DSPRINT(DSFLAG,"Source  warp: %d shiftsize:%d \n", SourceWID, DS_Max_Worker_Warp_Size_Log2);

  void *P = DataSharingState.FramePtr[SourceWID];
  DSPRINT(DSFLAG,"Exiting __kmpc_get_data_sharing_environment_frame %p for tid:%d\n",P,tid);
  return P;
}

//EXTERN void __kmpc_samuel_print(int64_t Bla){
//  DSPRINT(DSFLAG,"Sam print: %016llx\n",Bla);
//
//}
