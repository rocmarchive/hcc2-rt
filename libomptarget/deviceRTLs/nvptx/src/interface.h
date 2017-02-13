//===------- interface.h - NVPTX OpenMP interface definitions ---- CUDA -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.txt for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains debug macros to be used in the application.
//
//  This file contains all the definitions that are relevant to
//  the interface. The first section contains the interface as
//  declared by OpenMP.  A second section includes library private calls
//  (mostly debug, temporary?) The third section includes the compiler
//  specific interfaces.
//
//===----------------------------------------------------------------------===//

#ifndef _INTERFACES_H_
#define _INTERFACES_H_

////////////////////////////////////////////////////////////////////////////////
// OpenMP interface
////////////////////////////////////////////////////////////////////////////////

typedef uint32_t omp_lock_t;      /* arbitrary type of the right length */
typedef uint64_t omp_nest_lock_t; /* arbitrary type of the right length */

typedef enum omp_sched_t {
  omp_sched_static = 1,  /* chunkSize >0 */
  omp_sched_dynamic = 2, /* chunkSize >0 */
  omp_sched_guided = 3,  /* chunkSize >0 */
  omp_sched_auto = 4,    /* no chunkSize */
} omp_sched_t;

typedef enum omp_proc_bind_t {
  omp_proc_bind_false = 0,
  omp_proc_bind_true = 1,
  omp_proc_bind_master = 2,
  omp_proc_bind_close = 3,
  omp_proc_bind_spread = 4
} omp_proc_bind_t;

EXTERN double omp_get_wtick(void);
EXTERN double omp_get_wtime(void);

EXTERN void omp_set_num_threads(int num);
EXTERN int omp_get_num_threads(void);
EXTERN int omp_get_max_threads(void);
EXTERN int omp_get_thread_limit(void);
EXTERN int omp_get_thread_num(void);
EXTERN int omp_get_num_procs(void);
EXTERN int omp_in_parallel(void);
EXTERN int omp_in_final(void);
EXTERN void omp_set_dynamic(int flag);
EXTERN int omp_get_dynamic(void);
EXTERN void omp_set_nested(int flag);
EXTERN int omp_get_nested(void);
EXTERN void omp_set_max_active_levels(int level);
EXTERN int omp_get_max_active_levels(void);
EXTERN int omp_get_level(void);
EXTERN int omp_get_active_level(void);
EXTERN int omp_get_ancestor_thread_num(int level);
EXTERN int omp_get_team_size(int level);

EXTERN void omp_init_lock(omp_lock_t *lock);
EXTERN void omp_init_nest_lock(omp_nest_lock_t *lock);
EXTERN void omp_destroy_lock(omp_lock_t *lock);
EXTERN void omp_destroy_nest_lock(omp_nest_lock_t *lock);
EXTERN void omp_set_lock(omp_lock_t *lock);
EXTERN void omp_set_nest_lock(omp_nest_lock_t *lock);
EXTERN void omp_unset_lock(omp_lock_t *lock);
EXTERN void omp_unset_nest_lock(omp_nest_lock_t *lock);
EXTERN int omp_test_lock(omp_lock_t *lock);
EXTERN int omp_test_nest_lock(omp_nest_lock_t *lock);

EXTERN void omp_get_schedule(omp_sched_t *kind, int *modifier);
EXTERN void omp_set_schedule(omp_sched_t kind, int modifier);
EXTERN omp_proc_bind_t omp_get_proc_bind(void);
EXTERN int omp_get_cancellation(void);
EXTERN void omp_set_default_device(int deviceId);
EXTERN int omp_get_default_device(void);
EXTERN int omp_get_num_devices(void);
EXTERN int omp_get_num_teams(void);
EXTERN int omp_get_team_num(void);
EXTERN int omp_is_initial_device(void);
EXTERN int omp_get_initial_device(void);
EXTERN int omp_get_max_task_priority(void);

////////////////////////////////////////////////////////////////////////////////
// OMPTARGET_NVPTX private (debug / temportary?) interface
////////////////////////////////////////////////////////////////////////////////

// for debug
EXTERN void __kmpc_print_str(char *title);
EXTERN void __kmpc_print_title_int(char *title, int data);
EXTERN void __kmpc_print_index(char *title, int i);
EXTERN void __kmpc_print_int(int data);
EXTERN void __kmpc_print_double(double data);
EXTERN void __kmpc_print_address_int64(int64_t data);

////////////////////////////////////////////////////////////////////////////////
// file below is swiped from kmpc host interface
////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////
// kmp specifc types
////////////////////////////////////////////////////////////////////////////////

typedef enum kmp_sched_t {
  kmp_sched_static_chunk = 33,
  kmp_sched_static_nochunk = 34,
  kmp_sched_dynamic = 35,
  kmp_sched_guided = 36,
  kmp_sched_runtime = 37,
  kmp_sched_auto = 38,

  kmp_sched_static_ordered = 65,
  kmp_sched_static_nochunk_ordered = 66,
  kmp_sched_dynamic_ordered = 67,
  kmp_sched_guided_ordered = 68,
  kmp_sched_runtime_ordered = 69,
  kmp_sched_auto_ordered = 70,

  kmp_sched_distr_static_chunk = 91,
  kmp_sched_distr_static_nochunk = 92,
  kmp_sched_distr_static_chunk_sched_static_chunkone = 93,

  kmp_sched_default = kmp_sched_static_nochunk,
  kmp_sched_unordered_first = kmp_sched_static_chunk,
  kmp_sched_unordered_last = kmp_sched_auto,
  kmp_sched_ordered_first = kmp_sched_static_ordered,
  kmp_sched_ordered_last = kmp_sched_auto_ordered,
  kmp_sched_distribute_first = kmp_sched_distr_static_chunk,
  kmp_sched_distribute_last = kmp_sched_distr_static_chunk_sched_static_chunkone,
} kmp_sched_t;

// parallel defs
typedef void kmp_Indent;
typedef void (*kmp_ParFctPtr)(int32_t *global_tid, int32_t *bound_tid, ...);
typedef void (*kmp_ReductFctPtr)(void *lhsData, void *rhsData);
typedef void (*kmp_InterWarpCopyFctPtr)(void* src, int32_t warp_num);
typedef void (*kmp_ShuffleReductFctPtr)(void *rhsData, int16_t lane_id, int16_t lane_offset, int16_t shortCircuit);
typedef void (*kmp_CopyToScratchpadFctPtr)(void *reduceData, void * scratchpad, int32_t index, int32_t width);
typedef void (*kmp_LoadReduceFctPtr)(void *reduceData, void * scratchpad, int32_t index, int32_t width, int32_t reduce);

// task defs
typedef struct kmp_TaskDescr kmp_TaskDescr;
typedef int32_t (*kmp_TaskFctPtr)(int32_t global_tid, kmp_TaskDescr *taskDescr);
typedef struct kmp_TaskDescr {
  void *sharedPointerTable;   // ptr to a table of shared var ptrs
  kmp_TaskFctPtr sub;         // task subroutine
  int32_t partId;             // unused
  kmp_TaskFctPtr destructors; // destructor of c++ first private
} kmp_TaskDescr;
// task dep defs
#define KMP_TASKDEP_IN 0x1u
#define KMP_TASKDEP_OUT 0x2u
typedef struct kmp_TaskDep_Public {
  void *addr;
  size_t len;
  uint8_t flags; // bit 0: in, bit 1: out
} kmp_TaskDep_Public;

// flags that interpret the interface part of tasking flags
#define KMP_TASK_IS_TIED 0x1
#define KMP_TASK_FINAL 0x2
#define KMP_TASK_MERGED_IF0 0x4 /* unused */
#define KMP_TASK_DESTRUCTOR_THUNK 0x8

// flags for task setup return
#define KMP_CURRENT_TASK_NOT_SUSPENDED 0
#define KMP_CURRENT_TASK_SUSPENDED 1

// sync defs
typedef int32_t kmp_CriticalName[8];

////////////////////////////////////////////////////////////////////////////////
// flags for kstate (all bits initially off)
////////////////////////////////////////////////////////////////////////////////

// first 2 bits used by kmp_Reduction (defined in kmp_reduction.cpp)
#define KMP_REDUCTION_MASK 0x3
#define KMP_SKIP_NEXT_CALL 0x4
#define KMP_SKIP_NEXT_CANCEL_BARRIER 0x8

////////////////////////////////////////////////////////////////////////////////
// data
////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////
// external interface
////////////////////////////////////////////////////////////////////////////////

// query
EXTERN int32_t __kmpc_global_num_threads(kmp_Indent *loc); // missing
EXTERN int32_t __kmpc_bound_thread_num(kmp_Indent *loc);   // missing
EXTERN int32_t __kmpc_bound_num_threads(kmp_Indent *loc);  // missing
EXTERN int32_t __kmpc_in_parallel(kmp_Indent *loc);        // missing

// parallel
EXTERN int32_t __kmpc_global_thread_num(kmp_Indent *loc);
EXTERN void __kmpc_push_num_threads(kmp_Indent *loc, int32_t global_tid,
                                    int32_t num_threads);
// simd
EXTERN void __kmpc_push_simd_limit(kmp_Indent *loc, int32_t global_tid,
                                   int32_t simd_limit);
// aee ... not supported
// EXTERN void __kmpc_fork_call(kmp_Indent *loc, int32_t argc, kmp_ParFctPtr
// microtask, ...);
EXTERN void __kmpc_serialized_parallel(kmp_Indent *loc, uint32_t global_tid);
EXTERN void __kmpc_end_serialized_parallel(kmp_Indent *loc,
                                           uint32_t global_tid);
EXTERN uint16_t __kmpc_parallel_level(kmp_Indent *loc,
                                      uint32_t global_tid);

// proc bind
EXTERN void __kmpc_push_proc_bind(kmp_Indent *loc, uint32_t global_tid, int proc_bind);
EXTERN int omp_get_num_places(void);
EXTERN int omp_get_place_num_procs(int place_num);
EXTERN void omp_get_place_proc_ids(int place_num, int *ids);
EXTERN int omp_get_place_num(void);
EXTERN int omp_get_partition_num_places(void);
EXTERN void omp_get_partition_place_nums(int *place_nums);

// for static (no chunk or chunk)
EXTERN void __kmpc_for_static_init_4(kmp_Indent *loc, int32_t global_tid,
                                     int32_t sched, int32_t *plastiter,
                                     int32_t *plower, int32_t *pupper,
                                     int32_t *pstride, int32_t incr,
                                     int32_t chunk);
EXTERN void __kmpc_for_static_init_4u(kmp_Indent *loc, int32_t global_tid,
                                      int32_t sched, int32_t *plastiter,
                                      uint32_t *plower, uint32_t *pupper,
                                      int32_t *pstride, int32_t incr,
                                      int32_t chunk);
EXTERN void __kmpc_for_static_init_8(kmp_Indent *loc, int32_t global_tid,
                                     int32_t sched, int32_t *plastiter,
                                     int64_t *plower, int64_t *pupper,
                                     int64_t *pstride, int64_t incr,
                                     int64_t chunk);
EXTERN void __kmpc_for_static_init_8u(kmp_Indent *loc, int32_t global_tid,
                                      int32_t sched, int32_t *plastiter1,
                                      uint64_t *plower, uint64_t *pupper,
                                      int64_t *pstride, int64_t incr,
                                      int64_t chunk);

EXTERN void __kmpc_for_static_fini(kmp_Indent *loc, int32_t global_tid);

// for dynamic
EXTERN void __kmpc_dispatch_init_4(kmp_Indent *loc, int32_t global_tid,
                                   int32_t sched, int32_t lower, int32_t upper,
                                   int32_t incr, int32_t chunk);
EXTERN void __kmpc_dispatch_init_4u(kmp_Indent *loc, int32_t global_tid,
                                    int32_t sched, uint32_t lower,
                                    uint32_t upper, int32_t incr,
                                    int32_t chunk);
EXTERN void __kmpc_dispatch_init_8(kmp_Indent *loc, int32_t global_tid,
                                   int32_t sched, int64_t lower, int64_t upper,
                                   int64_t incr, int64_t chunk);
EXTERN void __kmpc_dispatch_init_8u(kmp_Indent *loc, int32_t global_tid,
                                    int32_t sched, uint64_t lower,
                                    uint64_t upper, int64_t incr,
                                    int64_t chunk);

EXTERN int __kmpc_dispatch_next_4(kmp_Indent *loc, int32_t global_tid,
                                  int32_t *plastiter, int32_t *plower,
                                  int32_t *pupper, int32_t *pstride);
EXTERN int __kmpc_dispatch_next_4u(kmp_Indent *loc, int32_t global_tid,
                                   int32_t *plastiter, uint32_t *plower,
                                   uint32_t *pupper, int32_t *pstride);
EXTERN int __kmpc_dispatch_next_8(kmp_Indent *loc, int32_t global_tid,
                                  int32_t *plastiter, int64_t *plower,
                                  int64_t *pupper, int64_t *pstride);
EXTERN int __kmpc_dispatch_next_8u(kmp_Indent *loc, int32_t global_tid,
                                   int32_t *plastiter, uint64_t *plower,
                                   uint64_t *pupper, int64_t *pstride);

EXTERN void __kmpc_dispatch_fini_4(kmp_Indent *loc, int32_t global_tid);
EXTERN void __kmpc_dispatch_fini_4u(kmp_Indent *loc, int32_t global_tid);
EXTERN void __kmpc_dispatch_fini_8(kmp_Indent *loc, int32_t global_tid);
EXTERN void __kmpc_dispatch_fini_8u(kmp_Indent *loc, int32_t global_tid);

// reduction
EXTERN int32_t __kmpc_reduce41(kmp_Indent *loc, int32_t global_tid,
                               int32_t varNum, size_t reduceSize,
                               void *reduceData, void *reduceArraySize,
                               kmp_ReductFctPtr *reductFct,
                               kmp_CriticalName *lock);
EXTERN void __kmpc_nvptx_end_reduce(int32_t global_tid);
EXTERN void __kmpc_nvptx_end_reduce_nowait(int32_t global_tid);
EXTERN int32_t __kmpc_nvptx_parallel_reduce_nowait(int32_t global_tid,
                             int32_t num_vars, size_t reduce_size,
                             void *reduce_data, kmp_ShuffleReductFctPtr shflFct,
                             kmp_InterWarpCopyFctPtr cpyFct);
EXTERN int32_t __kmpc_nvptx_simd_reduce_nowait(int32_t global_tid,
                             int32_t num_vars, size_t reduce_size,
                             void *reduce_data, kmp_ShuffleReductFctPtr shflFct,
                             kmp_InterWarpCopyFctPtr cpyFct);
EXTERN int32_t __kmpc_nvptx_teams_reduce_nowait(int32_t global_tid,
                             int32_t num_vars, size_t reduce_size,
                             void *reduce_data,
                             kmp_ShuffleReductFctPtr shflFct,
                             kmp_InterWarpCopyFctPtr cpyFct,
                             kmp_CopyToScratchpadFctPtr sratchFct,
                             kmp_LoadReduceFctPtr ldFct);
EXTERN int32_t __kmpc_shuffle_int32(int32_t val, int16_t delta, int16_t size);
EXTERN int64_t __kmpc_shuffle_int64(int64_t val, int16_t delta, int16_t size);

// sync barrier
EXTERN int32_t __kmpc_cancel_barrier(kmp_Indent *loc, int32_t global_tid);

// single
EXTERN int32_t __kmpc_single(kmp_Indent *loc, int32_t global_tid);
EXTERN void __kmpc_end_single(kmp_Indent *loc, int32_t global_tid);

// sync
EXTERN int32_t __kmpc_master(kmp_Indent *loc, int32_t global_tid);
EXTERN void __kmpc_end_master(kmp_Indent *loc, int32_t global_tid);
EXTERN void __kmpc_ordered(kmp_Indent *loc, int32_t global_tid);
EXTERN void __kmpc_end_ordered(kmp_Indent *loc, int32_t global_tid);
EXTERN void __kmpc_critical(kmp_Indent *loc, int32_t global_tid,
                            kmp_CriticalName *crit);
EXTERN void __kmpc_end_critical(kmp_Indent *loc, int32_t global_tid,
                                kmp_CriticalName *crit);
EXTERN void __kmpc_flush(kmp_Indent *loc);

// vote
EXTERN int32_t __kmpc_warp_active_thread_mask();

// tasks
EXTERN kmp_TaskDescr *__kmpc_omp_task_alloc(kmp_Indent *loc,
                                            uint32_t global_tid, int32_t flag,
                                            size_t sizeOfTaskInclPrivate,
                                            size_t sizeOfSharedTable,
                                            kmp_TaskFctPtr sub);
EXTERN int32_t __kmpc_omp_task(kmp_Indent *loc, uint32_t global_tid,
                               kmp_TaskDescr *newLegacyTaskDescr);
EXTERN int32_t __kmpc_omp_task_with_deps(kmp_Indent *loc, uint32_t global_tid,
                                         kmp_TaskDescr *newLegacyTaskDescr,
                                         int32_t depNum, void *depList,
                                         int32_t noAliasDepNum,
                                         void *noAliasDepList);
EXTERN void __kmpc_omp_task_begin_if0(kmp_Indent *loc, uint32_t global_tid,
                                      kmp_TaskDescr *newLegacyTaskDescr);
EXTERN void __kmpc_omp_task_complete_if0(kmp_Indent *loc, uint32_t global_tid,
                                         kmp_TaskDescr *newLegacyTaskDescr);
EXTERN void __kmpc_omp_wait_deps(kmp_Indent *loc, uint32_t global_tid,
                                 int32_t depNum, void *depList,
                                 int32_t noAliasDepNum, void *noAliasDepList);
EXTERN void __kmpc_taskgroup(kmp_Indent *loc, uint32_t global_tid);
EXTERN void __kmpc_end_taskgroup(kmp_Indent *loc, uint32_t global_tid);
EXTERN int32_t __kmpc_omp_taskyield(kmp_Indent *loc, uint32_t global_tid,
    int end_part);
EXTERN int32_t __kmpc_omp_taskwait(kmp_Indent *loc, uint32_t global_tid);
EXTERN void __kmpc_taskloop(kmp_Indent *loc, uint32_t global_tid,
    kmp_TaskDescr *newKmpTaskDescr, int if_val, uint64_t *lb, uint64_t *ub,
    int64_t st, int nogroup, int32_t sched, uint64_t grainsize,
    void *task_dup);

// cancel
EXTERN int32_t __kmpc_cancellationpoint(kmp_Indent *loc, int32_t global_tid,
                                        int32_t cancelVal);
EXTERN int32_t __kmpc_cancel(kmp_Indent *loc, int32_t global_tid,
                             int32_t cancelVal);

// target (no target call here)

// atomic
// EXTERN void __array_atomicfixed4_add(kmp_Indent *loc, int tid, int32_t
// *addr1, int32_t *val, int64_t n);
EXTERN void __kmpc_atomic_fixed1_wr(kmp_Indent *loc, int tid, int8_t *addr,
                                    int8_t val);
EXTERN void __kmpc_atomic_fixed1_add(kmp_Indent *loc, int tid, int8_t *addr,
                                     int8_t val);
EXTERN void __kmpc_atomic_fixed1_sub(kmp_Indent *loc, int tid, int8_t *addr,
                                     int8_t val);
EXTERN void __kmpc_atomic_fixed1_sub_rev(kmp_Indent *loc, int tid,
                                         int8_t *addr, int8_t val);
EXTERN void __kmpc_atomic_fixed1_mul(kmp_Indent *loc, int tid, int8_t *addr,
                                     int8_t val);
EXTERN void __kmpc_atomic_fixed1_div(kmp_Indent *loc, int tid, int8_t *addr,
                                     int8_t val);
EXTERN void __kmpc_atomic_fixed1_div_rev(kmp_Indent *loc, int tid,
                                         int8_t *addr, int8_t val);
EXTERN void __kmpc_atomic_fixed1u_div(kmp_Indent *loc, int tid, uint8_t *addr,
                                      uint8_t val);
EXTERN void __kmpc_atomic_fixed1u_div_rev(kmp_Indent *loc, int tid,
                                          uint8_t *addr, uint8_t val);
EXTERN void __kmpc_atomic_fixed1_min(kmp_Indent *loc, int tid, int8_t *addr,
                                     int8_t val);
EXTERN void __kmpc_atomic_fixed1_max(kmp_Indent *loc, int tid, int8_t *addr,
                                     int8_t val);
EXTERN void __kmpc_atomic_fixed1_andb(kmp_Indent *loc, int tid, int8_t *addr,
                                      int8_t val);
EXTERN void __kmpc_atomic_fixed1_orb(kmp_Indent *loc, int tid, int8_t *addr,
                                     int8_t val);
EXTERN void __kmpc_atomic_fixed1_xor(kmp_Indent *loc, int tid, int8_t *addr,
                                     int8_t val);
EXTERN void __kmpc_atomic_fixed1_andl(kmp_Indent *loc, int tid, int8_t *addr,
                                      int8_t val);
EXTERN void __kmpc_atomic_fixed1_orl(kmp_Indent *loc, int tid, int8_t *addr,
                                     int8_t val);
EXTERN void __kmpc_atomic_fixed1_eqv(kmp_Indent *loc, int tid, int8_t *addr,
                                     int8_t val);
EXTERN void __kmpc_atomic_fixed1_neqv(kmp_Indent *loc, int tid, int8_t *addr,
                                      int8_t val);
EXTERN void __kmpc_atomic_fixed1_shl(kmp_Indent *loc, int tid, int8_t *addr,
                                     int8_t val);
EXTERN void __kmpc_atomic_fixed1_shl_rev(kmp_Indent *loc, int tid,
                                         int8_t *addr, int8_t val);
EXTERN void __kmpc_atomic_fixed1_shr(kmp_Indent *loc, int tid, int8_t *addr,
                                     int8_t val);
EXTERN void __kmpc_atomic_fixed1_shr_rev(kmp_Indent *loc, int tid,
                                         int8_t *addr, int8_t val);
EXTERN void __kmpc_atomic_fixed1u_shr(kmp_Indent *loc, int tid, uint8_t *addr,
                                      uint8_t val);
EXTERN void __kmpc_atomic_fixed1u_shr_rev(kmp_Indent *loc, int tid,
                                          uint8_t *addr, uint8_t val);
EXTERN void __kmpc_atomic_fixed2_wr(kmp_Indent *loc, int tid, int16_t *addr,
                                    int16_t val);
EXTERN void __kmpc_atomic_fixed2_add(kmp_Indent *loc, int tid, int16_t *addr,
                                     int16_t val);
EXTERN void __kmpc_atomic_fixed2_sub(kmp_Indent *loc, int tid, int16_t *addr,
                                     int16_t val);
EXTERN void __kmpc_atomic_fixed2_sub_rev(kmp_Indent *loc, int tid,
                                         int16_t *addr, int16_t val);
EXTERN void __kmpc_atomic_fixed2_mul(kmp_Indent *loc, int tid, int16_t *addr,
                                     int16_t val);
EXTERN void __kmpc_atomic_fixed2_div(kmp_Indent *loc, int tid, int16_t *addr,
                                     int16_t val);
EXTERN void __kmpc_atomic_fixed2_div_rev(kmp_Indent *loc, int tid,
                                         int16_t *addr, int16_t val);
EXTERN void __kmpc_atomic_fixed2u_div(kmp_Indent *loc, int tid, uint16_t *addr,
                                      uint16_t val);
EXTERN void __kmpc_atomic_fixed2u_div_rev(kmp_Indent *loc, int tid,
                                          uint16_t *addr, uint16_t val);
EXTERN void __kmpc_atomic_fixed2_min(kmp_Indent *loc, int tid, int16_t *addr,
                                     int16_t val);
EXTERN void __kmpc_atomic_fixed2_max(kmp_Indent *loc, int tid, int16_t *addr,
                                     int16_t val);
EXTERN void __kmpc_atomic_fixed2_andb(kmp_Indent *loc, int tid, int16_t *addr,
                                      int16_t val);
EXTERN void __kmpc_atomic_fixed2_orb(kmp_Indent *loc, int tid, int16_t *addr,
                                     int16_t val);
EXTERN void __kmpc_atomic_fixed2_xor(kmp_Indent *loc, int tid, int16_t *addr,
                                     int16_t val);
EXTERN void __kmpc_atomic_fixed2_andl(kmp_Indent *loc, int tid, int16_t *addr,
                                      int16_t val);
EXTERN void __kmpc_atomic_fixed2_orl(kmp_Indent *loc, int tid, int16_t *addr,
                                     int16_t val);
EXTERN void __kmpc_atomic_fixed2_eqv(kmp_Indent *loc, int tid, int16_t *addr,
                                     int16_t val);
EXTERN void __kmpc_atomic_fixed2_neqv(kmp_Indent *loc, int tid, int16_t *addr,
                                      int16_t val);
EXTERN void __kmpc_atomic_fixed2_shl(kmp_Indent *loc, int tid, int16_t *addr,
                                     int16_t val);
EXTERN void __kmpc_atomic_fixed2_shl_rev(kmp_Indent *loc, int tid,
                                         int16_t *addr, int16_t val);
EXTERN void __kmpc_atomic_fixed2_shr(kmp_Indent *loc, int tid, int16_t *addr,
                                     int16_t val);
EXTERN void __kmpc_atomic_fixed2_shr_rev(kmp_Indent *loc, int tid,
                                         int16_t *addr, int16_t val);
EXTERN void __kmpc_atomic_fixed2u_shr(kmp_Indent *loc, int tid, uint16_t *addr,
                                      uint16_t val);
EXTERN void __kmpc_atomic_fixed2u_shr_rev(kmp_Indent *loc, int tid,
                                          uint16_t *addr, uint16_t val);
EXTERN void __kmpc_atomic_fixed2_swp(kmp_Indent *loc, int tid, int16_t *addr,
                                     int16_t val);
EXTERN void __kmpc_atomic_fixed4_wr(kmp_Indent *loc, int tid, int32_t *addr,
                                    int32_t val);
EXTERN void __kmpc_atomic_fixed4_add(kmp_Indent *loc, int tid, int32_t *addr,
                                     int32_t val);
EXTERN void __kmpc_atomic_fixed4_sub(kmp_Indent *loc, int tid, int32_t *addr,
                                     int32_t val);
EXTERN void __kmpc_atomic_fixed4_sub_rev(kmp_Indent *loc, int tid,
                                         int32_t *addr, int32_t val);
EXTERN void __kmpc_atomic_fixed4_mul(kmp_Indent *loc, int tid, int32_t *addr,
                                     int32_t val);
EXTERN void __kmpc_atomic_fixed4_div(kmp_Indent *loc, int tid, int32_t *addr,
                                     int32_t val);
EXTERN void __kmpc_atomic_fixed4_div_rev(kmp_Indent *loc, int tid,
                                         int32_t *addr, int32_t val);
EXTERN void __kmpc_atomic_fixed4u_div(kmp_Indent *loc, int tid, uint32_t *addr,
                                      uint32_t val);
EXTERN void __kmpc_atomic_fixed4u_div_rev(kmp_Indent *loc, int tid,
                                          uint32_t *addr, uint32_t val);
EXTERN void __kmpc_atomic_fixed4_min(kmp_Indent *loc, int tid, int32_t *addr,
                                     int32_t val);
EXTERN void __kmpc_atomic_fixed4_max(kmp_Indent *loc, int tid, int32_t *addr,
                                     int32_t val);
EXTERN void __kmpc_atomic_fixed4_andb(kmp_Indent *loc, int tid, int32_t *addr,
                                      int32_t val);
EXTERN void __kmpc_atomic_fixed4_orb(kmp_Indent *loc, int tid, int32_t *addr,
                                     int32_t val);
EXTERN void __kmpc_atomic_fixed4_xor(kmp_Indent *loc, int tid, int32_t *addr,
                                     int32_t val);
EXTERN void __kmpc_atomic_fixed4_andl(kmp_Indent *loc, int tid, int32_t *addr,
                                      int32_t val);
EXTERN void __kmpc_atomic_fixed4_orl(kmp_Indent *loc, int tid, int32_t *addr,
                                     int32_t val);
EXTERN void __kmpc_atomic_fixed4_eqv(kmp_Indent *loc, int tid, int32_t *addr,
                                     int32_t val);
EXTERN void __kmpc_atomic_fixed4_neqv(kmp_Indent *loc, int tid, int32_t *addr,
                                      int32_t val);
EXTERN void __kmpc_atomic_fixed4_shl(kmp_Indent *loc, int tid, int32_t *addr,
                                     int32_t val);
EXTERN void __kmpc_atomic_fixed4_shl_rev(kmp_Indent *loc, int tid,
                                         int32_t *addr, int32_t val);
EXTERN void __kmpc_atomic_fixed4_shr(kmp_Indent *loc, int tid, int32_t *addr,
                                     int32_t val);
EXTERN void __kmpc_atomic_fixed4_shr_rev(kmp_Indent *loc, int tid,
                                         int32_t *addr, int32_t val);
EXTERN void __kmpc_atomic_fixed4u_shr(kmp_Indent *loc, int tid, uint32_t *addr,
                                      uint32_t val);
EXTERN void __kmpc_atomic_fixed4u_shr_rev(kmp_Indent *loc, int tid,
                                          uint32_t *addr, uint32_t val);
EXTERN void __kmpc_atomic_fixed4_swp(kmp_Indent *loc, int tid, int32_t *addr,
                                     int32_t val);
EXTERN void __kmpc_atomic_fixed8_wr(kmp_Indent *loc, int tid, int64_t *addr,
                                    int64_t val);
EXTERN void __kmpc_atomic_fixed8_add(kmp_Indent *loc, int tid, int64_t *addr,
                                     int64_t val);
EXTERN void __kmpc_atomic_fixed8_sub(kmp_Indent *loc, int tid, int64_t *addr,
                                     int64_t val);
EXTERN void __kmpc_atomic_fixed8_sub_rev(kmp_Indent *loc, int tid,
                                         int64_t *addr, int64_t val);
EXTERN void __kmpc_atomic_fixed8_mul(kmp_Indent *loc, int tid, int64_t *addr,
                                     int64_t val);
EXTERN void __kmpc_atomic_fixed8_div(kmp_Indent *loc, int tid, int64_t *addr,
                                     int64_t val);
EXTERN void __kmpc_atomic_fixed8_div_rev(kmp_Indent *loc, int tid,
                                         int64_t *addr, int64_t val);
EXTERN void __kmpc_atomic_fixed8u_div(kmp_Indent *loc, int tid, uint64_t *addr,
                                      uint64_t val);
EXTERN void __kmpc_atomic_fixed8u_div_rev(kmp_Indent *loc, int tid,
                                          uint64_t *addr, uint64_t val);
EXTERN void __kmpc_atomic_fixed8_min(kmp_Indent *loc, int tid, int64_t *addr,
                                     int64_t val);
EXTERN void __kmpc_atomic_fixed8_max(kmp_Indent *loc, int tid, int64_t *addr,
                                     int64_t val);
EXTERN void __kmpc_atomic_fixed8_andb(kmp_Indent *loc, int tid, int64_t *addr,
                                      int64_t val);
EXTERN void __kmpc_atomic_fixed8_orb(kmp_Indent *loc, int tid, int64_t *addr,
                                     int64_t val);
EXTERN void __kmpc_atomic_fixed8_xor(kmp_Indent *loc, int tid, int64_t *addr,
                                     int64_t val);
EXTERN void __kmpc_atomic_fixed8_andl(kmp_Indent *loc, int tid, int64_t *addr,
                                      int64_t val);
EXTERN void __kmpc_atomic_fixed8_orl(kmp_Indent *loc, int tid, int64_t *addr,
                                     int64_t val);
EXTERN void __kmpc_atomic_fixed8_eqv(kmp_Indent *loc, int tid, int64_t *addr,
                                     int64_t val);
EXTERN void __kmpc_atomic_fixed8_neqv(kmp_Indent *loc, int tid, int64_t *addr,
                                      int64_t val);
EXTERN void __kmpc_atomic_fixed8_shl(kmp_Indent *loc, int tid, int64_t *addr,
                                     int64_t val);
EXTERN void __kmpc_atomic_fixed8_shl_rev(kmp_Indent *loc, int tid,
                                         int64_t *addr, int64_t val);
EXTERN void __kmpc_atomic_fixed8_shr(kmp_Indent *loc, int tid, int64_t *addr,
                                     int64_t val);
EXTERN void __kmpc_atomic_fixed8_shr_rev(kmp_Indent *loc, int tid,
                                         int64_t *addr, int64_t val);
EXTERN void __kmpc_atomic_fixed8u_shr(kmp_Indent *loc, int tid, uint64_t *addr,
                                      uint64_t val);
EXTERN void __kmpc_atomic_fixed8u_shr_rev(kmp_Indent *loc, int tid,
                                          uint64_t *addr, uint64_t val);
EXTERN void __kmpc_atomic_fixed8_swp(kmp_Indent *loc, int tid, int64_t *addr,
                                     int64_t val);
EXTERN void __kmpc_atomic_float4_add(kmp_Indent *loc, int tid, float *addr,
                                     float val);
EXTERN void __kmpc_atomic_float4_sub(kmp_Indent *loc, int tid, float *addr,
                                     float val);
EXTERN void __kmpc_atomic_float4_sub_rev(kmp_Indent *loc, int tid, float *addr,
                                         float val);
EXTERN void __kmpc_atomic_float4_mul(kmp_Indent *loc, int tid, float *addr,
                                     float val);
EXTERN void __kmpc_atomic_float4_div(kmp_Indent *loc, int tid, float *addr,
                                     float val);
EXTERN void __kmpc_atomic_float4_div_rev(kmp_Indent *loc, int tid, float *addr,
                                         float val);
EXTERN void __kmpc_atomic_float4_min(kmp_Indent *loc, int tid, float *addr,
                                     float val);
EXTERN void __kmpc_atomic_float4_max(kmp_Indent *loc, int tid, float *addr,
                                     float val);
EXTERN void __kmpc_atomic_float8_add(kmp_Indent *loc, int tid, double *addr,
                                     double val);
EXTERN void __kmpc_atomic_float8_sub(kmp_Indent *loc, int tid, double *addr,
                                     double val);
EXTERN void __kmpc_atomic_float8_sub_rev(kmp_Indent *loc, int tid,
                                         double *addr, double val);
EXTERN void __kmpc_atomic_float8_mul(kmp_Indent *loc, int tid, double *addr,
                                     double val);
EXTERN void __kmpc_atomic_float8_div(kmp_Indent *loc, int tid, double *addr,
                                     double val);
EXTERN void __kmpc_atomic_float8_div_rev(kmp_Indent *loc, int tid,
                                         double *addr, double val);
EXTERN void __kmpc_atomic_float8_min(kmp_Indent *loc, int tid, double *addr,
                                     double val);
EXTERN void __kmpc_atomic_float8_max(kmp_Indent *loc, int tid, double *addr,
                                     double val);
EXTERN void __kmpc_atomic_cmplx4_add(kmp_Indent *loc, int tid,
                                     float _Complex *addr, float _Complex val);
EXTERN void __kmpc_atomic_cmplx4_sub(kmp_Indent *loc, int tid,
                                     float _Complex *addr, float _Complex val);
EXTERN void __kmpc_atomic_cmplx4_sub_rev(kmp_Indent *loc, int tid,
                                         float _Complex *addr,
                                         float _Complex val);
EXTERN void __kmpc_atomic_cmplx4_mul(kmp_Indent *loc, int tid,
                                     float _Complex *addr, float _Complex val);
EXTERN void __kmpc_atomic_cmplx4_div(kmp_Indent *loc, int tid,
                                     float _Complex *addr, float _Complex val);
EXTERN void __kmpc_atomic_cmplx4_div_rev(kmp_Indent *loc, int tid,
                                         float _Complex *addr,
                                         float _Complex val);
EXTERN void __kmpc_atomic_cmplx4_swp(kmp_Indent *loc, int tid,
                                     float _Complex *addr, float _Complex val);
EXTERN void __kmpc_atomic_cmplx8_add(kmp_Indent *loc, int tid,
                                     double _Complex *addr,
                                     double _Complex val);
EXTERN void __kmpc_atomic_cmplx8_sub(kmp_Indent *loc, int tid,
                                     double _Complex *addr,
                                     double _Complex val);
EXTERN void __kmpc_atomic_cmplx8_sub_rev(kmp_Indent *loc, int tid,
                                         double _Complex *addr,
                                         double _Complex val);
EXTERN void __kmpc_atomic_cmplx8_mul(kmp_Indent *loc, int tid,
                                     double _Complex *addr,
                                     double _Complex val);
EXTERN void __kmpc_atomic_cmplx8_div(kmp_Indent *loc, int tid,
                                     double _Complex *addr,
                                     double _Complex val);
EXTERN void __kmpc_atomic_cmplx8_div_rev(kmp_Indent *loc, int tid,
                                         double _Complex *addr,
                                         double _Complex val);
EXTERN void __kmpc_atomic_cmplx8_swp(kmp_Indent *loc, int tid,
                                     double _Complex *addr,
                                     double _Complex val);
EXTERN int8_t __kmpc_atomic_fixed1_wr_cpt(kmp_Indent *loc, int tid,
                                          int8_t *lhs, int8_t rhs,
                                          int atomicFlag);
EXTERN int8_t __kmpc_atomic_fixed1_add_cpt(kmp_Indent *loc, int tid,
                                           int8_t *lhs, int8_t rhs,
                                           int atomicFlag);
EXTERN int8_t __kmpc_atomic_fixed1_sub_cpt(kmp_Indent *loc, int tid,
                                           int8_t *lhs, int8_t rhs,
                                           int atomicFlag);
EXTERN int8_t __kmpc_atomic_fixed1_sub_cpt_rev(kmp_Indent *loc, int tid,
                                               int8_t *lhs, int8_t rhs,
                                               int atomicFlag);
EXTERN int8_t __kmpc_atomic_fixed1_mul_cpt(kmp_Indent *loc, int tid,
                                           int8_t *lhs, int8_t rhs,
                                           int atomicFlag);
EXTERN int8_t __kmpc_atomic_fixed1_div_cpt(kmp_Indent *loc, int tid,
                                           int8_t *lhs, int8_t rhs,
                                           int atomicFlag);
EXTERN int8_t __kmpc_atomic_fixed1_div_cpt_rev(kmp_Indent *loc, int tid,
                                               int8_t *lhs, int8_t rhs,
                                               int atomicFlag);
EXTERN uint8_t __kmpc_atomic_fixed1u_div_cpt(kmp_Indent *loc, int tid,
                                             uint8_t *lhs, uint8_t rhs,
                                             int atomicFlag);
EXTERN uint8_t __kmpc_atomic_fixed1u_div_cpt_rev(kmp_Indent *loc, int tid,
                                                 uint8_t *lhs, uint8_t rhs,
                                                 int atomicFlag);
EXTERN int8_t __kmpc_atomic_fixed1_min_cpt(kmp_Indent *loc, int tid,
                                           int8_t *lhs, int8_t rhs,
                                           int atomicFlag);
EXTERN int8_t __kmpc_atomic_fixed1_max_cpt(kmp_Indent *loc, int tid,
                                           int8_t *lhs, int8_t rhs,
                                           int atomicFlag);
EXTERN int8_t __kmpc_atomic_fixed1_andb_cpt(kmp_Indent *loc, int tid,
                                            int8_t *lhs, int8_t rhs,
                                            int atomicFlag);
EXTERN int8_t __kmpc_atomic_fixed1_orb_cpt(kmp_Indent *loc, int tid,
                                           int8_t *lhs, int8_t rhs,
                                           int atomicFlag);
EXTERN int8_t __kmpc_atomic_fixed1_xor_cpt(kmp_Indent *loc, int tid,
                                           int8_t *lhs, int8_t rhs,
                                           int atomicFlag);
EXTERN int8_t __kmpc_atomic_fixed1_andl_cpt(kmp_Indent *loc, int tid,
                                            int8_t *lhs, int8_t rhs,
                                            int atomicFlag);
EXTERN int8_t __kmpc_atomic_fixed1_orl_cpt(kmp_Indent *loc, int tid,
                                           int8_t *lhs, int8_t rhs,
                                           int atomicFlag);
EXTERN int8_t __kmpc_atomic_fixed1_eqv_cpt(kmp_Indent *loc, int tid,
                                           int8_t *lhs, int8_t rhs,
                                           int atomicFlag);
EXTERN int8_t __kmpc_atomic_fixed1_neqv_cpt(kmp_Indent *loc, int tid,
                                            int8_t *lhs, int8_t rhs,
                                            int atomicFlag);
EXTERN int8_t __kmpc_atomic_fixed1_shl_cpt(kmp_Indent *loc, int tid,
                                           int8_t *lhs, int8_t rhs,
                                           int atomicFlag);
EXTERN int8_t __kmpc_atomic_fixed1_shl_cpt_rev(kmp_Indent *loc, int tid,
                                               int8_t *lhs, int8_t rhs,
                                               int atomicFlag);
EXTERN int8_t __kmpc_atomic_fixed1_shr_cpt(kmp_Indent *loc, int tid,
                                           int8_t *lhs, int8_t rhs,
                                           int atomicFlag);
EXTERN int8_t __kmpc_atomic_fixed1_shr_cpt_rev(kmp_Indent *loc, int tid,
                                               int8_t *lhs, int8_t rhs,
                                               int atomicFlag);
EXTERN uint8_t __kmpc_atomic_fixed1u_shr_cpt(kmp_Indent *loc, int tid,
                                             uint8_t *lhs, uint8_t rhs,
                                             int atomicFlag);
EXTERN uint8_t __kmpc_atomic_fixed1u_shr_cpt_rev(kmp_Indent *loc, int tid,
                                                 uint8_t *lhs, uint8_t rhs,
                                                 int atomicFlag);
EXTERN int16_t __kmpc_atomic_fixed2_wr_cpt(kmp_Indent *loc, int tid,
                                           int16_t *lhs, int16_t rhs,
                                           int atomicFlag);
EXTERN int16_t __kmpc_atomic_fixed2_add_cpt(kmp_Indent *loc, int tid,
                                            int16_t *lhs, int16_t rhs,
                                            int atomicFlag);
EXTERN int16_t __kmpc_atomic_fixed2_sub_cpt(kmp_Indent *loc, int tid,
                                            int16_t *lhs, int16_t rhs,
                                            int atomicFlag);
EXTERN int16_t __kmpc_atomic_fixed2_sub_cpt_rev(kmp_Indent *loc, int tid,
                                                int16_t *lhs, int16_t rhs,
                                                int atomicFlag);
EXTERN int16_t __kmpc_atomic_fixed2_mul_cpt(kmp_Indent *loc, int tid,
                                            int16_t *lhs, int16_t rhs,
                                            int atomicFlag);
EXTERN int16_t __kmpc_atomic_fixed2_div_cpt(kmp_Indent *loc, int tid,
                                            int16_t *lhs, int16_t rhs,
                                            int atomicFlag);
EXTERN int16_t __kmpc_atomic_fixed2_div_cpt_rev(kmp_Indent *loc, int tid,
                                                int16_t *lhs, int16_t rhs,
                                                int atomicFlag);
EXTERN uint16_t __kmpc_atomic_fixed2u_div_cpt(kmp_Indent *loc, int tid,
                                              uint16_t *lhs, uint16_t rhs,
                                              int atomicFlag);
EXTERN uint16_t __kmpc_atomic_fixed2u_div_cpt_rev(kmp_Indent *loc, int tid,
                                                  uint16_t *lhs, uint16_t rhs,
                                                  int atomicFlag);
EXTERN int16_t __kmpc_atomic_fixed2_min_cpt(kmp_Indent *loc, int tid,
                                            int16_t *lhs, int16_t rhs,
                                            int atomicFlag);
EXTERN int16_t __kmpc_atomic_fixed2_max_cpt(kmp_Indent *loc, int tid,
                                            int16_t *lhs, int16_t rhs,
                                            int atomicFlag);
EXTERN int16_t __kmpc_atomic_fixed2_andb_cpt(kmp_Indent *loc, int tid,
                                             int16_t *lhs, int16_t rhs,
                                             int atomicFlag);
EXTERN int16_t __kmpc_atomic_fixed2_orb_cpt(kmp_Indent *loc, int tid,
                                            int16_t *lhs, int16_t rhs,
                                            int atomicFlag);
EXTERN int16_t __kmpc_atomic_fixed2_xor_cpt(kmp_Indent *loc, int tid,
                                            int16_t *lhs, int16_t rhs,
                                            int atomicFlag);
EXTERN int16_t __kmpc_atomic_fixed2_andl_cpt(kmp_Indent *loc, int tid,
                                             int16_t *lhs, int16_t rhs,
                                             int atomicFlag);
EXTERN int16_t __kmpc_atomic_fixed2_orl_cpt(kmp_Indent *loc, int tid,
                                            int16_t *lhs, int16_t rhs,
                                            int atomicFlag);
EXTERN int16_t __kmpc_atomic_fixed2_eqv_cpt(kmp_Indent *loc, int tid,
                                            int16_t *lhs, int16_t rhs,
                                            int atomicFlag);
EXTERN int16_t __kmpc_atomic_fixed2_neqv_cpt(kmp_Indent *loc, int tid,
                                             int16_t *lhs, int16_t rhs,
                                             int atomicFlag);
EXTERN int16_t __kmpc_atomic_fixed2_shl_cpt(kmp_Indent *loc, int tid,
                                            int16_t *lhs, int16_t rhs,
                                            int atomicFlag);
EXTERN int16_t __kmpc_atomic_fixed2_shl_cpt_rev(kmp_Indent *loc, int tid,
                                                int16_t *lhs, int16_t rhs,
                                                int atomicFlag);
EXTERN int16_t __kmpc_atomic_fixed2_shr_cpt(kmp_Indent *loc, int tid,
                                            int16_t *lhs, int16_t rhs,
                                            int atomicFlag);
EXTERN int16_t __kmpc_atomic_fixed2_shr_cpt_rev(kmp_Indent *loc, int tid,
                                                int16_t *lhs, int16_t rhs,
                                                int atomicFlag);
EXTERN uint16_t __kmpc_atomic_fixed2u_shr_cpt(kmp_Indent *loc, int tid,
                                              uint16_t *lhs, uint16_t rhs,
                                              int atomicFlag);
EXTERN uint16_t __kmpc_atomic_fixed2u_shr_cpt_rev(kmp_Indent *loc, int tid,
                                                  uint16_t *lhs, uint16_t rhs,
                                                  int atomicFlag);
EXTERN int16_t __kmpc_atomic_fixed2_swp_cpt(kmp_Indent *loc, int tid,
                                            int16_t *lhs, int16_t rhs,
                                            int atomicFlag);
EXTERN int32_t __kmpc_atomic_fixed4_wr_cpt(kmp_Indent *loc, int tid,
                                           int32_t *lhs, int32_t rhs,
                                           int atomicFlag);
EXTERN int32_t __kmpc_atomic_fixed4_add_cpt(kmp_Indent *loc, int tid,
                                            int32_t *lhs, int32_t rhs,
                                            int atomicFlag);
EXTERN int32_t __kmpc_atomic_fixed4_sub_cpt(kmp_Indent *loc, int tid,
                                            int32_t *lhs, int32_t rhs,
                                            int atomicFlag);
EXTERN int32_t __kmpc_atomic_fixed4_sub_cpt_rev(kmp_Indent *loc, int tid,
                                                int32_t *lhs, int32_t rhs,
                                                int atomicFlag);
EXTERN int32_t __kmpc_atomic_fixed4_mul_cpt(kmp_Indent *loc, int tid,
                                            int32_t *lhs, int32_t rhs,
                                            int atomicFlag);
EXTERN int32_t __kmpc_atomic_fixed4_div_cpt(kmp_Indent *loc, int tid,
                                            int32_t *lhs, int32_t rhs,
                                            int atomicFlag);
EXTERN int32_t __kmpc_atomic_fixed4_div_cpt_rev(kmp_Indent *loc, int tid,
                                                int32_t *lhs, int32_t rhs,
                                                int atomicFlag);
EXTERN uint32_t __kmpc_atomic_fixed4u_div_cpt(kmp_Indent *loc, int tid,
                                              uint32_t *lhs, uint32_t rhs,
                                              int atomicFlag);
EXTERN uint32_t __kmpc_atomic_fixed4u_div_cpt_rev(kmp_Indent *loc, int tid,
                                                  uint32_t *lhs, uint32_t rhs,
                                                  int atomicFlag);
EXTERN int32_t __kmpc_atomic_fixed4_min_cpt(kmp_Indent *loc, int tid,
                                            int32_t *lhs, int32_t rhs,
                                            int atomicFlag);
EXTERN int32_t __kmpc_atomic_fixed4_max_cpt(kmp_Indent *loc, int tid,
                                            int32_t *lhs, int32_t rhs,
                                            int atomicFlag);
EXTERN int32_t __kmpc_atomic_fixed4_andb_cpt(kmp_Indent *loc, int tid,
                                             int32_t *lhs, int32_t rhs,
                                             int atomicFlag);
EXTERN int32_t __kmpc_atomic_fixed4_orb_cpt(kmp_Indent *loc, int tid,
                                            int32_t *lhs, int32_t rhs,
                                            int atomicFlag);
EXTERN int32_t __kmpc_atomic_fixed4_xor_cpt(kmp_Indent *loc, int tid,
                                            int32_t *lhs, int32_t rhs,
                                            int atomicFlag);
EXTERN int32_t __kmpc_atomic_fixed4_andl_cpt(kmp_Indent *loc, int tid,
                                             int32_t *lhs, int32_t rhs,
                                             int atomicFlag);
EXTERN int32_t __kmpc_atomic_fixed4_orl_cpt(kmp_Indent *loc, int tid,
                                            int32_t *lhs, int32_t rhs,
                                            int atomicFlag);
EXTERN int32_t __kmpc_atomic_fixed4_eqv_cpt(kmp_Indent *loc, int tid,
                                            int32_t *lhs, int32_t rhs,
                                            int atomicFlag);
EXTERN int32_t __kmpc_atomic_fixed4_neqv_cpt(kmp_Indent *loc, int tid,
                                             int32_t *lhs, int32_t rhs,
                                             int atomicFlag);
EXTERN int32_t __kmpc_atomic_fixed4_shl_cpt(kmp_Indent *loc, int tid,
                                            int32_t *lhs, int32_t rhs,
                                            int atomicFlag);
EXTERN int32_t __kmpc_atomic_fixed4_shl_cpt_rev(kmp_Indent *loc, int tid,
                                                int32_t *lhs, int32_t rhs,
                                                int atomicFlag);
EXTERN int32_t __kmpc_atomic_fixed4_shr_cpt(kmp_Indent *loc, int tid,
                                            int32_t *lhs, int32_t rhs,
                                            int atomicFlag);
EXTERN int32_t __kmpc_atomic_fixed4_shr_cpt_rev(kmp_Indent *loc, int tid,
                                                int32_t *lhs, int32_t rhs,
                                                int atomicFlag);
EXTERN uint32_t __kmpc_atomic_fixed4u_shr_cpt(kmp_Indent *loc, int tid,
                                              uint32_t *lhs, uint32_t rhs,
                                              int atomicFlag);
EXTERN uint32_t __kmpc_atomic_fixed4u_shr_cpt_rev(kmp_Indent *loc, int tid,
                                                  uint32_t *lhs, uint32_t rhs,
                                                  int atomicFlag);
EXTERN int32_t __kmpc_atomic_fixed4_swp_cpt(kmp_Indent *loc, int tid,
                                            int32_t *lhs, int32_t rhs,
                                            int atomicFlag);
EXTERN int64_t __kmpc_atomic_fixed8_wr_cpt(kmp_Indent *loc, int tid,
                                           int64_t *lhs, int64_t rhs,
                                           int atomicFlag);
EXTERN int64_t __kmpc_atomic_fixed8_add_cpt(kmp_Indent *loc, int tid,
                                            int64_t *lhs, int64_t rhs,
                                            int atomicFlag);
EXTERN int64_t __kmpc_atomic_fixed8_sub_cpt(kmp_Indent *loc, int tid,
                                            int64_t *lhs, int64_t rhs,
                                            int atomicFlag);
EXTERN int64_t __kmpc_atomic_fixed8_sub_cpt_rev(kmp_Indent *loc, int tid,
                                                int64_t *lhs, int64_t rhs,
                                                int atomicFlag);
EXTERN int64_t __kmpc_atomic_fixed8_mul_cpt(kmp_Indent *loc, int tid,
                                            int64_t *lhs, int64_t rhs,
                                            int atomicFlag);
EXTERN int64_t __kmpc_atomic_fixed8_div_cpt(kmp_Indent *loc, int tid,
                                            int64_t *lhs, int64_t rhs,
                                            int atomicFlag);
EXTERN int64_t __kmpc_atomic_fixed8_div_cpt_rev(kmp_Indent *loc, int tid,
                                                int64_t *lhs, int64_t rhs,
                                                int atomicFlag);
EXTERN uint64_t __kmpc_atomic_fixed8u_div_cpt(kmp_Indent *loc, int tid,
                                              uint64_t *lhs, uint64_t rhs,
                                              int atomicFlag);
EXTERN uint64_t __kmpc_atomic_fixed8u_div_cpt_rev(kmp_Indent *loc, int tid,
                                                  uint64_t *lhs, uint64_t rhs,
                                                  int atomicFlag);
EXTERN int64_t __kmpc_atomic_fixed8_min_cpt(kmp_Indent *loc, int tid,
                                            int64_t *lhs, int64_t rhs,
                                            int atomicFlag);
EXTERN int64_t __kmpc_atomic_fixed8_max_cpt(kmp_Indent *loc, int tid,
                                            int64_t *lhs, int64_t rhs,
                                            int atomicFlag);
EXTERN int64_t __kmpc_atomic_fixed8_andb_cpt(kmp_Indent *loc, int tid,
                                             int64_t *lhs, int64_t rhs,
                                             int atomicFlag);
EXTERN int64_t __kmpc_atomic_fixed8_orb_cpt(kmp_Indent *loc, int tid,
                                            int64_t *lhs, int64_t rhs,
                                            int atomicFlag);
EXTERN int64_t __kmpc_atomic_fixed8_xor_cpt(kmp_Indent *loc, int tid,
                                            int64_t *lhs, int64_t rhs,
                                            int atomicFlag);
EXTERN int64_t __kmpc_atomic_fixed8_andl_cpt(kmp_Indent *loc, int tid,
                                             int64_t *lhs, int64_t rhs,
                                             int atomicFlag);
EXTERN int64_t __kmpc_atomic_fixed8_orl_cpt(kmp_Indent *loc, int tid,
                                            int64_t *lhs, int64_t rhs,
                                            int atomicFlag);
EXTERN int64_t __kmpc_atomic_fixed8_eqv_cpt(kmp_Indent *loc, int tid,
                                            int64_t *lhs, int64_t rhs,
                                            int atomicFlag);
EXTERN int64_t __kmpc_atomic_fixed8_neqv_cpt(kmp_Indent *loc, int tid,
                                             int64_t *lhs, int64_t rhs,
                                             int atomicFlag);
EXTERN int64_t __kmpc_atomic_fixed8_shl_cpt(kmp_Indent *loc, int tid,
                                            int64_t *lhs, int64_t rhs,
                                            int atomicFlag);
EXTERN int64_t __kmpc_atomic_fixed8_shl_cpt_rev(kmp_Indent *loc, int tid,
                                                int64_t *lhs, int64_t rhs,
                                                int atomicFlag);
EXTERN int64_t __kmpc_atomic_fixed8_shr_cpt(kmp_Indent *loc, int tid,
                                            int64_t *lhs, int64_t rhs,
                                            int atomicFlag);
EXTERN int64_t __kmpc_atomic_fixed8_shr_cpt_rev(kmp_Indent *loc, int tid,
                                                int64_t *lhs, int64_t rhs,
                                                int atomicFlag);
EXTERN uint64_t __kmpc_atomic_fixed8u_shr_cpt(kmp_Indent *loc, int tid,
                                              uint64_t *lhs, uint64_t rhs,
                                              int atomicFlag);
EXTERN uint64_t __kmpc_atomic_fixed8u_shr_cpt_rev(kmp_Indent *loc, int tid,
                                                  uint64_t *lhs, uint64_t rhs,
                                                  int atomicFlag);
EXTERN int64_t __kmpc_atomic_fixed8_swp_cpt(kmp_Indent *loc, int tid,
                                            int64_t *lhs, int64_t rhs,
                                            int atomicFlag);
EXTERN float __kmpc_atomic_float4_add_cpt(kmp_Indent *loc, int tid, float *lhs,
                                          float rhs, int atomicFlag);
EXTERN float __kmpc_atomic_float4_sub_cpt(kmp_Indent *loc, int tid, float *lhs,
                                          float rhs, int atomicFlag);
EXTERN float __kmpc_atomic_float4_sub_cpt_rev(kmp_Indent *loc, int tid,
                                              float *lhs, float rhs,
                                              int atomicFlag);
EXTERN float __kmpc_atomic_float4_mul_cpt(kmp_Indent *loc, int tid, float *lhs,
                                          float rhs, int atomicFlag);
EXTERN float __kmpc_atomic_float4_div_cpt(kmp_Indent *loc, int tid, float *lhs,
                                          float rhs, int atomicFlag);
EXTERN float __kmpc_atomic_float4_div_cpt_rev(kmp_Indent *loc, int tid,
                                              float *lhs, float rhs,
                                              int atomicFlag);
EXTERN float __kmpc_atomic_float4_min_cpt(kmp_Indent *loc, int tid, float *lhs,
                                          float rhs, int atomicFlag);
EXTERN float __kmpc_atomic_float4_max_cpt(kmp_Indent *loc, int tid, float *lhs,
                                          float rhs, int atomicFlag);
EXTERN double __kmpc_atomic_float8_add_cpt(kmp_Indent *loc, int tid,
                                           double *lhs, double rhs,
                                           int atomicFlag);
EXTERN double __kmpc_atomic_float8_sub_cpt(kmp_Indent *loc, int tid,
                                           double *lhs, double rhs,
                                           int atomicFlag);
EXTERN double __kmpc_atomic_float8_sub_cpt_rev(kmp_Indent *loc, int tid,
                                               double *lhs, double rhs,
                                               int atomicFlag);
EXTERN double __kmpc_atomic_float8_mul_cpt(kmp_Indent *loc, int tid,
                                           double *lhs, double rhs,
                                           int atomicFlag);
EXTERN double __kmpc_atomic_float8_div_cpt(kmp_Indent *loc, int tid,
                                           double *lhs, double rhs,
                                           int atomicFlag);
EXTERN double __kmpc_atomic_float8_div_cpt_rev(kmp_Indent *loc, int tid,
                                               double *lhs, double rhs,
                                               int atomicFlag);
EXTERN double __kmpc_atomic_float8_min_cpt(kmp_Indent *loc, int tid,
                                           double *lhs, double rhs,
                                           int atomicFlag);
EXTERN double __kmpc_atomic_float8_max_cpt(kmp_Indent *loc, int tid,
                                           double *lhs, double rhs,
                                           int atomicFlag);

// special case according to iomp reference
EXTERN void __kmpc_atomic_cmplx4_add_cpt(kmp_Indent *loc, int tid,
                                         float _Complex *lhs,
                                         float _Complex rhs,
                                         float _Complex *out, int atomicFlag);
EXTERN void __kmpc_atomic_cmplx4_sub_cpt(kmp_Indent *loc, int tid,
                                         float _Complex *lhs,
                                         float _Complex rhs,
                                         float _Complex *out, int atomicFlag);
EXTERN void __kmpc_atomic_cmplx4_sub_cpt_rev(kmp_Indent *loc, int tid,
                                             float _Complex *lhs,
                                             float _Complex rhs,
                                             float _Complex *out,
                                             int atomicFlag);
EXTERN void __kmpc_atomic_cmplx4_mul_cpt(kmp_Indent *loc, int tid,
                                         float _Complex *lhs,
                                         float _Complex rhs,
                                         float _Complex *out, int atomicFlag);
EXTERN void __kmpc_atomic_cmplx4_div_cpt(kmp_Indent *loc, int tid,
                                         float _Complex *lhs,
                                         float _Complex rhs,
                                         float _Complex *out, int atomicFlag);
EXTERN void __kmpc_atomic_cmplx4_div_cpt_rev(kmp_Indent *loc, int tid,
                                             float _Complex *lhs,
                                             float _Complex rhs,
                                             float _Complex *out,
                                             int atomicFlag);
EXTERN void __kmpc_atomic_cmplx4_swp_cpt(kmp_Indent *loc, int tid,
                                         float _Complex *lhs,
                                         float _Complex rhs,
                                         float _Complex *out, int atomicFlag);

EXTERN double _Complex __kmpc_atomic_cmplx8_add_cpt(kmp_Indent *loc, int tid,
                                                    double _Complex *lhs,
                                                    double _Complex rhs,
                                                    int atomicFlag);
EXTERN double _Complex __kmpc_atomic_cmplx8_sub_cpt(kmp_Indent *loc, int tid,
                                                    double _Complex *lhs,
                                                    double _Complex rhs,
                                                    int atomicFlag);
EXTERN double _Complex __kmpc_atomic_cmplx8_sub_cpt_rev(kmp_Indent *loc,
                                                        int tid,
                                                        double _Complex *lhs,
                                                        double _Complex rhs,
                                                        int atomicFlag);
EXTERN double _Complex __kmpc_atomic_cmplx8_mul_cpt(kmp_Indent *loc, int tid,
                                                    double _Complex *lhs,
                                                    double _Complex rhs,
                                                    int atomicFlag);
EXTERN double _Complex __kmpc_atomic_cmplx8_div_cpt(kmp_Indent *loc, int tid,
                                                    double _Complex *lhs,
                                                    double _Complex rhs,
                                                    int atomicFlag);
EXTERN double _Complex __kmpc_atomic_cmplx8_div_cpt_rev(kmp_Indent *loc,
                                                        int tid,
                                                        double _Complex *lhs,
                                                        double _Complex rhs,
                                                        int atomicFlag);
EXTERN double _Complex __kmpc_atomic_cmplx8_swp_cpt(kmp_Indent *loc, int tid,
                                                    double _Complex *lhs,
                                                    double _Complex rhs,
                                                    int atomicFlag);

// atomic for array reduction

EXTERN void __array_atomic_fixed1_add(kmp_Indent *id_ref, int32_t tid,
                                      int8_t *lhs, int8_t *rhs, int64_t n);
EXTERN void __array_atomic_fixed1_sub(kmp_Indent *id_ref, int32_t tid,
                                      int8_t *lhs, int8_t *rhs, int64_t n);
EXTERN void __array_atomic_fixed1_mul(kmp_Indent *id_ref, int32_t tid,
                                      int8_t *lhs, int8_t *rhs, int64_t n);
EXTERN void __array_atomic_fixed1_div(kmp_Indent *id_ref, int32_t tid,
                                      int8_t *lhs, int8_t *rhs, int64_t n);
EXTERN void __array_atomic_fixed1_min(kmp_Indent *id_ref, int32_t tid,
                                      int8_t *lhs, int8_t *rhs, int64_t n);
EXTERN void __array_atomic_fixed1_max(kmp_Indent *id_ref, int32_t tid,
                                      int8_t *lhs, int8_t *rhs, int64_t n);
EXTERN void __array_atomic_fixed1_andb(kmp_Indent *id_ref, int32_t tid,
                                       int8_t *lhs, int8_t *rhs, int64_t n);
EXTERN void __array_atomic_fixed1_orb(kmp_Indent *id_ref, int32_t tid,
                                      int8_t *lhs, int8_t *rhs, int64_t n);
EXTERN void __array_atomic_fixed1_xor(kmp_Indent *id_ref, int32_t tid,
                                      int8_t *lhs, int8_t *rhs, int64_t n);
EXTERN void __array_atomic_fixed1_shl(kmp_Indent *id_ref, int32_t tid,
                                      int8_t *lhs, int8_t *rhs, int64_t n);
EXTERN void __array_atomic_fixed1_shr(kmp_Indent *id_ref, int32_t tid,
                                      int8_t *lhs, int8_t *rhs, int64_t n);
EXTERN void __array_atomic_fixed1_andl(kmp_Indent *id_ref, int32_t tid,
                                       int8_t *lhs, int8_t *rhs, int64_t n);
EXTERN void __array_atomic_fixed1_orl(kmp_Indent *id_ref, int32_t tid,
                                      int8_t *lhs, int8_t *rhs, int64_t n);
EXTERN void __array_atomic_fixed1_eqv(kmp_Indent *id_ref, int32_t tid,
                                      int8_t *lhs, int8_t *rhs, int64_t n);
EXTERN void __array_atomic_fixed1_neqv(kmp_Indent *id_ref, int32_t tid,
                                       int8_t *lhs, int8_t *rhs, int64_t n);
EXTERN void __array_atomic_fixed2_add(kmp_Indent *id_ref, int32_t tid,
                                      int16_t *lhs, int16_t *rhs, int64_t n);
EXTERN void __array_atomic_fixed2_sub(kmp_Indent *id_ref, int32_t tid,
                                      int16_t *lhs, int16_t *rhs, int64_t n);
EXTERN void __array_atomic_fixed2_mul(kmp_Indent *id_ref, int32_t tid,
                                      int16_t *lhs, int16_t *rhs, int64_t n);
EXTERN void __array_atomic_fixed2_div(kmp_Indent *id_ref, int32_t tid,
                                      int16_t *lhs, int16_t *rhs, int64_t n);
EXTERN void __array_atomic_fixed2_min(kmp_Indent *id_ref, int32_t tid,
                                      int16_t *lhs, int16_t *rhs, int64_t n);
EXTERN void __array_atomic_fixed2_max(kmp_Indent *id_ref, int32_t tid,
                                      int16_t *lhs, int16_t *rhs, int64_t n);
EXTERN void __array_atomic_fixed2_andb(kmp_Indent *id_ref, int32_t tid,
                                       int16_t *lhs, int16_t *rhs, int64_t n);
EXTERN void __array_atomic_fixed2_orb(kmp_Indent *id_ref, int32_t tid,
                                      int16_t *lhs, int16_t *rhs, int64_t n);
EXTERN void __array_atomic_fixed2_xor(kmp_Indent *id_ref, int32_t tid,
                                      int16_t *lhs, int16_t *rhs, int64_t n);
EXTERN void __array_atomic_fixed2_shl(kmp_Indent *id_ref, int32_t tid,
                                      int16_t *lhs, int16_t *rhs, int64_t n);
EXTERN void __array_atomic_fixed2_shr(kmp_Indent *id_ref, int32_t tid,
                                      int16_t *lhs, int16_t *rhs, int64_t n);
EXTERN void __array_atomic_fixed2_andl(kmp_Indent *id_ref, int32_t tid,
                                       int16_t *lhs, int16_t *rhs, int64_t n);
EXTERN void __array_atomic_fixed2_orl(kmp_Indent *id_ref, int32_t tid,
                                      int16_t *lhs, int16_t *rhs, int64_t n);
EXTERN void __array_atomic_fixed2_eqv(kmp_Indent *id_ref, int32_t tid,
                                      int16_t *lhs, int16_t *rhs, int64_t n);
EXTERN void __array_atomic_fixed2_neqv(kmp_Indent *id_ref, int32_t tid,
                                       int16_t *lhs, int16_t *rhs, int64_t n);
EXTERN void __array_atomic_fixed4_add(kmp_Indent *id_ref, int32_t tid,
                                      int32_t *lhs, int32_t *rhs, int64_t n);
EXTERN void __array_atomic_fixed4_sub(kmp_Indent *id_ref, int32_t tid,
                                      int32_t *lhs, int32_t *rhs, int64_t n);
EXTERN void __array_atomic_fixed4_mul(kmp_Indent *id_ref, int32_t tid,
                                      int32_t *lhs, int32_t *rhs, int64_t n);
EXTERN void __array_atomic_fixed4_div(kmp_Indent *id_ref, int32_t tid,
                                      int32_t *lhs, int32_t *rhs, int64_t n);
EXTERN void __array_atomic_fixed4_min(kmp_Indent *id_ref, int32_t tid,
                                      int32_t *lhs, int32_t *rhs, int64_t n);
EXTERN void __array_atomic_fixed4_max(kmp_Indent *id_ref, int32_t tid,
                                      int32_t *lhs, int32_t *rhs, int64_t n);
EXTERN void __array_atomic_fixed4_andb(kmp_Indent *id_ref, int32_t tid,
                                       int32_t *lhs, int32_t *rhs, int64_t n);
EXTERN void __array_atomic_fixed4_orb(kmp_Indent *id_ref, int32_t tid,
                                      int32_t *lhs, int32_t *rhs, int64_t n);
EXTERN void __array_atomic_fixed4_xor(kmp_Indent *id_ref, int32_t tid,
                                      int32_t *lhs, int32_t *rhs, int64_t n);
EXTERN void __array_atomic_fixed4_shl(kmp_Indent *id_ref, int32_t tid,
                                      int32_t *lhs, int32_t *rhs, int64_t n);
EXTERN void __array_atomic_fixed4_shr(kmp_Indent *id_ref, int32_t tid,
                                      int32_t *lhs, int32_t *rhs, int64_t n);
EXTERN void __array_atomic_fixed4_andl(kmp_Indent *id_ref, int32_t tid,
                                       int32_t *lhs, int32_t *rhs, int64_t n);
EXTERN void __array_atomic_fixed4_orl(kmp_Indent *id_ref, int32_t tid,
                                      int32_t *lhs, int32_t *rhs, int64_t n);
EXTERN void __array_atomic_fixed4_eqv(kmp_Indent *id_ref, int32_t tid,
                                      int32_t *lhs, int32_t *rhs, int64_t n);
EXTERN void __array_atomic_fixed4_neqv(kmp_Indent *id_ref, int32_t tid,
                                       int32_t *lhs, int32_t *rhs, int64_t n);
EXTERN void __array_atomic_fixed8_add(kmp_Indent *id_ref, int32_t tid,
                                      int64_t *lhs, int64_t *rhs, int64_t n);
EXTERN void __array_atomic_fixed8_sub(kmp_Indent *id_ref, int32_t tid,
                                      int64_t *lhs, int64_t *rhs, int64_t n);
EXTERN void __array_atomic_fixed8_mul(kmp_Indent *id_ref, int32_t tid,
                                      int64_t *lhs, int64_t *rhs, int64_t n);
EXTERN void __array_atomic_fixed8_div(kmp_Indent *id_ref, int32_t tid,
                                      int64_t *lhs, int64_t *rhs, int64_t n);
EXTERN void __array_atomic_fixed8_min(kmp_Indent *id_ref, int32_t tid,
                                      int64_t *lhs, int64_t *rhs, int64_t n);
EXTERN void __array_atomic_fixed8_max(kmp_Indent *id_ref, int32_t tid,
                                      int64_t *lhs, int64_t *rhs, int64_t n);
EXTERN void __array_atomic_fixed8_andb(kmp_Indent *id_ref, int32_t tid,
                                       int64_t *lhs, int64_t *rhs, int64_t n);
EXTERN void __array_atomic_fixed8_orb(kmp_Indent *id_ref, int32_t tid,
                                      int64_t *lhs, int64_t *rhs, int64_t n);
EXTERN void __array_atomic_fixed8_xor(kmp_Indent *id_ref, int32_t tid,
                                      int64_t *lhs, int64_t *rhs, int64_t n);
EXTERN void __array_atomic_fixed8_shl(kmp_Indent *id_ref, int32_t tid,
                                      int64_t *lhs, int64_t *rhs, int64_t n);
EXTERN void __array_atomic_fixed8_shr(kmp_Indent *id_ref, int32_t tid,
                                      int64_t *lhs, int64_t *rhs, int64_t n);
EXTERN void __array_atomic_fixed8_andl(kmp_Indent *id_ref, int32_t tid,
                                       int64_t *lhs, int64_t *rhs, int64_t n);
EXTERN void __array_atomic_fixed8_orl(kmp_Indent *id_ref, int32_t tid,
                                      int64_t *lhs, int64_t *rhs, int64_t n);
EXTERN void __array_atomic_fixed8_eqv(kmp_Indent *id_ref, int32_t tid,
                                      int64_t *lhs, int64_t *rhs, int64_t n);
EXTERN void __array_atomic_fixed8_neqv(kmp_Indent *id_ref, int32_t tid,
                                       int64_t *lhs, int64_t *rhs, int64_t n);
EXTERN void __array_atomic_float4_add(kmp_Indent *id_ref, int32_t tid,
                                      float *lhs, float *rhs, int64_t n);
EXTERN void __array_atomic_float4_sub(kmp_Indent *id_ref, int32_t tid,
                                      float *lhs, float *rhs, int64_t n);
EXTERN void __array_atomic_float4_mul(kmp_Indent *id_ref, int32_t tid,
                                      float *lhs, float *rhs, int64_t n);
EXTERN void __array_atomic_float4_div(kmp_Indent *id_ref, int32_t tid,
                                      float *lhs, float *rhs, int64_t n);
EXTERN void __array_atomic_float4_min(kmp_Indent *id_ref, int32_t tid,
                                      float *lhs, float *rhs, int64_t n);
EXTERN void __array_atomic_float4_max(kmp_Indent *id_ref, int32_t tid,
                                      float *lhs, float *rhs, int64_t n);
EXTERN void __array_atomic_float8_add(kmp_Indent *id_ref, int32_t tid,
                                      double *lhs, double *rhs, int64_t n);
EXTERN void __array_atomic_float8_sub(kmp_Indent *id_ref, int32_t tid,
                                      double *lhs, double *rhs, int64_t n);
EXTERN void __array_atomic_float8_mul(kmp_Indent *id_ref, int32_t tid,
                                      double *lhs, double *rhs, int64_t n);
EXTERN void __array_atomic_float8_div(kmp_Indent *id_ref, int32_t tid,
                                      double *lhs, double *rhs, int64_t n);
EXTERN void __array_atomic_float8_min(kmp_Indent *id_ref, int32_t tid,
                                      double *lhs, double *rhs, int64_t n);
EXTERN void __array_atomic_float8_max(kmp_Indent *id_ref, int32_t tid,
                                      double *lhs, double *rhs, int64_t n);

// non standard
EXTERN void __kmpc_kernel_init(int ThreadLimit);
EXTERN void __kmpc_kernel_deinit();
EXTERN void __kmpc_spmd_kernel_init(int ThreadLimit,
                                    short RequiresOMPRuntime,
                                    short RequiresDataSharing);
EXTERN void __kmpc_spmd_kernel_deinit();
EXTERN void __kmpc_kernel_prepare_parallel(void *WorkFn);
EXTERN bool __kmpc_kernel_parallel(void **WorkFn);
EXTERN void __kmpc_kernel_end_parallel();
EXTERN bool __kmpc_kernel_convergent_parallel(void *buffer, bool *IsFinal, int32_t *LaneSource);
EXTERN void __kmpc_kernel_end_convergent_parallel(void *buffer);
EXTERN bool __kmpc_kernel_convergent_simd(void *buffer, bool *IsFinal, int32_t *LaneSource,
                                          int32_t *LaneId, int32_t *NumLanes);
EXTERN void __kmpc_kernel_end_convergent_simd(void *buffer);

// The slot used for data sharing by the master and worker threads. We use a complete (default size version and an incomplete one so that we allow sizes greater than the default).
struct __kmpc_data_sharing_slot {
  __kmpc_data_sharing_slot *Next;
  void *DataEnd;
  char Data[];
};
EXTERN void __kmpc_initialize_data_sharing_environment(
    __kmpc_data_sharing_slot *RootS,
    size_t InitialDataSize);
EXTERN void* __kmpc_data_sharing_environment_begin(
    __kmpc_data_sharing_slot **SavedSharedSlot,
    void **SavedSharedStack,
    void **SavedSharedFrame,
    int32_t *SavedActiveThreads,
    size_t SharingDataSize,
    size_t SharingDefaultDataSize );
EXTERN void __kmpc_data_sharing_environment_end(
    __kmpc_data_sharing_slot **SavedSharedSlot,
    void **SavedSharedStack,
    void **SavedSharedFrame,
    int32_t *SavedActiveThreads,
    int32_t IsEntryPoint );

EXTERN void* __kmpc_get_data_sharing_environment_frame(int32_t SourceThreadID);
//EXTERN void __kmpc_samuel_print(int64_t Bla);
#endif
