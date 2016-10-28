//===------ omptarget.cpp - Target independent OpenMP target RTL -- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.txt for details.
//
//===----------------------------------------------------------------------===//
//
// Implementation of the interface to be used by Clang during the codegen of a
// target region.
//
//===----------------------------------------------------------------------===//

#include <algorithm>
#include <cassert>
#include <climits>
#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <dlfcn.h>
#include <list>
#include <map>
#include <mutex>
#include <queue>
#include <set>
#include <sstream>
#include <string>
#include <vector>

// Header file global to this project
#include "omptarget.h"

#define DP(...) DEBUGP("Libomptarget", __VA_ARGS__)
#define INF_REF_CNT (LONG_MAX>>1) // leave room for additions/subtractions

// List of all plugins that can support offloading.
static const char *RTLNames[] = {
    /* PowerPC target */ "libomptarget.rtl.ppc64.so",
    /* x86_64 target  */ "libomptarget.rtl.x86_64.so",
    /* CUDA target    */ "libomptarget.rtl.cuda.so"};

// forward declarations
struct RTLInfoTy;
static int target(int32_t device_id, void *host_ptr, int32_t arg_num,
    void **args_base, void **args, int64_t *arg_sizes, int64_t *arg_types,
    int32_t team_num, int32_t thread_limit, int IsTeamConstruct);

/// Map between host data and target data.
struct HostDataToTargetTy {
  long HstPtrBase; // host info.
  long HstPtrBegin;
  long HstPtrEnd; // non-inclusive.

  long TgtPtrBegin; // target info.

  long RefCount;

  HostDataToTargetTy()
      : HstPtrBase(0), HstPtrBegin(0), HstPtrEnd(0),
        TgtPtrBegin(0), RefCount(0) {}
  HostDataToTargetTy(long BP, long B, long E, long TB)
      : HstPtrBase(BP), HstPtrBegin(B), HstPtrEnd(E),
        TgtPtrBegin(TB), RefCount(1) {}
};

typedef std::list<HostDataToTargetTy> HostDataToTargetListTy;

/// Map for shadow pointers
struct ShadowPtrValTy {
  void *HstPtrVal;
  void *TgtPtrAddr;
  void *TgtPtrVal;
};
typedef std::map<void *, ShadowPtrValTy> ShadowPtrListTy;

///
struct PendingCtorDtorListsTy {
  std::list<void *> PendingCtors;
  std::list<void *> PendingDtors;
};
typedef std::map<__tgt_bin_desc *, PendingCtorDtorListsTy>
    PendingCtorsDtorsPerLibrary;

struct DeviceTy {
  int32_t DeviceID;
  RTLInfoTy *RTL;
  int32_t RTLDeviceID;

  bool IsInit;
  std::once_flag InitFlag;
  bool HasPendingGlobals;

  HostDataToTargetListTy HostDataToTargetMap;
  PendingCtorsDtorsPerLibrary PendingCtorsDtors;

  ShadowPtrListTy ShadowPtrMap;

  std::mutex DataMapMtx, PendingGlobalsMtx, ShadowMtx;

  uint64_t loopTripCnt;

  DeviceTy(RTLInfoTy *RTL)
      : DeviceID(-1), RTL(RTL), RTLDeviceID(-1), IsInit(false), InitFlag(),
        HasPendingGlobals(false), HostDataToTargetMap(),
        PendingCtorsDtors(), ShadowPtrMap(), DataMapMtx(), PendingGlobalsMtx(),
        ShadowMtx(), loopTripCnt(0) {}

  // The existence of mutexes makes DeviceTy non-copyable. We need to
  // provide a copy constructor and an assignment operator explicitly.
  DeviceTy(const DeviceTy &d)
      : InitFlag(), DataMapMtx(), PendingGlobalsMtx(), ShadowMtx() {
    DeviceID = d.DeviceID;
    RTL = d.RTL;
    RTLDeviceID = d.RTLDeviceID;
    IsInit = d.IsInit;
    HasPendingGlobals = d.HasPendingGlobals;
    HostDataToTargetMap = d.HostDataToTargetMap;
    PendingCtorsDtors = d.PendingCtorsDtors;
    ShadowPtrMap = d.ShadowPtrMap;
    loopTripCnt = d.loopTripCnt;
  }

  DeviceTy& operator=(const DeviceTy &d) {
    DeviceID = d.DeviceID;
    RTL = d.RTL;
    RTLDeviceID = d.RTLDeviceID;
    IsInit = d.IsInit;
    HasPendingGlobals = d.HasPendingGlobals;
    HostDataToTargetMap = d.HostDataToTargetMap;
    PendingCtorsDtors = d.PendingCtorsDtors;
    ShadowPtrMap = d.ShadowPtrMap;
    loopTripCnt = d.loopTripCnt;

    return *this;
  }

  void *getOrAllocTgtPtr(void *HstPtrBegin, void *HstPtrBase, long Size,
                         long &IsNew, long UpdateRefCount = true);
  void *getTgtPtrBegin(void *HstPtrBegin, long Size);
  void *getTgtPtrBegin(void *HstPtrBegin, long Size, long &IsLast,
                       long UpdateRefCount = true);
  void deallocTgtPtr(void *TgtPtrBegin, long Size, long ForceDelete);
  int associatePtr(void *HstPtrBegin, void *TgtPtrBegin, long Size);
  int disassociatePtr(void *HstPtrBegin);
  HostDataToTargetTy *getMapEntry(void *HstPtrBegin);

  // calls to RTL
  int32_t initOnce();
  __tgt_target_table *load_binary(void *Img);

  int32_t data_submit(void *TgtPtrBegin, void *HstPtrBegin, int64_t Size);
  int32_t data_retrieve(void *HstPtrBegin, void *TgtPtrBegin, int64_t Size);

  int32_t run_region(void *TgtEntryPtr, void **TgtVarsPtr, int32_t TgtVarsSize);
  int32_t run_team_region(void *TgtEntryPtr, void **TgtVarsPtr,
                          int32_t TgtVarsSize, int32_t NumTeams,
                          int32_t ThreadLimit, uint64_t LoopTripCount);

private:
  // call to RTL
  void init(); // To be called only via DeviceTy::initOnce()
};

struct RTLInfoTy {
  typedef int32_t(is_valid_binary_ty)(void *);
  typedef int32_t(number_of_devices_ty)();
  typedef int32_t(init_device_ty)(int32_t);
  typedef __tgt_target_table *(load_binary_ty)(int32_t, void *);
  typedef void *(data_alloc_ty)(int32_t, int64_t);
  typedef int32_t(data_submit_ty)(int32_t, void *, void *, int64_t);
  typedef int32_t(data_retrieve_ty)(int32_t, void *, void *, int64_t);
  typedef int32_t(data_delete_ty)(int32_t, void *);
  typedef int32_t(run_region_ty)(int32_t, void *, void **, int32_t);
  typedef int32_t(run_team_region_ty)(int32_t, void *, void **, int32_t,
                                      int32_t, int32_t, uint64_t);

  int32_t Idx;                     // RTL index, index is the number of devices
                                   // of other RTLs that were registered before,
                                   // i.e. the OpenMP index of the first device
                                   // to be registered with this RTL.
  int32_t NumberOfDevices;         // Number of devices this RTL deals with.
  std::vector<DeviceTy *> Devices; // one per device (NumberOfDevices in total).

  void *LibraryHandler;
  // Functions implemented in the RTL.
  is_valid_binary_ty *is_valid_binary;
  number_of_devices_ty *number_of_devices;
  init_device_ty *init_device;
  load_binary_ty *load_binary;
  data_alloc_ty *data_alloc;
  data_submit_ty *data_submit;
  data_retrieve_ty *data_retrieve;
  data_delete_ty *data_delete;
  run_region_ty *run_region;
  run_team_region_ty *run_team_region;

  // Are there images associated with this RTL.
  bool isUsed;

  // Mutex for thread-safety when calling RTL interface functions.
  // It is easier to enforce thread-safety at the libomptarget level,
  // so that developers of new RTLs do not have to worry about it.
  std::mutex Mtx;

  // The existence of the mutex above makes RTLInfoTy non-copyable.
  // We need to provide a copy constructor explicitly.
  RTLInfoTy()
      : Idx(-1), NumberOfDevices(-1), Devices(), LibraryHandler(0),
        is_valid_binary(0), number_of_devices(0), init_device(0),
        load_binary(0), data_alloc(0), data_submit(0), data_retrieve(0),
        data_delete(0), run_region(0), run_team_region(0), isUsed(false),
        Mtx() {}

  RTLInfoTy(const RTLInfoTy &r) : Mtx() {
    Idx = r.Idx;
    NumberOfDevices = r.NumberOfDevices;
    Devices = r.Devices;
    LibraryHandler = r.LibraryHandler;
    is_valid_binary = r.is_valid_binary;
    number_of_devices = r.number_of_devices;
    init_device = r.init_device;
    load_binary = r.load_binary;
    data_alloc = r.data_alloc;
    data_submit = r.data_submit;
    data_retrieve = r.data_retrieve;
    data_delete = r.data_delete;
    run_region = r.run_region;
    run_team_region = r.run_team_region;
    isUsed = r.isUsed;
  }
};

/// Map between Device ID (i.e. openmp device id) and its DeviceTy.
typedef std::vector<DeviceTy> DevicesTy;
static DevicesTy Devices;

/// RTLs identified in the system.
class RTLsTy {
private:
  // Mutex-like object to guarantee thread-safety and unique initialization
  // (i.e. the library attempts to load the RTLs (plugins) only once).
  std::once_flag initFlag;
  void LoadRTLs(); // not thread-safe

public:
  // List of the detected runtime libraries.
  std::list<RTLInfoTy> AllRTLs;

  // Array of pointers to the detected runtime libraries that have compatible
  // binaries.
  std::vector<RTLInfoTy *> UsedRTLs;

  explicit RTLsTy() {}

  // Load all the runtime libraries (plugins) if not done before.
  void LoadRTLsOnce();
};

void RTLsTy::LoadRTLs() {
  // Attempt to open all the plugins and, if they exist, check if the interface
  // is correct and if they are supporting any devices.
  for (auto *Name : RTLNames) {
    void *dynlib_handle = dlopen(Name, RTLD_NOW);

    if (!dynlib_handle) {
      // Library does not exist or cannot be found.
      DP("Unable to load library '%s': %s!\n", Name, dlerror());
      continue;
    }

    DP("Successfully loaded library '%s'!\n", Name);

    // Retrieve the RTL information from the runtime library.
    RTLInfoTy R;

    R.LibraryHandler = dynlib_handle;
    R.isUsed = false;
    if (!(R.is_valid_binary = (RTLInfoTy::is_valid_binary_ty *)dlsym(
              dynlib_handle, "__tgt_rtl_is_valid_binary")))
      continue;
    if (!(R.number_of_devices = (RTLInfoTy::number_of_devices_ty *)dlsym(
              dynlib_handle, "__tgt_rtl_number_of_devices")))
      continue;
    if (!(R.init_device = (RTLInfoTy::init_device_ty *)dlsym(
              dynlib_handle, "__tgt_rtl_init_device")))
      continue;
    if (!(R.load_binary = (RTLInfoTy::load_binary_ty *)dlsym(
              dynlib_handle, "__tgt_rtl_load_binary")))
      continue;
    if (!(R.data_alloc = (RTLInfoTy::data_alloc_ty *)dlsym(
              dynlib_handle, "__tgt_rtl_data_alloc")))
      continue;
    if (!(R.data_submit = (RTLInfoTy::data_submit_ty *)dlsym(
              dynlib_handle, "__tgt_rtl_data_submit")))
      continue;
    if (!(R.data_retrieve = (RTLInfoTy::data_retrieve_ty *)dlsym(
              dynlib_handle, "__tgt_rtl_data_retrieve")))
      continue;
    if (!(R.data_delete = (RTLInfoTy::data_delete_ty *)dlsym(
              dynlib_handle, "__tgt_rtl_data_delete")))
      continue;
    if (!(R.run_region = (RTLInfoTy::run_region_ty *)dlsym(
              dynlib_handle, "__tgt_rtl_run_target_region")))
      continue;
    if (!(R.run_team_region = (RTLInfoTy::run_team_region_ty *)dlsym(
              dynlib_handle, "__tgt_rtl_run_target_team_region")))
      continue;

    // No devices are supported by this RTL?
    if (!(R.NumberOfDevices = R.number_of_devices())) {
      DP("No devices supported in this RTL\n");
      continue;
    }

    DP("Registering RTL %016lx supporting %d devices!\n", (long)dynlib_handle,
       R.NumberOfDevices);

    // The RTL is valid! Will save the information in the RTLs list.
    AllRTLs.push_back(R);
  }
  return;
}

void RTLsTy::LoadRTLsOnce() {
  // RTL.LoadRTLs() is called only once in a thread-safe fashion.
  std::call_once(initFlag, &RTLsTy::LoadRTLs, this);
}

static RTLsTy RTLs;
static std::mutex RTLsMtx;

/// Map between the host entry begin and the translation table. Each
/// registered library gets one TranslationTable. Use the map from
/// __tgt_offload_entry so that we may quickly determine whether we
/// are trying to (re)register an existing lib or really have a new one.
struct TranslationTable {
  __tgt_target_table HostTable;

  // Image assigned to a given device.
  std::vector<__tgt_device_image *> TargetsImages; // One image per device ID.

  // Table of entry points or NULL if it was not already computed.
  std::vector<__tgt_target_table *> TargetsTable; // One table per device ID.
};
typedef std::map<__tgt_offload_entry *, TranslationTable>
    HostEntriesBeginToTransTableTy;
static HostEntriesBeginToTransTableTy HostEntriesBeginToTransTable;
static std::mutex TrlTblMtx;

/// Map between the host ptr and a table index
struct TableMap {
  TranslationTable *Table; // table associated with the host ptr.
  uint32_t Index; // index in which the host ptr translated entry is found.
  TableMap() : Table(0), Index(0) {}
  TableMap(TranslationTable *table, uint32_t index)
      : Table(table), Index(index) {}
};
typedef std::map<void *, TableMap> HostPtrToTableMapTy;
static HostPtrToTableMapTy HostPtrToTableMap;
static std::mutex TblMapMtx;

// Check whether a device has an associated RTL and initialize it if it's not
// already initialized.
static bool device_is_ready(int device_num) {
  // Devices.size() can only change while registering a new
  // library, so try to acquire the lock of RTLs' mutex.
  RTLsMtx.lock();
  size_t Devices_size = Devices.size();
  RTLsMtx.unlock();
  if (Devices_size <= (size_t)device_num) {
    DP("Device ID  %d does not have a matching RTL\n", device_num);
    return false;
  }

  // Get device info
  DeviceTy &Device = Devices[device_num];
  // Init the device if not done before
  if (!Device.IsInit) {
    if (Device.initOnce() != OFFLOAD_SUCCESS) {
      DP("Failed to init device %d\n", device_num);
      return false;
    }
  }

  return true;
}

////////////////////////////////////////////////////////////////////////////////
// getter and setter
//

EXTERN int omp_get_num_devices(void) {
  RTLsMtx.lock();
  size_t Devices_size = Devices.size();
  RTLsMtx.unlock();

  return Devices_size;
}

EXTERN int omp_get_initial_device(void) {
  return HOST_DEVICE;
}

EXTERN void *omp_target_alloc(size_t size, int device_num) {
  DP("Call to omp_target_alloc for device %d requesting %lu bytes\n",
      device_num, (unsigned long) size);

  if (size <= 0) {
    DP("Call to omp_target_alloc with non-positive length\n");
    return NULL;
  }

  void *rc = NULL;

  if (device_num == omp_get_initial_device()) {
    rc = malloc(size);
    DP("omp_target_alloc returns host ptr 0x%016llx\n", (long long) rc);
    return rc;
  }

  if (!device_is_ready(device_num)) {
    DP("omp_target_alloc returns NULL ptr\n");
    return NULL;
  }

  DeviceTy &Device = Devices[device_num];
  rc = Device.RTL->data_alloc(Device.RTLDeviceID, size);
  DP("omp_target_alloc returns device ptr 0x%016llx\n", (long long) rc);
  return rc;
}

EXTERN void omp_target_free(void *device_ptr, int device_num) {
  DP("Call to omp_target_free for device %d and address 0x%016llx\n",
      device_num, (long long) device_ptr);

  if (!device_ptr) {
    DP("Call to omp_target_free with NULL ptr\n");
    return;
  }

  if (device_num == omp_get_initial_device()) {
    free(device_ptr);
    DP("omp_target_free deallocated host ptr\n");
    return;
  }

  DeviceTy &Device = Devices[device_num];
  Device.RTL->data_delete(Device.RTLDeviceID, (void *)device_ptr);
  DP("omp_target_free deallocated device ptr\n");
}

EXTERN int omp_target_is_present(void *ptr, int device_num) {
  DP("Call to omp_target_is_present for device %d and address 0x%016llx\n",
      device_num, (long long) ptr);

  if (!ptr) {
    DP("Call to omp_target_is_present with NULL ptr, returning false\n");
    return false;
  }

  if (device_num == omp_get_initial_device()) {
    DP("Call to omp_target_is_present on host, returning true\n");
    return true;
  }

  DeviceTy& Device = Devices[device_num];
  long IsLast; // not used
  int rc = (Device.getTgtPtrBegin(ptr, 0, IsLast, false) != NULL);
  DP("Call to omp_target_is_present returns %d\n", rc);
  return rc;
}

EXTERN int omp_target_memcpy(void *dst, void *src, size_t length,
    size_t dst_offset, size_t src_offset, int dst_device, int src_device) {
  DP("Call to omp_target_memcpy, dst device %d, src device %d, "
      "dst addr 0x%016llx, src addr 0x%016llx, dst offset %lu, src offset %lu, "
      "length %lu\n", dst_device, src_device, (long long) dst, (long long) src,
      (unsigned long) dst_offset, (unsigned long) src_offset,
      (unsigned long) length);

  if (!dst || !src || length <= 0) {
    DP("Call to omp_target_memcpy with invalid arguments\n");
    return OFFLOAD_FAIL;
  }

  if (src_device != omp_get_initial_device())
    if (!device_is_ready(src_device)) {
      DP("omp_target_memcpy returns OFFLOAD_FAIL\n");
      return OFFLOAD_FAIL;
    }

  if (dst_device != omp_get_initial_device())
    if (!device_is_ready(dst_device)) {
      DP("omp_target_memcpy returns OFFLOAD_FAIL\n");
      return OFFLOAD_FAIL;
    }

  int rc = OFFLOAD_SUCCESS;
  void *srcAddr = (char *)src + src_offset;
  void *dstAddr = (char *)dst + dst_offset;

  if (src_device == omp_get_initial_device() &&
      dst_device == omp_get_initial_device()) {
    DP("copy from host to host\n");
    const void *p = memcpy(dstAddr, srcAddr, length);
    if (p == NULL)
      rc = OFFLOAD_FAIL;
  } else if (src_device == omp_get_initial_device()) {
    DP("copy from host to device\n");
    DeviceTy& DstDev = Devices[dst_device];
    rc = DstDev.data_submit(dstAddr, srcAddr, length);
  } else if (dst_device == omp_get_initial_device()) {
    DP("copy from device to host\n");
    DeviceTy& SrcDev = Devices[src_device];
    rc = SrcDev.data_retrieve(dstAddr, srcAddr, length);
  } else {
    DP("copy from device to device\n");
    void *buffer = malloc(length);
    DeviceTy& SrcDev = Devices[src_device];
    DeviceTy& DstDev = Devices[dst_device];
    rc = SrcDev.data_retrieve(buffer, srcAddr, length);
    if (rc == OFFLOAD_SUCCESS)
      rc = DstDev.data_submit(dstAddr, buffer, length);
  }

  DP("omp_target_memcpy returns %d\n", rc);
  return rc;
}

EXTERN int omp_target_memcpy_rect(void *dst, void *src, size_t element_size,
    int num_dims, const size_t *volume, const size_t *dst_offsets,
    const size_t *src_offsets, const size_t *dst_dimensions,
    const size_t *src_dimensions, int dst_device, int src_device) {
  DP("Call to omp_target_memcpy_rect, dst device %d, src device %d, "
      "dst addr 0x%016llx, src addr 0x%016llx, dst offsets 0x%016llx, "
      "src offsets 0x%016llx, dst dims 0x%016llx, src dims 0x%016llx, "
      "volume 0x%016llx, element size %lu, num_dims %d\n", dst_device,
      src_device, (long long) dst, (long long) src, (long long) dst_offsets,
      (long long) src_offsets, (long long) dst_dimensions,
      (long long) src_dimensions, (long long) volume,
      (unsigned long) element_size, num_dims);

  if (!(dst || src)) {
    DP("Call to omp_target_memcpy_rect returns max supported dimensions %d\n",
        INT_MAX);
    return INT_MAX;
  }

  if (!dst || !src || element_size < 1 || num_dims < 1 || !volume ||
      !dst_offsets || !src_offsets || !dst_dimensions || !src_dimensions) {
    DP("Call to omp_target_memcpy_rect with invalid arguments\n");
    return OFFLOAD_FAIL;
  }

  int rc;
  if (num_dims == 1) {
    rc = omp_target_memcpy(dst, src, element_size * volume[0],
        element_size * dst_offsets[0], element_size * src_offsets[0],
        dst_device, src_device);
  } else {
    size_t dst_slice_size = element_size;
    size_t src_slice_size = element_size;
    for (int i=1; i<num_dims; ++i) {
      dst_slice_size *= dst_dimensions[i];
      src_slice_size *= src_dimensions[i];
    }

    size_t dst_off = dst_offsets[0] * dst_slice_size;
    size_t src_off = src_offsets[0] * src_slice_size;
    for (size_t i=0; i<volume[0]; ++i) {
      rc = omp_target_memcpy_rect((char *) dst + dst_off + dst_slice_size * i,
          (char *) src + src_off + src_slice_size * i, element_size,
          num_dims - 1, volume + 1, dst_offsets + 1, src_offsets + 1,
          dst_dimensions + 1, src_dimensions + 1, dst_device, src_device);

      if (rc) {
        DP("Recursive call to omp_target_memcpy_rect returns unsuccessfully\n");
        return rc;
      }
    }
  }

  DP("omp_target_memcpy_rect returns %d\n", rc);
  return rc;
}

EXTERN int omp_target_associate_ptr(void *host_ptr, void *device_ptr,
    size_t size, size_t device_offset, int device_num) {
  DP("Call to omp_target_associate_ptr with host_ptr 0x%016llx, "
      "device_ptr 0x%016llx, size %lu, device_offset %lu, device_num %d\n",
      (long long) host_ptr, (long long) device_ptr, (unsigned long) size,
      (unsigned long) device_offset, device_num);

  if (!host_ptr || !device_ptr || size <= 0) {
    DP("Call to omp_target_associate_ptr with invalid arguments\n");
    return OFFLOAD_FAIL;
  }

  if (device_num == omp_get_initial_device()) {
    DP("omp_target_associate_ptr: no association possible on the host\n");
    return OFFLOAD_FAIL;
  }

  if (!device_is_ready(device_num)) {
    DP("omp_target_associate_ptr returns OFFLOAD_FAIL\n");
    return OFFLOAD_FAIL;
  }

  DeviceTy& Device = Devices[device_num];
  void *device_addr = (void *)((uint64_t)device_ptr + (uint64_t)device_offset);
  int rc = Device.associatePtr(host_ptr, device_addr, size);
  DP("omp_target_associate_ptr returns %d\n", rc);
  return rc;
}

EXTERN int omp_target_disassociate_ptr(void *host_ptr, int device_num) {
  DP("Call to omp_target_disassociate_ptr with host_ptr 0x%016llx, "
      "device_num %d\n", (long long) host_ptr, device_num);

  if (!host_ptr) {
    DP("Call to omp_target_associate_ptr with invalid host_ptr\n");
    return OFFLOAD_FAIL;
  }

  if (device_num == omp_get_initial_device()) {
    DP("omp_target_disassociate_ptr: no association possible on the host\n");
    return OFFLOAD_FAIL;
  }

  if (!device_is_ready(device_num)) {
    DP("omp_target_disassociate_ptr returns OFFLOAD_FAIL\n");
    return OFFLOAD_FAIL;
  }

  DeviceTy& Device = Devices[device_num];
  int rc = Device.disassociatePtr(host_ptr);
  DP("omp_target_disassociate_ptr returns %d\n", rc);
  return rc;
}

////////////////////////////////////////////////////////////////////////////////
// functionality for device

int DeviceTy::associatePtr(void *HstPtrBegin, void *TgtPtrBegin, long Size) {
  DataMapMtx.lock();

  // Check if entry exists
  for (auto &HT : HostDataToTargetMap) {
    if ((long)HstPtrBegin == HT.HstPtrBegin) {
      // Mapping already exists
      bool isValid = HT.HstPtrBegin == (long) HstPtrBegin &&
                     HT.HstPtrEnd == (long) HstPtrBegin + Size &&
                     HT.TgtPtrBegin == (long) TgtPtrBegin;
      DataMapMtx.unlock();
      if (isValid) {
        DP("Attempt to re-associate the same device ptr+offset with the same "
            "host ptr, nothing to do\n");
        return OFFLOAD_SUCCESS;
      } else {
        DP("Not allowed to re-associate a different device ptr+offset with the "
            "same host ptr\n");
        return OFFLOAD_FAIL;
      }
    }
  }

  // Mapping does not exist, allocate it
  HostDataToTargetTy newEntry;

  // Set up missing fields
  newEntry.HstPtrBase = (long) HstPtrBegin;
  newEntry.HstPtrBegin = (long) HstPtrBegin;
  newEntry.HstPtrEnd = (long) HstPtrBegin + Size;
  newEntry.TgtPtrBegin = (long) TgtPtrBegin;
  // refCount must be infinite
  newEntry.RefCount = INF_REF_CNT;

  HostDataToTargetMap.push_front(newEntry);

  DataMapMtx.unlock();

  return OFFLOAD_SUCCESS;
}

int DeviceTy::disassociatePtr(void *HstPtrBegin) {
  DataMapMtx.lock();

  // Check if entry exists
  for (HostDataToTargetListTy::iterator ii = HostDataToTargetMap.begin();
      ii != HostDataToTargetMap.end(); ++ii) {
    if ((long)HstPtrBegin == ii->HstPtrBegin) {
      // Mapping exists
      if (ii->RefCount > INF_REF_CNT>>1) {
        DP("Association found, removing it\n");
        HostDataToTargetMap.erase(ii);
        DataMapMtx.unlock();
        return OFFLOAD_SUCCESS;
      } else {
        DP("Trying to disassociate a pointer which was not mapped via "
            "omp_target_associate_ptr\n");
        break;
      }
    }
  }

  // Mapping not found
  DataMapMtx.unlock();
  DP("Association not found\n");
  return OFFLOAD_FAIL;
}

// return the target pointer begin (where the data will be moved).
// lock-free version called from within assertions
void *DeviceTy::getTgtPtrBegin(void *HstPtrBegin, long Size) {
  long hp = (long)HstPtrBegin;
  for (auto &HT : HostDataToTargetMap) {
    if (hp >= HT.HstPtrBegin && hp < HT.HstPtrEnd) {
      long tp = HT.TgtPtrBegin + (hp - HT.HstPtrBegin);
      return (void *)tp;
    }
  }
  return NULL;
}

// return the target pointer begin (where the data will be moved).
void *DeviceTy::getTgtPtrBegin(void *HstPtrBegin, long Size, long &IsLast,
                               long UpdateRefCount) {
  long hp = (long)HstPtrBegin;
  IsLast = false;

  DataMapMtx.lock();
  for (auto &HT : HostDataToTargetMap) {
    bool isContained = hp >= HT.HstPtrBegin && hp < HT.HstPtrEnd;
    bool extendsBefore = hp < HT.HstPtrBegin && (hp + Size) > HT.HstPtrBegin;
    bool extendsAfter = isContained && (hp + Size) > HT.HstPtrEnd;
    if (extendsBefore) {
      DP("WARNING: Pointer is not mapped but section extends into already "
          "mapped region\n");
    }
    if (extendsAfter) {
      DP("WARNING: Pointer is already mapped but section extends beyond mapped "
          "region \n");
    }
    if (isContained || extendsBefore || extendsAfter) {
      IsLast = !(HT.RefCount > 1);

      if (HT.RefCount > 1 && UpdateRefCount)
        --HT.RefCount;

      long tp = HT.TgtPtrBegin + (hp - HT.HstPtrBegin);
      DataMapMtx.unlock();
      return (void *)tp;
    }
  }
  DataMapMtx.unlock();

  return NULL;
}

// return the target pointer begin (where the data will be moved).
void *DeviceTy::getOrAllocTgtPtr(void *HstPtrBegin, void *HstPtrBase, long Size,
                                 long &IsNew, long UpdateRefCount) {
  long hp = (long)HstPtrBegin;
  IsNew = false;

  // Check if the pointer is contained.
  DataMapMtx.lock();
  for (auto &HT : HostDataToTargetMap) {
    // Is it contained?
    bool isContained = hp >= HT.HstPtrBegin && hp < HT.HstPtrEnd;
    // Does it extend into an already mapped region?
    bool extendsBefore = hp < HT.HstPtrBegin && (hp + Size) > HT.HstPtrBegin;
    // Does it extend beyond the mapped region?
    bool extendsAfter = isContained && (hp + Size) > HT.HstPtrEnd;
    if (extendsBefore) {
      DP("WARNING: Pointer is not mapped but section extends into already "
          "mapped data\n");
    }
    if (extendsAfter) {
      DP("WARNING: Pointer is already mapped but section extends beyond mapped "
          "region\n");
    }
    if (isContained || extendsBefore || extendsAfter) {
      if (UpdateRefCount)
        ++HT.RefCount;
      long tp = HT.TgtPtrBegin + (hp - HT.HstPtrBegin);
      DataMapMtx.unlock();
      return (void *)tp;
    }
  }

  // If it is not contained we should create a new entry for it.
  IsNew = true;
  long tp = (long)RTL->data_alloc(RTLDeviceID, Size);
  HostDataToTargetMap.push_front(
      HostDataToTargetTy((long)HstPtrBase, hp, hp + Size, tp));
  DataMapMtx.unlock();
  return (void *)tp;
}

void DeviceTy::deallocTgtPtr(void *HstPtrBegin, long Size, long ForceDelete) {
  long hp = (long)HstPtrBegin;

  // Check if the pointer is contained in any sub-nodes.
  DataMapMtx.lock();
  for (auto ii = HostDataToTargetMap.begin(), ie = HostDataToTargetMap.end();
       ii != ie; ++ii) {
    auto &HT = *ii;
    // Is it contained?
    if (hp >= HT.HstPtrBegin && hp < HT.HstPtrEnd) {
      if ((hp + Size) > HT.HstPtrEnd) {
        DP("WARNING: Array contains pointer but does not contain the complete "
           "section\n");
      }
      if (ForceDelete)
        HT.RefCount = 1;
      if (--HT.RefCount <= 0) {
        assert(HT.RefCount == 0 && "did not expect a negative ref count");
        DP("Deleting tgt data 0x%016llx of size %lld\n",
           (long long)HT.TgtPtrBegin, (long long)Size);
        RTL->data_delete(RTLDeviceID, (void *)HT.TgtPtrBegin);
        HostDataToTargetMap.erase(ii);
      }
      DataMapMtx.unlock();
      return;
    }
  }
  DataMapMtx.unlock();
  DP("Section to delete (hst addr 0x%llx) does not exist in the allocated "
     "memory\n",
     (unsigned long long)hp);
}

HostDataToTargetTy *DeviceTy::getMapEntry(void *HstPtrBegin) {
  long hp = (long)HstPtrBegin;
  for (auto &HT : HostDataToTargetMap) {
    if (hp >= HT.HstPtrBegin && hp < HT.HstPtrEnd) {
      return &HT;
    }
  }
  return NULL;
}

// init device.
void DeviceTy::init() {
  int32_t rc = RTL->init_device(RTLDeviceID);
  if (rc == OFFLOAD_SUCCESS) {
    IsInit = true;
  }
}

// thread-safe method to initialize the device only once.
int32_t DeviceTy::initOnce() {
  std::call_once(InitFlag, &DeviceTy::init, this);

  // At this point, if IsInit is true, then either this thread or some other
  // thread in the past successfully initialized the device, so we can return
  // OFFLOAD_SUCCESS. If this thread executed init() via call_once() and it
  // failed, return OFFLOAD_FAIL. If call_once did not invoke init(), it means
  // that some other thread already attempted to execute init() and if IsInit
  // is still false, return OFFLOAD_FAIL.
  if (IsInit)
    return OFFLOAD_SUCCESS;
  else
    return OFFLOAD_FAIL;
}

// load binary to device.
__tgt_target_table *DeviceTy::load_binary(void *Img) {
  RTL->Mtx.lock();
  __tgt_target_table *rc = RTL->load_binary(RTLDeviceID, Img);
  RTL->Mtx.unlock();
  return rc;
}

// submit data to device.
int32_t DeviceTy::data_submit(void *TgtPtrBegin, void *HstPtrBegin,
                              int64_t Size) {
  return RTL->data_submit(RTLDeviceID, TgtPtrBegin, HstPtrBegin, Size);
}

// retrieve data from device.
int32_t DeviceTy::data_retrieve(void *HstPtrBegin, void *TgtPtrBegin,
                                int64_t Size) {
  return RTL->data_retrieve(RTLDeviceID, HstPtrBegin, TgtPtrBegin, Size);
}

// run region on device
int32_t DeviceTy::run_region(void *TgtEntryPtr, void **TgtVarsPtr,
                             int32_t TgtVarsSize) {
  return RTL->run_region(RTLDeviceID, TgtEntryPtr, TgtVarsPtr, TgtVarsSize);
}

// run team region on device.
int32_t DeviceTy::run_team_region(void *TgtEntryPtr, void **TgtVarsPtr,
                                  int32_t TgtVarsSize, int32_t NumTeams,
                                  int32_t ThreadLimit, uint64_t LoopTripCount) {
  return RTL->run_team_region(RTLDeviceID, TgtEntryPtr, TgtVarsPtr, TgtVarsSize,
                              NumTeams, ThreadLimit, LoopTripCount);
}

////////////////////////////////////////////////////////////////////////////////
// functionality for registering libs

static void RegisterImageIntoTranslationTable(TranslationTable &TT,
                                              RTLInfoTy &RTL,
                                              __tgt_device_image *image) {

  // same size, as when we increase one, we also increase the other.
  assert(TT.TargetsTable.size() == TT.TargetsImages.size() &&
         "We should have as many images as we have tables!");

  // Resize the Targets Table and Images to accommodate the new targets if
  // required
  unsigned TargetsTableMinimumSize = RTL.Idx + RTL.NumberOfDevices;

  if (TT.TargetsTable.size() < TargetsTableMinimumSize) {
    TT.TargetsImages.resize(TargetsTableMinimumSize, 0);
    TT.TargetsTable.resize(TargetsTableMinimumSize, 0);
  }

  // Register the image in all devices for this target type.
  for (int32_t i = 0; i < RTL.NumberOfDevices; ++i) {
    // If we are changing the image we are also invalidating the target table.
    if (TT.TargetsImages[RTL.Idx + i] != image) {
      TT.TargetsImages[RTL.Idx + i] = image;
      TT.TargetsTable[RTL.Idx + i] = 0; // lazy initialization of target table.
    }
  }
}

////////////////////////////////////////////////////////////////////////////////
/// adds a target shared library to the target execution image
EXTERN void __tgt_register_lib(__tgt_bin_desc *desc) {

  // Attempt to load all the plugins available in the system.
  RTLs.LoadRTLsOnce();

  RTLsMtx.lock();
  // Register the images with the RTLs that understand them, if any.
  for (int32_t i = 0; i < desc->NumDevices; ++i) {
    // Obtain the image.
    __tgt_device_image *img = &desc->DeviceImages[i];

    RTLInfoTy *FoundRTL = NULL;

    // Scan the RTLs that have associated images until we find one that supports
    // the current image.
    for (auto &R : RTLs.AllRTLs) {
      if (!R.is_valid_binary(img)) {
        DP("Image %016lx is NOT compatible with RTL %016lx!\n",
           (long)img->ImageStart, (long)R.LibraryHandler);
        continue;
      }

      DP("Image %016lx is compatible with RTL %016lx!\n", (long)img->ImageStart,
         (long)R.LibraryHandler);

      // If this RTL is not already in use, initialize it.
      if (!R.isUsed) {
        // Initialize the device information for the RTL we are about to use.
        DeviceTy device(&R);

        size_t start = Devices.size();
        Devices.resize(start + R.NumberOfDevices, device);
        for (int32_t device_id = 0; device_id < R.NumberOfDevices;
            device_id++) {
          // global device ID
          Devices[start + device_id].DeviceID = start + device_id;
          // RTL local device ID
          Devices[start + device_id].RTLDeviceID = device_id;

          // Save pointer to device in RTL in case we want to unregister the RTL
          R.Devices.push_back(&Devices[start + device_id]);
        }

        // Initialize the index of this RTL and save it in the used RTLs.
        R.Idx = (RTLs.UsedRTLs.empty())
                    ? 0
                    : RTLs.UsedRTLs.back()->Idx +
                          RTLs.UsedRTLs.back()->NumberOfDevices;
        assert((size_t) R.Idx == start &&
            "RTL index should equal the number of devices used so far.");
        R.isUsed = true;
        RTLs.UsedRTLs.push_back(&R);

        DP("RTL %016lx has index %d!\n", (long)R.LibraryHandler, R.Idx);
      }

      // Initialize (if necessary) translation table for this library.
      TrlTblMtx.lock();
      if(!HostEntriesBeginToTransTable.count(desc->EntriesBegin)){
        TranslationTable &tt =
            HostEntriesBeginToTransTable[desc->EntriesBegin];
        tt.HostTable.EntriesBegin = desc->EntriesBegin;
        tt.HostTable.EntriesEnd = desc->EntriesEnd;
      }

      // Retrieve translation table for this library.
      TranslationTable &TransTable =
          HostEntriesBeginToTransTable[desc->EntriesBegin];

      DP("Registering image %016lx with RTL %016lx!\n", (long)img->ImageStart,
         (long)R.LibraryHandler);
      RegisterImageIntoTranslationTable(TransTable, R, img);
      TrlTblMtx.unlock();
      FoundRTL = &R;
      break;
    }

    // if an RTL was found we are done - proceed to register the next image
    if (!FoundRTL)
      DP("No RTL found for image %016lx!\n", (long)img->ImageStart);

    // Load ctors/dtors for static objects
    for (int32_t i = 0; i < FoundRTL->NumberOfDevices; ++i) {
      DeviceTy &Device = Devices[i];
      Device.PendingGlobalsMtx.lock();
      Device.HasPendingGlobals = true;
      for (__tgt_offload_entry *entry = img->EntriesBegin;
          entry != img->EntriesEnd; ++entry) {
        if (entry->flags & OMP_DECLARE_TARGET_CTOR) {
          DP("Adding ctor %016lx to the pending list.\n", (long) entry->addr);
          Device.PendingCtorsDtors[desc].PendingCtors.push_back(entry->addr);
        } else if (entry->flags & OMP_DECLARE_TARGET_DTOR) {
          // Dtors are pushed in reverse order so they are executed from end
          // to beginning when unregistering the library!
          DP("Adding dtor %016lx to the pending list.\n", (long) entry->addr);
          Device.PendingCtorsDtors[desc].PendingDtors.push_front(entry->addr);
        }

        if (entry->flags & OMP_DECLARE_TARGET_LINK) {
          DP("The \"link\" attribute is not yet supported!\n");
        }
      }
      Device.PendingGlobalsMtx.unlock();
    }
  }
  RTLsMtx.unlock();


  DP("Done registering entries!\n");
}

////////////////////////////////////////////////////////////////////////////////
/// unloads a target shared library
EXTERN void __tgt_unregister_lib(__tgt_bin_desc *desc) {
  DP("Unloading target library!\n");

  RTLsMtx.lock();
  // Find which RTL understands each image, if any.
  for (int32_t i = 0; i < desc->NumDevices; ++i) {
    // Obtain the image.
    __tgt_device_image *img = &desc->DeviceImages[i];

    RTLInfoTy *FoundRTL = NULL;

    // Scan the RTLs that have associated images until we find one that supports
    // the current image. We only need to scan RTLs that are already being used.
    for (auto *R : RTLs.UsedRTLs) {

      assert(R->isUsed && "Expecting used RTLs.");

      if (!R->is_valid_binary(img)) {
        DP("Image %016lx is NOT compatible with RTL %016lx!\n",
           (long)img->ImageStart, (long)R->LibraryHandler);
        continue;
      }

      DP("Image %016lx is compatible with RTL %016lx!\n", (long)img->ImageStart,
         (long)R->LibraryHandler);

      FoundRTL = R;

      // Execute dtors for static objects if the device has been used, i.e.
      // if its PendingCtors list has been emptied.
      for (int32_t i = 0; i < FoundRTL->NumberOfDevices; ++i) {
        DeviceTy &Device = Devices[i];
        Device.PendingGlobalsMtx.lock();
        if (Device.PendingCtorsDtors[desc].PendingCtors.empty()) {
          for (auto &dtor : Device.PendingCtorsDtors[desc].PendingDtors) {
            int rc = target(Device.DeviceID, dtor, 0, NULL, NULL, NULL, NULL, 1,
                1, true /*team*/);
            if (rc != OFFLOAD_SUCCESS) {
              DP("Running destructor %016lx failed.\n", (long) dtor);
            }
          }
          // Remove this library's entry from PendingCtorsDtors
          Device.PendingCtorsDtors.erase(desc);
        }
        Device.PendingGlobalsMtx.unlock();
      }

      // Remove translation table for this image.
      TrlTblMtx.lock();
      auto tt = HostEntriesBeginToTransTable.find(desc->EntriesBegin);
      if (tt != HostEntriesBeginToTransTable.end()) {
        HostEntriesBeginToTransTable.erase(tt);
        DP("Unregistering image %016lx from RTL %016lx!\n",
            (long)img->ImageStart, (long)R->LibraryHandler);
      } else {
        DP("Translation table for image %016lx cannot be found, probably it "
            "has been already removed.\n", (long)img->ImageStart);
      }

      TrlTblMtx.unlock();
      break;
    }

    // if an RTL was not found proceed to unregister the next image
    if (!FoundRTL){
      DP("No RTLs in use support the image %016lx!\n", (long)img->ImageStart);
      continue;
    }
  }
  RTLsMtx.unlock();
  DP("Done unregistering images!\n");

  // Remove entries from HostPtrToTableMap
  // TODO: This seems quite time consuming...
  TblMapMtx.lock();
  for (__tgt_offload_entry *cur = desc->EntriesBegin;
      cur < desc->EntriesEnd; ++cur) {
    HostPtrToTableMap.erase(cur->addr);
  }
  TblMapMtx.unlock();

  // TODO: Remove RTL and the devices it manages if it's not used by any library anymore?
  // TODO: Write some RTL->unload_image(...) function.

  DP("Done unregistering library!\n");
}

/// Map global data and execute pending ctors
static int InitLibrary(DeviceTy& Device) {
  /*
   * Map global data
   */
  int32_t device_id = Device.DeviceID;
  int rc = OFFLOAD_SUCCESS;

  Device.PendingGlobalsMtx.lock();
  TrlTblMtx.lock();
  for (HostEntriesBeginToTransTableTy::iterator
      ii = HostEntriesBeginToTransTable.begin();
      ii != HostEntriesBeginToTransTable.end(); ++ii) {
    TranslationTable *TransTable = &ii->second;
    if (TransTable->TargetsTable[device_id] != 0) {
      // Library entries have already been processed
      continue;
    }

    // 1) get image.
    assert(TransTable->TargetsImages.size() > (size_t)device_id &&
           "Not expecting a device ID outside the table's bounds!");
    __tgt_device_image *img = TransTable->TargetsImages[device_id];
    if (!img) {
      DP("No image loaded for device id %d.\n", device_id);
      rc = OFFLOAD_FAIL;
      break;
    }
    // 2) load image into the target table.
    __tgt_target_table *TargetTable =
        TransTable->TargetsTable[device_id] = Device.load_binary(img);
    // Unable to get table for this image: invalidate image and fail.
    if (!TargetTable) {
      DP("Unable to generate entries table for device id %d.\n", device_id);
      TransTable->TargetsImages[device_id] = 0;
      rc = OFFLOAD_FAIL;
      break;
    }

    // Verify whether the two table sizes match.
    size_t hsize =
        TransTable->HostTable.EntriesEnd - TransTable->HostTable.EntriesBegin;
    size_t tsize = TargetTable->EntriesEnd - TargetTable->EntriesBegin;

    // Invalid image for these host entries!
    if (hsize != tsize) {
      DP("Host and Target tables mismatch for device id %d [%lx != %lx].\n",
         device_id, hsize, tsize);
      TransTable->TargetsImages[device_id] = 0;
      TransTable->TargetsTable[device_id] = 0;
      rc = OFFLOAD_FAIL;
      break;
    }

    // process global data that needs to be mapped.
    Device.DataMapMtx.lock();
    __tgt_target_table *HostTable = &TransTable->HostTable;
    for (__tgt_offload_entry *CurrDeviceEntry = TargetTable->EntriesBegin,
                             *CurrHostEntry = HostTable->EntriesBegin,
                             *EntryDeviceEnd = TargetTable->EntriesEnd;
         CurrDeviceEntry != EntryDeviceEnd;
         CurrDeviceEntry++, CurrHostEntry++) {
      if (CurrDeviceEntry->size != 0) {
        // has data.
        assert(CurrDeviceEntry->size == CurrHostEntry->size &&
               "data size mismatch");
        assert(Device.getTgtPtrBegin(CurrHostEntry->addr,
                                     CurrHostEntry->size) == NULL &&
               "data in declared target should not be already mapped");
        // add entry to map.
        DP("add mapping from host 0x%llx to 0x%llx with size %lld\n\n",
           (unsigned long long)CurrHostEntry->addr,
           (unsigned long long)CurrDeviceEntry->addr,
           (unsigned long long)CurrDeviceEntry->size);
        Device.HostDataToTargetMap.push_front(HostDataToTargetTy(
            (long)CurrHostEntry->addr, (long)CurrHostEntry->addr,
            (long)CurrHostEntry->addr + CurrHostEntry->size,
            (long)CurrDeviceEntry->addr));
      }
    }
    Device.DataMapMtx.unlock();
  }
  TrlTblMtx.unlock();

  if (rc != OFFLOAD_SUCCESS) {
    Device.PendingGlobalsMtx.unlock();
    return rc;
  }

  /*
   * Run ctors for static objects
   */
  if (!Device.PendingCtorsDtors.empty()) {
    // Call all ctors for all libraries registered so far
    for (auto &lib : Device.PendingCtorsDtors) {
      if (!lib.second.PendingCtors.empty()) {
        DP("Has pending ctors... call now\n");
        for (auto &entry : lib.second.PendingCtors) {
          void *ctor = entry;
          int rc = target(device_id, ctor, 0, NULL, NULL, NULL,
                          NULL, 1, 1, true /*team*/);
          if (rc != OFFLOAD_SUCCESS) {
            DP("Running ctor %016lx failed.\n", (long) ctor);
            Device.PendingGlobalsMtx.unlock();
            return OFFLOAD_FAIL;
          }
        }
        // Clear the list to indicate that this device has been used
        lib.second.PendingCtors.clear();
        DP("Done with pending ctors for lib %016lx\n", (long) lib.first);
      }
    }
  }
  Device.HasPendingGlobals = false;
  Device.PendingGlobalsMtx.unlock();

  return OFFLOAD_SUCCESS;
}

static int CheckDevice(int32_t device_id) {
  // Get device info.
  DeviceTy &Device = Devices[device_id];

  // No devices available?
  // Devices.size() can only change while registering a new
  // library, so try to acquire the lock of RTLs' mutex.
  RTLsMtx.lock();
  size_t Devices_size = Devices.size();
  RTLsMtx.unlock();
  if (!(device_id >= 0 && (size_t)device_id < Devices_size)) {
    DP("Device ID %d does not have a matching RTL.\n", device_id);
    return OFFLOAD_FAIL;
  }

  DP("Is the device %d (local ID %d) initialized? %d\n", device_id,
     Device.RTLDeviceID, (int)Device.IsInit);

  // Init the device if not done before.
  if (!Device.IsInit) {
    if (Device.initOnce() != OFFLOAD_SUCCESS) {
      DP("Failed to init device %d\n", device_id);
      return OFFLOAD_FAIL;
    }
  }

  // Check whether global data has been mapped for this device
  Device.PendingGlobalsMtx.lock();
  bool hasPendingGlobals = Device.HasPendingGlobals;
  Device.PendingGlobalsMtx.unlock();
  if (hasPendingGlobals) {
    if (InitLibrary(Device) != OFFLOAD_SUCCESS) {
      DP("failed to init globals on device %d\n", device_id);
      return OFFLOAD_FAIL;
    }
  }

  return OFFLOAD_SUCCESS;
}

// Old map types
enum tgt_oldmap_type {
  OMP_TGT_OLDMAPTYPE_TO          = 0x001, // copy data from host to device
  OMP_TGT_OLDMAPTYPE_FROM        = 0x002, // copy data from device to host
  OMP_TGT_OLDMAPTYPE_ALWAYS      = 0x004, // copy regardless of the reference count
  OMP_TGT_OLDMAPTYPE_DELETE      = 0x008, // force unmapping of data
  OMP_TGT_OLDMAPTYPE_MAP_PTR     = 0x010, // map the pointer as well as the pointee
  OMP_TGT_OLDMAPTYPE_FIRST_MAP   = 0x020, // first occurrence of mapped variable
  OMP_TGT_OLDMAPTYPE_RETURN_PTR  = 0x040, // return base device addr of mapped data
  OMP_TGT_OLDMAPTYPE_PRIVATE_PTR = 0x080, // private variable - not mapped
  OMP_TGT_OLDMAPTYPE_PRIVATE_VAL = 0x100  // copy by value - not mapped
};

// Temporary functions for map translation and cleanup
struct combined_entry_t {
  int num_members; // number of members in combined entry
  void *base_addr; // base address of combined entry
  void *begin_addr; // begin address of combined entry
  void *end_addr; // size of combined entry
};

static void translate_map(int32_t arg_num, void **args_base, void **args,
    int64_t *arg_sizes, int32_t *arg_types, int32_t &new_arg_num,
    void **&new_args_base, void **&new_args, int64_t *&new_arg_sizes,
    int64_t *&new_arg_types, long is_target_construct) {
  if (arg_num <= 0) {
    DP("Nothing to translate\n");
    new_arg_num = 0;
    return;
  }

  // array of combined entries
  combined_entry_t *cmb_entries =
      (combined_entry_t *) alloca(arg_num * sizeof(combined_entry_t));
  // number of combined entries
  long num_combined = 0;
  // old entry is MAP_PTR?
  bool *is_ptr_old = (bool *) alloca(arg_num * sizeof(bool));
  // old entry is member of member_of[old] cmb_entry
  int *member_of = (int *) alloca(arg_num * sizeof(int));

  DP("Translating %d map entries\n", arg_num);
  for (int i = 0; i < arg_num; ++i) {
    member_of[i] = -1;
    is_ptr_old[i] = false;
    // Scan previous entries to see whether this entry shares the same base
    for (int j = 0; j < i; ++j) {
      void *new_begin_addr = NULL;
      void *new_end_addr = NULL;

      if (arg_types[i] & OMP_TGT_OLDMAPTYPE_MAP_PTR) {
        if (args_base[i] == args[j]) {
          if (!(arg_types[j] & OMP_TGT_OLDMAPTYPE_MAP_PTR)) {
            DP("Entry %d has the same base as entry %d's begin address\n", i,
                j);
            new_begin_addr = args_base[i];
            new_end_addr = (char *)args_base[i] + sizeof(void *);
            assert(arg_sizes[j] == sizeof(void *));
            is_ptr_old[j] = true;
          } else {
            DP("Entry %d has the same base as entry %d's begin address, but "
                "%d's base was a MAP_PTR too\n", i, j, j);
          }
        }
      } else {
        if (!(arg_types[i] & OMP_TGT_OLDMAPTYPE_FIRST_MAP) &&
            args_base[i] == args_base[j]) {
          DP("Entry %d has the same base address as entry %d\n", i, j);
          new_begin_addr = args[i];
          new_end_addr = (char *)args[i] + arg_sizes[i];
        }
      }

      // If we have combined the entry with a previous one
      if (new_begin_addr) {
        int id;
        if(member_of[j] == -1) {
          // We have a new entry
          id = num_combined++;
          DP("Creating new combined entry %d for old entry %d\n", id, j);
          // Initialize new entry
          cmb_entries[id].num_members = 1;
          cmb_entries[id].base_addr = args_base[j];
          if (arg_types[j] & OMP_TGT_OLDMAPTYPE_MAP_PTR) {
            cmb_entries[id].begin_addr = args_base[j];
            cmb_entries[id].end_addr = (char *)args_base[j] + arg_sizes[j];
          } else {
            cmb_entries[id].begin_addr = args[j];
            cmb_entries[id].end_addr = (char *)args[j] + arg_sizes[j];
          }
          member_of[j] = id;
        } else {
          // Reuse existing combined entry
          DP("Reusing existing combined entry %d\n", member_of[j]);
          id = member_of[j];
        }

        // Update combined entry
        DP("Adding entry %d to combined entry %d\n", i, id);
        cmb_entries[id].num_members++;
        // base_addr stays the same
        cmb_entries[id].begin_addr =
            std::min(cmb_entries[id].begin_addr, new_begin_addr);
        cmb_entries[id].end_addr =
            std::max(cmb_entries[id].end_addr, new_end_addr);
        member_of[i] = id;
        break;
      }
    }
  }

  DP("New entries: %ld combined + %d original\n", num_combined, arg_num);
  new_arg_num = arg_num + num_combined;
  new_args_base = (void **) malloc(new_arg_num * sizeof(void *));
  new_args = (void **) malloc(new_arg_num * sizeof(void *));
  new_arg_sizes = (int64_t *) malloc(new_arg_num * sizeof(int64_t));
  new_arg_types = (int64_t *) malloc(new_arg_num * sizeof(int64_t));

  const int64_t alignment = 8;

  int next_id = 0; // next ID
  int next_cid = 0; // next combined ID
  int *combined_to_new_id = (int *) alloca(num_combined * sizeof(int));
  for (int i = 0; i < arg_num; ++i) {
    // It is member_of
    if (member_of[i] == next_cid) {
      int cid = next_cid++; // ID of this combined entry
      int nid = next_id++; // ID of the new (global) entry
      combined_to_new_id[cid] = nid;
      DP("Combined entry %3d will become new entry %3d\n", cid, nid);

      int64_t padding = (int64_t)cmb_entries[cid].begin_addr % alignment;
      if (padding) {
        DP("Using a padding of %ld for begin address 0x%016llx\n", padding,
            (long long) cmb_entries[cid].begin_addr);
        cmb_entries[cid].begin_addr =
            (char *)cmb_entries[cid].begin_addr - padding;
      }

      new_args_base[nid] = cmb_entries[cid].base_addr;
      new_args[nid] = cmb_entries[cid].begin_addr;
      new_arg_sizes[nid] = (int64_t) ((char *)cmb_entries[cid].end_addr -
          (char *)cmb_entries[cid].begin_addr);
      new_arg_types[nid] = OMP_TGT_MAPTYPE_TARGET_PARAM;
      DP("Entry %3d: base_addr 0x%016llx, begin_addr 0x%016llx, "
          "size %lu, type 0x%llx\n", nid, (long long) new_args_base[nid],
          (long long) new_args[nid], (unsigned long) new_arg_sizes[nid],
          (long long) new_arg_types[nid]);
    } else if (member_of[i] != -1) {
      DP("Combined entry %3d has been encountered before, do nothing\n",
          member_of[i]);
    }

    // Now that the combined entry (the one the old entry was a member of) has
    // been inserted into the new arguments list, proceed with the old entry.
    int nid = next_id++;
    DP("Old entry %3d will become new entry %3d\n", i, nid);

    new_args_base[nid] = args_base[i];
    new_args[nid] = args[i];
    new_arg_sizes[nid] = arg_sizes[i];
    int64_t old_type = arg_types[i];

    if (is_ptr_old[i]) {
      // Reset TO and FROM flags
      old_type &= ~(OMP_TGT_OLDMAPTYPE_TO | OMP_TGT_OLDMAPTYPE_FROM);
    }

    if (member_of[i] == -1) {
      if (!is_target_construct)
        old_type &= ~OMP_TGT_MAPTYPE_TARGET_PARAM;
      new_arg_types[nid] = old_type;
      DP("Entry %3d: base_addr 0x%016llx, begin_addr 0x%016llx, size %lu, "
        "type 0x%llx (old entry %d not MEMBER_OF)\n", nid,
        (long long) new_args_base[nid], (long long) new_args[nid],
        (unsigned long) new_arg_sizes[nid], (long long) new_arg_types[nid], i);
    } else {
      // Old entry is not FIRST_MAP
      old_type &= ~OMP_TGT_OLDMAPTYPE_FIRST_MAP;
      // Add MEMBER_OF
      int new_member_of = combined_to_new_id[member_of[i]];
      old_type |= ((int64_t)new_member_of + 1) << 48;
      new_arg_types[nid] = old_type;
      DP("Entry %3d: base_addr 0x%016llx, begin_addr 0x%016llx, size %lu, "
        "type 0x%llx (old entry %d MEMBER_OF %d)\n", nid,
        (long long) new_args_base[nid], (long long) new_args[nid],
        (unsigned long) new_arg_sizes[nid], (long long) new_arg_types[nid], i,
        new_member_of);
    }
  }
}

static void cleanup_map(int32_t new_arg_num, void **new_args_base,
    void **new_args, int64_t *new_arg_sizes, int64_t *new_arg_types,
    int32_t arg_num, void **args_base) {
  if (new_arg_num > 0) {
    int offset = new_arg_num - arg_num;
    for (int32_t i = 0; i < arg_num; ++i) {
      // Restore old base address
      args_base[i] = new_args_base[i+offset];
    }
    free(new_args_base);
    free(new_args);
    free(new_arg_sizes);
    free(new_arg_types);
  }
}

static short member_of(int64_t type) {
  return ((type & OMP_TGT_MAPTYPE_MEMBER_OF) >> 48) - 1;
}

/// Internal function to do the mapping and transfer the data to the device
static void target_data_begin(DeviceTy &Device, int32_t arg_num,
                              void **args_base, void **args, int64_t *arg_sizes,
                              int64_t *arg_types) {
  // process each input.
  for (int32_t i = 0; i < arg_num; ++i) {
    // Ignore private variables and arrays - there is no mapping for them.
    if ((arg_types[i] & OMP_TGT_MAPTYPE_LITERAL) ||
        (arg_types[i] & OMP_TGT_MAPTYPE_PRIVATE))
      continue;

    void *HstPtrBegin = args[i];
    void *HstPtrBase = args_base[i];
    // Address of pointer on the host and device, respectively.
    void *Pointer_HstPtrBegin, *Pointer_TgtPtrBegin;
    long IsNew, Pointer_IsNew;
    long UpdateRef = !(arg_types[i] & OMP_TGT_MAPTYPE_MEMBER_OF);
    if (arg_types[i] & OMP_TGT_MAPTYPE_PTR_AND_OBJ) {
      DP("has a pointer entry: \n");
      // base is address of pointer.
      Pointer_TgtPtrBegin = Device.getOrAllocTgtPtr(HstPtrBase, HstPtrBase,
          sizeof(void *), Pointer_IsNew, UpdateRef);
      DP("There are %ld bytes allocated at target address %016lx - is new %ld"
          "\n", (long)sizeof(void *), (long)Pointer_TgtPtrBegin, Pointer_IsNew);
      assert(Pointer_TgtPtrBegin &&
             "Data allocation by RTL returned invalid ptr");
      Pointer_HstPtrBegin = HstPtrBase;
      // modify current entry.
      HstPtrBase = *(void **)HstPtrBase;
      UpdateRef = true; // subsequently update ref count of pointee
    }

    void *TgtPtrBegin = Device.getOrAllocTgtPtr(HstPtrBegin, HstPtrBase,
        arg_sizes[i], IsNew, UpdateRef);
    DP("There are %ld bytes allocated at target address %016lx - is new %ld"
        "\n", (long)arg_sizes[i], (long)TgtPtrBegin, IsNew);
    assert((TgtPtrBegin || !arg_sizes[i]) &&
        "Data allocation by RTL returned invalid ptr");

    if (arg_types[i] & OMP_TGT_MAPTYPE_RETURN_PARAM) {
      void *ret_ptr;
      if (arg_types[i] & OMP_TGT_MAPTYPE_PTR_AND_OBJ)
        ret_ptr = Pointer_TgtPtrBegin;
      else {
        long IsLast; // not used
        ret_ptr = Device.getTgtPtrBegin(HstPtrBegin, 0, IsLast, false);
      }

      DP("Returning device pointer %016lx\n", (long)ret_ptr);
      args_base[i] = ret_ptr;
    }

    if (arg_types[i] & OMP_TGT_MAPTYPE_TO) {
      bool copy = false;
      if (IsNew || (arg_types[i] & OMP_TGT_MAPTYPE_ALWAYS)) {
        copy = true;
      } else if (arg_types[i] & OMP_TGT_MAPTYPE_MEMBER_OF) {
        // Copy data only if the "parent" struct has RefCount==1.
        short parent_idx = member_of(arg_types[i]);
        HostDataToTargetTy *entry = Device.getMapEntry(args[parent_idx]);
        if (entry->RefCount == 1) {
          copy = true;
        }
      }

      if (copy) {
        DP("Moving %ld bytes (hst:%016lx) -> (tgt:%016lx)\n", (long)arg_sizes[i],
           (long)HstPtrBegin, (long)TgtPtrBegin);
        Device.data_submit(TgtPtrBegin, HstPtrBegin, arg_sizes[i]);
      }
    }

    if (arg_types[i] & OMP_TGT_MAPTYPE_PTR_AND_OBJ) {
      DP("Update pointer (%016lx) -> [%016lx]\n", (long)Pointer_TgtPtrBegin,
         (long)TgtPtrBegin);
      uint64_t Delta = (uint64_t)HstPtrBegin - (uint64_t)HstPtrBase;
      void *TgtPtrBase = (void *)((uint64_t)TgtPtrBegin - Delta);
      Device.data_submit(Pointer_TgtPtrBegin, &TgtPtrBase, sizeof(void *));
      // create shadow pointers for this entry
      Device.ShadowMtx.lock();
      Device.ShadowPtrMap[Pointer_HstPtrBegin] = {HstPtrBase,
          Pointer_TgtPtrBegin, TgtPtrBase};
      Device.ShadowMtx.unlock();
    }
  }
}

EXTERN void __tgt_target_data_begin_nowait(int32_t device_id, int32_t arg_num,
                                           void **args_base, void **args,
                                           int64_t *arg_sizes,
                                           int32_t *arg_types, int32_t depNum,
                                           void *depList, int32_t noAliasDepNum,
                                           void *noAliasDepList) {
  if (depNum + noAliasDepNum > 0)
    __kmpc_omp_taskwait(NULL, 0);

  __tgt_target_data_begin(device_id, arg_num, args_base, args, arg_sizes,
                          arg_types);
}

/// creates host-to-target data mapping, stores it in the
/// libomptarget.so internal structure (an entry in a stack of data maps)
/// and passes the data to the device.
EXTERN void __tgt_target_data_begin(int32_t device_id, int32_t arg_num,
                                    void **args_base, void **args,
                                    int64_t *arg_sizes, int32_t *arg_types) {
  DP("Entering data begin region for device %d with %d mappings\n", device_id,
     arg_num);

  // No devices available?
  if (device_id == OFFLOAD_DEVICE_DEFAULT) {
    device_id = omp_get_default_device();
    DP("Use default device id %d\n", device_id);
  }

  if (CheckDevice(device_id) != OFFLOAD_SUCCESS) {
    DP("Failed to get device %d ready\n", device_id);
    return;
  }

  DeviceTy& Device = Devices[device_id];

  // Translate maps
  int32_t new_arg_num;
  void **new_args_base;
  void **new_args;
  int64_t *new_arg_sizes;
  int64_t *new_arg_types;
  translate_map(arg_num, args_base, args, arg_sizes, arg_types, new_arg_num,
      new_args_base, new_args, new_arg_sizes, new_arg_types, false);

  //target_data_begin(Device, arg_num, args_base, args, arg_sizes, arg_types);
  target_data_begin(Device, new_arg_num, new_args_base, new_args, new_arg_sizes,
      new_arg_types);

  // Cleanup translation memory
  cleanup_map(new_arg_num, new_args_base, new_args, new_arg_sizes,
      new_arg_types, arg_num, args_base);
}

/// Internal function to undo the mapping and retrieve the data from the device.
static void target_data_end(DeviceTy &Device, int32_t arg_num, void **args_base,
                            void **args, int64_t *arg_sizes,
                            int64_t *arg_types) {
  // process each input.
  for (int32_t i = arg_num - 1; i >= 0; --i) {
    // Ignore private variables and arrays - there is no mapping for them.
    // Also, ignore the use_device_ptr directive, it has no effect here.
    if ((arg_types[i] & OMP_TGT_MAPTYPE_LITERAL) ||
        (arg_types[i] & OMP_TGT_MAPTYPE_PRIVATE))
      continue;

    void *HstPtrBegin = args[i];
    long IsLast;
    long UpdateRef = !(arg_types[i] & OMP_TGT_MAPTYPE_MEMBER_OF) ||
        (arg_types[i] & OMP_TGT_MAPTYPE_PTR_AND_OBJ);
    long ForceDelete = arg_types[i] & OMP_TGT_MAPTYPE_DELETE;

    // If PTR_AND_OBJ, HstPtrBegin is address of pointee
    void *TgtPtrBegin = Device.getTgtPtrBegin(HstPtrBegin, arg_sizes[i], IsLast,
        UpdateRef);
    DP("There are %ld bytes allocated at target address %016lx - is last %ld\n",
       (long)arg_sizes[i], (long)TgtPtrBegin, IsLast);

    if ((arg_types[i] & OMP_TGT_MAPTYPE_MEMBER_OF) &&
        !(arg_types[i] & OMP_TGT_MAPTYPE_PTR_AND_OBJ)) {
      IsLast = false; // protect parent struct from being deallocated
    }

    long DelEntry = IsLast || ForceDelete;

    if ((arg_types[i] & OMP_TGT_MAPTYPE_FROM) || DelEntry) {
      // Move data back to the host
      if (arg_types[i] & OMP_TGT_MAPTYPE_FROM) {
        long Always = arg_types[i] & OMP_TGT_MAPTYPE_ALWAYS;
        long CopyMember = false;
        if ((arg_types[i] & OMP_TGT_MAPTYPE_MEMBER_OF) &&
            !(arg_types[i] & OMP_TGT_MAPTYPE_PTR_AND_OBJ)) {
          // Copy data only if the "parent" struct has RefCount==1.
          short parent_idx = member_of(arg_types[i]);
          HostDataToTargetTy *entry = Device.getMapEntry(args[parent_idx]);
          if (entry->RefCount == 1) {
            CopyMember = true;
          }
        }

        if (DelEntry || Always || CopyMember) {
          DP("Moving %ld bytes (tgt:%016lx) -> (hst:%016lx)\n",
              (long)arg_sizes[i], (long)TgtPtrBegin, (long)HstPtrBegin);
          Device.data_retrieve(HstPtrBegin, TgtPtrBegin, arg_sizes[i]);
        }
      }

      // If we copied back to the host a struct/array containing pointers, we
      // need to restore the original host pointer values from their shadow
      // copies. If the struct is going to be deallocated, remove any remaining
      // shadow pointer entries for this struct.
      long lb = (long) HstPtrBegin;
      long ub = (long) HstPtrBegin + arg_sizes[i];
      Device.ShadowMtx.lock();
      for (ShadowPtrListTy::iterator it = Device.ShadowPtrMap.begin();
          it != Device.ShadowPtrMap.end(); ++it) {
        void **ShadowHstPtrAddr = (void**) it->first;

        // An STL map is sorted on its keys; use this property
        // to quickly determine when to break out of the loop.
        if ((long) ShadowHstPtrAddr < lb)
          continue;
        if ((long) ShadowHstPtrAddr >= ub)
          break;

        // If we copied the struct to the host, we need to restore the pointer.
        if (arg_types[i] & OMP_TGT_MAPTYPE_FROM) {
          DP("Restoring original host pointer value %016lx for host pointer "
              "%016lx\n", (long)it->second.HstPtrVal, (long)ShadowHstPtrAddr);
          *ShadowHstPtrAddr = it->second.HstPtrVal;
        }
        // If the struct is to be deallocated, remove the shadow entry.
        if (DelEntry) {
          DP("Removing shadow pointer %016lx\n", (long)ShadowHstPtrAddr);
          Device.ShadowPtrMap.erase(it);
        }
      }
      Device.ShadowMtx.unlock();

      // Deallocate map
      if (DelEntry) {
        Device.deallocTgtPtr(HstPtrBegin, arg_sizes[i], ForceDelete);
      }
    }
  }
}

/// passes data from the target, releases target memory and destroys
/// the host-target mapping (top entry from the stack of data maps)
/// created by the last __tgt_target_data_begin.
EXTERN void __tgt_target_data_end(int32_t device_id, int32_t arg_num,
                                  void **args_base, void **args,
                                  int64_t *arg_sizes, int32_t *arg_types) {
  DP("Entering data end region with %d mappings\n", arg_num);

  // No devices available?
  if (device_id == OFFLOAD_DEVICE_DEFAULT) {
    device_id = omp_get_default_device();
  }

  RTLsMtx.lock();
  size_t Devices_size = Devices.size();
  RTLsMtx.unlock();
  if (Devices_size <= (size_t)device_id) {
    DP("Device ID  %d does not have a matching RTL.\n", device_id);
    return;
  }

  DeviceTy &Device = Devices[device_id];
  if (!Device.IsInit) {
    DP("uninit device: ignore");
    return;
  }

  // Translate maps
  int32_t new_arg_num;
  void **new_args_base;
  void **new_args;
  int64_t *new_arg_sizes;
  int64_t *new_arg_types;
  translate_map(arg_num, args_base, args, arg_sizes, arg_types, new_arg_num,
      new_args_base, new_args, new_arg_sizes, new_arg_types, false);

  //target_data_end(Device, arg_num, args_base, args, arg_sizes, arg_types);
  target_data_end(Device, new_arg_num, new_args_base, new_args, new_arg_sizes,
      new_arg_types);

  // Cleanup translation memory
  cleanup_map(new_arg_num, new_args_base, new_args, new_arg_sizes,
      new_arg_types, arg_num, args_base);
}

EXTERN void __tgt_target_data_end_nowait(int32_t device_id, int32_t arg_num,
                                         void **args_base, void **args,
                                         int64_t *arg_sizes, int32_t *arg_types,
                                         int32_t depNum, void *depList,
                                         int32_t noAliasDepNum,
                                         void *noAliasDepList) {
  if (depNum + noAliasDepNum > 0)
    __kmpc_omp_taskwait(NULL, 0);

  __tgt_target_data_end(device_id, arg_num, args_base, args, arg_sizes,
                        arg_types);
}

/// passes data to/from the target.
EXTERN void __tgt_target_data_update(int32_t device_id, int32_t arg_num,
                                     void **args_base, void **args,
                                     int64_t *arg_sizes, int32_t *arg_types) {
  DP("Entering data update with %d mappings\n", arg_num);

  // No devices available?
  if (device_id == OFFLOAD_DEVICE_DEFAULT) {
    device_id = omp_get_default_device();
  }

  if (CheckDevice(device_id) != OFFLOAD_SUCCESS) {
    DP("Failed to get device %d ready\n", device_id);
    return;
  }

  DeviceTy& Device = Devices[device_id];

  // process each input.
  for (int32_t i = 0; i < arg_num; ++i) {
    if ((arg_types[i] & OMP_TGT_MAPTYPE_LITERAL) ||
        (arg_types[i] & OMP_TGT_MAPTYPE_PRIVATE))
      continue;

    void *HstPtrBegin = args[i];
    int64_t MapSize = arg_sizes[i];
    long IsLast;
    void *TgtPtrBegin = Device.getTgtPtrBegin(HstPtrBegin, MapSize, IsLast,
        false);

    if (arg_types[i] & OMP_TGT_MAPTYPE_FROM) {
      DP("Moving %ld bytes (tgt:%016lx) -> (hst:%016lx)\n", (long)arg_sizes[i],
         (long)TgtPtrBegin, (long)HstPtrBegin);
      Device.data_retrieve(HstPtrBegin, TgtPtrBegin, MapSize);

      long lb = (long) HstPtrBegin;
      long ub = (long) HstPtrBegin + MapSize;
      Device.ShadowMtx.lock();
      for (ShadowPtrListTy::iterator it = Device.ShadowPtrMap.begin();
          it != Device.ShadowPtrMap.end(); ++it) {
        void **ShadowHstPtrAddr = (void**) it->first;
        if ((long) ShadowHstPtrAddr < lb)
          continue;
        if ((long) ShadowHstPtrAddr >= ub)
          break;
        DP("Restoring original host pointer value %016lx "
            "for host pointer %016lx\n",
            (long)it->second.HstPtrVal, (long)ShadowHstPtrAddr);
        *ShadowHstPtrAddr = it->second.HstPtrVal;
      }
      Device.ShadowMtx.unlock();
    }

    if (arg_types[i] & OMP_TGT_MAPTYPE_TO) {
      DP("Moving %ld bytes (hst:%016lx) -> (tgt:%016lx)\n", (long)arg_sizes[i],
         (long)HstPtrBegin, (long)TgtPtrBegin);
      Device.data_submit(TgtPtrBegin, HstPtrBegin, MapSize);

      long lb = (long) HstPtrBegin;
      long ub = (long) HstPtrBegin + MapSize;
      Device.ShadowMtx.lock();
      for (ShadowPtrListTy::iterator it = Device.ShadowPtrMap.begin();
          it != Device.ShadowPtrMap.end(); ++it) {
        void **ShadowHstPtrAddr = (void**) it->first;
        if ((long) ShadowHstPtrAddr < lb)
          continue;
        if ((long) ShadowHstPtrAddr >= ub)
          break;
        DP("Restoring original target pointer value %016lx "
            "for target pointer %016lx\n",
            (long)it->second.TgtPtrVal, (long)it->second.TgtPtrAddr);
        Device.data_submit(it->second.TgtPtrAddr,
            &it->second.TgtPtrVal, sizeof(void *));
      }
      Device.ShadowMtx.unlock();
    }
  }
}

EXTERN void __tgt_target_data_update_nowait(
    int32_t device_id, int32_t arg_num, void **args_base, void **args,
    int64_t *arg_sizes, int32_t *arg_types, int32_t depNum, void *depList,
    int32_t noAliasDepNum, void *noAliasDepList) {
  if (depNum + noAliasDepNum > 0)
    __kmpc_omp_taskwait(NULL, 0);

  __tgt_target_data_update(device_id, arg_num, args_base, args, arg_sizes,
                           arg_types);
}

/// performs the same actions as data_begin in case arg_num is
/// non-zero and initiates run of the offloaded region on the target platform;
/// if arg_num is non-zero after the region execution is done it also
/// performs the same action as data_update and data_end above. This function
/// returns 0 if it was able to transfer the execution to a target and an
/// integer different from zero otherwise.
static int target(int32_t device_id, void *host_ptr, int32_t arg_num,
                  void **args_base, void **args, int64_t *arg_sizes,
                  int64_t *arg_types, int32_t team_num, int32_t thread_limit,
                  int IsTeamConstruct) {
  DeviceTy &Device = Devices[device_id];

  // Find the table information in the map or look it up in the translation
  // tables.
  TableMap *TM = 0;
  TblMapMtx.lock();
  HostPtrToTableMapTy::iterator TableMapIt = HostPtrToTableMap.find(host_ptr);
  if (TableMapIt == HostPtrToTableMap.end()) {
    // We don't have a map. So search all the registered libraries.
    TrlTblMtx.lock();
    for (HostEntriesBeginToTransTableTy::iterator
             ii = HostEntriesBeginToTransTable.begin(),
             ie = HostEntriesBeginToTransTable.end();
         !TM && ii != ie; ++ii) {
      // get the translation table (which contains all the good info).
      TranslationTable *TransTable = &ii->second;
      // iterate over all the host table entries to see if we can locate the
      // host_ptr.
      __tgt_offload_entry *begin = TransTable->HostTable.EntriesBegin;
      __tgt_offload_entry *end = TransTable->HostTable.EntriesEnd;
      __tgt_offload_entry *cur = begin;
      for (uint32_t i = 0; cur < end; ++cur, ++i) {
        if (cur->addr != host_ptr)
          continue;
        // we got a match, now fill the HostPtrToTableMap so that we
        // may avoid this search next time.
        TM = &HostPtrToTableMap[host_ptr];
        TM->Table = TransTable;
        TM->Index = i;
        break;
      }
    }
    TrlTblMtx.unlock();
  } else {
    TM = &TableMapIt->second;
  }
  TblMapMtx.unlock();

  // No map for this host pointer found!
  if (!TM) {
    DP("Host ptr %016lx does not have a matching target pointer.\n",
       (long)host_ptr);
    return OFFLOAD_FAIL;
  }

  // get target table.
  TrlTblMtx.lock();
  assert(TM->Table->TargetsTable.size() > (size_t)device_id &&
         "Not expecting a device ID outside the table's bounds!");
  __tgt_target_table *TargetTable = TM->Table->TargetsTable[device_id];
  TrlTblMtx.unlock();
  assert(TargetTable && "Global data has not been mapped\n");

  // Move data to device.
  target_data_begin(Device, arg_num, args_base, args, arg_sizes, arg_types);

  std::vector<void *> tgt_args;

  // List of (first-)private arrays allocated for this target region
  std::vector<void *> fpArrays;

  for (int32_t i = 0; i < arg_num; ++i) {
    if (!(arg_types[i] & OMP_TGT_MAPTYPE_TARGET_PARAM)) {
      // This is not a target parameter, do not push it into tgt_args.
      continue;
    }
    void *HstPtrBegin = args[i];
    void *HstPtrBase = args_base[i];
    void *TgtPtrBase;
    long IsLast; // unused.
    if (arg_types[i] & OMP_TGT_MAPTYPE_LITERAL) {
      DP("Forwarding first-private value %016lx to the target construct\n",
          (long)HstPtrBase);
      TgtPtrBase = HstPtrBase;
    } else if (arg_types[i] & OMP_TGT_MAPTYPE_PRIVATE) {
      // Allocate memory for (first-)private array
      void *TgtPtrBegin = Device.RTL->data_alloc(Device.RTLDeviceID,
          arg_sizes[i]);
      fpArrays.push_back(TgtPtrBegin);
      uint64_t PtrDelta = (uint64_t)HstPtrBegin - (uint64_t)HstPtrBase;
      TgtPtrBase = (void *)((uint64_t)TgtPtrBegin - PtrDelta);
      // If first-private, copy data from host
      if (arg_types[i] & OMP_TGT_MAPTYPE_TO) {
        Device.data_submit(TgtPtrBegin, HstPtrBegin, arg_sizes[i]);
      }
    } else if (arg_types[i] & OMP_TGT_MAPTYPE_PTR_AND_OBJ) {
      DP("Obtaining target argument from host pointer %016lx to object %016lx "
         "\n", (long)HstPtrBase, (long)HstPtrBegin);
      void *TgtPtrBegin = Device.getTgtPtrBegin(HstPtrBase, sizeof(void *),
          IsLast, false);
      TgtPtrBase = TgtPtrBegin; // no offset for ptrs.
    } else {
      DP("Obtaining target argument from host pointer %016lx\n",
          (long)HstPtrBegin);
      void *TgtPtrBegin = Device.getTgtPtrBegin(HstPtrBegin, arg_sizes[i],
          IsLast, false);
      uint64_t PtrDelta = (uint64_t)HstPtrBegin - (uint64_t)HstPtrBase;
      TgtPtrBase = (void *)((uint64_t)TgtPtrBegin - PtrDelta);
    }
    tgt_args.push_back(TgtPtrBase);
  }
  // Push omp handle.
  tgt_args.push_back((void *)0);

  // Pop loop trip count
  uint64_t ltc = Device.loopTripCnt;
  Device.loopTripCnt = 0;

  // Launch device execution.
  int rc;
  DP("Launching target execution with pointer %016lx (index=%d).\n",
     (long)TargetTable->EntriesBegin[TM->Index].addr, TM->Index);
  if (IsTeamConstruct) {
    rc = Device.run_team_region(TargetTable->EntriesBegin[TM->Index].addr,
                                &tgt_args[0], tgt_args.size(), team_num,
                                thread_limit, ltc);
  } else {
    rc = Device.run_region(TargetTable->EntriesBegin[TM->Index].addr,
                           &tgt_args[0], tgt_args.size());
  }

  // Deallocate (first-)private arrays
  for (auto it : fpArrays) {
    Device.RTL->data_delete(Device.RTLDeviceID, it);
  }

  // Move data from device.
  target_data_end(Device, arg_num, args_base, args, arg_sizes, arg_types);

  if (rc)
    return OFFLOAD_FAIL;

  return OFFLOAD_SUCCESS;
}

EXTERN int __tgt_target(int32_t device_id, void *host_ptr, int32_t arg_num,
                        void **args_base, void **args, int64_t *arg_sizes,
                        int32_t *arg_types) {
  if (device_id == OFFLOAD_DEVICE_CONSTRUCTOR ||
      device_id == OFFLOAD_DEVICE_DESTRUCTOR) {
    // Return immediately for the time being, target calls with device_id
    // -2 or -3 will be removed from the compiler in the future.
    return OFFLOAD_SUCCESS;
  }

  DP("Entering target region with entry point %016lx and device Id %d\n",
     (long)host_ptr, device_id);

  if (device_id == OFFLOAD_DEVICE_DEFAULT) {
    device_id = omp_get_default_device();
  }

  if (CheckDevice(device_id) != OFFLOAD_SUCCESS) {
    DP("Failed to get device %d ready\n", device_id);
    return OFFLOAD_FAIL;
  }

  // Translate maps
  int32_t new_arg_num;
  void **new_args_base;
  void **new_args;
  int64_t *new_arg_sizes;
  int64_t *new_arg_types;
  translate_map(arg_num, args_base, args, arg_sizes, arg_types, new_arg_num,
      new_args_base, new_args, new_arg_sizes, new_arg_types, true);

  //return target(device_id, host_ptr, arg_num, args_base, args, arg_sizes,
  //    arg_types, 0, 0, false /*team*/, false /*recursive*/);
  int rc = target(device_id, host_ptr, new_arg_num, new_args_base, new_args,
      new_arg_sizes, new_arg_types, 0, 0, false /*team*/);

  // Cleanup translation memory
  cleanup_map(new_arg_num, new_args_base, new_args, new_arg_sizes,
      new_arg_types, arg_num, args_base);

  return rc;
}

EXTERN int __tgt_target_nowait(int32_t device_id, void *host_ptr,
                               int32_t arg_num, void **args_base, void **args,
                               int64_t *arg_sizes, int32_t *arg_types,
                               int32_t depNum, void *depList,
                               int32_t noAliasDepNum, void *noAliasDepList) {
  if (depNum + noAliasDepNum > 0)
    __kmpc_omp_taskwait(NULL, 0);

  return __tgt_target(device_id, host_ptr, arg_num, args_base, args, arg_sizes,
                      arg_types);
}

EXTERN int __tgt_target_teams(int32_t device_id, void *host_ptr,
                              int32_t arg_num, void **args_base, void **args,
                              int64_t *arg_sizes, int32_t *arg_types,
                              int32_t team_num, int32_t thread_limit) {
  if (device_id == OFFLOAD_DEVICE_CONSTRUCTOR ||
      device_id == OFFLOAD_DEVICE_DESTRUCTOR) {
    // Return immediately for the time being, target calls with device_id
    // -2 or -3 will be removed from the compiler in the future.
    return OFFLOAD_SUCCESS;
  }

  DP("Entering target region with entry point %016lx and device Id %d\n",
     (long)host_ptr, device_id);

  if (device_id == OFFLOAD_DEVICE_DEFAULT) {
    device_id = omp_get_default_device();
  }

  if (CheckDevice(device_id) != OFFLOAD_SUCCESS) {
    DP("Failed to get device %d ready\n", device_id);
    return OFFLOAD_FAIL;
  }

  // Translate maps
  int32_t new_arg_num;
  void **new_args_base;
  void **new_args;
  int64_t *new_arg_sizes;
  int64_t *new_arg_types;
  translate_map(arg_num, args_base, args, arg_sizes, arg_types, new_arg_num,
      new_args_base, new_args, new_arg_sizes, new_arg_types, true);

  //return target(device_id, host_ptr, arg_num, args_base, args, arg_sizes,
  //              arg_types, team_num, thread_limit, true /*team*/,
  //              false /*recursive*/);
  int rc = target(device_id, host_ptr, new_arg_num, new_args_base, new_args,
      new_arg_sizes, new_arg_types, team_num, thread_limit, true /*team*/);

  // Cleanup translation memory
  cleanup_map(new_arg_num, new_args_base, new_args, new_arg_sizes,
      new_arg_types, arg_num, args_base);

  return rc;
}

EXTERN int __tgt_target_teams_nowait(int32_t device_id, void *host_ptr,
                                     int32_t arg_num, void **args_base,
                                     void **args, int64_t *arg_sizes,
                                     int32_t *arg_types, int32_t team_num,
                                     int32_t thread_limit, int32_t depNum,
                                     void *depList, int32_t noAliasDepNum,
                                     void *noAliasDepList) {
  if (depNum + noAliasDepNum > 0)
    __kmpc_omp_taskwait(NULL, 0);

  return __tgt_target_teams(device_id, host_ptr, arg_num, args_base, args,
                            arg_sizes, arg_types, team_num, thread_limit);
}

EXTERN void __kmpc_push_target_tripcount(int32_t device_id,
    uint64_t loop_tripcount) {
  if (device_id == OFFLOAD_DEVICE_DEFAULT) {
    device_id = omp_get_default_device();
  }

  if (CheckDevice(device_id) != OFFLOAD_SUCCESS) {
    DP("Failed to get device %d ready\n", device_id);
    return;
  }

  DP("__kmpc_push_target_tripcount(%d, %lu)\n", device_id, loop_tripcount);
  Devices[device_id].loopTripCnt = loop_tripcount;
}

////////////////////////////////////////////////////////////////////////////////
// temporary for debugging (matching the ones in omptarget-nvptx)

EXTERN void __kmpc_kernel_print(char *title) { DP(" %s\n", title); }

EXTERN void __kmpc_kernel_print_int8(char *title, int64_t data) {
  DP(" %s val=%lld\n", title, (long long)data);
}
