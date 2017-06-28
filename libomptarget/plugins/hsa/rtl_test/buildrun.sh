CLOC_PATH=/usr/bin
ATMI_PATH=${HOME}/git/atmi
OMPTARGET_BUILD_PATH=${HOME}/git/coral2/coral-omprt/libomptarget
OMPTARGET_PATH=${HOME}/opt/amd/llvm/omprt

$CLOC_PATH/cloc.sh -ll -vv -opt 2  hw.cl

g++ rtl_test.cpp -lelf -L/usr/lib/x86_64-linux-gnu -lomptarget -lpthread -L${OMPTARGET_PATH}/lib -I${OMPTARGET_BUILD_PATH}/src -L/opt/rocm/lib -lhsa-runtime64 -g -o rtl_test

LD_LIBRARY_PATH=/opt/rocm/libatmi/lib:/opt/rocm/lib:${OMPTARGET_PATH}/lib:$LD_LIBRARY_PATH ./rtl_test hw.hsaco

#LD_LIBRARY_PATH=${ATMI_PATH}/lib:/opt/rocm/lib:${OMPTARGET_PATH}/lib:$LD_LIBRARY_PATH ./rtl_test vmul.so.tgt-amdgcn--amdhsa
