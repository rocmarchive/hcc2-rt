CLOC_PATH=/usr/bin
HCC2=${HCC2:-/opt/rocm/hcc2}
HCC2RT_REPOS=${HCC2RT_REPOS:-/home/$USER/git/hcc2}
RT_REPO_NAME=${RT_REPO_NAME:-hcc2-rt}

$CLOC_PATH/cloc.sh -ll -vv -opt 2  hw.cl

g++ rtl_test.cpp -lelf -L/usr/lib/x86_64-linux-gnu -lomptarget -lpthread -L${HCC2}/lib -I$HCC2RT_REPOS/$RT_REPO_NAME/libamdgcn}/src -L/opt/rocm/lib -lhsa-runtime64 -g -o rtl_test

LD_LIBRARY_PATH=/opt/rocm/lib:$HCC2/lib:$LD_LIBRARY_PATH ./rtl_test hw.hsaco

