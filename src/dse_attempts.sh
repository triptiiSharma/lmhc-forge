#!/bin/bash
set -e
cd "/mnt/c/Tripti Sharma/lmhc-forge"
NVCC=/usr/local/cuda-12.8/bin/nvcc
CUOBJ=/usr/local/cuda-12.8/bin/cuobjdump
GCC=/usr/bin/gcc-14
mkdir -p /tmp/dse_attempts

try() {
  local name="$1"; shift
  echo "=== $name : $* ==="
  if "$NVCC" -ccbin "$GCC" "$@" --cubin -o "/tmp/dse_attempts/$name.cubin" src/dse_test.cu 2> "/tmp/dse_attempts/$name.err"; then
    "$CUOBJ" -sass "/tmp/dse_attempts/$name.cubin" > "/tmp/dse_attempts/$name.sass" 2>&1
    n=$(awk '/Function : _Z11scrub_naive/,/Function : _Z14scrub_hardened/' "/tmp/dse_attempts/$name.sass" | grep -c STS || true)
    echo "  scrub_naive STS count: $n"
  else
    echo "  COMPILE FAILED:"
    cat "/tmp/dse_attempts/$name.err"
  fi
}

try O3plain -arch=sm_89 -O3
try O3_allow_expensive -arch=sm_89 -O3 -Xptxas -O3,--allow-expensive-optimizations=true
try O3_maxrreg16 -arch=sm_89 -O3 -Xptxas -O3 -maxrregcount=16
try O3_maxrreg8 -arch=sm_89 -O3 -Xptxas -O3 -maxrregcount=8
try O3_fastmath -arch=sm_89 -O3 --use_fast_math
try O3_dlto -arch=sm_89 -O3 -dlto
try O2plain -arch=sm_89 -O2
try O3_sm86 -arch=sm_86 -O3
