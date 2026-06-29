#!/usr/bin/env bash
# run.sh — LMHC-Forge multi-GPU replication: one-shot build + run
#
# This is the single command for collaborators:
#
#   cd /path/to/lmhc-forge/multi_gpu && bash run.sh
#
# What it does:
#   1. Detects compute capability of GPU 0 (via nvidia-smi or a probe binary).
#   2. Compiles multi_gpu_repro.cu against the detected arch.
#   3. Runs the harness (3 phases: control + residue + LMHC scrub), N=200.
#   4. Saves artifacts to runs/<YYYY-MM-DD_HH-MM-SS>_<gpu>/ :
#        <gpu>_<arch>_<driver>.json   — per-device result
#        run_metadata.txt             — exact compile + run command and flags
#        stderr.log                   — harness progress output
#   5. Also copies the JSON to results/ so aggregate.py works unchanged.
#   6. Prints the path to send back.
#
# Prerequisites: CUDA toolkit (nvcc), gcc-14 or gcc-12 or gcc, nvidia-smi.
# See README.md for full setup instructions.

set -euo pipefail
trap 'echo "[run.sh] ERROR: command failed at line $LINENO — aborting." >&2' ERR

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RESULTS_DIR="$SCRIPT_DIR/results"
RUNS_DIR="$SCRIPT_DIR/runs"
SRC="$SCRIPT_DIR/multi_gpu_repro.cu"
BIN="$SCRIPT_DIR/multi_gpu_repro"

mkdir -p "$RESULTS_DIR" "$RUNS_DIR"

# ── Locate nvcc ───────────────────────────────────────────────────────────────
if   command -v /usr/local/cuda-12.8/bin/nvcc >/dev/null 2>&1; then
    NVCC=/usr/local/cuda-12.8/bin/nvcc
elif command -v /usr/local/cuda/bin/nvcc       >/dev/null 2>&1; then
    NVCC=/usr/local/cuda/bin/nvcc
elif command -v nvcc                            >/dev/null 2>&1; then
    NVCC=$(command -v nvcc)
else
    echo "ERROR: nvcc not found. Install CUDA toolkit and ensure nvcc is on PATH." >&2
    exit 1
fi
echo "[run.sh] nvcc: $NVCC  ($("$NVCC" --version 2>&1 | grep 'release' | head -1 || true))"

# ── Locate host compiler ───────────────────────────────────────────────────────
for CC_CANDIDATE in gcc-14 gcc-12 gcc-11 gcc; do
    if command -v "$CC_CANDIDATE" >/dev/null 2>&1; then
        CCBIN=$(command -v "$CC_CANDIDATE")
        break
    fi
done
if [ -z "${CCBIN:-}" ]; then
    echo "ERROR: No suitable gcc found (tried gcc-14, gcc-12, gcc-11, gcc)." >&2
    exit 1
fi
echo "[run.sh] host compiler: $CCBIN"

# ── Detect compute capability ─────────────────────────────────────────────────
if command -v nvidia-smi >/dev/null 2>&1; then
    CC_RAW=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null \
             | head -1 | tr -d ' ')
    if [ -z "$CC_RAW" ]; then
        echo "ERROR: nvidia-smi returned empty compute capability. Check GPU driver." >&2
        exit 1
    fi
    MAJOR="${CC_RAW%%.*}"
    MINOR="${CC_RAW##*.}"
    ARCH="sm_${MAJOR}${MINOR}"
else
    # Fallback: compile a minimal probe and run it
    PROBE_SRC=$(mktemp /tmp/cc_probe.XXXXXX.cu)
    cat >"$PROBE_SRC" <<'CPROBE'
#include <stdio.h>
#include <cuda_runtime.h>
int main(){
    cudaDeviceProp p; cudaGetDeviceProperties(&p,0);
    printf("sm_%d%d\n",p.major,p.minor); return 0;
}
CPROBE
    PROBE_BIN=$(mktemp /tmp/cc_probe.XXXXXX)
    "$NVCC" -O0 -ccbin "$CCBIN" -o "$PROBE_BIN" "$PROBE_SRC" 2>/dev/null \
        && ARCH="$("$PROBE_BIN")" \
        || ARCH="sm_89"   # last-resort default; user can override below
    rm -f "$PROBE_SRC" "$PROBE_BIN"
fi
echo "[run.sh] target arch: $ARCH"

# ── Compile ───────────────────────────────────────────────────────────────────
# Array form handles spaces in any path component.
COMPILE_CMD=("$NVCC" -arch="$ARCH" -O2 -ccbin "$CCBIN" -o "$BIN" "$SRC")
# Display string for logging and run_metadata.txt (paths quoted so it's copy-pasteable).
COMPILE_CMD_DISPLAY="\"$NVCC\" -arch=$ARCH -O2 -ccbin \"$CCBIN\" -o \"$BIN\" \"$SRC\""
echo "[run.sh] Compiling: $COMPILE_CMD_DISPLAY"
"${COMPILE_CMD[@]}"
echo "[run.sh] Compiled: $BIN"

# ── Run (capture stderr to temp; echo it live to console) ─────────────────────
TS=$(date +%Y-%m-%d_%H-%M-%S)
STDERR_TMP=$(mktemp /tmp/multi_gpu_stderr.XXXXXX)
JSON_TMP=$(mktemp /tmp/multi_gpu_result.XXXXXX.json)
RUN_CMD_DISPLAY="\"$BIN\" -n 200 -o \"$JSON_TMP\""

echo "[run.sh] Running: $RUN_CMD_DISPLAY"
"$BIN" -n 200 -o "$JSON_TMP" 2>"$STDERR_TMP"
cat "$STDERR_TMP" >&2   # replay stderr so it's visible on the console

# ── Extract GPU info from JSON ────────────────────────────────────────────────
GPU=$(python3 -c "
import json, sys
d = json.load(open('$JSON_TMP'))
print(d['environment']['gpu_name'].replace(' ','_'))
" 2>/dev/null || echo "GPU")

ARCH_OUT=$(python3 -c "
import json
d = json.load(open('$JSON_TMP'))
print(d['environment']['compute_capability'])
" 2>/dev/null || echo "$ARCH")

DRIVER=$(python3 -c "
import json
d = json.load(open('$JSON_TMP'))
print(d['environment']['driver_version'].replace(' ','_'))
" 2>/dev/null || echo "driver_unknown")

OUT_NAME="${GPU}_${ARCH_OUT}_${DRIVER}.json"

# Sanitize GPU name for the directory component (keep alphanum, _, -, .)
GPU_SAFE=$(echo "$GPU" | sed 's/[^A-Za-z0-9._-]/_/g')

# ── Create timestamped run directory ─────────────────────────────────────────
RUN_DIR="$RUNS_DIR/${TS}_${GPU_SAFE}"
mkdir -p "$RUN_DIR"

# ── Persist artifacts into the run directory ──────────────────────────────────
JSON_IN_RUN="$RUN_DIR/$OUT_NAME"
cp "$JSON_TMP"    "$JSON_IN_RUN"
mv "$STDERR_TMP"  "$RUN_DIR/stderr.log"
rm -f "$JSON_TMP"

# Record the exact compile and run commands with all resolved flags
NVCC_VER=$("$NVCC" --version 2>&1 | grep -i 'release\|V[0-9]' | head -1 || true)
CCBIN_VER=$("$CCBIN" --version 2>&1 | head -1 || true)
cat >"$RUN_DIR/run_metadata.txt" <<METADATA
timestamp:       $TS
compile_command: $COMPILE_CMD_DISPLAY
run_command:     "$BIN" -n 200 -o "$JSON_IN_RUN"
arch:            $ARCH (auto-detected)
nvcc:            $NVCC
nvcc_version:    $NVCC_VER
ccbin:           $CCBIN
ccbin_version:   $CCBIN_VER
METADATA

# ── Also copy JSON to results/ so aggregate.py works unchanged ────────────────
cp "$JSON_IN_RUN" "$RESULTS_DIR/$OUT_NAME"

echo ""
echo "[run.sh] ===== DONE ====="
echo "[run.sh] Run directory : $RUN_DIR"
echo "[run.sh]   $OUT_NAME"
echo "[run.sh]   run_metadata.txt"
echo "[run.sh]   stderr.log"
echo "[run.sh] Also in       : $RESULTS_DIR/$OUT_NAME"
echo ""
echo "Please send back: $JSON_IN_RUN"
echo "  (or $RESULTS_DIR/$OUT_NAME — identical content)"
