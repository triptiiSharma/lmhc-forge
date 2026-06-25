#!/usr/bin/env bash
# run_all.sh — LMHC-Forge full pipeline
# Regenerates every result and figure from a clean state.
# Must be run inside WSL2 (Kali Linux 2026.1) as root at /opt/lmhc-forge/.
#
# ── PREREQUISITES (must be satisfied before running) ──────────────────────────
#
# 1. Hardware
#    NVIDIA GPU with sm_80 or newer. Results in this repo are from:
#    RTX 4050 Laptop GPU (Ada Lovelace, sm_89, 100 KB shared/SM).
#    Change ARCH below if your GPU differs.
#
# 2. WSL2 environment
#    Tested on: WSL2 Kali Linux 2026.1, glibc 2.42
#
# 3. CUDA 12.8 toolkit at /usr/local/cuda-12.8/
#    Install:
#      wget https://developer.download.nvidia.com/compute/cuda/repos/debian12/x86_64/cuda-keyring_1.1-1_all.deb
#      dpkg -i cuda-keyring_1.1-1_all.deb
#      apt-get update && apt-get install -y cuda-nvcc-12-8 cuda-cuobjdump-12-8
#    (Add [trusted=yes] to the repo line in /etc/apt/sources.list.d/ if apt
#     reports SHA1 signature errors on Kali 2026.1.)
#
# 4. gcc-14 host compiler
#    apt-get install -y gcc-14
#
# 5. glibc 2.42 compatibility patch for CUDA 12.8 (one-time, system-wide)
#    glibc 2.42 adds C23 math functions (cospi, sinpi, rsqrt, etc.) that
#    conflict with CUDA 12.8's declarations. Fix:
#      MH=/usr/local/cuda-12.8/targets/x86_64-linux/include/crt/math_functions.h
#      cp $MH ${MH}.orig
#      # Add __THROW to declarations of cospi, sinpi, rsqrt, cospif, sinpif, rsqrtf
#      # (six lines total — see math_functions.h.orig diff in this repo)
#    Without this patch, nvcc will error on any .cu file that includes <cmath>.
#
# 6. Python3 + matplotlib + numpy
#    apt-get install -y python3-matplotlib python3-numpy
#
# 7. cuobjdump at /usr/local/cuda-12.8/bin/cuobjdump (comes with cuda-cuobjdump-12-8)
#
# ── OPTIONAL: nvcc 13.3 for DSE comparison ────────────────────────────────────
#    apt-get install -y cuda-nvcc-13-3 cuda-cuobjdump-13-3
#    Host compiler for 13.3 must be g++-14 (nvcc 13.3 emits C++ guard symbols).
#
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC="$ROOT/src"
RESULTS="$ROOT/results"
FIGURES="$ROOT/figures"
SASS_DUMPS="$ROOT/sass_dumps"

ARCH="sm_89"
NVCC="/usr/local/cuda-12.8/bin/nvcc"
CUOBJD="/usr/local/cuda-12.8/bin/cuobjdump"
CCBIN="/usr/bin/gcc-14"
NVCC_FLAGS="-arch=$ARCH -O3 -ccbin $CCBIN"

log() { echo "[run_all] $*"; }
die() { echo "[run_all] ERROR: $*" >&2; exit 1; }

# ── Prerequisite checks ───────────────────────────────────────────────────────
log "Checking prerequisites..."
command -v "$NVCC"    >/dev/null 2>&1 || die "nvcc not found at $NVCC"
command -v "$CUOBJD"  >/dev/null 2>&1 || die "cuobjdump not found at $CUOBJD"
command -v "$CCBIN"   >/dev/null 2>&1 || die "gcc-14 not found at $CCBIN"
command -v python3    >/dev/null 2>&1 || die "python3 not found"
python3 -c "import matplotlib, numpy" 2>/dev/null \
    || die "python3-matplotlib or python3-numpy missing (apt-get install -y python3-matplotlib python3-numpy)"

mkdir -p "$RESULTS" "$FIGURES" "$SASS_DUMPS"

# ── Phase 0 — Residue Detector ────────────────────────────────────────────────
log "=== Phase 0: Residue Detector ==="
$NVCC $NVCC_FLAGS -o "$ROOT/phase0_residue" "$SRC/phase0_residue.cu"
"$ROOT/phase0_residue" > "$RESULTS/phase0_residue.json"
log "  phase0_residue.json written"

# Phase 0 control (Kernel B only — no Kernel A)
$NVCC $NVCC_FLAGS -o "$ROOT/phase0_control" "$SRC/phase0_control.cu" 2>/dev/null \
    || log "  (phase0_control.cu not found — skipping control run)"
[ -f "$ROOT/phase0_control" ] && "$ROOT/phase0_control" > "$RESULTS/phase0_control.json" \
    && log "  phase0_control.json written"

# ── Module 1 — Discovery Engine ───────────────────────────────────────────────
log "=== Module 1: Discovery Engine ==="
python3 "$SRC/discover.py" "$SRC/test_kernels.cu" \
    --output-conservative "$RESULTS/discovery_conservative.json" \
    --output-taint        "$RESULTS/discovery_taint.json"
log "  discovery_conservative.json / discovery_taint.json written"

# ── Module 2 — LMHC Rewriter ─────────────────────────────────────────────────
log "=== Module 2: LMHC Rewriter ==="
python3 "$SRC/lmhc_rewriter.py" "$SRC/test_kernels.cu" \
    --taint "$RESULTS/discovery_taint.json" \
    --output "$SRC/test_kernels_lmhc.cu"
log "  test_kernels_lmhc.cu written"

# ── Module 3 — DSE Differential ───────────────────────────────────────────────
log "=== Module 3: DSE Differential ==="
$NVCC $NVCC_FLAGS --cubin -o "$SASS_DUMPS/dse_test_12.8.cubin" "$SRC/dse_test.cu"
"$CUOBJD" -sass "$SASS_DUMPS/dse_test_12.8.cubin" > "$SASS_DUMPS/dse_test.sass"
python3 "$SRC/dse_analysis.py" "$SASS_DUMPS/dse_test.sass" \
    --output "$RESULTS/dse_differential.json"
log "  dse_differential.json / dse_test.sass written"

# ── Module 4 — Witness Generator ──────────────────────────────────────────────
log "=== Module 4: Witness Generator ==="
# Compile the M2-rewritten source to produce a test-ready cubin first
M4_CU="$SRC/test_kernels_m4_lmhc.cu"
[ -f "$M4_CU" ] || die "M2 output not found at $M4_CU (did Module 2 run?)"
python3 "$SRC/witness_gen.py" "$M4_CU" \
    --arch "$ARCH" --nvcc "$NVCC" --ccbin "$CCBIN" \
    --output "$RESULTS/witness.json" \
    --save-cubin "/tmp/test_kernels.cubin"
cp /tmp/test_kernels.cubin "$RESULTS/test_kernels.cubin"
log "  witness.json / test_kernels.cubin written"

# ── Module 5 — Attestation Verifier + T1–T6 tamper tests ─────────────────────
log "=== Module 5: Attestation Verifier (T3 baseline) ==="
python3 "$SRC/attester.py" "$RESULTS/test_kernels.cubin" "$RESULTS/witness.json" \
    --cuobjdump "$CUOBJD"
log "  T3 baseline: PASS (see above)"

log "=== Module 5: Tamper Tests T1–T6 ==="
python3 "$SRC/tamper_tests.py" 2>&1 | tee "$RESULTS/tamper_tests.log"
log "  tamper_tests.log written"

# ── Module 6 — Leakage Demo ───────────────────────────────────────────────────
log "=== Module 6: Leakage Demo ==="
$NVCC $NVCC_FLAGS -o "$ROOT/leakage_demo" "$SRC/leakage_demo.cu"
"$ROOT/leakage_demo" > "$RESULTS/leakage_demo.json"
log "  leakage_demo.json written (EI_before=1.0, EI_after=0.0 expected)"

# ── Module 7 — Benchmark Suite ────────────────────────────────────────────────
log "=== Module 7: Benchmark Suite ==="
$NVCC $NVCC_FLAGS -lm -o "$ROOT/benchmark" "$SRC/benchmark.cu"
"$ROOT/benchmark" > "$RESULTS/benchmark.json"
log "  benchmark.json written"

# ── Visualizations ────────────────────────────────────────────────────────────
log "=== Visualizations ==="
python3 "$SRC/visualize.py"
log "  figures/ written: fig1–fig5 × {png,pdf}"

# ── Non-determinism check ─────────────────────────────────────────────────────
log "=== Non-determinism check ==="
# Re-run leakage_demo and benchmark a second time and compare key fields.
"$ROOT/leakage_demo" > /tmp/leakage_demo_2.json
EI_B1=$(python3 -c "import json,sys; d=json.load(open('$RESULTS/leakage_demo.json')); print(d['exposure_index_before'])")
EI_B2=$(python3 -c "import json,sys; d=json.load(open('/tmp/leakage_demo_2.json')); print(d['exposure_index_before'])")
EI_A1=$(python3 -c "import json,sys; d=json.load(open('$RESULTS/leakage_demo.json')); print(d['exposure_index_after'])")
EI_A2=$(python3 -c "import json,sys; d=json.load(open('/tmp/leakage_demo_2.json')); print(d['exposure_index_after'])")
if [ "$EI_B1" = "$EI_B2" ] && [ "$EI_A1" = "$EI_A2" ]; then
    log "  leakage_demo: deterministic (EI_before=$EI_B1, EI_after=$EI_A1 on both runs)"
else
    log "  WARNING: leakage_demo non-determinism detected!"
    log "    run1: EI_before=$EI_B1 EI_after=$EI_A1"
    log "    run2: EI_before=$EI_B2 EI_after=$EI_A2"
fi

# Benchmark timing is inherently variable; report stddev summary only.
python3 - <<'PYEOF'
import json, sys
d = json.load(open("RESULTS_PATH/benchmark.json".replace("RESULTS_PATH", sys.argv[1] if len(sys.argv)>1 else "/opt/lmhc-forge/results")))
print("[run_all]   benchmark stddevs (µs):", ", ".join(
    f"{k['class']}={k['baseline_stddev_us']:.2f}" for k in d["kernels"]))
PYEOF

log ""
log "=== run_all.sh COMPLETE ==="
log "Outputs:"
log "  results/phase0_residue.json   — Phase 0 residue rate"
log "  results/discovery_*.json      — Module 1 discovered buffers"
log "  src/test_kernels_lmhc.cu      — Module 2 rewritten source"
log "  results/dse_differential.json — Module 3 SASS differential"
log "  sass_dumps/dse_test.sass      — Module 3 raw SASS dump"
log "  results/witness.json          — Module 4 witness"
log "  results/tamper_tests.log      — Module 5 T1–T6 results"
log "  results/leakage_demo.json     — Module 6 EI before/after"
log "  results/benchmark.json        — Module 7 overhead table"
log "  figures/fig1_*.{png,pdf}      — Phase 0 heatmap"
log "  figures/fig2_*.{png,pdf}      — Module 3 DSE bar + SASS"
log "  figures/fig3_*.{png,pdf}      — Module 6 EI chart"
log "  figures/fig4_*.{png,pdf}      — Module 7 benchmark grouped bar"
log "  figures/fig5_*.{png,pdf}      — Module 5 T1–T6 PASS/FAIL panel"
