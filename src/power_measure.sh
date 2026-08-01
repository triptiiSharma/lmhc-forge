#!/bin/bash
set -e
cd "/mnt/c/Tripti Sharma/lmhc-forge"
NVCC=/usr/local/cuda-12.8/bin/nvcc
GCC=/usr/bin/gcc-14
BIN=/tmp/power_sustained
DUR=4
OUT=results

"$NVCC" -arch=sm_89 -O3 -ccbin "$GCC" -o "$BIN" src/power_sustained.cu

sample() {
  local mode="$1" logfile="$2"
  nvidia-smi --query-gpu=power.draw --format=csv,noheader -lms 100 > "$logfile" &
  local smi_pid=$!
  sleep 0.3   # let sampler settle
  "$BIN" "$mode" "$DUR"
  sleep 0.3
  kill "$smi_pid" 2>/dev/null || true
  wait "$smi_pid" 2>/dev/null || true
}

echo "=== idle ===" >&2
sample idle  /tmp/power_idle.log
echo "=== empty (fill only, no scrub) ===" >&2
sample empty /tmp/power_empty.log
echo "=== scrub (fill + LMHC hardened scrub) ===" >&2
sample scrub /tmp/power_scrub.log

python3 - <<'PYEOF'
import re, json, statistics
def load(path):
    vals=[]
    with open(path) as f:
        for line in f:
            m = re.search(r'([\d.]+)\s*W', line)
            if m: vals.append(float(m.group(1)))
    return vals

idle = load('/tmp/power_idle.log')
empty = load('/tmp/power_empty.log')
scrub = load('/tmp/power_scrub.log')

def stats(v):
    return {"n_samples": len(v), "mean_w": round(statistics.mean(v),3) if v else None,
            "stdev_w": round(statistics.stdev(v),3) if len(v)>1 else 0.0,
            "min_w": min(v) if v else None, "max_w": max(v) if v else None}

out = {
  "experiment": "Sustained-load power draw sampling (RP Sec VI.D 'Energy and thermal cost')",
  "method": "nvidia-smi --query-gpu=power.draw --format=csv,noheader -lms 100, sampled in parallel with a sustained CUDA launch loop (~4s) for each of 3 modes: idle (no GPU work), empty (repeated fill-only kernel launches, no scrub), scrub (same fill + full LMHC hardened scrub of a ~93.75KB shared-memory footprint per launch, matching benchmark_clock64.cu's amplification kernel).",
  "caveat": "This isolates the power delta of SUSTAINED, BACK-TO-BACK scrub-kernel launches (thousands/sec) from empty-kernel launches -- NOT the per-exit energy of a single scrub in an otherwise-idle pipeline. nvidia-smi power.draw reflects whole-board power (not per-kernel), sampled at ~100ms/10Hz -- far coarser than the ~200ns-800ns scrub duration itself. A non-significant or small delta here does not by itself establish per-exit energy is negligible; it is the coarsest instrument available without CUPTI power counters (not installed -- see results/verifier_perf.json and benchmark_clock64.json for the equivalent CUPTI/Nsight Compute gap noted there).",
  "idle": stats(idle),
  "empty_fill_only": stats(empty),
  "scrub_fill_and_scrub": stats(scrub),
}
if empty and scrub:
    out["delta_scrub_minus_empty_w"] = round(out["scrub_fill_and_scrub"]["mean_w"] - out["empty_fill_only"]["mean_w"], 3)
if idle and empty:
    out["delta_empty_minus_idle_w"] = round(out["empty_fill_only"]["mean_w"] - out["idle"]["mean_w"], 3)

with open("results/power_sustained.json","w") as f:
    json.dump(out, f, indent=2)
print(json.dumps(out, indent=2))
PYEOF
