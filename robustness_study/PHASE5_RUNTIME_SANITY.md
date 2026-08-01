# PHASE5_RUNTIME_SANITY.md

## What this is NOT (stated first, prominently, per instruction)

- These 18 diagnostic variants are **not the original core-matrix cubins**.
  Each adds a post-cleanup-point global-memory readback of the sensitive
  buffer, which is itself new code — the compiled artifact differs from
  its corresponding core-matrix artifact.
- Adding the readback instrumentation **changes liveness and
  compilation**: a buffer that is read back is no longer "dead" in the
  sense the compiler analyzes, so this experiment validates the
  **diagnostic template's** behavior, not the exact core-matrix cubin's
  compiled structure. (This is exactly why K0, the pure-dead-allocation
  context, is excluded — instrumenting it would invalidate the property
  K0 exists to study.)
- This is **not** a new Exposure Index or cross-kernel residue-recovery
  experiment. There is no writer/listener pattern, no forced co-residency,
  no second kernel. It is a same-kernel, same-launch behavioral check.
- It does **not** establish the absence of recoverable residue in any
  general sense — it checks one specific buffer, read back by the same
  kernel, immediately after its own cleanup point, on one GPU.
- It does **not** generalize beyond the tested hardware: **RTX 4050 Laptop
  GPU (sm_89) only, CUDA 12.8 only.** No claim is made about sm_75/80/86/
  87/90 runtime behavior — those remain compile-target evidence only, as
  throughout this entire study.

## What was run

18 diagnostic variants: 6 constructs (**Z0, Z1, Z2, Z4, Z5, Z6** — Z3
excluded per the approved plan's explicit construct list) × 3 contexts
(**K2, K3, K4** — K0 excluded for the reason above; K1/K5 not included,
bounded subset per instruction). Each variant is the same
construct×context kernel body from `kernel_lib.py`, with one addition: a
post-cleanup readback of the sensitive shared buffer into global memory,
so the host can count nonzero words. 5 repeats per variant (90 total
launches) to check consistency.

## Results

**18/18 compiled successfully. 90/90 launches succeeded (`cudaGetLastError`
= "no error"). 90/90 synchronizations succeeded
(`cudaDeviceSynchronize` = "no error"). 90/90 ordinary computations
matched the expected value exactly** (K2's deterministic XOR output; K3's
reduction sum, exact float match against the host-computed expected sum;
K4's 16×16 GEMM, exact float match against the host-computed reference,
all 256 output elements).

Post-cleanup nonzero-word counts, **fully consistent across all 5 repeats
for every variant** (no run-to-run variation observed):

| Construct | K2 (1024 words) | K3 (256 words) | K4 (256 words) |
|---|---:|---:|---:|
| Z0 (no scrub) | 1024 | 256 | 219 |
| Z1 (plain) | 0 | 0 | 0 |
| Z2 (volatile) | 0 | 0 | 0 |
| Z4 (volatile+barrier) | 0 | 0 | 0 |
| Z5 (hardened) | 0 | 0 | 0 |
| Z6 (inline PTX) | 0 | 0 | 0 |

Z0 (negative/positive control) shows genuinely nonzero leftover data in
every context — K4's 219/256 (not 256/256) is expected and consistent with
GEMM's zero-padding at tile boundaries for out-of-range indices (some
entries are legitimately zero from the computation itself, not from any
scrub). This confirms the diagnostic methodology itself is sound: an
unscrubbed buffer shows real nonzero content, and the readback correctly
captures it.

All five scrub-bearing constructs — including **Z1, the plain non-volatile
loop** — show 0 nonzero words in every context, every repeat. This is
consistent with, and now runtime-confirmed for, the compile-time finding
already established in Phases 3/4/5: in the K2/K3/K4 contexts (where the
buffer is genuinely read before the scrub), ptxas does not eliminate the
scrub stores, so they execute and the buffer is actually zero at readback
time — not merely "the SASS contains a store instruction," but the buffer
demonstrably holds zero when read back from a real, running kernel on real
hardware.

## Files produced

- `results/runtime_sanity.jsonl` — full per-variant records (compile
  command/status, 5-repeat stdout, launch/sync error strings)
- `results/runtime_sanity_summary.csv` — the table above, machine-readable
- `kernels/runtime_sanity/*.cu` — the 18 diagnostic sources
- `artifacts/runtime_sanity/` — the 18 compiled executables
