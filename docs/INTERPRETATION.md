# LMHC-Forge — Multi-GPU Replication: Pre-Committed Outcome Interpretations

This document is written **before any collaborator results are collected**.
Every possible outcome is committed here in advance so that the readings cannot
be adjusted post-hoc to fit observed data. The table of results produced by
`aggregate.py` is generated purely from per-device JSON files; no device's
numbers are hardcoded anywhere in the harness or aggregation scripts.

---

## Scientific Objective

Test the reproducibility of the residue mechanism and the LMHC mitigation across
NVIDIA architectures (Ada Lovelace, Ampere, Turing, and any others submitted)
under identical adversarial (forced co-residency) conditions. The only independent
variable is the GPU. Methodology is held byte-for-byte identical to the existing
controlled experiment (`src/leakage_demo.cu`): same kernels, same
`__launch_bounds__(32,1)` single-block same-SM forced co-residency setup, same
sentinel (0xCAFEBABE), same buffer geometry (4096 words / 16 KB), same trial
count (N = 200 default), same Exposure Index definition.

## Scope Guard

This harness performs **controlled-placement replication only**. An unconstrained
or natural-scheduling mode is explicitly out of scope for this artifact. If such a
mode is ever added later, it must be a separate exploratory path that never alters
or replaces the controlled results, and its outputs must be labeled
device-specific and non-generalizing.

---

## Framing That Applies to Every Result on Every Device

> Residue is fully observable when the subsequent kernel is co-resident on the
> same SM — co-residency is forced in this experiment to demonstrate the window.
> This is in-process cross-kernel shared-memory persistence, NOT cross-tenant
> LeftoverLocals on NVIDIA.
>
> Co-residency is forced on every device, so these are controlled-placement
> results consistent with our adversarial threat model — not natural-scheduling
> frequencies, and not a claim about production prevalence or about non-tested
> hardware (including the H100 CC target).

This framing is mandatory and applies regardless of the EI value observed.

---

## Pre-Committed Reading of Every Possible Outcome

### Outcome A — All tested devices show residue (EI > 0) without scrub and EI = 0.0 after LMHC

**Reading**: The residue mechanism and its closure replicate across the tested
NVIDIA architectures under identical adversarial conditions. Each device's result
stands on its own; results from one device do not imply anything about devices not
tested. Do not use language such as "all NVIDIA GPUs" or "modern GPUs" — refer
only to the specific device names that appear in the results table.

**What this does and does not claim**: It establishes that, for each tested device
individually, (a) uninitialized `__shared__` memory holds prior-kernel data when
co-residency is forced, and (b) the LMHC hardened scrub reduces the Exposure
Index to zero. It does not claim this frequency reflects natural production
scheduling rates, nor does it generalize to hardware not in the table.

---

### Outcome B — Devices differ (some show residue, some do not)

**Reading**: Report each device individually. Do not average across devices, do
not compute a "mean EI", and do not treat the subset that shows residue as more
representative than the subset that does not. Each row in the table is a
self-contained result.

If a device in the residue-present subset shows EI = 0.0 after LMHC scrub, that
is a complete positive result for that device: the mechanism is present and the
mitigation closes it. If a device in the residue-absent subset shows EI = 0.0
without scrub, that bounds where the controlled demonstration holds (see
Outcome C).

---

### Outcome C — A device shows no residue (EI = 0.0) under forced co-residency, with a clean control

**Reading**: Report it exactly as measured. This is a valid scientific result —
it means the controlled demonstration of the residue window does not hold on
this specific device under these conditions. Possible explanations include
hardware-level shared-memory initialization, driver-level scrubbing, or
architectural differences in SM SRAM retention. The JSON for this device carries
`valid: true` and the residue EI is reported as measured (0.0). This result
**bounds** where the controlled demonstration holds; it is not a failure of the
experiment.

Do not suppress, relabel, or explain away this result. It is precisely the kind
of outcome the experiment is designed to be capable of reporting.

---

### Outcome D — A device shows residue and LMHC scrub drives EI to 0.0 (device-level closure)

**Reading**: For this specific device, the LMHC mitigation closes the residue
window under adversarial co-residency. The architecture-independent character of
the mitigation result — that `volatile` stores + `__threadfence_block()` +
`__syncthreads()` prevent the listener from reading non-zero words — holds for
all devices in this category. The shared-memory ordering guarantee is a
consequence of the C++ memory model and PTX semantics, not of any particular
microarchitecture. State per-device; do not generalize to untested hardware.

---

### Outcome E — Control run is non-zero on any device

**Reading**: Mark that device's result `valid: false` with reason
`"control non-zero (possible self-contamination or driver scrub)"`. Do NOT
report the residue EI for this device as a leak or as a negative result — the
validity gate has failed and the measurement is unreliable. Report the control EI
as measured and flag the entry clearly in the aggregation table with `Valid = N`.
Do not attempt to interpret the residue or scrub EIs from an invalid device.

A non-zero control most likely indicates either (a) a driver that pre-fills
shared memory with a non-zero value, (b) self-contamination from a prior run in
the same process (a harness bug — file an issue), or (c) hardware behavior
outside the model. In any case, the result cannot be cleanly attributed to
cross-kernel residue and must not be so attributed.

---

## What the Table Does Not Claim

- It does not characterize scheduler behavior or natural co-residency rates.
- It does not claim that residue prevalence in the table reflects production
  multi-tenant frequencies.
- It does not characterize the H100 Confidential Computing target (out of scope
  for this artifact; see [SPEC.md](SPEC.md)).
- A device not in the table makes no appearance in any claim.
- The DSE characterization (58-STS / ordering-differential finding, Module 3) is
  toolchain-specific to ptxas 12.8/13.3 and is explicitly excluded from this
  harness. Collaborators may run different CUDA versions; residue + scrub
  elimination are the only measurements here.
