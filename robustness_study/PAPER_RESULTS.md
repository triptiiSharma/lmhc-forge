# PAPER_RESULTS.md

For every proposed paper claim: exact data source, query/aggregation,
artifact count, scope, limitation, safe wording, and wording that would
overclaim. This document does not edit the manuscript — it exists so that
whoever does can copy exact, defensible sentences rather than paraphrase
from memory.

---

## Claim 1 — Corpus scale

**Data source**: `results/artifacts.jsonl`, `results/summary.json`.
**Query**: count of records with `compile_status == "COMPILE_OK"`, grouped
by `zeroization_construct`, `kernel_context`, `ptxas_optimization`,
`target_arch`, `cuda_version`.
**Artifact count**: 2,016 (100% of attempted configurations).
**Scope**: 7 constructs × 6 contexts × 4 ptxas opt levels × 6 archs × 2
toolkits, all compiled in WSL2 on one machine (AUDIT.md §4).
**Limitation**: only sm_89 has matching physical hardware; the other 5
targets are compile-target evidence only.

**Safe wording**: "We compiled and analyzed 2,016 GPU binaries across two
CUDA toolkits (12.8, 13.3), six architecture targets (sm_75/80/86/87/89/
90), four ptxas optimization levels, seven zeroization constructs, and six
kernel contexts; every configuration compiled successfully."

**Overclaiming wording to avoid**: "We validated zeroization on 2,016 GPU
configurations" (implies runtime validation, which only exists for sm_89
and only for the 18 diagnostic runtime-sanity variants, not the 2,016
core-matrix cubins themselves).

---

## Claim 2 — Structural category distribution

**Data source**: `results/classifications.csv`, `results/summary.json`.
**Artifact count**: 2,016.
**Result**: `FULL_SIGNATURE` 576, `PARTIAL_SIGNATURE` 864, `STORE_ONLY` 96,
`ABSENT` 480, `UNDECIDABLE` 0.
**Limitation**: category names describe static structural presence/
ordering only (`PLAN.md §5`), not a security guarantee.

**Safe wording**: "The structural-signature checker assigned each of the
2,016 artifacts to one of the five defined categories; four occurred in
this corpus (UNDECIDABLE did not)."

**Overclaiming wording to avoid**: "X% of binaries were secure" —
category is a structural-text classification, not a security verdict.

---

## Claim 3 — Hardened constructs' universal FULL_SIGNATURE

**Data source**: `results/classifications.csv` filtered to
`zeroization_construct in {Z5, Z6}`.
**Artifact count**: 576 (288 each).
**Result**: 288/288 `FULL_SIGNATURE` for each of Z5 (hardened
volatile+fence+barrier) and Z6 (inline-PTX hardened), across every
context/opt-level/architecture/toolkit combination tested.
**Limitation**: "FULL_SIGNATURE" is a static-text structural finding
(`PLAN.md §5`), not proof of dynamic execution on every thread path — see
Claim 7 (N6) for direct counter-evidence of why that distinction matters.

**Safe wording**: "The hardened zeroization constructs (volatile+fence+
barrier, and its inline-PTX-store variant) retained the complete
recognized structural signature — zero-store, fence, and barrier in the
required order before every reachable exit — across the entire tested
configuration matrix (576/576 artifacts)."

**Overclaiming wording to avoid**: "The hardened constructs guarantee
zeroization on every execution" — not established; see Claim 7.

---

## Claim 4 — Cross-tool agreement

**Data source**: `results/cross_tool_summary.json`,
`results/cross_tool_comparison.csv`.
**Artifact count**: 2,016 (preceded by a 96-artifact pilot gate,
`PHASE5_NVDISASM_PILOT.md`).
**Result**: 100% category agreement (2016/2016), 100% feature agreement
on all 8 security-relevant features, 0 disagreements, 0 unsupported, 0
UNDECIDABLE from either path.
**Limitation**: `nvdisasm` and `cuobjdump` both interpret the same NVIDIA
cubin — this is agreement between two parsing implementations, not
independent semantic ground truth.

**Safe wording**: "An independent disassembly and parsing path
(`nvdisasm`'s structured JSON output, a separate implementation from the
primary `cuobjdump`-based parser) agreed with the primary analysis on
2,016 of 2,016 supported artifacts (100%), on both structural category
and every extracted security-relevant feature."

**Overclaiming wording to avoid**: "Independently verified" / "confirmed
correct by a second tool" — both tools disassemble the same binary; use
"agreement between two parsing paths," not "independent verification of
correctness."

---

## Claim 5 — Controlled-case conformance

**Data source**: `results/controlled_cases.jsonl`,
`results/controlled_case_summary.csv`.
**Artifact count**: 55 configurations across 13 defined cases (N0–N12);
54 `REAL_COMPILED_CASE` + 1 `REAL_BINARY_INPUT_CASE`; 0
`SYNTHETIC_TEST_ONLY` pooled.
**Result**: 47/48 dedicated-case configurations conform exactly to their
intended structural condition (100% cross-tool agreement on every one).
The remaining 8 (all case N6) are a distinct, documented finding — see
Claim 7.
**Limitation**: "conformance" means the compiled binary matches the
intended structural condition, checked by both parsing paths — it is not
a general accuracy statement about the checker.

**Safe wording**: "A controlled corpus of 13 structural cases, realized
as genuine compiled source across both CUDA toolkits, two ptxas
optimization levels, and two architectures (55 configurations total),
showed the expected structural condition in 47 of 48 dedicated
configurations; the remaining case is discussed separately."

**Overclaiming wording to avoid**: "The checker was validated against 13
test cases with 96% accuracy" — this is conformance to intended
conditions in a controlled corpus, not accuracy against independent
ground truth, and the 8 "non-conforming" configs are a single documented
finding, not 8 independent failures.

---

## Claim 6 — Reproducibility

**Data source**: `results/reproducibility_summary.json`.
**Artifact count**: 50 configurations × 3 repeats = 150 new compiles.
**Result**: 150/150 byte-identical across cubin, raw SASS, and raw
`nvdisasm` JSON hashes; independently re-verified by a separate script.
**Limitation**: single machine, single WSL2 environment, same two
toolkit installs throughout — not evidence about cross-machine or
cross-driver reproducibility.

**Safe wording**: "Repeated compilation of 50 stratified configurations
(3 additional compiles each) preserved the extracted structural category,
features, and compiled binary bytes exactly, on the tested machine and
toolchain."

**Overclaiming wording to avoid**: "Compilation is fully deterministic" —
established only for the tested machine/toolchain combination, not as a
general property of the compilers.

---

## Claim 7 — The N6 finding (structural-checker scope limitation, evidenced)

**Data source**: `results/controlled_cases.jsonl`, case `N6`;
`PHASE5_CONTROLLED_CASES.md`.
**Artifact count**: 8 configurations (2 toolkits × 2 opt levels × 2
archs), all compiled and disassembled, **never executed** (source has a
genuinely divergent early return).
**Result**: source has a cleanup path intentionally uncovered for half
the threads (`if (tid < 16) return;` before the scrub); ptxas compiled
this to a kernel with **exactly one** `EXIT` instruction (predication,
not a second exit); both `cuobjdump`- and `nvdisasm`-based paths
correctly and identically report `FULL_SIGNATURE` from the static text
(8/8 cross-tool agreement).
**Limitation**: this is exactly what "structural-signature checker, not a
CFG/post-dominance prover" means in practice — the checker cannot see
that the returning threads never reach the scrub, because ptxas's code
generation doesn't expose that as a separate exit for the checker to
notice.

**Safe wording**: "A controlled case with a source-level intentionally
uncovered cleanup path compiled to a single-exit binary via predicated
control flow; the structural checker correctly reported the recognized
signature present in the static text relative to that single exit,
concretely illustrating that a FULL_SIGNATURE verdict reflects static
structural presence and ordering, not proof that every dynamic thread
path executes the scrub."

**Overclaiming wording to avoid**: "The checker failed to detect an
incomplete cleanup" (it correctly applied its documented, honest scope —
this is not a failure, it's the scope boundary functioning as designed
and disclosed) — and never state or imply the checker performs CFG/
post-dominance analysis, which it explicitly does not.

---

## Claim 8 — Runtime sanity (bounded)

**Data source**: `results/runtime_sanity.jsonl`,
`results/runtime_sanity_summary.csv`.
**Artifact count**: 18 diagnostic variants × 5 repeats = 90 launches.
**Result**: 90/90 launch/sync/computation success; the no-scrub control
(Z0) shows genuine nonzero leftover data in every tested context; all
five scrub-bearing constructs (Z1, Z2, Z4, Z5, Z6) show 0 nonzero words,
consistently across every repeat, in K2/K3/K4.
**Limitation**: bounded to sm_89/CUDA 12.8 only; diagnostic variants
differ from the core-matrix cubins (added readback instrumentation); not
an Exposure Index or cross-kernel residue-recovery experiment; does not
establish absence of recoverable residue in any general sense.

**Safe wording**: "A bounded runtime check on the one available physical
GPU (RTX 4050 Laptop, sm_89) confirmed that 18 representative diagnostic
kernel templates launched, synchronized, and computed their ordinary
result correctly across 5 repeats each, and that the templates
implementing a zeroization construct left the sensitive buffer's word
count at zero immediately after their own cleanup point, while an
unscrubbed control retained nonzero data."

**Overclaiming wording to avoid**: "Runtime-validated across the corpus"
(only 18 of 2,016 configurations, and only on one architecture, with
modified/instrumented binaries) — and never describe this as an Exposure
Index or residue-recovery result; it is not that experiment.

---

## Master limitations list (applies to every claim above unless narrower text is given)

- No human-labelled binary ground truth exists in this study.
- No precision/recall/F1 claim.
- No human-validated false-accept-rate estimate.
- No complete CFG or post-dominance proof.
- Compile-only evidence for sm_75/80/86/87/90.
- `nvdisasm`/`cuobjdump` agreement is not independent semantic ground truth.
- Runtime diagnostic binaries differ from the core-corpus artifacts.
- Structural evidence does not itself prove absence of recoverable residue.
