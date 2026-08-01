# PHASE6_STRUCTURAL_RELIABILITY.md

## Scope and terminology (frozen throughout this study)

The primary tool is a **binary structural-signature checker**. It is not:
a complete CFG verifier; a proof of post-dominance; a proof of dynamic
execution on every path; a proof that memory residue cannot be recovered;
or a human-validated classifier. Every number in this document is
automated or controlled evidence. **No human-labelled ground truth was
collected** (the prepared blinded packet in `phase5_review/` was archived
unused, `phase5_review/README_NOT_USED.md`) — this document therefore
contains no accuracy, precision/recall/F1, or false-accept-rate claim.

---

## 1. Core corpus

- **2,016 attempted configurations**: 7 zeroization constructs × 6 kernel
  contexts × 4 ptxas optimization levels (O0–O3) × 6 architecture targets
  (sm_75/80/86/87/89/90) × 2 CUDA toolkits (12.8, 13.3).
- **Compilation outcomes**: `COMPILE_OK` 2016/2016 (100%), `COMPILE_FAILED`
  0, `COMPILE_UNSUPPORTED` 0.
- **Structural-category distribution**: `FULL_SIGNATURE` 576,
  `PARTIAL_SIGNATURE` 864, `STORE_ONLY` 96, `ABSENT` 480, `UNDECIDABLE` 0.
- **Timing**: compile time median 313ms / p95 368ms; feature-extraction+
  classification time median 6ms / p95 13ms.
- **Storage**: 89MB for the core matrix (2,016 cubins + 2,016 SASS dumps +
  4,036 logs).
- **By every matrix dimension** (all perfectly balanced — 288/construct,
  336/context, 504/opt-level, 336/arch, 1008/toolkit): full breakdown in
  `VALIDATION_REPORT.md`'s Phase 4 section, construct×context table there
  showing a fully coherent pattern (Z0 → `ABSENT` universally; Z1/Z3 →
  `ABSENT` only where the buffer is truly dead, matching the original
  repo's whole-allocation-elimination finding; Z2 → `STORE_ONLY` only
  where no barrier exists in context; Z4 → `PARTIAL_SIGNATURE` everywhere;
  **Z5/Z6 → `FULL_SIGNATURE` in 100% of 288 cells each**).

## 2. Cross-tool validation

- **96-artifact stratified pilot**: 100% category agreement, 100%
  feature agreement (8 features), gate PASSED
  (`PHASE5_NVDISASM_PILOT.md`).
- **Full 2,016-artifact pass**: **category agreement 2016/2016 (100%)**;
  **feature agreement 2016/2016 (100%) on all 8 security-relevant
  features** (`zero_shared_store_direct_count`,
  `zero_shared_store_indirect_count`, `zero_shared_store_count`,
  `membar_count`, `membar_cta_count`, `bar_sync_count`, `exit_count`,
  `branch_count`); **disagreements: 0**; unsupported cases: 0; `nvdisasm`
  `UNDECIDABLE`: 0; `cuobjdump` `UNDECIDABLE`: 0. Agreement is 100% across
  every construct/context/opt-level/architecture/toolkit breakdown
  individually, not just in aggregate (`PHASE5_CROSS_TOOL_VALIDATION.md`).
- **Causes of disagreement**: none occurred; nothing to categorize.
- Reiterated: this is agreement between two parsing paths over the same
  cubins, not independent semantic ground truth.

## 3. Controlled cases

- **13 cases (N0–N12), all `REAL_COMPILED_CASE` or `REAL_BINARY_INPUT_CASE`
  — zero `SYNTHETIC_TEST_ONLY` cases pooled into this corpus.**
- 6 reused directly from the core matrix (N0, N1, N2, N9, N10, N12); 1
  (N11) a real-binary/nonexistent-kernel-name query; 6 dedicated new
  sources (N3–N8), each compiled across 2 toolkits × 2 opt levels × 2
  archs = 48 configurations.
- **A real bug was found and fixed during construction** — the new
  harness script looked up cuobjdump results by the wrong dictionary key
  (mangled vs. demangled name); traced to raw SASS, confirmed the tested
  `sass_parser.py`/`classify_binary.py` were not at fault, fixed in the
  harness. Documented in full in `PHASE5_CONTROLLED_CASES.md`.
- **Expected structural condition vs. observed**: of the 55 total
  configurations (48 dedicated + 6 core-matrix references + 1 N11),
  **47/55 conform** exactly to their intended condition. Broken out by
  type: **40/48 dedicated configurations** conform (all of N3, N4, N5,
  N7, N8 — 5 cases × 8 configs each — with 8/8 cross-tool agreement per
  case), and **7/7 reference/N11 configurations** conform. The remaining
  **8/55 non-conforming configurations are all case N6** (its full 8
  configurations), not a defect spread across multiple cases. **N6 is a
  genuine, evidenced finding, not a defect**: source has a
  divergent early return leaving one cleanup path uncovered for half the
  threads, but ptxas compiled this to a **single** `EXIT` via predication
  — both parsing paths correctly and identically report `FULL_SIGNATURE`
  from the static text, while the real per-thread behavior still skips
  the scrub for the returning threads. This is concrete, compiled,
  disassembled evidence for exactly the limitation this study states
  everywhere else: **a `FULL_SIGNATURE` verdict is not proof every
  dynamic thread path executes the scrub.**

## 4. Reproducibility

- 50 deterministically-stratified core configurations, each compiled 3
  additional times (150 new compiles).
- **Cubin-byte stability**: 150/150 byte-identical cubin SHA-256 (vs. the
  original core-matrix compile and vs. each other).
- **Normalized-SASS / nvdisasm-JSON stability**: not needed — raw
  SASS SHA-256 and raw nvdisasm JSON SHA-256 were also 150/150
  byte-identical, independently re-verified by a separate check script.
- **Feature stability**: 150/150 identical.
- **Category stability**: 150/150 identical.
- Normalization procedure frozen in writing (`PHASE5_REPRODUCIBILITY.md`)
  *before* any repeat was compiled — it went unused because no
  divergence occurred to normalize.
- Scope: single machine, single WSL2 environment, the same two toolkit
  installs used throughout. Not evidence about cross-machine or
  cross-driver reproducibility.

## 5. Runtime sanity

- 18 diagnostic variants (6 constructs × K2/K3/K4; **K0 excluded** to
  avoid invalidating its pure-dead-allocation property) × 5 repeats = 90
  launches, **bounded to sm_89 (RTX 4050 Laptop GPU) / CUDA 12.8 only**.
- 90/90 launches succeeded, 90/90 syncs succeeded, 90/90 ordinary
  computations matched the expected value exactly.
- Z0 (no scrub) shows genuinely nonzero leftover data in every context
  (positive control confirming the methodology). All five scrub
  constructs (Z1, Z2, Z4, Z5, Z6 — including the **plain, non-volatile**
  Z1) show **0 nonzero words**, consistently across all 5 repeats, in
  every one of K2/K3/K4.
- **Instrumentation limitation, stated prominently**: these are diagnostic
  templates with an added readback, not the original core-matrix cubins.
  **No cross-target generalization** — sm_75/80/86/87/90 remain
  compile-target evidence only.

## 6. Limitations (retained explicitly, not softened)

- **No human-labelled binary ground truth exists anywhere in this study.**
  The prepared, genuinely-blinded 254-artifact review packet
  (`phase5_review/`) was never labeled — archived for provenance only.
- **No precision/recall/F1 claim** is made or computable without human
  ground truth.
- **No human-validated false-accept-rate estimate** exists.
- **No complete CFG or post-dominance proof** exists or is claimed — the
  primary tool remains a structural-signature checker (task §13's
  framing, maintained throughout). Section 3's N6 finding is direct,
  compiled evidence of exactly this limitation's real-world relevance.
- **Compile-only evidence for sm_75/80/86/87/90** — only sm_89 (the one
  physical GPU available) has any runtime evidence at all, and only for
  the 18 diagnostic templates in Section 5, not the core-matrix cubins
  themselves.
- **Diagnostic runtime artifacts differ from the core-corpus artifacts**
  (readback instrumentation changes what was compiled).
- Structural evidence (any of Sections 1–5) does not itself establish
  absence of recoverable residue in any general sense — this study did
  not run a residue-recovery/Exposure-Index experiment; that evidence, if
  relevant, lives in the original LMHC-Forge artifact, and is not
  repeated or extended here.
- Cross-tool agreement (Section 2) is agreement between two disassembly
  paths over the *same* compiled binary, not independent semantic ground
  truth — both tools ultimately reflect what `ptxas` actually emitted.

## Files produced

- `results/phase6_structural_reliability.json` — machine-readable version
  of the numbers above
- `results/phase6_supported_claims.csv`
- `results/phase6_unsupported_claims.csv`
- `PAPER_RESULTS.md` — per-claim data source, scope, limitation, and safe
  vs. overclaiming wording
