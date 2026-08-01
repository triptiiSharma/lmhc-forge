# PHASE5_CROSS_TOOL_VALIDATION.md

## Terminology (frozen)

`nvdisasm` is a second, **independent disassembly and parsing path** over
the same NVIDIA cubins the core matrix already compiled. It is **not**
independent semantic ground truth — both `nvdisasm` and `cuobjdump`
interpret the same binary encoding produced by the same `ptxas` backend.
This document reports **category agreement** and **feature agreement**
between the two paths. It never reports "accuracy," "precision/recall,"
or a "false-accept rate" — those require independent human ground truth,
which this study does not have (see `phase5_review/README_NOT_USED.md`).

## What was run

`scripts/run_cross_tool_validation.py --all-core`, over all 2,016 core
artifacts, using each artifact's **matching** toolkit's `nvdisasm` (CUDA
12.8 cubins → CUDA 12.8 `nvdisasm` V12.8.90; CUDA 13.3 cubins → CUDA 13.3
`nvdisasm` V13.3.29). Completed in 535.7s (~9 minutes). Preceded by the
96-artifact pilot gate (`PHASE5_NVDISASM_PILOT.md`, PASS, 96/96 category
and feature agreement) as required before this full run was permitted.

Raw JSON preserved for every artifact: `artifacts/nvdisasm_json/<artifact_id>.json`
(2,016 files). Full per-artifact records: `results/nvdisasm_features.jsonl`.

## Headline results

```
total_artifacts: 2016
category_agreement_count: 2016
category_agreement_pct: 100.0
disagreement_count: 0
unsupported_count: 0
nvdisasm_undecidable_count: 0
cuobjdump_undecidable_count: 0
```

Verdict distribution — **identical** between the two independent paths:
`PARTIAL_SIGNATURE` 864, `FULL_SIGNATURE` 576, `ABSENT` 480, `STORE_ONLY` 96.

**Feature-level agreement, all 8 security-relevant features, 2016/2016
(100%) each**: `zero_shared_store_direct_count`,
`zero_shared_store_indirect_count`, `zero_shared_store_count`,
`membar_count`, `membar_cta_count`, `bar_sync_count`, `exit_count`,
`branch_count`.

Note: total raw instruction counts were **not** required to match (and
were not compared) between the two tools, per the frozen plan — cuobjdump
and nvdisasm may expose pseudo-instructions or section-boundary padding
differently; only the semantically load-bearing feature set above was
compared.

## Agreement by every matrix dimension

| Dimension | Result |
|---|---|
| by construct | Z0-Z6: **288/288 each** |
| by context | K0-K5: **336/336 each** |
| by ptxas opt level | O0-O3: **504/504 each** |
| by architecture | sm_75/80/86/87/89/90: **336/336 each** |
| by toolkit | cuda-12.8/cuda-13.3: **1008/1008 each** |

100% agreement on every dimension, no exceptions.

## Disagreements

**Zero.** `results/cross_tool_mismatches.csv` exists (per the required
file list) and contains a header row with no data rows. There was nothing
to triage into output-format/schema/opcode-normalization/predicate-
parsing/register-tracking/architecture-encoding/function-selection/
unsupported-record/genuine-interpretation-difference categories, because
no disagreement of any kind occurred across all 2,016 artifacts.

This null result is reported as exactly that — a null result — not
inflated into a stronger claim than it supports. It means: for this
corpus, on this hardware/toolchain combination, the two independent
parsing paths extracted identical security-relevant facts from every
compiled binary. It does not mean the underlying structural-signature
policy is proven correct in any absolute sense (see Limitations in
`PHASE6_STRUCTURAL_RELIABILITY.md`).

## Timing

nvdisasm analysis time: median 248ms, p95 379ms (includes subprocess
invocation, JSON parse, and classification — comparable to cuobjdump's
own median 313ms compile time from Phase 4, reported separately since
these measure different things: nvdisasm analysis vs. nvcc/ptxas compile).

## Independent re-verification

Beyond the script's own tally, a separate standalone script
(`dev_notes_phase1/analyze_full_cross_tool.py`) re-derived every number in
this document directly from `results/nvdisasm_features.jsonl` and
`results/artifacts.jsonl` from scratch, confirming the same counts —
category agreement 2016/2016, feature agreement 2016/2016 on all 8
features, 0 schema errors, 0 UNDECIDABLE from either path, identical
per-dimension breakdowns.

## Files produced

- `results/nvdisasm_features.jsonl` — one record per artifact (2,016 rows)
- `results/cross_tool_comparison.csv` — full per-artifact comparison table
- `results/cross_tool_summary.json` — the headline numbers above
- `results/cross_tool_mismatches.csv` — empty (header only; 0 mismatches)
- `artifacts/nvdisasm_json/` — 2,016 raw JSON files, one per artifact
