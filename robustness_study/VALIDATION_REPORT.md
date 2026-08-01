# VALIDATION_REPORT.md — Phase 3 Pilot Gate

**Decision: PASS. Proceed to Phase 4 (core matrix) pending your review of
this report and explicit go-ahead** (per the task brief's stop conditions
and the earlier agreement to pause before the core-matrix run).

## What was run

Pilot matrix (`configs/pilot_matrix.json`, PLAN.md §8): 5 constructs (Z0,
Z1, Z2, Z5, Z6) × 2 contexts (K0, K2) × 1 toolkit (CUDA 12.8) × 2 ptxas opt
levels (O0, O3) × 2 archs (sm_86, sm_89) = 40, plus 2 flag-recheck cells
(Z1×K0 and Z2×K3, each with `-maxrregcount=16 --use_fast_math`) =
**42 artifacts total**.

```
python3 scripts/generate_kernels.py
python3 scripts/compile_matrix.py --config configs/pilot_matrix.json --workers 1
python3 scripts/extract_features.py
python3 scripts/validate_dataset.py
python3 scripts/aggregate_results.py
python3 -m unittest discover -s tests
```

## Results

- **42/42 compiled successfully** (`COMPILE_OK`), 0 failures, 0 unsupported
  — expected, since the pilot only targets sm_86/sm_89, both supported by
  CUDA 12.8.
- **Compile time**: median ~305ms, range 291–770ms per artifact (single
  worker, first-compile-of-session outlier at 770ms likely nvcc/gcc-14 cold
  start). Cubin size: ~3–15 KB. This is the real basis for the runtime
  estimate below, not a guess.
- `validate_dataset.py`: **PASS**, no violations (unique IDs, all required
  fields present, no duplicate configs, all files exist, all hashes match,
  outcome counts reconcile).
- Unit tests: **36/36 pass** (`python3 -m unittest discover -s tests`),
  covering all 13 of the task's §22 required test areas.
- Checker verdict distribution across the 42 pilot artifacts: `ABSENT` 13,
  `PARTIAL_SIGNATURE` 9, `STORE_ONLY` 4, `FULL_SIGNATURE` 16 — all four
  non-compile-failure categories genuinely exercised, not a degenerate
  single-outcome run.

## The one real bug this pilot caught (exactly what the gate is for)

At `ptxas -O0`, every construct — **including the hardened Z5/Z6
constructs** — initially misclassified as `ABSENT`. Root cause: at `-O0`,
ptxas emits the zero literal via `MOV Rn, RZ` followed by `STS […], Rn`,
rather than folding it directly into `STS […], RZ` as it does at higher
optimization levels. The original parser (generalized from `dse_analysis.py`
/ `attester.py`, which were only ever exercised at `-O3`) only recognized
the direct-RZ form.

Fixed in `sass_parser.py` by adding a conservative, explicitly-scoped linear
register-tracking pass (last-write-wins per register, invalidated by any
subsequent write) — documented in the module docstring as exactly that, not
overclaimed as dataflow or CFG analysis. Manually verified against the raw
SASS of the specific artifact that exposed the bug
(`tests/fixtures/z2_k0_sm89_O0_indirect_zero_store.sass`): traced all 5
reported zero-stores back by hand to an unredefined `MOV Rn, RZ`, confirmed
correct. After the fix, Z5/Z6 correctly classify as `FULL_SIGNATURE` at both
O0 and O3, on both archs.

**This is exactly why the task specifies a pilot gate before the 2,016-cell
core run** — this bug would have silently produced ~250 false-`ABSENT`
records at O0 across the full core matrix if it had shipped unfixed.

## Manual inspection vs. automated extraction (task requirement)

- `docs/RESULTS.md`/`results/dse_differential.json` cross-check: the parser
  reproduces the original repo's published STS/MEMBAR/BAR.SYNC counts
  exactly (58 total STS, 29 zero-stores, 0/2 vs 1/3 MEMBAR/BAR for
  naive/hardened) — asserted as unit tests against the original fixture,
  not just eyeballed once.
- Z2×K0 (O0, sm_89) indirect-zero-store count: manually traced by hand
  against raw SASS (see above), matches parser output (5).
- Z5×K2 (O3, sm_89): manually confirmed `FULL_SIGNATURE` ordering
  (store@212 < membar@214 < bar@215 < exit@216) against the real
  disassembly.
- Multi-EXIT probe (`z2_k3_kernel`, `z6_k4_kernel`, from the Phase-1
  feasibility compile, PLAN.md §9): both kernels' real compiled-and-run
  behavior (empirically verified correct on RTX 4050 hardware — see
  PLAN.md §9) matches the classifier's multi-EXIT handling exactly.
- `COMPILE_UNSUPPORTED` path: exercised directly against a known-unsupported
  combination (`sm_70` on CUDA 13.3, confirmed unsupported in AUDIT.md §4.4)
  through the real `compile_one()` function (not just the raw subprocess) —
  correctly returns `COMPILE_UNSUPPORTED` / `unsupported_target`, matching
  `nvcc fatal : Unsupported gpu architecture 'sm_70'`.

## Known residual gaps (honestly reported, not blocking)

- `COMPILE_FAILED` (a genuine compile error, as opposed to an unsupported
  target) was not separately exercised in the pilot — the pilot's sources
  all compile validly by construction. Lower risk than the paths above
  (straightforward `returncode != 0, no unsupported-marker match` fallback),
  but not empirically exercised yet. Will be watched for in Phase 4;
  if a construct/context/arch combination that should logically fail never
  does, or a genuine failure gets misrouted to `COMPILE_UNSUPPORTED`, that's
  a concrete thing to check in Phase 5 review.
- `zero_shared_store_indirect_count` handling has only been manually
  verified on **one** real example. It's now covered by an automated
  regression test, but the register-tracking heuristic's behavior on more
  complex control flow (e.g., K3/K4/K5's internal branches, which the pilot
  only lightly exercises — pilot contexts are K0/K2 only) is unverified.
  Explicit action for Phase 4: re-run this same manual-trace exercise on at
  least one K3/K4/K5 artifact once the core matrix compiles them.
- Architecture-textual-encoding variance (task §11) has only been confirmed
  for sm_86/sm_89 (`BAR.SYNC.DEFER_BLOCKING`). sm_75/80/87/90 are unverified
  — a real risk given CUDA 13.3 dropped sm_70 and toolchains do change
  mnemonics across generations. This is exactly what negative-corpus case
  N9 (`configs/negative_cases.json`) exists to probe, but that corpus is
  design-only, not yet built. Flagging as something to watch during Phase 4
  rather than something already closed out.

## Runtime/storage estimate for the core matrix (measured, not guessed)

At the pilot's measured median compile time (~305ms) plus disassembly
overhead, 1 worker, the 2,016-cell core matrix is estimated at roughly
2016 × ~0.4s ≈ **13–15 minutes sequential**; with up to 4 workers (pending a
memory-pressure check under real parallel load, AUDIT.md §4.3's 7.6 GB WSL2
ceiling), plausibly 4–8 minutes. Disk: pilot artifacts (42) totaled a few
hundred KB including logs; extrapolated to 2,016 artifacts, well under
1 GB — not a constraint against the 114 GB free on `C:`.

## Test suite

```
Ran 36 tests in 0.095s
OK
```
Covers: kernel generation validity and uniqueness (42/42), matrix expansion
counts (pilot=42, core=2016), deterministic artifact IDs, SASS function
splitting, known-value cross-checks against the original repo's published
evidence, the -O0 indirect-zero-store fix, multi-EXIT real-fixture
classification, and all seven classification-boundary conditions including
the "parser failure must never become a silent pass" requirement.

## Recommendation (Phase 3)

Proceed to Phase 4. No repository safety issues, no unresolved tool-path
issues, no non-compiling source, no barrier-placement risk found, parser
and manual inspection agree on every case checked, dataset validates
cleanly, and the residual gaps above are specifically what Phase 4/5 will
continue to check — they are not reasons to stop, but they are not silently
declared closed either.

---

# Phase 4 — Core Compilation Matrix

**Decision: PASS. Dataset is validated and ready for Phase 5 (independent
manual review) — not yet started, pending your approval.**

## What was run

```
python3 scripts/compile_matrix.py --config configs/core_matrix.json --workers 1
python3 scripts/extract_features.py
python3 scripts/validate_dataset.py
python3 -m unittest discover -s tests
python3 scripts/aggregate_results.py
```

7 constructs × 6 contexts × 4 ptxas opt levels (O0-O3) × 6 arch targets
(sm_75/80/86/87/89/90) × 2 toolkits (CUDA 12.8, 13.3) = **2,016 core cells**.
Single worker throughout (WSL2's 7.6 GB RAM ceiling, AUDIT.md §4.3, made
this the conservative choice; observed compile time made it unnecessary to
add parallelism risk for a ~20-minute run).

## 1. Exit status and runtime

`compile_matrix.py` exited 0. Final summary line: `done in 1199.5s:
{'COMPILE_OK': 1976, 'COMPILE_FAILED': 0, 'COMPILE_UNSUPPORTED': 0}` — 1,976
is the *newly*-compiled count; 40 core cells were already satisfied by
resumability from the pilot (identical `artifact_id` — same construct,
context, opt, arch, toolkit). **Total wall-clock: ~20 minutes** for the
1,976 new compiles, consistent with the pilot's per-artifact measurement.

## 2. Configuration count

Confirmed by recomputing all 2,016 expected `artifact_id`s from
`configs/core_matrix.json` via `compile_matrix.expand_cells()` +
`artifact_id()` and diffing against `results/artifacts.jsonl`:
**2,016 expected, 2,016 present, 0 missing, 0 duplicates.** (The JSONL file
has 2,018 total records — the 2,016 core cells plus the 2 pilot-only
flag-recheck cells from Phase 3, which are intentionally outside the core
matrix and use distinct `artifact_id`s due to `extra_device_flags`.)

## 3-6. Feature extraction, dataset validation, test suite, aggregation

- `extract_features.py`: 1,976 newly processed (42 already done from the
  pilot), **2,018 total feature records**, in 14.1s.
- `validate_dataset.py`: **PASS** — 2,018 records, 0 violations.
- Independently re-verified file/hash integrity with a second, separately
  written script (not reusing `validate_dataset.py`'s code path): 0 missing
  files, 0 hash mismatches, across `cubin_path`, `source_file`, and
  `sass_path` for every record.
- Test suite: **38/38 pass** (36 from Phase 3 + 2 new, added during this
  phase — see §11 below for why).
- `aggregate_results.py`: wrote `results/summary.json`.

## 7-8. Cell representation and full reconciliation

All 2,016 core cells confirmed represented **exactly once** (§2). Full
breakdown, core-matrix-only (excludes the 2 non-core flag-recheck records):

| Dimension | Counts |
|---|---|
| `compile_status` | `COMPILE_OK`: 2016 |
| by construct | Z0-Z6: **288 each** (7 × 288 = 2016) |
| by context | K0-K5: **336 each** (6 × 336 = 2016) |
| by ptxas opt | O0-O3: **504 each** (4 × 504 = 2016) |
| by arch | sm_75/80/86/87/89/90: **336 each** (6 × 336 = 2016) |
| by toolkit | cuda-12.8 / cuda-13.3: **1008 each** |

Perfectly balanced on every axis — exactly what a correct full-factorial
expansion should produce, and itself a strong integrity signal (an
off-by-one in the generator or matrix expansion would have shown up here
immediately).

Artifact directory reconciliation: 2,018 `.cubin` files, 2,018 `.sass`
files, 4,036 log files on disk — all matching the 2,018 total JSONL records
exactly (2 stray orphan log files from an earlier standalone
`COMPILE_UNSUPPORTED` test, not referenced by any record, found and
removed during this check).

## 9. Full outcome report

**Compile status:** `COMPILE_OK` 2016, `COMPILE_FAILED` 0,
`COMPILE_UNSUPPORTED` 0 (core matrix only).

**Checker verdict (core matrix, 2016 cells):**

| Verdict | Count |
|---|---:|
| `FULL_SIGNATURE` | 576 |
| `PARTIAL_SIGNATURE` | 864 |
| `STORE_ONLY` | 96 |
| `ABSENT` | 480 |
| `UNDECIDABLE` | 0 |

(Sum = 2016. The all-record total in `results/summary.json`, 2018, includes
the 2 flag-recheck extras: +1 `ABSENT` for Z1×K0, +1 `PARTIAL_SIGNATURE` for
Z2×K3 — consistent with their core-matrix counterparts' verdicts.)

**By construct × context** (full 42-combination table, aggregated across
all opt/arch/toolkit — see PLAN.md-style construct roles):

```
Z0  K0-K5  ABSENT (48 each)                        — negative control, as designed
Z1  K0/K1  ABSENT (48/48)   K2-K5  PARTIAL (48 each) — whole-allocation elimination only where buffer is truly dead
Z2  K0/K1  STORE_ONLY (48/48)  K2-K5  PARTIAL (48 each) — no barrier only where context provides none
Z3  K0/K1  ABSENT (48/48)   K2-K5  PARTIAL (48 each) — same elimination pattern as Z1 (own barrier doesn't prevent DSE)
Z4  K0-K5  PARTIAL (48 each)                        — own barrier present even where Z2 lacked one
Z5  K0-K5  FULL_SIGNATURE (48 each)                 — 100% across every context/opt/arch/toolkit
Z6  K0-K5  FULL_SIGNATURE (48 each)                 — 100% across every context/opt/arch/toolkit
```
(48 = 4 opt × 6 arch × 2 toolkit; Z1×K0 and Z2×K3 show 49 due to the +1
flag-recheck cell each, already reconciled in §2.)

This is a clean, internally consistent, scientifically legible pattern —
not noise. It also reproduces the original repo's Module-3 finding (equal
STS presence, ordering is the differentiator for K2-shaped contexts) and
extends it, for the first time, across K3/K4/K5, all 6 architecture
targets, and both toolkits — genuinely new evidence, not a repeat.

**Failures / warnings / undecidable:** 0 `COMPILE_FAILED`, 0
`COMPILE_UNSUPPORTED`, 0 `UNDECIDABLE` — nothing to enumerate in these
categories for the core matrix. 842 records carry non-error
`parser_warnings` (722 are informational multi-EXIT notices; 192 are
"shared stores present but not recognized as zero-valued," which is the
*expected* signal on `ABSENT`-with-nonzero-stores records, e.g. Z0/Z1/Z3 at
K2-K5 where the context's own fill-phase data stores are present but
correctly not counted as zero-stores) — no warning indicates a parser
failure being silently swallowed.

**Timing/storage:**
- Compile time: median 313ms, p95 368ms.
- Analysis (parse+classify) time: median 6ms, p95 13ms.
- Total `artifacts/` directory: 89 MB (2,018 cubins + 2,018 SASS dumps +
  4,036 logs) — far under the 114 GB free on `C:`.
- Duplicate/missing-record count: **0 / 0** (§2, §8).

## 10. Manual inspection of the stratified sample

- **Construct × context table** (§9): systematically reviewed all 42
  combinations for anomalies — none found; the pattern is fully explicable
  from each construct's design (PLAN.md §1) and each context's structure
  (PLAN.md §2).
- **O0 in a previously-unverified context**: `Z5×K4` (tiled-GEMM), sm_89,
  O0, CUDA 12.8 — manually traced. Correctly found 5 indirect zero-stores
  from the scrub loop (`FULL_SIGNATURE`, store@334 < membar@342 < bar@344 <
  exit@364), **and** correctly did *not* count two earlier `STS` instructions
  that stored a register last touched by `MOV Rn,RZ` many lines earlier but
  since overwritten by a real `LDG.E` global-memory load (`b_tile`'s
  data-dependent fill) — direct confirmation that the register-tracking
  heuristic correctly invalidates on redefinition rather than
  over-triggering. This closes the residual gap flagged in the Phase-3
  report ("re-run this manual-trace exercise on a K3/K4/K5 artifact").
- **Multi-EXIT, `FULL_SIGNATURE`, `exit_count==2`** (`Z5×K4`, sm_75, O1,
  CUDA 12.8): manually confirmed store@120 < membar@124 < bar@125, both
  exits (128, 133) after the barrier → correctly `FULL_SIGNATURE`. Same
  structural pattern as the Phase-1 probe, now confirmed at a different
  arch/opt/toolkit.
- **Multi-EXIT, `exit_count==4`** (`Z1×K2`, sm_86, O3, CUDA 12.8): no
  `MEMBAR.SC.CTA` present at all, so classification short-circuits to
  `PARTIAL_SIGNATURE` before the multi-EXIT ordering branch is even reached
  — a legitimate, different code path, correctly exercised.
- **A real gap found and closed**: every real `multi_exit_scope_assumed=True`
  case in the entire 2,018-record corpus has `exit_count==2` — **zero**
  real cases with `exit_count>=3` ever reach `FULL_SIGNATURE`, because
  those higher exit counts only occur in constructs that never have both
  `MEMBAR.SC.CTA` and `BAR.SYNC` present, so they short-circuit earlier in
  `classify_kernel`. This means the "signature must hold across every
  relevant exit" branch of the multi-EXIT logic (PLAN.md §9) was **never
  exercised by real compiled data**, only by the Phase-1 synthetic-adjacent
  2-exit probe. Rather than leave this untested, I added two new synthetic
  unit tests (`test_full_signature_holds_across_three_exits`,
  `test_undecidable_when_signature_precedes_only_some_relevant_exits`,
  now part of the 38-test suite) that directly exercise the ≥3-exit
  ordering and the correct `UNDECIDABLE` fallback when the signature holds
  for some but not all relevant exits.

## 11. VALIDATION_REPORT.md

This section. Test suite grew from 36 to 38 (see §10's last bullet).

## Residual gaps carried into Phase 5 (still honestly open, not resolved by Phase 4)

- Architecture-textual-encoding variance beyond sm_86/89: sm_75/80/87/90
  compiled successfully and their SASS parses without parser warnings
  specific to those archs, which is *some* evidence the mnemonics are
  compatible, but no dedicated manual trace like the sm_86/89 ones has been
  done for those four targets yet. Phase 5's stratified sample (below)
  deliberately covers all six architectures for this reason.
- `COMPILE_FAILED` remains unexercised by real data — every core-matrix
  cell compiled successfully. This is itself a finding (no requested
  construct/context/opt/arch/toolkit combination in this design was
  actually invalid), not a gap in coverage of the *design space* tested,
  but it does mean the `COMPILE_FAILED` code path in `compile_matrix.py`
  has only synthetic/direct-call coverage, not a real trigger.
- The construct×context table (§9) is aggregate pattern-checking, not a
  substitute for Phase 5's independently-labeled ground truth — it shows
  the automated pipeline is *internally consistent*, not that its verdicts
  are *correct* against independent human judgment. That's exactly what
  Phase 5 exists to establish.

## Recommendation (Phase 4)

Dataset is complete, internally consistent, fully reconciled, and every
manually-checked sample agrees with the automated output. No repository
safety issues. Ready for Phase 5 (independent manual review) — proposed
sample design below — pending your approval to proceed.
