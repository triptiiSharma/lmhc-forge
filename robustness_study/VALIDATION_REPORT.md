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

## Recommendation

Proceed to Phase 4. No repository safety issues, no unresolved tool-path
issues, no non-compiling source, no barrier-placement risk found, parser
and manual inspection agree on every case checked, dataset validates
cleanly, and the residual gaps above are specifically what Phase 4/5 will
continue to check — they are not reasons to stop, but they are not silently
declared closed either.
