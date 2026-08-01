# PHASE5_CONTROLLED_CASES.md

## Terminology (frozen)

This reports **controlled-case conformance** — whether the compiled binary
actually exhibits the intended structural condition — never "general
classifier accuracy." A case counts as conforming only when the *observed*
binary structure (from real compiled/disassembled output) matches the
intended condition; source intent alone is never sufficient, per the task's
explicit requirement.

## Master reconciliation table (N0–N12, every configuration accounted for)

This table resolves an inconsistency in earlier drafts of this study's
summary documents, which stated "47/48 dedicated configurations conformed"
— an error that conflated two different denominators (47 is the count of
conforming configurations out of the **full 55-configuration corpus**, not
out of the 48-configuration **dedicated subset**). The correct, verified
figures: **47/55 total configurations conform; 40/48 dedicated
configurations conform; 8/8 non-conforming configurations are case N6.**
Recomputed directly from `results/controlled_cases.jsonl` (not from prior
prose) and independently re-verified: 55 attempted = 55 compiled = 47
matched + 8 non-matched, exactly.

| Case | Type | Intended structural condition | Attempted | Compiled | Matched intended | cuobjdump outcome | nvdisasm outcome | Cross-tool agreement | Runtime execution | Interpretation | Limitation |
|---|---|---|---:|---:|---:|---|---|---|---|---|---|
| N0 | REAL_COMPILED_CASE (reference: Z0×K2) | scrub absent | 1 | 1 | 1 | ABSENT | ABSENT | 1/1 | Not executed as N0; underlying Z0×K2 construct has separate runtime evidence in Section H (Z0 shows genuine nonzero leftover data) | Negative control behaves as expected | None beyond the general study scope |
| N1 | REAL_COMPILED_CASE (reference: Z5×K2) | complete hardened recognized signature | 1 | 1 | 1 | FULL_SIGNATURE | FULL_SIGNATURE | 1/1 | Not executed as N1; underlying Z5×K2 construct has separate runtime evidence in Section H (0 nonzero words) | Hardened construct's signature recognized correctly | Static signature, not dynamic proof (see N6) |
| N2 | REAL_COMPILED_CASE (reference: Z2×K0) | zero stores present, fence absent | 1 | 1 | 1 | STORE_ONLY | STORE_ONLY | 1/1 | Not executed | Correctly distinguishes store-only from full signature | None beyond general scope |
| N3 | REAL_COMPILED_CASE (dedicated) | zero stores and fence present, final barrier absent | 8 | 8 | 8 | PARTIAL_SIGNATURE (8/8) | PARTIAL_SIGNATURE (8/8) | 8/8 | Not executed | Fence-without-barrier correctly distinguished from full signature | None beyond general scope |
| N4 | REAL_COMPILED_CASE (dedicated) | barrier present, scrub stores absent | 8 | 8 | 8 | ABSENT (8/8) | ABSENT (8/8) | 8/8 | Not executed | Barrier alone is not mistaken for scrub evidence | None beyond general scope |
| N5 | REAL_COMPILED_CASE (dedicated) | stores/fence/barrier present, intentionally invalid order | 8 | 8 | 8 | PARTIAL_SIGNATURE (8/8) | PARTIAL_SIGNATURE (8/8) | 8/8 | **Never executed** (intentionally invalid instruction ordering; not launched by design) | Ordering violation correctly downgrades the verdict | None beyond general scope |
| N6 | REAL_COMPILED_CASE (dedicated) | multi-source-exit, one cleanup path intentionally uncovered | 8 | 8 | **0** | FULL_SIGNATURE (8/8) | FULL_SIGNATURE (8/8) | 8/8 | **Never executed** (genuinely divergent early return before a fence/barrier sequence; not launched by design) | **Central deployment-boundary finding — see dedicated section below.** ptxas compiled the divergent early return to a single `EXIT` via predication; the checker correctly (per its documented scope) reports `FULL_SIGNATURE` from the static text, while half the threads never reach the scrub | Static structural presence is not proof of path-complete dynamic execution |
| N7 | REAL_COMPILED_CASE (dedicated) | intended scrub stores write a nonzero value | 8 | 8 | 8 | ABSENT (8/8) | ABSENT (8/8) | 8/8 | Not executed | Nonzero "scrub" correctly not mistaken for a zero-store | None beyond general scope |
| N8 | REAL_COMPILED_CASE (dedicated) | full recognized signature plus unrelated shared-memory stores | 8 | 8 | 8 | FULL_SIGNATURE (8/8) | FULL_SIGNATURE (8/8) | 8/8 | Not executed | Unrelated stores do not mask or corrupt the correct verdict | None beyond general scope |
| N9 | REAL_COMPILED_CASE (reference: Z5×K4, sm_89, O0) | valid register-mediated (indirect) zero encoding, real ptxas-O0 form | 1 | 1 | 1 | FULL_SIGNATURE, 5 indirect / 0 direct zero-stores confirmed | FULL_SIGNATURE | 1/1 | Not executed as N9; underlying Z5×K4 has separate runtime evidence in Section H | Confirms the real O0 register-mediated encoding is correctly recognized | Demonstrated for this construct/context only |
| N10 | REAL_COMPILED_CASE (reference: Z5×K4) | multi-exit control flow, intended cleanup structure present on all paths | 1 | 1 | 1 | FULL_SIGNATURE, exit_count=2 | FULL_SIGNATURE | 1/1 | Not executed | Multi-exit with genuine cleanup on both paths correctly recognized (contrast with N6) | None beyond general scope |
| N11 | REAL_BINARY_INPUT_CASE (reference: Z5×K2 cubin, wrong kernel name) | requested/nominated kernel absent from the binary | 1 | 1 | 1 | UNDECIDABLE | N/A — not run through the nvdisasm cross-tool pipeline; this case queries `classify_artifact()` directly against an existing real cubin with a deliberately wrong name, not a new compile | N/A (no nvdisasm comparison for this case; see outcome column) | Not executed (binary/witness-input scenario, not a launch scenario) | Correctly conservative: a witness/tooling mismatch never becomes a false pass or fail | A real, valid binary; the mismatch is in the query, not GPU-memory security |
| N12 | REAL_COMPILED_CASE (reference: Z1×K0) | pure-dead plain scrub eliminated by compilation | 1 | 1 | 1 | ABSENT (whole-allocation elimination) | ABSENT | 1/1 | Not executed | Reproduces the original repo's documented DSE finding | None beyond general scope |
| **TOTAL** | | | **55** | **55** | **47** | | | **54/54 compared (N11 excluded)** | | | |

**Reading the totals row**: 55 attempted, 55 compiled (100%), 47 matched
their intended condition, 8 did not (all N6). Cross-tool agreement is
100% on every configuration that was run through both parsing paths (54
of 55 — N11 is a single-tool witness-input test, not a compile-and-compare
case, so it has no nvdisasm counterpart by design, not by omission).

## Corpus composition — all 13 cases

| Case | Category | Description |
|---|---|---|
| N0 | REAL_COMPILED_CASE (reference) | scrub absent |
| N1 | REAL_COMPILED_CASE (reference) | complete hardened recognized signature |
| N2 | REAL_COMPILED_CASE (reference) | zero stores present, fence absent |
| N3 | REAL_COMPILED_CASE (dedicated) | zero stores and fence present, final barrier absent |
| N4 | REAL_COMPILED_CASE (dedicated) | barrier present, scrub stores absent |
| N5 | REAL_COMPILED_CASE (dedicated, NOT EXECUTED) | stores/fence/barrier present, intentionally invalid order |
| N6 | REAL_COMPILED_CASE (dedicated, NOT EXECUTED) | multi-source-exit, one cleanup path intentionally uncovered |
| N7 | REAL_COMPILED_CASE (dedicated) | intended scrub stores write a nonzero value |
| N8 | REAL_COMPILED_CASE (dedicated) | full recognized signature plus unrelated shared-memory stores |
| N9 | REAL_COMPILED_CASE (reference) | valid register-mediated zero encoding (real ptxas-O0 form) |
| N10 | REAL_COMPILED_CASE (reference) | multi-exit control flow, intended cleanup structure present |
| N11 | REAL_BINARY_INPUT_CASE | requested/nominated kernel absent from the binary |
| N12 | REAL_COMPILED_CASE (reference) | pure-dead plain scrub eliminated by compilation |

**No `SYNTHETIC_TEST_ONLY` cases were needed anywhere in this corpus** — every
case was realized as genuine compiled source or a genuine real-binary query.
(Synthetic SASS fixtures exist separately in `tests/` for parser-branch unit
tests only, e.g. `tests/test_classifier.py`'s boundary-condition tests —
those are never pooled with this corpus and are not referenced here.)

## Reference cases (N0, N1, N2, N9, N10, N12) — reused from the core matrix

No new compilation. Pulled by reference from `results/artifacts.jsonl` /
`results/features.jsonl` / `results/nvdisasm_features.jsonl`:

| Case | Reused artifact | cuobjdump verdict | Conforms |
|---|---|---|---|
| N0 | Z0×K2 | ABSENT | ✓ |
| N1 | Z5×K2 | FULL_SIGNATURE | ✓ |
| N2 | Z2×K0 | STORE_ONLY | ✓ |
| N9 | Z5×K4 (sm_89, O0) | FULL_SIGNATURE, **5 indirect / 0 direct zero-stores confirmed** | ✓ (demonstrates the real ptxas-O0 register-mediated encoding, as intended) |
| N10 | Z5×K4 | FULL_SIGNATURE, exit_count=2 (real multi-exit, cleanup present on both) | ✓ |
| N12 | Z1×K0 | ABSENT (whole-allocation elimination) | ✓ |

## N11 — REAL_BINARY_INPUT_CASE

A real, valid Z5×K2 cubin (genuinely `FULL_SIGNATURE`), queried through
`classify_artifact()` for a kernel name (`kernel_z5_k2_NONEXISTENT_NAME`)
that is not present in it — simulating a witness/tooling input error (a
nominated kernel name that doesn't match the binary), not a GPU-memory-
security scenario. Result: `UNDECIDABLE`, with detail explaining the named
kernel wasn't found among the kernels actually present. **Conforms** — this
is exactly the intended, correct, conservative behavior for a
binary/witness mismatch: never silently coerced to a pass or a fail.

## Dedicated cases (N3, N4, N5, N6, N7, N8)

Each compiled across 8 representative configurations: {CUDA 12.8, CUDA
13.3} × {ptxas O0, O3} × {sm_86, sm_89} = 48 new compiles, all against a
shared K2-shaped host context (fill→use→scrub) so the only variable under
test is the scrub construct itself, not the surrounding context.

**Real bug found and fixed while building this corpus** (not a checker
bug): the harness script's first version looked up cuobjdump results by
*demangled* name in a dict that `sass_parser.parse_sass_file()` actually
keys by *mangled* name — every one of the 48 dedicated-case cuobjdump
lookups came back `UNDECIDABLE` as a result. Traced directly against raw
SASS (`grep "Function :"` showed the real mangled name
`_Z9kernel_n3Pj`), confirmed `sass_parser.py`/`classify_binary.py`
themselves were correct and already covered by existing tests — the bug
was in the new harness (`build_controlled_cases.py`), fixed to match the
same lookup pattern `extract_features.py` already uses correctly
(`next(f for f in functions.values() if f.demangled_name == target)`).
Not a parser correction, no new parser regression test needed — but
recorded here in full per the task's transparency requirement, since it
did initially produce 48 wrong-looking results before being traced and
fixed.

### Results after the fix

| Case | Intended condition | Result (all 8 configs) | Cross-tool agreement | Conforms |
|---|---|---|---|---|
| N3 | fence present, barrier absent | `PARTIAL_SIGNATURE`, 8/8 | 8/8 | ✓ 8/8 |
| N4 | barrier present, stores absent | `ABSENT`, 8/8 | 8/8 | ✓ 8/8 |
| N5 | invalid order (not executed) | `PARTIAL_SIGNATURE`, 8/8 | 8/8 | ✓ 8/8 |
| N6 | one cleanup path uncovered (not executed) | `FULL_SIGNATURE`, 8/8 | 8/8 | **✗ 8/8 — see finding below, not a bug** |
| N7 | nonzero "scrub" value | `ABSENT`, 8/8 | 8/8 | ✓ 8/8 |
| N8 | full signature + unrelated stores | `FULL_SIGNATURE`, 8/8 | 8/8 | ✓ 8/8 |

40/48 marked "conforms" against my own pre-registered guess at N6's
category; the other 8 (all N6) are a genuine, important finding, not a
defect — detailed below.

## N6 — the most important finding in this corpus

**Source**: `if (tid < 16) return;` immediately before the scrub loop —
16 of 32 threads never reach the zero-store/fence/barrier sequence in the
*source*. This is a textbook "one cleanup path intentionally uncovered."

**What actually compiled**: `exit_count == 1`. Across all 8 configurations,
ptxas compiled the divergent early return down to a **single** `EXIT`
instruction, using predication for the tid<16 threads' skip rather than a
separate early-exit instruction — the same phenomenon this study already
documented in PLAN.md §9 (compiler-generated control flow doesn't map
1:1 onto source-level exit statements), just in the opposite direction
this time (source has 2 logical exit paths, SASS has 1 `EXIT`
instruction).

**What the checker concluded**: `FULL_SIGNATURE` — `store@166 < membar@174
< bar@176 < exit@177`, all correctly ordered relative to the single EXIT.
This is not a bug in the checker: by its own documented, honest scope (a
**structural-signature checker**, not a CFG/post-dominance prover, PLAN.md
§13), it correctly found that the signature textually precedes the only
`EXIT` instruction in the disassembly. Both cuobjdump-based and
nvdisasm-based paths agree exactly (8/8 cross-tool agreement) — this is
not a parsing discrepancy, it is the correct output of both independent
implementations applied to the actual compiled binary.

**Why this matters**: it is a **real, concrete, evidenced demonstration**
of exactly the limitation this study has stated in every design document
so far in the abstract — that `FULL_SIGNATURE` means the recognizable
structural sequence is present in the static text, **not** that every
dynamic thread/warp path executes it. N6 shows this is not a hypothetical
caveat: a genuinely compiled binary, where half the threads' cleanup path
is source-level uncovered, textually classifies as `FULL_SIGNATURE`
because ptxas's predication-based code generation doesn't produce a
second `EXIT` instruction for the checker to notice. No parser fix
addresses this — doing so would require real control-flow/predicate-
reachability analysis, which this study has repeatedly and explicitly
declined to claim (PLAN.md §13, this document's terminology section, and
`PHASE6_STRUCTURAL_RELIABILITY.md`'s limitations).

**N6 was never executed** (per instruction — a genuinely divergent early
return before a barrier/fence sequence is exactly the kind of pattern this
study avoids running on real hardware).

## Files produced

- `results/controlled_cases.jsonl` — 55 records (48 dedicated + 7 reference/N11)
- `results/controlled_case_summary.csv`
- `results/controlled_case_mismatches.csv` — 8 rows, all N6 (documented above,
  not unexplained)
- `kernels/controlled/{n3,n4,n5,n6,n7,n8}.cu` — dedicated sources
- `artifacts/controlled/{cubin,sass,nvdisasm_json,logs}/` — full raw evidence
