# FINAL_RESULTS_LEDGER.md

Frozen evidence commit: `91f284f61a738ab24cd89bcf7dd50ebb6133647c` (branch
`upes-robustness-study`, working tree clean, all 49 tests passing, dataset
and artifact/hash validation PASS with 0 errors — see the final-freeze
verification in the commit immediately preceding this document).

Every row below is a single numerical claim, traceable to one source file
and one exact query/aggregation. Nothing in this document is estimated,
rounded beyond what the source data itself supports, or inferred. Use this
document, not memory or prose summaries, as the source of truth when
filling manuscript placeholders.

---

## Core corpus (2,016-artifact compilation matrix)

| Claim ID | Exact value | Denominator | Source file | Query / aggregation | Scope | Limitation | Safe wording | Prohibited overclaim |
|---|---|---|---|---|---|---|---|---|
| CORE-1 | 2016 | attempted configurations | `results/artifacts.jsonl` | count of records matching the 2,016-cell core-matrix `artifact_id` set (recomputed from `configs/core_matrix.json` via `expand_cells()`) | 7 constructs × 6 contexts × 4 opt levels × 6 archs × 2 toolkits | — | "2,016 attempted configurations" | "2,016 validated deployments" |
| CORE-2 | 2016 | of 2016 (100%) | `results/artifacts.jsonl` | `compile_status == "COMPILE_OK"` count | as CORE-1 | compile-only for 5 of 6 archs (no local hardware) | "100% (2016/2016) compiled successfully" | "100% deployment-ready" |
| CORE-3 | 0 | of 2016 | `results/artifacts.jsonl` | `compile_status == "COMPILE_FAILED"` count | as CORE-1 | — | "0 compile failures" | — |
| CORE-4 | 0 | of 2016 | `results/artifacts.jsonl` | `compile_status == "COMPILE_UNSUPPORTED"` count | as CORE-1 | — | "0 unsupported configurations" | — |
| CORE-5 | 576 | of 2016 | `results/classifications.csv` | `checker_verdict == "FULL_SIGNATURE"` count | as CORE-1 | structural category, not a security guarantee | "576/2016 artifacts classified FULL_SIGNATURE" | "576 secure binaries" |
| CORE-6 | 864 | of 2016 | `results/classifications.csv` | `checker_verdict == "PARTIAL_SIGNATURE"` count | as CORE-1 | — | "864/2016 artifacts classified PARTIAL_SIGNATURE" | — |
| CORE-7 | 96 | of 2016 | `results/classifications.csv` | `checker_verdict == "STORE_ONLY"` count | as CORE-1 | — | "96/2016 artifacts classified STORE_ONLY" | — |
| CORE-8 | 480 | of 2016 | `results/classifications.csv` | `checker_verdict == "ABSENT"` count | as CORE-1 | ABSENT is not itself evidence of a leak (task-wide framing rule) | "480/2016 artifacts classified ABSENT" | "480 leaking binaries" |
| CORE-9 | 0 | of 2016 | `results/classifications.csv` | `checker_verdict == "UNDECIDABLE"` count | as CORE-1 | — | "0 UNDECIDABLE outcomes in the core corpus" | — |
| CORE-10 | 288 | of 288 (100%) | `results/classifications.csv` | `checker_verdict == "FULL_SIGNATURE"` count where `zeroization_construct == "Z5"` | Z5 (hardened volatile+fence+barrier), all contexts/opts/archs/toolkits | see CORE-5's limitation; see controlled-case N6 for the static-vs-dynamic caveat | "the hardened construct Z5 retained the complete recognized structural signature in 288/288 (100%) of its tested configurations" | "Z5 guarantees zeroization on every execution" |
| CORE-11 | 288 | of 288 (100%) | `results/classifications.csv` | as CORE-10, `zeroization_construct == "Z6"` | Z6 (inline-PTX hardened), all contexts/opts/archs/toolkits | as CORE-10 | as CORE-10 for Z6 | as CORE-10 |
| CORE-12 | 313 | ms (median) | `results/artifacts.jsonl` | median of `compile_time_ms` across 2016 records | single WSL2 machine, single worker | not representative of other hardware/parallelism | "median compile time 313ms" | — |
| CORE-13 | 368 | ms (p95) | `results/artifacts.jsonl` | 95th percentile of `compile_time_ms` | as CORE-12 | as CORE-12 | "p95 compile time 368ms" | — |
| CORE-14 | 6 | ms (median) | `results/features.jsonl` | median of `analysis_time_ms` across 2016 records | as CORE-12 | — | "median feature-extraction+classification time 6ms" | — |
| CORE-15 | 13 | ms (p95) | `results/features.jsonl` | 95th percentile of `analysis_time_ms` | as CORE-12 | — | "p95 analysis time 13ms" | — |
| CORE-16 | 89 | MB | `VALIDATION_REPORT.md` Phase 4 section (`du -sh` on core-matrix `cubin`/`sass`/`logs`) | directory size of the 2,016-cubin + 2,016-SASS + 4,036-log core subset | as CORE-1 | — | "89MB total storage for the core matrix" | — |
| CORE-17 | 6 | architecture targets | `configs/core_matrix.json` | `archs` list length | sm_75/80/86/87/89/90 | only sm_89 has physical hardware in this study | "six architecture targets (sm_75/80/86/87/89/90)" | "validated on six GPU architectures" |
| CORE-18 | 2 | CUDA toolkits | `configs/core_matrix.json` | `toolkits` list length | CUDA 12.8 (V12.8.90/93), CUDA 13.3 (V13.3.29/33) | one machine, one WSL2 environment | "two CUDA toolkits (12.8, 13.3)" | — |
| CORE-19 | 1 | physical GPU | `AUDIT.md` §4.3 | `nvidia-smi` output, this study's environment | RTX 4050 Laptop GPU, Ada Lovelace, sm_89 | all other 5 arch targets are compile-only | "one physical GPU (RTX 4050 Laptop, sm_89) was available for runtime evidence" | "evaluated on three consumer GPUs" (that claim belongs to the *prior* LMHC work, not this study) |

## Cross-tool validation (nvdisasm vs. cuobjdump)

| Claim ID | Exact value | Denominator | Source file | Query / aggregation | Scope | Limitation | Safe wording | Prohibited overclaim |
|---|---|---|---|---|---|---|---|---|
| XT-1 | 96 | of 96 (100%) | `PHASE5_NVDISASM_PILOT.md` | pilot category-agreement count | stratified 96-artifact pilot, gate before full run | pilot, not the full corpus | "the 96-artifact stratified pilot showed 100% category agreement" | — |
| XT-2 | 2016 | of 2016 (100%) | `results/cross_tool_summary.json` | `category_agreement_count` | full core corpus | agreement between two parsers over the same binary, not independent semantic ground truth | "an independent disassembly and parsing path agreed with the primary analysis on 2016 of 2016 (100%) supported artifacts" | "independently verified," "confirmed correct by a second tool" |
| XT-3 | 8 | features, each 2016/2016 (100%) | `results/cross_tool_summary.json` | per-feature `feature_agreement` counts | as XT-2 | as XT-2 | "100% feature agreement across all 8 extracted security-relevant features" | — |
| XT-4 | 0 | disagreements | `results/cross_tool_mismatches.csv` | row count (header only) | as XT-2 | — | "zero disagreements occurred" | — |
| XT-5 | 0 | unsupported/schema-error records | `results/nvdisasm_features.jsonl` | `schema_error` present count | as XT-2 | — | "zero unsupported records" | — |
| XT-6 | 248 | ms (median) | `results/nvdisasm_features.jsonl` | median `analysis_time_ms` | as CORE-12 | — | "nvdisasm analysis median 248ms" | — |
| XT-7 | 379 | ms (p95) | `results/nvdisasm_features.jsonl` | 95th percentile `analysis_time_ms` | as CORE-12 | — | "nvdisasm analysis p95 379ms" | — |

## Controlled N0–N12 case corpus

| Claim ID | Exact value | Denominator | Source file | Query / aggregation | Scope | Limitation | Safe wording | Prohibited overclaim |
|---|---|---|---|---|---|---|---|---|
| CC-1 | 55 | configurations | `results/controlled_cases.jsonl` | total record count | 13 defined cases (N0–N12) | — | "55 configurations across 13 controlled structural cases" | — |
| CC-2 | 47 | of 55 | `results/controlled_cases.jsonl` | `conforms_to_intended == true` count | as CC-1 | conformance ≠ general accuracy | "47 of 55 configurations conformed to their intended structural condition" | "96% classifier accuracy" (denominator/framing error — see PHASE5_CONTROLLED_CASES.md's reconciliation) |
| CC-3 | 40 | of 48 | `results/controlled_cases.jsonl` | as CC-2, filtered to the 6 dedicated cases (N3–N8) | dedicated cases only | — | "40 of 48 dedicated-case configurations conformed" | — |
| CC-4 | 8 | of 8 | `results/controlled_cases.jsonl` | non-conforming count, all `case_id == "N6"` | N6 only | this is one case's full config set, not 8 independent failures | "the 8 non-conforming configurations are all case N6" | "8 classifier failures" |
| CC-5 | 1 | (single EXIT) | `results/controlled_cases.jsonl`, N6 records | `cuobjdump_features.exit_count` for N6 | N6, all 8 configs | — | "N6 compiled to a single-EXIT binary in all 8 tested configurations" | — |
| CC-6 | 8 | of 8 | `results/controlled_cases.jsonl` | N6 cross-tool agreement count | N6 | — | "both parsing paths agreed exactly on N6 in all 8 configurations" | — |
| CC-7 | 0 | `SYNTHETIC_TEST_ONLY` cases | `results/controlled_cases.jsonl` | count of records with that category | as CC-1 | — | "zero synthetic cases were pooled into this corpus" | — |
| CC-8 | 5 | direct / 0 indirect | `results/controlled_cases.jsonl`, N9 record | `cuobjdump_features` for N9 | N9 (sm_89, ptxas-O0) | one construct/context/config only | "N9 confirmed the real ptxas-O0 register-mediated zero-store encoding (5 indirect, 0 direct zero-stores)" | — |

## Reproducibility

| Claim ID | Exact value | Denominator | Source file | Query / aggregation | Scope | Limitation | Safe wording | Prohibited overclaim |
|---|---|---|---|---|---|---|---|---|
| REPRO-1 | 50 | configurations | `results/reproducibility_summary.json` | `n_configs` | deterministically stratified core subset | — | "50 stratified core configurations" | — |
| REPRO-2 | 150 | repeat compiles | `results/reproducibility_summary.json` | `n_total_repeat_compiles` | 50 × 3 | — | "150 additional compiles (3 per configuration)" | — |
| REPRO-3 | 150 | of 150 (100%) | `results/reproducibility_summary.json` + independent re-check | `classification_counts["byte-identical"]`, re-verified against cubin/SASS/nvdisasm-JSON SHA-256 directly | as REPRO-1 | single machine, single toolchain install | "150 of 150 repeat compiles were byte-identical to the original (cubin, raw SASS, and raw nvdisasm JSON hashes)" | "compilation is deterministic in general" |
| REPRO-4 | 0 | category/feature/instruction-stream differences | `results/reproducibility_summary.json` | `any_category_difference`, `any_feature_difference` | as REPRO-1 | — | "zero category or feature differences occurred" | — |

## Runtime sanity (bounded, sm_89 only)

| Claim ID | Exact value | Denominator | Source file | Query / aggregation | Scope | Limitation | Safe wording | Prohibited overclaim |
|---|---|---|---|---|---|---|---|---|
| RT-1 | 18 | diagnostic variants | `results/runtime_sanity.jsonl` | record count | 6 constructs (Z0,Z1,Z2,Z4,Z5,Z6) × 3 contexts (K2,K3,K4) | sm_89/CUDA 12.8 only; instrumented, not the core-matrix cubins | "18 diagnostic runtime variants" | "18 core-matrix artifacts validated at runtime" |
| RT-2 | 90 | launches | `results/runtime_sanity.jsonl` | 18 × 5 repeats | as RT-1 | — | "90 total launches (5 repeats each)" | — |
| RT-3 | 90 | of 90 (100%) | `results/runtime_sanity.jsonl` | launches with `launch=no error` | as RT-1 | — | "100% launch success" | — |
| RT-4 | 90 | of 90 (100%) | `results/runtime_sanity.jsonl` | launches with `sync=no error` | as RT-1 | — | "100% synchronization success" | — |
| RT-5 | 90 | of 90 (100%) | `results/runtime_sanity.jsonl` | launches with correct ordinary computation | as RT-1 | — | "100% ordinary computation correctness" | — |
| RT-6 | 1024 / 256 / 219 | nonzero words (K2/K3/K4) | `results/runtime_sanity_summary.csv` | Z0 (no-scrub control) nonzero-word counts, consistent across 5 repeats | Z0 only, sm_89 | positive-control result, not a residue-recovery claim | "the unscrubbed control retained nonzero data (1024/256/219 words in K2/K3/K4 respectively)" | — |
| RT-7 | 0 | nonzero words, all contexts/repeats | `results/runtime_sanity_summary.csv` | Z1,Z2,Z4,Z5,Z6 nonzero-word counts | 5 scrub constructs × 3 contexts × 5 repeats = 75 measurements | instrumented diagnostic template, not the exact core-matrix cubin; sm_89 only; does not establish absence of recoverable residue in general | "every scrub-bearing construct tested (including the plain, non-volatile Z1) left zero nonzero words after its cleanup point, consistently across all repeats and contexts tested" | "proven free of residue," "no recoverable residue exists" |

## Test suite and reproducible tooling

| Claim ID | Exact value | Denominator | Source file | Query / aggregation | Scope | Limitation | Safe wording | Prohibited overclaim |
|---|---|---|---|---|---|---|---|---|
| TEST-1 | 49 | of 49 (100%) | `python3 -m unittest discover -s tests` output | test pass count | this study's own test suite | — | "49/49 automated tests pass" | — |

---

## How to use this ledger

Every number that appears in the revised manuscript, its figures, or its
tables must trace to exactly one row above. If a number is needed that is
not in this ledger, it does not go in the manuscript — either compute it
from the frozen commit and add a row here first, or leave the placeholder
in place and note it in the final audit as unresolved.
