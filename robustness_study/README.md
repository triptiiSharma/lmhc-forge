# UPES Robustness Study

A new, isolated experimental study supporting the revised paper *"Beyond
Specification: Evaluating the Compilation and Verification Boundaries of
Source-Level GPU Memory Zeroization."* Investigates zeroization
survivability and binary-only detectability across compiler/optimization/
kernel-context/toolkit-version/architecture-target conditions — distinct
from, and does not repeat, the original LMHC-Forge artifact's
residue-recovery / Exposure Index / overhead / tamper-ladder evidence (see
`../docs/RESULTS.md` for that prior work).

Lives entirely under this directory. Nothing outside `robustness_study/`
was modified. Branch: `upes-robustness-study`.

## Read these in order

1. **`AUDIT.md`** — Phase 0: repository + environment inventory (real
   command output, not assumed). Establishes that all compilation happens
   inside WSL2 (native Windows CUDA lacks a working host compiler) using two
   real toolkits, CUDA 12.8 and 13.3, both already validated end-to-end.
2. **`PLAN.md`** — Phase 1: the Z0-Z6 zeroization constructs, K0-K5 kernel
   contexts, the core/pilot/sensitivity matrices, classification categories,
   and — from a real feasibility compile, not a paper design — the
   multi-EXIT SASS finding that shapes the checker's design.
3. **`DATA_DICTIONARY.md`** — exact schema of every output file.
4. **`VALIDATION_REPORT.md`** — Phase 3 pilot gate result (PASS) and the
   real parser bug it caught before it could pollute the core matrix.

## Reproducing the pilot

All commands run inside WSL2 (`wsl.exe -d kali-linux -- bash -c '...'`)
from this directory:

```bash
python3 scripts/generate_kernels.py                                   # 42 .cu sources
python3 scripts/compile_matrix.py --config configs/pilot_matrix.json --workers 1
python3 scripts/extract_features.py
python3 scripts/validate_dataset.py
python3 scripts/aggregate_results.py
python3 -m unittest discover -s tests
```

## Layout

```
configs/     pilot/core/sensitivity/negative-case matrix definitions (JSON, not YAML —
             PyYAML isn't installed and this study doesn't install packages; see AUDIT.md)
kernels/     generated/  — 42 auto-generated .cu sources (7 constructs x 6 contexts)
             metadata/   — one JSON per source (hash, construct/context info)
             templates/  — not used; kernel_lib.py is the single source of truth instead
                           of separate template files, so the design in PLAN.md and the
                           generated corpus cannot drift apart
scripts/     kernel_lib.py       — Z0-Z6 / K0-K5 definitions (PLAN.md §1-2, code)
             generate_kernels.py — expands the 7x6 matrix into kernels/generated/
             compile_matrix.py   — invokes nvcc/ptxas/cuobjdump, writes artifacts.jsonl
             sass_parser.py      — generalized cuobjdump -sass parser (from dse_analysis.py
                                   / attester.py's splitting logic, extended)
             classify_binary.py  — structural-signature checker (renamed, generalized,
                                   honestly-scoped descendant of attester.py's _verify_scrub)
             extract_features.py — runs the parser+classifier over artifacts.jsonl
             validate_dataset.py — integrity checks (unique IDs, hashes, file existence)
             aggregate_results.py — outcome counts + scalability stats -> summary.json
tests/       unittest-based (no pytest installed); real SASS fixtures in tests/fixtures/,
             copied from the original repo's sass_dumps/ and from this study's own probes
artifacts/   cubin/ sass/ logs/ metadata/ — raw compiled evidence, never deleted
results/     artifacts.jsonl (source of truth) + features.jsonl + classifications.csv +
             summary.json — all derived, all re-derivable from artifacts/
dev_notes_phase1/  Phase-1 feasibility-compile evidence (kept, not deleted — it's
                   load-bearing justification for a design decision in PLAN.md §9)
```

## Status

Phase 0-3 complete (audit, design, scaffolding+tests, pilot gate — see
`VALIDATION_REPORT.md`). Phase 4 (2,016-artifact core matrix) has **not**
been run — pending review of the pilot results.
