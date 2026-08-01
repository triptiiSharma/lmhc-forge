# PHASE5_BLINDING_AUDIT.md

> **STATUS: CANCELLED_BY_STUDY_DECISION.** The packet audited here was
> never used for human review — the study decided not to conduct human
> review. Archived for provenance only; see
> `phase5_review/README_NOT_USED.md`. This document remains a valid,
> useful record of the blinding methodology and the real leak it found and
> fixed, independent of whether the packet was ultimately used.

## Why this document exists

The first version of the Phase 5 packet (`results/validation_sample.csv`,
now deleted) was reviewed and found to leak heavily: it had
`zeroization_construct`, `kernel_context`, `target_arch`,
`ptxas_optimization`, `cuda_version`, and `selection_reason` as plain CSV
columns, and pointed reviewers at the **original, unmodified** SASS files —
whose kernel function names are literally `kernel_z5_k2`-style strings
containing the construct and context identifiers. A reviewer would have
known the answer before reading a single instruction. This document is the
record of correcting that and then actually verifying the fix, rather than
asserting it.

## What was audited

Every file under `phase5_review/reviewer_A/` and `phase5_review/reviewer_B/`
after `scripts/build_phase5_review_packet.py` ran: both reviewer CSVs and
all 254×2 sanitized `R####.sass` files (510 files total).

## Automated audit (`scripts/audit_blinding.py`)

Scans every reviewer-visible file for:
- `Z0`-`Z6` / `K0`-`K5` construct/context tokens
- every original `artifact_id` (from `artifacts.jsonl`, 2018 values checked
  against each file)
- checker verdict strings (`FULL_SIGNATURE`, `PARTIAL_SIGNATURE`,
  `STORE_ONLY` — auto-fail; `ABSENT`/`UNDECIDABLE` flagged separately as
  "ambiguous" since they're also plain English words that could
  legitimately appear, to avoid false-failing on innocuous text)
- raw architecture strings (`sm_75`/`80`/`86`/`87`/`89`/`90`)
- raw toolkit version strings (`cuda-12.8`/`cuda-13.3`)
- original `kernel_zN_kN`-style function names
- every original source/cubin/SASS filename

### First run: FAILED — found a real, second leak

```
Files scanned: 510
Violations: 232
  VIOLATION: phase5_review/reviewer_A/sass/R0002.sass: contains raw architecture string: ['89']
  ... (232 total, all the same root cause)
```

Investigation traced this to a **fourth line format** the initial
sanitizer didn't know about: `.target	sm_89` (a PTX-target directive line
that `cuobjdump` emits for some but not all compiled cubins — present in
roughly half the sampled files, absent in the others, which is presumably
a function of whether that particular fatbinary also embeds PTX). The
initial 3-line inspection (`code for smXX`, `Function :`, `.headerflags`)
had genuinely missed this because it was based on eyeballing only a
handful of example files rather than scanning every sampled file for every
line containing an architecture string.

**Fix verification, done properly this time**: before patching, every line
of every one of the 254 sampled *original* SASS files was scanned for any
line containing `sm_` or `SM<digit>`, and the results deduplicated by
structural pattern (digits normalized out). This produced an exhaustive
list of exactly 4 distinct line formats:
```
'\t.headerflags\t@"EF_CUDA_SMNN EF_CUDA_VIRTUAL_SM(EF_CUDA_SMNN)"'
'\t.headerflags\t@"EF_CUDA_TEXMODE_UNIFIED EF_CUDA_NNBIT_ADDRESS EF_CUDA_SMNN EF_CUDA_VIRTUAL_SM(EF_CUDA_SMNN)"'
'\t.target\tsm_NN'
'\tcode for sm_NN'
```
All 4 are now redacted by `sanitize_sass()`. The packet was regenerated
from scratch (old, leaky `phase5_review/` deleted first) and re-audited.

### Second run: PASS

```
Files scanned: 510
Violations: 0
Ambiguous hits (manual review, not auto-fail): 0

PASS -- no prohibited identifying strings found in any reviewer-visible file.
```

No `ABSENT`/`UNDECIDABLE`-as-plain-English false positives occurred either
— the sanitized instruction bodies don't happen to contain those words.

## Manual inspection (required in addition to the automated scan)

10 files spanning the full `R0001`–`R0254` range
(`R0001, R0025, R0050, R0075, R0100, R0125, R0150, R0175, R0200, R0254`)
read directly, not just grepped:

- All 10 correctly show `Function : R####` (matching their own filename,
  never the original mangled kernel name).
- All 10 correctly show `code for [ARCH-REDACTED]` and
  `.headerflags @"...EF_CUDA_SM[REDACTED]..."`.
- The 5 of these 10 that had a `.target` line correctly show
  `.target [ARCH-REDACTED]` (confirming the fix, not just the automated
  scan's word).
- Instruction bodies (opcodes, registers, hex encodings, addresses,
  predicates) are visibly untouched and structurally coherent —
  `LDC`, `IMAD.MOV.U32`, `ISETP.NE.U32.AND`, etc. — nothing looks
  truncated or corrupted by the redaction.
- File lengths vary naturally (55–487 lines) across the 10, consistent
  with genuinely different kernel bodies rather than some uniform
  artifact of the sanitizer.

## Structural checks (packet integrity, not text content)

- `results/phase5_private_mapping.csv` is **not** present anywhere under
  `phase5_review/` (confirmed by `find phase5_review -iname "*private*" -o
  -iname "*mapping*" -o -iname "*answer*"` → no results).
- `reviewer_A.csv`/`reviewer_B.csv` contain exactly the 5 required columns
  (`review_id, reviewer_label, confidence, evidence_notes,
  undecidable_reason`), no extra columns, no prepopulated labels.
- Both reviewer CSVs contain the identical set of 254 `review_id` values
  but in confirmed **different** row orders (independently seeded
  shuffles) — `same set: True`, `same order: False`.

## Result

**PASS.** Both the automated scan (0/510 violations after the fix) and
manual inspection (10/10 clean, redactions correct, instructions intact)
confirm the packet under `phase5_review/` contains no construct, context,
architecture, toolkit, checker-verdict, source-intent, or original-identity
information. Ready for human labeling once approved — not yet started.
