# STATUS: CANCELLED_BY_STUDY_DECISION — NOT USED AS EVIDENCE

The contents of this directory (`reviewer_A/`, `reviewer_B/`, and the
associated `results/phase5_blinded_manifest.csv` /
`results/phase5_private_mapping.csv`) were prepared as a genuinely-blinded
human manual-review packet for 254 sampled artifacts (252 from the 2,016
core matrix + 2 supplemental pilot flag-recheck cells), following the
methodology in `PHASE5_SAMPLE_DESIGN.md`, `PHASE5_REVIEW_CODEBOOK.md`, and
`PHASE5_BLINDING_AUDIT.md`.

**The study decision was made not to conduct human review.** No human ever
labeled any artifact in this packet. `reviewer_A.csv` and `reviewer_B.csv`
have empty `reviewer_label` columns, exactly as generated.

**These files are archived for provenance only.** They:
- must **not** be treated as completed evidence;
- must **not** be included in the paper's results;
- must **not** have Cohen's kappa, inter-reviewer agreement, precision,
  recall, F1, false-accept rate, or any human-ground-truth accuracy metric
  computed from them, because no ground truth was ever collected;
- are kept, not deleted, so the blinding methodology itself (including the
  real leak found and fixed — see `PHASE5_BLINDING_AUDIT.md`) remains
  auditable and citable as a record of process, separate from any claim
  about its results.

Phase 5 (independent validation) and Phase 6 (reliability reporting) were
replaced with an automated, reproducible validation approach: cross-tool
comparison (cuobjdump vs. `nvdisasm`), a controlled real-compiled-case
corpus, a compilation-reproducibility experiment, and a bounded runtime
sanity subset. See `PHASE5_CROSS_TOOL_VALIDATION.md`,
`PHASE5_CONTROLLED_CASES.md`, `PHASE5_REPRODUCIBILITY.md`,
`PHASE5_RUNTIME_SANITY.md`, and `PHASE6_STRUCTURAL_RELIABILITY.md` (once
produced) for what actually supports the paper's claims.
