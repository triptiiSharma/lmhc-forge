# FINAL_AUDIT.md

Final audit of the manuscript-revision phase (controlled-case
reconciliation, evidence freeze, results ledger, figures/tables, docx
revision, reviewer-response mapping). Produced after the revised
manuscript, figures, tables, results ledger, and reviewer-response mapping
— per instruction, execution stops here.

## Frozen evidence commit

`91f284f61a738ab24cd89bcf7dd50ebb6133647c` (branch `upes-robustness-study`).
Working tree clean at that commit; 49/49 tests passing; dataset and
artifact/hash validation PASS with 0 errors across `artifacts.jsonl` (2018
records), `controlled_cases.jsonl` (55), `reproducibility_runs.jsonl`
(50x3), `runtime_sanity.jsonl` (18), `nvdisasm_features.jsonl` (2016).
Controlled-case totals independently re-verified twice: 55 attempted = 55
compiled = 47 matched + 8 non-matched, exact.

## Placeholders replaced (UPES_Pivot_Draft_v1.docx, saved alongside the
original at `C:\Users\pandi\Downloads\UPES_Pivot_Draft_v1.docx` --
`UPES_Pivot_Draft_v0.docx` untouched)

21 targeted, individually-verified XML replacements, each anchored on a
unique substring of the original document and applied only after
confirming that anchor occurred exactly once:

1. Toolchain inventory (§VII-A)
2. Hardware/platform scope correction (§VII-A -- corrects a claim of
   three-GPU hardware reuse to accurately state one physical GPU + 5
   compile-only targets, since that is what this study actually did)
3. Experiment E1 results (§VII-C)
4. Table I sub-caption (§VII-C)
5. Figure 1 placeholder, sub-caption, and caption text (§VII-C)
6. Experiment E3 methodology correction (§VI-C -- removes a "compare "
   against ground truth" / "true/false positive rates" framing that this "
   study's methodology does not support, replaces with the cross-tool-"
   agreement + controlled-case-conformance framing actually used)
7. Experiment E2 results (§VII-D)
8. Experiment E3 description correction (§VII-E -- same correction as #6,
   applied to the experiment-design paragraph)
9. Experiment E3 results (§VII-E)
10. Table II caption and sub-caption (§VII-E -- retitled from "Binary-only
    detection accuracy" to "Controlled-case cross-tool comparison," since
    accuracy against ground truth was never computed)
11. Experiment E4 results (§VII-F -- states plainly that E4 was not
    executed)
12. Section VIII-A/B/C/D headline results (RQ1-RQ4)
13. Additional limitations (§X)
14. Stale "results not yet complete" intro to Section VIII
15. Table I data (rebuilt: 7 constructs x 6 outcome columns, real
    structural-category counts)
16. Table II data (rebuilt: 13 controlled cases x cuobjdump/nvdisasm
    verdict/agreement, real data)

Verified: 0 remaining `[TBD]` cells; document.xml is well-formed XML after
every edit; final document re-extracted and read through in full for
coherence.

## Placeholders still unresolved (left untouched, deliberately, not
missed)

1. **Manuscript status banner** (top of document) -- governance/ethics
   content (disclosure and withdrawal preconditions per the overlap
   audit), not an experimental placeholder. Not this task's decision to
   resolve.
2. **Related-work gap citations** (§III) -- requires a literature search
   for verified citations on GPU-specific SASS analysis tooling, ptxas
   optimization-stability literature, and prior compiler-configuration-
   matrix studies. No citation was fabricated; this remains open.
3. **Provenance and disclosure statement** (§XI) -- explicitly gated on
   the dual-submission disclosure/withdrawal process described in
   `LMHC_overlap_audit_and_pivot_plan.md` §9-10, which is outside this
   task's scope and not something this session can resolve or verify has
   happened.
4. **Reference list completion** (end of document) -- depends on
   resolving #2 and on the prior LMHC manuscript's final venue/DOI, also
   outside scope.

## Supported claims (see `PAPER_RESULTS.md` and `FINAL_RESULTS_LEDGER.md`
for the complete, sourced list)

- 2,016-configuration core compilation matrix, 100% compiled successfully.
- Structural-category distribution by construct, context, optimization
  level, architecture, and toolkit (Table I/II in this study's own
  documents; Table I in the revised manuscript).
- Hardened constructs (Z5, Z6) retained the complete recognized signature
  in 100% (288/288 each) of tested configurations.
- 100% cross-tool category and feature agreement across all 2,016
  artifacts (nvdisasm vs. cuobjdump), 0 disagreements.
- 47/55 controlled-case configurations conformed to their intended
  structural condition; the N6 finding as a central, evidenced
  deployment-boundary result.
- 150/150 byte-identical repeated compilations (50 configs x 3 repeats).
- Bounded runtime sanity check on sm_89: 90/90 launch/sync/computation
  success; 0 nonzero words for every scrub construct tested, in every
  context and repeat.

## Unsupported claims (explicitly refused everywhere in this work)

Classifier accuracy; false-accept or false-reject rate; human-validated
ground truth; universal LMHC/GPU-memory-hygiene security; absence of
recoverable residue in any general sense; cross-vendor (AMD/Intel)
validation of any kind; runtime validation of the five non-sm_89
architecture targets; "independently verified" language for the
cross-tool comparison (correct framing: agreement between two parsing
paths over the same binary).

## Remaining citation needs

- 3-5 verified citations for GPU-specific binary/SASS analysis tooling
  used by security researchers.
- Citations for prior work specifically on ptxas optimization stability
  or dead-code elimination in the CUDA back end, if any exists.
- Citations for prior compiler-configuration-matrix studies in adjacent
  domains (kernel hardening flags, stack-protector survivability).
- The GPU memory-protection comparison literature requested by R2.6/R4.1
  (Graviton, HIX, Honeycomb, Telekine, NVIDIA CC, driver/runtime clearing,
  CPU-side `memset_s`/`explicit_bzero` lineage, Common Criteria
  zeroization requirements) -- entirely unaddressed, flagged in
  `REVIEWER_RESPONSE_MAPPING.md`.
- Final bibliographic entry for the prior LMHC manuscript, pending its own
  venue/DOI resolution.

None of these were fabricated or estimated.

## Remaining manual edits

- The abstract (§Abstract, not a blue placeholder) still contains the
  sentence "Where results are not yet available we say so explicitly
  rather than estimating them" -- truthful (E4's absence is stated
  explicitly in §VII-F) but written in a fully prospective voice that now
  reads oddly given most sections have real headline results. Recommend a
  human pass to tighten the abstract to lead with the real numbers (2,016
  configurations, 100% cross-tool agreement, the N6 finding) rather than
  reframing it programmatically, since abstract-level rhetorical choices
  are an authorial judgment call outside a placeholder-replacement task.
- Figures were **not embedded as images** inside the docx (risk of
  corrupting the file via raw XML media-relationship manipulation without
  a proper library outweighed the benefit); the docx instead points to
  the separate `fig1_pipeline.pdf/png` through `fig4_n6_boundary_case.pdf/
  png` files (plus their source CSVs) in `robustness_study/figures/`. A
  human should embed these images in Word directly (Insert > Picture) —
  mechanical, low-risk, better done in the native tool than via scripted
  XML surgery.
- Table III and Table IV (per this task's own numbering, distinct from
  the draft's internal "Table I/Table II") were generated as data
  (`robustness_study/figures/table3_controlled_cases.csv`,
  `table4_cross_tool_repro_cost.csv`, and `TABLES.md`) but were not
  inserted into the docx as additional Word tables — the draft's own
  structure only had two table slots (which now carry the Table I/Table
  II content). Recommend a human decide where Table III/IV best fit in
  the revised section structure (likely §VII-E and §VII-D respectively)
  and insert them as native Word tables.
- Full independent copyedit (R1.4) was not performed beyond the targeted
  placeholder replacements.
- No `python-docx` or similar library was available in this environment;
  the revision was done via direct, individually-verified XML
  replacement. Recommend opening the file in Word and confirming visual
  layout (spacing, page breaks) looks correct, since raw XML editing
  cannot preview pagination.

## Git state

Robustness-study repository: clean at commit `91f284f` at the start of
this phase; figures, tables, `FINAL_RESULTS_LEDGER.md`,
`REVIEWER_RESPONSE_MAPPING.md`, and this audit are committed separately
(see final commit hash in the closing report). The manuscript itself
(`UPES_Pivot_Draft_v1.docx`) lives outside the git repository, alongside
the original draft in the user's own Downloads folder, per the task's
"do not alter the experimental data" instruction and because the
manuscript is not part of the git-tracked experimental artifact.
