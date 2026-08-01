# REVIEWER_RESPONSE_MAPPING.md

Source of the reviewer comments: `LMHC_overlap_audit_and_pivot_plan.md` §5
("Exact requests extracted from the CMT report"), the actual review report
for ICACSDF #931. These are real reviewer comments, not fabricated for
this document.

**Honesty note before the table**: the pivot plan (§4/§6 of the overlap
audit) envisioned a substantially larger paper than what this robustness
study actually executed. What was actually built and run is the
compiler-configuration survivability matrix and binary-only structural
detection (the audit's "D1" direction) — 2,016 compiled configurations,
cross-tool validation, a controlled case corpus, reproducibility, and a
bounded runtime check. The audit's "D2" direction (an ecosystem audit of
shipped third-party GPU libraries — cuBLAS, cuDNN, CUTLASS, FlashAttention,
etc.) and the cross-vendor compilation evidence (ROCm/HIP, oneAPI/clspv)
were **not executed**. Several reviewer comments below are therefore
honestly marked **not addressed by this round of work**, not stretched to
appear satisfied.

| # | Reviewer comment | Manuscript change | New experiment/discussion added | Exact result supporting the response | Location | Remaining limitation |
|---|---|---|---|---|---|---|
| R1.1 | "Interesting and relevant problem with a well-defined solution" (compliment) | None needed | — | — | — | — |
| R1.2 | "Some sections are quite lengthy; simplifying them would improve readability" | The pivot draft already cites LMHC's specification/proofs/verifier rather than re-deriving them (Section II is a summary, not a re-statement) | — | Structural, not a numeric result | Section II, throughout | Full copyedit pass for length/clarity not performed by this round of work |
| R1.3 | "Adding results on more GPU platforms" | Compiler-configuration matrix now spans 6 architecture targets (was 1 in the original evaluation) | Core compilation matrix (Section VI-A, Table I) | 6 arch targets x 2 toolkits x 4 opt levels x 7 constructs x 6 contexts = 2,016 configurations, 100% compiled | Table I, VII-A, VIII-A | **Partially addressed only**: 5 of 6 architecture targets are compile-and-disassemble evidence only; only sm_89 (one physical GPU) has any runtime evidence. Free-tier cloud GPU replication (the audit's "E8") was not attempted. |
| R1.4 | "Careful proof reading for language and formatting" | Blue placeholders replaced with real prose in this revision pass | — | — | Throughout | Full independent copyedit not performed |
| R2.1 | LMHC portability beyond NVIDIA, including AMD and Intel | None | None | None | — | **Not addressed by this round of work.** No ROCm/HIP or oneAPI/clspv compilation was attempted; ADDITIONAL_LIMITATIONS text (Section X) does not claim otherwise |
| R2.2 | Confidential-GPU attestation "is still theoretical" | N/A -- this pivot draft never included the confidential-GPU attestation section to begin with | — | — | — | Already structurally satisfied by scope, not by new work |
| R2.3 | "Study bigger workloads in reality" | Kernel-context axis reuses the three synthetic LMHC workloads (reduction, tiled GEMM, attention-style) plus three additional structural contexts (pure-dead allocation, fill-then-scrub, fill-use-then-scrub) | Core matrix's 6-context axis | 6 kernel contexts x everything else in the matrix | Table I, VI-A, VII-B | **Not a real-world-scale workload audit** (that is the audit's "D2" direction, cuBLAS/cuDNN/CUTLASS/etc., not executed). Do not claim this satisfies "bigger workloads" in the ecosystem-audit sense the reviewer likely meant. |
| R2.4 | "Ability of the verifier to operate with binary and kernel code" | This is now the paper's centerpiece section (VI-C, VII-D-E, VIII-C) | Cross-tool validation (nvdisasm vs. cuobjdump) + controlled N0-N12 corpus | 2016/2016 (100%) cross-tool category agreement; 47/55 controlled-case conformance; N6 boundary finding | Table II, VIII-C, Section IX | No human-validated ground truth exists; "operates with binary and kernel code" is demonstrated via agreement and conformance, not accuracy/precision/recall |
| R2.5 | "Consequences of leaving out implicit and alias-generated flows" | None -- Section X still states this as a caveat, not a measured quantity | None | None | Section X (limitation carried forward, unchanged in substance) | **Not addressed.** The audit's own guidance (§5, R2.5 row) explicitly asked for the *incidence* of this to be measured in an audited corpus (its "E6") -- that experiment was not run. This remains a stated limitation, not new evidence. |
| R2.6 | Comparison with other GPU memory-protection methods | None | None | None | Related-Work-gap placeholder (Section III) remains unresolved | **Not addressed.** No verified citations were sourced for this study; fabricating them was explicitly out of scope. This is the single most-repeated reviewer request across R2.6/R4.1 and remains open. |
| R4.1 | Differentiate LMHC from prior GPU memory protection; compare with prior secure zeroization, runtime clearing, hardware protections | None | None | None | Same placeholder as R2.6 | **Not addressed**, same reason as R2.6 |
| R4.2 | Validate on AMD, Intel, or other GPU platforms | None | None | None | Section X states vendor-neutrality remains unproven outside NVIDIA | **Not addressed.** No AMD/Intel compilation or hardware evidence exists anywhere in this study. |
| R4.3 | Side-channel attacks, malicious kernels, multi-tenant cloud, attacker models beyond residue recovery | None | None | None | Out of scope, not discussed in the current draft | **Not addressed.** This remains entirely outside this study's evidence. |
| R4.4 | More comprehensive baselines with additional security and performance metrics | New baseline set: 7 zeroization constructs (was 2: naive vs. hardened); new metrics: structural-category distribution, cross-tool agreement, controlled-case conformance, reproducibility rate | Full core matrix + cross-tool + controlled cases + reproducibility | Table I, Table II, `results/reproducibility_summary.json` | Tables I-IV, Sections VII-VIII | Per the audit's own explicit warning, this is intentionally **not** a re-run of the original EI/overhead baselines -- confirmed: no Exposure Index or overhead measurement appears anywhere in this study |
| R4.5 | Deployment limitations: compiler compatibility, toolchain integration, closed-source libraries, production scalability | Compiler/toolchain compatibility directly addressed (2 toolkit versions, 6 arch targets); closed-source-library scalability NOT addressed | Core matrix + cross-version comparison (VI-B, VIII-B) | 2016/2016 configurations compiled; identical outcomes across both toolkit versions | Table I, VIII-A, VIII-B | **Partially addressed.** The audit's "D2" ecosystem-scale closed-source-library audit was not executed; N11 (a single synthetic witness-mismatch probe) is not a substitute for it and is not presented as one. |
| R4.6 | Future applicability: heterogeneous ecosystems, confidential GPU environments, AI accelerators, adoption roadmap | Discussion section (IX) frames hardware confidential computing as the complement for scenarios binary-only checking cannot cover | Synthesis from RQ1-RQ3 results | N6 finding as the concrete boundary case | Section IX, VIII-D | Heterogeneous ecosystems (ROCm/oneAPI) **not addressed**; confidential GPU discussion is cited/deferred, not re-derived, consistent with the firewall against UPWIECON overlap |

## Summary

**Directly and substantively addressed by this round of work**: R2.4, R4.4
(centerpiece results: the 2,016-configuration matrix, 100% cross-tool
agreement, the controlled-case corpus, and the N6 deployment-boundary
finding).

**Partially addressed**: R1.3, R2.3, R4.5, R4.6 (real new evidence exists,
but does not fully satisfy what the reviewer likely intended — see each
row's limitation column for exactly what is still missing).

**Not addressed by this round of work, and not claimed to be**: R2.1,
R2.5, R2.6, R4.1, R4.2, R4.3. These require either a literature search this
study did not perform (R2.6/R4.1), cross-vendor compilation infrastructure
not built (R2.1/R4.2), a measured-incidence experiment not run (R2.5), or
are explicitly out of this study's scope (R4.3). None of these are papered
over with a vague gesture at generality — each row states plainly what
would need to happen next.
