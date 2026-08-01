# PHASE5_REVIEW_CODEBOOK.md

> **STATUS: CANCELLED_BY_STUDY_DECISION.** This codebook was never given to
> any reviewer — the study decided not to conduct human review. Archived
> for provenance only; see `phase5_review/README_NOT_USED.md`. The category
> definitions below (operational meaning of FULL_SIGNATURE/PARTIAL_SIGNATURE/
> STORE_ONLY/ABSENT/UNDECIDABLE) remain the authoritative definitions used
> by the automated checker and by `PHASE5_CROSS_TOOL_VALIDATION.md` and
> `PHASE5_CONTROLLED_CASES.md`, so this document is still referenced, just
> not as a human-review artifact.

Given verbatim to both reviewers before labeling starts. The wording here
must match `PLAN.md §5`'s classifier rules exactly, so that any
disagreement between a reviewer and the automated checker reflects a
genuine judgment call about the SASS, not a difference in what the
category names mean.

## What you are doing

For each `R####.sass` file, you are assigning **one structural category**
describing what zeroization-related evidence is present in that single
compiled GPU kernel's disassembly, and in what order. You are reading
static machine code — you are not executing it, and you are not being
asked to prove anything about runtime behavior on every possible input or
thread-divergence path.

**You are assigning a structural SASS category, not certifying universal
security or full LMHC policy compliance.** A `FULL_SIGNATURE` label means
the required recognizable instruction sequence is present within the scope
of static reading described below — it does **not** mean you have proven
full control-flow post-dominance, and it does **not** mean you have proven
the code is safe on every dynamic execution path. If you cannot establish
that from the static text with confidence, the correct answer is
`UNDECIDABLE`, not a forced guess in either direction.

## What "zero-valued store to shared memory" means

An instruction that writes a literal zero value into GPU shared memory
(`STS` — store-to-shared — where the value being stored is the constant
zero). In these files this appears either directly (the source operand is
the hardware zero register, printed `RZ`) or indirectly (a register was
most recently set to zero by a preceding move-from-zero instruction, and
has not been reassigned since). Both count as evidence of a zero-store.
**A store of a non-zero, arbitrary, or data-dependent value does not
count**, even if it targets the same address.

## What "fence" and "barrier" mean

- **Fence**: an instruction that enforces memory ordering between the
  zero-store(s) and later block-wide synchronization (in this
  architecture's disassembly, this appears as `MEMBAR.SC.CTA`).
- **Barrier**: a block-wide thread synchronization point (appears as
  `BAR.SYNC`, sometimes with a suffix like `.DEFER_BLOCKING` — that suffix
  is still the same barrier instruction, not a different one).

## The five categories

### `ABSENT`
No zero-valued store to shared memory is present anywhere in the kernel.
The kernel may contain other shared-memory stores (real, data-dependent
values), but none of them are zero.

*Neutral example:*
```
Function : R0000
    STS [R2], R5 ;      // R5 holds a data-dependent value, not zero
    EXIT ;
```
→ `ABSENT` — a shared store exists, but it is not a zero-store.

### `STORE_ONLY`
At least one zero-valued store to shared memory is present, but **neither**
a fence (`MEMBAR.SC.CTA`) **nor** a barrier (`BAR.SYNC`) appears anywhere
after it.

*Neutral example:*
```
Function : R0000
    STS [R2], RZ ;
    EXIT ;
```
→ `STORE_ONLY` — zero-store present, nothing enforces it completed before
the kernel exits.

### `PARTIAL_SIGNATURE`
A zero-valued store is present, and at least one of {fence, barrier} is
also present, but not both in the correct relative order (store, then
fence, then barrier), or one of the two is simply missing while the other
is present.

*Neutral examples:*
```
Function : R0000                       Function : R0000
    STS [R2], RZ ;                          BAR.SYNC ;
    BAR.SYNC ;                              MEMBAR.SC.CTA ;
    EXIT ;                                  STS [R2], RZ ;
                                             EXIT ;
→ PARTIAL_SIGNATURE                    → PARTIAL_SIGNATURE
  (barrier present, fence missing)       (right instructions, wrong order —
                                           store must come first)
```

### `FULL_SIGNATURE`
A zero-valued store, followed by a fence (`MEMBAR.SC.CTA`), followed by a
barrier (`BAR.SYNC`), all in that relative order, and all of that
sequence occurs before every kernel exit (`EXIT`) that you judge to be
reachable after the store. If the kernel has more than one `EXIT`
instruction (this happens even for source code with a single `return`
statement — the compiler can generate multiple predicated exits), use your
judgment about which exits are reachable after the store; if you cannot
tell, use `UNDECIDABLE` instead of guessing.

*Neutral example:*
```
Function : R0000
    STS [R2], RZ ;
    MEMBAR.SC.CTA ;
    BAR.SYNC ;
    EXIT ;
```
→ `FULL_SIGNATURE`.

**Important scope limitation, stated explicitly**: `FULL_SIGNATURE` means
this recognizable sequence is present in the static text, within the
supported static-analysis scope described above. It does **not** mean you
have performed full control-flow-graph post-dominance analysis, and it
does **not** mean you have proven the sequence executes on every dynamic
thread/warp path. If the file's control flow is complex enough that you
cannot confidently make that static judgment, label it `UNDECIDABLE`.

### `UNDECIDABLE`
You cannot confidently assign one of the above four categories from static
reading alone — for example, genuinely ambiguous multi-exit control flow
where you cannot tell which exit a given thread reaches, an unfamiliar
instruction encoding you don't recognize, or any other case where a
confident call isn't possible from the text in front of you.

**This is a valid, expected answer.** Do not force a case into `ABSENT`,
`STORE_ONLY`, `PARTIAL_SIGNATURE`, or `FULL_SIGNATURE` just to avoid using
it. If you use `UNDECIDABLE`, fill in `undecidable_reason` with a short
note on what specifically made it undecidable.

## Practical notes

- Every file you receive is disassembly from a real compiled GPU binary
  (`cuobjdump -sass` output), with only the kernel's function name and
  target-architecture strings redacted for blinding. Every instruction,
  predicate, address, branch, `EXIT`, store, fence, and barrier is exactly
  as compiled — nothing about the instruction content itself has been
  altered.
- Record your `confidence` (`high`/`medium`/`low`) for every row, even
  ones you're sure about — this is used later to help interpret any
  disagreements, not to second-guess your answer.
- Use `evidence_notes` to note the specific instruction(s) and their
  approximate location that led to your label — this is what makes
  disagreement resolution possible without re-reading everything from
  scratch.
- You will not be shown what the automated checker concluded, what
  construct or context this artifact came from, what architecture it was
  compiled for, or the other reviewer's answers, until after both of you
  have completed all 254 rows.
