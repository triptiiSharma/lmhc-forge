# PHASE5_REPRODUCIBILITY.md

## Normalization procedure — FROZEN BEFORE EXAMINING ANY RESULTS

This section is written and committed to before `scripts/reproducibility_run.py`
is executed or any comparison is examined, per the requirement not to tune
normalization after seeing results.

Three raw byte-level hashes are always computed and compared first, with no
normalization applied:
1. **Raw cubin SHA-256** — the compiled binary, byte-for-byte.
2. **Raw cuobjdump SASS SHA-256** — `cuobjdump -sass` stdout, byte-for-byte.
3. **Raw nvdisasm JSON SHA-256** — `nvdisasm -c -json` stdout, byte-for-byte.

If all three are byte-identical across all 3 repeats, the artifact is
**byte-identical** and no normalization step is needed for it. Normalization
exists only to distinguish, for artifacts that are *not* byte-identical,
whether the difference is meaningless presentation data or an actual
instruction-stream difference.

### What normalization removes (and only this)

From cuobjdump SASS text:
- The leading `/*hex-offset*/` address prefix on each instruction line.
- The entire second "control-code" line that follows every instruction
  line (pure encoding-bytes metadata, e.g. `/* 0x003fde0000000f00 */` on
  its own line) — dropped entirely, not kept in any form.
- Blank lines and leading/trailing whitespace.
- Local branch-label *numbering* is left untouched — cuobjdump's SASS
  output for this study's kernels does not use symbolic labels (branch
  targets are literal hex offsets like `` `(.L_x_0) ``, and the label
  definition itself, e.g. `.L_x_0:`, is a fixed textual anchor, not a
  generated counter that could vary between identical recompiles) — so
  nothing further is stripped here beyond the address/control-code lines.

From nvdisasm JSON:
- Each instruction is reduced to exactly the 4-tuple
  `(predicate, opcode, operands, control_flow_flag)` — this already
  excludes any positional/address metadata nvdisasm's JSON does not even
  expose at the instruction level (no per-instruction offset field exists
  in `-c -json` output, confirmed by inspection before writing this
  procedure).
- The function-level `start` and `length` metadata fields (byte offset
  and size within the section) are explicitly dropped — these are
  layout/presentation facts about where the function sits, not about what
  it does.
- Top-level metadata (`SchemaVersion`, `Producer` — which embeds the exact
  nvdisasm build string, a legitimate but non-semantic difference source —
  and toolkit-specific extra keys) is excluded from the normalized
  representation entirely.

### What normalization does NOT remove (frozen, non-negotiable)

- Opcode text (including any `.` suffix modifiers, e.g. `.SC.CTA`,
  `.DEFER_BLOCKING`).
- Predicate (`@P0`, `@!P0`, etc.).
- Operands, including immediate constants (e.g. `0x20` in
  `IADD3 R0,R0,0x20,RZ` is a genuine semantic immediate and is preserved
  exactly).
- Instruction ordering — the normalized representation is a **sequence**,
  compared element-by-element in order, never re-sorted or bagged.
- Branch/control-flow relationships — the `control-flow` flag is
  preserved; branch target *operands* are preserved as literal text
  exactly as normalization does not attempt symbolic branch-target
  resolution (this study's controlled/generated kernels are simple enough
  that recompiling byte-identical source through the identical compiler
  is expected to be genuinely deterministic even at the raw-address
  level — if that expectation turns out to be wrong for some artifact,
  it will show up as a normalized-sequence difference and be reported as
  such, not silently absorbed by an aggressive target-resolution
  normalization pass this study did not build).
- Fence (`MEMBAR`/`MEMBAR.SC.CTA`), barrier (`BAR.SYNC*`), and `EXIT`
  instructions — never touched by normalization, always compared exactly.

### Difference categories reported

For every one of the 3 repeats of each of the 50 selected configurations,
each comparison axis is classified into exactly one of:
- **byte-identical** — all 3 raw hashes match the original compile.
- **metadata/presentation-only difference** — raw hashes differ, but the
  normalized instruction sequence, extracted features, and structural
  category are all identical to the original.
- **instruction-stream difference** — the normalized sequence itself
  differs (a real semantic difference in what was compiled), independent
  of whether extracted features/category happen to still agree.
- **feature difference** — the normalized sequence may or may not match,
  but at least one extracted structural feature (zero-store counts,
  membar/bar/exit/branch counts) differs from the original compile.
- **category difference** — the final structural category
  (`FULL_SIGNATURE`/`PARTIAL_SIGNATURE`/`STORE_ONLY`/`ABSENT`/
  `UNDECIDABLE`) differs from the original compile. This is the most
  severe category and, if it ever occurs, is investigated individually
  before any aggregate claim is made.

## Selection

50 core configurations, deterministically stratified (seed `20260801`),
cycling through all 42 construct×context combinations until 50 unique
artifacts were drawn. Each compiled 3 additional times in clean, separate
output directories (`artifacts/reproducibility/<artifact_id>/repeat_{1,2,3}/`),
producing 150 new compilation runs.

## Results

```
n_configs: 50
n_repeats_per_config: 3
n_total_repeat_compiles: 150
classification_counts: {"byte-identical": 150}
any_category_difference: false
any_feature_difference: false
any_compile_failure_on_repeat: false
total elapsed: 130.3s
```

**Independently re-verified** (not just trusting the classification loop's
own tally): a separate check script confirmed, for every one of the 50
configs, that the cubin SHA-256, raw SASS SHA-256, and raw nvdisasm JSON
SHA-256 are each identical across all 3 repeats **and** identical to the
original core-matrix compile's cubin hash. **0 mismatches found.**

## Interpretation

Within the tested scope — a single machine, a single WSL2 environment, the
same two toolkit installs used throughout this study, no cross-machine or
cross-driver variation — compilation of these 50 stratified configurations
(spanning all 7 constructs, both toolkits, and a spread of contexts/opt
levels/archs) was **fully byte-deterministic**: every repeat produced a
cubin, SASS dump, and nvdisasm JSON dump identical down to the byte, not
merely equivalent after normalization. No instruction-stream difference,
no feature difference, and no category difference occurred anywhere in the
150 repeat compiles, so the normalization procedure above — while frozen
and documented — was not actually needed to explain any observed
difference, because there were no differences to explain.

This is evidence of **feature and structural-category reproducibility**
under the tested conditions. It is not evidence about reproducibility
across different machines, driver versions, or CUDA installations, which
this study did not test.
