# DATA_DICTIONARY.md

Documents the actual schema produced by the Phase-2 scripts, reconciled
against the task brief's §16 field list. Two JSONL files, joined by
`artifact_id`, rather than one denormalized record per artifact — a
deliberate implementation choice (compile-time facts vs. SASS-derived
facts are produced by different scripts at different times and it's
cheaper to resume/re-run one without re-running the other); the join is
performed for you by `extract_features.py` (writes `classifications.csv`)
and `aggregate_results.py` (writes `summary.json`).

## `results/artifacts.jsonl` — one record per attempted compile

Written by `compile_matrix.py`. One line per (construct, context, ptxas
opt level, target arch, toolkit, extra_device_flags) cell, appended as each
compile completes; resumable (a cell already present with a terminal
`compile_status` and existing referenced files is skipped on the next run).

| Field | Type | Nullable | Notes |
|---|---|---|---|
| `artifact_id` | str (16 hex chars) | no | `sha256(construct\|context\|opt\|arch\|toolkit_id\|extra_flags)[:16]` — deterministic, so re-runs are idempotent |
| `study_phase` | str | no | `"pilot"`, `"pilot_flag_recheck"`, `"core"`, `"sensitivity"`, etc. — taken from the config's `name` |
| `source_template` | str | no | `"{construct}x{context}"`, e.g. `"Z5xK2"` |
| `source_file` | str (path) | no | one of the 42 files under `kernels/generated/` |
| `source_hash_sha256` | str | no | of the generated `.cu` file, from `kernels/metadata/*.json` |
| `zeroization_construct` | str | no | `Z0`-`Z6` |
| `kernel_context` | str | no | `K0`-`K5` |
| `buffer_words` | int | no | `SBUF`'s `NWORDS` for this context (kernel_lib.py) |
| `block_threads` | int | no | this context's `BLOCK` |
| `toolkit_path` | str (path) | no | absolute path to the `nvcc` used |
| `cuda_version` | str | no | `"cuda-12.8"` or `"cuda-13.3"` (toolkit id, not a parsed version string) |
| `host_compiler` | str | no | `"gcc-14"` for both toolkits (AUDIT.md §4.4) |
| `host_optimization` | str | no | fixed `"-O3"` for this study (task §6: host opt is not the device-optimization axis) |
| `ptxas_optimization` | int (0-3) | no | passed as `-Xptxas -O{n}` |
| `target_arch` | str | no | `sm_75`/`sm_80`/`sm_86`/`sm_87`/`sm_89`/`sm_90` |
| `maxrregcount` | str | yes | only set on the two pilot flag-recheck cells |
| `extra_device_flags` | list[str] | no (`[]` default) | e.g. `["-maxrregcount=16", "--use_fast_math"]` |
| `compile_command` | str | no | full argv, space-joined, for exact reproduction |
| `compile_status` | str | no | `COMPILE_OK` \| `COMPILE_FAILED` \| `COMPILE_UNSUPPORTED` |
| `compile_return_code` | int | yes (null on timeout) | raw nvcc exit code |
| `compile_time_ms` | int | no | wall-clock for the nvcc subprocess only |
| `compile_stdout_path` / `compile_stderr_path` | str (path) | no | full logs, always saved even on success (warnings) |
| `cubin_path` / `cubin_hash_sha256` / `cubin_size_bytes` | — | only on `COMPILE_OK` | |
| `sass_path` / `sass_size_bytes` | — | only if disassembly succeeded | |
| `kernel_mangled_name` | str | **currently always null here** | the mangled name is only known after disassembly; it lives in `features.jsonl`'s `kernel_features.mangled_name` instead. Documented gap, not a silent omission. |
| `kernel_demangled_name` | str | no | deterministic from the generator (`kernel_{z}_{k}`), matches `KernelFeatures.demangled_name` after Itanium de-mangling |
| `source_intent_label` | str | no | == `zeroization_construct`; kept as a separate field per task §15's non-circularity requirement even though redundant today, because Phase 6's negative corpus (`configs/negative_cases.json`) will populate it independently of `zeroization_construct` for dedicated (non-core-matrix) sources |
| `binary_ground_truth` | str | yes | **only ever set by Phase 5 manual review** — never derived automatically anywhere in this codebase |
| `checker_verdict` | str | yes | **not populated in this file** — lives in `features.jsonl`/`classifications.csv`, joined by `artifact_id`. Left here as a schema placeholder per the task's field list. |
| `manual_review_status` / `manual_reviewer_1` / `manual_reviewer_2` / `manual_disagreement` | — | yes | Phase 5 fields, all null until then |
| `error_category` / `error_message` | str | yes (only on failure) | `"timeout"`, `"unsupported_target"`, or `"compile_error"` |
| `timestamp_utc` / `git_commit` | str | no | provenance |
| `notes` | list[str] | no (`[]` default) | nvcc warnings, cuobjdump stderr, free-text |

## `results/features.jsonl` — one record per successfully-disassembled artifact

Written by `extract_features.py`. Joined to `artifacts.jsonl` by `artifact_id`.

| Field | Type | Notes |
|---|---|---|
| `artifact_id` | str | join key |
| `kernel_features` | object \| null | `KernelFeatures.to_dict()` from `sass_parser.py`, or null if the named kernel wasn't found |
| `checker_verdict` | str | one of `classify_binary.py`'s `VALID_CATEGORIES`, or `UNDECIDABLE` if the kernel was missing |
| `checker_detail` | str | human-readable justification, always present |
| `multi_exit_scope_assumed` | bool | true when the classification relied on the multi-EXIT handling in PLAN.md §9 rather than the simple single-EXIT scoping |
| `analysis_time_ms` | int | parse+classify wall-clock for this one artifact |

### `kernel_features` object fields (from `sass_parser.KernelFeatures`)

`mangled_name`, `demangled_name`, `instruction_count`, `shared_store_count`
(any STS, any value), `zero_shared_store_count` (= direct + indirect, see
below), `zero_shared_store_direct_count` (`STS […], RZ` literally),
`zero_shared_store_indirect_count` (`MOV Rn,RZ` … `STS […], Rn` — the -O0
pattern found and fixed during Phase 3, PLAN.md §9), `membar_count` (any
`MEMBAR`), `membar_cta_count` (`MEMBAR.SC.CTA` specifically), `bar_sync_count`
(`BAR.SYNC`, matches sm_89's `BAR.SYNC.DEFER_BLOCKING` variant too — verified
against a real fixture), `exit_count`, `branch_count`, `exit_line_indices`,
`last_zero_store_index`, `first_membar_cta_index`, `last_bar_sync_index`,
`parser_warnings` (list[str] — never silently dropped), `unsupported_encodings`.

## `results/classifications.csv`

Flat projection of `artifacts.jsonl` × `features.jsonl`, one row per
COMPILE_OK artifact: `artifact_id, zeroization_construct, kernel_context,
target_arch, cuda_version, ptxas_optimization, source_intent_label,
checker_verdict, checker_detail, multi_exit_scope_assumed,
analysis_time_ms`. Rewritten in full on every `extract_features.py` run
(cheap — it's a projection, not a source of truth).

## `results/summary.json`

Written by `aggregate_results.py`: outcome counts by compile_status ×
construct/context/arch/toolkit, checker-verdict counts and breakdowns, and
scalability stats (`stat_block`: n/min/max/mean/median) for compile time,
cubin size, SASS size, and analysis time. Read-only reduction — safe to
re-run any time, never mutates `artifacts.jsonl`/`features.jsonl`.

## Classification categories (`classify_binary.VALID_CATEGORIES`)

`FULL_SIGNATURE`, `STORE_ONLY`, `PARTIAL_SIGNATURE`, `ABSENT`,
`UNDECIDABLE` — exact decision logic in PLAN.md §5, code in
`scripts/classify_binary.py`. `COMPILE_FAILED`/`COMPILE_UNSUPPORTED` are
`compile_status` values, not checker verdicts — a cell that didn't compile
is never fed to the classifier at all.
