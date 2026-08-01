#!/usr/bin/env python3
"""
sass_parser.py — generalized cuobjdump -sass parser.

Direct generalization of the function-splitting logic in src/dse_analysis.py
and src/attester.py (AUDIT.md §2): same Itanium-length-prefix demangle, same
"Function :" section splitter. Extended per PLAN.md §11/§7 to extract the
full feature set for arbitrary kernel names (not hardcoded to the original
3-kernel comparison) and to be explicit about architecture-textual variants
observed in this study (e.g. sm_89's `BAR.SYNC.DEFER_BLOCKING`, confirmed by
the Phase-1 feasibility probe in dev_notes_phase1/).

Does NOT decide PASS/FAIL — that's classify_binary.py's job, operating on
the features this module extracts. Kept deliberately dumb and auditable:
every count is re-derivable by grepping the raw SASS by hand.
"""

import re
from dataclasses import dataclass, field, asdict


_FUNC_RE      = re.compile(r'Function\s*:\s*(\S+)')
_STS_RE       = re.compile(r'\bSTS\b')                 # any shared store (any width)
_STS_RZ_RE    = re.compile(r'\bSTS\b[^;]*\bRZ\b')       # zero-valued shared store (RZ = zero register, direct)
_STS_REG_RE   = re.compile(r'\bSTS\s*\[[^]]*\]\s*,\s*(R\d+)\s*;')  # STS storing a general-purpose register
_MOV_ZERO_RE  = re.compile(r'^\s*/\*[0-9a-f]+\*/\s*(?:@!?P\d+\s+)?MOV\s+(R\d+)\s*,\s*RZ\s*;')
# Any instruction that writes some register Rn as its first (destination)
# operand — used only to know when a prior "MOV Rn, RZ" has been
# invalidated by a later write to the same register. Deliberately broad
# (over-invalidates rather than under-invalidates) so this stays a
# conservative "was the last write to this register a zero-move" heuristic,
# not a claim of real dataflow/CFG analysis (task §13).
_DEST_REG_RE  = re.compile(r'^\s*/\*[0-9a-f]+\*/\s*(?:@!?P\d+\s+)?[A-Z][\w.]*\s+(R\d+)\s*[,;]')
_ST_SHARED_RE = re.compile(r'\bst\.shared\b')            # PTX-text fallback, not expected in SASS but checked
_MEMBAR_RE    = re.compile(r'\bMEMBAR\b')
_MEMBAR_CTA_RE = re.compile(r'\bMEMBAR\.SC\.CTA\b')
# BAR.SYNC matches both plain "BAR.SYNC ;" and sm_89's "BAR.SYNC.DEFER_BLOCKING" —
# \b after SYNC is a boundary against the following '.' either way (verified in
# dev_notes_phase1/scratch_feasibility_probe.sass).
_BAR_SYNC_RE  = re.compile(r'\bBAR\.SYNC\b')
_EXIT_RE      = re.compile(r'\bEXIT\b')
_BRANCH_RE    = re.compile(r'\bBRA\b|\bBSYNC\b|\bBSSY\b')
_PREDICATE_RE = re.compile(r'^\s*/\*[0-9a-f]+\*/\s*(@!?P\d)\s')


def demangle_simple(mangled: str) -> str:
    """Itanium ABI length-prefix demangle: _Z11scrub_naivePj -> scrub_naive.
    Identical logic to src/dse_analysis.py / src/attester.py."""
    m = re.match(r'_Z(\d+)(\w+)', mangled)
    if m:
        length = int(m.group(1))
        rest = m.group(2)
        return rest[:length] if len(rest) >= length else rest
    return mangled


@dataclass
class KernelFeatures:
    mangled_name: str
    demangled_name: str
    instruction_count: int = 0
    shared_store_count: int = 0
    zero_shared_store_count: int = 0
    zero_shared_store_direct_count: int = 0     # STS [...], RZ
    zero_shared_store_indirect_count: int = 0   # MOV Rn,RZ ... STS [...], Rn (see PLAN.md O0 finding)
    membar_count: int = 0
    membar_cta_count: int = 0
    bar_sync_count: int = 0
    exit_count: int = 0
    branch_count: int = 0
    exit_line_indices: list[int] = field(default_factory=list)
    last_zero_store_index: int | None = None
    first_membar_cta_index: int | None = None
    last_bar_sync_index: int | None = None
    parser_warnings: list[str] = field(default_factory=list)
    unsupported_encodings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


def split_functions(sass_text: str) -> dict[str, list[str]]:
    """Split cuobjdump -sass text into {mangled_name: [instruction_lines]}.
    Identical splitting logic to attester.py's _parse_sass, generalized to
    not assume any particular kernel names."""
    functions: dict[str, list[str]] = {}
    current: str | None = None
    buf: list[str] = []

    for line in sass_text.splitlines():
        fm = _FUNC_RE.search(line)
        if fm:
            if current is not None:
                functions[current] = buf
            current = fm.group(1)
            buf = []
        elif current is not None:
            # Skip pure-encoding continuation lines (the second line of each
            # instruction pair, containing only the control-code word) and
            # blank lines — they carry no mnemonic information.
            if line.strip() and not re.match(r'^\s*/\*\s*0x[0-9a-f]+\s*\*/\s*$', line):
                buf.append(line)

    if current is not None:
        functions[current] = buf
    return functions


def extract_features(mangled: str, lines: list[str]) -> KernelFeatures:
    feat = KernelFeatures(mangled_name=mangled, demangled_name=demangle_simple(mangled))

    # Only lines that look like actual instruction lines (contain the
    # "/*hex*/" offset marker) count toward instruction_count; this excludes
    # header/blank lines that may have survived split_functions.
    instr_lines = [l for l in lines if re.match(r'^\s*/\*[0-9a-f]+\*/', l)]
    feat.instruction_count = len(instr_lines)

    # Linear "last write wins" register tracking (NOT real dataflow/CFG —
    # see the module docstring and PLAN.md §9's O0 finding). At -O0, ptxas
    # emits `MOV Rn, RZ` followed later by `STS [...], Rn` instead of
    # folding straight to `STS [...], RZ`; without tracking this, every
    # -O0 artifact — including hardened constructs — misclassifies as
    # ABSENT. This scan is a single top-to-bottom pass per kernel: any
    # instruction writing Rn as its first operand invalidates a prior
    # "Rn holds zero" fact (conservative — over-invalidates rather than
    # risk a false zero-store claim).
    reg_holds_zero: dict[str, bool] = {}

    for idx, line in enumerate(lines):
        mz = _MOV_ZERO_RE.match(line)
        if mz:
            reg_holds_zero[mz.group(1)] = True
        else:
            dm = _DEST_REG_RE.match(line)
            if dm:
                reg_holds_zero[dm.group(1)] = False

        if _STS_RE.search(line) or _ST_SHARED_RE.search(line):
            feat.shared_store_count += 1
        if _STS_RZ_RE.search(line):
            feat.zero_shared_store_count += 1
            feat.zero_shared_store_direct_count += 1
            feat.last_zero_store_index = idx
        else:
            sr = _STS_REG_RE.search(line)
            if sr and reg_holds_zero.get(sr.group(1)):
                feat.zero_shared_store_count += 1
                feat.zero_shared_store_indirect_count += 1
                feat.last_zero_store_index = idx
        if _MEMBAR_RE.search(line):
            feat.membar_count += 1
        if _MEMBAR_CTA_RE.search(line):
            feat.membar_cta_count += 1
            if feat.first_membar_cta_index is None:
                feat.first_membar_cta_index = idx
        if _BAR_SYNC_RE.search(line):
            feat.bar_sync_count += 1
            feat.last_bar_sync_index = idx
        if _EXIT_RE.search(line):
            feat.exit_count += 1
            feat.exit_line_indices.append(idx)
        if _BRANCH_RE.search(line):
            feat.branch_count += 1

    if feat.shared_store_count > 0 and feat.zero_shared_store_count == 0:
        feat.parser_warnings.append(
            "shared stores present but none recognized as zero-valued (RZ) — "
            "either a genuinely nonzero store (expected for N7-style negative "
            "cases) or an encoding this parser doesn't recognize"
        )
    if feat.exit_count == 0:
        feat.parser_warnings.append("no EXIT instruction found — kernel body may be truncated")
    if feat.exit_count > 1:
        feat.parser_warnings.append(
            f"{feat.exit_count} EXIT instructions found for what may be single-source-exit "
            "source — see PLAN.md §9 (confirmed benign per-warp scheduling artifact in at "
            "least one case, empirically verified on hardware, but not assumed safe in "
            "general; classify_binary.py handles this conservatively)"
        )

    return feat


def parse_sass_file(sass_text: str) -> dict[str, KernelFeatures]:
    functions = split_functions(sass_text)
    return {name: extract_features(name, lines) for name, lines in functions.items()}
