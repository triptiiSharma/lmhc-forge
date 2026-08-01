#!/usr/bin/env python3
"""
nvdisasm_parser.py — independent disassembly-and-parsing path using
nvdisasm's structured JSON output (`nvdisasm -c -json <cubin>`).

This is NOT the cuobjdump/SASS-text parser reused or renamed. It consumes
nvdisasm's JSON structure directly (opcode/operands/predicate/control-flow
fields) and re-implements the same instruction-level taxonomy independently
against that structure. It is an independent DISASSEMBLY AND PARSING PATH
over the same NVIDIA cubin — not independent semantic ground truth, and
this module makes no claim otherwise.

JSON shape (confirmed against real CUDA 12.8 and 13.3 output before writing
this parser, not assumed):
    [
      {"ELF": {...}, "SM": {...}, "SchemaVersion": {...}, "Producer": "...",
       "Description": "...", [toolkit-specific extra keys...]},
      [
        {"function-name": "<mangled>", "start": int, "length": int,
         "other-attributes": [...],
         "sass-instructions": [
             {"predicate": "@P0"?, "opcode": "STS", "operands": "[R0],RZ",
              "other-attributes": {"control-flow": "True"}?},
             ...
         ]},
        ...
      ]
    ]

Required keys are checked explicitly; a record missing a required key is
never silently skipped or guessed at — it's recorded as an unsupported
record and the whole artifact is scored UNDECIDABLE rather than partially
parsed with a hole in it.
"""

import re
from dataclasses import dataclass, field, asdict

REQUIRED_TOP_LEVEL_LEN = 2
REQUIRED_FUNCTION_KEYS = {"function-name", "sass-instructions"}
REQUIRED_INSTR_KEYS = {"opcode"}  # operands/predicate/other-attributes are optional per-instruction

_STS_RE = re.compile(r'^STS(\.\S+)?$')          # STS, STS.128, etc. (any shared store)
_MEMBAR_RE = re.compile(r'^MEMBAR(\..+)?$')
_MEMBAR_CTA_RE = re.compile(r'^MEMBAR\.SC\.CTA$')
_BAR_SYNC_RE = re.compile(r'^BAR\.SYNC(\..+)?$')
_EXIT_RE = re.compile(r'^EXIT$')
_BRANCH_OPCODES = {"BRA", "BSYNC", "BSSY", "BRX", "BRXU"}
# Operand pattern for a register that is the SOURCE (value) operand of an STS:
# "STS" operands look like "[addr-expr],SRC" -- SRC is everything after the
# last top-level comma outside brackets.
_STS_OPERANDS_RE = re.compile(r'^\[[^\]]*\]\s*,\s*(\S+)$')
_REG_RE = re.compile(r'^R\d+$')
_ZERO_MOVE_OPERANDS_RE = re.compile(r'^(R\d+)\s*,\s*RZ$')
_DEST_REG_FROM_OPERANDS_RE = re.compile(r'^(R\d+)\b')


@dataclass
class NvdisasmKernelFeatures:
    function_name: str
    demangled_name: str
    instruction_count: int = 0
    shared_store_count: int = 0
    zero_shared_store_count: int = 0
    zero_shared_store_direct_count: int = 0
    zero_shared_store_indirect_count: int = 0
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
    unsupported_records: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


def demangle_simple(mangled: str) -> str:
    """Same generic Itanium-ABI length-prefix algorithm as sass_parser.py's
    demangle_simple -- this is a tool-independent utility (the mangling
    scheme is a C++ ABI fact, not something specific to either disassembler),
    not a reuse of cuobjdump-specific parsing logic."""
    m = re.match(r'_Z(\d+)(\w+)', mangled)
    if m:
        n = int(m.group(1))
        rest = m.group(2)
        return rest[:n] if len(rest) >= n else rest
    return mangled


class SchemaError(Exception):
    pass


def validate_schema(data) -> tuple[bool, str]:
    """Conservative schema check. Returns (ok, reason). Never raises for a
    malformed-but-parseable document -- callers use the reason to mark the
    artifact UNDECIDABLE rather than crash the whole run."""
    if not isinstance(data, list) or len(data) != REQUIRED_TOP_LEVEL_LEN:
        return False, f"top-level JSON is not a 2-element list (got {type(data).__name__}, len={len(data) if isinstance(data, list) else 'n/a'})"
    meta, functions = data
    if not isinstance(meta, dict):
        return False, "element 0 (metadata) is not a dict"
    if not isinstance(functions, list):
        return False, "element 1 (functions) is not a list"
    return True, ""


def extract_features(func_record: dict) -> NvdisasmKernelFeatures:
    missing = REQUIRED_FUNCTION_KEYS - set(func_record.keys())
    mangled = func_record.get("function-name", "<unknown>")
    feat = NvdisasmKernelFeatures(function_name=mangled, demangled_name=demangle_simple(mangled))
    if missing:
        feat.unsupported_records.append(f"function record missing required keys: {missing}")
        return feat

    instrs = func_record["sass-instructions"]
    if not isinstance(instrs, list):
        feat.unsupported_records.append("sass-instructions is not a list")
        return feat

    feat.instruction_count = len(instrs)

    # Linear, conservative register-zero tracking over structured operands
    # (independent re-implementation of the same principle as
    # sass_parser.py's text-based tracker, applied to JSON fields instead of
    # regex-matched text lines -- last-write-wins, any destination write
    # invalidates, never assumes cross-branch dataflow).
    reg_holds_zero: dict[str, bool] = {}

    for idx, instr in enumerate(instrs):
        if not isinstance(instr, dict) or "opcode" not in instr:
            feat.unsupported_records.append(f"instruction[{idx}] missing 'opcode'")
            continue
        opcode = instr["opcode"]
        operands = instr.get("operands", "")

        # Track zero-holding registers: a plain "MOV Rn,RZ" sets Rn zero;
        # any other instruction whose operands begin with "Rn" as the first
        # (destination) token invalidates Rn. STS never writes a register
        # (its first operand is a bracketed address), so it never
        # invalidates anything via this path.
        if opcode == "MOV":
            mz = _ZERO_MOVE_OPERANDS_RE.match(operands)
            if mz:
                reg_holds_zero[mz.group(1)] = True
            else:
                dm = _DEST_REG_FROM_OPERANDS_RE.match(operands)
                if dm:
                    reg_holds_zero[dm.group(1)] = False
        elif not _STS_RE.match(opcode):
            dm = _DEST_REG_FROM_OPERANDS_RE.match(operands)
            if dm:
                reg_holds_zero[dm.group(1)] = False

        if _STS_RE.match(opcode):
            feat.shared_store_count += 1
            m = _STS_OPERANDS_RE.match(operands)
            if m:
                src = m.group(1)
                if src == "RZ":
                    feat.zero_shared_store_count += 1
                    feat.zero_shared_store_direct_count += 1
                    feat.last_zero_store_index = idx
                elif _REG_RE.match(src) and reg_holds_zero.get(src):
                    feat.zero_shared_store_count += 1
                    feat.zero_shared_store_indirect_count += 1
                    feat.last_zero_store_index = idx
            else:
                feat.parser_warnings.append(
                    f"instruction[{idx}]: STS operands '{operands}' did not match "
                    f"expected '[addr],SRC' pattern -- not counted as zero or nonzero"
                )

        if _MEMBAR_RE.match(opcode):
            feat.membar_count += 1
        if _MEMBAR_CTA_RE.match(opcode):
            feat.membar_cta_count += 1
            if feat.first_membar_cta_index is None:
                feat.first_membar_cta_index = idx
        if _BAR_SYNC_RE.match(opcode):
            feat.bar_sync_count += 1
            feat.last_bar_sync_index = idx
        if _EXIT_RE.match(opcode):
            feat.exit_count += 1
            feat.exit_line_indices.append(idx)
        if opcode in _BRANCH_OPCODES:
            feat.branch_count += 1

    if feat.shared_store_count > 0 and feat.zero_shared_store_count == 0:
        feat.parser_warnings.append(
            "shared stores present but none recognized as zero-valued -- "
            "either genuinely nonzero or an unrecognized encoding"
        )
    if feat.exit_count == 0:
        feat.parser_warnings.append("no EXIT instruction found")
    if feat.exit_count > 1:
        feat.parser_warnings.append(f"{feat.exit_count} EXIT instructions found (multi-exit)")

    return feat


def parse_nvdisasm_json(data) -> tuple[dict[str, NvdisasmKernelFeatures], str | None]:
    """Returns ({demangled_name: features}, schema_error_or_None)."""
    ok, reason = validate_schema(data)
    if not ok:
        return {}, reason
    _, functions = data
    result = {}
    for func_record in functions:
        if not isinstance(func_record, dict):
            continue
        feat = extract_features(func_record)
        result[feat.demangled_name] = feat
    return result, None
