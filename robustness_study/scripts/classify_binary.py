#!/usr/bin/env python3
"""
classify_binary.py — structural-signature checker and classifier.

This is a RENAMED, GENERALIZED, HONESTLY-SCOPED descendant of
src/attester.py's `_verify_scrub()` (AUDIT.md §2, task §13). It is a
**structural-signature checker**, not a CFG verifier: it does not construct
basic blocks, does not compute post-dominators, and does not prove that
every dynamic execution path reaches the scrub. It only checks, from static
SASS text, whether the required instruction patterns are present and in the
expected relative textual order.

Classification categories are exactly PLAN.md §5 / task §12's seven:
FULL_SIGNATURE, STORE_ONLY, PARTIAL_SIGNATURE, ABSENT, UNDECIDABLE,
COMPILE_FAILED, COMPILE_UNSUPPORTED (the latter two are assigned upstream by
compile_matrix.py, not here).

Multi-EXIT handling: PLAN.md §9 found, and empirically verified on real
hardware, that a single-source-exit kernel can legitimately compile to
multiple SASS EXIT instructions (per-warp predicated retirement). This
checker does NOT assume the first EXIT is the right scope boundary (that
was attester.py's assumption, and the task explicitly requires this to be
named and not silently carried forward). Instead:
  - exit_count == 1: score relative to that one EXIT (identical to
    attester.py's original logic).
  - exit_count > 1: require the signature ordering to hold relative to
    EVERY EXIT that appears after the last zero-store (i.e. every exit a
    thread could plausibly reach post-scrub). If that holds uniformly,
    classify normally. If it holds for some exits but not others, that is
    exactly the kind of case a textual checker cannot safely resolve —
    return UNDECIDABLE rather than guess.

Never coerces a parser warning or an unrecognized encoding into a passing
verdict (task §12, §22 test requirement 8).
"""

import sys
from pathlib import Path
from dataclasses import dataclass

sys.path.insert(0, str(Path(__file__).parent))
from sass_parser import KernelFeatures, parse_sass_file


VALID_CATEGORIES = {
    "FULL_SIGNATURE", "STORE_ONLY", "PARTIAL_SIGNATURE",
    "ABSENT", "UNDECIDABLE",
}


@dataclass
class ClassificationResult:
    category: str
    detail: str
    exit_count: int
    multi_exit_scope_assumed: bool = False


def classify_kernel(feat: KernelFeatures) -> ClassificationResult:
    if feat.exit_count == 0:
        return ClassificationResult(
            "UNDECIDABLE", "no EXIT instruction found — cannot bound the analysis scope",
            exit_count=0,
        )

    if feat.zero_shared_store_count == 0:
        detail = "no zero-valued shared store (STS…RZ) found"
        if feat.shared_store_count > 0:
            detail += (
                f" ({feat.shared_store_count} nonzero/unclassified shared store(s) "
                "present — see parser_warnings)"
            )
        return ClassificationResult("ABSENT", detail, exit_count=feat.exit_count)

    has_cta = feat.membar_cta_count > 0
    has_bar = feat.bar_sync_count > 0

    if not has_cta and not has_bar:
        return ClassificationResult(
            "STORE_ONLY",
            f"{feat.zero_shared_store_count} zero-store(s) found; no MEMBAR.SC.CTA, no BAR.SYNC",
            exit_count=feat.exit_count,
        )

    if has_cta != has_bar:
        missing = "BAR.SYNC" if has_cta else "MEMBAR.SC.CTA"
        return ClassificationResult(
            "PARTIAL_SIGNATURE",
            f"zero-store(s) + {'MEMBAR.SC.CTA' if has_cta else 'BAR.SYNC'} present, {missing} missing",
            exit_count=feat.exit_count,
        )

    # Both present — check ordering: last zero-store < first MEMBAR.SC.CTA < last BAR.SYNC.
    order_ok = (
        feat.last_zero_store_index is not None
        and feat.first_membar_cta_index is not None
        and feat.last_bar_sync_index is not None
        and feat.last_zero_store_index < feat.first_membar_cta_index < feat.last_bar_sync_index
    )
    if not order_ok:
        return ClassificationResult(
            "PARTIAL_SIGNATURE",
            f"zero-store, MEMBAR.SC.CTA, BAR.SYNC all present but not in required order "
            f"(store@{feat.last_zero_store_index}, membar@{feat.first_membar_cta_index}, "
            f"bar@{feat.last_bar_sync_index})",
            exit_count=feat.exit_count,
        )

    if feat.exit_count == 1:
        exit_idx = feat.exit_line_indices[0]
        if feat.last_bar_sync_index < exit_idx:
            return ClassificationResult(
                "FULL_SIGNATURE",
                f"store@{feat.last_zero_store_index} < membar@{feat.first_membar_cta_index} "
                f"< bar@{feat.last_bar_sync_index} < exit@{exit_idx}",
                exit_count=1,
            )
        return ClassificationResult(
            "PARTIAL_SIGNATURE",
            f"signature ordering holds but BAR.SYNC@{feat.last_bar_sync_index} is not "
            f"before the single EXIT@{exit_idx}",
            exit_count=1,
        )

    # exit_count > 1: signature must precede every EXIT reachable after the scrub.
    relevant_exits = [e for e in feat.exit_line_indices if e > feat.last_zero_store_index]
    if not relevant_exits:
        return ClassificationResult(
            "UNDECIDABLE",
            f"{feat.exit_count} EXITs found but all precede the last zero-store — "
            "cannot establish that any exit is reached post-scrub without CFG analysis",
            exit_count=feat.exit_count,
        )
    all_after_bar = all(feat.last_bar_sync_index < e for e in relevant_exits)
    if all_after_bar:
        return ClassificationResult(
            "FULL_SIGNATURE",
            f"store@{feat.last_zero_store_index} < membar@{feat.first_membar_cta_index} "
            f"< bar@{feat.last_bar_sync_index} < every relevant EXIT {relevant_exits} "
            "(multi-EXIT kernel — see PLAN.md §9 for why first-EXIT-only scoping was rejected)",
            exit_count=feat.exit_count,
            multi_exit_scope_assumed=True,
        )
    some_after_bar = any(feat.last_bar_sync_index < e for e in relevant_exits)
    if some_after_bar:
        return ClassificationResult(
            "UNDECIDABLE",
            f"{feat.exit_count} EXITs found; signature precedes some relevant exits "
            f"{relevant_exits} but not all — a textual checker cannot resolve which "
            "thread paths actually reach the scrub without real CFG analysis "
            "(task §13: not implemented, not overclaimed)",
            exit_count=feat.exit_count,
        )
    return ClassificationResult(
        "PARTIAL_SIGNATURE",
        f"signature ordering holds internally but BAR.SYNC@{feat.last_bar_sync_index} "
        f"precedes none of the relevant EXITs {relevant_exits}",
        exit_count=feat.exit_count,
    )


def classify_artifact(sass_text: str, kernel_demangled_name: str) -> ClassificationResult:
    """Entry point matching one artifact record: find the named kernel in
    the SASS, classify it. UNDECIDABLE (never a silent pass) if missing."""
    functions = parse_sass_file(sass_text)
    match = next((f for f in functions.values() if f.demangled_name == kernel_demangled_name), None)
    if match is None:
        return ClassificationResult(
            "UNDECIDABLE",
            f"kernel '{kernel_demangled_name}' not found in cubin "
            f"(kernels present: {sorted(f.demangled_name for f in functions.values())})",
            exit_count=0,
        )
    return classify_kernel(match)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("sass_file")
    ap.add_argument("kernel_name")
    args = ap.parse_args()
    text = Path(args.sass_file).read_text(encoding="utf-8", errors="replace")
    result = classify_artifact(text, args.kernel_name)
    print(f"{result.category}: {result.detail}")
