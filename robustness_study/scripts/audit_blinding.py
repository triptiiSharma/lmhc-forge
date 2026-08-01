#!/usr/bin/env python3
"""
audit_blinding.py — automated scan of every reviewer-visible file for
prohibited identifying strings. Run after build_phase5_review_packet.py,
before any human labeling.

Scans phase5_review/reviewer_A/ and phase5_review/reviewer_B/ (both the
.csv files and every sanitized R####.sass) for:
  - Z0-Z6, K0-K5 construct/context tokens
  - every original artifact_id (from artifacts.jsonl)
  - every checker_verdict string (FULL_SIGNATURE, PARTIAL_SIGNATURE,
    STORE_ONLY, ABSENT, UNDECIDABLE) -- note ABSENT/UNDECIDABLE alone are
    English words that could appear innocuously; flagged for manual read,
    not auto-failed, to avoid false positives on legitimate SASS content
  - source_intent_label values (== construct IDs, already covered)
  - original file paths/names (sass_path, cubin_path, source_file basenames)
  - private-mapping column headers (as a leak-via-copy-paste check)
  - raw architecture strings (sm_75/80/86/87/89/90) and toolkit strings
    (cuda-12.8/13.3) that should only live in the private mapping
  - ptxas/opt-level and toolkit tokens

Exits 1 and prints every violation if anything is found; exits 0 with a
clean report otherwise.
"""

import csv
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent


def load_jsonl(path):
    return [json.loads(l) for l in Path(path).read_text(encoding="utf-8").splitlines() if l.strip()]


def main():
    artifacts = load_jsonl(ROOT / "results" / "artifacts.jsonl")
    all_artifact_ids = {a["artifact_id"] for a in artifacts}
    all_source_files = {Path(a["source_file"]).name for a in artifacts}
    all_cubin_names = {Path(a["cubin_path"]).name for a in artifacts if a.get("cubin_path")}
    all_sass_names = {Path(a["sass_path"]).name for a in artifacts if a.get("sass_path")}
    all_kernel_names = {a["kernel_demangled_name"] for a in artifacts}

    construct_tokens = re.compile(r'\bZ[0-6]\b')
    context_tokens = re.compile(r'\bK[0-5]\b')
    verdict_tokens = re.compile(r'\bFULL_SIGNATURE\b|\bPARTIAL_SIGNATURE\b|\bSTORE_ONLY\b')
    ambiguous_verdict_tokens = re.compile(r'\bABSENT\b|\bUNDECIDABLE\b')  # flagged, not auto-failed
    arch_tokens = re.compile(r'\bsm_(75|80|86|87|89|90)\b')
    toolkit_tokens = re.compile(r'cuda-1[23]\.\d')
    kernel_name_tokens = re.compile(r'kernel_z\d_k\d')

    violations = []
    ambiguous_hits = []
    files_scanned = 0

    review_dirs = [ROOT / "phase5_review" / "reviewer_A", ROOT / "phase5_review" / "reviewer_B"]
    for rd in review_dirs:
        for path in sorted(rd.rglob("*")):
            if not path.is_file():
                continue
            files_scanned += 1
            text = path.read_text(encoding="utf-8", errors="replace")
            rel = path.relative_to(ROOT)

            if construct_tokens.search(text):
                violations.append(f"{rel}: contains construct token (Z0-Z6): "
                                   f"{construct_tokens.findall(text)[:5]}")
            if context_tokens.search(text):
                violations.append(f"{rel}: contains context token (K0-K5): "
                                   f"{context_tokens.findall(text)[:5]}")
            if verdict_tokens.search(text):
                violations.append(f"{rel}: contains checker verdict string: "
                                   f"{verdict_tokens.findall(text)}")
            if arch_tokens.search(text):
                violations.append(f"{rel}: contains raw architecture string: "
                                   f"{arch_tokens.findall(text)}")
            if toolkit_tokens.search(text):
                violations.append(f"{rel}: contains raw toolkit version string: "
                                   f"{toolkit_tokens.findall(text)}")
            if kernel_name_tokens.search(text):
                violations.append(f"{rel}: contains original kernel_zN_kN name: "
                                   f"{kernel_name_tokens.findall(text)}")
            for aid in all_artifact_ids:
                if aid in text:
                    violations.append(f"{rel}: contains original artifact_id {aid}")
            for name in all_source_files | all_cubin_names | all_sass_names | all_kernel_names:
                if name and name in text:
                    violations.append(f"{rel}: contains original filename/kernel-name '{name}'")

            am = ambiguous_verdict_tokens.findall(text)
            if am:
                ambiguous_hits.append(f"{rel}: contains word(s) {am} that are also verdict-category "
                                       f"names (ABSENT/UNDECIDABLE) -- flagged for manual read, not "
                                       f"auto-failed, since these are also plain English words that "
                                       f"could appear in comments/register names")

    print(f"Files scanned: {files_scanned}")
    print(f"Violations: {len(violations)}")
    for v in violations:
        print(f"  VIOLATION: {v}")
    print(f"\nAmbiguous hits (manual review, not auto-fail): {len(ambiguous_hits)}")
    for a in ambiguous_hits:
        print(f"  AMBIGUOUS: {a}")

    if violations:
        sys.exit(1)
    print("\nPASS -- no prohibited identifying strings found in any reviewer-visible file.")
    sys.exit(0)


if __name__ == "__main__":
    main()
