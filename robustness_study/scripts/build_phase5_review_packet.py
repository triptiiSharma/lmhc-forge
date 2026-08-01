#!/usr/bin/env python3
"""
build_phase5_review_packet.py — corrected, genuinely-blinded Phase 5 packet.

Supersedes select_review_sample.py's output (which leaked construct/context/
arch/opt columns directly, and did not sanitize the SASS files' kernel names,
which contain the construct/context identifiers literally, e.g.
"kernel_z5_k2"). This script:

  1. Selects the primary sample from the 2,016-cell CORE matrix only (the
     2 pilot flag-recheck cells are handled separately, tagged
     SUPPLEMENTAL_PILOT, and never influence the primary selection logic).
  2. Assigns opaque review IDs (R0001..).
  3. Writes results/phase5_private_mapping.csv (review_id -> everything,
     including checker_verdict) -- NOT placed in either reviewer directory.
  4. Writes results/phase5_blinded_manifest.csv (review_id only, canonical
     order -- the un-randomized master list).
  5. Sanitizes each sampled artifact's real, compiled SASS: extracts the
     single kernel's instruction body via sass_parser.split_functions,
     replaces the "Function : <mangled-name>" line and "code for smXX" /
     ".headerflags ... SMXX ..." lines (the three identified leak points --
     verified by direct inspection, not assumed) with redacted/opaque
     text, and otherwise copies every instruction line byte-for-byte.
     Writes the sanitized file as R####.sass into BOTH reviewer_A/sass/ and
     reviewer_B/sass/ (self-contained packets, no shared directory).
  6. Writes reviewer_A/reviewer_A.csv and reviewer_B/reviewer_B.csv: same
     254(+2) review_ids, independently randomized row order, only the
     columns review_id/reviewer_label/confidence/evidence_notes/
     undecidable_reason -- no construct/context/arch/opt/toolkit/verdict/
     selection_reason/paths anywhere in these files.
  7. Prints full verification data (exit_count>=3 list, checker-verdict
     distribution, supplemental handling) for the written report.
"""

import csv
import hashlib
import json
import random
import re
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
from compile_matrix import expand_cells, artifact_id as compute_artifact_id

SEED = 20260801
BASE_PER_COMBO = 5
TIER3_CAP = 15

CONSTRUCTS = [f"Z{i}" for i in range(7)]
CONTEXTS = [f"K{i}" for i in range(6)]
OPTS = [0, 1, 2, 3]
ARCHS = ["sm_75", "sm_80", "sm_86", "sm_87", "sm_89", "sm_90"]
TOOLKITS = ["cuda-12.8", "cuda-13.3"]

_FUNC_LINE_RE = re.compile(r'^(\s*Function\s*:\s*)(\S+)(\s*)$')
_CODE_FOR_RE = re.compile(r'^(\s*code for\s+)(sm_\d+)(\s*)$')
_TARGET_RE = re.compile(r'^(\s*\.target\s+)(sm_\d+)(\s*)$')
_HEADERFLAGS_SM_RE = re.compile(r'EF_CUDA_SM\d+|EF_CUDA_VIRTUAL_SM\(EF_CUDA_SM\d+\)')
# Exhaustive: these are the only 4 line-formats found to contain an
# architecture-identifying string across all 254 sampled original SASS
# files (verified by scanning every line of every sampled file for
# "sm_"/"SM<digit>" before writing this, not assumed from a small
# hand-inspection — the first version of this script missed .target and
# the automated blinding audit (audit_blinding.py) caught it).


def load_jsonl(path):
    return [json.loads(l) for l in Path(path).read_text(encoding="utf-8").splitlines() if l.strip()]


def sanitize_sass(raw_text: str, review_id: str) -> tuple[str, list[str]]:
    """Redact the 3 verified leak points; leave every instruction line
    byte-for-byte untouched. Returns (sanitized_text, redaction_log)."""
    lines = raw_text.splitlines()
    out = []
    redactions = []
    func_count = 0
    for line in lines:
        m_func = _FUNC_LINE_RE.match(line)
        if m_func:
            func_count += 1
            out.append(f"{m_func.group(1)}{review_id}{m_func.group(3)}")
            redactions.append(f"Function line -> replaced mangled name with {review_id}")
            continue
        m_code = _CODE_FOR_RE.match(line)
        if m_code:
            out.append(f"{m_code.group(1)}[ARCH-REDACTED]{m_code.group(3)}")
            redactions.append("code-for-arch line -> [ARCH-REDACTED]")
            continue
        m_target = _TARGET_RE.match(line)
        if m_target:
            out.append(f"{m_target.group(1)}[ARCH-REDACTED]{m_target.group(3)}")
            redactions.append(".target line -> [ARCH-REDACTED]")
            continue
        if "EF_CUDA_SM" in line:
            new_line = _HEADERFLAGS_SM_RE.sub("EF_CUDA_SM[REDACTED]", line)
            out.append(new_line)
            redactions.append("headerflags line -> SM references redacted")
            continue
        out.append(line)
    if func_count != 1:
        raise ValueError(f"expected exactly 1 Function line, found {func_count}")
    return "\n".join(out) + "\n", redactions


def main():
    artifacts = {a["artifact_id"]: a for a in load_jsonl(ROOT / "results" / "artifacts.jsonl")}
    features = {f["artifact_id"]: f for f in load_jsonl(ROOT / "results" / "features.jsonl")}

    config = json.loads((ROOT / "configs" / "core_matrix.json").read_text())
    cells = expand_cells(config)
    core_ids = set()
    for cell in cells:
        aid = compute_artifact_id(cell["construct"], cell["context"], cell["ptxas_opt"],
                                   cell["arch"], cell["toolkit"]["id"], cell.get("extra_device_flags", []))
        core_ids.add(aid)
    assert len(core_ids) == 2016, f"expected 2016 core ids, got {len(core_ids)}"

    supplemental_ids = {aid for aid, a in artifacts.items() if a.get("study_phase") == "pilot_flag_recheck"}
    assert len(supplemental_ids) == 2

    rng = random.Random(SEED)
    selected: dict[str, str] = {}

    # Tier 1: unconditional inclusion of any non-OK / UNDECIDABLE core record
    for aid in core_ids:
        a = artifacts[aid]
        if a["compile_status"] != "COMPILE_OK":
            selected[aid] = "tier1_non_ok_compile_status"
        elif features.get(aid, {}).get("checker_verdict") == "UNDECIDABLE":
            selected[aid] = "tier1_undecidable"

    # Tier 2: base stratified layer, core-only
    by_combo = defaultdict(list)
    for aid in core_ids:
        a = artifacts[aid]
        by_combo[(a["zeroization_construct"], a["kernel_context"])].append(aid)

    rotation_i = 0
    for z in CONSTRUCTS:
        for k in CONTEXTS:
            candidates = by_combo.get((z, k), [])
            if not candidates:
                continue
            targets = []
            for j in range(BASE_PER_COMBO):
                idx = rotation_i + j
                opt = OPTS[idx % len(OPTS)]
                arch = ARCHS[(idx // len(OPTS)) % len(ARCHS)]
                toolkit = TOOLKITS[(idx // (len(OPTS) * len(ARCHS))) % len(TOOLKITS)]
                targets.append((opt, arch, toolkit))
            rotation_i += BASE_PER_COMBO

            by_target = {}
            for aid in candidates:
                a = artifacts[aid]
                by_target[(a["ptxas_optimization"], a["target_arch"], a["cuda_version"])] = aid

            for t in targets:
                aid = by_target.get(t)
                if aid and aid not in selected:
                    selected[aid] = "tier2_base_stratified"
                elif aid is None:
                    remaining = [c for c in candidates if c not in selected]
                    pick = rng.choice(remaining or candidates)
                    selected.setdefault(pick, "tier2_base_stratified_fallback")

    def stratified_subsample(pool, cap, reason):
        pool = [aid for aid in pool if aid in core_ids and aid not in selected]
        rng.shuffle(pool)
        by_combo_pool = defaultdict(list)
        for aid in pool:
            a = artifacts[aid]
            by_combo_pool[(a["zeroization_construct"], a["kernel_context"])].append(aid)
        combos = list(by_combo_pool.keys())
        rng.shuffle(combos)
        picked, ci = 0, 0
        while picked < cap and combos:
            combo = combos[ci % len(combos)]
            bucket = by_combo_pool[combo]
            if bucket:
                aid = bucket.pop()
                if aid not in selected:
                    selected[aid] = reason
                    picked += 1
            else:
                combos.remove(combo)
                continue
            ci += 1

    # Verify + report exit_count>=3 population BEFORE capping (step 6)
    exit_ge3_core = [aid for aid in core_ids
                      if features.get(aid, {}).get("kernel_features")
                      and features[aid]["kernel_features"].get("exit_count", 0) >= 3]
    print(f"[verify] genuine core artifacts with exit_count>=3: {len(exit_ge3_core)}")

    pool_a = [aid for aid in core_ids
              if features.get(aid, {}).get("kernel_features")
              and features[aid]["kernel_features"].get("zero_shared_store_indirect_count", 0) > 0]
    stratified_subsample(pool_a, TIER3_CAP, "tier3_indirect_zero_store")

    if len(exit_ge3_core) < TIER3_CAP:
        print(f"[verify] fewer than {TIER3_CAP} genuine cases -> including ALL {len(exit_ge3_core)}, "
              f"reallocating remaining {TIER3_CAP - len(exit_ge3_core)} slots to stratified sampling")
        for aid in exit_ge3_core:
            selected.setdefault(aid, "tier3_multi_exit_ge3")
        shortfall = TIER3_CAP - sum(1 for v in selected.values() if v == "tier3_multi_exit_ge3")
        if shortfall > 0:
            extra_pool = [aid for aid in core_ids if aid not in selected]
            stratified_subsample(extra_pool, shortfall, "tier2_extra_stratified_reallocated")
    else:
        stratified_subsample(exit_ge3_core, TIER3_CAP, "tier3_multi_exit_ge3")

    for arch in ("sm_75", "sm_90"):
        candidates = [aid for aid in core_ids
                      if artifacts[aid]["target_arch"] == arch and aid not in selected]
        rng.shuffle(candidates)
        seen_constructs, picked = set(), 0
        for aid in candidates:
            z = artifacts[aid]["zeroization_construct"]
            if z in seen_constructs:
                continue
            selected[aid] = f"tier3_underverified_arch_{arch}"
            seen_constructs.add(z)
            picked += 1
            if picked >= 6:
                break

    print(f"\n[selection] primary CORE sample size: {len(selected)}")
    reason_counts = defaultdict(int)
    for r in selected.values():
        reason_counts[r] += 1
    for r, c in sorted(reason_counts.items()):
        print(f"  {r}: {c}")

    # Exact list + verdict distribution of the >=3-exit sub-sample actually included
    included_ge3 = [aid for aid, r in selected.items() if r == "tier3_multi_exit_ge3"]
    print(f"\n[verify] included exit_count>=3 artifacts: {len(included_ge3)}")
    verdict_dist = defaultdict(int)
    for aid in included_ge3:
        v = features[aid]["checker_verdict"]
        verdict_dist[v] += 1
        a = artifacts[aid]
        print(f"    {aid}  {a['zeroization_construct']}x{a['kernel_context']}  "
              f"{a['target_arch']} O{a['ptxas_optimization']} {a['cuda_version']}  -> {v}")
    print(f"  verdict distribution: {dict(verdict_dist)}")

    # Full primary sample (core) + 2 supplemental, all get opaque IDs.
    all_ids_ordered = sorted(selected.keys()) + sorted(supplemental_ids)
    n = len(all_ids_ordered)
    review_ids = [f"R{i+1:04d}" for i in range(n)]

    # Private mapping (NOT in either reviewer dir)
    priv_path = ROOT / "results" / "phase5_private_mapping.csv"
    original_hashes = {}
    with priv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["review_id", "status", "selection_reason", "original_artifact_id",
                     "zeroization_construct", "kernel_context", "ptxas_optimization",
                     "target_arch", "cuda_version", "checker_verdict", "checker_detail",
                     "source_intent_label", "original_sass_path", "original_cubin_path",
                     "original_source_file", "original_sass_sha256"])
        for rid, aid in zip(review_ids, all_ids_ordered):
            a = artifacts[aid]
            feat = features.get(aid, {})
            status = "SUPPLEMENTAL_PILOT" if aid in supplemental_ids else "CORE"
            reason = selected.get(aid, "tier1_flag_recheck_supplemental")
            sass_text = Path(a["sass_path"]).read_text(encoding="utf-8", errors="replace")
            orig_hash = hashlib.sha256(sass_text.encode("utf-8")).hexdigest()
            original_hashes[aid] = (sass_text, orig_hash)
            w.writerow([rid, status, reason, aid, a["zeroization_construct"], a["kernel_context"],
                        a["ptxas_optimization"], a["target_arch"], a["cuda_version"],
                        feat.get("checker_verdict"), feat.get("checker_detail"),
                        a["source_intent_label"], a["sass_path"], a["cubin_path"],
                        a["source_file"], orig_hash])
    print(f"\n[write] {priv_path}")

    # Blinded manifest (canonical order, review_id only + blank label fields)
    manifest_path = ROOT / "results" / "phase5_blinded_manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["review_id"])
        for rid in review_ids:
            w.writerow([rid])
    print(f"[write] {manifest_path}")

    # Sanitize SASS, write into both reviewer dirs, track sanitized hashes.
    for reviewer in ("reviewer_A", "reviewer_B"):
        (ROOT / "phase5_review" / reviewer / "sass").mkdir(parents=True, exist_ok=True)

    sanitized_hashes = {}
    redaction_samples = []
    for rid, aid in zip(review_ids, all_ids_ordered):
        a = artifacts[aid]
        full_text = Path(a["sass_path"]).read_text(encoding="utf-8", errors="replace")
        # Every sampled file has exactly one kernel function (verified for
        # the whole 254-artifact sample before this script was written), so
        # the full cuobjdump output IS "just the one relevant kernel" --
        # sanitize_sass itself asserts func_count==1 as a hard safety check.
        sanitized, redactions = sanitize_sass(full_text, rid)
        sanitized_hashes[rid] = hashlib.sha256(sanitized.encode("utf-8")).hexdigest()
        if len(redaction_samples) < 3:
            redaction_samples.append((rid, redactions))
        for reviewer in ("reviewer_A", "reviewer_B"):
            out_path = ROOT / "phase5_review" / reviewer / "sass" / f"{rid}.sass"
            out_path.write_text(sanitized, encoding="utf-8", newline="\n")

    # Append sanitized hash to the private mapping (rewrite with extra column).
    priv_rows = list(csv.DictReader(priv_path.open(encoding="utf-8")))
    fieldnames = list(priv_rows[0].keys()) + ["sanitized_sass_sha256"]
    for row in priv_rows:
        row["sanitized_sass_sha256"] = sanitized_hashes[row["review_id"]]
    with priv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(priv_rows)

    print(f"\n[write] sanitized SASS: {len(review_ids)} files x2 (reviewer_A/sass, reviewer_B/sass)")
    print("[sample redactions]:")
    for rid, reds in redaction_samples:
        print(f"  {rid}: {reds}")

    # Reviewer CSVs, independently randomized order, minimal columns only.
    rng_a = random.Random(SEED + 1)
    rng_b = random.Random(SEED + 2)
    order_a = review_ids[:]
    order_b = review_ids[:]
    rng_a.shuffle(order_a)
    rng_b.shuffle(order_b)

    fieldnames_reviewer = ["review_id", "reviewer_label", "confidence", "evidence_notes", "undecidable_reason"]
    for reviewer, order in (("reviewer_A", order_a), ("reviewer_B", order_b)):
        path = ROOT / "phase5_review" / reviewer / f"{reviewer}.csv"
        with path.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames_reviewer)
            w.writeheader()
            for rid in order:
                w.writerow({"review_id": rid, "reviewer_label": "", "confidence": "",
                            "evidence_notes": "", "undecidable_reason": ""})
        print(f"[write] {path} ({len(order)} rows, randomized order)")

    print(f"\n[summary] total reviewed: {n} (core: {len(selected)}, supplemental: {len(supplemental_ids)})")


if __name__ == "__main__":
    main()
