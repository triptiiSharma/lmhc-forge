#!/usr/bin/env python3
"""
select_review_sample.py — Phase 5 stratified sample selection.

Design (documented in full in PHASE5_SAMPLE_DESIGN.md):

Tier 1 (mandatory, all-inclusive):
  - every COMPILE_FAILED / COMPILE_UNSUPPORTED / UNDECIDABLE record (0 exist
    in this corpus, but the rule is unconditional so it stays correct if
    that ever changes on a re-run)
  - both pilot flag-recheck cells

Tier 2 (base stratified layer): for each of the 42 (construct, context)
combinations, deterministically select BASE_PER_COMBO artifacts, rotating
through (ptxas_opt, arch, toolkit) combinations so that across the 42
combos every value of every dimension is hit many times over, not just
once — using a fixed seed for reproducibility, not hand-picked.

Tier 3 (oversampling layer): every record with
zero_shared_store_indirect_count > 0 (the O0 indirect-store code path that
already produced one real bug), every record with exit_count >= 3 (the
multi-EXIT code path that turned out to have zero real FULL_SIGNATURE
coverage — Phase 4 finding), and extra draws from the two least-verified
architectures (sm_75, sm_90) not already selected.

Output: results/validation_sample.csv (BLINDED — no checker_verdict,
binary_ground_truth columns) and results/validation_sample_answer_key.csv
(same rows, WITH checker_verdict — kept separate for Phase 6 comparison
only, never shown to reviewers during labeling).
"""

import csv
import json
import random
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).parent.parent
SEED = 20260801  # fixed for reproducibility; = today's date at time of the study
BASE_PER_COMBO = 5
TIER3_CAP = 15  # each oversampling category is capped and stratified, not exhaustive —
                 # these turned out to be ~15-17% of the whole corpus, not rare edge
                 # cases, so "oversample" means proportionally more than their base
                 # rate would get in a pure random sample, not "include all of them"

CONSTRUCTS = [f"Z{i}" for i in range(7)]
CONTEXTS = [f"K{i}" for i in range(6)]
OPTS = [0, 1, 2, 3]
ARCHS = ["sm_75", "sm_80", "sm_86", "sm_87", "sm_89", "sm_90"]
TOOLKITS = ["cuda-12.8", "cuda-13.3"]


def load_jsonl(path):
    return [json.loads(l) for l in Path(path).read_text(encoding="utf-8").splitlines() if l.strip()]


def main():
    artifacts = {a["artifact_id"]: a for a in load_jsonl(ROOT / "results" / "artifacts.jsonl")}
    features = {f["artifact_id"]: f for f in load_jsonl(ROOT / "results" / "features.jsonl")}

    rng = random.Random(SEED)
    selected: dict[str, str] = {}  # artifact_id -> reason

    # Tier 1a: every non-OK compile status (unconditional, even though 0 exist today)
    for aid, a in artifacts.items():
        if a["compile_status"] != "COMPILE_OK":
            selected[aid] = "tier1_non_ok_compile_status"

    # Tier 1b: every UNDECIDABLE checker verdict (unconditional, 0 exist today)
    for aid, f in features.items():
        if f.get("checker_verdict") == "UNDECIDABLE":
            selected.setdefault(aid, "tier1_undecidable")

    # Tier 1c: both pilot flag-recheck cells
    for aid, a in artifacts.items():
        if a.get("study_phase") == "pilot_flag_recheck":
            selected.setdefault(aid, "tier1_flag_recheck")

    # Tier 2: base stratified layer, deterministic rotation per (construct, context)
    by_combo = defaultdict(list)
    for aid, a in artifacts.items():
        if a["compile_status"] != "COMPILE_OK":
            continue
        by_combo[(a["zeroization_construct"], a["kernel_context"])].append(aid)

    rotation_i = 0
    for z in CONSTRUCTS:
        for k in CONTEXTS:
            candidates = by_combo.get((z, k), [])
            if not candidates:
                continue
            # Build the desired (opt, arch, toolkit) targets for this combo,
            # rotating the starting offset so different combos hit different
            # points in the opt x arch x toolkit space rather than always
            # picking the same corner.
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
                    # exact target combo not present for this (z,k) pair (shouldn't
                    # happen for the core matrix, but fall back to a random pick
                    # from this combo rather than silently under-sampling it)
                    pick = rng.choice([c for c in candidates if c not in selected] or candidates)
                    selected[pick] = "tier2_base_stratified_fallback"

    def stratified_subsample(pool: list[str], cap: int, reason: str) -> None:
        """Spread `cap` picks across as many distinct (construct, context)
        combos as possible, rather than letting one combo dominate — a
        capped tier-3 category should still sample the *breadth* of where
        the pattern occurs, not just its most common corner."""
        pool = [aid for aid in pool if aid not in selected]
        rng.shuffle(pool)
        by_combo_pool = defaultdict(list)
        for aid in pool:
            a = artifacts[aid]
            by_combo_pool[(a["zeroization_construct"], a["kernel_context"])].append(aid)
        combos = list(by_combo_pool.keys())
        rng.shuffle(combos)
        picked = 0
        ci = 0
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

    # Tier 3a: O0 indirect zero-store code path — capped, stratified subsample
    # (343 records match; that's ~17% of the corpus, too large to include whole).
    pool_a = [aid for aid, f in features.items()
              if f.get("kernel_features") and f["kernel_features"].get("zero_shared_store_indirect_count", 0) > 0]
    stratified_subsample(pool_a, TIER3_CAP, "tier3_indirect_zero_store")

    # Tier 3b: multi-EXIT (exit_count>=3) code path — capped, stratified subsample
    # (258 records match; Phase 4 finding was that NONE of these reach
    # FULL_SIGNATURE, so this sample specifically checks that's really true,
    # not an artifact of under-sampling).
    pool_b = [aid for aid, f in features.items()
              if f.get("kernel_features") and f["kernel_features"].get("exit_count", 0) >= 3]
    stratified_subsample(pool_b, TIER3_CAP, "tier3_multi_exit_ge3")

    # Tier 3c: extra draws from the two least hardware-verified archs
    # (sm_75, sm_90 — everything else has at least one hand-traced example
    # from Phase 3/4; these two don't yet), 6 more each, spread across constructs.
    for arch in ("sm_75", "sm_90"):
        candidates = [aid for aid, a in artifacts.items()
                      if a["target_arch"] == arch and a["compile_status"] == "COMPILE_OK"
                      and aid not in selected]
        rng.shuffle(candidates)
        # spread across distinct constructs rather than clustering
        seen_constructs = set()
        picked = 0
        for aid in candidates:
            z = artifacts[aid]["zeroization_construct"]
            if z in seen_constructs:
                continue
            selected[aid] = f"tier3_underverified_arch_{arch}"
            seen_constructs.add(z)
            picked += 1
            if picked >= 6:
                break

    print(f"Total selected: {len(selected)}")
    reason_counts = defaultdict(int)
    for r in selected.values():
        reason_counts[r] += 1
    for r, c in sorted(reason_counts.items()):
        print(f"  {r}: {c}")

    # Write blinded CSV (no checker_verdict / binary_ground_truth) and a
    # separate answer key.
    fieldnames_blind = [
        "artifact_id", "selection_reason", "zeroization_construct", "kernel_context",
        "target_arch", "ptxas_optimization", "cuda_version", "sass_path", "source_file",
        "reviewer_1_label", "reviewer_1_notes", "reviewer_2_label", "reviewer_2_notes",
        "disagreement", "resolved_binary_ground_truth",
    ]
    blind_path = ROOT / "results" / "validation_sample.csv"
    with blind_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames_blind)
        w.writeheader()
        for aid, reason in sorted(selected.items()):
            a = artifacts[aid]
            w.writerow({
                "artifact_id": aid, "selection_reason": reason,
                "zeroization_construct": a["zeroization_construct"],
                "kernel_context": a["kernel_context"],
                "target_arch": a["target_arch"],
                "ptxas_optimization": a["ptxas_optimization"],
                "cuda_version": a["cuda_version"],
                "sass_path": a["sass_path"], "source_file": a["source_file"],
                "reviewer_1_label": "", "reviewer_1_notes": "",
                "reviewer_2_label": "", "reviewer_2_notes": "",
                "disagreement": "", "resolved_binary_ground_truth": "",
            })

    answer_path = ROOT / "results" / "validation_sample_answer_key.csv"
    with answer_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["artifact_id", "selection_reason", "checker_verdict", "checker_detail",
                     "source_intent_label"])
        for aid, reason in sorted(selected.items()):
            a = artifacts[aid]
            feat = features.get(aid, {})
            w.writerow([aid, reason, feat.get("checker_verdict"), feat.get("checker_detail"),
                        a["source_intent_label"]])

    print(f"\nwrote {blind_path} (blinded, {len(selected)} rows)")
    print(f"wrote {answer_path} (answer key, NOT for reviewer use)")

    # Coverage confirmation
    def coverage(field, values, getter):
        present = {getter(artifacts[aid]) for aid in selected}
        missing = set(values) - present
        return missing

    print("\n=== coverage check ===")
    print("constructs missing:", coverage("construct", CONSTRUCTS, lambda a: a["zeroization_construct"]))
    print("contexts missing:", coverage("context", CONTEXTS, lambda a: a["kernel_context"]))
    print("opts missing:", coverage("opt", OPTS, lambda a: a["ptxas_optimization"]))
    print("archs missing:", coverage("arch", ARCHS, lambda a: a["target_arch"]))
    print("toolkits missing:", coverage("toolkit", TOOLKITS, lambda a: a["cuda_version"]))
    verdicts_present = {features[aid]["checker_verdict"] for aid in selected if aid in features}
    print("verdicts present:", verdicts_present)


if __name__ == "__main__":
    main()
