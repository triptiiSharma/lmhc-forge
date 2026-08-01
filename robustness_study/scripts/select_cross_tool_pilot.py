#!/usr/bin/env python3
"""
select_cross_tool_pilot.py — deterministic ~96-artifact stratified pilot
for cross-tool (nvdisasm vs cuobjdump) validation (Section D of the
approved plan).

Must cover: all 7 constructs, all 6 contexts, all 4 ptxas opt levels, all
6 archs, both toolkits, all 4 observed categories, direct zero stores, O0
register-mediated (indirect) zero stores, exit_count==1, exit_count==2,
exit_count>=3, attention-style K5 artifacts, and every one of
sm_75/80/86/87/89/90 individually.

Two-phase construction: (1) a base stratified layer (2 per construct x
context combo, rotating opt/arch/toolkit -- 84 artifacts), then (2)
explicit top-up draws for any required coverage dimension the base layer
didn't happen to hit, verified programmatically rather than assumed.
"""

import json
import random
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
from compile_matrix import expand_cells, artifact_id as compute_artifact_id

SEED = 20260801
BASE_PER_COMBO = 2

CONSTRUCTS = [f"Z{i}" for i in range(7)]
CONTEXTS = [f"K{i}" for i in range(6)]
OPTS = [0, 1, 2, 3]
ARCHS = ["sm_75", "sm_80", "sm_86", "sm_87", "sm_89", "sm_90"]
TOOLKITS = ["cuda-12.8", "cuda-13.3"]
CATEGORIES = ["FULL_SIGNATURE", "PARTIAL_SIGNATURE", "STORE_ONLY", "ABSENT"]


def load_jsonl(path):
    return [json.loads(l) for l in Path(path).read_text(encoding="utf-8").splitlines() if l.strip()]


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
    assert len(core_ids) == 2016

    rng = random.Random(SEED)
    selected: set[str] = set()

    # Base layer: 2 per (construct, context), rotating opt/arch/toolkit
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
            by_target = {}
            for aid in candidates:
                a = artifacts[aid]
                by_target[(a["ptxas_optimization"], a["target_arch"], a["cuda_version"])] = aid
            for j in range(BASE_PER_COMBO):
                idx = rotation_i + j
                opt = OPTS[idx % len(OPTS)]
                arch = ARCHS[(idx // len(OPTS)) % len(ARCHS)]
                toolkit = TOOLKITS[(idx // (len(OPTS) * len(ARCHS))) % len(TOOLKITS)]
                aid = by_target.get((opt, arch, toolkit))
                if aid:
                    selected.add(aid)
                else:
                    remaining = [c for c in candidates if c not in selected]
                    if remaining:
                        selected.add(rng.choice(remaining))
            rotation_i += BASE_PER_COMBO

    print(f"base layer: {len(selected)} artifacts")

    def feat_of(aid):
        return features.get(aid, {})

    def has_property(aid, prop):
        f = feat_of(aid)
        kf = f.get("kernel_features") or {}
        if prop == "direct_zero":
            return kf.get("zero_shared_store_direct_count", 0) > 0
        if prop == "o0_indirect_zero":
            a = artifacts[aid]
            return a["ptxas_optimization"] == 0 and kf.get("zero_shared_store_indirect_count", 0) > 0
        if prop == "exit1":
            return kf.get("exit_count") == 1
        if prop == "exit2":
            return kf.get("exit_count") == 2
        if prop == "exit_ge3":
            return kf.get("exit_count", 0) >= 3
        if prop == "k5":
            return artifacts[aid]["kernel_context"] == "K5"
        raise ValueError(prop)

    def top_up(predicate_fn, label, min_count=2):
        current = sum(1 for aid in selected if predicate_fn(aid))
        if current >= min_count:
            print(f"  {label}: already have {current} (>= {min_count})")
            return
        pool = [aid for aid in core_ids if aid not in selected and predicate_fn(aid)]
        rng.shuffle(pool)
        added = 0
        for aid in pool:
            if current + added >= min_count:
                break
            selected.add(aid)
            added += 1
        print(f"  {label}: had {current}, added {added}, pool available was {len(pool)}")

    print("\ntop-up pass:")
    for arch in ARCHS:
        top_up(lambda aid, a=arch: artifacts[aid]["target_arch"] == a, f"arch={arch}", min_count=2)
    for opt in OPTS:
        top_up(lambda aid, o=opt: artifacts[aid]["ptxas_optimization"] == o, f"opt=O{opt}", min_count=2)
    for tk in TOOLKITS:
        top_up(lambda aid, t=tk: artifacts[aid]["cuda_version"] == t, f"toolkit={tk}", min_count=2)
    for cat in CATEGORIES:
        top_up(lambda aid, c=cat: feat_of(aid).get("checker_verdict") == c, f"category={cat}", min_count=2)
    top_up(lambda aid: has_property(aid, "direct_zero"), "direct_zero_store", min_count=3)
    top_up(lambda aid: has_property(aid, "o0_indirect_zero"), "O0_indirect_zero_store", min_count=3)
    top_up(lambda aid: has_property(aid, "exit1"), "exit_count==1", min_count=3)
    top_up(lambda aid: has_property(aid, "exit2"), "exit_count==2", min_count=3)
    top_up(lambda aid: has_property(aid, "exit_ge3"), "exit_count>=3", min_count=3)
    top_up(lambda aid: has_property(aid, "k5"), "K5_attention_style", min_count=3)

    # Round up toward the ~96 target with additional stratified draws (one
    # extra per construct x context combo, cycling until the target is hit)
    # -- coverage is already complete at this point, this just adds breadth.
    TARGET = 96
    if len(selected) < TARGET:
        combo_order = [(z, k) for z in CONSTRUCTS for k in CONTEXTS]
        rng.shuffle(combo_order)
        ci = 0
        while len(selected) < TARGET and ci < len(combo_order) * 10:
            z, k = combo_order[ci % len(combo_order)]
            candidates = [aid for aid in by_combo.get((z, k), []) if aid not in selected]
            if candidates:
                selected.add(rng.choice(candidates))
            ci += 1

    print(f"\nfinal pilot size: {len(selected)}")

    out_path = ROOT / "results" / "cross_tool_pilot_ids.txt"
    with out_path.open("w", encoding="utf-8") as f:
        for aid in sorted(selected):
            f.write(aid + "\n")
    print(f"wrote {out_path}")

    # Coverage confirmation
    def coverage(values, getter):
        present = {getter(aid) for aid in selected}
        return set(values) - present

    print("\ncoverage check:")
    print("  constructs missing:", coverage(CONSTRUCTS, lambda a: artifacts[a]["zeroization_construct"]))
    print("  contexts missing:", coverage(CONTEXTS, lambda a: artifacts[a]["kernel_context"]))
    print("  opts missing:", coverage(OPTS, lambda a: artifacts[a]["ptxas_optimization"]))
    print("  archs missing:", coverage(ARCHS, lambda a: artifacts[a]["target_arch"]))
    print("  toolkits missing:", coverage(TOOLKITS, lambda a: artifacts[a]["cuda_version"]))
    print("  categories missing:", coverage(CATEGORIES, lambda a: feat_of(a).get("checker_verdict")))
    print("  has direct_zero:", sum(1 for a in selected if has_property(a, "direct_zero")))
    print("  has o0_indirect_zero:", sum(1 for a in selected if has_property(a, "o0_indirect_zero")))
    print("  exit1/exit2/exit_ge3:",
          sum(1 for a in selected if has_property(a, "exit1")),
          sum(1 for a in selected if has_property(a, "exit2")),
          sum(1 for a in selected if has_property(a, "exit_ge3")))
    print("  K5 count:", sum(1 for a in selected if has_property(a, "k5")))


if __name__ == "__main__":
    main()
