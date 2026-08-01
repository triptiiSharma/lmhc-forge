#!/usr/bin/env python3
"""
aggregate_results.py — reduce results/artifacts.jsonl + results/features.jsonl
into results/summary.json: outcome counts by construct/context/arch/toolkit/
opt-level, plus scalability metrics (task §21). Read-only over the JSONL
source of truth; safe to re-run any time.
"""

import argparse
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).parent.parent


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def group_counts(records, key_fn):
    counts = defaultdict(Counter)
    for r in records:
        counts[key_fn(r)][r.get("compile_status")] += 1
    return {str(k): dict(v) for k, v in counts.items()}


def stat_block(values):
    values = [v for v in values if v is not None]
    if not values:
        return None
    return {
        "n": len(values), "min": min(values), "max": max(values),
        "mean": round(statistics.mean(values), 2),
        "median": statistics.median(values),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifacts", default=str(ROOT / "results" / "artifacts.jsonl"))
    ap.add_argument("--features", default=str(ROOT / "results" / "features.jsonl"))
    ap.add_argument("--out", default=str(ROOT / "results" / "summary.json"))
    args = ap.parse_args()

    artifacts = load_jsonl(Path(args.artifacts))
    features = load_jsonl(Path(args.features))
    features_by_id = {f["artifact_id"]: f for f in features}

    verdict_counts = Counter()
    for a in artifacts:
        if a.get("compile_status") != "COMPILE_OK":
            continue
        f = features_by_id.get(a["artifact_id"])
        verdict_counts[f["checker_verdict"] if f else "NOT_YET_EXTRACTED"] += 1

    verdict_by_construct = defaultdict(Counter)
    verdict_by_context = defaultdict(Counter)
    for a in artifacts:
        if a.get("compile_status") != "COMPILE_OK":
            continue
        f = features_by_id.get(a["artifact_id"])
        v = f["checker_verdict"] if f else "NOT_YET_EXTRACTED"
        verdict_by_construct[a["zeroization_construct"]][v] += 1
        verdict_by_context[a["kernel_context"]][v] += 1

    summary = {
        "total_records": len(artifacts),
        "by_compile_status": dict(Counter(a.get("compile_status") for a in artifacts)),
        "by_compile_status_and_construct": group_counts(artifacts, lambda r: r.get("zeroization_construct")),
        "by_compile_status_and_context": group_counts(artifacts, lambda r: r.get("kernel_context")),
        "by_compile_status_and_arch": group_counts(artifacts, lambda r: r.get("target_arch")),
        "by_compile_status_and_toolkit": group_counts(artifacts, lambda r: r.get("cuda_version")),
        "checker_verdict_counts": dict(verdict_counts),
        "checker_verdict_by_construct": {k: dict(v) for k, v in verdict_by_construct.items()},
        "checker_verdict_by_context": {k: dict(v) for k, v in verdict_by_context.items()},
        "scalability": {
            "compile_time_ms": stat_block([a.get("compile_time_ms") for a in artifacts]),
            "cubin_size_bytes": stat_block([a.get("cubin_size_bytes") for a in artifacts]),
            "sass_size_bytes": stat_block([a.get("sass_size_bytes") for a in artifacts]),
            "analysis_time_ms": stat_block([f.get("analysis_time_ms") for f in features]),
        },
        "manual_review_status_counts": dict(Counter(a.get("manual_review_status") for a in artifacts)),
    }

    Path(args.out).write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[aggregate_results] wrote {args.out}")
    print(json.dumps(summary["by_compile_status"], indent=2))
    print(json.dumps(summary["checker_verdict_counts"], indent=2))


if __name__ == "__main__":
    main()
