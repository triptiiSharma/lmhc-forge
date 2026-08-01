#!/usr/bin/env python3
"""
extract_features.py — read results/artifacts.jsonl, run sass_parser +
classify_binary over every COMPILE_OK artifact's saved SASS, and write:
  - results/features.jsonl   (one KernelFeatures dict per artifact, plus artifact_id)
  - results/classifications.csv  (artifact_id, source_intent_label, checker_verdict, detail)

Does NOT set binary_ground_truth (task §15 — that's Phase 5 manual review
only, never derived automatically). checker_verdict here is exactly what
classify_binary.py produces; it is explicitly not equated with ground truth
anywhere downstream.

Resumable: skips any artifact_id already present in features.jsonl.

Usage:
    python3 extract_features.py --jsonl results/artifacts.jsonl
"""

import argparse
import csv
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from sass_parser import parse_sass_file
from classify_binary import classify_kernel

ROOT = Path(__file__).parent.parent


def load_jsonl(path: Path) -> list[dict]:
    records = []
    if not path.exists():
        return records
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def already_done(features_path: Path) -> set[str]:
    done = set()
    for rec in load_jsonl(features_path):
        done.add(rec["artifact_id"])
    return done


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl", default=str(ROOT / "results" / "artifacts.jsonl"))
    ap.add_argument("--features-out", default=str(ROOT / "results" / "features.jsonl"))
    ap.add_argument("--classifications-out", default=str(ROOT / "results" / "classifications.csv"))
    args = ap.parse_args()

    artifacts = load_jsonl(Path(args.jsonl))
    ok_artifacts = [a for a in artifacts if a.get("compile_status") == "COMPILE_OK"]

    features_path = Path(args.features_out)
    done_ids = already_done(features_path)

    pending = [a for a in ok_artifacts if a["artifact_id"] not in done_ids]
    print(f"[extract_features] {len(ok_artifacts)} COMPILE_OK artifacts, "
          f"{len(done_ids)} already extracted, {len(pending)} pending")

    csv_rows = []
    t0 = time.time()
    with features_path.open("a", encoding="utf-8") as feat_f:
        for i, art in enumerate(pending, 1):
            sass_path = Path(art["sass_path"]) if art.get("sass_path") else None
            aid = art["artifact_id"]
            t_start = time.time()

            if not sass_path or not sass_path.exists():
                result_rec = {
                    "artifact_id": aid, "kernel_features": None,
                    "analysis_time_ms": 0,
                    "extraction_error": "sass_path missing or file not found",
                }
                feat_f.write(json.dumps(result_rec) + "\n")
                csv_rows.append([aid, art["source_intent_label"], "UNDECIDABLE",
                                  "sass file not found", 0])
                continue

            sass_text = sass_path.read_text(encoding="utf-8", errors="replace")
            functions = parse_sass_file(sass_text)
            target_name = art["kernel_demangled_name"]
            match = next((f for f in functions.values() if f.demangled_name == target_name), None)

            analysis_ms = int((time.time() - t_start) * 1000)

            if match is None:
                classification = None
                verdict, detail = "UNDECIDABLE", (
                    f"kernel '{target_name}' not found in SASS "
                    f"(found: {sorted(f.demangled_name for f in functions.values())})"
                )
                feat_dict = None
            else:
                classification = classify_kernel(match)
                verdict, detail = classification.category, classification.detail
                feat_dict = match.to_dict()

            result_rec = {
                "artifact_id": aid,
                "kernel_features": feat_dict,
                "checker_verdict": verdict,
                "checker_detail": detail,
                "multi_exit_scope_assumed": classification.multi_exit_scope_assumed if classification else False,
                "analysis_time_ms": analysis_ms,
            }
            feat_f.write(json.dumps(result_rec) + "\n")
            csv_rows.append([aid, art["source_intent_label"], verdict, detail, analysis_ms])

            if i % 50 == 0 or i == len(pending):
                print(f"  [{i}/{len(pending)}] processed")

    # classifications.csv is rewritten in full each run from the union of all
    # features records (cheap — CSV is a projection, not the source of truth).
    all_features = load_jsonl(features_path)
    features_by_id = {r["artifact_id"]: r for r in all_features}
    artifacts_by_id = {a["artifact_id"]: a for a in artifacts}

    csv_path = Path(args.classifications_out)
    with csv_path.open("w", newline="", encoding="utf-8") as cf:
        writer = csv.writer(cf)
        writer.writerow(["artifact_id", "zeroization_construct", "kernel_context",
                          "target_arch", "cuda_version", "ptxas_optimization",
                          "source_intent_label", "checker_verdict", "checker_detail",
                          "multi_exit_scope_assumed", "analysis_time_ms"])
        for aid, feat_rec in features_by_id.items():
            art = artifacts_by_id.get(aid, {})
            writer.writerow([
                aid, art.get("zeroization_construct"), art.get("kernel_context"),
                art.get("target_arch"), art.get("cuda_version"), art.get("ptxas_optimization"),
                art.get("source_intent_label"), feat_rec.get("checker_verdict"),
                feat_rec.get("checker_detail"), feat_rec.get("multi_exit_scope_assumed"),
                feat_rec.get("analysis_time_ms"),
            ])

    elapsed = time.time() - t0
    print(f"[extract_features] {len(pending)} artifacts processed in {elapsed:.1f}s")
    print(f"[extract_features] wrote {features_path} ({len(all_features)} total records)")
    print(f"[extract_features] wrote {csv_path} ({len(features_by_id)} rows)")


if __name__ == "__main__":
    main()
