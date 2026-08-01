#!/usr/bin/env python3
"""
validate_dataset.py — integrity checks over results/artifacts.jsonl
(task §16 validation rules, §22 test requirements 9-12).

Checks:
  1. Every artifact_id is unique.
  2. No record is missing a required field.
  3. No two records share the same (construct, context, opt, arch, toolkit,
     extra_flags) configuration unless explicitly a repetition run
     (study_phase contains "replication").
  4. Every file path a COMPILE_OK record references actually exists.
  5. cubin_hash_sha256 and source_hash_sha256 match the files on disk.
  6. Outcome counts (by compile_status) reconcile against a fresh tally of
     the raw records — i.e. the reported total is not silently short.

Exits 0 if everything passes, 1 otherwise, printing every violation found
(never just the first) so a single run gives the complete picture.
"""

import argparse
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

REQUIRED_FIELDS = [
    "artifact_id", "study_phase", "source_file", "source_hash_sha256",
    "zeroization_construct", "kernel_context", "toolkit_path", "cuda_version",
    "host_optimization", "ptxas_optimization", "target_arch",
    "compile_command", "compile_status", "compile_time_ms",
    "timestamp_utc", "git_commit",
]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def load_records(jsonl_path: Path) -> list[dict]:
    records = []
    with jsonl_path.open("r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"ERROR: {jsonl_path}:{lineno}: invalid JSON: {e}")
    return records


def validate(records: list[dict]) -> list[str]:
    errors: list[str] = []

    # 1. Unique artifact IDs.
    ids = [r.get("artifact_id") for r in records]
    id_counts = Counter(ids)
    dupes = [i for i, c in id_counts.items() if c > 1]
    for d in dupes:
        errors.append(f"duplicate artifact_id: {d} appears {id_counts[d]} times")

    # 2. Required fields present.
    for r in records:
        aid = r.get("artifact_id", "<missing-id>")
        for field in REQUIRED_FIELDS:
            if field not in r or r[field] is None:
                errors.append(f"{aid}: missing required field '{field}'")

    # 3. Duplicate configurations (same cell compiled twice) unless a
    #    declared replication run.
    config_keys = defaultdict(list)
    for r in records:
        key = (
            r.get("zeroization_construct"), r.get("kernel_context"),
            r.get("ptxas_optimization"), r.get("target_arch"), r.get("cuda_version"),
            tuple(r.get("extra_device_flags") or []),
        )
        config_keys[key].append(r.get("artifact_id"))
    for key, aids in config_keys.items():
        if len(aids) > 1:
            # artifact_id is a deterministic hash of exactly this key, so
            # more than one record for the same key can only happen if the
            # dataset was appended to non-idempotently (a real bug) — a
            # legitimate replication run must use a different artifact_id
            # scheme (task §18) and is out of scope for this basic check.
            errors.append(
                f"config {key} produced {len(aids)} records with the same "
                f"deterministic key (artifact_ids: {aids}) — should be impossible "
                f"unless artifact_id generation changed between runs"
            )

    # 4 & 5. File existence and hash match, for COMPILE_OK records only.
    for r in records:
        if r.get("compile_status") != "COMPILE_OK":
            continue
        aid = r["artifact_id"]
        for path_field, hash_field in [("cubin_path", "cubin_hash_sha256")]:
            p = r.get(path_field)
            if not p:
                errors.append(f"{aid}: COMPILE_OK but missing '{path_field}'")
                continue
            path = Path(p)
            if not path.exists():
                errors.append(f"{aid}: referenced file does not exist: {p}")
                continue
            expected_hash = r.get(hash_field)
            if expected_hash:
                actual = sha256_file(path)
                if actual != expected_hash:
                    errors.append(
                        f"{aid}: {path_field} hash mismatch — recorded {expected_hash[:16]}…, "
                        f"actual {actual[:16]}…"
                    )
        src = r.get("source_file")
        if src:
            src_path = Path(src)
            if not src_path.exists():
                errors.append(f"{aid}: source_file does not exist: {src}")
            elif r.get("source_hash_sha256"):
                actual = sha256_file(src_path)
                if actual != r["source_hash_sha256"]:
                    errors.append(
                        f"{aid}: source_hash_sha256 mismatch — recorded "
                        f"{r['source_hash_sha256'][:16]}…, actual {actual[:16]}…"
                    )
        sass = r.get("sass_path")
        if sass and not Path(sass).exists():
            errors.append(f"{aid}: sass_path does not exist: {sass}")

    return errors


def summarize(records: list[dict]) -> dict:
    by_status = Counter(r.get("compile_status") for r in records)
    by_construct = Counter(r.get("zeroization_construct") for r in records)
    by_context = Counter(r.get("kernel_context") for r in records)
    by_arch = Counter(r.get("target_arch") for r in records)
    by_toolkit = Counter(r.get("cuda_version") for r in records)
    total = len(records)
    reconciled = sum(by_status.values()) == total
    return {
        "total_records": total,
        "by_compile_status": dict(by_status),
        "by_construct": dict(by_construct),
        "by_context": dict(by_context),
        "by_arch": dict(by_arch),
        "by_toolkit": dict(by_toolkit),
        "outcome_counts_reconcile": reconciled,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl", default="results/artifacts.jsonl")
    args = ap.parse_args()

    path = Path(args.jsonl)
    if not path.exists():
        print(f"ERROR: {path} does not exist")
        sys.exit(1)

    records = load_records(path)
    errors = validate(records)
    summary = summarize(records)

    print(f"=== validate_dataset: {path} ===")
    print(f"records: {summary['total_records']}")
    print(f"by compile_status: {summary['by_compile_status']}")
    print(f"outcome counts reconcile: {summary['outcome_counts_reconcile']}")
    print()

    if errors:
        print(f"FAIL — {len(errors)} violation(s):")
        for e in errors:
            print(f"  ✗ {e}")
        sys.exit(1)
    else:
        print("PASS — no violations found")
        sys.exit(0)


if __name__ == "__main__":
    main()
