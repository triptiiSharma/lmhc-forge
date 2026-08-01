import csv

supported = [
    ["We compiled and analyzed 2,016 GPU binaries across two CUDA toolkits, six architecture "
     "targets, four ptxas optimization levels, seven zeroization constructs, and six kernel "
     "contexts.", "results/artifacts.jsonl", "2016", "100% COMPILE_OK, 0 failed, 0 unsupported"],
    ["The structural checker assigned each artifact to one of four observed binary-signature "
     "categories (FULL_SIGNATURE, PARTIAL_SIGNATURE, STORE_ONLY, ABSENT); a fifth (UNDECIDABLE) "
     "is defined but did not occur in this corpus.",
     "results/features.jsonl", "2016", "576/864/96/480/0 respectively"],
    ["The hardened constructions (Z5, Z6) retained the complete recognized structural signature "
     "across the entire tested configuration matrix.",
     "results/features.jsonl, filtered construct in {Z5,Z6}", "576",
     "288/288 for each of Z5 and Z6, across all contexts/opt-levels/archs/toolkits"],
    ["An independent disassembly path (nvdisasm) agreed with the primary cuobjdump-based "
     "analysis on 2,016 of 2,016 supported artifacts (100%), on both structural category and "
     "all 8 extracted security-relevant features.",
     "results/nvdisasm_features.jsonl, results/cross_tool_summary.json", "2016",
     "0 disagreements, 0 unsupported, 0 UNDECIDABLE from either path"],
    ["Repeated compilation preserved the extracted structural category, features, and cubin "
     "bytes across the tested replication subset.",
     "results/reproducibility_summary.json", "50 configs x 3 repeats = 150",
     "150/150 byte-identical cubin/SASS/nvdisasm-JSON hashes"],
    ["A bounded runtime sanity check on real hardware (sm_89) confirmed the hardened and plain "
     "scrub constructs left zero nonzero words in the sensitive buffer after their cleanup "
     "point, across repeated launches, while the no-scrub control retained nonzero data.",
     "results/runtime_sanity.jsonl", "18 variants x 5 repeats = 90 launches",
     "90/90 launch/sync/compute success; 0 nonzero words for Z1/Z2/Z4/Z5/Z6; nonzero for Z0"],
    ["A controlled corpus of real compiled cases, including one with a source-level uncovered "
     "cleanup path, demonstrated that a FULL_SIGNATURE verdict reflects the static text and "
     "does not by itself prove every dynamic thread path executes the scrub.",
     "results/controlled_cases.jsonl, case N6", "8 configurations",
     "8/8 compiled to a single-EXIT binary via ptxas predication; checker correctly reports "
     "FULL_SIGNATURE from the static text on both parsing paths"],
]

unsupported = [
    ["The checker is X% accurate.", "no independent human ground truth was collected",
     "Do not state an accuracy percentage anywhere."],
    ["False-accept rate is Y%.", "requires human-validated ground truth, not available",
     "Report cross-tool agreement or controlled-case conformance instead, never as a "
     "false-accept rate."],
    ["FULL_SIGNATURE proves every thread/warp path executes the scrub.",
     "case N6 is direct, compiled counter-evidence: predication can collapse a source-level "
     "uncovered path into a single EXIT the checker cannot distinguish",
     "State the structural-signature scope limitation explicitly whenever FULL_SIGNATURE is "
     "mentioned."],
    ["The generated binaries are secure.", "structural presence of a signature is not a "
     "security property claim", "Describe only the structural category observed."],
    ["No residue can be recovered.", "this study ran no residue-recovery/Exposure-Index "
     "experiment", "Do not make any residue-recovery claim from this study's evidence."],
    ["All NVIDIA/AMD/Intel GPUs are protected.", "only sm_89 (one physical GPU) has any "
     "runtime evidence; sm_75/80/86/87/90 are compile-target only; no AMD/Intel evidence "
     "exists in this study", "Scope every claim to the specific architecture(s) actually "
     "tested, and label non-sm_89 targets as compile-only."],
    ["nvdisasm independently confirms the checker is correct.", "nvdisasm and cuobjdump both "
     "interpret the same NVIDIA cubin -- agreement is not independent semantic ground truth",
     "Describe as cross-tool agreement between two parsing paths, never as independent "
     "confirmation of correctness."],
]

with open("results/phase6_supported_claims.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["claim", "data_source", "artifact_count", "result_supporting_it"])
    w.writerows(supported)

with open("results/phase6_unsupported_claims.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["overclaiming_wording", "why_unsupported", "safe_alternative"])
    w.writerows(unsupported)

print("wrote results/phase6_supported_claims.csv and results/phase6_unsupported_claims.csv")
