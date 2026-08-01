#!/usr/bin/env python3
"""model_predict.py -- RP Sec VI.D analytical cost model validation.

Implements C_scrub ~= F / min(b, P*w*f) + L and calibrates b (bandwidth,
bytes/cycle) and L (barrier cost, cycles) from TWO of the four measured
clock64() data points, then predicts the OTHER TWO (held out) and reports
whether prediction and measurement agree -- not circular curve-fitting to
every point, which would prove nothing.

Calibration set: {full_sm_amplification, reduction} -- the two largest-F
kernels, where the model's own reasoning says the bandwidth-bound term
min(b, P*w*f) = b should dominate (RP Sec VI.D: "the term is bandwidth-bound
for large F"), making the 2-point exact solve for (b, L) meaningful rather
than an arbitrary pair.

Held-out validation set: {gemm, attention} -- smaller F, smaller/differently
sized blocks, where the model itself anticipates the thread-issue-bound term
or barrier-cost-scaling assumptions may not transfer cleanly.

Input: results/benchmark_clock64.json (real clock64() measurements -- not
invented). Output: results/model_validation.json.
"""
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
IN = REPO / "results" / "benchmark_clock64.json"
OUT = REPO / "results" / "model_validation.json"

W_BYTES = 4  # bytes/store (uint32/float)


def main():
    data = json.loads(IN.read_text())
    khz = data["gpu_clock_rate_khz"]

    kernels = {k["class"]: k for k in data["kernels"]}
    amp = data["full_sm_amplification"]

    # threads per kernel (must match src/benchmark_clock64.cu launch config)
    threads = {"reduction": 256, "gemm": 256, "attention": 64}
    amp_threads = amp["threads"]

    points = {
        "full_sm_amplification": {"F_bytes": amp["words"] * W_BYTES,
                                   "P": amp_threads,
                                   "cycles_measured": amp["cycles_median"]},
        "reduction": {"F_bytes": kernels["reduction"]["scrub_words"] * W_BYTES,
                      "P": threads["reduction"],
                      "cycles_measured": kernels["reduction"]["cycles_median"]},
        "gemm": {"F_bytes": kernels["gemm"]["scrub_words"] * W_BYTES,
                 "P": threads["gemm"],
                 "cycles_measured": kernels["gemm"]["cycles_median"]},
        "attention": {"F_bytes": kernels["attention"]["scrub_words"] * W_BYTES,
                      "P": threads["attention"],
                      "cycles_measured": kernels["attention"]["cycles_median"]},
    }

    # --- Calibrate (b, L) exactly from the 2-point system:
    #     cycles = F/b + L
    p1, p2 = points["full_sm_amplification"], points["reduction"]
    F1, C1 = p1["F_bytes"], p1["cycles_measured"]
    F2, C2 = p2["F_bytes"], p2["cycles_measured"]
    # C1 - C2 = (F1 - F2) / b  =>  b = (F1-F2) / (C1-C2)
    b_bytes_per_cycle = (F1 - F2) / (C1 - C2)
    L_cycles = C1 - F1 / b_bytes_per_cycle
    b_gbps = b_bytes_per_cycle * (khz * 1000) / 1e9

    results = []
    for name, pt in points.items():
        predicted_cycles = pt["F_bytes"] / b_bytes_per_cycle + L_cycles
        measured_cycles = pt["cycles_measured"]
        predicted_ns = predicted_cycles * 1e6 / khz
        measured_ns = measured_cycles * 1e6 / khz
        ratio = predicted_cycles / measured_cycles if measured_cycles else float("nan")
        is_calibration_point = name in ("full_sm_amplification", "reduction")
        results.append({
            "kernel": name,
            "role": "calibration" if is_calibration_point else "held_out_prediction",
            "F_bytes": pt["F_bytes"],
            "P_threads": pt["P"],
            "predicted_cycles": round(predicted_cycles, 2),
            "measured_cycles": round(measured_cycles, 2),
            "predicted_ns": round(predicted_ns, 4),
            "measured_ns": round(measured_ns, 4),
            "predicted_over_measured_ratio": round(ratio, 4),
            "within_2x": bool(0.5 <= ratio <= 2.0) if measured_cycles else False,
        })

    out = {
        "experiment": "Analytical cost model validation (RP Sec VI.D): C_scrub ~= F / min(b, P*w*f) + L",
        "hardware": data["hardware"],
        "input_file": "results/benchmark_clock64.json",
        "calibration_method": "Exact 2-point solve of cycles=F/b+L using {full_sm_amplification, reduction} "
                               "(the two largest-F points, where the model's own reasoning predicts the "
                               "bandwidth-bound term dominates). {gemm, attention} are held out and PREDICTED, "
                               "not fit -- this is a falsifiable check, not curve-fitting to every point.",
        "calibrated_b_bytes_per_cycle": round(b_bytes_per_cycle, 4),
        "calibrated_b_GB_per_s": round(b_gbps, 2),
        "calibrated_L_cycles": round(L_cycles, 2),
        "calibrated_L_ns": round(L_cycles * 1e6 / khz, 4),
        "note_on_b": "b is per-SM shared-store bandwidth implied by the fit; for reference, a documented "
                     "back-of-envelope figure for Ada-class shared memory is ~128 bytes/clock/SM (32 lanes x 4B, "
                     "no bank conflicts) -- the fitted value is reported as-derived, not adjusted to match.",
        "results": results,
    }
    OUT.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))
    print(f"\ncalibrated b = {b_bytes_per_cycle:.4f} bytes/cycle ({b_gbps:.2f} GB/s), L = {L_cycles:.2f} cycles",
          file=sys.stderr)
    for r in results:
        print(f"  {r['kernel']:22s} [{r['role']:20s}] predicted={r['predicted_ns']:8.2f} ns  "
              f"measured={r['measured_ns']:8.2f} ns  ratio={r['predicted_over_measured_ratio']:.3f}",
              file=sys.stderr)


if __name__ == "__main__":
    main()
