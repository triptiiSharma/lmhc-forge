from pathlib import Path

ids = ["R0001", "R0025", "R0050", "R0075", "R0100", "R0125", "R0150", "R0175", "R0200", "R0254"]
for rid in ids:
    p = Path(f"phase5_review/reviewer_A/sass/{rid}.sass")
    text = p.read_text(encoding="utf-8")
    lines = text.splitlines()
    print(f"=== {rid} ({len(lines)} lines) ===")
    for l in lines[:6]:
        print(" ", l)
    print("  ...")
    print()
