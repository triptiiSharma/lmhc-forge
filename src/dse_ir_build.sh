#!/bin/bash
set -e
cd "/mnt/c/Tripti Sharma/lmhc-forge"
mkdir -p sass_dumps
OUT=sass_dumps

count_stores() {
  local file="$1" func="$2"
  awk -v f="$func" '$0 ~ "^define.*"f"\\(" {p=1} p{print} p && /^}/{exit}' "$file" | grep -c '  store ' || true
}
show_stores() {
  local file="$1" func="$2"
  awk -v f="$func" '$0 ~ "^define.*"f"\\(" {p=1} p{print} p && /^}/{exit}' "$file" | grep '  store ' || echo "  (none)"
}

echo "############ PIPELINE 1: raw -O0 IR + explicit opt -passes=dse ############"
clang -O0 -S -emit-llvm -o "$OUT/dse_ir_exhibit.O0.ll" src/dse_ir_exhibit.c
opt -passes=dse -S "$OUT/dse_ir_exhibit.O0.ll" -o "$OUT/dse_ir_exhibit.O0.dse.ll"
echo "-- naive: before=$(count_stores "$OUT/dse_ir_exhibit.O0.ll" fill_read_scrub_naive)  after-dse=$(count_stores "$OUT/dse_ir_exhibit.O0.dse.ll" fill_read_scrub_naive) --"
echo "-- hardened: before=$(count_stores "$OUT/dse_ir_exhibit.O0.ll" fill_read_scrub_hardened)  after-dse=$(count_stores "$OUT/dse_ir_exhibit.O0.dse.ll" fill_read_scrub_hardened) --"

echo ""
echo "############ PIPELINE 2: mem2reg+sroa cleanup, THEN opt -passes=dse ############"
opt -passes=mem2reg,sroa,instcombine,loop-simplify,lcssa,indvars -S "$OUT/dse_ir_exhibit.O0.ll" -o "$OUT/dse_ir_exhibit.clean.ll"
opt -passes=dse -S "$OUT/dse_ir_exhibit.clean.ll" -o "$OUT/dse_ir_exhibit.clean.dse.ll"
echo "-- naive: before=$(count_stores "$OUT/dse_ir_exhibit.clean.ll" fill_read_scrub_naive)  after-dse=$(count_stores "$OUT/dse_ir_exhibit.clean.dse.ll" fill_read_scrub_naive) --"
show_stores "$OUT/dse_ir_exhibit.clean.dse.ll" fill_read_scrub_naive
echo "-- hardened: before=$(count_stores "$OUT/dse_ir_exhibit.clean.ll" fill_read_scrub_hardened)  after-dse=$(count_stores "$OUT/dse_ir_exhibit.clean.dse.ll" fill_read_scrub_hardened) --"
show_stores "$OUT/dse_ir_exhibit.clean.dse.ll" fill_read_scrub_hardened

echo ""
echo "############ PIPELINE 3: standard aggressive optimizer, clang -O2 direct ############"
clang -O2 -S -emit-llvm -o "$OUT/dse_ir_exhibit.O2.ll" src/dse_ir_exhibit.c
echo "-- naive store count at -O2: $(count_stores "$OUT/dse_ir_exhibit.O2.ll" fill_read_scrub_naive) --"
show_stores "$OUT/dse_ir_exhibit.O2.ll" fill_read_scrub_naive
echo "-- hardened store count at -O2: $(count_stores "$OUT/dse_ir_exhibit.O2.ll" fill_read_scrub_hardened) --"
show_stores "$OUT/dse_ir_exhibit.O2.ll" fill_read_scrub_hardened
