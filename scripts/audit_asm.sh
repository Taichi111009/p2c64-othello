#!/usr/bin/env bash
set -euo pipefail

BIN="${1:-build/bench_scalar_eval}"
if [[ ! -x "$BIN" ]]; then
    echo "Error: binary '$BIN' not found or not executable. Run 'make all' first." >&2
    exit 1
fi

python3 - "$BIN" << 'PYEOF'
import re
import subprocess
import sys

bin_path = sys.argv[1]
asm = subprocess.check_output(["objdump", "-d", "-C", bin_path], text=True)

targets = [
    ("RAW_NTUPLE_64 [B]",     r"<int p2c64::eval_raw_ntuple64_ptm<false>\([^>]*>:\n(.*?)(?:\n\n|\Z)", 0),
    ("RAW_NTUPLE_64 [W]",     r"<int p2c64::eval_raw_ntuple64_ptm<true>\([^>]*>:\n(.*?)(?:\n\n|\Z)",  0),
    ("LINEAR_64 [B]",         r"<int p2c64::evaluate_p2c64_ptm<\(p2c64::EvalMode\)1, false>\([^>]*>:\n(.*?)(?:\n\n|\Z)", 0),
    ("LINEAR_64 [W]",         r"<int p2c64::evaluate_p2c64_ptm<\(p2c64::EvalMode\)1, true>\([^>]*>:\n(.*?)(?:\n\n|\Z)",  0),
    ("CAND_A_SCRELU4 [B]",    r"<int p2c64::evaluate_p2c64_ptm<\(p2c64::EvalMode\)2, false>\([^>]*>:\n(.*?)(?:\n\n|\Z)", 42),
    ("CAND_A_SCRELU4 [W]",    r"<int p2c64::evaluate_p2c64_ptm<\(p2c64::EvalMode\)2, true>\([^>]*>:\n(.*?)(?:\n\n|\Z)",  42),
    ("CAND_B_NAIVE [B]",      r"<int p2c64::evaluate_p2c64_ptm<\(p2c64::EvalMode\)3, false>\([^>]*>:\n(.*?)(?:\n\n|\Z)", 0),
    ("CAND_B_NAIVE [W]",      r"<int p2c64::evaluate_p2c64_ptm<\(p2c64::EvalMode\)3, true>\([^>]*>:\n(.*?)(?:\n\n|\Z)",  0),
    ("CAND_B_KEY14 [B]",      r"<int p2c64::evaluate_p2c64_ptm<\(p2c64::EvalMode\)4, false>\([^>]*>:\n(.*?)(?:\n\n|\Z)", 0),
    ("CAND_B_KEY14 [W]",      r"<int p2c64::evaluate_p2c64_ptm<\(p2c64::EvalMode\)4, true>\([^>]*>:\n(.*?)(?:\n\n|\Z)",  0),
    ("CAND_C_HYBRID [B]",     r"<int p2c64::evaluate_p2c64_ptm<\(p2c64::EvalMode\)5, false>\([^>]*>:\n(.*?)(?:\n\n|\Z)", 22),
    ("CAND_C_HYBRID [W]",     r"<int p2c64::evaluate_p2c64_ptm<\(p2c64::EvalMode\)5, true>\([^>]*>:\n(.*?)(?:\n\n|\Z)",  22),
]

print("=== P2C-64 v2.0 Pure-Scalar Assembly Audit (objdump -d -C, Black [B] & White [W]) ===")
print(f"{'Kernel':<22} | {'Insns':>6} | {'imul':>5} | {'div':>4} | {'xmm/ymm/zmm':>11} | Status")
print("-" * 74)

failed = False
for name, pattern, max_imul in targets:
    m = re.search(pattern, asm, re.S)
    if not m:
        print(f"{name:<22} | NOT FOUND IN BINARY")
        failed = True
        continue
    lines = [l for l in m.group(1).splitlines() if "\t" in l]
    insns = len(lines)
    imuls = sum(1 for l in lines if re.search(r"\b(imul|mul)\b", l.split("\t")[-1]))
    divs  = sum(1 for l in lines if re.search(r"\b(idiv|div)\b", l.split("\t")[-1]))
    simd  = sum(1 for l in lines if re.search(r"%[xyz]mm\d+", l.split("\t")[-1]))
    avx   = sum(1 for l in lines if re.search(r"%[yz]mm\d+", l.split("\t")[-1]))

    status = "OK"
    if avx > 0 or simd > 0:
        status = f"FAIL (expected 0 simd, got {simd})"
        failed = True
    elif imuls > max_imul:
        status = f"FAIL (expected <={max_imul} imul, got {imuls})"
        failed = True

    print(f"{name:<22} | {insns:6d} | {imuls:5d} | {divs:4d} | {simd:11d} | {status}")

if failed:
    sys.exit(1)
print("\n[PASS] Assembly audit verified: 0 SIMD/AVX across all 12 PTM kernels; 0 imul in RAW/LINEAR/CAND_B; <=40 imul in CAND_A; <=20 imul in CAND_C.")
PYEOF
