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
    ("RAW_NTUPLE_64",        r"<p2c64::eval_raw_ntuple64\([^>]*>:\n(.*?)(?:\n\n|\Z)"),
    ("LINEAR_64",            r"<int p2c64::evaluate_p2c64<\(p2c64::EvalMode\)1>\([^>]*>:\n(.*?)(?:\n\n|\Z)"),
    ("CAND_A_SCRELU_K2",     r"<int p2c64::evaluate_p2c64<\(p2c64::EvalMode\)2>\([^>]*>:\n(.*?)(?:\n\n|\Z)"),
    ("CAND_B_3W3B_NAIVE",    r"<int p2c64::evaluate_p2c64<\(p2c64::EvalMode\)3>\([^>]*>:\n(.*?)(?:\n\n|\Z)"),
    ("CAND_B_3W3B_KEY14",    r"<int p2c64::evaluate_p2c64<\(p2c64::EvalMode\)4>\([^>]*>:\n(.*?)(?:\n\n|\Z)"),
    ("CAND_C_HYBRID",        r"<int p2c64::evaluate_p2c64<\(p2c64::EvalMode\)5>\([^>]*>:\n(.*?)(?:\n\n|\Z)"),
]

print("=== P2C-64 Pure-Scalar Assembly Audit (objdump -d -C) ===")
print(f"{'Kernel':<22} | {'Insns':>6} | {'imul':>5} | {'div':>4} | {'xmm/ymm/zmm':>11} | Status")
print("-" * 74)

failed = False
for name, pattern in targets:
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
    if name in ("RAW_NTUPLE_64", "LINEAR_64", "CAND_B_3W3B_NAIVE", "CAND_B_3W3B_KEY14"):
        if imuls != 0:
            status = f"FAIL (expected 0 imul, got {imuls})"
            failed = True
    elif name == "CAND_A_SCRELU_K2" and imuls > 40:
        status = f"FAIL (expected <=40 imul after factoring, got {imuls})"
        failed = True
    elif name == "CAND_C_HYBRID" and imuls > 12:
        status = f"FAIL (expected <=12 imul after factoring, got {imuls})"
        failed = True

    print(f"{name:<22} | {insns:6d} | {imuls:5d} | {divs:4d} | {simd:11d} | {status}")

if failed:
    sys.exit(1)
print("\n[PASS] Assembly audit verified: 0 SIMD/AVX across all 6 kernels; 0 imul in RAW/LINEAR/CAND_B; factored imul in CAND_A & CAND_C.")
PYEOF
