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
    ("WACA_G2 Stage1 [B]",    r"<int p2c64::eval_waca_stage1_ptm<false>\([^>]*>:\n(.*?)(?:\n\n|\Z)", 0, False),
    ("WACA_G2 Stage1 [W]",    r"<int p2c64::eval_waca_stage1_ptm<true>\([^>]*>:\n(.*?)(?:\n\n|\Z)",  0, False),
    ("WACA_G2 Stage2.5",      r"<(?:int )?p2c64::eval_waca_stage25_kernel\([^>]*>:\n(.*?)(?:\n\n|\Z)", 36, True),
    ("WACA_G2 Stage3 [B]",    r"<int p2c64::eval_waca_stage3_ptm<false>\([^>]*>:\n(.*?)(?:\n\n|\Z)", 128, False),
    ("WACA_G2 Stage3 [W]",    r"<int p2c64::eval_waca_stage3_ptm<true>\([^>]*>:\n(.*?)(?:\n\n|\Z)",  128, False),
    ("WACA_V31 Stage1 [B]",   r"<int p2c64::eval_waca_v31_stage1_ptm<false>\([^>]*>:\n(.*?)(?:\n\n|\Z)", 0, False),
    ("WACA_V31 Stage1 [W]",   r"<int p2c64::eval_waca_v31_stage1_ptm<true>\([^>]*>:\n(.*?)(?:\n\n|\Z)",  0, False),
    ("WACA_V31 Stage2.5",     r"<(?:int )?p2c64::eval_waca_v31_stage25_kernel\([^>]*>:\n(.*?)(?:\n\n|\Z)", 16, True),
    ("WACA_V31 Stage3 [B]",   r"<int p2c64::eval_waca_v31_stage3_ptm<false>\([^>]*>:\n(.*?)(?:\n\n|\Z)", 160, False),
    ("WACA_V31 Stage3 [W]",   r"<int p2c64::eval_waca_v31_stage3_ptm<true>\([^>]*>:\n(.*?)(?:\n\n|\Z)",  160, False),
    ("WACA_V32 Stage1 [B]",   r"<int p2c64::eval_waca_v32_stage1_ptm<false>\([^>]*>:\n(.*?)(?:\n\n|\Z)", 0, False),
    ("WACA_V32 Stage1 [W]",   r"<int p2c64::eval_waca_v32_stage1_ptm<true>\([^>]*>:\n(.*?)(?:\n\n|\Z)",  0, False),
    ("WACA_V32 Stage2.5",     r"<(?:int )?p2c64::eval_waca_v32_stage25_kernel\([^>]*>:\n(.*?)(?:\n\n|\Z)", 16, True),
    ("WACA_V32 Stage3 [B]",   r"<int p2c64::eval_waca_v32_stage3_ptm<false>\([^>]*>:\n(.*?)(?:\n\n|\Z)", 160, True),
    ("WACA_V32 Stage3 [W]",   r"<int p2c64::eval_waca_v32_stage3_ptm<true>\([^>]*>:\n(.*?)(?:\n\n|\Z)",  160, True),
    ("WACA_V33 Stage1 [B]",   r"<int p2c64::eval_waca_v33_stage1_ptm<false>\([^>]*>:\n(.*?)(?:\n\n|\Z)", 0, False),
    ("WACA_V33 Stage1 [W]",   r"<int p2c64::eval_waca_v33_stage1_ptm<true>\([^>]*>:\n(.*?)(?:\n\n|\Z)",  0, False),
    ("WACA_V33 Stage2.5",     r"<(?:int )?p2c64::eval_waca_v33_stage25_kernel\([^>]*>:\n(.*?)(?:\n\n|\Z)", 20, True),
    ("WACA_V33 ExtraFeat",    r"<(?:void )?p2c64::extract_waca_v33_stage3_extra_features\([^>]*>:\n(.*?)(?:\n\n|\Z)", 40, True),
    ("WACA_V33 MLP Kernel",   r"<(?:int )?p2c64::eval_waca_v33_mlp_kernel\([^>]*>:\n(.*?)(?:\n\n|\Z)", 40, True),
    ("WACA_V33 Stage3 [B]",   r"<int p2c64::eval_waca_v33_stage3_ptm<false>\([^>]*>:\n(.*?)(?:\n\n|\Z)", 120, "v33_internal"),
    ("WACA_V33 Stage3 [W]",   r"<int p2c64::eval_waca_v33_stage3_ptm<true>\([^>]*>:\n(.*?)(?:\n\n|\Z)",  120, "v33_internal"),
    ("SWAR_V40 4c/G [B]",     r"<int p2c64::eval_swar_v40_ptm<false, false, false>\([^>]*>:\n(.*?)(?:\n\n|\Z)", 14, True),
    ("SWAR_V40 4c/G [W]",     r"<int p2c64::eval_swar_v40_ptm<true, false, false>\([^>]*>:\n(.*?)(?:\n\n|\Z)",  14, True),
    ("SWAR_V40 4c/L [B]",     r"<int p2c64::eval_swar_v40_ptm<false, false, true>\([^>]*>:\n(.*?)(?:\n\n|\Z)",  24, True),
    ("SWAR_V40 4c/L [W]",     r"<int p2c64::eval_swar_v40_ptm<true, false, true>\([^>]*>:\n(.*?)(?:\n\n|\Z)",   24, True),
    ("SWAR_V40 8c/G [B]",     r"<int p2c64::eval_swar_v40_ptm<false, true, false>\([^>]*>:\n(.*?)(?:\n\n|\Z)",  26, True),
    ("SWAR_V40 8c/G [W]",     r"<int p2c64::eval_swar_v40_ptm<true, true, false>\([^>]*>:\n(.*?)(?:\n\n|\Z)",   26, True),
    ("SWAR_V40 8c/L [B]",     r"<int p2c64::eval_swar_v40_ptm<false, true, true>\([^>]*>:\n(.*?)(?:\n\n|\Z)",   36, True),
    ("SWAR_V40 8c/L [W]",     r"<int p2c64::eval_swar_v40_ptm<true, true, true>\([^>]*>:\n(.*?)(?:\n\n|\Z)",    36, True),
]

print("=== P2C-64 v4.0 Pure-Scalar Assembly Audit (objdump -d -C, Black [B] & White [W]) ===")
print(f"{'Kernel':<22} | {'Insns':>6} | {'imul':>5} | {'div':>4} | {'calls':>5} | {'plt':>4} | {'xmm/ymm/zmm':>11} | Status")
print("-" * 89)

failed = False
for item in targets:
    if len(item) == 3:
        name, pattern, max_imul = item
        call_policy = False
    else:
        name, pattern, max_imul, call_policy = item
    m = re.search(pattern, asm, re.S)
    if not m:
        print(f"{name:<22} | NOT FOUND IN BINARY")
        failed = True
        continue
    lines = [l for l in m.group(1).splitlines() if "\t" in l]
    insns = len(lines)
    imuls = sum(1 for l in lines if re.search(r"\b(imul|mul)\b", l.split("\t")[-1]))
    divs  = sum(1 for l in lines if re.search(r"\b(idiv|div)\b", l.split("\t")[-1]))
    pop_plt = sum(1 for l in lines if "__popcountdi2" in l)
    call_lines = [l for l in lines if re.search(r"\bcall\b", l.split("\t")[-1]) and "__stack_chk_fail" not in l]
    calls = len(call_lines)
    plt_calls = sum(1 for l in call_lines if "@plt" in l)
    simd  = sum(1 for l in lines if re.search(r"%[xyz]mm\d+", l.split("\t")[-1]))
    avx   = sum(1 for l in lines if re.search(r"%[yz]mm\d+", l.split("\t")[-1]))

    status = "OK"
    if avx > 0 or simd > 0:
        status = f"FAIL (expected 0 simd, got {simd})"
        failed = True
    elif divs > 0:
        status = f"FAIL (expected 0 div, got {divs})"
        failed = True
    elif pop_plt > 0 or plt_calls > 0:
        status = f"FAIL (expected 0 PLT/popcnt calls, got plt={plt_calls} pop={pop_plt})"
        failed = True
    elif call_policy is True and calls > 0:
        status = f"FAIL (expected 0 calls, got calls={calls})"
        failed = True
    elif call_policy == "v33_internal":
        bad = [l for l in call_lines if ("extract_waca_v33_stage3_extra_features" not in l and "eval_waca_v33_mlp_kernel" not in l)]
        if bad:
            status = f"FAIL (unexpected external call: {bad[0].strip()})"
            failed = True
    elif imuls > max_imul:
        status = f"FAIL (expected <={max_imul} imul, got {imuls})"
        failed = True

    print(f"{name:<22} | {insns:6d} | {imuls:5d} | {divs:4d} | {calls:5d} | {plt_calls:4d} | {simd:11d} | {status}")

if failed:
    sys.exit(1)
print(f"\n[PASS] Assembly audit verified: 0 SIMD/AVX, 0 div & 0 PLT calls across all {len(targets)} v2.0 + v3.0 + v3.1 + v3.2 + v3.3/v3.4 + v4.0 kernels!")
PYEOF
