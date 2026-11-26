#!/usr/bin/env python3
"""Bit-exact verification of Python .p2c I/O + integer simulator vs C++ pure-scalar kernels,
plus full 16-symmetry (D4 x Z2) invariance audit and OpenMP batch-bridge throughput benchmark."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import List, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from p2c64_bridge import (
    CORNER_CANON_SIZE,
    CORNER_RAW_SIZE,
    KEY_CANON_ENTRIES,
    NONKEY_RAW_ENTRIES,
    NUM_BLOCKS,
    NUM_PLIES,
    PACKED_SAMPLE_DTYPE,
    TOTAL_RAW_ENTRIES,
    P2C64Bridge,
)
from p2c64_io import (
    EvalMode,
    P2C64Model,
    RawNTupleModel,
    eval_p2c64_int_batch,
    eval_raw_ntuple64_int_batch,
    load_p2c,
    make_empty_p2c64,
    save_p2c64,
    save_raw_p2c,
    symmetrize_p2c64_model,
    symmetrize_raw_model,
)


def load_positions(data_bin: Path | None, n: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if data_bin is not None and data_bin.exists():
        arr = np.fromfile(data_bin, dtype=PACKED_SAMPLE_DTYPE)
        if len(arr) == 0:
            raise RuntimeError(f"Empty dataset: {data_bin}")
        if len(arr) > n:
            idx = np.linspace(0, len(arr) - 1, n, dtype=np.int64)
            arr = arr[idx]
        return (
            np.ascontiguousarray(arr["black"], dtype=np.uint64),
            np.ascontiguousarray(arr["white"], dtype=np.uint64),
            np.ascontiguousarray(arr["stm"], dtype=np.int8),
            np.ascontiguousarray(arr["ply"], dtype=np.uint8),
        )

    # Fallback: synthetic valid random board states
    rng = np.random.default_rng(20261125)
    blacks = np.zeros(n, dtype=np.uint64)
    whites = np.zeros(n, dtype=np.uint64)
    stms = np.where(rng.integers(0, 2, size=n) == 0, -1, 1).astype(np.int8)
    plies = np.zeros(n, dtype=np.uint8)
    for i in range(n):
        ply = int(rng.integers(0, 60))
        discs = ply + 4
        sqs = rng.choice(64, size=discs, replace=False)
        nb = int(rng.integers(1, discs))
        b = np.uint64(0)
        w = np.uint64(0)
        for s in sqs[:nb]:
            b |= np.uint64(1) << np.uint64(s)
        for s in sqs[nb:]:
            w |= np.uint64(1) << np.uint64(s)
        blacks[i] = b
        whites[i] = w
        plies[i] = ply
    return blacks, whites, stms, plies


def verify_p2c_file(
    bridge: P2C64Bridge,
    p2c_path: Path,
    black: np.ndarray,
    white: np.ndarray,
    stm: np.ndarray,
    ply: np.ndarray,
) -> None:
    model = load_p2c(p2c_path)
    if isinstance(model, RawNTupleModel):
        ext = bridge.extract_batch(
            black, white, stm, enable_aug=False, need_raw_idx=True, need_p2c=False, need_raw_flat=True
        )
        py_scores = eval_raw_ntuple64_int_batch(model, ext["raw_flat"], ext["raw_idx"], stm, ply)
        mode_name = "RAW_NTUPLE_64"
    else:
        ext = bridge.extract_batch(
            black, white, stm, enable_aug=False, need_raw_idx=False, need_p2c=True, need_raw_flat=False
        )
        py_scores = eval_p2c64_int_batch(
            model, ext["key_cx"], ext["key_flip"], ext["nonkey_flat"], stm, ply
        )
        mode_name = model.mode.name

    cpp_mode, cpp_scores, max_sym_err = bridge.eval_file_batch(
        p2c_path, black, white, stm, ply, check_16sym=True
    )
    diff = np.abs(py_scores.astype(np.int64) - cpp_scores.astype(np.int64))
    max_err = int(diff.max()) if len(diff) else 0
    non_zero = int(np.count_nonzero(cpp_scores))

    status = "PASS" if (max_err == 0 and max_sym_err == 0) else "FAIL"
    print(
        f"[{status}] {p2c_path.name:<28} ({mode_name:<18} mode={cpp_mode}) | "
        f"N={len(black)} | py_vs_cpp_max_err={max_err} | 16sym_max_err={max_sym_err} | "
        f"non_zero={non_zero}/{len(black)}"
    )
    if max_err != 0 or max_sym_err != 0:
        bad = np.where(diff != 0)[0]
        if len(bad) > 0:
            idx = int(bad[0])
            print(f"       First mismatch at i={idx}: ply={ply[idx]} stm={stm[idx]} py={py_scores[idx]} cpp={cpp_scores[idx]}")
        sys.exit(1)


def run_synthetic_roundtrip(bridge: P2C64Bridge, tmp_dir: Path, black, white, stm, ply) -> None:
    orbits = bridge.get_orbit_tables()
    rng = np.random.default_rng(12345)
    tmp_dir.mkdir(parents=True, exist_ok=True)

    # 1) RAW_NTUPLE_64
    raw = RawNTupleModel(
        w_ply=rng.integers(-2000, 2001, size=(NUM_PLIES, TOTAL_RAW_ENTRIES), dtype=np.int16),
        corner_tempo=rng.integers(-64, 65, size=(NUM_PLIES, CORNER_RAW_SIZE), dtype=np.int8),
    )
    raw = symmetrize_raw_model(raw, orbits)
    raw_p = tmp_dir / "synth_raw.p2c"
    save_raw_p2c(raw_p, raw)
    verify_p2c_file(bridge, raw_p, black, white, stm, ply)

    # 2..5) LINEAR_64, CAND_A, CAND_B, CAND_C
    for mode in (
        EvalMode.LINEAR_64,
        EvalMode.CAND_A_SCRELU_K2,
        EvalMode.CAND_B_3W3B,
        EvalMode.CAND_C_HYBRID,
    ):
        m = make_empty_p2c64(mode)
        m.key_blocks["w"] = rng.integers(-2000, 2001, size=(NUM_BLOCKS, KEY_CANON_ENTRIES, 6), dtype=np.int16)
        m.key_blocks["aux"] = rng.integers(0, 256, size=(NUM_BLOCKS, KEY_CANON_ENTRIES, 4), dtype=np.uint8)
        m.nonkey_w = rng.integers(-2000, 2001, size=(NUM_PLIES, NONKEY_RAW_ENTRIES), dtype=np.int16)
        m.corner_tempo = rng.integers(-64, 65, size=(NUM_BLOCKS, CORNER_CANON_SIZE), dtype=np.int8)
        m.screlu_mu = rng.integers(8, 64, size=(NUM_BLOCKS, 3, 2), dtype=np.int32)
        m.screlu_beta_sq = rng.integers(-400, 401, size=(NUM_BLOCKS, 3, 2), dtype=np.int32)
        m.screlu_beta_pr = rng.integers(-400, 401, size=(NUM_BLOCKS, 3), dtype=np.int32)
        m.macro_cw = rng.integers(-600, 601, size=(NUM_BLOCKS, 512), dtype=np.int16)
        m.macro_ea = rng.integers(-600, 601, size=(NUM_BLOCKS, 512), dtype=np.int16)
        m.macro_cd = rng.integers(-600, 601, size=(NUM_BLOCKS, 512), dtype=np.int16)
        m = symmetrize_p2c64_model(m, orbits)
        p = tmp_dir / f"synth_{mode.name.lower()}.p2c"
        save_p2c64(p, m)
        verify_p2c_file(bridge, p, black, white, stm, ply)


def bench_bridge(bridge: P2C64Bridge, black, white, stm) -> None:
    reps = max(1, 100000 // len(black))
    b_big = np.tile(black, reps)
    w_big = np.tile(white, reps)
    s_big = np.tile(stm, reps)
    n = len(b_big)

    # Warmup
    bridge.extract_batch(b_big[:1024], w_big[:1024], s_big[:1024], aug_seed=1, enable_aug=True, need_p2c=True)

    t0 = time.perf_counter()
    bridge.extract_batch(b_big, w_big, s_big, aug_seed=42, enable_aug=True, need_p2c=True)
    dt = time.perf_counter() - t0
    print(f"[BENCH] C++ OpenMP Batch Bridge (aug=ON, 64-pat P2C): {n:,} pos in {dt*1000:.1f} ms -> {n/dt/1e6:.2f} M pos/s")


def main() -> None:
    ap = argparse.ArgumentParser(description="Verify C++/Python bit-exact .p2c I/O & 16-symmetry invariance")
    ap.add_argument("--data", type=Path, default=None, help="Optional val.bin for real test positions")
    ap.add_argument("--num-pos", type=int, default=1000, help="Number of test positions (default: 1000)")
    ap.add_argument("--models", type=Path, nargs="*", default=None, help="Explicit .p2c files to verify")
    ap.add_argument("--work-dir", type=Path, default=None, help="Directory containing trained .p2c files")
    ap.add_argument("--synthetic", action="store_true", help="Also run synthetic 5-mode roundtrip test")
    ap.add_argument("--bench-bridge", action="store_true", help="Benchmark C++ OpenMP batch bridge throughput")
    args = ap.parse_args()

    bridge = P2C64Bridge()
    black, white, stm, ply = load_positions(args.data, args.num_pos)

    print(f"=== P2C-64 C++/Python Bit-Exact I/O & 16-Symmetry Test ({len(black)} positions) ===")
    files: List[Path] = []
    if args.models:
        files.extend(args.models)
    if args.work_dir:
        for name in (
            "raw_ntuple64.p2c",
            "linear64.p2c",
            "cand_a_screlu_k2.p2c",
            "cand_b_3w3b.p2c",
            "cand_c_hybrid.p2c",
        ):
            p = args.work_dir / name
            if p.exists():
                files.append(p)

    if args.synthetic or not files:
        tmp_dir = (args.work_dir / "_synth_tmp") if args.work_dir else Path("build/_synth_tmp")
        run_synthetic_roundtrip(bridge, tmp_dir, black, white, stm, ply)

    for p in files:
        verify_p2c_file(bridge, p, black, white, stm, ply)

    if args.bench_bridge:
        bench_bridge(bridge, black, white, stm)

    print("=== BIT-EXACT VERIFICATION PASSED (0 ERRORS) ===")


if __name__ == "__main__":
    main()
