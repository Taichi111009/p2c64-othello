#!/usr/bin/env python3
"""Bit-exact verification of Python .p2c v2.0 I/O + integer simulator vs C++ pure-scalar kernels,
RAW==LINEAR identity verification, 16-symmetry (D4 x Z2) audit, and OpenMP bridge throughput."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import List, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from p2c64_bridge import (
    KEY_RAW_ENTRIES,
    NONKEY_RAW_ENTRIES,
    NUM_BLOCKS,
    NUM_PLIES,
    PACKED_SAMPLE_DTYPE,
    TOTAL_RAW_ENTRIES,
    P2C64Bridge,
)
from p2c64_io import (
    WACA_EO_BLOCKS,
    WACA_F25,
    WACA_F3,
    EvalMode,
    RawNTupleModel,
    WacaG2Model,
    eval_p2c64_int_batch,
    eval_raw_ntuple64_int_batch,
    eval_waca_g2_int_batch,
    load_p2c,
    make_empty_p2c64,
    make_empty_waca_g2,
    raw_to_linear_p2c64,
    save_p2c64,
    save_raw_p2c,
    save_waca_g2,
    symmetrize_p2c64_model,
    symmetrize_raw_model,
    symmetrize_waca_g2_model,
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
) -> np.ndarray:
    model = load_p2c(p2c_path)
    if isinstance(model, WacaG2Model):
        ext = bridge.extract_waca_g2_batch(
            black, white, stm, ply, model.tier1_w, enable_aug=False, need_stage3=True
        )
        py_v1, py_v25, py_v3 = eval_waca_g2_int_batch(
            model, ext["v1"], ext["f25"], ext["f3_geom"], ext["key_ptm"], ply
        )
        cpp_h, cpp_v1, cpp_v25, cpp_v3, sym3 = bridge.eval_waca_g2_file_stages(
            p2c_path, black, white, stm, ply, check_16sym=True
        )
        max_sym_err = int(sym3.max())
        err_s1 = int(np.abs(py_v1.astype(np.int64) - cpp_v1.astype(np.int64)).max()) if len(black) else 0
        err_s25 = int(np.abs(py_v25.astype(np.int64) - cpp_v25.astype(np.int64)).max()) if len(black) else 0
        err_s3 = int(np.abs(py_v3.astype(np.int64) - cpp_v3.astype(np.int64)).max()) if len(black) else 0
        max_err = max(err_s1, err_s25, err_s3)
        non_zero = int(np.count_nonzero(cpp_v3))
        status = "PASS" if (max_err == 0 and max_sym_err == 0 and cpp_h == model.mlp_hidden) else "FAIL"
        print(
            f"[{status}] {p2c_path.name:<28} ({f'WACA_G2(H={cpp_h})':<18} mode=6) | "
            f"N={len(black)} | py_vs_cpp: S1={err_s1} S2.5={err_s25} S3={err_s3} | "
            f"16sym: S1={int(sym3[0])} S2.5={int(sym3[1])} S3={int(sym3[2])} | non_zero={non_zero}/{len(black)}"
        )
        if max_err != 0 or max_sym_err != 0:
            for tag, py_a, cpp_a in (("S1", py_v1, cpp_v1), ("S2.5", py_v25, cpp_v25), ("S3", py_v3, cpp_v3)):
                bad = np.where(py_a != cpp_a)[0]
                if len(bad) > 0:
                    idx = int(bad[0])
                    print(f"       First {tag} mismatch at i={idx}: ply={ply[idx]} stm={stm[idx]} py={py_a[idx]} cpp={cpp_a[idx]}")
            sys.exit(1)
        return cpp_v3

    if isinstance(model, RawNTupleModel):
        ext = bridge.extract_batch_ptm(black, white, stm, enable_aug=False, need_p2c=False, need_raw=True)
        py_scores = eval_raw_ntuple64_int_batch(model, ext["raw_ptm"], ply)
        mode_name = "RAW_NTUPLE_64"
    else:
        ext = bridge.extract_batch_ptm(black, white, stm, enable_aug=False, need_p2c=True, need_raw=False)
        py_scores = eval_p2c64_int_batch(model, ext["key_ptm"], ext["nonkey_ptm"], ply)
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
    return cpp_scores


def run_synthetic_roundtrip(bridge: P2C64Bridge, tmp_dir: Path, black, white, stm, ply) -> None:
    orbits = bridge.get_orbit_tables()
    rng = np.random.default_rng(12345)
    tmp_dir.mkdir(parents=True, exist_ok=True)

    # 1) RAW_NTUPLE_64 + shared LINEAR_64 bit-exactness check
    raw = RawNTupleModel(
        w_ply=rng.integers(-2000, 2001, size=(NUM_PLIES, TOTAL_RAW_ENTRIES), dtype=np.int16),
    )
    raw = symmetrize_raw_model(raw, orbits)
    raw_p = tmp_dir / "synth_raw.p2c"
    save_raw_p2c(raw_p, raw)
    sc_raw = verify_p2c_file(bridge, raw_p, black, white, stm, ply)

    lin = raw_to_linear_p2c64(raw)
    lin_p = tmp_dir / "synth_linear64.p2c"
    save_p2c64(lin_p, lin)
    sc_lin = verify_p2c_file(bridge, lin_p, black, white, stm, ply)
    raw_lin_err = int(np.abs(sc_raw.astype(np.int64) - sc_lin.astype(np.int64)).max())
    if raw_lin_err != 0:
        print(f"[FAIL] synth RAW != LINEAR (max_err={raw_lin_err})")
        sys.exit(1)
    print(f"[PASS] synth RAW_NTUPLE_64 == LINEAR_64 bit-exact identity (max_err=0)")

    # 2) CAND_A, CAND_B, CAND_C (v2.0 backward compatibility)
    for mode in (
        EvalMode.CAND_A_PTM_SCRELU4,
        EvalMode.CAND_B_PTM_3W3B,
        EvalMode.CAND_C_PTM_HYBRID,
    ):
        m = make_empty_p2c64(mode)
        m.key_blocks["w"] = rng.integers(-2000, 2001, size=(NUM_BLOCKS, KEY_RAW_ENTRIES, 6), dtype=np.int16)
        m.key_blocks["aux"] = rng.integers(0, 256, size=(NUM_BLOCKS, KEY_RAW_ENTRIES, 4), dtype=np.uint8)
        m.nonkey_w = rng.integers(-2000, 2001, size=(NUM_PLIES, NONKEY_RAW_ENTRIES), dtype=np.int16)
        m.screlu_mu = rng.integers(8, 64, size=(NUM_PLIES, 3, 4), dtype=np.int16)
        m.screlu_beta = rng.integers(-400, 401, size=(NUM_PLIES, 3, 4), dtype=np.int16)
        m.macro_cw = rng.integers(-600, 601, size=(NUM_PLIES, 512), dtype=np.int16)
        m.macro_ea = rng.integers(-600, 601, size=(NUM_PLIES, 512), dtype=np.int16)
        m.macro_cd = rng.integers(-600, 601, size=(NUM_PLIES, 512), dtype=np.int16)
        m = symmetrize_p2c64_model(m, orbits)
        p = tmp_dir / f"synth_{mode.name.lower()}.p2c"
        save_p2c64(p, m)
        verify_p2c_file(bridge, p, black, white, stm, ply)

    # 3) WACA_G2 v3.0 (both H=16 and H=24) + Stage1 == RAW_NTUPLE_64 bit-exact check
    for h in (16, 24):
        wm = make_empty_waca_g2(mlp_hidden=h)
        wm.tier1_w = raw.w_ply.copy()
        wm.s25_w = rng.integers(-600, 601, size=(NUM_PLIES, WACA_F25), dtype=np.int16)
        wm.s25_b = rng.integers(-8000, 8001, size=(NUM_PLIES,), dtype=np.int32)
        wm.h_emb = rng.integers(-60, 61, size=(WACA_EO_BLOCKS, KEY_RAW_ENTRIES, 4), dtype=np.int8)
        wm.screlu_mu = rng.integers(8, 64, size=(NUM_PLIES, 3, 4), dtype=np.int16)
        wm.screlu_beta = rng.integers(-300, 301, size=(NUM_PLIES, 3, 4), dtype=np.int16)
        wm.mlp_w1 = rng.integers(-80, 81, size=(NUM_PLIES, h, WACA_F3), dtype=np.int8)
        wm.mlp_b1 = rng.integers(-8000, 8001, size=(NUM_PLIES, h), dtype=np.int32)
        wm.mlp_w2 = rng.integers(-800, 801, size=(NUM_PLIES, h), dtype=np.int16)
        wm.mlp_b2 = rng.integers(-8000, 8001, size=(NUM_PLIES,), dtype=np.int32)
        wm = symmetrize_waca_g2_model(wm, orbits)
        wp = tmp_dir / f"synth_waca_g2_h{h}.p2c"
        save_waca_g2(wp, wm)
        verify_p2c_file(bridge, wp, black, white, stm, ply)
        _, cpp_v1, _, _, _ = bridge.eval_waca_g2_file_stages(wp, black, white, stm, ply, check_16sym=False)
        t1_err = int(np.abs(sc_raw.astype(np.int64) - cpp_v1.astype(np.int64)).max())
        if t1_err != 0:
            print(f"[FAIL] synth RAW != WACA_G2(H={h}).Stage1 (max_err={t1_err})")
            sys.exit(1)
        print(f"[PASS] synth RAW_NTUPLE_64 == WACA_G2(H={h}).Stage1 bit-exact identity (max_err=0)")


def bench_bridge(bridge: P2C64Bridge, black, white, stm) -> None:
    reps = max(1, 100000 // len(black))
    b_big = np.tile(black, reps)
    w_big = np.tile(white, reps)
    s_big = np.tile(stm, reps)
    n = len(b_big)

    bridge.extract_batch_ptm(b_big[:1024], w_big[:1024], s_big[:1024], aug_seed=1, enable_aug=True, need_p2c=True)
    t0 = time.perf_counter()
    bridge.extract_batch_ptm(b_big, w_big, s_big, aug_seed=42, enable_aug=True, need_p2c=True)
    dt = time.perf_counter() - t0
    print(f"[BENCH] C++ OpenMP PTM Batch Bridge (aug=ON, 64-pat P2C v2.0): {n:,} pos in {dt*1000:.1f} ms -> {n/dt/1e6:.2f} M pos/s")


def main() -> None:
    ap = argparse.ArgumentParser(description="Verify C++/Python bit-exact .p2c v2.0/v3.0 I/O & 16-symmetry invariance")
    ap.add_argument("--data", type=Path, default=None)
    ap.add_argument("--num-pos", type=int, default=1000)
    ap.add_argument("--models", type=Path, nargs="*", default=None)
    ap.add_argument("--work-dir", type=Path, default=None)
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--bench-bridge", action="store_true")
    args = ap.parse_args()

    bridge = P2C64Bridge()
    black, white, stm, ply = load_positions(args.data, args.num_pos)

    print(f"=== P2C-64 v3.0 C++/Python Bit-Exact I/O & 16-Symmetry Test ({len(black)} positions) ===")
    files: List[Path] = []
    if args.models:
        files.extend(args.models)
    if args.work_dir:
        for name in (
            "raw_ntuple64.p2c",
            "linear64.p2c",
            "cand_a_ptm_screlu4.p2c",
            "cand_b_ptm_3w3b.p2c",
            "cand_c_ptm_hybrid.p2c",
            "waca_g2.p2c",
        ):
            p = args.work_dir / name
            if p.exists():
                files.append(p)

    if args.synthetic or not files:
        tmp_dir = (args.work_dir / "_synth_tmp") if args.work_dir else Path("build/_synth_tmp")
        run_synthetic_roundtrip(bridge, tmp_dir, black, white, stm, ply)

    scores_by_name = {}
    for p in files:
        scores_by_name[p.name] = verify_p2c_file(bridge, p, black, white, stm, ply)

    if "raw_ntuple64.p2c" in scores_by_name and "linear64.p2c" in scores_by_name:
        err = int(np.abs(scores_by_name["raw_ntuple64.p2c"].astype(np.int64) - scores_by_name["linear64.p2c"].astype(np.int64)).max())
        if err != 0:
            print(f"[FAIL] Trained RAW_NTUPLE_64 != LINEAR_64 (max_err={err})")
            sys.exit(1)
        print("[PASS] Trained RAW_NTUPLE_64 == LINEAR_64 bit-exact identity (max_err=0)")

    if args.bench_bridge:
        bench_bridge(bridge, black, white, stm)

    print("=== BIT-EXACT VERIFICATION PASSED (0 ERRORS) ===")


if __name__ == "__main__":
    main()
