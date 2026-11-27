"""P2C-64 v2.0 (Full PTM) binary (.p2c) I/O, D4 mirror symmetrization, and bit-exact integer simulator.

Guarantees 0-error bit-exact identity between Python integer simulation and C++
`evaluate_p2c64` / `eval_raw_ntuple64` across all 5 model modes.
"""

from __future__ import annotations

import ctypes
import struct
from dataclasses import dataclass
from enum import IntEnum
from pathlib import Path
from typing import Dict, Optional

import numpy as np

from p2c64_bridge import (
    EDGE_OFF,
    EDGE_RAW_SIZE,
    KEY_RAW_ENTRIES,
    NONKEY_RAW_ENTRIES,
    NUM_BLOCKS,
    NUM_PLIES,
    PLIES_PER_BLOCK,
    TOTAL_RAW_ENTRIES,
    P2C64Bridge,
)

P2C_MAGIC = b"P2C64V20"
P2C_FORMAT_VERSION = 2
P2C_MAGIC_V3 = b"P2C64V30"
P2C_FORMAT_VERSION_V3 = 3
P2C_MAGIC_V31 = b"P2C64V31"
P2C_FORMAT_VERSION_V31 = 4
P2C_MAGIC_V32 = b"P2C64V32"
P2C_FORMAT_VERSION_V32 = 5
P2C_MAGIC_V33 = b"P2C64V33"
P2C_FORMAT_VERSION_V33 = 6
P2C_DEFAULT_SCALE = 128.0
HEADER_STRUCT = struct.Struct("<8sIIfIIIIIIIQQ")
assert HEADER_STRUCT.size == 64

SCRELU_Q = 127
SCRELU_BETA_MAX = 2047
SCRELU_SHIFT = 7

# WACA-G2 v3.0 fixed-point constants (MUST match include/p2c64_eval.hpp bit-for-bit!)
WACA_EO_BLOCKS = 20
WACA_F25 = 16
WACA_F3 = 24
WACA_MAX_HID = 24
WACA_FEAT_CLAMP = 512
WACA_S25_W_MAX = 16384
WACA_S25_B_MAX = 1048576
WACA_S25_SHIFT = 10
WACA_MLP_W1_MAX = 127
WACA_MLP_B1_MAX = 1048576
WACA_MLP_S1 = 9
WACA_MLP_W2_MAX = 4095
WACA_MLP_B2_MAX = 1048576
WACA_MLP_S2 = 10
WACA_MAX_RES = 1024
PLY_TO_EO_BLK = np.array([(p // 6) * 2 + (p & 1) for p in range(NUM_PLIES)], dtype=np.int32)

# WACA-G2 v3.1 fixed-point constants
WACA_V31_EO_BLOCKS = 30
WACA_V31_F25 = 8
WACA_V31_F3_EXTRA = 20
WACA_V31_F3 = 32
WACA_V31_MAX_HID = 24
PLY_TO_EO30_BLK = np.array([(p // 4) * 2 + (p & 1) for p in range(NUM_PLIES)], dtype=np.int32)

# WACA-G2 v3.2 fixed-point constants
WACA_V32_EO_BLOCKS = 30
WACA_V32_F25 = 8
WACA_V32_F3_EXTRA = 16
WACA_V32_F3 = 28
WACA_V32_MAX_HID = 24

# WACA-G2 v3.3 fixed-point constants
WACA_V33_EO_BLOCKS = 30
WACA_V33_F25 = 12
WACA_V33_F3_EXTRA = 16
WACA_V33_F3 = 32
WACA_V33_MAX_HID = 24
WACA_V33_CTR_PATTERNS = 10
WACA_V33_CTR_RAW_ENTRIES = 19683
WACA_V33_SCRELU_TERMS = 44
WACA_V33_SCRELU_BETA_MAX = 1023

BLOCK_ENTRY_DTYPE = np.dtype([("w", "<i2", (6,)), ("aux", "u1", (4,))], align=False)
assert BLOCK_ENTRY_DTYPE.itemsize == 16


class EvalMode(IntEnum):
    RAW_NTUPLE_64 = 0
    LINEAR_64 = 1
    CAND_A_PTM_SCRELU4 = 2
    CAND_B_PTM_3W3B_NAIVE = 3
    CAND_B_PTM_3W3B = 4
    CAND_C_PTM_HYBRID = 5
    WACA_G2 = 6
    WACA_G2_V31 = 7
    WACA_G2_V32 = 8
    WACA_G2_V33 = 9


_DEFAULT_BRIDGE: Optional[P2C64Bridge] = None


def _get_default_bridge() -> P2C64Bridge:
    global _DEFAULT_BRIDGE
    if _DEFAULT_BRIDGE is None:
        _DEFAULT_BRIDGE = P2C64Bridge()
    return _DEFAULT_BRIDGE


def _fast_fnv1a_64(data: bytes) -> int:
    try:
        bridge = _get_default_bridge()
        return int(bridge.lib.p2c64_fnv1a_64(ctypes.c_char_p(data), ctypes.c_size_t(len(data))))
    except Exception:
        h = 14695981039346656037
        for b in data:
            h = ((h ^ b) * 1099511628211) & 0xFFFFFFFFFFFFFFFF
        return h


@dataclass
class RawNTupleModel:
    w_ply: np.ndarray  # [60, 410022] int16
    score_scale: float = P2C_DEFAULT_SCALE
    flags: int = 1


@dataclass
class P2C64Model:
    mode: EvalMode
    key_blocks: np.ndarray   # [10, 91854] BLOCK_ENTRY_DTYPE
    nonkey_w: np.ndarray     # [60, 318168] int16
    screlu_mu: np.ndarray    # [60, 3, 4] int16
    screlu_beta: np.ndarray  # [60, 3, 4] int16
    macro_cw: np.ndarray     # [60, 512] int16
    macro_ea: np.ndarray     # [60, 512] int16
    macro_cd: np.ndarray     # [60, 512] int16
    score_scale: float = P2C_DEFAULT_SCALE
    flags: int = 1


def make_empty_p2c64(mode: EvalMode, score_scale: float = P2C_DEFAULT_SCALE) -> P2C64Model:
    return P2C64Model(
        mode=mode,
        key_blocks=np.zeros((NUM_BLOCKS, KEY_RAW_ENTRIES), dtype=BLOCK_ENTRY_DTYPE),
        nonkey_w=np.zeros((NUM_PLIES, NONKEY_RAW_ENTRIES), dtype=np.int16),
        screlu_mu=np.full((NUM_PLIES, 3, 4), 16, dtype=np.int16),
        screlu_beta=np.zeros((NUM_PLIES, 3, 4), dtype=np.int16),
        macro_cw=np.zeros((NUM_PLIES, 512), dtype=np.int16),
        macro_ea=np.zeros((NUM_PLIES, 512), dtype=np.int16),
        macro_cd=np.zeros((NUM_PLIES, 512), dtype=np.int16),
        score_scale=score_scale,
        flags=1,
    )


@dataclass
class WacaG2Model:
    tier1_w: np.ndarray      # [60, 410022] int16
    s25_w: np.ndarray        # [60, 16] int16
    s25_b: np.ndarray        # [60] int32
    h_emb: np.ndarray        # [20, 91854, 4] int8 (Even/Odd 20-block Key-14)
    screlu_mu: np.ndarray    # [60, 3, 4] int16
    screlu_beta: np.ndarray  # [60, 3, 4] int16
    mlp_w1: np.ndarray       # [60, H, 24] int8
    mlp_b1: np.ndarray       # [60, H] int32
    mlp_w2: np.ndarray       # [60, H] int16
    mlp_b2: np.ndarray       # [60] int32
    mlp_hidden: int = 16
    score_scale: float = P2C_DEFAULT_SCALE
    flags: int = 1


def make_empty_waca_g2(mlp_hidden: int = 16, score_scale: float = P2C_DEFAULT_SCALE) -> WacaG2Model:
    h = int(np.clip(mlp_hidden, 1, WACA_MAX_HID))
    return WacaG2Model(
        tier1_w=np.zeros((NUM_PLIES, TOTAL_RAW_ENTRIES), dtype=np.int16),
        s25_w=np.zeros((NUM_PLIES, WACA_F25), dtype=np.int16),
        s25_b=np.zeros((NUM_PLIES,), dtype=np.int32),
        h_emb=np.zeros((WACA_EO_BLOCKS, KEY_RAW_ENTRIES, 4), dtype=np.int8),
        screlu_mu=np.full((NUM_PLIES, 3, 4), 16, dtype=np.int16),
        screlu_beta=np.zeros((NUM_PLIES, 3, 4), dtype=np.int16),
        mlp_w1=np.zeros((NUM_PLIES, h, WACA_F3), dtype=np.int8),
        mlp_b1=np.zeros((NUM_PLIES, h), dtype=np.int32),
        mlp_w2=np.zeros((NUM_PLIES, h), dtype=np.int16),
        mlp_b2=np.zeros((NUM_PLIES,), dtype=np.int32),
        mlp_hidden=h,
        score_scale=score_scale,
        flags=1,
    )


@dataclass
class WacaG2V31Model:
    tier1_w: np.ndarray      # [60, 410022] int16
    s25_w: np.ndarray        # [60, 8] int16
    s25_b: np.ndarray        # [60] int32
    h_emb: np.ndarray        # [30, 91854, 4] int8 (Even/Odd 30-block Key-14)
    screlu_mu: np.ndarray    # [60, 3, 4] int16
    screlu_beta: np.ndarray  # [60, 3, 4] int16
    mlp_w1: np.ndarray       # [60, H, 32] int8
    mlp_b1: np.ndarray       # [60, H] int32
    mlp_w2: np.ndarray       # [60, H] int16
    mlp_b2: np.ndarray       # [60] int32
    mlp_hidden: int = 16
    score_scale: float = P2C_DEFAULT_SCALE
    flags: int = 1


def make_empty_waca_g2_v31(mlp_hidden: int = 16, score_scale: float = P2C_DEFAULT_SCALE) -> WacaG2V31Model:
    h = int(np.clip(mlp_hidden, 1, WACA_V31_MAX_HID))
    return WacaG2V31Model(
        tier1_w=np.zeros((NUM_PLIES, TOTAL_RAW_ENTRIES), dtype=np.int16),
        s25_w=np.zeros((NUM_PLIES, WACA_V31_F25), dtype=np.int16),
        s25_b=np.zeros((NUM_PLIES,), dtype=np.int32),
        h_emb=np.zeros((WACA_V31_EO_BLOCKS, KEY_RAW_ENTRIES, 4), dtype=np.int8),
        screlu_mu=np.full((NUM_PLIES, 3, 4), 16, dtype=np.int16),
        screlu_beta=np.zeros((NUM_PLIES, 3, 4), dtype=np.int16),
        mlp_w1=np.zeros((NUM_PLIES, h, WACA_V31_F3), dtype=np.int8),
        mlp_b1=np.zeros((NUM_PLIES, h), dtype=np.int32),
        mlp_w2=np.zeros((NUM_PLIES, h), dtype=np.int16),
        mlp_b2=np.zeros((NUM_PLIES,), dtype=np.int32),
        mlp_hidden=h,
        score_scale=score_scale,
        flags=1,
    )


@dataclass
class WacaG2V32Model:
    tier1_w: np.ndarray      # [60, 410022] int16
    s25_w: np.ndarray        # [60, 8] int16
    s25_b: np.ndarray        # [60] int32
    h_emb: np.ndarray        # [30, 91854, 4] int8 (Even/Odd 30-block Key-14)
    screlu_mu: np.ndarray    # [60, 3, 4] int16
    screlu_beta: np.ndarray  # [60, 3, 4] int16
    mlp_w1: np.ndarray       # [60, H, 28] int8
    mlp_b1: np.ndarray       # [60, H] int32
    mlp_w2: np.ndarray       # [60, H] int16
    mlp_b2: np.ndarray       # [60] int32
    mlp_hidden: int = 16
    score_scale: float = P2C_DEFAULT_SCALE
    flags: int = 1


def make_empty_waca_g2_v32(mlp_hidden: int = 16, score_scale: float = P2C_DEFAULT_SCALE) -> WacaG2V32Model:
    h = int(np.clip(mlp_hidden, 1, WACA_V32_MAX_HID))
    return WacaG2V32Model(
        tier1_w=np.zeros((NUM_PLIES, TOTAL_RAW_ENTRIES), dtype=np.int16),
        s25_w=np.zeros((NUM_PLIES, WACA_V32_F25), dtype=np.int16),
        s25_b=np.zeros((NUM_PLIES,), dtype=np.int32),
        h_emb=np.zeros((WACA_V32_EO_BLOCKS, KEY_RAW_ENTRIES, 4), dtype=np.int8),
        screlu_mu=np.full((NUM_PLIES, 3, 4), 16, dtype=np.int16),
        screlu_beta=np.zeros((NUM_PLIES, 3, 4), dtype=np.int16),
        mlp_w1=np.zeros((NUM_PLIES, h, WACA_V32_F3), dtype=np.int8),
        mlp_b1=np.zeros((NUM_PLIES, h), dtype=np.int32),
        mlp_w2=np.zeros((NUM_PLIES, h), dtype=np.int16),
        mlp_b2=np.zeros((NUM_PLIES,), dtype=np.int32),
        mlp_hidden=h,
        score_scale=score_scale,
        flags=1,
    )


@dataclass
class WacaG2V33Model:
    tier1_w: np.ndarray        # [60, 410022] int16
    s25_w: np.ndarray          # [60, 12] int16
    s25_b: np.ndarray          # [60] int32
    h_emb: np.ndarray          # [30, 91854, 8] int8 (Even/Odd 30-block Key-14 8ch)
    ctr_emb: np.ndarray        # [30, 19683, 4] int8 (Even/Odd 30-block Center-10 4ch)
    screlu_mu_key: np.ndarray  # [60, 3, 8] int16
    screlu_mu_ctr: np.ndarray  # [60, 4] int16
    screlu_beta: np.ndarray    # [60, 44] int16
    mlp_w1: np.ndarray         # [60, H, 32] int8
    mlp_b1: np.ndarray         # [60, H] int32
    mlp_w2: np.ndarray         # [60, H] int16
    mlp_b2: np.ndarray         # [60] int32
    mlp_hidden: int = 16
    score_scale: float = P2C_DEFAULT_SCALE
    flags: int = 1


def make_empty_waca_g2_v33(mlp_hidden: int = 16, score_scale: float = P2C_DEFAULT_SCALE) -> WacaG2V33Model:
    h = int(np.clip(mlp_hidden, 1, WACA_V33_MAX_HID))
    return WacaG2V33Model(
        tier1_w=np.zeros((NUM_PLIES, TOTAL_RAW_ENTRIES), dtype=np.int16),
        s25_w=np.zeros((NUM_PLIES, WACA_V33_F25), dtype=np.int16),
        s25_b=np.zeros((NUM_PLIES,), dtype=np.int32),
        h_emb=np.zeros((WACA_V33_EO_BLOCKS, KEY_RAW_ENTRIES, 8), dtype=np.int8),
        ctr_emb=np.zeros((WACA_V33_EO_BLOCKS, WACA_V33_CTR_RAW_ENTRIES, 4), dtype=np.int8),
        screlu_mu_key=np.full((NUM_PLIES, 3, 8), 16, dtype=np.int16),
        screlu_mu_ctr=np.full((NUM_PLIES, 4), 16, dtype=np.int16),
        screlu_beta=np.zeros((NUM_PLIES, WACA_V33_SCRELU_TERMS), dtype=np.int16),
        mlp_w1=np.zeros((NUM_PLIES, h, WACA_V33_F3), dtype=np.int8),
        mlp_b1=np.zeros((NUM_PLIES, h), dtype=np.int32),
        mlp_w2=np.zeros((NUM_PLIES, h), dtype=np.int16),
        mlp_b2=np.zeros((NUM_PLIES,), dtype=np.int32),
        mlp_hidden=h,
        score_scale=score_scale,
        flags=1,
    )


def raw_to_linear_p2c64(raw: RawNTupleModel) -> P2C64Model:
    """Create a LINEAR_64 P2C64Model with bit-identical linear weights to RawNTupleModel."""
    m = make_empty_p2c64(EvalMode.LINEAR_64, score_scale=raw.score_scale)
    w_key = raw.w_ply[:, :KEY_RAW_ENTRIES]  # [60, 91854]
    m.key_blocks["w"] = w_key.reshape(NUM_BLOCKS, PLIES_PER_BLOCK, KEY_RAW_ENTRIES).transpose(0, 2, 1).copy()
    m.nonkey_w = raw.w_ply[:, KEY_RAW_ENTRIES:].copy()
    m.flags = raw.flags
    return m


def symmetrize_macro_table(tbl: np.ndarray, swap_mode: str) -> np.ndarray:
    """Enforce exact D4 mirror symmetry on a [60, 512] int16 macro table."""
    arr = np.array(tbl, dtype=np.int16, copy=True).reshape(NUM_PLIES, 8, 8, 8)
    if swap_mode == "cw":
        arr_t = arr.transpose(0, 1, 3, 2)
    elif swap_mode == "ea":
        arr_t = arr.transpose(0, 2, 1, 3)
    elif swap_mode == "cd":
        arr_t = arr.transpose(0, 3, 2, 1)
    else:
        raise ValueError(f"Unknown swap_mode: {swap_mode}")
    avg = np.round(0.5 * (arr.astype(np.float32) + arr_t.astype(np.float32)))
    return np.clip(avg, -32767, 32767).astype(np.int16).reshape(NUM_PLIES, 512)


def symmetrize_raw_model(model: RawNTupleModel, orbits: Dict[str, np.ndarray]) -> RawNTupleModel:
    raw_rep = orbits["raw_orb_rep"]
    w = model.w_ply[:, raw_rep].astype(np.int16, copy=True)
    return RawNTupleModel(w_ply=w, score_scale=model.score_scale, flags=1)


def symmetrize_p2c64_model(model: P2C64Model, orbits: Dict[str, np.ndarray]) -> P2C64Model:
    """Enforce exact D4 internal mirror invariance matching C++ P2C64Weights::symmetrize_for_mode."""
    k_rep = orbits["key_orb_rep"]
    e_mx = orbits["edge_mirror_x"]
    nk_rep = orbits["nk_orb_rep"]

    kb = model.key_blocks.copy()
    kb["w"] = kb["w"][:, k_rep, :]
    nk_sym = model.nonkey_w[:, nk_rep].astype(np.int16, copy=True)

    if model.mode == EvalMode.CAND_A_PTM_SCRELU4:
        aux_i8 = np.clip(kb["aux"].view(np.int8).astype(np.int32), -127, 127).astype(np.int8)
        kb["aux"] = aux_i8[:, k_rep, :].view(np.uint8)
    elif model.mode in (EvalMode.CAND_B_PTM_3W3B, EvalMode.CAND_B_PTM_3W3B_NAIVE):
        zf = kb["aux"][:, :, 0] & 7
        zr = kb["aux"][:, :, 1] & 7
        z_single = zf[:, k_rep]
        kb["aux"][:, :, 0] = z_single
        kb["aux"][:, :, 1] = z_single
        kb["aux"][:, :, 2] = 0
        kb["aux"][:, :, 3] = 0
        # Edge+2X directional: code_fwd(rev(x)) == code_rev(x)
        ezf = zf[:, EDGE_OFF : EDGE_OFF + EDGE_RAW_SIZE].copy()
        ezr = zr[:, EDGE_OFF : EDGE_OFF + EDGE_RAW_SIZE].copy()
        idx_arr = np.arange(EDGE_RAW_SIZE, dtype=np.int32)
        rep_e = np.minimum(idx_arr, e_mx)
        is_swapped = idx_arr > e_mx
        is_self = idx_arr == e_mx
        f0 = ezf[:, rep_e]
        r0 = ezr[:, rep_e]
        out_f = np.where(is_swapped[None, :], r0, f0)
        out_r = np.where(is_swapped[None, :], f0, r0)
        out_r = np.where(is_self[None, :], out_f, out_r)
        kb["aux"][:, EDGE_OFF : EDGE_OFF + EDGE_RAW_SIZE, 0] = out_f.astype(np.uint8)
        kb["aux"][:, EDGE_OFF : EDGE_OFF + EDGE_RAW_SIZE, 1] = out_r.astype(np.uint8)
    elif model.mode == EvalMode.CAND_C_PTM_HYBRID:
        aux_i8 = np.clip(kb["aux"][:, :, :2].view(np.int8).astype(np.int32), -127, 127).astype(np.int8)
        kb["aux"][:, :, :2] = aux_i8[:, k_rep, :].view(np.uint8)
        pk = kb["aux"][:, :, 2].astype(np.int32)
        zf = pk & 7
        zr = (pk >> 3) & 7
        z_single = zf[:, k_rep]
        out_pk = z_single | (z_single << 3)
        ezf = zf[:, EDGE_OFF : EDGE_OFF + EDGE_RAW_SIZE]
        ezr = zr[:, EDGE_OFF : EDGE_OFF + EDGE_RAW_SIZE]
        idx_arr = np.arange(EDGE_RAW_SIZE, dtype=np.int32)
        rep_e = np.minimum(idx_arr, e_mx)
        is_swapped = idx_arr > e_mx
        is_self = idx_arr == e_mx
        f0 = ezf[:, rep_e]
        r0 = ezr[:, rep_e]
        out_f = np.where(is_swapped[None, :], r0, f0)
        out_r = np.where(is_swapped[None, :], f0, r0)
        out_r = np.where(is_self[None, :], out_f, out_r)
        out_pk[:, EDGE_OFF : EDGE_OFF + EDGE_RAW_SIZE] = out_f | (out_r << 3)
        kb["aux"][:, :, 2] = out_pk.astype(np.uint8)
        kb["aux"][:, :, 3] = 0

    sc_beta = np.clip(model.screlu_beta.astype(np.int32), -SCRELU_BETA_MAX, SCRELU_BETA_MAX).astype(np.int16)
    return P2C64Model(
        mode=model.mode,
        key_blocks=kb,
        nonkey_w=nk_sym,
        screlu_mu=model.screlu_mu.astype(np.int16, copy=True),
        screlu_beta=sc_beta,
        macro_cw=symmetrize_macro_table(model.macro_cw, "cw"),
        macro_ea=symmetrize_macro_table(model.macro_ea, "ea"),
        macro_cd=symmetrize_macro_table(model.macro_cd, "cd"),
        score_scale=model.score_scale,
        flags=1,
    )


def save_raw_p2c(path: str | Path, model: RawNTupleModel) -> None:
    w = np.ascontiguousarray(model.w_ply, dtype="<i2")
    assert w.shape == (NUM_PLIES, TOTAL_RAW_ENTRIES)
    payload = w.tobytes()
    chk = _fast_fnv1a_64(payload)
    hdr = HEADER_STRUCT.pack(
        P2C_MAGIC,
        P2C_FORMAT_VERSION,
        int(EvalMode.RAW_NTUPLE_64),
        float(model.score_scale),
        int(model.flags),
        KEY_RAW_ENTRIES,
        NONKEY_RAW_ENTRIES,
        TOTAL_RAW_ENTRIES,
        NUM_PLIES,
        NUM_BLOCKS,
        0,
        len(payload),
        chk,
    )
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        f.write(hdr)
        f.write(payload)


def save_p2c64(path: str | Path, model: P2C64Model) -> None:
    assert model.mode != EvalMode.RAW_NTUPLE_64
    kb = np.ascontiguousarray(model.key_blocks, dtype=BLOCK_ENTRY_DTYPE)
    nk = np.ascontiguousarray(model.nonkey_w, dtype="<i2")
    assert kb.shape == (NUM_BLOCKS, KEY_RAW_ENTRIES)
    assert nk.shape == (NUM_PLIES, NONKEY_RAW_ENTRIES)

    sc_chunks = []
    for ply in range(NUM_PLIES):
        sc_chunks.append(np.ascontiguousarray(model.screlu_mu[ply], dtype="<i2").tobytes())
        sc_chunks.append(np.ascontiguousarray(model.screlu_beta[ply], dtype="<i2").tobytes())

    ml_chunks = []
    for ply in range(NUM_PLIES):
        ml_chunks.append(np.ascontiguousarray(model.macro_cw[ply], dtype="<i2").tobytes())
        ml_chunks.append(np.ascontiguousarray(model.macro_ea[ply], dtype="<i2").tobytes())
        ml_chunks.append(np.ascontiguousarray(model.macro_cd[ply], dtype="<i2").tobytes())

    payload = kb.tobytes() + nk.tobytes() + b"".join(sc_chunks) + b"".join(ml_chunks)
    chk = _fast_fnv1a_64(payload)
    hdr = HEADER_STRUCT.pack(
        P2C_MAGIC,
        P2C_FORMAT_VERSION,
        int(model.mode),
        float(model.score_scale),
        int(model.flags),
        KEY_RAW_ENTRIES,
        NONKEY_RAW_ENTRIES,
        TOTAL_RAW_ENTRIES,
        NUM_PLIES,
        NUM_BLOCKS,
        0,
        len(payload),
        chk,
    )
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        f.write(hdr)
        f.write(payload)


def symmetrize_waca_g2_model(model: WacaG2Model, orbits: Dict[str, np.ndarray]) -> WacaG2Model:
    """Enforce exact D4 internal mirror invariance & overflow-safe parameter bounds matching C++ WacaG2Weights::symmetrize."""
    raw_rep = orbits["raw_orb_rep"]
    k_rep = orbits["key_orb_rep"]
    h = int(np.clip(model.mlp_hidden, 1, WACA_MAX_HID))

    t1 = model.tier1_w[:, raw_rep].astype(np.int16, copy=True)
    h_i8 = np.clip(model.h_emb.astype(np.int32), -127, 127).astype(np.int8)
    h_sym = h_i8[:, k_rep, :].copy()

    s25_w = np.clip(model.s25_w.astype(np.int32), -WACA_S25_W_MAX, WACA_S25_W_MAX).astype(np.int16)
    s25_b = np.clip(model.s25_b.astype(np.int64), -WACA_S25_B_MAX, WACA_S25_B_MAX).astype(np.int32)
    sc_mu = model.screlu_mu.astype(np.int16, copy=True)
    sc_beta = np.clip(model.screlu_beta.astype(np.int32), -SCRELU_BETA_MAX, SCRELU_BETA_MAX).astype(np.int16)
    w1 = np.clip(model.mlp_w1[:, :h, :].astype(np.int32), -WACA_MLP_W1_MAX, WACA_MLP_W1_MAX).astype(np.int8)
    b1 = np.clip(model.mlp_b1[:, :h].astype(np.int64), -WACA_MLP_B1_MAX, WACA_MLP_B1_MAX).astype(np.int32)
    w2 = np.clip(model.mlp_w2[:, :h].astype(np.int32), -WACA_MLP_W2_MAX, WACA_MLP_W2_MAX).astype(np.int16)
    b2 = np.clip(model.mlp_b2.astype(np.int64), -WACA_MLP_B2_MAX, WACA_MLP_B2_MAX).astype(np.int32)

    return WacaG2Model(
        tier1_w=t1,
        s25_w=s25_w,
        s25_b=s25_b,
        h_emb=h_sym,
        screlu_mu=sc_mu,
        screlu_beta=sc_beta,
        mlp_w1=w1,
        mlp_b1=b1,
        mlp_w2=w2,
        mlp_b2=b2,
        mlp_hidden=h,
        score_scale=model.score_scale,
        flags=1,
    )


def save_waca_g2(path: str | Path, model: WacaG2Model) -> None:
    h = int(np.clip(model.mlp_hidden, 1, WACA_MAX_HID))
    t1 = np.ascontiguousarray(model.tier1_w, dtype="<i2")
    s25_w = np.ascontiguousarray(model.s25_w, dtype="<i2")
    s25_b = np.ascontiguousarray(model.s25_b, dtype="<i4")
    h_emb = np.ascontiguousarray(model.h_emb, dtype="i1")
    w1 = np.ascontiguousarray(model.mlp_w1[:, :h, :], dtype="i1")
    b1 = np.ascontiguousarray(model.mlp_b1[:, :h], dtype="<i4")
    w2 = np.ascontiguousarray(model.mlp_w2[:, :h], dtype="<i2")
    b2 = np.ascontiguousarray(model.mlp_b2, dtype="<i4")

    assert t1.shape == (NUM_PLIES, TOTAL_RAW_ENTRIES)
    assert s25_w.shape == (NUM_PLIES, WACA_F25)
    assert s25_b.shape == (NUM_PLIES,)
    assert h_emb.shape == (WACA_EO_BLOCKS, KEY_RAW_ENTRIES, 4)
    assert w1.shape == (NUM_PLIES, h, WACA_F3)
    assert b1.shape == (NUM_PLIES, h)
    assert w2.shape == (NUM_PLIES, h)
    assert b2.shape == (NUM_PLIES,)

    sc_chunks = []
    for ply in range(NUM_PLIES):
        sc_chunks.append(np.ascontiguousarray(model.screlu_mu[ply], dtype="<i2").tobytes())
        sc_chunks.append(np.ascontiguousarray(model.screlu_beta[ply], dtype="<i2").tobytes())

    payload = (
        t1.tobytes()
        + s25_w.tobytes()
        + s25_b.tobytes()
        + h_emb.tobytes()
        + b"".join(sc_chunks)
        + w1.tobytes()
        + b1.tobytes()
        + w2.tobytes()
        + b2.tobytes()
    )
    chk = _fast_fnv1a_64(payload)
    hdr = HEADER_STRUCT.pack(
        P2C_MAGIC_V3,
        P2C_FORMAT_VERSION_V3,
        int(EvalMode.WACA_G2),
        float(model.score_scale),
        int(model.flags),
        KEY_RAW_ENTRIES,
        NONKEY_RAW_ENTRIES,
        TOTAL_RAW_ENTRIES,
        NUM_PLIES,
        WACA_EO_BLOCKS,
        h,
        len(payload),
        chk,
    )
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        f.write(hdr)
        f.write(payload)


def symmetrize_waca_g2_v31_model(model: WacaG2V31Model, orbits: Dict[str, np.ndarray]) -> WacaG2V31Model:
    """Enforce exact D4 internal mirror invariance & overflow-safe bounds matching C++ WacaG2V31Weights::symmetrize."""
    raw_rep = orbits["raw_orb_rep"]
    k_rep = orbits["key_orb_rep"]
    h = int(np.clip(model.mlp_hidden, 1, WACA_V31_MAX_HID))

    t1 = model.tier1_w[:, raw_rep].astype(np.int16, copy=True)
    h_i8 = np.clip(model.h_emb.astype(np.int32), -127, 127).astype(np.int8)
    h_sym = h_i8[:, k_rep, :].copy()

    s25_w = np.clip(model.s25_w.astype(np.int32), -WACA_S25_W_MAX, WACA_S25_W_MAX).astype(np.int16)
    s25_b = np.clip(model.s25_b.astype(np.int64), -WACA_S25_B_MAX, WACA_S25_B_MAX).astype(np.int32)
    sc_mu = model.screlu_mu.astype(np.int16, copy=True)
    sc_beta = np.clip(model.screlu_beta.astype(np.int32), -SCRELU_BETA_MAX, SCRELU_BETA_MAX).astype(np.int16)
    w1 = np.clip(model.mlp_w1[:, :h, :].astype(np.int32), -WACA_MLP_W1_MAX, WACA_MLP_W1_MAX).astype(np.int8)
    b1 = np.clip(model.mlp_b1[:, :h].astype(np.int64), -WACA_MLP_B1_MAX, WACA_MLP_B1_MAX).astype(np.int32)
    w2 = np.clip(model.mlp_w2[:, :h].astype(np.int32), -WACA_MLP_W2_MAX, WACA_MLP_W2_MAX).astype(np.int16)
    b2 = np.clip(model.mlp_b2.astype(np.int64), -WACA_MLP_B2_MAX, WACA_MLP_B2_MAX).astype(np.int32)

    return WacaG2V31Model(
        tier1_w=t1,
        s25_w=s25_w,
        s25_b=s25_b,
        h_emb=h_sym,
        screlu_mu=sc_mu,
        screlu_beta=sc_beta,
        mlp_w1=w1,
        mlp_b1=b1,
        mlp_w2=w2,
        mlp_b2=b2,
        mlp_hidden=h,
        score_scale=model.score_scale,
        flags=1,
    )


def save_waca_g2_v31(path: str | Path, model: WacaG2V31Model) -> None:
    h = int(np.clip(model.mlp_hidden, 1, WACA_V31_MAX_HID))
    t1 = np.ascontiguousarray(model.tier1_w, dtype="<i2")
    s25_w = np.ascontiguousarray(model.s25_w, dtype="<i2")
    s25_b = np.ascontiguousarray(model.s25_b, dtype="<i4")
    h_emb = np.ascontiguousarray(model.h_emb, dtype="i1")
    w1 = np.ascontiguousarray(model.mlp_w1[:, :h, :], dtype="i1")
    b1 = np.ascontiguousarray(model.mlp_b1[:, :h], dtype="<i4")
    w2 = np.ascontiguousarray(model.mlp_w2[:, :h], dtype="<i2")
    b2 = np.ascontiguousarray(model.mlp_b2, dtype="<i4")

    assert t1.shape == (NUM_PLIES, TOTAL_RAW_ENTRIES)
    assert s25_w.shape == (NUM_PLIES, WACA_V31_F25)
    assert s25_b.shape == (NUM_PLIES,)
    assert h_emb.shape == (WACA_V31_EO_BLOCKS, KEY_RAW_ENTRIES, 4)
    assert w1.shape == (NUM_PLIES, h, WACA_V31_F3)
    assert b1.shape == (NUM_PLIES, h)
    assert w2.shape == (NUM_PLIES, h)
    assert b2.shape == (NUM_PLIES,)

    sc_chunks = []
    for ply in range(NUM_PLIES):
        sc_chunks.append(np.ascontiguousarray(model.screlu_mu[ply], dtype="<i2").tobytes())
        sc_chunks.append(np.ascontiguousarray(model.screlu_beta[ply], dtype="<i2").tobytes())

    payload = (
        t1.tobytes()
        + s25_w.tobytes()
        + s25_b.tobytes()
        + h_emb.tobytes()
        + b"".join(sc_chunks)
        + w1.tobytes()
        + b1.tobytes()
        + w2.tobytes()
        + b2.tobytes()
    )
    chk = _fast_fnv1a_64(payload)
    hdr = HEADER_STRUCT.pack(
        P2C_MAGIC_V31,
        P2C_FORMAT_VERSION_V31,
        int(EvalMode.WACA_G2_V31),
        float(model.score_scale),
        int(model.flags),
        KEY_RAW_ENTRIES,
        NONKEY_RAW_ENTRIES,
        TOTAL_RAW_ENTRIES,
        NUM_PLIES,
        WACA_V31_EO_BLOCKS,
        h,
        len(payload),
        chk,
    )
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        f.write(hdr)
        f.write(payload)


def symmetrize_waca_g2_v32_model(model: WacaG2V32Model, orbits: Dict[str, np.ndarray]) -> WacaG2V32Model:
    """Enforce exact D4 internal mirror invariance & overflow-safe bounds matching C++ WacaG2V32Weights::symmetrize."""
    raw_rep = orbits["raw_orb_rep"]
    k_rep = orbits["key_orb_rep"]
    h = int(np.clip(model.mlp_hidden, 1, WACA_V32_MAX_HID))

    t1 = model.tier1_w[:, raw_rep].astype(np.int16, copy=True)
    h_i8 = np.clip(model.h_emb.astype(np.int32), -127, 127).astype(np.int8)
    h_sym = h_i8[:, k_rep, :].copy()

    s25_w = np.clip(model.s25_w.astype(np.int32), -WACA_S25_W_MAX, WACA_S25_W_MAX).astype(np.int16)
    s25_b = np.clip(model.s25_b.astype(np.int64), -WACA_S25_B_MAX, WACA_S25_B_MAX).astype(np.int32)
    sc_mu = model.screlu_mu.astype(np.int16, copy=True)
    sc_beta = np.clip(model.screlu_beta.astype(np.int32), -SCRELU_BETA_MAX, SCRELU_BETA_MAX).astype(np.int16)
    w1 = np.clip(model.mlp_w1[:, :h, :].astype(np.int32), -WACA_MLP_W1_MAX, WACA_MLP_W1_MAX).astype(np.int8)
    b1 = np.clip(model.mlp_b1[:, :h].astype(np.int64), -WACA_MLP_B1_MAX, WACA_MLP_B1_MAX).astype(np.int32)
    w2 = np.clip(model.mlp_w2[:, :h].astype(np.int32), -WACA_MLP_W2_MAX, WACA_MLP_W2_MAX).astype(np.int16)
    b2 = np.clip(model.mlp_b2.astype(np.int64), -WACA_MLP_B2_MAX, WACA_MLP_B2_MAX).astype(np.int32)

    return WacaG2V32Model(
        tier1_w=t1,
        s25_w=s25_w,
        s25_b=s25_b,
        h_emb=h_sym,
        screlu_mu=sc_mu,
        screlu_beta=sc_beta,
        mlp_w1=w1,
        mlp_b1=b1,
        mlp_w2=w2,
        mlp_b2=b2,
        mlp_hidden=h,
        score_scale=model.score_scale,
        flags=1,
    )


def save_waca_g2_v32(path: str | Path, model: WacaG2V32Model) -> None:
    h = int(np.clip(model.mlp_hidden, 1, WACA_V32_MAX_HID))
    t1 = np.ascontiguousarray(model.tier1_w, dtype="<i2")
    s25_w = np.ascontiguousarray(model.s25_w, dtype="<i2")
    s25_b = np.ascontiguousarray(model.s25_b, dtype="<i4")
    h_emb = np.ascontiguousarray(model.h_emb, dtype="i1")
    w1 = np.ascontiguousarray(model.mlp_w1[:, :h, :], dtype="i1")
    b1 = np.ascontiguousarray(model.mlp_b1[:, :h], dtype="<i4")
    w2 = np.ascontiguousarray(model.mlp_w2[:, :h], dtype="<i2")
    b2 = np.ascontiguousarray(model.mlp_b2, dtype="<i4")

    assert t1.shape == (NUM_PLIES, TOTAL_RAW_ENTRIES)
    assert s25_w.shape == (NUM_PLIES, WACA_V32_F25)
    assert s25_b.shape == (NUM_PLIES,)
    assert h_emb.shape == (WACA_V32_EO_BLOCKS, KEY_RAW_ENTRIES, 4)
    assert w1.shape == (NUM_PLIES, h, WACA_V32_F3)
    assert b1.shape == (NUM_PLIES, h)
    assert w2.shape == (NUM_PLIES, h)
    assert b2.shape == (NUM_PLIES,)

    sc_chunks = []
    for ply in range(NUM_PLIES):
        sc_chunks.append(np.ascontiguousarray(model.screlu_mu[ply], dtype="<i2").tobytes())
        sc_chunks.append(np.ascontiguousarray(model.screlu_beta[ply], dtype="<i2").tobytes())

    payload = (
        t1.tobytes()
        + s25_w.tobytes()
        + s25_b.tobytes()
        + h_emb.tobytes()
        + b"".join(sc_chunks)
        + w1.tobytes()
        + b1.tobytes()
        + w2.tobytes()
        + b2.tobytes()
    )
    chk = _fast_fnv1a_64(payload)
    hdr = HEADER_STRUCT.pack(
        P2C_MAGIC_V32,
        P2C_FORMAT_VERSION_V32,
        int(EvalMode.WACA_G2_V32),
        float(model.score_scale),
        int(model.flags),
        KEY_RAW_ENTRIES,
        NONKEY_RAW_ENTRIES,
        TOTAL_RAW_ENTRIES,
        NUM_PLIES,
        WACA_V32_EO_BLOCKS,
        h,
        len(payload),
        chk,
    )
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        f.write(hdr)
        f.write(payload)


def symmetrize_waca_g2_v33_model(model: WacaG2V33Model, orbits: Dict[str, np.ndarray]) -> WacaG2V33Model:
    """Enforce exact D4 internal mirror invariance & overflow-safe bounds matching C++ WacaG2V33Weights::symmetrize."""
    raw_rep = orbits["raw_orb_rep"]
    k_rep = orbits["key_orb_rep"]
    c_rep = orbits["ctr_orb_rep"]
    h = int(np.clip(model.mlp_hidden, 1, WACA_V33_MAX_HID))

    t1 = model.tier1_w[:, raw_rep].astype(np.int16, copy=True)
    h_i8 = np.clip(model.h_emb.astype(np.int32), -127, 127).astype(np.int8)
    h_sym = h_i8[:, k_rep, :].copy()
    c_i8 = np.clip(model.ctr_emb.astype(np.int32), -127, 127).astype(np.int8)
    c_sym = c_i8[:, c_rep, :].copy()

    s25_w = np.clip(model.s25_w.astype(np.int32), -WACA_S25_W_MAX, WACA_S25_W_MAX).astype(np.int16)
    s25_b = np.clip(model.s25_b.astype(np.int64), -WACA_S25_B_MAX, WACA_S25_B_MAX).astype(np.int32)
    sc_mu_k = model.screlu_mu_key.astype(np.int16, copy=True)
    sc_mu_c = model.screlu_mu_ctr.astype(np.int16, copy=True)
    sc_beta = np.clip(
        model.screlu_beta.astype(np.int32), -WACA_V33_SCRELU_BETA_MAX, WACA_V33_SCRELU_BETA_MAX
    ).astype(np.int16)
    w1 = np.clip(model.mlp_w1[:, :h, :].astype(np.int32), -WACA_MLP_W1_MAX, WACA_MLP_W1_MAX).astype(np.int8)
    b1 = np.clip(model.mlp_b1[:, :h].astype(np.int64), -WACA_MLP_B1_MAX, WACA_MLP_B1_MAX).astype(np.int32)
    w2 = np.clip(model.mlp_w2[:, :h].astype(np.int32), -WACA_MLP_W2_MAX, WACA_MLP_W2_MAX).astype(np.int16)
    b2 = np.clip(model.mlp_b2.astype(np.int64), -WACA_MLP_B2_MAX, WACA_MLP_B2_MAX).astype(np.int32)

    return WacaG2V33Model(
        tier1_w=t1,
        s25_w=s25_w,
        s25_b=s25_b,
        h_emb=h_sym,
        ctr_emb=c_sym,
        screlu_mu_key=sc_mu_k,
        screlu_mu_ctr=sc_mu_c,
        screlu_beta=sc_beta,
        mlp_w1=w1,
        mlp_b1=b1,
        mlp_w2=w2,
        mlp_b2=b2,
        mlp_hidden=h,
        score_scale=model.score_scale,
        flags=1,
    )


def save_waca_g2_v33(path: str | Path, model: WacaG2V33Model) -> None:
    h = int(np.clip(model.mlp_hidden, 1, WACA_V33_MAX_HID))
    t1 = np.ascontiguousarray(model.tier1_w, dtype="<i2")
    s25_w = np.ascontiguousarray(model.s25_w, dtype="<i2")
    s25_b = np.ascontiguousarray(model.s25_b, dtype="<i4")
    h_emb = np.ascontiguousarray(model.h_emb, dtype="i1")
    ctr_emb = np.ascontiguousarray(model.ctr_emb, dtype="i1")
    w1 = np.ascontiguousarray(model.mlp_w1[:, :h, :], dtype="i1")
    b1 = np.ascontiguousarray(model.mlp_b1[:, :h], dtype="<i4")
    w2 = np.ascontiguousarray(model.mlp_w2[:, :h], dtype="<i2")
    b2 = np.ascontiguousarray(model.mlp_b2, dtype="<i4")

    assert t1.shape == (NUM_PLIES, TOTAL_RAW_ENTRIES)
    assert s25_w.shape == (NUM_PLIES, WACA_V33_F25)
    assert s25_b.shape == (NUM_PLIES,)
    assert h_emb.shape == (WACA_V33_EO_BLOCKS, KEY_RAW_ENTRIES, 8)
    assert ctr_emb.shape == (WACA_V33_EO_BLOCKS, WACA_V33_CTR_RAW_ENTRIES, 4)
    assert model.screlu_mu_key.shape == (NUM_PLIES, 3, 8)
    assert model.screlu_mu_ctr.shape == (NUM_PLIES, 4)
    assert model.screlu_beta.shape == (NUM_PLIES, WACA_V33_SCRELU_TERMS)
    assert w1.shape == (NUM_PLIES, h, WACA_V33_F3)
    assert b1.shape == (NUM_PLIES, h)
    assert w2.shape == (NUM_PLIES, h)
    assert b2.shape == (NUM_PLIES,)

    sc_chunks = []
    for ply in range(NUM_PLIES):
        sc_chunks.append(np.ascontiguousarray(model.screlu_mu_key[ply], dtype="<i2").tobytes())
        sc_chunks.append(np.ascontiguousarray(model.screlu_mu_ctr[ply], dtype="<i2").tobytes())
        sc_chunks.append(np.ascontiguousarray(model.screlu_beta[ply], dtype="<i2").tobytes())

    payload = (
        t1.tobytes()
        + s25_w.tobytes()
        + s25_b.tobytes()
        + h_emb.tobytes()
        + ctr_emb.tobytes()
        + b"".join(sc_chunks)
        + w1.tobytes()
        + b1.tobytes()
        + w2.tobytes()
        + b2.tobytes()
    )
    chk = _fast_fnv1a_64(payload)
    hdr = HEADER_STRUCT.pack(
        P2C_MAGIC_V33,
        P2C_FORMAT_VERSION_V33,
        int(EvalMode.WACA_G2_V33),
        float(model.score_scale),
        int(model.flags),
        KEY_RAW_ENTRIES,
        NONKEY_RAW_ENTRIES,
        TOTAL_RAW_ENTRIES,
        NUM_PLIES,
        WACA_V33_EO_BLOCKS,
        h,
        len(payload),
        chk,
    )
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        f.write(hdr)
        f.write(payload)


def load_p2c(path: str | Path) -> RawNTupleModel | P2C64Model | WacaG2Model | WacaG2V31Model | WacaG2V32Model | WacaG2V33Model:
    with open(path, "rb") as f:
        hdr_bytes = f.read(HEADER_STRUCT.size)
        if len(hdr_bytes) != HEADER_STRUCT.size:
            raise ValueError(f"Truncated .p2c header in {path}")
        (
            magic, version, mode_i, scale, flags,
            kre, nre, tre, n_plies, n_blks, res, payload_bytes, chk,
        ) = HEADER_STRUCT.unpack(hdr_bytes)
        is_v2 = (magic == P2C_MAGIC and version == P2C_FORMAT_VERSION)
        is_v3 = (magic == P2C_MAGIC_V3 and version == P2C_FORMAT_VERSION_V3)
        is_v31 = (magic == P2C_MAGIC_V31 and version == P2C_FORMAT_VERSION_V31)
        is_v32 = (magic == P2C_MAGIC_V32 and version == P2C_FORMAT_VERSION_V32)
        is_v33 = (magic == P2C_MAGIC_V33 and version == P2C_FORMAT_VERSION_V33)
        if not (is_v2 or is_v3 or is_v31 or is_v32 or is_v33):
            raise ValueError(f"Invalid .p2c magic/version in {path} (got {magic!r}, v={version})")
        if (kre, nre, tre, n_plies) != (KEY_RAW_ENTRIES, NONKEY_RAW_ENTRIES, TOTAL_RAW_ENTRIES, NUM_PLIES):
            raise ValueError("Table dimension mismatch in .p2c header")
        payload = f.read(payload_bytes)
        if len(payload) != payload_bytes:
            raise ValueError("Truncated .p2c payload")
        if _fast_fnv1a_64(payload) != chk:
            raise ValueError(f"FNV-1a checksum mismatch in {path}")

    mode = EvalMode(mode_i)
    mv = memoryview(payload)
    off = 0

    def take(nbytes: int) -> memoryview:
        nonlocal off
        chunk = mv[off : off + nbytes]
        off += nbytes
        return chunk

    if is_v33:
        if mode != EvalMode.WACA_G2_V33 or n_blks != WACA_V33_EO_BLOCKS:
            raise ValueError(f"Invalid v3.3 header fields (mode={mode}, num_blocks={n_blks})")
        h = int(res)
        if h < 1 or h > WACA_V33_MAX_HID:
            raise ValueError(f"Invalid v3.3 mlp_hidden={h}")
        t1 = np.frombuffer(take(NUM_PLIES * TOTAL_RAW_ENTRIES * 2), dtype="<i2").reshape(
            NUM_PLIES, TOTAL_RAW_ENTRIES
        ).copy()
        s25_w = np.frombuffer(take(NUM_PLIES * WACA_V33_F25 * 2), dtype="<i2").reshape(NUM_PLIES, WACA_V33_F25).copy()
        s25_b = np.frombuffer(take(NUM_PLIES * 4), dtype="<i4").reshape(NUM_PLIES).copy()
        h_emb = np.frombuffer(take(WACA_V33_EO_BLOCKS * KEY_RAW_ENTRIES * 8), dtype="i1").reshape(
            WACA_V33_EO_BLOCKS, KEY_RAW_ENTRIES, 8
        ).copy()
        ctr_emb = np.frombuffer(take(WACA_V33_EO_BLOCKS * WACA_V33_CTR_RAW_ENTRIES * 4), dtype="i1").reshape(
            WACA_V33_EO_BLOCKS, WACA_V33_CTR_RAW_ENTRIES, 4
        ).copy()
        sc_mu_k = np.zeros((NUM_PLIES, 3, 8), dtype=np.int16)
        sc_mu_c = np.zeros((NUM_PLIES, 4), dtype=np.int16)
        sc_beta = np.zeros((NUM_PLIES, WACA_V33_SCRELU_TERMS), dtype=np.int16)
        for ply in range(NUM_PLIES):
            sc_mu_k[ply] = np.frombuffer(take(48), dtype="<i2").reshape(3, 8)
            sc_mu_c[ply] = np.frombuffer(take(8), dtype="<i2").reshape(4)
            sc_beta[ply] = np.frombuffer(take(WACA_V33_SCRELU_TERMS * 2), dtype="<i2").reshape(WACA_V33_SCRELU_TERMS)
        w1 = np.frombuffer(take(NUM_PLIES * h * WACA_V33_F3), dtype="i1").reshape(NUM_PLIES, h, WACA_V33_F3).copy()
        b1 = np.frombuffer(take(NUM_PLIES * h * 4), dtype="<i4").reshape(NUM_PLIES, h).copy()
        w2 = np.frombuffer(take(NUM_PLIES * h * 2), dtype="<i2").reshape(NUM_PLIES, h).copy()
        b2 = np.frombuffer(take(NUM_PLIES * 4), dtype="<i4").reshape(NUM_PLIES).copy()
        if off != len(payload):
            raise ValueError("Unread trailing bytes in v3.3 .p2c payload")
        return WacaG2V33Model(
            tier1_w=t1,
            s25_w=s25_w,
            s25_b=s25_b,
            h_emb=h_emb,
            ctr_emb=ctr_emb,
            screlu_mu_key=sc_mu_k,
            screlu_mu_ctr=sc_mu_c,
            screlu_beta=sc_beta,
            mlp_w1=w1,
            mlp_b1=b1,
            mlp_w2=w2,
            mlp_b2=b2,
            mlp_hidden=h,
            score_scale=scale,
            flags=flags,
        )

    if is_v32:
        if mode != EvalMode.WACA_G2_V32 or n_blks != WACA_V32_EO_BLOCKS:
            raise ValueError(f"Invalid v3.2 header fields (mode={mode}, num_blocks={n_blks})")
        h = int(res)
        if h < 1 or h > WACA_V32_MAX_HID:
            raise ValueError(f"Invalid v3.2 mlp_hidden={h}")
        t1 = np.frombuffer(take(NUM_PLIES * TOTAL_RAW_ENTRIES * 2), dtype="<i2").reshape(
            NUM_PLIES, TOTAL_RAW_ENTRIES
        ).copy()
        s25_w = np.frombuffer(take(NUM_PLIES * WACA_V32_F25 * 2), dtype="<i2").reshape(NUM_PLIES, WACA_V32_F25).copy()
        s25_b = np.frombuffer(take(NUM_PLIES * 4), dtype="<i4").reshape(NUM_PLIES).copy()
        h_emb = np.frombuffer(take(WACA_V32_EO_BLOCKS * KEY_RAW_ENTRIES * 4), dtype="i1").reshape(
            WACA_V32_EO_BLOCKS, KEY_RAW_ENTRIES, 4
        ).copy()
        sc_mu = np.zeros((NUM_PLIES, 3, 4), dtype=np.int16)
        sc_beta = np.zeros((NUM_PLIES, 3, 4), dtype=np.int16)
        for ply in range(NUM_PLIES):
            sc_mu[ply] = np.frombuffer(take(24), dtype="<i2").reshape(3, 4)
            sc_beta[ply] = np.frombuffer(take(24), dtype="<i2").reshape(3, 4)
        w1 = np.frombuffer(take(NUM_PLIES * h * WACA_V32_F3), dtype="i1").reshape(NUM_PLIES, h, WACA_V32_F3).copy()
        b1 = np.frombuffer(take(NUM_PLIES * h * 4), dtype="<i4").reshape(NUM_PLIES, h).copy()
        w2 = np.frombuffer(take(NUM_PLIES * h * 2), dtype="<i2").reshape(NUM_PLIES, h).copy()
        b2 = np.frombuffer(take(NUM_PLIES * 4), dtype="<i4").reshape(NUM_PLIES).copy()
        if off != len(payload):
            raise ValueError("Unread trailing bytes in v3.2 .p2c payload")
        return WacaG2V32Model(
            tier1_w=t1,
            s25_w=s25_w,
            s25_b=s25_b,
            h_emb=h_emb,
            screlu_mu=sc_mu,
            screlu_beta=sc_beta,
            mlp_w1=w1,
            mlp_b1=b1,
            mlp_w2=w2,
            mlp_b2=b2,
            mlp_hidden=h,
            score_scale=scale,
            flags=flags,
        )

    if is_v31:
        if mode != EvalMode.WACA_G2_V31 or n_blks != WACA_V31_EO_BLOCKS:
            raise ValueError(f"Invalid v3.1 header fields (mode={mode}, num_blocks={n_blks})")
        h = int(res)
        if h < 1 or h > WACA_V31_MAX_HID:
            raise ValueError(f"Invalid v3.1 mlp_hidden={h}")
        t1 = np.frombuffer(take(NUM_PLIES * TOTAL_RAW_ENTRIES * 2), dtype="<i2").reshape(
            NUM_PLIES, TOTAL_RAW_ENTRIES
        ).copy()
        s25_w = np.frombuffer(take(NUM_PLIES * WACA_V31_F25 * 2), dtype="<i2").reshape(NUM_PLIES, WACA_V31_F25).copy()
        s25_b = np.frombuffer(take(NUM_PLIES * 4), dtype="<i4").reshape(NUM_PLIES).copy()
        h_emb = np.frombuffer(take(WACA_V31_EO_BLOCKS * KEY_RAW_ENTRIES * 4), dtype="i1").reshape(
            WACA_V31_EO_BLOCKS, KEY_RAW_ENTRIES, 4
        ).copy()
        sc_mu = np.zeros((NUM_PLIES, 3, 4), dtype=np.int16)
        sc_beta = np.zeros((NUM_PLIES, 3, 4), dtype=np.int16)
        for ply in range(NUM_PLIES):
            sc_mu[ply] = np.frombuffer(take(24), dtype="<i2").reshape(3, 4)
            sc_beta[ply] = np.frombuffer(take(24), dtype="<i2").reshape(3, 4)
        w1 = np.frombuffer(take(NUM_PLIES * h * WACA_V31_F3), dtype="i1").reshape(NUM_PLIES, h, WACA_V31_F3).copy()
        b1 = np.frombuffer(take(NUM_PLIES * h * 4), dtype="<i4").reshape(NUM_PLIES, h).copy()
        w2 = np.frombuffer(take(NUM_PLIES * h * 2), dtype="<i2").reshape(NUM_PLIES, h).copy()
        b2 = np.frombuffer(take(NUM_PLIES * 4), dtype="<i4").reshape(NUM_PLIES).copy()
        return WacaG2V31Model(
            tier1_w=t1,
            s25_w=s25_w,
            s25_b=s25_b,
            h_emb=h_emb,
            screlu_mu=sc_mu,
            screlu_beta=sc_beta,
            mlp_w1=w1,
            mlp_b1=b1,
            mlp_w2=w2,
            mlp_b2=b2,
            mlp_hidden=h,
            score_scale=scale,
            flags=flags,
        )

    if is_v3:
        if mode != EvalMode.WACA_G2 or n_blks != WACA_EO_BLOCKS:
            raise ValueError(f"Invalid v3.0 header fields (mode={mode}, num_blocks={n_blks})")
        h = int(res)
        if h < 1 or h > WACA_MAX_HID:
            raise ValueError(f"Invalid v3.0 mlp_hidden={h}")
        t1 = np.frombuffer(take(NUM_PLIES * TOTAL_RAW_ENTRIES * 2), dtype="<i2").reshape(
            NUM_PLIES, TOTAL_RAW_ENTRIES
        ).copy()
        s25_w = np.frombuffer(take(NUM_PLIES * WACA_F25 * 2), dtype="<i2").reshape(NUM_PLIES, WACA_F25).copy()
        s25_b = np.frombuffer(take(NUM_PLIES * 4), dtype="<i4").reshape(NUM_PLIES).copy()
        h_emb = np.frombuffer(take(WACA_EO_BLOCKS * KEY_RAW_ENTRIES * 4), dtype="i1").reshape(
            WACA_EO_BLOCKS, KEY_RAW_ENTRIES, 4
        ).copy()
        sc_mu = np.zeros((NUM_PLIES, 3, 4), dtype=np.int16)
        sc_beta = np.zeros((NUM_PLIES, 3, 4), dtype=np.int16)
        for ply in range(NUM_PLIES):
            sc_mu[ply] = np.frombuffer(take(24), dtype="<i2").reshape(3, 4)
            sc_beta[ply] = np.frombuffer(take(24), dtype="<i2").reshape(3, 4)
        w1 = np.frombuffer(take(NUM_PLIES * h * WACA_F3), dtype="i1").reshape(NUM_PLIES, h, WACA_F3).copy()
        b1 = np.frombuffer(take(NUM_PLIES * h * 4), dtype="<i4").reshape(NUM_PLIES, h).copy()
        w2 = np.frombuffer(take(NUM_PLIES * h * 2), dtype="<i2").reshape(NUM_PLIES, h).copy()
        b2 = np.frombuffer(take(NUM_PLIES * 4), dtype="<i4").reshape(NUM_PLIES).copy()
        return WacaG2Model(
            tier1_w=t1,
            s25_w=s25_w,
            s25_b=s25_b,
            h_emb=h_emb,
            screlu_mu=sc_mu,
            screlu_beta=sc_beta,
            mlp_w1=w1,
            mlp_b1=b1,
            mlp_w2=w2,
            mlp_b2=b2,
            mlp_hidden=h,
            score_scale=scale,
            flags=flags,
        )

    if n_blks != NUM_BLOCKS:
        raise ValueError(f"Table block count mismatch in .p2c v2.0 header (got {n_blks})")

    if mode == EvalMode.RAW_NTUPLE_64:
        w = np.frombuffer(take(NUM_PLIES * TOTAL_RAW_ENTRIES * 2), dtype="<i2").reshape(
            NUM_PLIES, TOTAL_RAW_ENTRIES
        ).copy()
        return RawNTupleModel(w_ply=w, score_scale=scale, flags=flags)

    kb = np.frombuffer(take(NUM_BLOCKS * KEY_RAW_ENTRIES * 16), dtype=BLOCK_ENTRY_DTYPE).reshape(
        NUM_BLOCKS, KEY_RAW_ENTRIES
    ).copy()
    nk = np.frombuffer(take(NUM_PLIES * NONKEY_RAW_ENTRIES * 2), dtype="<i2").reshape(
        NUM_PLIES, NONKEY_RAW_ENTRIES
    ).copy()

    sc_mu = np.zeros((NUM_PLIES, 3, 4), dtype=np.int16)
    sc_beta = np.zeros((NUM_PLIES, 3, 4), dtype=np.int16)
    for ply in range(NUM_PLIES):
        sc_mu[ply] = np.frombuffer(take(24), dtype="<i2").reshape(3, 4)
        sc_beta[ply] = np.frombuffer(take(24), dtype="<i2").reshape(3, 4)

    mcw = np.zeros((NUM_PLIES, 512), dtype=np.int16)
    mea = np.zeros((NUM_PLIES, 512), dtype=np.int16)
    mcd = np.zeros((NUM_PLIES, 512), dtype=np.int16)
    for ply in range(NUM_PLIES):
        mcw[ply] = np.frombuffer(take(1024), dtype="<i2")
        mea[ply] = np.frombuffer(take(1024), dtype="<i2")
        mcd[ply] = np.frombuffer(take(1024), dtype="<i2")

    return P2C64Model(
        mode=mode,
        key_blocks=kb,
        nonkey_w=nk,
        screlu_mu=sc_mu,
        screlu_beta=sc_beta,
        macro_cw=mcw,
        macro_ea=mea,
        macro_cd=mcd,
        score_scale=scale,
        flags=flags,
    )


# ============================================================================
# Exact Integer Evaluation Simulator (Bit-Exact with C++ evaluate_p2c64_ptm)
# ============================================================================
def _sym_rshift_i32(x: np.ndarray, shift: int = SCRELU_SHIFT) -> np.ndarray:
    x32 = x.astype(np.int32)
    add = (x32 >> np.int32(31)) & np.int32((1 << shift) - 1)
    return ((x32 + add) >> np.int32(shift)).astype(np.int32)


def eval_raw_ntuple64_int_batch(
    model: RawNTupleModel,
    raw_ptm: np.ndarray,  # [N, 64] int32
    ply: np.ndarray,      # [N] uint8
) -> np.ndarray:
    ply_i = ply.astype(np.int32)
    return model.w_ply[ply_i[:, None], raw_ptm].astype(np.int32).sum(axis=1, dtype=np.int32)


def eval_p2c64_int_batch(
    model: P2C64Model,
    key_ptm: np.ndarray,     # [N, 14] int32
    nonkey_ptm: np.ndarray,  # [N, 50] int32
    ply: np.ndarray,         # [N] uint8
) -> np.ndarray:
    ply_i = ply.astype(np.int32)
    blk = ply_i // PLIES_PER_BLOCK
    sub = ply_i % PLIES_PER_BLOCK
    n = len(ply_i)

    entries = model.key_blocks[blk[:, None], key_ptm]
    w_key = entries["w"][np.arange(n)[:, None], np.arange(14)[None, :], sub[:, None]].astype(np.int32)
    total = w_key.sum(axis=1, dtype=np.int32)
    total += model.nonkey_w[ply_i[:, None], nonkey_ptm].astype(np.int32).sum(axis=1, dtype=np.int32)

    if model.mode == EvalMode.LINEAR_64:
        return total

    aux_u8 = entries["aux"]
    aux_i8 = aux_u8.view(np.int8).astype(np.int32)

    if model.mode == EvalMode.CAND_A_PTM_SCRELU4:
        res_acc = np.zeros(n, dtype=np.int32)
        mu_p = model.screlu_mu[ply_i].astype(np.int32)     # [N, 3, 4]
        beta_p = model.screlu_beta[ply_i].astype(np.int32) # [N, 3, 4]

        s0 = np.zeros((n, 4), dtype=np.int32)
        for r in range(4):
            c1 = (r + 1) & 3
            u = np.clip(
                aux_i8[:, 4 + r] + aux_i8[:, 8 + r] + aux_i8[:, r] + aux_i8[:, c1] + mu_p[:, 0],
                0, SCRELU_Q,
            )
            s0 += u * u
        res_acc += (beta_p[:, 0] * s0).sum(axis=1, dtype=np.int32)

        s1 = np.zeros((n, 4), dtype=np.int32)
        for d in range(2):
            u = np.clip(
                aux_i8[:, 12 + d] + aux_i8[:, d] + aux_i8[:, d + 2] + mu_p[:, 1],
                0, SCRELU_Q,
            )
            s1 += u * u
        res_acc += (beta_p[:, 1] * s1).sum(axis=1, dtype=np.int32)

        ug = np.clip(aux_i8.sum(axis=1, dtype=np.int32) + mu_p[:, 2], 0, SCRELU_Q)
        res_acc += (beta_p[:, 2] * (ug * ug)).sum(axis=1, dtype=np.int32)
        return (total + _sym_rshift_i32(res_acc, SCRELU_SHIFT)).astype(np.int32)

    if model.mode in (EvalMode.CAND_B_PTM_3W3B, EvalMode.CAND_B_PTM_3W3B_NAIVE):
        zf = (aux_u8[:, :, 0] & 7).astype(np.int32)
        zr = (aux_u8[:, 4:8, 1] & 7).astype(np.int32)
        a_lut = np.zeros(n, dtype=np.int32)
        for i in range(4):
            a_lut += model.macro_cw[ply_i, (zf[:, i] << 6) | (zf[:, 4 + i] << 3) | zr[:, (i + 3) & 3]].astype(np.int32)
            a_lut += model.macro_ea[ply_i, (zf[:, 4 + i] << 6) | (zr[:, i] << 3) | zf[:, 8 + i]].astype(np.int32)
        a_lut += model.macro_cd[ply_i, (zf[:, 0] << 6) | (zf[:, 12] << 3) | zf[:, 2]].astype(np.int32)
        a_lut += model.macro_cd[ply_i, (zf[:, 1] << 6) | (zf[:, 13] << 3) | zf[:, 3]].astype(np.int32)
        return (total + a_lut).astype(np.int32)

    # EvalMode.CAND_C_PTM_HYBRID
    h = aux_i8[:, :, :2]  # [N, 14, 2]
    pk = aux_u8[:, :, 2].astype(np.int32)
    zf = pk & 7
    zr = (pk[:, 4:8] >> 3) & 7
    a_lut = np.zeros(n, dtype=np.int32)
    for i in range(4):
        a_lut += model.macro_cw[ply_i, (zf[:, i] << 6) | (zf[:, 4 + i] << 3) | zr[:, (i + 3) & 3]].astype(np.int32)
        a_lut += model.macro_ea[ply_i, (zf[:, 4 + i] << 6) | (zr[:, i] << 3) | zf[:, 8 + i]].astype(np.int32)
    a_lut += model.macro_cd[ply_i, (zf[:, 0] << 6) | (zf[:, 12] << 3) | zf[:, 2]].astype(np.int32)
    a_lut += model.macro_cd[ply_i, (zf[:, 1] << 6) | (zf[:, 13] << 3) | zf[:, 3]].astype(np.int32)

    mu_p = model.screlu_mu[ply_i, :, :2].astype(np.int32)
    beta_p = model.screlu_beta[ply_i, :, :2].astype(np.int32)
    res_acc = np.zeros(n, dtype=np.int32)
    s0 = np.zeros((n, 2), dtype=np.int32)
    for r in range(4):
        c1 = (r + 1) & 3
        u = np.clip(h[:, 4 + r] + h[:, 8 + r] + h[:, r] + h[:, c1] + mu_p[:, 0], 0, SCRELU_Q)
        s0 += u * u
    res_acc += (beta_p[:, 0] * s0).sum(axis=1, dtype=np.int32)

    s1 = np.zeros((n, 2), dtype=np.int32)
    for d in range(2):
        u = np.clip(h[:, 12 + d] + h[:, d] + h[:, d + 2] + mu_p[:, 1], 0, SCRELU_Q)
        s1 += u * u
    res_acc += (beta_p[:, 1] * s1).sum(axis=1, dtype=np.int32)

    ug = np.clip(h.sum(axis=1, dtype=np.int32) + mu_p[:, 2], 0, SCRELU_Q)
    res_acc += (beta_p[:, 2] * (ug * ug)).sum(axis=1, dtype=np.int32)
    return (total + a_lut + _sym_rshift_i32(res_acc, SCRELU_SHIFT)).astype(np.int32)


def eval_waca_g2_int_batch(
    model: WacaG2Model,
    v1: np.ndarray,       # [N] int32
    f25: np.ndarray,      # [N, 16] int32
    f3_geom: np.ndarray,  # [N, 4] int32
    key_ptm: np.ndarray,  # [N, 14] int32
    ply: np.ndarray,      # [N] uint8
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Bit-exact Python integer simulator matching C++ eval_waca_g2_stages across all 3 stages."""
    ply_i = ply.astype(np.int32)
    n = len(ply_i)
    v1_i = v1.astype(np.int32)
    f25_i = f25.astype(np.int32)
    f3g_i = f3_geom.astype(np.int32)

    # Stage 2.5
    acc25 = model.s25_b[ply_i].astype(np.int32) + (
        model.s25_w[ply_i].astype(np.int32) * f25_i
    ).sum(axis=1, dtype=np.int32)
    r25 = np.clip(_sym_rshift_i32(acc25, WACA_S25_SHIFT), -WACA_MAX_RES, WACA_MAX_RES).astype(np.int32)
    v25 = (v1_i + r25).astype(np.int32)

    # Stage 3 Part 1: 20-block EO Key-14 4ch SCReLU
    eo_blk = PLY_TO_EO_BLK[ply_i]
    h = model.h_emb[eo_blk[:, None], key_ptm].astype(np.int32)  # [N, 14, 4]
    mu_p = model.screlu_mu[ply_i].astype(np.int32)              # [N, 3, 4]
    beta_p = model.screlu_beta[ply_i].astype(np.int32)          # [N, 3, 4]

    s0 = np.zeros((n, 4), dtype=np.int32)
    for r in range(4):
        c1 = (r + 1) & 3
        u = np.clip(h[:, 4 + r] + h[:, 8 + r] + h[:, r] + h[:, c1] + mu_p[:, 0], 0, SCRELU_Q)
        s0 += u * u

    s1 = np.zeros((n, 4), dtype=np.int32)
    for d in range(2):
        u = np.clip(h[:, 12 + d] + h[:, d] + h[:, d + 2] + mu_p[:, 1], 0, SCRELU_Q)
        s1 += u * u

    ug = np.clip(h.sum(axis=1, dtype=np.int32) + mu_p[:, 2], 0, SCRELU_Q)
    s2 = (ug * ug).astype(np.int32)

    sc_acc = (beta_p[:, 0] * s0 + beta_p[:, 1] * s1 + beta_p[:, 2] * s2).sum(axis=1, dtype=np.int32)
    r_key = _sym_rshift_i32(sc_acc, SCRELU_SHIFT)
    f_key = np.clip((s0 + s1 + s2) >> np.int32(8), -WACA_FEAT_CLAMP, WACA_FEAT_CLAMP).astype(np.int32)

    # Stage 3 Part 2: 60-ply Pure-Integer SCReLU MLP (24 -> H -> 1)
    H = int(model.mlp_hidden)
    f_all = np.concatenate([f25_i, f3g_i, f_key], axis=1)  # [N, 24]
    w1_p = model.mlp_w1[ply_i, :H, :].astype(np.int32)     # [N, H, 24]
    b1_p = model.mlp_b1[ply_i, :H].astype(np.int32)        # [N, H]
    z = b1_p + (w1_p * f_all[:, None, :]).sum(axis=2, dtype=np.int32)
    u_mlp = np.where(
        z <= 0, np.int32(0), np.minimum(z >> np.int32(WACA_MLP_S1), np.int32(SCRELU_Q))
    ).astype(np.int32)
    w2_p = model.mlp_w2[ply_i, :H].astype(np.int32)
    acc2 = model.mlp_b2[ply_i].astype(np.int32) + (w2_p * (u_mlp * u_mlp)).sum(axis=1, dtype=np.int32)
    r_mlp = _sym_rshift_i32(acc2, WACA_MLP_S2)

    rt = np.clip(r25 + r_mlp + r_key, -WACA_MAX_RES, WACA_MAX_RES).astype(np.int32)
    v3 = (v1_i + rt).astype(np.int32)
    return v1_i, v25, v3


def eval_waca_g2_v31_int_batch(
    model: WacaG2V31Model,
    v1: np.ndarray,        # [N] int32
    f25: np.ndarray,       # [N, 8] int32
    f3_extra: np.ndarray,  # [N, 20] int32
    key_ptm: np.ndarray,   # [N, 14] int32
    ply: np.ndarray,       # [N] uint8
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Bit-exact Python integer simulator matching C++ eval_waca_g2_v31_stages across all 3 stages."""
    ply_i = ply.astype(np.int32)
    n = len(ply_i)
    v1_i = v1.astype(np.int32)
    f25_i = f25.astype(np.int32)
    f3e_i = f3_extra.astype(np.int32)

    # Stage 2.5 (slim 8-dim linear)
    acc25 = model.s25_b[ply_i].astype(np.int32) + (
        model.s25_w[ply_i].astype(np.int32) * f25_i
    ).sum(axis=1, dtype=np.int32)
    r25 = np.clip(_sym_rshift_i32(acc25, WACA_S25_SHIFT), -WACA_MAX_RES, WACA_MAX_RES).astype(np.int32)
    v25 = (v1_i + r25).astype(np.int32)

    # Stage 3 Part 1: 30-block EO Key-14 4ch SCReLU
    eo_blk = PLY_TO_EO30_BLK[ply_i]
    h = model.h_emb[eo_blk[:, None], key_ptm].astype(np.int32)  # [N, 14, 4]
    mu_p = model.screlu_mu[ply_i].astype(np.int32)              # [N, 3, 4]
    beta_p = model.screlu_beta[ply_i].astype(np.int32)          # [N, 3, 4]

    s0 = np.zeros((n, 4), dtype=np.int32)
    for r in range(4):
        c1 = (r + 1) & 3
        u = np.clip(h[:, 4 + r] + h[:, 8 + r] + h[:, r] + h[:, c1] + mu_p[:, 0], 0, SCRELU_Q)
        s0 += u * u

    s1 = np.zeros((n, 4), dtype=np.int32)
    for d in range(2):
        u = np.clip(h[:, 12 + d] + h[:, d] + h[:, d + 2] + mu_p[:, 1], 0, SCRELU_Q)
        s1 += u * u

    ug = np.clip(h.sum(axis=1, dtype=np.int32) + mu_p[:, 2], 0, SCRELU_Q)
    s2 = (ug * ug).astype(np.int32)

    sc_acc = (beta_p[:, 0] * s0 + beta_p[:, 1] * s1 + beta_p[:, 2] * s2).sum(axis=1, dtype=np.int32)
    r_key = _sym_rshift_i32(sc_acc, SCRELU_SHIFT)
    f_key = np.clip((s0 + s1 + s2) >> np.int32(8), -WACA_FEAT_CLAMP, WACA_FEAT_CLAMP).astype(np.int32)

    # Stage 3 Part 2: 60-ply Pure-Integer SCReLU MLP (32 -> H -> 1)
    H = int(model.mlp_hidden)
    f_all = np.concatenate([f25_i, f3e_i, f_key], axis=1)  # [N, 32]
    w1_p = model.mlp_w1[ply_i, :H, :].astype(np.int32)     # [N, H, 32]
    b1_p = model.mlp_b1[ply_i, :H].astype(np.int32)        # [N, H]
    z = b1_p + (w1_p * f_all[:, None, :]).sum(axis=2, dtype=np.int32)
    u_mlp = np.where(
        z <= 0, np.int32(0), np.minimum(z >> np.int32(WACA_MLP_S1), np.int32(SCRELU_Q))
    ).astype(np.int32)
    w2_p = model.mlp_w2[ply_i, :H].astype(np.int32)
    acc2 = model.mlp_b2[ply_i].astype(np.int32) + (w2_p * (u_mlp * u_mlp)).sum(axis=1, dtype=np.int32)
    r_mlp = _sym_rshift_i32(acc2, WACA_MLP_S2)

    rt = np.clip(r25 + r_mlp + r_key, -WACA_MAX_RES, WACA_MAX_RES).astype(np.int32)
    v3 = (v1_i + rt).astype(np.int32)
    return v1_i, v25, v3


def eval_waca_g2_v32_int_batch(
    model: WacaG2V32Model,
    v1: np.ndarray,        # [N] int32
    f25: np.ndarray,       # [N, 8] int32
    f3_extra: np.ndarray,  # [N, 16] int32
    key_ptm: np.ndarray,   # [N, 14] int32
    ply: np.ndarray,       # [N] uint8
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Bit-exact Python integer simulator matching C++ eval_waca_g2_v32_stages across all 3 stages."""
    ply_i = ply.astype(np.int32)
    n = len(ply_i)
    v1_i = v1.astype(np.int32)
    f25_i = f25.astype(np.int32)
    f3e_i = f3_extra.astype(np.int32)

    # Stage 2.5 (slim 8-dim linear)
    acc25 = model.s25_b[ply_i].astype(np.int32) + (
        model.s25_w[ply_i].astype(np.int32) * f25_i
    ).sum(axis=1, dtype=np.int32)
    r25 = np.clip(_sym_rshift_i32(acc25, WACA_S25_SHIFT), -WACA_MAX_RES, WACA_MAX_RES).astype(np.int32)
    v25 = (v1_i + r25).astype(np.int32)

    # Stage 3 Part 1: 30-block EO Key-14 4ch SCReLU
    eo_blk = PLY_TO_EO30_BLK[ply_i]
    h = model.h_emb[eo_blk[:, None], key_ptm].astype(np.int32)  # [N, 14, 4]
    mu_p = model.screlu_mu[ply_i].astype(np.int32)              # [N, 3, 4]
    beta_p = model.screlu_beta[ply_i].astype(np.int32)          # [N, 3, 4]

    s0 = np.zeros((n, 4), dtype=np.int32)
    for r in range(4):
        c1 = (r + 1) & 3
        u = np.clip(h[:, 4 + r] + h[:, 8 + r] + h[:, r] + h[:, c1] + mu_p[:, 0], 0, SCRELU_Q)
        s0 += u * u

    s1 = np.zeros((n, 4), dtype=np.int32)
    for d in range(2):
        u = np.clip(h[:, 12 + d] + h[:, d] + h[:, d + 2] + mu_p[:, 1], 0, SCRELU_Q)
        s1 += u * u

    ug = np.clip(h.sum(axis=1, dtype=np.int32) + mu_p[:, 2], 0, SCRELU_Q)
    s2 = (ug * ug).astype(np.int32)

    sc_acc = (beta_p[:, 0] * s0 + beta_p[:, 1] * s1 + beta_p[:, 2] * s2).sum(axis=1, dtype=np.int32)
    r_key = _sym_rshift_i32(sc_acc, SCRELU_SHIFT)
    f_key = np.clip((s0 + s1 + s2) >> np.int32(8), -WACA_FEAT_CLAMP, WACA_FEAT_CLAMP).astype(np.int32)

    # Stage 3 Part 2: 60-ply Pure-Integer SCReLU MLP (28 -> H -> 1)
    H = int(model.mlp_hidden)
    f_all = np.concatenate([f25_i, f3e_i, f_key], axis=1)  # [N, 28]
    w1_p = model.mlp_w1[ply_i, :H, :].astype(np.int32)     # [N, H, 28]
    b1_p = model.mlp_b1[ply_i, :H].astype(np.int32)        # [N, H]
    z = b1_p + (w1_p * f_all[:, None, :]).sum(axis=2, dtype=np.int32)
    u_mlp = np.where(
        z <= 0, np.int32(0), np.minimum(z >> np.int32(WACA_MLP_S1), np.int32(SCRELU_Q))
    ).astype(np.int32)
    w2_p = model.mlp_w2[ply_i, :H].astype(np.int32)
    acc2 = model.mlp_b2[ply_i].astype(np.int32) + (w2_p * (u_mlp * u_mlp)).sum(axis=1, dtype=np.int32)
    r_mlp = _sym_rshift_i32(acc2, WACA_MLP_S2)

    rt = np.clip(r25 + r_mlp + r_key, -WACA_MAX_RES, WACA_MAX_RES).astype(np.int32)
    v3 = (v1_i + rt).astype(np.int32)
    return v1_i, v25, v3


def eval_waca_g2_v33_int_batch(
    model: WacaG2V33Model,
    v1: np.ndarray,        # [N] int32
    f25: np.ndarray,       # [N, 12] int32
    f3_extra: np.ndarray,  # [N, 16] int32
    key_ptm: np.ndarray,   # [N, 14] int32
    ctr_ptm: np.ndarray,   # [N, 10] int32
    ply: np.ndarray,       # [N] uint8
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Bit-exact Python integer simulator matching C++ eval_waca_g2_v33_stages across all 3 stages."""
    ply_i = ply.astype(np.int32)
    n = len(ply_i)
    v1_i = v1.astype(np.int32)
    f25_i = f25.astype(np.int32)
    f3e_i = f3_extra.astype(np.int32)

    # Stage 2.5 (12-dim polynomial linear)
    acc25 = model.s25_b[ply_i].astype(np.int32) + (
        model.s25_w[ply_i].astype(np.int32) * f25_i
    ).sum(axis=1, dtype=np.int32)
    r25 = np.clip(_sym_rshift_i32(acc25, WACA_S25_SHIFT), -WACA_MAX_RES, WACA_MAX_RES).astype(np.int32)
    v25 = (v1_i + r25).astype(np.int32)

    # Stage 3 Part 1: 30-block EO Key-14 8ch + Center-10 4ch + 44-term Pairwise/Cross SCReLU
    eo_blk = PLY_TO_EO30_BLK[ply_i]
    h = model.h_emb[eo_blk[:, None], key_ptm].astype(np.int32)     # [N, 14, 8]
    hc = model.ctr_emb[eo_blk[:, None], ctr_ptm].astype(np.int32)  # [N, 10, 4]
    gc = hc.sum(axis=1, dtype=np.int32)                            # [N, 4]

    mu_k = model.screlu_mu_key[ply_i].astype(np.int32)             # [N, 3, 8]
    mu_c = model.screlu_mu_ctr[ply_i].astype(np.int32)             # [N, 4]
    beta = model.screlu_beta[ply_i].astype(np.int32)               # [N, 44]

    s0_sq = np.zeros((n, 8), dtype=np.int32)
    p0 = np.zeros((n, 4), dtype=np.int32)
    for r in range(4):
        c1 = (r + 1) & 3
        u = np.clip(h[:, 4 + r] + h[:, 8 + r] + h[:, r] + h[:, c1] + mu_k[:, 0], 0, SCRELU_Q)
        s0_sq += u * u
        p0 += u[:, :4] * u[:, 4:]

    s1_sq = np.zeros((n, 8), dtype=np.int32)
    p1 = np.zeros((n, 4), dtype=np.int32)
    for d in range(2):
        u = np.clip(h[:, 12 + d] + h[:, d] + h[:, d + 2] + mu_k[:, 1], 0, SCRELU_Q)
        s1_sq += u * u
        p1 += u[:, :4] * u[:, 4:]

    ug = np.clip(h.sum(axis=1, dtype=np.int32) + mu_k[:, 2], 0, SCRELU_Q)
    s2_sq = (ug * ug).astype(np.int32)
    p2 = (ug[:, :4] * ug[:, 4:]).astype(np.int32)

    uc = np.clip(gc + mu_c, 0, SCRELU_Q)
    s3_sq = (uc * uc).astype(np.int32)
    pc = (ug[:, :4] * uc).astype(np.int32)

    sc_terms = np.concatenate([s0_sq, p0, s1_sq, p1, s2_sq, p2, s3_sq, pc], axis=1)  # [N, 44]
    sc_acc = (beta * sc_terms).sum(axis=1, dtype=np.int32)
    r_key = _sym_rshift_i32(sc_acc, SCRELU_SHIFT)

    sum_sq = s0_sq + s1_sq + s2_sq
    f_24_27 = np.clip((sum_sq[:, :4] + sum_sq[:, 4:]) >> np.int32(9), -WACA_FEAT_CLAMP, WACA_FEAT_CLAMP)
    p_sum = p0 + p1 + p2
    f_28 = np.clip((p_sum[:, 0] + p_sum[:, 1]) >> np.int32(9), -WACA_FEAT_CLAMP, WACA_FEAT_CLAMP)[:, None]
    f_29 = np.clip((p_sum[:, 2] + p_sum[:, 3]) >> np.int32(9), -WACA_FEAT_CLAMP, WACA_FEAT_CLAMP)[:, None]
    f_30 = np.clip(s3_sq.sum(axis=1, dtype=np.int32) >> np.int32(8), -WACA_FEAT_CLAMP, WACA_FEAT_CLAMP)[:, None]
    f_31 = np.clip(pc.sum(axis=1, dtype=np.int32) >> np.int32(8), -WACA_FEAT_CLAMP, WACA_FEAT_CLAMP)[:, None]
    f_sparse = np.concatenate([f_24_27, f_28, f_29, f_30, f_31], axis=1).astype(np.int32)

    # Stage 3 Part 2: 60-ply Pure-Integer SCReLU MLP (32 -> H -> 1)
    H = int(model.mlp_hidden)
    f_all = np.concatenate([f25_i[:, :8], f3e_i, f_sparse], axis=1)  # [N, 32]
    w1_p = model.mlp_w1[ply_i, :H, :].astype(np.int32)               # [N, H, 32]
    b1_p = model.mlp_b1[ply_i, :H].astype(np.int32)                  # [N, H]
    z = b1_p + (w1_p * f_all[:, None, :]).sum(axis=2, dtype=np.int32)
    u_mlp = np.where(
        z <= 0, np.int32(0), np.minimum(z >> np.int32(WACA_MLP_S1), np.int32(SCRELU_Q))
    ).astype(np.int32)
    w2_p = model.mlp_w2[ply_i, :H].astype(np.int32)
    acc2 = model.mlp_b2[ply_i].astype(np.int32) + (w2_p * (u_mlp * u_mlp)).sum(axis=1, dtype=np.int32)
    r_mlp = _sym_rshift_i32(acc2, WACA_MLP_S2)

    rt = np.clip(r25 + r_mlp + r_key, -WACA_MAX_RES, WACA_MAX_RES).astype(np.int32)
    v3 = (v1_i + rt).astype(np.int32)
    return v1_i, v25, v3
