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
P2C_DEFAULT_SCALE = 128.0
HEADER_STRUCT = struct.Struct("<8sIIfIIIIIIIQQ")
assert HEADER_STRUCT.size == 64

SCRELU_Q = 127
SCRELU_BETA_MAX = 2047
SCRELU_SHIFT = 7

BLOCK_ENTRY_DTYPE = np.dtype([("w", "<i2", (6,)), ("aux", "u1", (4,))], align=False)
assert BLOCK_ENTRY_DTYPE.itemsize == 16


class EvalMode(IntEnum):
    RAW_NTUPLE_64 = 0
    LINEAR_64 = 1
    CAND_A_PTM_SCRELU4 = 2
    CAND_B_PTM_3W3B_NAIVE = 3
    CAND_B_PTM_3W3B = 4
    CAND_C_PTM_HYBRID = 5


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


def load_p2c(path: str | Path) -> RawNTupleModel | P2C64Model:
    with open(path, "rb") as f:
        hdr_bytes = f.read(HEADER_STRUCT.size)
        if len(hdr_bytes) != HEADER_STRUCT.size:
            raise ValueError(f"Truncated .p2c header in {path}")
        (
            magic, version, mode_i, scale, flags,
            kre, nre, tre, n_plies, n_blks, _res, payload_bytes, chk,
        ) = HEADER_STRUCT.unpack(hdr_bytes)
        if magic != P2C_MAGIC or version != P2C_FORMAT_VERSION:
            raise ValueError(f"Invalid .p2c v2.0 magic/version in {path} (got {magic!r}, v={version})")
        if (kre, nre, tre, n_plies, n_blks) != (
            KEY_RAW_ENTRIES, NONKEY_RAW_ENTRIES, TOTAL_RAW_ENTRIES, NUM_PLIES, NUM_BLOCKS
        ):
            raise ValueError("Table dimension mismatch in .p2c v2.0 header")
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
