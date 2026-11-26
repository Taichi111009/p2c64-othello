"""ctypes wrapper around build/libp2c64_batch.so for fast OpenMP pattern extraction,
exact C++ orbit/symmetry metadata export, and C++ .p2c verification."""

from __future__ import annotations

import ctypes
import os
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np

NUM_PATTERN_TYPES = 16
NUM_PATTERNS = 64
NUM_KEY_PATTERNS = 14
NUM_NONKEY_PATTERNS = 50
NUM_PLIES = 60
PLIES_PER_BLOCK = 6
NUM_BLOCKS = 10

KEY_CANON_ENTRIES = 45929
NONKEY_RAW_ENTRIES = 318168
TOTAL_RAW_ENTRIES = 410022
CORNER_CANON_SIZE = 9842
CORNER_RAW_SIZE = 19683
EDGE_CANON_SIZE = 29525

PACKED_SAMPLE_DTYPE = np.dtype(
    [
        ("black", "<u8"),
        ("white", "<u8"),
        ("score", "<i1"),
        ("stm", "<i1"),
        ("ply", "<u1"),
        ("flags", "<u1"),
    ],
    align=False,
)
assert PACKED_SAMPLE_DTYPE.itemsize == 20


def _find_library(lib_path: Optional[str] = None) -> Path:
    if lib_path:
        p = Path(lib_path)
        if p.exists():
            return p
        raise FileNotFoundError(f"libp2c64_batch.so not found at {p}")
    candidates = [
        Path(__file__).resolve().parent.parent / "build" / "libp2c64_batch.so",
        Path.cwd() / "build" / "libp2c64_batch.so",
    ]
    for c in candidates:
        if c.exists():
            return c
    raise FileNotFoundError(
        "Could not locate build/libp2c64_batch.so. Run `make tools` first."
    )


class P2C64Bridge:
    def __init__(self, lib_path: Optional[str] = None) -> None:
        self.lib_path = _find_library(lib_path)
        self.lib = ctypes.CDLL(str(self.lib_path))

        c_u64_p = ctypes.POINTER(ctypes.c_uint64)
        c_u16_p = ctypes.POINTER(ctypes.c_uint16)
        c_u8_p = ctypes.POINTER(ctypes.c_uint8)
        c_i32_p = ctypes.POINTER(ctypes.c_int32)
        c_i8_p = ctypes.POINTER(ctypes.c_int8)

        self.lib.p2c64_extract_batch.argtypes = [
            c_u64_p, c_u64_p, c_i8_p,
            ctypes.c_int32, ctypes.c_uint64, ctypes.c_int32,
            c_u16_p, c_i32_p, c_i8_p, c_i32_p, c_i32_p, c_i8_p,
        ]
        self.lib.p2c64_extract_batch.restype = None

        self.lib.p2c64_get_constants.argtypes = [c_i32_p] * 10
        self.lib.p2c64_get_constants.restype = None

        self.lib.p2c64_get_orbit_tables.argtypes = [
            c_i32_p, c_i8_p, c_i8_p,
            c_i32_p, c_i8_p,
            c_i32_p, c_i8_p,
            c_i32_p, c_i8_p,
            c_i32_p, c_i32_p,
        ]
        self.lib.p2c64_get_orbit_tables.restype = None

        self.lib.p2c64_eval_file_batch.argtypes = [
            ctypes.c_char_p,
            c_u64_p, c_u64_p, c_i8_p, c_u8_p,
            ctypes.c_int32, c_i32_p, c_i32_p,
        ]
        self.lib.p2c64_eval_file_batch.restype = ctypes.c_int32

        self._orbit_cache: Optional[Dict[str, np.ndarray]] = None

    def get_constants(self) -> Dict[str, np.ndarray | int]:
        kce = ctypes.c_int32()
        nre = ctypes.c_int32()
        tre = ctypes.c_int32()
        ccs = ctypes.c_int32()
        crs = ctypes.c_int32()
        t_co = np.zeros(NUM_PATTERN_TYPES, dtype=np.int32)
        t_cs = np.zeros(NUM_PATTERN_TYPES, dtype=np.int32)
        t_ro = np.zeros(NUM_PATTERN_TYPES, dtype=np.int32)
        t_rs = np.zeros(NUM_PATTERN_TYPES, dtype=np.int32)
        t_no = np.zeros(NUM_PATTERN_TYPES, dtype=np.int32)
        self.lib.p2c64_get_constants(
            ctypes.byref(kce), ctypes.byref(nre), ctypes.byref(tre),
            ctypes.byref(ccs), ctypes.byref(crs),
            t_co.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
            t_cs.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
            t_ro.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
            t_rs.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
            t_no.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
        )
        return {
            "key_canon_entries": int(kce.value),
            "nonkey_raw_entries": int(nre.value),
            "total_raw_entries": int(tre.value),
            "corner_canon_size": int(ccs.value),
            "corner_raw_size": int(crs.value),
            "type_canon_off": t_co,
            "type_canon_size": t_cs,
            "type_raw_off": t_ro,
            "type_raw_size": t_rs,
            "type_nonkey_off": t_no,
        }

    def get_orbit_tables(self) -> Dict[str, np.ndarray]:
        if self._orbit_cache is not None:
            return self._orbit_cache
        key_orb_rep = np.zeros(KEY_CANON_ENTRIES, dtype=np.int32)
        key_orb_sgn = np.zeros(KEY_CANON_ENTRIES, dtype=np.int8)
        key_anti_mirror = np.zeros(KEY_CANON_ENTRIES, dtype=np.int8)
        edge_mirror_cx = np.zeros(EDGE_CANON_SIZE, dtype=np.int32)
        edge_mirror_flip = np.zeros(EDGE_CANON_SIZE, dtype=np.int8)
        nk_orb_rep = np.zeros(NONKEY_RAW_ENTRIES, dtype=np.int32)
        nk_orb_sgn = np.zeros(NONKEY_RAW_ENTRIES, dtype=np.int8)
        raw_orb_rep = np.zeros(TOTAL_RAW_ENTRIES, dtype=np.int32)
        raw_orb_sgn = np.zeros(TOTAL_RAW_ENTRIES, dtype=np.int8)
        ct_canon_rep = np.zeros(CORNER_CANON_SIZE, dtype=np.int32)
        ct_raw_rep = np.zeros(CORNER_RAW_SIZE, dtype=np.int32)

        c_i32_p = ctypes.POINTER(ctypes.c_int32)
        c_i8_p = ctypes.POINTER(ctypes.c_int8)
        self.lib.p2c64_get_orbit_tables(
            key_orb_rep.ctypes.data_as(c_i32_p),
            key_orb_sgn.ctypes.data_as(c_i8_p),
            key_anti_mirror.ctypes.data_as(c_i8_p),
            edge_mirror_cx.ctypes.data_as(c_i32_p),
            edge_mirror_flip.ctypes.data_as(c_i8_p),
            nk_orb_rep.ctypes.data_as(c_i32_p),
            nk_orb_sgn.ctypes.data_as(c_i8_p),
            raw_orb_rep.ctypes.data_as(c_i32_p),
            raw_orb_sgn.ctypes.data_as(c_i8_p),
            ct_canon_rep.ctypes.data_as(c_i32_p),
            ct_raw_rep.ctypes.data_as(c_i32_p),
        )
        self._orbit_cache = {
            "key_orb_rep": key_orb_rep,
            "key_orb_sgn": key_orb_sgn,
            "key_anti_mirror": key_anti_mirror,
            "edge_mirror_cx": edge_mirror_cx,
            "edge_mirror_flip": edge_mirror_flip,
            "nk_orb_rep": nk_orb_rep,
            "nk_orb_sgn": nk_orb_sgn,
            "raw_orb_rep": raw_orb_rep,
            "raw_orb_sgn": raw_orb_sgn,
            "ct_canon_rep": ct_canon_rep,
            "ct_raw_rep": ct_raw_rep,
        }
        return self._orbit_cache

    def extract_batch(
        self,
        black: np.ndarray,
        white: np.ndarray,
        stm: np.ndarray,
        aug_seed: int = 0,
        enable_aug: bool = False,
        need_raw_idx: bool = False,
        need_p2c: bool = True,
        need_raw_flat: bool = False,
    ) -> Dict[str, np.ndarray]:
        black = np.ascontiguousarray(black, dtype=np.uint64)
        white = np.ascontiguousarray(white, dtype=np.uint64)
        stm = np.ascontiguousarray(stm, dtype=np.int8)
        n = int(black.shape[0])

        c_u64_p = ctypes.POINTER(ctypes.c_uint64)
        c_u16_p = ctypes.POINTER(ctypes.c_uint16)
        c_i32_p = ctypes.POINTER(ctypes.c_int32)
        c_i8_p = ctypes.POINTER(ctypes.c_int8)

        out: Dict[str, np.ndarray] = {}
        out_stm = np.empty(n, dtype=np.int8)
        out["stm"] = out_stm

        if need_raw_idx:
            raw_idx = np.empty((n, NUM_PATTERNS), dtype=np.uint16)
            p_raw_idx = raw_idx.ctypes.data_as(c_u16_p)
            out["raw_idx"] = raw_idx
        else:
            p_raw_idx = ctypes.cast(0, c_u16_p)

        if need_p2c:
            key_cx = np.empty((n, NUM_KEY_PATTERNS), dtype=np.int32)
            key_flip = np.empty((n, NUM_KEY_PATTERNS), dtype=np.int8)
            nk_flat = np.empty((n, NUM_NONKEY_PATTERNS), dtype=np.int32)
            p_kcx = key_cx.ctypes.data_as(c_i32_p)
            p_kfl = key_flip.ctypes.data_as(c_i8_p)
            p_nkf = nk_flat.ctypes.data_as(c_i32_p)
            out["key_cx"] = key_cx
            out["key_flip"] = key_flip
            out["nonkey_flat"] = nk_flat
        else:
            p_kcx = ctypes.cast(0, c_i32_p)
            p_kfl = ctypes.cast(0, c_i8_p)
            p_nkf = ctypes.cast(0, c_i32_p)

        if need_raw_flat:
            raw_flat = np.empty((n, NUM_PATTERNS), dtype=np.int32)
            p_rf = raw_flat.ctypes.data_as(c_i32_p)
            out["raw_flat"] = raw_flat
        else:
            p_rf = ctypes.cast(0, c_i32_p)

        self.lib.p2c64_extract_batch(
            black.ctypes.data_as(c_u64_p),
            white.ctypes.data_as(c_u64_p),
            stm.ctypes.data_as(c_i8_p),
            ctypes.c_int32(n),
            ctypes.c_uint64(aug_seed & 0xFFFFFFFFFFFFFFFF),
            ctypes.c_int32(1 if enable_aug else 0),
            p_raw_idx,
            p_kcx,
            p_kfl,
            p_nkf,
            p_rf,
            out_stm.ctypes.data_as(c_i8_p),
        )
        return out

    def eval_file_batch(
        self,
        p2c_path: str | os.PathLike,
        black: np.ndarray,
        white: np.ndarray,
        stm: np.ndarray,
        ply: np.ndarray,
        check_16sym: bool = False,
    ) -> Tuple[int, np.ndarray, int]:
        black = np.ascontiguousarray(black, dtype=np.uint64)
        white = np.ascontiguousarray(white, dtype=np.uint64)
        stm = np.ascontiguousarray(stm, dtype=np.int8)
        ply = np.ascontiguousarray(ply, dtype=np.uint8)
        n = int(black.shape[0])
        scores = np.empty(n, dtype=np.int32)
        max_sym_err = ctypes.c_int32(0)

        mode = self.lib.p2c64_eval_file_batch(
            os.fsencode(str(p2c_path)),
            black.ctypes.data_as(ctypes.POINTER(ctypes.c_uint64)),
            white.ctypes.data_as(ctypes.POINTER(ctypes.c_uint64)),
            stm.ctypes.data_as(ctypes.POINTER(ctypes.c_int8)),
            ply.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
            ctypes.c_int32(n),
            scores.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
            ctypes.byref(max_sym_err) if check_16sym else ctypes.cast(0, ctypes.POINTER(ctypes.c_int32)),
        )
        if mode < 0:
            raise RuntimeError(f"C++ p2c64_eval_file_batch failed on {p2c_path}")
        return int(mode), scores, int(max_sym_err.value)
