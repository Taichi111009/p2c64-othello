"""ctypes wrapper around build/libp2c64_batch.so for P2C-64 v2.0 (Full PTM):
fast OpenMP PTM pattern extraction, exact C++ D4 mirror orbit export, and C++ .p2c evaluation."""

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

KEY_RAW_ENTRIES = 91854
NONKEY_RAW_ENTRIES = 318168
TOTAL_RAW_ENTRIES = 410022
EDGE_OFF = 19683       # 3^9 (Corner 3x3)
EDGE_RAW_SIZE = 59049  # 3^10 (Edge+2X)

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
    raise FileNotFoundError("Could not locate build/libp2c64_batch.so. Run `make tools` first.")


class P2C64Bridge:
    def __init__(self, lib_path: Optional[str] = None) -> None:
        self.lib_path = _find_library(lib_path)
        self.lib = ctypes.CDLL(str(self.lib_path))

        c_u64_p = ctypes.POINTER(ctypes.c_uint64)
        c_u8_p = ctypes.POINTER(ctypes.c_uint8)
        c_i32_p = ctypes.POINTER(ctypes.c_int32)
        c_i8_p = ctypes.POINTER(ctypes.c_int8)

        self.lib.p2c64_extract_batch_ptm.argtypes = [
            c_u64_p, c_u64_p, c_i8_p,
            ctypes.c_int32, ctypes.c_uint64, ctypes.c_int32,
            c_i32_p, c_i32_p, c_i32_p,
        ]
        self.lib.p2c64_extract_batch_ptm.restype = None

        self.lib.p2c64_get_constants_v2.argtypes = [c_i32_p] * 6
        self.lib.p2c64_get_constants_v2.restype = None

        self.lib.p2c64_get_orbit_tables_v2.argtypes = [c_i32_p] * 4
        self.lib.p2c64_get_orbit_tables_v2.restype = None

        self.lib.p2c64_eval_file_batch.argtypes = [
            ctypes.c_char_p,
            c_u64_p, c_u64_p, c_i8_p, c_u8_p,
            ctypes.c_int32, c_i32_p, c_i32_p,
        ]
        self.lib.p2c64_eval_file_batch.restype = ctypes.c_int32

        c_i16_p = ctypes.POINTER(ctypes.c_int16)
        self.lib.waca_g2_extract_batch_features.argtypes = [
            c_u64_p, c_u64_p, c_i8_p, c_u8_p,
            ctypes.c_int32, c_i16_p, ctypes.c_uint64, ctypes.c_int32,
            c_i32_p, c_i32_p, c_i32_p, c_i32_p,
        ]
        self.lib.waca_g2_extract_batch_features.restype = None

        self.lib.waca_g2_eval_file_batch_stages.argtypes = [
            ctypes.c_char_p,
            c_u64_p, c_u64_p, c_i8_p, c_u8_p,
            ctypes.c_int32, c_i32_p, c_i32_p, c_i32_p, c_i32_p,
        ]
        self.lib.waca_g2_eval_file_batch_stages.restype = ctypes.c_int32

        self.lib.waca_g2_v31_extract_batch_features.argtypes = [
            c_u64_p, c_u64_p, c_i8_p, c_u8_p,
            ctypes.c_int32, c_i16_p, ctypes.c_uint64, ctypes.c_int32,
            c_i32_p, c_i32_p, c_i32_p, c_i32_p,
        ]
        self.lib.waca_g2_v31_extract_batch_features.restype = None

        self.lib.waca_g2_v31_eval_file_batch_stages.argtypes = [
            ctypes.c_char_p,
            c_u64_p, c_u64_p, c_i8_p, c_u8_p,
            ctypes.c_int32, c_i32_p, c_i32_p, c_i32_p, c_i32_p,
        ]
        self.lib.waca_g2_v31_eval_file_batch_stages.restype = ctypes.c_int32

        self.lib.waca_g2_v32_extract_batch_features.argtypes = [
            c_u64_p, c_u64_p, c_i8_p, c_u8_p,
            ctypes.c_int32, c_i16_p, ctypes.c_uint64, ctypes.c_int32,
            c_i32_p, c_i32_p, c_i32_p, c_i32_p,
        ]
        self.lib.waca_g2_v32_extract_batch_features.restype = None

        self.lib.waca_g2_v32_eval_file_batch_stages.argtypes = [
            ctypes.c_char_p,
            c_u64_p, c_u64_p, c_i8_p, c_u8_p,
            ctypes.c_int32, c_i32_p, c_i32_p, c_i32_p, c_i32_p,
        ]
        self.lib.waca_g2_v32_eval_file_batch_stages.restype = ctypes.c_int32

        self.lib.p2c64_get_ctr_orbit_table.argtypes = [c_i32_p]
        self.lib.p2c64_get_ctr_orbit_table.restype = None

        self.lib.waca_g2_v33_extract_batch_features.argtypes = [
            c_u64_p, c_u64_p, c_i8_p, c_u8_p,
            ctypes.c_int32, c_i16_p, ctypes.c_uint64, ctypes.c_int32,
            c_i32_p, c_i32_p, c_i32_p, c_i32_p, c_i32_p,
        ]
        self.lib.waca_g2_v33_extract_batch_features.restype = None

        self.lib.waca_g2_v33_eval_file_batch_stages.argtypes = [
            ctypes.c_char_p,
            c_u64_p, c_u64_p, c_i8_p, c_u8_p,
            ctypes.c_int32, c_i32_p, c_i32_p, c_i32_p, c_i32_p,
        ]
        self.lib.waca_g2_v33_eval_file_batch_stages.restype = ctypes.c_int32

        self.lib.swar_v40_eval_file_batch.argtypes = [
            ctypes.c_char_p,
            c_u64_p, c_u64_p, c_i8_p, c_u8_p,
            ctypes.c_int32, c_i32_p, c_i32_p,
        ]
        self.lib.swar_v40_eval_file_batch.restype = ctypes.c_int32

        self.lib.p2c64_fnv1a_64.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
        self.lib.p2c64_fnv1a_64.restype = ctypes.c_uint64

        self._orbit_cache: Optional[Dict[str, np.ndarray]] = None

    def get_orbit_tables(self) -> Dict[str, np.ndarray]:
        if self._orbit_cache is not None:
            return self._orbit_cache
        key_orb_rep = np.zeros(KEY_RAW_ENTRIES, dtype=np.int32)
        edge_mirror_x = np.zeros(EDGE_RAW_SIZE, dtype=np.int32)
        nk_orb_rep = np.zeros(NONKEY_RAW_ENTRIES, dtype=np.int32)
        raw_orb_rep = np.zeros(TOTAL_RAW_ENTRIES, dtype=np.int32)
        ctr_orb_rep = np.zeros(19683, dtype=np.int32)

        c_i32_p = ctypes.POINTER(ctypes.c_int32)
        self.lib.p2c64_get_orbit_tables_v2(
            key_orb_rep.ctypes.data_as(c_i32_p),
            edge_mirror_x.ctypes.data_as(c_i32_p),
            nk_orb_rep.ctypes.data_as(c_i32_p),
            raw_orb_rep.ctypes.data_as(c_i32_p),
        )
        self.lib.p2c64_get_ctr_orbit_table(ctr_orb_rep.ctypes.data_as(c_i32_p))
        self._orbit_cache = {
            "key_orb_rep": key_orb_rep,
            "edge_mirror_x": edge_mirror_x,
            "nk_orb_rep": nk_orb_rep,
            "raw_orb_rep": raw_orb_rep,
            "ctr_orb_rep": ctr_orb_rep,
        }
        return self._orbit_cache

    def extract_batch_ptm(
        self,
        black: np.ndarray,
        white: np.ndarray,
        stm: np.ndarray,
        aug_seed: int = 0,
        enable_aug: bool = False,
        need_p2c: bool = True,
        need_raw: bool = False,
    ) -> Dict[str, np.ndarray]:
        black = np.ascontiguousarray(black, dtype=np.uint64)
        white = np.ascontiguousarray(white, dtype=np.uint64)
        stm = np.ascontiguousarray(stm, dtype=np.int8)
        n = int(black.shape[0])

        c_u64_p = ctypes.POINTER(ctypes.c_uint64)
        c_i32_p = ctypes.POINTER(ctypes.c_int32)
        c_i8_p = ctypes.POINTER(ctypes.c_int8)

        out: Dict[str, np.ndarray] = {}
        if need_p2c:
            key_ptm = np.empty((n, NUM_KEY_PATTERNS), dtype=np.int32)
            nk_ptm = np.empty((n, NUM_NONKEY_PATTERNS), dtype=np.int32)
            p_k = key_ptm.ctypes.data_as(c_i32_p)
            p_nk = nk_ptm.ctypes.data_as(c_i32_p)
            out["key_ptm"] = key_ptm
            out["nonkey_ptm"] = nk_ptm
        else:
            p_k = ctypes.cast(0, c_i32_p)
            p_nk = ctypes.cast(0, c_i32_p)

        if need_raw:
            raw_ptm = np.empty((n, NUM_PATTERNS), dtype=np.int32)
            p_r = raw_ptm.ctypes.data_as(c_i32_p)
            out["raw_ptm"] = raw_ptm
        else:
            p_r = ctypes.cast(0, c_i32_p)

        self.lib.p2c64_extract_batch_ptm(
            black.ctypes.data_as(c_u64_p),
            white.ctypes.data_as(c_u64_p),
            stm.ctypes.data_as(c_i8_p),
            ctypes.c_int32(n),
            ctypes.c_uint64(aug_seed & 0xFFFFFFFFFFFFFFFF),
            ctypes.c_int32(1 if enable_aug else 0),
            p_k,
            p_nk,
            p_r,
        )
        return out

    def extract_waca_g2_batch(
        self,
        black: np.ndarray,
        white: np.ndarray,
        stm: np.ndarray,
        ply: np.ndarray,
        tier1_w_ply: np.ndarray,
        aug_seed: int = 0,
        enable_aug: bool = False,
        need_stage3: bool = True,
    ) -> Dict[str, np.ndarray]:
        black = np.ascontiguousarray(black, dtype=np.uint64)
        white = np.ascontiguousarray(white, dtype=np.uint64)
        stm = np.ascontiguousarray(stm, dtype=np.int8)
        ply = np.ascontiguousarray(ply, dtype=np.uint8)
        t1w = np.ascontiguousarray(tier1_w_ply, dtype=np.int16)
        assert t1w.shape == (NUM_PLIES, TOTAL_RAW_ENTRIES)
        n = int(black.shape[0])

        c_u64_p = ctypes.POINTER(ctypes.c_uint64)
        c_u8_p = ctypes.POINTER(ctypes.c_uint8)
        c_i8_p = ctypes.POINTER(ctypes.c_int8)
        c_i16_p = ctypes.POINTER(ctypes.c_int16)
        c_i32_p = ctypes.POINTER(ctypes.c_int32)

        v1 = np.empty(n, dtype=np.int32)
        f25 = np.empty((n, 16), dtype=np.int32)
        out: Dict[str, np.ndarray] = {"v1": v1, "f25": f25}
        if need_stage3:
            f3_geom = np.empty((n, 4), dtype=np.int32)
            key_ptm = np.empty((n, NUM_KEY_PATTERNS), dtype=np.int32)
            p_f3 = f3_geom.ctypes.data_as(c_i32_p)
            p_kp = key_ptm.ctypes.data_as(c_i32_p)
            out["f3_geom"] = f3_geom
            out["key_ptm"] = key_ptm
        else:
            p_f3 = ctypes.cast(0, c_i32_p)
            p_kp = ctypes.cast(0, c_i32_p)

        self.lib.waca_g2_extract_batch_features(
            black.ctypes.data_as(c_u64_p),
            white.ctypes.data_as(c_u64_p),
            stm.ctypes.data_as(c_i8_p),
            ply.ctypes.data_as(c_u8_p),
            ctypes.c_int32(n),
            t1w.ctypes.data_as(c_i16_p),
            ctypes.c_uint64(aug_seed & 0xFFFFFFFFFFFFFFFF),
            ctypes.c_int32(1 if enable_aug else 0),
            v1.ctypes.data_as(c_i32_p),
            f25.ctypes.data_as(c_i32_p),
            p_f3,
            p_kp,
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

    def eval_waca_g2_file_stages(
        self,
        p2c_path: str | os.PathLike,
        black: np.ndarray,
        white: np.ndarray,
        stm: np.ndarray,
        ply: np.ndarray,
        check_16sym: bool = False,
    ) -> Tuple[int, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        black = np.ascontiguousarray(black, dtype=np.uint64)
        white = np.ascontiguousarray(white, dtype=np.uint64)
        stm = np.ascontiguousarray(stm, dtype=np.int8)
        ply = np.ascontiguousarray(ply, dtype=np.uint8)
        n = int(black.shape[0])
        v1 = np.empty(n, dtype=np.int32)
        v25 = np.empty(n, dtype=np.int32)
        v3 = np.empty(n, dtype=np.int32)
        sym3 = np.zeros(3, dtype=np.int32)
        c_i32_p = ctypes.POINTER(ctypes.c_int32)

        h = self.lib.waca_g2_eval_file_batch_stages(
            os.fsencode(str(p2c_path)),
            black.ctypes.data_as(ctypes.POINTER(ctypes.c_uint64)),
            white.ctypes.data_as(ctypes.POINTER(ctypes.c_uint64)),
            stm.ctypes.data_as(ctypes.POINTER(ctypes.c_int8)),
            ply.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
            ctypes.c_int32(n),
            v1.ctypes.data_as(c_i32_p),
            v25.ctypes.data_as(c_i32_p),
            v3.ctypes.data_as(c_i32_p),
            sym3.ctypes.data_as(c_i32_p) if check_16sym else ctypes.cast(0, c_i32_p),
        )
        if h < 0:
            raise RuntimeError(f"C++ waca_g2_eval_file_batch_stages failed on {p2c_path}")
        return int(h), v1, v25, v3, sym3

    def extract_waca_g2_v31_batch(
        self,
        black: np.ndarray,
        white: np.ndarray,
        stm: np.ndarray,
        ply: np.ndarray,
        tier1_w_ply: np.ndarray,
        aug_seed: int = 0,
        enable_aug: bool = False,
        need_stage3: bool = True,
    ) -> Dict[str, np.ndarray]:
        black = np.ascontiguousarray(black, dtype=np.uint64)
        white = np.ascontiguousarray(white, dtype=np.uint64)
        stm = np.ascontiguousarray(stm, dtype=np.int8)
        ply = np.ascontiguousarray(ply, dtype=np.uint8)
        t1w = np.ascontiguousarray(tier1_w_ply, dtype=np.int16)
        assert t1w.shape == (NUM_PLIES, TOTAL_RAW_ENTRIES)
        n = int(black.shape[0])

        c_u64_p = ctypes.POINTER(ctypes.c_uint64)
        c_u8_p = ctypes.POINTER(ctypes.c_uint8)
        c_i8_p = ctypes.POINTER(ctypes.c_int8)
        c_i16_p = ctypes.POINTER(ctypes.c_int16)
        c_i32_p = ctypes.POINTER(ctypes.c_int32)

        v1 = np.empty(n, dtype=np.int32)
        f25 = np.empty((n, 8), dtype=np.int32)
        out: Dict[str, np.ndarray] = {"v1": v1, "f25": f25}
        if need_stage3:
            f3_extra = np.empty((n, 20), dtype=np.int32)
            key_ptm = np.empty((n, NUM_KEY_PATTERNS), dtype=np.int32)
            p_f3 = f3_extra.ctypes.data_as(c_i32_p)
            p_kp = key_ptm.ctypes.data_as(c_i32_p)
            out["f3_extra"] = f3_extra
            out["key_ptm"] = key_ptm
        else:
            p_f3 = ctypes.cast(0, c_i32_p)
            p_kp = ctypes.cast(0, c_i32_p)

        self.lib.waca_g2_v31_extract_batch_features(
            black.ctypes.data_as(c_u64_p),
            white.ctypes.data_as(c_u64_p),
            stm.ctypes.data_as(c_i8_p),
            ply.ctypes.data_as(c_u8_p),
            ctypes.c_int32(n),
            t1w.ctypes.data_as(c_i16_p),
            ctypes.c_uint64(aug_seed & 0xFFFFFFFFFFFFFFFF),
            ctypes.c_int32(1 if enable_aug else 0),
            v1.ctypes.data_as(c_i32_p),
            f25.ctypes.data_as(c_i32_p),
            p_f3,
            p_kp,
        )
        return out

    def eval_waca_g2_v31_file_stages(
        self,
        p2c_path: str | os.PathLike,
        black: np.ndarray,
        white: np.ndarray,
        stm: np.ndarray,
        ply: np.ndarray,
        check_16sym: bool = False,
    ) -> Tuple[int, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        black = np.ascontiguousarray(black, dtype=np.uint64)
        white = np.ascontiguousarray(white, dtype=np.uint64)
        stm = np.ascontiguousarray(stm, dtype=np.int8)
        ply = np.ascontiguousarray(ply, dtype=np.uint8)
        n = int(black.shape[0])
        v1 = np.empty(n, dtype=np.int32)
        v25 = np.empty(n, dtype=np.int32)
        v3 = np.empty(n, dtype=np.int32)
        sym3 = np.zeros(3, dtype=np.int32)
        c_i32_p = ctypes.POINTER(ctypes.c_int32)

        h = self.lib.waca_g2_v31_eval_file_batch_stages(
            os.fsencode(str(p2c_path)),
            black.ctypes.data_as(ctypes.POINTER(ctypes.c_uint64)),
            white.ctypes.data_as(ctypes.POINTER(ctypes.c_uint64)),
            stm.ctypes.data_as(ctypes.POINTER(ctypes.c_int8)),
            ply.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
            ctypes.c_int32(n),
            v1.ctypes.data_as(c_i32_p),
            v25.ctypes.data_as(c_i32_p),
            v3.ctypes.data_as(c_i32_p),
            sym3.ctypes.data_as(c_i32_p) if check_16sym else ctypes.cast(0, c_i32_p),
        )
        if h < 0:
            raise RuntimeError(f"C++ waca_g2_v31_eval_file_batch_stages failed on {p2c_path}")
        return int(h), v1, v25, v3, sym3

    def extract_waca_g2_v32_batch(
        self,
        black: np.ndarray,
        white: np.ndarray,
        stm: np.ndarray,
        ply: np.ndarray,
        tier1_w_ply: np.ndarray,
        aug_seed: int = 0,
        enable_aug: bool = False,
        need_stage3: bool = True,
    ) -> Dict[str, np.ndarray]:
        black = np.ascontiguousarray(black, dtype=np.uint64)
        white = np.ascontiguousarray(white, dtype=np.uint64)
        stm = np.ascontiguousarray(stm, dtype=np.int8)
        ply = np.ascontiguousarray(ply, dtype=np.uint8)
        t1w = np.ascontiguousarray(tier1_w_ply, dtype=np.int16)
        assert t1w.shape == (NUM_PLIES, TOTAL_RAW_ENTRIES)
        n = int(black.shape[0])

        c_u64_p = ctypes.POINTER(ctypes.c_uint64)
        c_u8_p = ctypes.POINTER(ctypes.c_uint8)
        c_i8_p = ctypes.POINTER(ctypes.c_int8)
        c_i16_p = ctypes.POINTER(ctypes.c_int16)
        c_i32_p = ctypes.POINTER(ctypes.c_int32)

        v1 = np.empty(n, dtype=np.int32)
        f25 = np.empty((n, 8), dtype=np.int32)
        out: Dict[str, np.ndarray] = {"v1": v1, "f25": f25}
        if need_stage3:
            f3_extra = np.empty((n, 16), dtype=np.int32)
            key_ptm = np.empty((n, NUM_KEY_PATTERNS), dtype=np.int32)
            p_f3 = f3_extra.ctypes.data_as(c_i32_p)
            p_kp = key_ptm.ctypes.data_as(c_i32_p)
            out["f3_extra"] = f3_extra
            out["key_ptm"] = key_ptm
        else:
            p_f3 = ctypes.cast(0, c_i32_p)
            p_kp = ctypes.cast(0, c_i32_p)

        self.lib.waca_g2_v32_extract_batch_features(
            black.ctypes.data_as(c_u64_p),
            white.ctypes.data_as(c_u64_p),
            stm.ctypes.data_as(c_i8_p),
            ply.ctypes.data_as(c_u8_p),
            ctypes.c_int32(n),
            t1w.ctypes.data_as(c_i16_p),
            ctypes.c_uint64(aug_seed & 0xFFFFFFFFFFFFFFFF),
            ctypes.c_int32(1 if enable_aug else 0),
            v1.ctypes.data_as(c_i32_p),
            f25.ctypes.data_as(c_i32_p),
            p_f3,
            p_kp,
        )
        return out

    def eval_waca_g2_v32_file_stages(
        self,
        p2c_path: str | os.PathLike,
        black: np.ndarray,
        white: np.ndarray,
        stm: np.ndarray,
        ply: np.ndarray,
        check_16sym: bool = False,
    ) -> Tuple[int, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        black = np.ascontiguousarray(black, dtype=np.uint64)
        white = np.ascontiguousarray(white, dtype=np.uint64)
        stm = np.ascontiguousarray(stm, dtype=np.int8)
        ply = np.ascontiguousarray(ply, dtype=np.uint8)
        n = int(black.shape[0])
        v1 = np.empty(n, dtype=np.int32)
        v25 = np.empty(n, dtype=np.int32)
        v3 = np.empty(n, dtype=np.int32)
        sym3 = np.zeros(3, dtype=np.int32)
        c_i32_p = ctypes.POINTER(ctypes.c_int32)

        h = self.lib.waca_g2_v32_eval_file_batch_stages(
            os.fsencode(str(p2c_path)),
            black.ctypes.data_as(ctypes.POINTER(ctypes.c_uint64)),
            white.ctypes.data_as(ctypes.POINTER(ctypes.c_uint64)),
            stm.ctypes.data_as(ctypes.POINTER(ctypes.c_int8)),
            ply.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
            ctypes.c_int32(n),
            v1.ctypes.data_as(c_i32_p),
            v25.ctypes.data_as(c_i32_p),
            v3.ctypes.data_as(c_i32_p),
            sym3.ctypes.data_as(c_i32_p) if check_16sym else ctypes.cast(0, c_i32_p),
        )
        if h < 0:
            raise RuntimeError(f"C++ waca_g2_v32_eval_file_batch_stages failed on {p2c_path}")
        return int(h), v1, v25, v3, sym3

    def extract_waca_g2_v33_batch(
        self,
        black: np.ndarray,
        white: np.ndarray,
        stm: np.ndarray,
        ply: np.ndarray,
        tier1_w_ply: np.ndarray,
        aug_seed: int = 0,
        enable_aug: bool = False,
        need_stage3: bool = True,
    ) -> Dict[str, np.ndarray]:
        black = np.ascontiguousarray(black, dtype=np.uint64)
        white = np.ascontiguousarray(white, dtype=np.uint64)
        stm = np.ascontiguousarray(stm, dtype=np.int8)
        ply = np.ascontiguousarray(ply, dtype=np.uint8)
        t1w = np.ascontiguousarray(tier1_w_ply, dtype=np.int16)
        assert t1w.shape == (NUM_PLIES, TOTAL_RAW_ENTRIES)
        n = int(black.shape[0])

        c_u64_p = ctypes.POINTER(ctypes.c_uint64)
        c_u8_p = ctypes.POINTER(ctypes.c_uint8)
        c_i8_p = ctypes.POINTER(ctypes.c_int8)
        c_i16_p = ctypes.POINTER(ctypes.c_int16)
        c_i32_p = ctypes.POINTER(ctypes.c_int32)

        v1 = np.empty(n, dtype=np.int32)
        f25 = np.empty((n, 12), dtype=np.int32)
        out: Dict[str, np.ndarray] = {"v1": v1, "f25": f25}
        if need_stage3:
            f3_extra = np.empty((n, 16), dtype=np.int32)
            key_ptm = np.empty((n, NUM_KEY_PATTERNS), dtype=np.int32)
            ctr_ptm = np.empty((n, 10), dtype=np.int32)
            p_f3 = f3_extra.ctypes.data_as(c_i32_p)
            p_kp = key_ptm.ctypes.data_as(c_i32_p)
            p_cp = ctr_ptm.ctypes.data_as(c_i32_p)
            out["f3_extra"] = f3_extra
            out["key_ptm"] = key_ptm
            out["ctr_ptm"] = ctr_ptm
        else:
            p_f3 = ctypes.cast(0, c_i32_p)
            p_kp = ctypes.cast(0, c_i32_p)
            p_cp = ctypes.cast(0, c_i32_p)

        self.lib.waca_g2_v33_extract_batch_features(
            black.ctypes.data_as(c_u64_p),
            white.ctypes.data_as(c_u64_p),
            stm.ctypes.data_as(c_i8_p),
            ply.ctypes.data_as(c_u8_p),
            ctypes.c_int32(n),
            t1w.ctypes.data_as(c_i16_p),
            ctypes.c_uint64(aug_seed & 0xFFFFFFFFFFFFFFFF),
            ctypes.c_int32(1 if enable_aug else 0),
            v1.ctypes.data_as(c_i32_p),
            f25.ctypes.data_as(c_i32_p),
            p_f3,
            p_kp,
            p_cp,
        )
        return out

    def eval_waca_g2_v33_file_stages(
        self,
        p2c_path: str | os.PathLike,
        black: np.ndarray,
        white: np.ndarray,
        stm: np.ndarray,
        ply: np.ndarray,
        check_16sym: bool = False,
    ) -> Tuple[int, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        black = np.ascontiguousarray(black, dtype=np.uint64)
        white = np.ascontiguousarray(white, dtype=np.uint64)
        stm = np.ascontiguousarray(stm, dtype=np.int8)
        ply = np.ascontiguousarray(ply, dtype=np.uint8)
        n = int(black.shape[0])
        v1 = np.empty(n, dtype=np.int32)
        v25 = np.empty(n, dtype=np.int32)
        v3 = np.empty(n, dtype=np.int32)
        sym3 = np.zeros(3, dtype=np.int32)
        c_i32_p = ctypes.POINTER(ctypes.c_int32)

        h = self.lib.waca_g2_v33_eval_file_batch_stages(
            os.fsencode(str(p2c_path)),
            black.ctypes.data_as(ctypes.POINTER(ctypes.c_uint64)),
            white.ctypes.data_as(ctypes.POINTER(ctypes.c_uint64)),
            stm.ctypes.data_as(ctypes.POINTER(ctypes.c_int8)),
            ply.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
            ctypes.c_int32(n),
            v1.ctypes.data_as(c_i32_p),
            v25.ctypes.data_as(c_i32_p),
            v3.ctypes.data_as(c_i32_p),
            sym3.ctypes.data_as(c_i32_p) if check_16sym else ctypes.cast(0, c_i32_p),
        )
        if h < 0:
            raise RuntimeError(f"C++ waca_g2_v33_eval_file_batch_stages failed on {p2c_path}")
        return int(h), v1, v25, v3, sym3

    def eval_swar_v40_file(
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
        sym_err = ctypes.c_int32(0)
        c_i32_p = ctypes.POINTER(ctypes.c_int32)

        mode = self.lib.swar_v40_eval_file_batch(
            os.fsencode(str(p2c_path)),
            black.ctypes.data_as(ctypes.POINTER(ctypes.c_uint64)),
            white.ctypes.data_as(ctypes.POINTER(ctypes.c_uint64)),
            stm.ctypes.data_as(ctypes.POINTER(ctypes.c_int8)),
            ply.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
            ctypes.c_int32(n),
            scores.ctypes.data_as(c_i32_p),
            ctypes.byref(sym_err) if check_16sym else ctypes.cast(0, c_i32_p),
        )
        if mode < 0:
            raise RuntimeError(f"C++ swar_v40_eval_file_batch failed on {p2c_path}")
        return int(mode), scores, int(sym_err.value)
