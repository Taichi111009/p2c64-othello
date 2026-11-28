#!/usr/bin/env python3
"""P2C-64 v4.0 (Scenario 3: Path A-I + Axis B) End-to-End PyTorch Training Pipeline.

Supports two modes sharing identical relabeled blended targets & dual-axis validation:
  1. --mode swar_v40    : Loads raw_ntuple64.p2c as Tier-1 init, UNFREEZES Tier-1
                          (requires_grad=True), and jointly trains SWAR_V40
                          (emb_ch in {4, 8}, num_blocks in {2, 30}, local_relu in {0, 1})
                          with 7 Float + 2 QAT epochs, Best-Float reload + AdamW reset,
                          and same-parity EO block proximal smoothing.
  2. --mode raw_relabel : Trains control baseline raw_ntuple64_relabel.p2c on the exact
                          same relabeled blended targets to cleanly isolate data gain vs
                          model architecture gain.

Blended targets:
  - Sidecar int16 files (--relabel-train / --relabel-val, 128x fixed-point, INT16_MIN=-32768
    sentinel for unrelabeled positions):
      y = (1 - lambda) * y_final + lambda * (y_relabel / 128.0)   where sidecar != -32768
      y = y_final                                                 otherwise
Dual-axis validation:
  - Every epoch & final quantized export logs BOTH:
      (1) Original val.bin (lambda=0) Val RMSE (all, 0-19, 20-39, 40-59)
      (2) Relabeled Val (blended lambda AND pure d6 subset) Val RMSE
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))

from p2c64_bridge import (
    KEY_RAW_ENTRIES,
    NUM_PLIES,
    PACKED_SAMPLE_DTYPE,
    TOTAL_RAW_ENTRIES,
    P2C64Bridge,
)
from p2c64_io import (
    P2C_DEFAULT_SCALE,
    PLY_TO_EO30_BLK,
    SCRELU_Q,
    SCRELU_SHIFT,
    SWAR_V40_BETA_MAX,
    SWAR_V40_H_MAX,
    SWAR_V40_MU_MAX,
    WACA_MAX_RES,
    RawNTupleModel,
    SwarV40Model,
    eval_raw_ntuple64_int_batch,
    eval_swar_v40_int_batch,
    load_p2c,
    make_empty_swar_v40,
    save_raw_p2c,
    save_swar_v40,
    symmetrize_raw_model,
    symmetrize_swar_v40_model,
)

EMB_SCALE = 32.0
RELABEL_SENTINEL = -32768


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_packed_dataset(path: Path, max_samples: int = 0) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    arr = np.fromfile(path, dtype=PACKED_SAMPLE_DTYPE)
    if len(arr) == 0:
        raise RuntimeError(f"Dataset file is empty: {path}")
    if max_samples > 0 and len(arr) > max_samples:
        idx = np.linspace(0, len(arr) - 1, max_samples, dtype=np.int64)
        return np.ascontiguousarray(arr[idx]), idx
    return arr, None


def load_relabel_sidecar(path: Optional[Path], n_expected: int, sub_idx: Optional[np.ndarray]) -> Optional[np.ndarray]:
    if path is None:
        return None
    sc = np.fromfile(path, dtype="<i2")
    if sub_idx is not None:
        if len(sc) < int(sub_idx[-1]) + 1:
            raise RuntimeError(f"Sidecar '{path}' too short ({len(sc)}) for sub_idx max {sub_idx[-1]}")
        sc = np.ascontiguousarray(sc[sub_idx])
    elif len(sc) != n_expected:
        raise RuntimeError(f"Sidecar '{path}' length {len(sc)} != dataset length {n_expected}")
    return sc


def build_blended_targets(
    y_final: np.ndarray,
    sidecar: Optional[np.ndarray],
    lam: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns (y_blend_f32, y_relabel_f32, relabel_mask_bool)."""
    yf = y_final.astype(np.float32)
    if sidecar is None:
        mask = np.zeros(len(yf), dtype=bool)
        return yf.copy(), yf.copy(), mask
    mask = sidecar != RELABEL_SENTINEL
    yr = np.where(mask, sidecar.astype(np.float32) / 128.0, yf).astype(np.float32)
    yb = np.where(mask, (1.0 - lam) * yf + lam * yr, yf).astype(np.float32)
    return yb, yr, mask


def ste_round_clamp(x: torch.Tensor, lo: float, hi: float) -> torch.Tensor:
    y = torch.clamp(torch.round(x), lo, hi)
    return x + (y - x).detach()


def ste_sym_rshift_int(x: torch.Tensor, shift: int) -> torch.Tensor:
    denom = float(1 << shift)
    xf = x / denom
    y = torch.trunc(xf)
    return xf + (y - xf).detach()


@torch.no_grad()
def apply_axis0_smoothness_(w: torch.Tensor, alpha: float) -> None:
    if alpha <= 0.0 or w.shape[0] < 2:
        return
    a = min(0.20, float(alpha))
    w_prev = w.clone()
    w[0].add_(w_prev[1] - w_prev[0], alpha=a)
    w[-1].add_(w_prev[-2] - w_prev[-1], alpha=a)
    if w.shape[0] > 2:
        w[1:-1].add_(w_prev[:-2] + w_prev[2:] - 2.0 * w_prev[1:-1], alpha=a)


@torch.no_grad()
def apply_eo_block_smoothness_(emb: torch.Tensor, alpha: float) -> None:
    if alpha <= 0.0 or emb.shape[0] < 4:
        return
    for parity in (0, 1):
        sub = emb[parity::2].clone()
        apply_axis0_smoothness_(sub, alpha)
        emb[parity::2].copy_(sub)


def compute_metrics_dict(preds_discs: np.ndarray, y_true: np.ndarray, plies: np.ndarray) -> Dict[str, float]:
    err = preds_discs.astype(np.float64) - y_true.astype(np.float64)
    res: Dict[str, float] = {
        "rmse_all": float(np.sqrt(np.mean(err * err))),
        "mae_all": float(np.mean(np.abs(err))),
    }
    for name, lo, hi in (("0_19", 0, 19), ("20_39", 20, 39), ("40_59", 40, 59)):
        m = (plies >= lo) & (plies <= hi)
        if np.any(m):
            em = err[m]
            res[f"rmse_{name}"] = float(np.sqrt(np.mean(em * em)))
            res[f"mae_{name}"] = float(np.mean(np.abs(em)))
        else:
            res[f"rmse_{name}"] = 0.0
            res[f"mae_{name}"] = 0.0
    return res


def compute_dual_val_metrics(
    preds_discs: np.ndarray,
    y_final: np.ndarray,
    y_blend: np.ndarray,
    y_relabel: np.ndarray,
    rel_mask: np.ndarray,
    plies: np.ndarray,
) -> Dict[str, float]:
    m_orig = compute_metrics_dict(preds_discs, y_final, plies)
    m_blend = compute_metrics_dict(preds_discs, y_blend, plies)
    out: Dict[str, float] = {}
    for k, v in m_orig.items():
        out[f"orig_{k}"] = v
    for k, v in m_blend.items():
        out[f"blend_{k}"] = v
    if np.any(rel_mask):
        e_sub = preds_discs[rel_mask].astype(np.float64) - y_relabel[rel_mask].astype(np.float64)
        out["relabel_sub_rmse"] = float(np.sqrt(np.mean(e_sub * e_sub)))
        out["relabel_sub_mae"] = float(np.mean(np.abs(e_sub)))
    else:
        out["relabel_sub_rmse"] = m_orig["rmse_all"]
        out["relabel_sub_mae"] = m_orig["mae_all"]
    return out


# ============================================================================
# PyTorch Module 1: End-to-End Unfrozen Tier-1 + SWAR_V40
# ============================================================================
class SwarV40Net(nn.Module):
    def __init__(
        self,
        init_tier1_i16: np.ndarray,
        emb_ch: int = 8,
        num_blocks: int = 30,
        local_relu: int = 0,
    ) -> None:
        super().__init__()
        self.emb_ch = int(emb_ch)
        self.num_blocks = int(num_blocks)
        self.local_relu = int(local_relu)
        self.n_g_terms = 12 if self.emb_ch == 8 else 6

        # Unfrozen Tier-1 weights in disc units (initialized from raw_ntuple64.p2c)
        t1_init = torch.from_numpy(init_tier1_i16.astype(np.float32) / P2C_DEFAULT_SCALE)
        self.tier1_w = nn.Parameter(t1_init)  # [60, 410022]

        # Key-14 SWAR embeddings (scaled by EMB_SCALE=32 -> int8 [-127, 127])
        h_init = torch.empty((self.num_blocks, KEY_RAW_ENTRIES, self.emb_ch), dtype=torch.float32)
        nn.init.normal_(h_init, mean=0.0, std=0.15)
        self.h_emb = nn.Parameter(h_init)

        # 60-ply head parameters (in integer-ready units)
        self.mu_g = nn.Parameter(torch.full((NUM_PLIES, self.emb_ch), 12.0, dtype=torch.float32))
        self.beta_g = nn.Parameter(torch.zeros((NUM_PLIES, self.n_g_terms), dtype=torch.float32))
        self.mu_e = nn.Parameter(torch.full((NUM_PLIES, 4), 12.0, dtype=torch.float32))
        self.mu_d = nn.Parameter(torch.full((NUM_PLIES, 4), 12.0, dtype=torch.float32))
        self.beta_e = nn.Parameter(torch.zeros((NUM_PLIES, 4), dtype=torch.float32))
        self.beta_d = nn.Parameter(torch.zeros((NUM_PLIES, 4), dtype=torch.float32))
        self.b_ply = nn.Parameter(torch.zeros((NUM_PLIES,), dtype=torch.float32))

        self.register_buffer("ply_to_eo30", torch.from_numpy(PLY_TO_EO30_BLK.astype(np.int64)))

    def forward(self, raw_ptm: torch.Tensor, ply: torch.Tensor, qat: bool = False) -> torch.Tensor:
        # raw_ptm: [B, 64] int64, ply: [B] int64
        ply_idx = ply.unsqueeze(1)  # [B, 1]
        blk = (ply & 1) if self.num_blocks == 2 else self.ply_to_eo30[ply]
        key_ptm = raw_ptm[:, :14]  # [B, 14]

        if qat:
            w1_q = ste_round_clamp(self.tier1_w[ply_idx, raw_ptm] * P2C_DEFAULT_SCALE, -32767.0, 32767.0)
            v1_int = w1_q.sum(dim=1)

            h_q = ste_round_clamp(self.h_emb[blk.unsqueeze(1), key_ptm] * EMB_SCALE, -float(SWAR_V40_H_MAX), float(SWAR_V40_H_MAX))
            mu_g_q = ste_round_clamp(self.mu_g[ply], -float(SWAR_V40_MU_MAX), float(SWAR_V40_MU_MAX))
            beta_g_q = ste_round_clamp(self.beta_g[ply], -float(SWAR_V40_BETA_MAX), float(SWAR_V40_BETA_MAX))
            b_ply_q = ste_round_clamp(self.b_ply[ply], -float(WACA_MAX_RES), float(WACA_MAX_RES))

            g_sum = h_q.sum(dim=1)
            ug = torch.clamp(g_sum + mu_g_q, 0.0, float(SCRELU_Q))
            if self.emb_ch == 8:
                terms = torch.cat([ug * ug, ug[:, :4] * ug[:, 4:]], dim=1)
            else:
                terms = torch.cat([ug * ug, (ug[:, 0] * ug[:, 2]).unsqueeze(1), (ug[:, 1] * ug[:, 3]).unsqueeze(1)], dim=1)
            sc_acc = (beta_g_q * terms).sum(dim=1)

            if self.local_relu:
                h4 = h_q[:, :, :4]
                mu_e_q = ste_round_clamp(self.mu_e[ply], -float(SWAR_V40_MU_MAX), float(SWAR_V40_MU_MAX))
                mu_d_q = ste_round_clamp(self.mu_d[ply], -float(SWAR_V40_MU_MAX), float(SWAR_V40_MU_MAX))
                beta_e_q = ste_round_clamp(self.beta_e[ply], -float(SWAR_V40_BETA_MAX), float(SWAR_V40_BETA_MAX))
                beta_d_q = ste_round_clamp(self.beta_d[ply], -float(SWAR_V40_BETA_MAX), float(SWAR_V40_BETA_MAX))
                sum_y0 = (
                    F.relu(h4[:, 4] + h4[:, 8] + h4[:, 0] + h4[:, 1] + mu_e_q)
                    + F.relu(h4[:, 5] + h4[:, 9] + h4[:, 1] + h4[:, 2] + mu_e_q)
                    + F.relu(h4[:, 6] + h4[:, 10] + h4[:, 2] + h4[:, 3] + mu_e_q)
                    + F.relu(h4[:, 7] + h4[:, 11] + h4[:, 3] + h4[:, 0] + mu_e_q)
                )
                sum_y1 = (
                    F.relu(h4[:, 12] + h4[:, 0] + h4[:, 2] + mu_d_q)
                    + F.relu(h4[:, 13] + h4[:, 1] + h4[:, 3] + mu_d_q)
                )
                sc_acc = sc_acc + (beta_e_q * (sum_y0 * 16.0)).sum(dim=1) + (beta_d_q * (sum_y1 * 16.0)).sum(dim=1)

            res_int = torch.clamp(ste_sym_rshift_int(sc_acc, SCRELU_SHIFT) + b_ply_q, -float(WACA_MAX_RES), float(WACA_MAX_RES))
            return (v1_int + res_int) / P2C_DEFAULT_SCALE
        else:
            v1_discs = self.tier1_w[ply_idx, raw_ptm].sum(dim=1)
            h_f = torch.clamp(self.h_emb[blk.unsqueeze(1), key_ptm] * EMB_SCALE, -float(SWAR_V40_H_MAX), float(SWAR_V40_H_MAX))
            mu_g_f = torch.clamp(self.mu_g[ply], -float(SWAR_V40_MU_MAX), float(SWAR_V40_MU_MAX))
            beta_g_f = torch.clamp(self.beta_g[ply], -float(SWAR_V40_BETA_MAX), float(SWAR_V40_BETA_MAX))
            b_ply_f = torch.clamp(self.b_ply[ply], -float(WACA_MAX_RES), float(WACA_MAX_RES))

            g_sum = h_f.sum(dim=1)
            ug = torch.clamp(g_sum + mu_g_f, 0.0, float(SCRELU_Q))
            if self.emb_ch == 8:
                terms = torch.cat([ug * ug, ug[:, :4] * ug[:, 4:]], dim=1)
            else:
                terms = torch.cat([ug * ug, (ug[:, 0] * ug[:, 2]).unsqueeze(1), (ug[:, 1] * ug[:, 3]).unsqueeze(1)], dim=1)
            sc_acc = (beta_g_f * terms).sum(dim=1)

            if self.local_relu:
                h4 = h_f[:, :, :4]
                mu_e_f = torch.clamp(self.mu_e[ply], -float(SWAR_V40_MU_MAX), float(SWAR_V40_MU_MAX))
                mu_d_f = torch.clamp(self.mu_d[ply], -float(SWAR_V40_MU_MAX), float(SWAR_V40_MU_MAX))
                beta_e_f = torch.clamp(self.beta_e[ply], -float(SWAR_V40_BETA_MAX), float(SWAR_V40_BETA_MAX))
                beta_d_f = torch.clamp(self.beta_d[ply], -float(SWAR_V40_BETA_MAX), float(SWAR_V40_BETA_MAX))
                sum_y0 = (
                    F.relu(h4[:, 4] + h4[:, 8] + h4[:, 0] + h4[:, 1] + mu_e_f)
                    + F.relu(h4[:, 5] + h4[:, 9] + h4[:, 1] + h4[:, 2] + mu_e_f)
                    + F.relu(h4[:, 6] + h4[:, 10] + h4[:, 2] + h4[:, 3] + mu_e_f)
                    + F.relu(h4[:, 7] + h4[:, 11] + h4[:, 3] + h4[:, 0] + mu_e_f)
                )
                sum_y1 = (
                    F.relu(h4[:, 12] + h4[:, 0] + h4[:, 2] + mu_d_f)
                    + F.relu(h4[:, 13] + h4[:, 1] + h4[:, 3] + mu_d_f)
                )
                sc_acc = sc_acc + (beta_e_f * (sum_y0 * 16.0)).sum(dim=1) + (beta_d_f * (sum_y1 * 16.0)).sum(dim=1)

            res_int = torch.clamp(sc_acc / float(1 << SCRELU_SHIFT) + b_ply_f, -float(WACA_MAX_RES), float(WACA_MAX_RES))
            return v1_discs + res_int / P2C_DEFAULT_SCALE

    @torch.no_grad()
    def export_quantized(self, bridge: P2C64Bridge) -> SwarV40Model:
        orbs = bridge.get_orbit_tables()
        raw_rep = orbs["raw_orb_rep"]
        key_rep = orbs["key_orb_rep"]

        # Average D4 orbits in float BEFORE quantization to eliminate mirror drift
        t1_np = self.tier1_w.detach().cpu().numpy()
        t1_q = np.empty((NUM_PLIES, TOTAL_RAW_ENTRIES), dtype=np.int16)
        raw_cnt = np.bincount(raw_rep, minlength=TOTAL_RAW_ENTRIES).astype(np.float64)
        raw_cnt = np.maximum(raw_cnt, 1.0)
        for p in range(NUM_PLIES):
            sums = np.bincount(raw_rep, weights=t1_np[p].astype(np.float64), minlength=TOTAL_RAW_ENTRIES)
            avg = (sums / raw_cnt)[raw_rep]
            t1_q[p] = np.clip(np.rint(avg * P2C_DEFAULT_SCALE), -32767, 32767).astype(np.int16)

        h_np = self.h_emb.detach().cpu().numpy() * EMB_SCALE
        h_q = np.empty((self.num_blocks, KEY_RAW_ENTRIES, self.emb_ch), dtype=np.int8)
        key_cnt = np.bincount(key_rep, minlength=KEY_RAW_ENTRIES).astype(np.float64)
        key_cnt = np.maximum(key_cnt, 1.0)
        for b in range(self.num_blocks):
            for c in range(self.emb_ch):
                sums = np.bincount(key_rep, weights=h_np[b, :, c].astype(np.float64), minlength=KEY_RAW_ENTRIES)
                avg = (sums / key_cnt)[key_rep]
                h_q[b, :, c] = np.clip(np.rint(avg), -SWAR_V40_H_MAX, SWAR_V40_H_MAX).astype(np.int8)

        m = make_empty_swar_v40(emb_ch=self.emb_ch, num_blocks=self.num_blocks, local_relu=self.local_relu)
        m.tier1_w = t1_q
        m.h_emb = h_q
        m.mu_g[:, : self.emb_ch] = np.clip(
            np.rint(self.mu_g.detach().cpu().numpy()), -SWAR_V40_MU_MAX, SWAR_V40_MU_MAX
        ).astype(np.int16)
        m.beta_g[:, : self.n_g_terms] = np.clip(
            np.rint(self.beta_g.detach().cpu().numpy()), -SWAR_V40_BETA_MAX, SWAR_V40_BETA_MAX
        ).astype(np.int16)
        if self.local_relu:
            m.mu_e[:] = np.clip(
                np.rint(self.mu_e.detach().cpu().numpy()), -SWAR_V40_MU_MAX, SWAR_V40_MU_MAX
            ).astype(np.int16)
            m.mu_d[:] = np.clip(
                np.rint(self.mu_d.detach().cpu().numpy()), -SWAR_V40_MU_MAX, SWAR_V40_MU_MAX
            ).astype(np.int16)
            m.beta_e[:] = np.clip(
                np.rint(self.beta_e.detach().cpu().numpy()), -SWAR_V40_BETA_MAX, SWAR_V40_BETA_MAX
            ).astype(np.int16)
            m.beta_d[:] = np.clip(
                np.rint(self.beta_d.detach().cpu().numpy()), -SWAR_V40_BETA_MAX, SWAR_V40_BETA_MAX
            ).astype(np.int16)
        m.b_ply[:] = np.clip(
            np.rint(self.b_ply.detach().cpu().numpy()), -WACA_MAX_RES, WACA_MAX_RES
        ).astype(np.int32)
        symmetrize_swar_v40_model(m, bridge)
        return m


# ============================================================================
# PyTorch Module 2: Control Baseline raw_ntuple64_relabel.p2c
# ============================================================================
class RawRelabelNet(nn.Module):
    def __init__(self, init_tier1_i16: np.ndarray) -> None:
        super().__init__()
        t1_init = torch.from_numpy(init_tier1_i16.astype(np.float32) / P2C_DEFAULT_SCALE)
        self.tier1_w = nn.Parameter(t1_init)

    def forward(self, raw_ptm: torch.Tensor, ply: torch.Tensor, qat: bool = False) -> torch.Tensor:
        ply_idx = ply.unsqueeze(1)
        if qat:
            w_q = ste_round_clamp(self.tier1_w[ply_idx, raw_ptm] * P2C_DEFAULT_SCALE, -32767.0, 32767.0)
            return w_q.sum(dim=1) / P2C_DEFAULT_SCALE
        return self.tier1_w[ply_idx, raw_ptm].sum(dim=1)

    @torch.no_grad()
    def export_quantized(self, bridge: P2C64Bridge) -> RawNTupleModel:
        orbs = bridge.get_orbit_tables()
        raw_rep = orbs["raw_orb_rep"]
        t1_np = self.tier1_w.detach().cpu().numpy()
        t1_q = np.empty((NUM_PLIES, TOTAL_RAW_ENTRIES), dtype=np.int16)
        raw_cnt = np.bincount(raw_rep, minlength=TOTAL_RAW_ENTRIES).astype(np.float64)
        raw_cnt = np.maximum(raw_cnt, 1.0)
        for p in range(NUM_PLIES):
            sums = np.bincount(raw_rep, weights=t1_np[p].astype(np.float64), minlength=TOTAL_RAW_ENTRIES)
            avg = (sums / raw_cnt)[raw_rep]
            t1_q[p] = np.clip(np.rint(avg * P2C_DEFAULT_SCALE), -32767, 32767).astype(np.int16)
        m = RawNTupleModel(w_ply=t1_q, score_scale=P2C_DEFAULT_SCALE, flags=1)
        symmetrize_raw_model(m, orbs)
        return m


@torch.no_grad()
def eval_model_float(
    net: nn.Module,
    bridge: P2C64Bridge,
    val_data: np.ndarray,
    device: torch.device,
    batch_size: int,
    qat: bool,
) -> np.ndarray:
    net.eval()
    n = len(val_data)
    preds = np.empty(n, dtype=np.float64)
    for s in range(0, n, batch_size):
        e = min(n, s + batch_size)
        sl = val_data[s:e]
        ext = bridge.extract_batch_ptm(sl["black"], sl["white"], sl["stm"], enable_aug=False, need_p2c=False, need_raw=True)
        rp_t = torch.from_numpy(ext["raw_ptm"].astype(np.int64)).to(device, non_blocking=True)
        pl_t = torch.from_numpy(sl["ply"].astype(np.int64)).to(device, non_blocking=True)
        out = net(rp_t, pl_t, qat=qat)
        preds[s:e] = out.detach().cpu().numpy().astype(np.float64)
    return preds


def eval_deployed_int(
    bridge: P2C64Bridge,
    model: SwarV40Model | RawNTupleModel,
    val_data: np.ndarray,
    batch_size: int = 32768,
) -> np.ndarray:
    n = len(val_data)
    preds = np.empty(n, dtype=np.float64)
    scale = float(model.score_scale)
    for s in range(0, n, batch_size):
        e = min(n, s + batch_size)
        sl = val_data[s:e]
        ext = bridge.extract_batch_ptm(sl["black"], sl["white"], sl["stm"], enable_aug=False, need_p2c=False, need_raw=True)
        if isinstance(model, SwarV40Model):
            sc = eval_swar_v40_int_batch(model, ext["raw_ptm"], sl["ply"])
        else:
            sc = eval_raw_ntuple64_int_batch(model, ext["raw_ptm"], sl["ply"])
        preds[s:e] = sc.astype(np.float64) / scale
    return preds


def make_optimizer(net: nn.Module, args: argparse.Namespace, lr_scale: float = 1.0) -> torch.optim.Optimizer:
    if isinstance(net, SwarV40Net):
        head_params = [net.mu_g, net.beta_g, net.b_ply]
        if net.local_relu:
            head_params.extend([net.mu_e, net.mu_d, net.beta_e, net.beta_d])
        groups = [
            {"params": [net.tier1_w], "lr": args.t1_lr * lr_scale, "weight_decay": args.t1_wd},
            {"params": [net.h_emb], "lr": args.emb_lr * lr_scale, "weight_decay": args.emb_wd},
            {"params": head_params, "lr": args.head_lr * lr_scale, "weight_decay": 0.0},
        ]
        return torch.optim.AdamW(groups)
    return torch.optim.AdamW(
        [{"params": [net.tier1_w], "lr": args.t1_lr * lr_scale, "weight_decay": args.t1_wd}]
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="P2C-64 v4.0 End-to-End SWAR_V40 / RAW Relabel Trainer")
    ap.add_argument("--mode", choices=("swar_v40", "raw_relabel"), default="swar_v40")
    ap.add_argument("--train", type=Path, required=True)
    ap.add_argument("--val", type=Path, required=True)
    ap.add_argument("--init-raw", type=Path, default=None, help="Initial raw_ntuple64.p2c for Tier-1 warm start")
    ap.add_argument("--relabel-train", type=Path, default=None, help="int16 sidecar for train.bin")
    ap.add_argument("--relabel-val", type=Path, default=None, help="int16 sidecar for val.bin")
    ap.add_argument("--lambda", dest="lam", type=float, default=0.65, help="Blend weight on relabel target")
    ap.add_argument("--out", type=Path, required=True, help="Output .p2c path")
    ap.add_argument("--metrics-out", type=Path, default=None, help="Output metrics JSON/CSV base path")
    ap.add_argument("--emb-ch", type=int, choices=(4, 8), default=8)
    ap.add_argument("--num-blocks", type=int, choices=(2, 30), default=30)
    ap.add_argument("--local-relu", type=int, choices=(0, 1), default=0)
    ap.add_argument("--float-epochs", type=int, default=7)
    ap.add_argument("--qat-epochs", type=int, default=2)
    ap.add_argument("--batch-size", type=int, default=8192)
    ap.add_argument("--t1-lr", type=float, default=2e-3)
    ap.add_argument("--emb-lr", type=float, default=6e-3)
    ap.add_argument("--head-lr", type=float, default=4e-3)
    ap.add_argument("--t1-wd", type=float, default=1e-5)
    ap.add_argument("--emb-wd", type=float, default=1e-5)
    ap.add_argument("--t1-ply-smooth", type=float, default=8e-4)
    ap.add_argument("--blk-smooth", type=float, default=4e-3)
    ap.add_argument("--head-ply-smooth", type=float, default=1.2e-3)
    ap.add_argument("--huber-delta", type=float, default=4.0)
    ap.add_argument("--max-train", type=int, default=0)
    ap.add_argument("--max-val", type=int, default=0)
    ap.add_argument("--seed", type=int, default=20261128)
    ap.add_argument("--device", type=str, default="auto")
    args = ap.parse_args()

    set_seed(args.seed)
    device = torch.device(
        ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    )
    bridge = P2C64Bridge()

    train_data, tr_idx = load_packed_dataset(args.train, args.max_train)
    val_data, va_idx = load_packed_dataset(args.val, args.max_val)
    sc_train = load_relabel_sidecar(args.relabel_train, len(train_data), tr_idx)
    sc_val = load_relabel_sidecar(args.relabel_val, len(val_data), va_idx)

    y_tr_blend, _, tr_rel_mask = build_blended_targets(train_data["score"], sc_train, args.lam)
    y_va_final = val_data["score"].astype(np.float32)
    y_va_blend, y_va_relabel, va_rel_mask = build_blended_targets(val_data["score"], sc_val, args.lam)

    if args.init_raw is not None:
        raw_init = load_p2c(args.init_raw)
        if not isinstance(raw_init, RawNTupleModel):
            raise RuntimeError(f"--init-raw must be a RAW_NTUPLE_64 .p2c file: {args.init_raw}")
        init_t1 = raw_init.w_ply.copy()
    else:
        init_t1 = np.zeros((NUM_PLIES, TOTAL_RAW_ENTRIES), dtype=np.int16)

    if args.mode == "swar_v40":
        net: nn.Module = SwarV40Net(
            init_tier1_i16=init_t1,
            emb_ch=args.emb_ch,
            num_blocks=args.num_blocks,
            local_relu=args.local_relu,
        ).to(device)
    else:
        net = RawRelabelNet(init_tier1_i16=init_t1).to(device)

    print(f"=== P2C-64 v4.0 Trainer (mode={args.mode}, device={device}) ===")
    if args.mode == "swar_v40":
        print(f"Arch    : SWAR_V40 (emb_ch={args.emb_ch}, num_blocks={args.num_blocks}, local_relu={args.local_relu}, Tier-1 UNFROZEN)")
    else:
        print("Arch    : RAW_NTUPLE_64 (relabel control baseline)")
    print(
        f"Data    : train={len(train_data):,} (relabeled={int(tr_rel_mask.sum()):,}), "
        f"val={len(val_data):,} (relabeled={int(va_rel_mask.sum()):,}), lambda={args.lam:.2f}"
    )
    print(
        f"Schedule: {args.float_epochs} Float + {args.qat_epochs} QAT epochs | "
        f"bs={args.batch_size} | lr=(t1={args.t1_lr}, emb={args.emb_lr}, head={args.head_lr})"
    )

    history = []
    best_float_rmse = float("inf")
    best_float_state = None
    best_qat_rmse = float("inf")
    best_qat_state = None

    def run_phase(phase: str, n_epochs: int, qat: bool, lr_mult_start: float, lr_mult_end: float) -> None:
        nonlocal best_float_rmse, best_float_state, best_qat_rmse, best_qat_state
        if n_epochs <= 0:
            return
        opt = make_optimizer(net, args, lr_scale=lr_mult_start)
        base_lrs = [g["lr"] for g in opt.param_groups]
        n_tr = len(train_data)
        steps_per_ep = max(1, (n_tr + args.batch_size - 1) // args.batch_size)
        total_steps = max(1, n_epochs * steps_per_ep)
        gstep = 0

        for ep in range(1, n_epochs + 1):
            t0 = time.perf_counter()
            net.train()
            perm = np.random.permutation(n_tr)
            loss_sum = 0.0
            n_seen = 0

            for s in range(0, n_tr, args.batch_size):
                e = min(n_tr, s + args.batch_size)
                idx = perm[s:e]
                sl = train_data[idx]
                ext = bridge.extract_batch_ptm(
                    sl["black"], sl["white"], sl["stm"],
                    aug_seed=args.seed + ep * 1000003 + s,
                    enable_aug=True, need_p2c=False, need_raw=True,
                )
                rp_t = torch.from_numpy(ext["raw_ptm"].astype(np.int64)).to(device, non_blocking=True)
                pl_t = torch.from_numpy(sl["ply"].astype(np.int64)).to(device, non_blocking=True)
                yt_t = torch.from_numpy(y_tr_blend[idx]).to(device, non_blocking=True)

                # Cosine LR schedule
                frac = gstep / max(1, total_steps - 1)
                ratio = lr_mult_end / max(1e-9, lr_mult_start)
                mult = ratio + 0.5 * (1.0 - ratio) * (1.0 + math.cos(math.pi * frac))
                for pg, blr in zip(opt.param_groups, base_lrs):
                    pg["lr"] = blr * mult

                opt.zero_grad(set_to_none=True)
                pred = net(rp_t, pl_t, qat=qat)
                loss = F.huber_loss(pred, yt_t, delta=args.huber_delta)
                loss.backward()
                opt.step()

                # Step-size-scaled proximal smoothing
                eff_mult = lr_mult_start * mult
                apply_axis0_smoothness_(net.tier1_w, args.t1_ply_smooth * eff_mult)
                if isinstance(net, SwarV40Net):
                    if net.num_blocks == 30:
                        apply_eo_block_smoothness_(net.h_emb, args.blk_smooth * eff_mult)
                    apply_axis0_smoothness_(net.mu_g, args.head_ply_smooth * eff_mult)
                    apply_axis0_smoothness_(net.beta_g, args.head_ply_smooth * eff_mult)
                    apply_axis0_smoothness_(net.b_ply, args.head_ply_smooth * eff_mult)
                    if net.local_relu:
                        apply_axis0_smoothness_(net.mu_e, args.head_ply_smooth * eff_mult)
                        apply_axis0_smoothness_(net.mu_d, args.head_ply_smooth * eff_mult)
                        apply_axis0_smoothness_(net.beta_e, args.head_ply_smooth * eff_mult)
                        apply_axis0_smoothness_(net.beta_d, args.head_ply_smooth * eff_mult)

                bs = e - s
                loss_sum += float(loss.item()) * bs
                n_seen += bs
                gstep += 1

            preds_val = eval_model_float(net, bridge, val_data, device, max(args.batch_size, 16384), qat=qat)
            vm = compute_dual_val_metrics(preds_val, y_va_final, y_va_blend, y_va_relabel, va_rel_mask, val_data["ply"])
            dt = time.perf_counter() - t0
            sel_rmse = vm["blend_rmse_all"]
            row = {
                "phase": phase,
                "epoch": ep,
                "train_loss": loss_sum / max(1, n_seen),
                "elapsed_s": round(dt, 2),
                **vm,
            }
            history.append(row)

            marker = ""
            if not qat:
                if sel_rmse < best_float_rmse:
                    best_float_rmse = sel_rmse
                    best_float_state = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
                    marker = " *"
            else:
                if sel_rmse < best_qat_rmse:
                    best_qat_rmse = sel_rmse
                    best_qat_state = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
                    marker = " *"

            print(
                f"[{phase:<5} ep {ep}/{n_epochs}] loss={row['train_loss']:.4f} | "
                f"val_blend={vm['blend_rmse_all']:.4f} | val_orig(lam=0)={vm['orig_rmse_all']:.4f} "
                f"(0-19:{vm['orig_rmse_0_19']:.3f} 20-39:{vm['orig_rmse_20_39']:.3f} 40-59:{vm['orig_rmse_40_59']:.3f}) | "
                f"val_rel_sub={vm['relabel_sub_rmse']:.4f} | {dt:.1f}s{marker}",
                flush=True,
            )

    # Phase 1: Float training
    run_phase("FLOAT", args.float_epochs, qat=False, lr_mult_start=1.0, lr_mult_end=0.10)

    # Reload Best-Float before QAT + fresh AdamW reset
    if best_float_state is not None:
        net.load_state_dict({k: v.to(device) for k, v in best_float_state.items()})
        print(f"[RELOAD] Loaded Best-Float checkpoint (val_blend_rmse={best_float_rmse:.4f}) before QAT.")

    # Phase 2: QAT fine-tuning (0.12x -> 0.02x)
    run_phase("QAT", args.qat_epochs, qat=True, lr_mult_start=0.12, lr_mult_end=0.02)

    if best_qat_state is not None:
        net.load_state_dict({k: v.to(device) for k, v in best_qat_state.items()})

    # Export quantized .p2c model & verify deployed integer RMSE on both axes
    deployed = net.export_quantized(bridge)
    if isinstance(deployed, SwarV40Model):
        save_swar_v40(args.out, deployed)
    else:
        save_raw_p2c(args.out, deployed)

    preds_int = eval_deployed_int(bridge, deployed, val_data)
    dep_metrics = compute_dual_val_metrics(preds_int, y_va_final, y_va_blend, y_va_relabel, va_rel_mask, val_data["ply"])
    qat_delta = dep_metrics["blend_rmse_all"] - best_float_rmse if math.isfinite(best_float_rmse) else 0.0

    print("\n=== Deployed Quantized Integer Model Summary ===")
    print(f"Saved   : {args.out}")
    print(
        f"Orig Val (lam=0) : RMSE={dep_metrics['orig_rmse_all']:.4f} "
        f"(0-19:{dep_metrics['orig_rmse_0_19']:.4f} | 20-39:{dep_metrics['orig_rmse_20_39']:.4f} | 40-59:{dep_metrics['orig_rmse_40_59']:.4f})"
    )
    print(
        f"Blend Val (lam={args.lam:.2f}): RMSE={dep_metrics['blend_rmse_all']:.4f} | "
        f"Pure Relabel Sub RMSE={dep_metrics['relabel_sub_rmse']:.4f} | "
        f"QAT delta vs Best-Float={qat_delta:+.4f}"
    )

    metrics_base = args.metrics_out if args.metrics_out is not None else args.out.with_name("metrics_v40")
    json_path = metrics_base.with_suffix(".json")
    csv_path = metrics_base.with_suffix(".csv")
    json_path.parent.mkdir(parents=True, exist_ok=True)
    summary = {
        "mode": args.mode,
        "out_p2c": str(args.out),
        "emb_ch": args.emb_ch if args.mode == "swar_v40" else 0,
        "num_blocks": args.num_blocks if args.mode == "swar_v40" else 0,
        "local_relu": args.local_relu if args.mode == "swar_v40" else 0,
        "lambda": args.lam,
        "best_float_blend_rmse": best_float_rmse,
        "best_qat_blend_rmse": best_qat_rmse,
        "qat_delta_vs_float": qat_delta,
        "deployed_metrics": dep_metrics,
        "history": history,
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    if history:
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(history[0].keys()))
            writer.writeheader()
            writer.writerows(history)
    print(f"Metrics : {json_path} & {csv_path}")


if __name__ == "__main__":
    main()
