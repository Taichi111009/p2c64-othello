#!/usr/bin/env python3
"""P2C-64 v1.1 End-to-End PyTorch Training Pipeline (CUDA A100 / CPU).

Trains and exports all 5 models to .p2c v1.1 format:
  Phase 1: RAW_NTUPLE_64 & LINEAR_64 (60-ply linear skeleton + Corner-4 tempo + smoothness reg)
  Phase 2: Zero-Init Residual (frozen quantized LINEAR_64 W & T):
    - CAND_A_SCRELU_K2 : Dual-SCReLU K=2 + QAT (mu_0=[+1.0, +0.35], beta=0)
    - CAND_B_3W3B      : Soft Multilinear Tensor Contraction (iota_3 exact)
                         -> Hard Switch + Empirical Bayes Shrinkage & Amplitude Clip Convex Refit
                         -> Exact EM-LUT Coordinate Descent (NO STE, NO DWN-EFD)
    - CAND_C_HYBRID    : Phase 2-1 Dual-SCReLU K=1 (frozen quantized)
                         -> Phase 2-2 3way-3bit SCONE-D pipeline
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))

from p2c64_bridge import (
    CORNER_CANON_SIZE,
    CORNER_RAW_SIZE,
    EDGE_CANON_SIZE,
    KEY_CANON_ENTRIES,
    NONKEY_RAW_ENTRIES,
    NUM_BLOCKS,
    NUM_PLIES,
    PACKED_SAMPLE_DTYPE,
    PLIES_PER_BLOCK,
    TOTAL_RAW_ENTRIES,
    P2C64Bridge,
)
from p2c64_io import (
    IOTA3,
    P2C_DEFAULT_SCALE,
    SCRELU_BETA_MAX,
    SCRELU_Q,
    SCRELU_SHIFT,
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

EMB_SCALE = 32.0
SCRELU_QF = SCRELU_Q / EMB_SCALE  # 127 / 32 = 3.96875
EDGE_OFF = CORNER_CANON_SIZE      # 9,842
EDGE_HALF = (3**10 - 1) // 2      # 29,524


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_packed_dataset(path: Path) -> np.ndarray:
    arr = np.fromfile(path, dtype=PACKED_SAMPLE_DTYPE)
    if len(arr) == 0:
        raise RuntimeError(f"Dataset file is empty: {path}")
    return arr


def ste_round_clamp(x: torch.Tensor, lo: float, hi: float) -> torch.Tensor:
    y = torch.clamp(torch.round(x), lo, hi)
    return x + (y - x).detach()


def ste_sym_rshift_int(x: torch.Tensor, shift: int = SCRELU_SHIFT) -> torch.Tensor:
    """Differentiable STE version of round-toward-zero right shift."""
    denom = float(1 << shift)
    y = torch.trunc(x / denom)
    return (x / denom) + (y - x / denom).detach()


# ============================================================================
# Macro-LUT Orbit Precomputation (exact match with C++ symmetrize_macro_luts)
# ============================================================================
def build_macro_orbit_map(swap_mode: str) -> Tuple[np.ndarray, np.ndarray, int]:
    """Returns (orb_id[512] in [0..n_orb-1], orb_sgn[512] in {-1, 0, +1}, n_orb)."""
    orb_id = np.full(512, -1, dtype=np.int32)
    orb_sgn = np.zeros(512, dtype=np.int8)
    cur_orb = 0
    for k in range(512):
        if orb_id[k] != -1:
            continue
        vis = {k: 1}
        orb = [(k, 1)]
        force_zero = False
        qi = 0
        while qi < len(orb):
            ck, sg = orb[qi]
            qi += 1
            z0, z1, z2 = (ck >> 6) & 7, (ck >> 3) & 7, ck & 7
            ik = (int(IOTA3[z0]) << 6) | (int(IOTA3[z1]) << 3) | int(IOTA3[z2])
            if ik not in vis:
                vis[ik] = -sg
                orb.append((ik, -sg))
            elif vis[ik] != -sg:
                force_zero = True

            if swap_mode == "cw":
                mk = (z0 << 6) | (z2 << 3) | z1
            elif swap_mode == "ea":
                mk = (z1 << 6) | (z0 << 3) | z2
            else:
                mk = (z2 << 6) | (z1 << 3) | z0
            if mk not in vis:
                vis[mk] = sg
                orb.append((mk, sg))
            elif vis[mk] != sg:
                force_zero = True

        for idx, sg in orb:
            orb_id[idx] = cur_orb
            orb_sgn[idx] = 0 if force_zero else sg
        cur_orb += 1
    return orb_id, orb_sgn, cur_orb


# ============================================================================
# Deployed Quantized Model Evaluation (RMSE / MAE Overall + Ply Buckets)
# ============================================================================
def evaluate_deployed_p2c(
    bridge: P2C64Bridge,
    model: RawNTupleModel | P2C64Model,
    val_data: np.ndarray,
    batch_size: int = 32768,
) -> Dict[str, float]:
    n = len(val_data)
    preds = np.empty(n, dtype=np.float64)
    y_true = val_data["score"].astype(np.float64)
    plies = val_data["ply"]

    for s in range(0, n, batch_size):
        e = min(n, s + batch_size)
        sl = val_data[s:e]
        if isinstance(model, RawNTupleModel):
            ext = bridge.extract_batch(
                sl["black"], sl["white"], sl["stm"],
                enable_aug=False, need_raw_idx=True, need_p2c=False, need_raw_flat=True,
            )
            sc = eval_raw_ntuple64_int_batch(model, ext["raw_flat"], ext["raw_idx"], sl["stm"], sl["ply"])
        else:
            ext = bridge.extract_batch(
                sl["black"], sl["white"], sl["stm"],
                enable_aug=False, need_raw_idx=False, need_p2c=True, need_raw_flat=False,
            )
            sc = eval_p2c64_int_batch(model, ext["key_cx"], ext["key_flip"], ext["nonkey_flat"], sl["stm"], sl["ply"])
        preds[s:e] = sc.astype(np.float64) / float(model.score_scale)

    err = preds - y_true
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


# ============================================================================
# Decoupled Adjacent-Ply Smoothness Step (Unconditionally Stable Laplacian)
# ============================================================================
@torch.no_grad()
def apply_ply_smoothness_(w: torch.Tensor, alpha: float) -> None:
    """Explicit proximal diffusion for lambda_sm * sum_{t=1}^{59} ||W_t - W_{t-1}||^2."""
    if alpha <= 0.0 or w.shape[0] < 2:
        return
    a = min(0.20, float(alpha))
    w_prev = w.clone()
    w[0].add_(w_prev[1] - w_prev[0], alpha=a)
    w[-1].add_(w_prev[-2] - w_prev[-1], alpha=a)
    if w.shape[0] > 2:
        w[1:-1].add_(w_prev[:-2] + w_prev[2:] - 2.0 * w_prev[1:-1], alpha=a)


# ============================================================================
# Phase 1A: RAW_NTUPLE_64 Baseline Trainer
# ============================================================================
def train_raw_ntuple64(
    bridge: P2C64Bridge,
    orbits: Dict[str, np.ndarray],
    train_data: np.ndarray,
    val_data: np.ndarray,
    args: argparse.Namespace,
    device: torch.device,
    out_p2c: Path,
    ckpt_path: Path,
) -> Tuple[RawNTupleModel, Dict[str, float]]:
    print("\n=== [Phase 1A] Training RAW_NTUPLE_64 Baseline ===")
    raw_rep = torch.from_numpy(orbits["raw_orb_rep"]).long().to(device)
    raw_sgn = torch.from_numpy(orbits["raw_orb_sgn"].astype(np.float32)).to(device)
    ct_rep = torch.from_numpy(orbits["ct_raw_rep"]).long().to(device)

    W = nn.Parameter(torch.zeros((NUM_PLIES, TOTAL_RAW_ENTRIES), dtype=torch.float32, device=device))
    T = nn.Parameter(torch.zeros((NUM_PLIES, CORNER_RAW_SIZE), dtype=torch.float32, device=device))
    opt = torch.optim.AdamW([W, T], lr=args.p1_lr, weight_decay=0.0)

    start_epoch = 0
    if args.resume and ckpt_path.exists():
        ck = torch.load(ckpt_path, map_location=device, weights_only=True)
        W.data.copy_(ck["W"])
        T.data.copy_(ck["T"])
        opt.load_state_dict(ck["opt"])
        start_epoch = int(ck["epoch"]) + 1
        print(f"  Resumed RAW_NTUPLE_64 from epoch {start_epoch}")

    n = len(train_data)
    rng = np.random.default_rng(args.seed + 101)

    for epoch in range(start_epoch, args.p1_epochs):
        t0 = time.perf_counter()
        perm = rng.permutation(n)
        lr_now = args.p1_lr * 0.5 * (1.0 + math.cos(math.pi * epoch / max(1, args.p1_epochs)))
        lr_now = max(lr_now, args.p1_lr * 0.05)
        for pg in opt.param_groups:
            pg["lr"] = lr_now

        total_loss = 0.0
        steps = 0
        for s in range(0, n, args.batch_size):
            idx_b = perm[s : min(n, s + args.batch_size)]
            sl = train_data[idx_b]
            ext = bridge.extract_batch(
                sl["black"], sl["white"], sl["stm"],
                aug_seed=int(rng.integers(0, 2**62)), enable_aug=True,
                need_raw_idx=True, need_p2c=False, need_raw_flat=True,
            )
            rf = torch.from_numpy(ext["raw_flat"]).long().to(device, non_blocking=True)
            c4 = torch.from_numpy(ext["raw_idx"][:, :4].astype(np.int64)).to(device, non_blocking=True)
            ply_t = torch.from_numpy(sl["ply"].astype(np.int64)).to(device, non_blocking=True)
            stm_t = torch.from_numpy(ext["stm"].astype(np.float32)).to(device, non_blocking=True)
            y_t = torch.from_numpy(sl["score"].astype(np.float32)).to(device, non_blocking=True)

            rf_rep = raw_rep[rf]
            rf_sgn = raw_sgn[rf]
            sum_A = (W[ply_t[:, None], rf_rep] * rf_sgn).sum(dim=1)
            sum_T = T[ply_t[:, None], ct_rep[c4]].sum(dim=1)
            pred = stm_t * sum_A + sum_T

            loss = F.smooth_l1_loss(pred, y_t, beta=args.huber_beta)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            apply_ply_smoothness_(W.data, args.smooth_lambda * lr_now * 50.0)
            apply_ply_smoothness_(T.data, args.smooth_lambda * lr_now * 50.0)
            total_loss += float(loss.item())
            steps += 1

        torch.save({"epoch": epoch, "W": W.detach().cpu(), "T": T.detach().cpu(), "opt": opt.state_dict()}, ckpt_path)
        dt = time.perf_counter() - t0
        print(f"  [RAW] Epoch {epoch+1}/{args.p1_epochs} | lr={lr_now:.4f} | huber={total_loss/max(1,steps):.4f} | {dt:.1f}s")

    # Bake quantized & symmetrized .p2c
    w_q = np.clip(np.round(W.detach().cpu().numpy() * args.score_scale), -32767, 32767).astype(np.int16)
    t_q = np.clip(np.round(T.detach().cpu().numpy() * args.score_scale), -127, 127).astype(np.int8)
    raw_model = symmetrize_raw_model(RawNTupleModel(w_ply=w_q, corner_tempo=t_q, score_scale=args.score_scale), orbits)
    save_raw_p2c(out_p2c, raw_model)
    metrics = evaluate_deployed_p2c(bridge, raw_model, val_data, args.val_batch_size)
    print(f"  [RAW] Saved {out_p2c} | Val RMSE={metrics['rmse_all']:.4f} MAE={metrics['mae_all']:.4f}")
    return raw_model, metrics


# ============================================================================
# Phase 1B: LINEAR_64 Shared Skeleton Trainer
# ============================================================================
def train_linear64(
    bridge: P2C64Bridge,
    orbits: Dict[str, np.ndarray],
    train_data: np.ndarray,
    val_data: np.ndarray,
    args: argparse.Namespace,
    device: torch.device,
    out_p2c: Path,
    ckpt_path: Path,
) -> Tuple[P2C64Model, Dict[str, float]]:
    print("\n=== [Phase 1B] Training LINEAR_64 Shared Skeleton ===")
    k_rep = torch.from_numpy(orbits["key_orb_rep"]).long().to(device)
    k_sgn = torch.from_numpy(orbits["key_orb_sgn"].astype(np.float32)).to(device)
    nk_rep = torch.from_numpy(orbits["nk_orb_rep"]).long().to(device)
    nk_sgn = torch.from_numpy(orbits["nk_orb_sgn"].astype(np.float32)).to(device)
    ct_rep = torch.from_numpy(orbits["ct_canon_rep"]).long().to(device)

    W_key = nn.Parameter(torch.zeros((NUM_PLIES, KEY_CANON_ENTRIES), dtype=torch.float32, device=device))
    W_nk = nn.Parameter(torch.zeros((NUM_PLIES, NONKEY_RAW_ENTRIES), dtype=torch.float32, device=device))
    T_c4 = nn.Parameter(torch.zeros((NUM_BLOCKS, CORNER_CANON_SIZE), dtype=torch.float32, device=device))
    opt = torch.optim.AdamW([W_key, W_nk, T_c4], lr=args.p1_lr, weight_decay=0.0)

    start_epoch = 0
    if args.resume and ckpt_path.exists():
        ck = torch.load(ckpt_path, map_location=device, weights_only=True)
        W_key.data.copy_(ck["W_key"])
        W_nk.data.copy_(ck["W_nk"])
        T_c4.data.copy_(ck["T_c4"])
        opt.load_state_dict(ck["opt"])
        start_epoch = int(ck["epoch"]) + 1
        print(f"  Resumed LINEAR_64 from epoch {start_epoch}")

    n = len(train_data)
    rng = np.random.default_rng(args.seed + 202)

    for epoch in range(start_epoch, args.p1_epochs):
        t0 = time.perf_counter()
        perm = rng.permutation(n)
        lr_now = args.p1_lr * 0.5 * (1.0 + math.cos(math.pi * epoch / max(1, args.p1_epochs)))
        lr_now = max(lr_now, args.p1_lr * 0.05)
        for pg in opt.param_groups:
            pg["lr"] = lr_now

        total_loss = 0.0
        steps = 0
        for s in range(0, n, args.batch_size):
            idx_b = perm[s : min(n, s + args.batch_size)]
            sl = train_data[idx_b]
            ext = bridge.extract_batch(
                sl["black"], sl["white"], sl["stm"],
                aug_seed=int(rng.integers(0, 2**62)), enable_aug=True,
                need_raw_idx=False, need_p2c=True, need_raw_flat=False,
            )
            kcx = torch.from_numpy(ext["key_cx"]).long().to(device, non_blocking=True)
            kfl = torch.from_numpy(ext["key_flip"].astype(np.float32)).to(device, non_blocking=True)
            nkf = torch.from_numpy(ext["nonkey_flat"]).long().to(device, non_blocking=True)
            ply_t = torch.from_numpy(sl["ply"].astype(np.int64)).to(device, non_blocking=True)
            blk_t = ply_t // PLIES_PER_BLOCK
            stm_t = torch.from_numpy(ext["stm"].astype(np.float32)).to(device, non_blocking=True)
            y_t = torch.from_numpy(sl["score"].astype(np.float32)).to(device, non_blocking=True)

            # Orbit-tied lookup with exact sign
            ksgn = (1.0 - 2.0 * kfl) * k_sgn[kcx]
            sum_A = (W_key[ply_t[:, None], k_rep[kcx]] * ksgn).sum(dim=1)
            sum_A = sum_A + (W_nk[ply_t[:, None], nk_rep[nkf]] * nk_sgn[nkf]).sum(dim=1)
            sum_T = T_c4[blk_t[:, None], ct_rep[kcx[:, :4]]].sum(dim=1)
            pred = stm_t * sum_A + sum_T

            loss = F.smooth_l1_loss(pred, y_t, beta=args.huber_beta)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            apply_ply_smoothness_(W_key.data, args.smooth_lambda * lr_now * 50.0)
            apply_ply_smoothness_(W_nk.data, args.smooth_lambda * lr_now * 50.0)
            apply_ply_smoothness_(T_c4.data, args.smooth_lambda * lr_now * 20.0)
            total_loss += float(loss.item())
            steps += 1

        torch.save(
            {"epoch": epoch, "W_key": W_key.detach().cpu(), "W_nk": W_nk.detach().cpu(),
             "T_c4": T_c4.detach().cpu(), "opt": opt.state_dict()},
            ckpt_path,
        )
        dt = time.perf_counter() - t0
        print(f"  [LIN] Epoch {epoch+1}/{args.p1_epochs} | lr={lr_now:.4f} | huber={total_loss/max(1,steps):.4f} | {dt:.1f}s")

    # Pack into P2C64Model v1.1 Asymmetric Layout
    model = make_empty_p2c64(EvalMode.LINEAR_64, score_scale=args.score_scale)
    wk_np = np.clip(np.round(W_key.detach().cpu().numpy() * args.score_scale), -32767, 32767).astype(np.int16)
    # Reshape [60, 45929] -> [10, 6, 45929] -> transpose [10, 45929, 6]
    model.key_blocks["w"] = wk_np.reshape(NUM_BLOCKS, PLIES_PER_BLOCK, KEY_CANON_ENTRIES).transpose(0, 2, 1)
    model.nonkey_w = np.clip(np.round(W_nk.detach().cpu().numpy() * args.score_scale), -32767, 32767).astype(np.int16)
    model.corner_tempo = np.clip(np.round(T_c4.detach().cpu().numpy() * args.score_scale), -127, 127).astype(np.int8)
    model = symmetrize_p2c64_model(model, orbits)
    save_p2c64(out_p2c, model)
    metrics = evaluate_deployed_p2c(bridge, model, val_data, args.val_batch_size)
    print(f"  [LIN] Saved {out_p2c} | Val RMSE={metrics['rmse_all']:.4f} MAE={metrics['mae_all']:.4f}")
    return model, metrics


# ============================================================================
# Frozen Quantized LINEAR_64 Evaluator on GPU/CPU (Exact Deployed Baseline)
# ============================================================================
class FrozenLinearEvaluator:
    def __init__(self, lin_model: P2C64Model, device: torch.device) -> None:
        self.scale = float(lin_model.score_scale)
        # [10, 45929, 6] -> [60, 45929] float32 exact integer values
        wk = lin_model.key_blocks["w"].transpose(0, 2, 1).reshape(NUM_PLIES, KEY_CANON_ENTRIES).astype(np.float32)
        self.W_key = torch.from_numpy(wk).to(device)
        self.W_nk = torch.from_numpy(lin_model.nonkey_w.astype(np.float32)).to(device)
        self.T_c4 = torch.from_numpy(lin_model.corner_tempo.astype(np.float32)).to(device)

    @torch.no_grad()
    def eval_disc(
        self,
        kcx: torch.Tensor,
        kfl: torch.Tensor,
        nkf: torch.Tensor,
        ply_t: torch.Tensor,
        stm_t: torch.Tensor,
    ) -> torch.Tensor:
        blk_t = ply_t // PLIES_PER_BLOCK
        sgn = 1.0 - 2.0 * kfl
        sum_A = (self.W_key[ply_t[:, None], kcx] * sgn).sum(dim=1)
        sum_A = sum_A + self.W_nk[ply_t[:, None], nkf].sum(dim=1)
        sum_T = self.T_c4[blk_t[:, None], kcx[:, :4]].sum(dim=1)
        return (stm_t * sum_A + sum_T) / self.scale


# ============================================================================
# Dual-SCReLU Residual Module (K=2 for CAND_A, K=1 for CAND_C Phase 2-1)
# ============================================================================
class DualSCReLUResidual(nn.Module):
    def __init__(
        self,
        k_channels: int,
        orbits: Dict[str, np.ndarray],
        mu_init: Tuple[float, float],
        score_scale: float,
        device: torch.device,
    ) -> None:
        super().__init__()
        assert k_channels in (1, 2)
        self.K = k_channels
        self.score_scale = float(score_scale)
        self.beta_scale = self.score_scale * float(1 << SCRELU_SHIFT) / (EMB_SCALE * EMB_SCALE)
        self.register_buffer("k_rep", torch.from_numpy(orbits["key_orb_rep"]).long().to(device))
        self.register_buffer("k_sgn", torch.from_numpy(orbits["key_orb_sgn"].astype(np.float32)).to(device))

        self.o_emb = nn.Parameter(torch.randn((NUM_BLOCKS, KEY_CANON_ENTRIES, k_channels), device=device) * 0.10)
        self.e_emb = nn.Parameter(torch.randn((NUM_BLOCKS, KEY_CANON_ENTRIES, k_channels), device=device) * 0.10)

        mu_t = torch.zeros((NUM_BLOCKS, 3, k_channels), dtype=torch.float32, device=device)
        mu_t[:, :, 0] = mu_init[0]
        if k_channels == 2:
            mu_t[:, :, 1] = mu_init[1]
        self.mu = nn.Parameter(mu_t)
        # Zero-Init Residual: beta starts strictly at 0
        self.beta_sq = nn.Parameter(torch.zeros((NUM_BLOCKS, 3, k_channels), dtype=torch.float32, device=device))
        self.beta_pr = nn.Parameter(torch.zeros((NUM_BLOCKS, 3), dtype=torch.float32, device=device))

    def forward(
        self, kcx: torch.Tensor, kfl: torch.Tensor, blk_t: torch.Tensor, qat: bool = False
    ) -> torch.Tensor:
        rep = self.k_rep[kcx]                       # [B, 14]
        sgn = (1.0 - 2.0 * kfl) * self.k_sgn[kcx]   # [B, 14]

        o_raw = self.o_emb[blk_t[:, None], rep]     # [B, 14, K]
        e_raw = self.e_emb[blk_t[:, None], rep]     # [B, 14, K]
        mu_b = self.mu[blk_t]                       # [B, 3, K]
        bsq_b = self.beta_sq[blk_t]                 # [B, 3, K]
        bpr_b = self.beta_pr[blk_t]                 # [B, 3]

        if qat:
            o_int = ste_round_clamp(o_raw * EMB_SCALE, -127.0, 127.0)
            e_int = ste_round_clamp(e_raw * EMB_SCALE, -127.0, 127.0)
            mu_int = ste_round_clamp(mu_b * EMB_SCALE, -1024.0, 1024.0)
            bsq_int = ste_round_clamp(bsq_b * self.beta_scale, -float(SCRELU_BETA_MAX), float(SCRELU_BETA_MAX))
            bpr_int = ste_round_clamp(bpr_b * self.beta_scale, -float(SCRELU_BETA_MAX), float(SCRELU_BETA_MAX))

            ko = o_int * sgn.unsqueeze(-1)
            ke = e_int
            q_max = float(SCRELU_Q)
        else:
            ko = o_raw * sgn.unsqueeze(-1)
            ke = e_raw
            mu_int = mu_b
            bsq_int = bsq_b
            bpr_int = bpr_b
            q_max = SCRELU_QF

        # 4 Edge strips (r=0..3)
        s_sq_e = torch.zeros((kcx.shape[0], self.K), dtype=ko.dtype, device=ko.device)
        s_pr_e = torch.zeros(kcx.shape[0], dtype=ko.dtype, device=ko.device)
        for r in range(4):
            c1 = (r + 1) & 3
            o_r = ko[:, 4 + r] + ko[:, 8 + r] + ko[:, r] + ko[:, c1]
            e_r = ke[:, 4 + r] + ke[:, 8 + r] + ke[:, r] + ke[:, c1] + mu_int[:, 0]
            b_r = torch.clamp(e_r + o_r, 0.0, q_max)
            w_r = torch.clamp(e_r - o_r, 0.0, q_max)
            s_sq_e = s_sq_e + (b_r - w_r) * (b_r + w_r)
            if self.K == 2:
                s_pr_e = s_pr_e + (b_r[:, 0] * b_r[:, 1] - w_r[:, 0] * w_r[:, 1])

        # 2 Diag strips (d=0..1)
        s_sq_d = torch.zeros((kcx.shape[0], self.K), dtype=ko.dtype, device=ko.device)
        s_pr_d = torch.zeros(kcx.shape[0], dtype=ko.dtype, device=ko.device)
        for d in range(2):
            o_d = ko[:, 12 + d] + ko[:, d] + ko[:, d + 2]
            e_d = ke[:, 12 + d] + ke[:, d] + ke[:, d + 2] + mu_int[:, 1]
            b_d = torch.clamp(e_d + o_d, 0.0, q_max)
            w_d = torch.clamp(e_d - o_d, 0.0, q_max)
            s_sq_d = s_sq_d + (b_d - w_d) * (b_d + w_d)
            if self.K == 2:
                s_pr_d = s_pr_d + (b_d[:, 0] * b_d[:, 1] - w_d[:, 0] * w_d[:, 1])

        # 1 Global sum
        g_o = ko.sum(dim=1)
        g_e = ke.sum(dim=1) + mu_int[:, 2]
        b_g = torch.clamp(g_e + g_o, 0.0, q_max)
        w_g = torch.clamp(g_e - g_o, 0.0, q_max)
        s_sq_g = (b_g - w_g) * (b_g + w_g)

        res = (bsq_int[:, 0] * s_sq_e + bsq_int[:, 1] * s_sq_d + bsq_int[:, 2] * s_sq_g).sum(dim=1)
        if self.K == 2:
            s_pr_g = b_g[:, 0] * b_g[:, 1] - w_g[:, 0] * w_g[:, 1]
            res = res + bpr_int[:, 0] * s_pr_e + bpr_int[:, 1] * s_pr_d + bpr_int[:, 2] * s_pr_g

        if qat:
            return ste_sym_rshift_int(res, SCRELU_SHIFT) / self.score_scale
        return res

    @torch.no_grad()
    def export_quantized_arrays(self) -> Dict[str, np.ndarray]:
        o_q = np.clip(np.round(self.o_emb.detach().cpu().numpy() * EMB_SCALE), -127, 127).astype(np.int8)
        e_q = np.clip(np.round(self.e_emb.detach().cpu().numpy() * EMB_SCALE), -127, 127).astype(np.int8)
        mu_q = np.clip(np.round(self.mu.detach().cpu().numpy() * EMB_SCALE), -1024, 1024).astype(np.int32)
        bsq_q = np.clip(
            np.round(self.beta_sq.detach().cpu().numpy() * self.beta_scale), -SCRELU_BETA_MAX, SCRELU_BETA_MAX
        ).astype(np.int32)
        bpr_q = np.clip(
            np.round(self.beta_pr.detach().cpu().numpy() * self.beta_scale), -SCRELU_BETA_MAX, SCRELU_BETA_MAX
        ).astype(np.int32)
        return {"o": o_q, "e": e_q, "mu": mu_q, "beta_sq": bsq_q, "beta_pr": bpr_q}


def train_screlu_residual(
    bridge: P2C64Bridge,
    orbits: Dict[str, np.ndarray],
    frozen_lin: FrozenLinearEvaluator,
    train_data: np.ndarray,
    k_channels: int,
    float_epochs: int,
    qat_epochs: int,
    lr: float,
    mu_init: Tuple[float, float],
    args: argparse.Namespace,
    device: torch.device,
    tag: str,
) -> DualSCReLUResidual:
    mod = DualSCReLUResidual(k_channels, orbits, mu_init, args.score_scale, device)
    opt = torch.optim.AdamW(mod.parameters(), lr=lr, weight_decay=1e-4)
    n = len(train_data)
    rng = np.random.default_rng(args.seed + 300 + k_channels * 17)
    total_ep = float_epochs + qat_epochs

    for ep in range(total_ep):
        is_qat = ep >= float_epochs
        t0 = time.perf_counter()
        perm = rng.permutation(n)
        lr_now = (lr * 0.25 if is_qat else lr) * 0.5 * (
            1.0 + math.cos(math.pi * ep / max(1, total_ep))
        )
        lr_now = max(lr_now, lr * 0.03)
        for pg in opt.param_groups:
            pg["lr"] = lr_now

        loss_sum = 0.0
        steps = 0
        for s in range(0, n, args.batch_size):
            idx_b = perm[s : min(n, s + args.batch_size)]
            sl = train_data[idx_b]
            ext = bridge.extract_batch(
                sl["black"], sl["white"], sl["stm"],
                aug_seed=int(rng.integers(0, 2**62)), enable_aug=True,
                need_raw_idx=False, need_p2c=True, need_raw_flat=False,
            )
            kcx = torch.from_numpy(ext["key_cx"]).long().to(device, non_blocking=True)
            kfl = torch.from_numpy(ext["key_flip"].astype(np.float32)).to(device, non_blocking=True)
            nkf = torch.from_numpy(ext["nonkey_flat"]).long().to(device, non_blocking=True)
            ply_t = torch.from_numpy(sl["ply"].astype(np.int64)).to(device, non_blocking=True)
            blk_t = ply_t // PLIES_PER_BLOCK
            stm_t = torch.from_numpy(ext["stm"].astype(np.float32)).to(device, non_blocking=True)
            y_t = torch.from_numpy(sl["score"].astype(np.float32)).to(device, non_blocking=True)

            with torch.no_grad():
                y_lin = frozen_lin.eval_disc(kcx, kfl, nkf, ply_t, stm_t)
            a_res = mod(kcx, kfl, blk_t, qat=is_qat)
            pred = y_lin + stm_t * a_res

            loss = F.smooth_l1_loss(pred, y_t, beta=args.huber_beta)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            loss_sum += float(loss.item())
            steps += 1

        stage = "QAT" if is_qat else "FLT"
        print(f"  [{tag}/{stage}] Epoch {ep+1}/{total_ep} | lr={lr_now:.5f} | huber={loss_sum/max(1,steps):.4f} | {time.perf_counter()-t0:.1f}s")
    return mod


# ============================================================================
# Phase 2B / 2C-2: SCONE-D 3way-3bit Pipeline (NO STE, NO DWN-EFD!)
#   Stage 1: Soft Multilinear Tensor Contraction with exact iota_3 & anti-mirror
#   Stage 2: Hard Switch + Empirical Bayes Shrinkage & Amplitude Clipping Convex Refit
#   Stage 3: Exact EM-LUT Coordinate Descent
# ============================================================================
class SoftSconeD3W3B(nn.Module):
    def __init__(self, orbits: Dict[str, np.ndarray], device: torch.device) -> None:
        super().__init__()
        self.register_buffer("iota3", torch.from_numpy(IOTA3).long().to(device))
        self.register_buffer("k_rep", torch.from_numpy(orbits["key_orb_rep"]).long().to(device))
        self.register_buffer("k_sgn", torch.from_numpy(orbits["key_orb_sgn"].astype(np.int8)).to(device))
        self.register_buffer("k_am", torch.from_numpy(orbits["key_anti_mirror"].astype(np.bool_)).to(device))

        e_mcx = orbits["edge_mirror_cx"]
        e_mfl = orbits["edge_mirror_flip"]
        e_rep = np.minimum(np.arange(EDGE_CANON_SIZE, dtype=np.int32), e_mcx)
        e_is_swapped = (np.arange(EDGE_CANON_SIZE, dtype=np.int32) > e_mcx)
        e_is_self = (np.arange(EDGE_CANON_SIZE, dtype=np.int32) == e_mcx)
        self.register_buffer("e_rep", torch.from_numpy(e_rep).long().to(device))
        self.register_buffer("e_mfl", torch.from_numpy(e_mfl.astype(np.bool_)).to(device))
        self.register_buffer("e_is_swapped", torch.from_numpy(e_is_swapped).to(device))
        self.register_buffer("e_is_self", torch.from_numpy(e_is_self).to(device))

        # Soft code logits (small random init for symmetry breaking)
        self.logits_single = nn.Parameter(torch.randn((NUM_BLOCKS, KEY_CANON_ENTRIES, 8), device=device) * 0.25)
        self.logits_edge_f = nn.Parameter(torch.randn((NUM_BLOCKS, EDGE_CANON_SIZE, 8), device=device) * 0.25)
        self.logits_edge_r = nn.Parameter(torch.randn((NUM_BLOCKS, EDGE_CANON_SIZE, 8), device=device) * 0.25)

        # Zero-Init Residual Macro-LUTs [10, 8, 8, 8]
        self.T_cw_raw = nn.Parameter(torch.zeros((NUM_BLOCKS, 8, 8, 8), dtype=torch.float32, device=device))
        self.T_ea_raw = nn.Parameter(torch.zeros((NUM_BLOCKS, 8, 8, 8), dtype=torch.float32, device=device))
        self.T_cd_raw = nn.Parameter(torch.zeros((NUM_BLOCKS, 8, 8, 8), dtype=torch.float32, device=device))

        # Fixed-point mask for anti-mirror/all-empty states: allow only {6, 7}
        sd_mask = torch.tensor([-1e9, -1e9, -1e9, -1e9, -1e9, -1e9, 0.0, 0.0], dtype=torch.float32, device=device)
        self.register_buffer("sd_mask", sd_mask)

    def _sym_macro(self, T: torch.Tensor, swap_axes: Tuple[int, int]) -> torch.Tensor:
        # 1) D4 mirror symmetry
        Tm = 0.5 * (T + T.transpose(swap_axes[0], swap_axes[1]))
        # 2) Z2 iota_3^3 antisymmetry
        ii = self.iota3
        Tinv = Tm[:, ii][:, :, ii][:, :, :, ii]
        return 0.5 * (Tm - Tinv)

    def _single_probs(self, blk_t: torch.Tensor, cx: torch.Tensor, flip_b: torch.Tensor, tau: float) -> torch.Tensor:
        rep = self.k_rep[cx]
        am = self.k_am[cx]
        sgn_neg = self.k_sgn[cx] < 0
        lg = self.logits_single[blk_t, rep]
        lg = torch.where(am.unsqueeze(-1), lg + self.sd_mask, lg)
        p = F.softmax(lg / tau, dim=-1)
        eff_flip = flip_b ^ sgn_neg
        p_inv = p[:, self.iota3]
        return torch.where(eff_flip.unsqueeze(-1), p_inv, p)

    def _edge_probs(self, blk_t: torch.Tensor, cx_global: torch.Tensor, flip_b: torch.Tensor, tau: float) -> Tuple[torch.Tensor, torch.Tensor]:
        cx = cx_global - EDGE_OFF
        rep = self.e_rep[cx]
        mfl = self.e_mfl[cx]
        swapped = self.e_is_swapped[cx]
        is_self = self.e_is_self[cx]
        is_half = (cx == EDGE_HALF)

        lg_f0 = self.logits_edge_f[blk_t, rep]
        lg_r0 = self.logits_edge_r[blk_t, rep]
        lg_f0 = torch.where(is_half.unsqueeze(-1), lg_f0 + self.sd_mask, lg_f0)

        pf0 = F.softmax(lg_f0 / tau, dim=-1)
        pr0 = F.softmax(lg_r0 / tau, dim=-1)
        pf0_inv = pf0[:, self.iota3]
        pr0_inv = pr0[:, self.iota3]

        # Build canonical (pf, pr) honoring all 3 Edge+2X orbit cases
        # Default (cx < cy): pf = pf0, pr = pr0
        pf = pf0
        pr = pr0
        # Case cx > cy (swapped): pf = IOTA3(pr0) if mfl else pr0; pr = IOTA3(pf0) if mfl else pf0
        pf_sw = torch.where(mfl.unsqueeze(-1), pr0_inv, pr0)
        pr_sw = torch.where(mfl.unsqueeze(-1), pf0_inv, pf0)
        pf = torch.where(swapped.unsqueeze(-1), pf_sw, pf)
        pr = torch.where(swapped.unsqueeze(-1), pr_sw, pr)
        # Case cx == cy (self-mirror): pf = pf0, pr = IOTA3(pf0) if mfl else pf0
        pr_self = torch.where(mfl.unsqueeze(-1), pf0_inv, pf0)
        pr = torch.where(is_self.unsqueeze(-1), pr_self, pr)
        # Case cx == half: pf = pr = pf0 (already restricted to {6,7})
        pr = torch.where(is_half.unsqueeze(-1), pf0, pr)

        # Runtime board color flip
        pf_out = torch.where(flip_b.unsqueeze(-1), pf[:, self.iota3], pf)
        pr_out = torch.where(flip_b.unsqueeze(-1), pr[:, self.iota3], pr)
        return pf_out, pr_out

    def forward(self, kcx: torch.Tensor, kfl: torch.Tensor, blk_t: torch.Tensor, tau: float) -> torch.Tensor:
        fl_b = kfl > 0.5
        T_cw = self._sym_macro(self.T_cw_raw, (2, 3))
        T_ea = self._sym_macro(self.T_ea_raw, (1, 2))
        T_cd = self._sym_macro(self.T_cd_raw, (1, 3))

        pf = [None] * 14
        pr = [None] * 4
        for p in (0, 1, 2, 3, 8, 9, 10, 11, 12, 13):
            pf[p] = self._single_probs(blk_t, kcx[:, p], fl_b[:, p], tau)
        for i in range(4):
            f_i, r_i = self._edge_probs(blk_t, kcx[:, 4 + i], fl_b[:, 4 + i], tau)
            pf[4 + i] = f_i
            pr[i] = r_i

        tcw_b = T_cw[blk_t]
        tea_b = T_ea[blk_t]
        tcd_b = T_cd[blk_t]

        out = torch.zeros(kcx.shape[0], dtype=torch.float32, device=kcx.device)
        for i in range(4):
            out = out + torch.einsum("bi,bj,bk,bijk->b", pf[i], pf[4 + i], pr[(i + 3) & 3], tcw_b)
            out = out + torch.einsum("bi,bj,bk,bijk->b", pf[4 + i], pr[i], pf[8 + i], tea_b)
        out = out + torch.einsum("bi,bj,bk,bijk->b", pf[0], pf[12], pf[2], tcd_b)
        out = out + torch.einsum("bi,bj,bk,bijk->b", pf[1], pf[13], pf[3], tcd_b)
        return out

    @torch.no_grad()
    def extract_hard_codes(self, orbits: Dict[str, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        """Extract exact D4xZ2-symmetric discrete codes z_fwd[10, 45929] and z_rev[10, 29525]."""
        k_rep = orbits["key_orb_rep"]
        k_sgn = orbits["key_orb_sgn"]
        k_am = orbits["key_anti_mirror"].astype(bool)
        e_mcx = orbits["edge_mirror_cx"]
        e_mfl = orbits["edge_mirror_flip"]

        ls = self.logits_single.detach().cpu().numpy()
        lf = self.logits_edge_f.detach().cpu().numpy()
        lr = self.logits_edge_r.detach().cpu().numpy()

        z_fwd = np.zeros((NUM_BLOCKS, KEY_CANON_ENTRIES), dtype=np.int32)
        z_rev = np.zeros((NUM_BLOCKS, EDGE_CANON_SIZE), dtype=np.int32)

        for b in range(NUM_BLOCKS):
            # Single-code types (0, 2, 3)
            ls_b = ls[b].copy()
            ls_b[k_am, :6] = -1e9
            z0 = np.argmax(ls_b, axis=-1).astype(np.int32)
            z_r = z0[k_rep]
            z_sym = np.where(k_sgn < 0, IOTA3[z_r], z_r)
            z_sym = np.where(k_am, 6 | (z_r & 1), z_sym)
            z_fwd[b] = z_sym

            # Edge+2X (type 1)
            lf_b = lf[b].copy()
            lf_b[EDGE_HALF, :6] = -1e9
            ef = np.argmax(lf_b, axis=-1).astype(np.int32)
            er = np.argmax(lr[b], axis=-1).astype(np.int32)
            out_f = np.zeros(EDGE_CANON_SIZE, dtype=np.int32)
            out_r = np.zeros(EDGE_CANON_SIZE, dtype=np.int32)
            for cx in range(EDGE_CANON_SIZE):
                cy = int(e_mcx[cx])
                fl = int(e_mfl[cx])
                if cx == EDGE_HALF:
                    zsd = int(ef[cx])
                    out_f[cx] = zsd
                    out_r[cx] = zsd
                elif cy == cx:
                    f0 = int(ef[cx])
                    out_f[cx] = f0
                    out_r[cx] = int(IOTA3[f0]) if fl else f0
                elif cx < cy:
                    f0, r0 = int(ef[cx]), int(er[cx])
                    out_f[cx], out_r[cx] = f0, r0
                    out_f[cy] = int(IOTA3[r0]) if fl else r0
                    out_r[cy] = int(IOTA3[f0]) if fl else f0
            z_fwd[b, EDGE_OFF : EDGE_OFF + EDGE_CANON_SIZE] = out_f
            z_rev[b] = out_r
        return z_fwd, z_rev


def _lookup_hard_z(
    z_fwd_t: torch.Tensor,
    z_rev_t: torch.Tensor,
    iota3_t: torch.Tensor,
    kcx: torch.Tensor,
    kfl_b: torch.Tensor,
    blk_t: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    zf0 = z_fwd_t[blk_t[:, None], kcx]  # [B, 14]
    zf = torch.where(kfl_b, iota3_t[zf0], zf0)
    zr0 = z_rev_t[blk_t[:, None], kcx[:, 4:8] - EDGE_OFF]  # [B, 4]
    zr = torch.where(kfl_b[:, 4:8], iota3_t[zr0], zr0)
    return zf, zr


@torch.no_grad()
def eb_convex_refit_macro_luts(
    bridge: P2C64Bridge,
    train_data: np.ndarray,
    base_eval_fn,
    z_fwd: np.ndarray,
    z_rev: np.ndarray,
    eb_lambda: float,
    eb_amax: float,
    args: argparse.Namespace,
    device: torch.device,
    passes: int = 4,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Provably Monotonic Majorization-Minimization (MM) Empirical Bayes Shrinkage +
    Amplitude Clipping Convex Refit for T_cw (x4), T_ea (x4), T_cd (x2)."""
    orb_cw, sgn_cw, n_cw = build_macro_orbit_map("cw")
    orb_ea, sgn_ea, n_ea = build_macro_orbit_map("ea")
    orb_cd, sgn_cd, n_cd = build_macro_orbit_map("cd")

    orb_maps = [
        (torch.from_numpy(orb_cw).long().to(device), torch.from_numpy(sgn_cw.astype(np.float32)).to(device), n_cw),
        (torch.from_numpy(orb_ea).long().to(device), torch.from_numpy(sgn_ea.astype(np.float32)).to(device), n_ea),
        (torch.from_numpy(orb_cd).long().to(device), torch.from_numpy(sgn_cd.astype(np.float32)).to(device), n_cd),
    ]
    theta_orb = [
        torch.zeros((NUM_BLOCKS, n_cw), dtype=torch.float64, device=device),
        torch.zeros((NUM_BLOCKS, n_ea), dtype=torch.float64, device=device),
        torch.zeros((NUM_BLOCKS, n_cd), dtype=torch.float64, device=device),
    ]
    T_list = [
        torch.zeros((NUM_BLOCKS, 512), dtype=torch.float32, device=device),
        torch.zeros((NUM_BLOCKS, 512), dtype=torch.float32, device=device),
        torch.zeros((NUM_BLOCKS, 512), dtype=torch.float32, device=device),
    ]
    z_fwd_t = torch.from_numpy(z_fwd).long().to(device)
    z_rev_t = torch.from_numpy(z_rev).long().to(device)
    iota3_t = torch.from_numpy(IOTA3).long().to(device)

    n = len(train_data)
    # 10 total macro terms per position -> joint MM majorizer factor m = 10 guarantees
    # strict monotonic decrease of the exact joint Ridge objective 0.5*||X*theta - r||^2 + 0.5*lam*||theta||^2
    M_TERMS = 10.0

    for _ in range(passes):
        grad_num = [
            torch.zeros((NUM_BLOCKS, n_cw), dtype=torch.float64, device=device),
            torch.zeros((NUM_BLOCKS, n_ea), dtype=torch.float64, device=device),
            torch.zeros((NUM_BLOCKS, n_cd), dtype=torch.float64, device=device),
        ]
        cnt_den = [
            torch.zeros((NUM_BLOCKS, n_cw), dtype=torch.float64, device=device),
            torch.zeros((NUM_BLOCKS, n_ea), dtype=torch.float64, device=device),
            torch.zeros((NUM_BLOCKS, n_cd), dtype=torch.float64, device=device),
        ]

        for s in range(0, n, args.val_batch_size):
            e = min(n, s + args.val_batch_size)
            sl = train_data[s:e]
            ext = bridge.extract_batch(
                sl["black"], sl["white"], sl["stm"],
                enable_aug=False, need_raw_idx=False, need_p2c=True, need_raw_flat=False,
            )
            kcx = torch.from_numpy(ext["key_cx"]).long().to(device, non_blocking=True)
            kfl = torch.from_numpy(ext["key_flip"].astype(np.float32)).to(device, non_blocking=True)
            nkf = torch.from_numpy(ext["nonkey_flat"]).long().to(device, non_blocking=True)
            ply_t = torch.from_numpy(sl["ply"].astype(np.int64)).to(device, non_blocking=True)
            blk_t = ply_t // PLIES_PER_BLOCK
            stm_t = torch.from_numpy(sl["stm"].astype(np.float32)).to(device, non_blocking=True)
            y_t = torch.from_numpy(sl["score"].astype(np.float32)).to(device, non_blocking=True)

            y_base = base_eval_fn(kcx, kfl, nkf, ply_t, stm_t)
            r_A = stm_t * (y_t - y_base)

            zf, zr = _lookup_hard_z(z_fwd_t, z_rev_t, iota3_t, kcx, kfl > 0.5, blk_t)
            keys_cw = [(zf[:, i] << 6) | (zf[:, 4 + i] << 3) | zr[:, (i + 3) & 3] for i in range(4)]
            keys_ea = [(zf[:, 4 + i] << 6) | (zr[:, i] << 3) | zf[:, 8 + i] for i in range(4)]
            keys_cd = [
                (zf[:, 0] << 6) | (zf[:, 12] << 3) | zf[:, 2],
                (zf[:, 1] << 6) | (zf[:, 13] << 3) | zf[:, 3],
            ]
            all_keys = [keys_cw, keys_ea, keys_cd]

            pred_cur = torch.zeros_like(r_A)
            for m_idx in range(3):
                Tm = T_list[m_idx]
                for kk in all_keys[m_idx]:
                    pred_cur = pred_cur + Tm[blk_t, kk]
            err_res = (r_A - pred_cur).double()

            for m_idx in range(3):
                oid_map, osgn_map, n_orb = orb_maps[m_idx]
                for kk in all_keys[m_idx]:
                    oid = oid_map[kk]
                    osg = osgn_map[kk].double()
                    flat_idx = blk_t * n_orb + oid
                    grad_num[m_idx].view(-1).scatter_add_(0, flat_idx, osg * err_res)
                    cnt_den[m_idx].view(-1).scatter_add_(0, flat_idx, (osg != 0.0).double())

        for m_idx in range(3):
            oid_map, osgn_map, _ = orb_maps[m_idx]
            # Exact MM Empirical Bayes Shrinkage + Amplitude Clipping update:
            #   theta_new = clip( (M * n_k * theta_old + sum s_n * (r_n - y_hat_old)) / (M * n_k + lambda_EB), -Amax, +Amax )
            mn = M_TERMS * cnt_den[m_idx]
            theta_new = torch.clamp((mn * theta_orb[m_idx] + grad_num[m_idx]) / (mn + eb_lambda), -eb_amax, eb_amax)
            theta_orb[m_idx] = theta_new
            T_list[m_idx] = theta_new.float()[:, oid_map] * osgn_map[None, :]

    return (
        T_list[0].cpu().numpy(),
        T_list[1].cpu().numpy(),
        T_list[2].cpu().numpy(),
    )


@torch.no_grad()
def exact_em_lut_coordinate_descent(
    bridge: P2C64Bridge,
    orbits: Dict[str, np.ndarray],
    train_data: np.ndarray,
    base_eval_fn,
    z_fwd: np.ndarray,
    z_rev: np.ndarray,
    T_cw: np.ndarray,
    T_ea: np.ndarray,
    T_cd: np.ndarray,
    eb_lambda: float,
    eb_amax: float,
    em_iters: int,
    args: argparse.Namespace,
    device: torch.device,
    tag: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Vectorized Exact EM-LUT Coordinate Descent over single-code orbits."""
    k_rep_np = orbits["key_orb_rep"]
    k_sgn_np = orbits["key_orb_sgn"]
    k_am_np = orbits["key_anti_mirror"].astype(bool)

    k_rep_t = torch.from_numpy(k_rep_np).long().to(device)
    k_sgn_t = torch.from_numpy(k_sgn_np).to(device)
    iota3_t = torch.from_numpy(IOTA3).long().to(device)
    cands = torch.arange(8, dtype=torch.long, device=device)

    n = len(train_data)
    min_orbit_hits = max(5, int(n // 5000))  # Prevent noisy code flips on rare 1-2 hit orbits
    for it in range(em_iters):
        t0 = time.perf_counter()
        z_fwd_t = torch.from_numpy(z_fwd).long().to(device)
        z_rev_t = torch.from_numpy(z_rev).long().to(device)
        tcw_t = torch.from_numpy(T_cw).float().to(device)
        tea_t = torch.from_numpy(T_ea).float().to(device)
        tcd_t = torch.from_numpy(T_cd).float().to(device)

        delta_single = torch.zeros((NUM_BLOCKS, KEY_CANON_ENTRIES, 8), dtype=torch.float64, device=device)
        hits_single = torch.zeros((NUM_BLOCKS, KEY_CANON_ENTRIES), dtype=torch.int64, device=device)
        ones_buf = None

        for s in range(0, n, args.val_batch_size):
            e = min(n, s + args.val_batch_size)
            sl = train_data[s:e]
            ext = bridge.extract_batch(
                sl["black"], sl["white"], sl["stm"],
                enable_aug=False, need_raw_idx=False, need_p2c=True, need_raw_flat=False,
            )
            kcx = torch.from_numpy(ext["key_cx"]).long().to(device, non_blocking=True)
            kfl = torch.from_numpy(ext["key_flip"].astype(np.float32)).to(device, non_blocking=True)
            nkf = torch.from_numpy(ext["nonkey_flat"]).long().to(device, non_blocking=True)
            ply_t = torch.from_numpy(sl["ply"].astype(np.int64)).to(device, non_blocking=True)
            blk_t = ply_t // PLIES_PER_BLOCK
            stm_t = torch.from_numpy(sl["stm"].astype(np.float32)).to(device, non_blocking=True)
            y_t = torch.from_numpy(sl["score"].astype(np.float32)).to(device, non_blocking=True)

            y_base = base_eval_fn(kcx, kfl, nkf, ply_t, stm_t)
            r_A = stm_t * (y_t - y_base)
            fl_b = kfl > 0.5
            zf, zr = _lookup_hard_z(z_fwd_t, z_rev_t, iota3_t, kcx, fl_b, blk_t)
            if ones_buf is None or ones_buf.shape[0] != kcx.shape[0]:
                ones_buf = torch.ones(kcx.shape[0], dtype=torch.int64, device=device)

            pred_cur = torch.zeros_like(r_A)
            for i in range(4):
                pred_cur = pred_cur + tcw_t[blk_t, (zf[:, i] << 6) | (zf[:, 4 + i] << 3) | zr[:, (i + 3) & 3]]
                pred_cur = pred_cur + tea_t[blk_t, (zf[:, 4 + i] << 6) | (zr[:, i] << 3) | zf[:, 8 + i]]
            pred_cur = pred_cur + tcd_t[blk_t, (zf[:, 0] << 6) | (zf[:, 12] << 3) | zf[:, 2]]
            pred_cur = pred_cur + tcd_t[blk_t, (zf[:, 1] << 6) | (zf[:, 13] << 3) | zf[:, 3]]
            err_cur = pred_cur - r_A
            cur_sq = err_cur * err_cur

            def accum_single(p_idx: int, new_term_fn, cur_term: torch.Tensor) -> None:
                cx_p = kcx[:, p_idx]
                rep_p = k_rep_t[cx_p]
                eff_flip = fl_b[:, p_idx] ^ (k_sgn_t[cx_p] < 0)
                z_cand = torch.where(eff_flip[:, None], iota3_t[cands][None, :], cands[None, :])
                new_t = new_term_fn(z_cand)
                err_new = err_cur[:, None] - cur_term[:, None] + new_t
                dloss = (err_new * err_new - cur_sq[:, None]).double()
                flat1 = blk_t * KEY_CANON_ENTRIES + rep_p
                hits_single.view(-1).scatter_add_(0, flat1, ones_buf)
                delta_single.view(-1, 8).scatter_add_(0, flat1[:, None].expand(-1, 8), dloss)

            for i in range(4):
                t_cw_cur = tcw_t[blk_t, (zf[:, i] << 6) | (zf[:, 4 + i] << 3) | zr[:, (i + 3) & 3]]
                if i < 2:
                    t_cd_cur = tcd_t[blk_t, (zf[:, i] << 6) | (zf[:, 12 + i] << 3) | zf[:, i + 2]]
                    accum_single(
                        i,
                        lambda zc, ii=i: tcw_t[blk_t[:, None], (zc << 6) | (zf[:, 4 + ii, None] << 3) | zr[:, (ii + 3) & 3, None]]
                                       + tcd_t[blk_t[:, None], (zc << 6) | (zf[:, 12 + ii, None] << 3) | zf[:, ii + 2, None]],
                        t_cw_cur + t_cd_cur,
                    )
                else:
                    d = i - 2
                    t_cd_cur = tcd_t[blk_t, (zf[:, d] << 6) | (zf[:, 12 + d] << 3) | zf[:, i]]
                    accum_single(
                        i,
                        lambda zc, ii=i, dd=d: tcw_t[blk_t[:, None], (zc << 6) | (zf[:, 4 + ii, None] << 3) | zr[:, (ii + 3) & 3, None]]
                                             + tcd_t[blk_t[:, None], (zf[:, dd, None] << 6) | (zf[:, 12 + dd, None] << 3) | zc],
                        t_cw_cur + t_cd_cur,
                    )

            for i in range(4):
                t_ea_cur = tea_t[blk_t, (zf[:, 4 + i] << 6) | (zr[:, i] << 3) | zf[:, 8 + i]]
                accum_single(
                    8 + i,
                    lambda zc, ii=i: tea_t[blk_t[:, None], (zf[:, 4 + ii, None] << 6) | (zr[:, ii, None] << 3) | zc],
                    t_ea_cur,
                )

            for d in range(2):
                t_cd_cur = tcd_t[blk_t, (zf[:, d] << 6) | (zf[:, 12 + d] << 3) | zf[:, d + 2]]
                accum_single(
                    12 + d,
                    lambda zc, dd=d: tcd_t[blk_t[:, None], (zf[:, dd, None] << 6) | (zc << 3) | zf[:, dd + 2, None]],
                    t_cd_cur,
                )

        ds_np = delta_single.cpu().numpy()
        hs_np = hits_single.cpu().numpy()
        flips_cnt = 0
        for b in range(NUM_BLOCKS):
            for cx in range(KEY_CANON_ENTRIES):
                if EDGE_OFF <= cx < EDGE_OFF + EDGE_CANON_SIZE:
                    continue
                if k_rep_np[cx] != cx or hs_np[b, cx] < min_orbit_hits:
                    continue
                row = ds_np[b, cx]
                best_c = (6 + int(np.argmin(row[6:8]))) if k_am_np[cx] else int(np.argmin(row))
                if row[best_c] < -1e-3 and best_c != z_fwd[b, cx]:
                    z_fwd[b, cx] = best_c
                    flips_cnt += 1
            z_r = z_fwd[b, k_rep_np]
            z_sym = np.where(k_sgn_np < 0, IOTA3[z_r], z_r)
            z_sym = np.where(k_am_np, 6 | (z_r & 1), z_sym)
            z_fwd[b, :EDGE_OFF] = z_sym[:EDGE_OFF]
            z_fwd[b, EDGE_OFF + EDGE_CANON_SIZE :] = z_sym[EDGE_OFF + EDGE_CANON_SIZE :]

        T_cw, T_ea, T_cd = eb_convex_refit_macro_luts(
            bridge, train_data, base_eval_fn, z_fwd, z_rev, eb_lambda, eb_amax, args, device, passes=3
        )
        print(f"  [{tag}/EM-LUT] Iter {it+1}/{em_iters} | updated_orbits={flips_cnt} (min_hits={min_orbit_hits}) | {time.perf_counter()-t0:.1f}s")

    return z_fwd, z_rev, T_cw, T_ea, T_cd


def train_sconed_3w3b_pipeline(
    bridge: P2C64Bridge,
    orbits: Dict[str, np.ndarray],
    train_data: np.ndarray,
    base_eval_fn,
    soft_epochs: int,
    em_iters: int,
    args: argparse.Namespace,
    device: torch.device,
    tag: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    # Stage 1: Soft Multilinear Tensor Contraction + Temperature Annealing
    mod = SoftSconeD3W3B(orbits, device)
    opt = torch.optim.AdamW(mod.parameters(), lr=args.b_lr, weight_decay=0.0)
    n = len(train_data)
    rng = np.random.default_rng(args.seed + 505)
    total_steps = max(1, soft_epochs * math.ceil(n / args.batch_size))
    gstep = 0

    for ep in range(soft_epochs):
        t0 = time.perf_counter()
        perm = rng.permutation(n)
        loss_sum = 0.0
        steps = 0
        for s in range(0, n, args.batch_size):
            idx_b = perm[s : min(n, s + args.batch_size)]
            sl = train_data[idx_b]
            ext = bridge.extract_batch(
                sl["black"], sl["white"], sl["stm"],
                aug_seed=int(rng.integers(0, 2**62)), enable_aug=True,
                need_raw_idx=False, need_p2c=True, need_raw_flat=False,
            )
            kcx = torch.from_numpy(ext["key_cx"]).long().to(device, non_blocking=True)
            kfl = torch.from_numpy(ext["key_flip"].astype(np.float32)).to(device, non_blocking=True)
            nkf = torch.from_numpy(ext["nonkey_flat"]).long().to(device, non_blocking=True)
            ply_t = torch.from_numpy(sl["ply"].astype(np.int64)).to(device, non_blocking=True)
            blk_t = ply_t // PLIES_PER_BLOCK
            stm_t = torch.from_numpy(ext["stm"].astype(np.float32)).to(device, non_blocking=True)
            y_t = torch.from_numpy(sl["score"].astype(np.float32)).to(device, non_blocking=True)

            frac = gstep / max(1, total_steps - 1)
            tau = args.tau_start * ((args.tau_end / args.tau_start) ** frac)
            gstep += 1

            with torch.no_grad():
                y_base = base_eval_fn(kcx, kfl, nkf, ply_t, stm_t)
            a_lut = mod(kcx, kfl, blk_t, tau=tau)
            pred = y_base + stm_t * a_lut

            loss = F.smooth_l1_loss(pred, y_t, beta=args.huber_beta)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            loss_sum += float(loss.item())
            steps += 1

        print(f"  [{tag}/Soft] Epoch {ep+1}/{soft_epochs} | tau={tau:.3f} | huber={loss_sum/max(1,steps):.4f} | {time.perf_counter()-t0:.1f}s")

    # Stage 2: Hard Switch + Empirical Bayes Shrinkage & Amplitude Clipping Convex Refit
    t1 = time.perf_counter()
    z_fwd, z_rev = mod.extract_hard_codes(orbits)
    T_cw, T_ea, T_cd = eb_convex_refit_macro_luts(
        bridge, train_data, base_eval_fn, z_fwd, z_rev, args.eb_lambda, args.eb_amax, args, device, passes=3
    )
    print(f"  [{tag}/EB-Refit] Hard switch + 3-pass EB shrinkage (lam={args.eb_lambda}, Amax={args.eb_amax}) done in {time.perf_counter()-t1:.1f}s")

    # Stage 3: Exact EM-LUT Coordinate Descent
    if em_iters > 0:
        z_fwd, z_rev, T_cw, T_ea, T_cd = exact_em_lut_coordinate_descent(
            bridge, orbits, train_data, base_eval_fn, z_fwd, z_rev,
            T_cw, T_ea, T_cd, args.eb_lambda, args.eb_amax, em_iters, args, device, tag,
        )
    return z_fwd, z_rev, T_cw, T_ea, T_cd


# ============================================================================
# Metrics Logging Helper (JSON + CSV + Pretty Comparison Table)
# ============================================================================
def save_metrics(work_dir: Path, all_metrics: Dict[str, Dict[str, float]]) -> None:
    (work_dir / "metrics.json").write_text(json.dumps(all_metrics, indent=2))
    csv_path = work_dir / "metrics.csv"
    cols = ["model", "rmse_all", "mae_all", "rmse_0_19", "mae_0_19", "rmse_20_39", "mae_20_39", "rmse_40_59", "mae_40_59"]
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for name, m in all_metrics.items():
            row = {"model": name}
            for c in cols[1:]:
                row[c] = f"{m.get(c, 0.0):.5f}"
            w.writerow(row)


def print_summary_table(all_metrics: Dict[str, Dict[str, float]]) -> None:
    print("\n================ P2C-64 Val RMSE / MAE Summary (Quantized .p2c) ================")
    print(f"{'Model':<20} | {'All RMSE':>8} {'All MAE':>8} | {'0-19':>8} | {'20-39':>8} | {'40-59':>8}")
    print("-" * 74)
    for name, m in all_metrics.items():
        print(
            f"{name:<20} | {m['rmse_all']:8.4f} {m['mae_all']:8.4f} | "
            f"{m['rmse_0_19']:8.4f} | {m['rmse_20_39']:8.4f} | {m['rmse_40_59']:8.4f}"
        )


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="P2C-64 v1.1 End-to-End Training Pipeline (A100 / CPU)")
    ap.add_argument("--train-bin", type=Path, required=True)
    ap.add_argument("--val-bin", type=Path, required=True)
    ap.add_argument("--work-dir", type=Path, required=True)
    ap.add_argument("--resume", action="store_true", help="Resume from checkpoints / skip completed phases")
    ap.add_argument(
        "--phases",
        type=str,
        default="all",
        help="Comma-separated phases: all | phase1 | phase2 | raw | linear | cand_a | cand_b | cand_c",
    )
    ap.add_argument("--batch-size", type=int, default=16384)
    ap.add_argument("--val-batch-size", type=int, default=32768)
    ap.add_argument("--score-scale", type=float, default=P2C_DEFAULT_SCALE)
    ap.add_argument("--huber-beta", type=float, default=4.0)
    ap.add_argument("--smooth-lambda", type=float, default=1e-4)

    # Phase 1
    ap.add_argument("--p1-epochs", type=int, default=6)
    ap.add_argument("--p1-lr", type=float, default=0.05)

    # Phase 2A (CAND_A)
    ap.add_argument("--a-epochs", type=int, default=4)
    ap.add_argument("--a-qat-epochs", type=int, default=2)
    ap.add_argument("--a-lr", type=float, default=0.01)
    ap.add_argument("--mu-init", type=str, default="1.0,0.35")

    # Phase 2B (CAND_B)
    ap.add_argument("--b-soft-epochs", type=int, default=4)
    ap.add_argument("--b-lr", type=float, default=0.02)
    ap.add_argument("--tau-start", type=float, default=1.5)
    ap.add_argument("--tau-end", type=float, default=0.15)
    ap.add_argument("--eb-lambda", type=float, default=200.0)
    ap.add_argument("--eb-amax", type=float, default=4.0)
    ap.add_argument("--em-iters", type=int, default=2)

    # Phase 2C (CAND_C)
    ap.add_argument("--c-screlu-epochs", type=int, default=3)
    ap.add_argument("--c-qat-epochs", type=int, default=1)
    ap.add_argument("--c-soft-epochs", type=int, default=3)
    ap.add_argument("--c-em-iters", type=int, default=2)

    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", type=str, default="auto")
    return ap.parse_args()


def expand_phases(spec: str) -> List[str]:
    out: List[str] = []
    for tok in [t.strip().lower() for t in spec.split(",") if t.strip()]:
        if tok == "all":
            out.extend(["raw", "linear", "cand_a", "cand_b", "cand_c"])
        elif tok == "phase1":
            out.extend(["raw", "linear"])
        elif tok == "phase2":
            out.extend(["cand_a", "cand_b", "cand_c"])
        else:
            out.append(tok)
    # Deduplicate preserving order
    return list(dict.fromkeys(out))


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    torch.set_float32_matmul_precision("high")

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    args.work_dir.mkdir(parents=True, exist_ok=True)
    phases = expand_phases(args.phases)
    mu_parts = [float(x) for x in args.mu_init.split(",")]
    mu_init = (mu_parts[0], mu_parts[1] if len(mu_parts) > 1 else 0.35)

    bridge = P2C64Bridge()
    orbits = bridge.get_orbit_tables()
    train_data = load_packed_dataset(args.train_bin)
    val_data = load_packed_dataset(args.val_bin)

    print(f"=== P2C-64 v1.1 Training Pipeline | device={device} | phases={phases} ===")
    print(f"  Train={len(train_data):,} | Val={len(val_data):,} | batch={args.batch_size} | scale={args.score_scale}")

    metrics_file = args.work_dir / "metrics.json"
    all_metrics: Dict[str, Dict[str, float]] = {}
    if args.resume and metrics_file.exists():
        all_metrics = json.loads(metrics_file.read_text())

    # ---- Phase 1A: RAW_NTUPLE_64 ----
    raw_p2c = args.work_dir / "raw_ntuple64.p2c"
    if "raw" in phases:
        if args.resume and raw_p2c.exists() and "RAW_NTUPLE_64" in all_metrics:
            print(f"[Skip] RAW_NTUPLE_64 already completed ({raw_p2c})")
        else:
            _, m_raw = train_raw_ntuple64(
                bridge, orbits, train_data, val_data, args, device,
                raw_p2c, args.work_dir / "ckpt_raw.pt",
            )
            all_metrics["RAW_NTUPLE_64"] = m_raw
            save_metrics(args.work_dir, all_metrics)

    # ---- Phase 1B: LINEAR_64 ----
    lin_p2c = args.work_dir / "linear64.p2c"
    lin_model: Optional[P2C64Model] = None
    if "linear" in phases:
        if args.resume and lin_p2c.exists() and "LINEAR_64" in all_metrics:
            print(f"[Skip] LINEAR_64 already completed ({lin_p2c})")
            lin_model = load_p2c(lin_p2c)  # type: ignore[assignment]
        else:
            lin_model, m_lin = train_linear64(
                bridge, orbits, train_data, val_data, args, device,
                lin_p2c, args.work_dir / "ckpt_linear.pt",
            )
            all_metrics["LINEAR_64"] = m_lin
            save_metrics(args.work_dir, all_metrics)

    need_p2 = any(p in phases for p in ("cand_a", "cand_b", "cand_c"))
    if need_p2:
        if lin_model is None:
            if not lin_p2c.exists():
                raise FileNotFoundError(f"Phase 2 requires trained LINEAR_64 at {lin_p2c}")
            lin_model = load_p2c(lin_p2c)  # type: ignore[assignment]
        assert isinstance(lin_model, P2C64Model)
        frozen_lin = FrozenLinearEvaluator(lin_model, device)

        # ---- Phase 2A: CAND_A_SCRELU_K2 ----
        cand_a_p2c = args.work_dir / "cand_a_screlu_k2.p2c"
        if "cand_a" in phases:
            if args.resume and cand_a_p2c.exists() and "CAND_A_SCRELU_K2" in all_metrics:
                print(f"[Skip] CAND_A_SCRELU_K2 already completed ({cand_a_p2c})")
            else:
                print("\n=== [Phase 2A] Training CAND_A_SCRELU_K2 (Zero-Init Residual + QAT) ===")
                mod_a = train_screlu_residual(
                    bridge, orbits, frozen_lin, train_data,
                    k_channels=2, float_epochs=args.a_epochs, qat_epochs=args.a_qat_epochs,
                    lr=args.a_lr, mu_init=mu_init, args=args, device=device, tag="CAND_A",
                )
                q = mod_a.export_quantized_arrays()
                m_a = make_empty_p2c64(EvalMode.CAND_A_SCRELU_K2, score_scale=args.score_scale)
                m_a.key_blocks["w"] = lin_model.key_blocks["w"].copy()
                m_a.nonkey_w = lin_model.nonkey_w.copy()
                m_a.corner_tempo = lin_model.corner_tempo.copy()
                m_a.key_blocks["aux"][:, :, 0] = q["o"][:, :, 0].view(np.uint8)
                m_a.key_blocks["aux"][:, :, 1] = q["o"][:, :, 1].view(np.uint8)
                m_a.key_blocks["aux"][:, :, 2] = q["e"][:, :, 0].view(np.uint8)
                m_a.key_blocks["aux"][:, :, 3] = q["e"][:, :, 1].view(np.uint8)
                m_a.screlu_mu = q["mu"]
                m_a.screlu_beta_sq = q["beta_sq"]
                m_a.screlu_beta_pr = q["beta_pr"]
                m_a = symmetrize_p2c64_model(m_a, orbits)
                save_p2c64(cand_a_p2c, m_a)
                met_a = evaluate_deployed_p2c(bridge, m_a, val_data, args.val_batch_size)
                all_metrics["CAND_A_SCRELU_K2"] = met_a
                save_metrics(args.work_dir, all_metrics)
                print(f"  [CAND_A] Saved {cand_a_p2c} | Val RMSE={met_a['rmse_all']:.4f} MAE={met_a['mae_all']:.4f}")

        # ---- Phase 2B: CAND_B_3W3B ----
        cand_b_p2c = args.work_dir / "cand_b_3w3b.p2c"
        if "cand_b" in phases:
            if args.resume and cand_b_p2c.exists() and "CAND_B_3W3B" in all_metrics:
                print(f"[Skip] CAND_B_3W3B already completed ({cand_b_p2c})")
            else:
                print("\n=== [Phase 2B] Training CAND_B_3W3B (Soft Contraction -> EB Convex Refit -> Exact EM-LUT) ===")
                z_fwd, z_rev, T_cw, T_ea, T_cd = train_sconed_3w3b_pipeline(
                    bridge, orbits, train_data, frozen_lin.eval_disc,
                    soft_epochs=args.b_soft_epochs, em_iters=args.em_iters,
                    args=args, device=device, tag="CAND_B",
                )
                m_b = make_empty_p2c64(EvalMode.CAND_B_3W3B, score_scale=args.score_scale)
                m_b.key_blocks["w"] = lin_model.key_blocks["w"].copy()
                m_b.nonkey_w = lin_model.nonkey_w.copy()
                m_b.corner_tempo = lin_model.corner_tempo.copy()
                m_b.key_blocks["aux"][:, :, 0] = z_fwd.astype(np.uint8)
                m_b.key_blocks["aux"][:, EDGE_OFF : EDGE_OFF + EDGE_CANON_SIZE, 1] = z_rev.astype(np.uint8)
                m_b.macro_cw = np.clip(np.round(T_cw * args.score_scale), -32767, 32767).astype(np.int16)
                m_b.macro_ea = np.clip(np.round(T_ea * args.score_scale), -32767, 32767).astype(np.int16)
                m_b.macro_cd = np.clip(np.round(T_cd * args.score_scale), -32767, 32767).astype(np.int16)
                m_b = symmetrize_p2c64_model(m_b, orbits)
                save_p2c64(cand_b_p2c, m_b)
                met_b = evaluate_deployed_p2c(bridge, m_b, val_data, args.val_batch_size)
                all_metrics["CAND_B_3W3B"] = met_b
                save_metrics(args.work_dir, all_metrics)
                print(f"  [CAND_B] Saved {cand_b_p2c} | Val RMSE={met_b['rmse_all']:.4f} MAE={met_b['mae_all']:.4f}")

        # ---- Phase 2C: CAND_C_HYBRID (Phase 2-1 Dual-SCReLU K=1 -> Phase 2-2 3way-3bit) ----
        cand_c_p2c = args.work_dir / "cand_c_hybrid.p2c"
        if "cand_c" in phases:
            if args.resume and cand_c_p2c.exists() and "CAND_C_HYBRID" in all_metrics:
                print(f"[Skip] CAND_C_HYBRID already completed ({cand_c_p2c})")
            else:
                print("\n=== [Phase 2C-1] Training CAND_C Dual-SCReLU K=1 Residual ===")
                mod_c1 = train_screlu_residual(
                    bridge, orbits, frozen_lin, train_data,
                    k_channels=1, float_epochs=args.c_screlu_epochs, qat_epochs=args.c_qat_epochs,
                    lr=args.a_lr, mu_init=mu_init, args=args, device=device, tag="CAND_C-K1",
                )
                qc1 = mod_c1.export_quantized_arrays()

                # Freeze exact quantized K=1 SCReLU + LINEAR_64 as base for Phase 2C-2
                k_rep_t = torch.from_numpy(orbits["key_orb_rep"]).long().to(device)
                k_sgn_t = torch.from_numpy(orbits["key_orb_sgn"].astype(np.float32)).to(device)
                o0_q = torch.from_numpy(qc1["o"][:, :, 0].astype(np.float32)).to(device)
                e0_q = torch.from_numpy(qc1["e"][:, :, 0].astype(np.float32)).to(device)
                mu0_q = torch.from_numpy(qc1["mu"][:, :, 0].astype(np.float32)).to(device)
                bsq0_q = torch.from_numpy(qc1["beta_sq"][:, :, 0].astype(np.float32)).to(device)

                @torch.no_grad()
                def eval_lin_plus_k1(kcx, kfl, nkf, ply_t, stm_t):
                    y_lin = frozen_lin.eval_disc(kcx, kfl, nkf, ply_t, stm_t)
                    blk_t = ply_t // PLIES_PER_BLOCK
                    rep = k_rep_t[kcx]
                    sgn = (1.0 - 2.0 * kfl) * k_sgn_t[kcx]
                    ko = o0_q[blk_t[:, None], rep] * sgn
                    ke = e0_q[blk_t[:, None], rep]
                    acc = torch.zeros(kcx.shape[0], dtype=torch.float32, device=device)
                    s_e = torch.zeros_like(acc)
                    for r in range(4):
                        c1 = (r + 1) & 3
                        o_r = ko[:, 4 + r] + ko[:, 8 + r] + ko[:, r] + ko[:, c1]
                        e_r = ke[:, 4 + r] + ke[:, 8 + r] + ke[:, r] + ke[:, c1] + mu0_q[blk_t, 0]
                        b_r = torch.clamp(e_r + o_r, 0.0, float(SCRELU_Q))
                        w_r = torch.clamp(e_r - o_r, 0.0, float(SCRELU_Q))
                        s_e = s_e + (b_r - w_r) * (b_r + w_r)
                    acc = acc + bsq0_q[blk_t, 0] * s_e
                    s_d = torch.zeros_like(acc)
                    for d in range(2):
                        o_d = ko[:, 12 + d] + ko[:, d] + ko[:, d + 2]
                        e_d = ke[:, 12 + d] + ke[:, d] + ke[:, d + 2] + mu0_q[blk_t, 1]
                        b_d = torch.clamp(e_d + o_d, 0.0, float(SCRELU_Q))
                        w_d = torch.clamp(e_d - o_d, 0.0, float(SCRELU_Q))
                        s_d = s_d + (b_d - w_d) * (b_d + w_d)
                    acc = acc + bsq0_q[blk_t, 1] * s_d
                    g_o = ko.sum(dim=1)
                    g_e = ke.sum(dim=1) + mu0_q[blk_t, 2]
                    b_g = torch.clamp(g_e + g_o, 0.0, float(SCRELU_Q))
                    w_g = torch.clamp(g_e - g_o, 0.0, float(SCRELU_Q))
                    acc = acc + bsq0_q[blk_t, 2] * ((b_g - w_g) * (b_g + w_g))
                    a_sc = torch.trunc(acc / float(1 << SCRELU_SHIFT)) / float(args.score_scale)
                    return y_lin + stm_t * a_sc

                print("\n=== [Phase 2C-2] Training CAND_C 3way-3bit Residual on (LINEAR + K1) ===")
                z_fwd, z_rev, T_cw, T_ea, T_cd = train_sconed_3w3b_pipeline(
                    bridge, orbits, train_data, eval_lin_plus_k1,
                    soft_epochs=args.c_soft_epochs, em_iters=args.c_em_iters,
                    args=args, device=device, tag="CAND_C-LUT",
                )
                m_c = make_empty_p2c64(EvalMode.CAND_C_HYBRID, score_scale=args.score_scale)
                m_c.key_blocks["w"] = lin_model.key_blocks["w"].copy()
                m_c.nonkey_w = lin_model.nonkey_w.copy()
                m_c.corner_tempo = lin_model.corner_tempo.copy()
                m_c.key_blocks["aux"][:, :, 0] = qc1["o"][:, :, 0].view(np.uint8)
                m_c.key_blocks["aux"][:, :, 1] = qc1["e"][:, :, 0].view(np.uint8)
                packed = z_fwd.copy()
                packed[:, EDGE_OFF : EDGE_OFF + EDGE_CANON_SIZE] |= (z_rev << 3)
                m_c.key_blocks["aux"][:, :, 2] = packed.astype(np.uint8)
                m_c.key_blocks["aux"][:, :, 3] = 0
                m_c.screlu_mu[:, :, 0] = qc1["mu"][:, :, 0]
                m_c.screlu_beta_sq[:, :, 0] = qc1["beta_sq"][:, :, 0]
                m_c.macro_cw = np.clip(np.round(T_cw * args.score_scale), -32767, 32767).astype(np.int16)
                m_c.macro_ea = np.clip(np.round(T_ea * args.score_scale), -32767, 32767).astype(np.int16)
                m_c.macro_cd = np.clip(np.round(T_cd * args.score_scale), -32767, 32767).astype(np.int16)
                m_c = symmetrize_p2c64_model(m_c, orbits)
                save_p2c64(cand_c_p2c, m_c)
                met_c = evaluate_deployed_p2c(bridge, m_c, val_data, args.val_batch_size)
                all_metrics["CAND_C_HYBRID"] = met_c
                save_metrics(args.work_dir, all_metrics)
                print(f"  [CAND_C] Saved {cand_c_p2c} | Val RMSE={met_c['rmse_all']:.4f} MAE={met_c['mae_all']:.4f}")

    if all_metrics:
        print_summary_table(all_metrics)


if __name__ == "__main__":
    main()
