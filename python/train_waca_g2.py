#!/usr/bin/env python3
"""P2C-64 v3.0 (WACA-G2) PyTorch Training Pipeline (CUDA A100/L4/T4 & CPU).

Reuses frozen Phase-1 64-pattern baseline (`raw_ntuple64.p2c`) with 0-min Phase 1
retraining, and trains the 2-stage cascade global nonlinear residual:
  - Stage 2.5 : 60-ply 16-dim D4xZ2 global mobility + geometry linear gate (2.16 KiB)
                trained on residual (Y - V_1) with adjacent-ply smoothing + exact QAT.
  - Stage 3   : 20-block Even/Odd parity-separated Key-14 4ch embeddings (7.01 MiB)
                + 60-ply Key-14 SCReLU head
                + 60-ply 24 -> H -> 1 pure-integer SCReLU MLP
                trained on residual (Y - V_25) with same-parity block smoothing,
                midgame + boundary-focus weighting, and bit-exact integer QAT.

Exports `waca_g2.p2c` (.p2c v3.0) and `metrics_v30.{json,csv}`.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))

from p2c64_bridge import (
    KEY_RAW_ENTRIES,
    NUM_PLIES,
    PACKED_SAMPLE_DTYPE,
    P2C64Bridge,
)
from p2c64_io import (
    P2C_DEFAULT_SCALE,
    PLY_TO_EO_BLK,
    SCRELU_BETA_MAX,
    SCRELU_Q,
    SCRELU_SHIFT,
    WACA_EO_BLOCKS,
    WACA_F25,
    WACA_F3,
    WACA_FEAT_CLAMP,
    WACA_MAX_RES,
    WACA_MLP_B1_MAX,
    WACA_MLP_B2_MAX,
    WACA_MLP_S1,
    WACA_MLP_S2,
    WACA_MLP_W1_MAX,
    WACA_MLP_W2_MAX,
    WACA_S25_B_MAX,
    WACA_S25_SHIFT,
    WACA_S25_W_MAX,
    RawNTupleModel,
    WacaG2Model,
    eval_waca_g2_int_batch,
    load_p2c,
    make_empty_waca_g2,
    save_waca_g2,
    symmetrize_waca_g2_model,
)

EMB_SCALE = 32.0


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_packed_dataset(path: Path, max_samples: int = 0) -> np.ndarray:
    arr = np.fromfile(path, dtype=PACKED_SAMPLE_DTYPE)
    if len(arr) == 0:
        raise RuntimeError(f"Dataset file is empty: {path}")
    if max_samples > 0 and len(arr) > max_samples:
        # Deterministic strided slice across full dataset to preserve ply/game balance
        idx = np.linspace(0, len(arr) - 1, max_samples, dtype=np.int64)
        arr = np.ascontiguousarray(arr[idx])
    return arr


def ste_round_clamp(x: torch.Tensor, lo: float, hi: float) -> torch.Tensor:
    y = torch.clamp(torch.round(x), lo, hi)
    return x + (y - x).detach()


def ste_floor_rshift_nonneg(x: torch.Tensor, shift: int) -> torch.Tensor:
    """Bit-exact STE for C++ `x >> shift` when x >= 0."""
    denom = float(1 << shift)
    xf = x / denom
    y = torch.floor(xf)
    return xf + (y - xf).detach()


def ste_sym_rshift_int(x: torch.Tensor, shift: int) -> torch.Tensor:
    """Bit-exact STE for C++ `sym_rshift(x, shift)` (truncation toward zero)."""
    denom = float(1 << shift)
    xf = x / denom
    y = torch.trunc(xf)
    return xf + (y - xf).detach()


@torch.no_grad()
def apply_axis0_smoothness_(w: torch.Tensor, alpha: float) -> None:
    """Decoupled proximal Laplacian diffusion along axis 0 (e.g., 60 plies)."""
    if alpha <= 0.0 or w.shape[0] < 2:
        return
    a = min(0.20, float(alpha))
    w_prev = w.clone()
    w[0].add_(w_prev[1] - w_prev[0], alpha=a)
    w[-1].add_(w_prev[-2] - w_prev[-1], alpha=a)
    if w.shape[0] > 2:
        w[1:-1].add_(w_prev[:-2] + w_prev[2:] - 2.0 * w_prev[1:-1], alpha=a)


@torch.no_grad()
def apply_eo_block_smoothness_(h_emb: torch.Tensor, alpha: float) -> None:
    """Same-parity proximal Laplacian smoothing on [20, 91854, 4] (even blocks & odd blocks independently!)."""
    if alpha <= 0.0 or h_emb.shape[0] < 4:
        return
    for parity in (0, 1):
        sub = h_emb[parity::2].clone()  # [10, 91854, 4]
        apply_axis0_smoothness_(sub, alpha)
        h_emb[parity::2].copy_(sub)


def compute_sample_weights(
    ply_t: torch.Tensor,
    v1_discs: torch.Tensor,
    midgame_boost: float = 1.25,
    bnd_boost: float = 0.60,
    bnd_sigma: float = 6.0,
) -> torch.Tensor:
    """Midgame + alpha-beta boundary-focus weighting (normalized to mean 1.0)."""
    w_ply = torch.where(
        ply_t <= 39,
        torch.full_like(v1_discs, midgame_boost),
        torch.where(ply_t <= 49, torch.full_like(v1_discs, 0.85), torch.full_like(v1_discs, 0.55)),
    )
    w_bnd = 1.0 + bnd_boost * torch.exp(-((v1_discs / bnd_sigma) ** 2))
    w = w_ply * w_bnd
    return w / (w.mean() + 1e-8)


def weighted_huber_loss(pred: torch.Tensor, target: torch.Tensor, weight: torch.Tensor, delta: float = 4.0) -> torch.Tensor:
    per = F.huber_loss(pred, target, reduction="none", delta=delta)
    return (per * weight).mean()


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


def evaluate_deployed_waca_g2(
    bridge: P2C64Bridge,
    model: WacaG2Model,
    val_data: np.ndarray,
    batch_size: int = 32768,
) -> Dict[str, Dict[str, float]]:
    """Evaluate exact quantized integer Stage 1 (RAW_64), Stage 2.5, and Stage 3 (WACA_G2 Full)."""
    n = len(val_data)
    p1 = np.empty(n, dtype=np.float64)
    p25 = np.empty(n, dtype=np.float64)
    p3 = np.empty(n, dtype=np.float64)
    y_true = val_data["score"].astype(np.float64)
    plies = val_data["ply"]
    scale = float(model.score_scale)

    for s in range(0, n, batch_size):
        e = min(n, s + batch_size)
        sl = val_data[s:e]
        ext = bridge.extract_waca_g2_batch(
            sl["black"], sl["white"], sl["stm"], sl["ply"],
            model.tier1_w, enable_aug=False, need_stage3=True,
        )
        v1, v25, v3 = eval_waca_g2_int_batch(
            model, ext["v1"], ext["f25"], ext["f3_geom"], ext["key_ptm"], sl["ply"]
        )
        p1[s:e] = v1.astype(np.float64) / scale
        p25[s:e] = v25.astype(np.float64) / scale
        p3[s:e] = v3.astype(np.float64) / scale

    r25_abs = np.abs(p25 - p1)
    r3_delta_abs = np.abs(p3 - p25)
    r_tot_abs = np.abs(p3 - p1)
    mid = plies <= 39

    out: Dict[str, Dict[str, float]] = {
        "stage1_raw64": compute_metrics_dict(p1, y_true, plies),
        "stage25": compute_metrics_dict(p25, y_true, plies),
        "stage3_full": compute_metrics_dict(p3, y_true, plies),
        "residual_stats": {
            "r25_p90_all": float(np.percentile(r25_abs, 90)),
            "r25_p95_all": float(np.percentile(r25_abs, 95)),
            "r25_p99_all": float(np.percentile(r25_abs, 99)),
            "r3_delta_p90_all": float(np.percentile(r3_delta_abs, 90)),
            "r3_delta_p95_all": float(np.percentile(r3_delta_abs, 95)),
            "r3_delta_p99_all": float(np.percentile(r3_delta_abs, 99)),
            "rtot_p95_mid": float(np.percentile(r_tot_abs[mid], 95)) if np.any(mid) else 0.0,
            "rtot_p99_mid": float(np.percentile(r_tot_abs[mid], 99)) if np.any(mid) else 0.0,
            "r3_delta_p95_mid": float(np.percentile(r3_delta_abs[mid], 95)) if np.any(mid) else 0.0,
        },
    }
    return out


# ============================================================================
# Stage 2.5 Training: 60-Ply 16-Dim Global Linear Gate (Frozen After Quantize)
# ============================================================================
def train_stage25(
    bridge: P2C64Bridge,
    tier1_w: np.ndarray,
    train_data: np.ndarray,
    val_data: np.ndarray,
    args: argparse.Namespace,
    device: torch.device,
    ckpt_path: Path,
) -> Tuple[np.ndarray, np.ndarray]:
    print("\n=== [Stage 2.5] Training 60-Ply 16-Dim Global Mobility+Geom Linear Gate ===")
    # Parameterize in scaled-down space (w_int = w_p * 1024, b_int = b_p * 1024) for clean AdamW conditioning
    w_p = nn.Parameter(torch.zeros((NUM_PLIES, WACA_F25), dtype=torch.float32, device=device))
    b_p = nn.Parameter(torch.zeros((NUM_PLIES,), dtype=torch.float32, device=device))
    opt = torch.optim.AdamW([w_p, b_p], lr=args.s25_lr, weight_decay=1e-4)

    start_ep = 0
    if args.resume and ckpt_path.exists():
        ck = torch.load(ckpt_path, map_location=device, weights_only=True)
        w_p.data.copy_(ck["w_p"])
        b_p.data.copy_(ck["b_p"])
        opt.load_state_dict(ck["opt"])
        start_ep = int(ck["epoch"]) + 1
        print(f"  Resumed Stage 2.5 from epoch {start_ep}")

    n = len(train_data)
    bs = args.batch_size
    total_steps = max(1, args.s25_epochs * math.ceil(n / bs))
    step_idx = start_ep * math.ceil(n / bs)
    rng = np.random.default_rng(args.seed + 101)

    for ep in range(start_ep, args.s25_epochs):
        t0 = time.perf_counter()
        perm = rng.permutation(n)
        qat = (ep == args.s25_epochs - 1)
        loss_sum = 0.0
        n_batches = 0

        for s in range(0, n, bs):
            e = min(n, s + bs)
            idx = perm[s:e]
            sl = train_data[idx]
            ext = bridge.extract_waca_g2_batch(
                sl["black"], sl["white"], sl["stm"], sl["ply"],
                tier1_w, enable_aug=False, need_stage3=False,
            )

            ply_t = torch.from_numpy(sl["ply"].astype(np.int64)).to(device)
            v1_t = torch.from_numpy(ext["v1"].astype(np.float32)).to(device)
            f25_t = torch.from_numpy(ext["f25"].astype(np.float32)).to(device)
            y_t = torch.from_numpy(sl["score"].astype(np.float32)).to(device)

            # Cosine LR schedule
            prog = min(1.0, step_idx / float(total_steps))
            lr = args.s25_lr * (0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * prog)))
            for pg in opt.param_groups:
                pg["lr"] = lr
            step_idx += 1

            opt.zero_grad(set_to_none=True)
            if qat:
                wq = ste_round_clamp(w_p[ply_t] * 1024.0, -WACA_S25_W_MAX, WACA_S25_W_MAX)
                bq = ste_round_clamp(b_p[ply_t] * (128.0 * 1024.0), -WACA_S25_B_MAX, WACA_S25_B_MAX)
                acc = bq + (wq * f25_t).sum(dim=1)
                r25 = torch.clamp(ste_sym_rshift_int(acc, WACA_S25_SHIFT), -WACA_MAX_RES, WACA_MAX_RES)
            else:
                acc = (b_p[ply_t] * 128.0) + (w_p[ply_t] * f25_t).sum(dim=1)
                r25 = torch.clamp(acc, -WACA_MAX_RES, WACA_MAX_RES)

            pred_discs = (v1_t + r25) / P2C_DEFAULT_SCALE
            w_smp = compute_sample_weights(ply_t, v1_t / P2C_DEFAULT_SCALE)
            loss = weighted_huber_loss(pred_discs, y_t, w_smp, delta=4.0)
            loss.backward()
            opt.step()

            apply_axis0_smoothness_(w_p.data, args.s25_smooth)
            apply_axis0_smoothness_(b_p.data, args.s25_smooth)
            w_p.data.clamp_(-WACA_S25_W_MAX / 1024.0, WACA_S25_W_MAX / 1024.0)
            b_p.data.clamp_(-WACA_S25_B_MAX / (128.0 * 1024.0), WACA_S25_B_MAX / (128.0 * 1024.0))

            loss_sum += float(loss.item())
            n_batches += 1

        # Quick quantized Val RMSE check
        s25_w_q = np.clip(np.round(w_p.detach().cpu().numpy() * 1024.0), -WACA_S25_W_MAX, WACA_S25_W_MAX).astype(np.int16)
        s25_b_q = np.clip(np.round(b_p.detach().cpu().numpy() * (128.0 * 1024.0)), -WACA_S25_B_MAX, WACA_S25_B_MAX).astype(np.int32)
        tmp_m = make_empty_waca_g2(mlp_hidden=args.mlp_hidden)
        tmp_m.tier1_w = tier1_w
        tmp_m.s25_w = s25_w_q
        tmp_m.s25_b = s25_b_q
        ev = evaluate_deployed_waca_g2(bridge, tmp_m, val_data, batch_size=max(bs, 16384))
        dt = time.perf_counter() - t0
        m1 = ev["stage1_raw64"]
        m25 = ev["stage25"]
        print(
            f"  [S2.5 Ep {ep+1}/{args.s25_epochs}{' QAT' if qat else '    '}] "
            f"loss={loss_sum/max(1,n_batches):.4f} | "
            f"Val RMSE: {m1['rmse_all']:.4f} -> {m25['rmse_all']:.4f} "
            f"(0-19:{m25['rmse_0_19']:.4f} 20-39:{m25['rmse_20_39']:.4f} 40-59:{m25['rmse_40_59']:.4f}) | {dt:.1f}s"
        )
        torch.save({"epoch": ep, "w_p": w_p.detach().cpu(), "b_p": b_p.detach().cpu(), "opt": opt.state_dict()}, ckpt_path)

    s25_w_final = np.clip(np.round(w_p.detach().cpu().numpy() * 1024.0), -WACA_S25_W_MAX, WACA_S25_W_MAX).astype(np.int16)
    s25_b_final = np.clip(np.round(b_p.detach().cpu().numpy() * (128.0 * 1024.0)), -WACA_S25_B_MAX, WACA_S25_B_MAX).astype(np.int32)
    return s25_w_final, s25_b_final


# ============================================================================
# Stage 3 Training: 20-Block EO Key-14 SCReLU + 60-Ply 24->H->1 Integer MLP
# ============================================================================
class WacaStage3Module(nn.Module):
    def __init__(self, mlp_hidden: int, key_orb_rep: np.ndarray, device: torch.device) -> None:
        super().__init__()
        self.H = int(mlp_hidden)
        self.register_buffer("key_orb_rep", torch.from_numpy(key_orb_rep).long().to(device))
        self.register_buffer("ply_to_eo", torch.from_numpy(PLY_TO_EO_BLK).long().to(device))

        # 20-block EO Key-14 4ch embeddings (in float units; int8 = round(h * 32.0))
        self.h_emb = nn.Parameter(torch.randn((WACA_EO_BLOCKS, KEY_RAW_ENTRIES, 4), device=device) * 0.05)

        # 60-ply Key-14 SCReLU head (O(1) scaled; Zero-Init beta -> starts at exact Stage 2.5!)
        #   mu_int   = round(sc_mu * 32.0)   in [-4096, 4096]
        #   beta_int = round(sc_beta * 100.0) in [-2047, 2047]
        self.sc_mu = nn.Parameter(torch.full((NUM_PLIES, 3, 4), 0.5, dtype=torch.float32, device=device))
        self.sc_beta = nn.Parameter(torch.zeros((NUM_PLIES, 3, 4), dtype=torch.float32, device=device))

        # 60-ply 24->H->1 SCReLU MLP (O(1) scaled; Zero-Init w2, b2 -> starts at exact Stage 2.5!)
        #   w1_int = round(mlp_w1 * 64.0)         in [-127, 127]
        #   b1_int = round(mlp_b1 * 16384.0)      in [-1048576, 1048576] (init 0.5 -> +8192 -> u~16)
        #   w2_int = round(mlp_w2 * 1024.0)       in [-4095, 4095]
        #   b2_int = round(mlp_b2 * 131072.0)     in [-1048576, 1048576] (discs!)
        self.mlp_w1 = nn.Parameter(torch.randn((NUM_PLIES, self.H, WACA_F3), device=device) * (16.0 / 64.0))
        self.mlp_b1 = nn.Parameter(torch.full((NUM_PLIES, self.H), 0.5, dtype=torch.float32, device=device))
        self.mlp_w2 = nn.Parameter(torch.zeros((NUM_PLIES, self.H), dtype=torch.float32, device=device))
        self.mlp_b2 = nn.Parameter(torch.zeros((NUM_PLIES,), dtype=torch.float32, device=device))

    def forward(
        self,
        f25: torch.Tensor,      # [B, 16] float (exact integer values)
        f3_geom: torch.Tensor,  # [B, 4]  float (exact integer values)
        key_ptm: torch.Tensor,  # [B, 14] long
        ply: torch.Tensor,      # [B]     long
        r25_fixed: torch.Tensor,  # [B]   float (exact frozen int32 r25)
        qat: bool = False,
    ) -> torch.Tensor:
        eo_blk = self.ply_to_eo[ply]                    # [B]
        key_rep = self.key_orb_rep[key_ptm]             # [B, 14]
        h_raw = self.h_emb[eo_blk[:, None], key_rep]    # [B, 14, 4]
        if qat:
            h = ste_round_clamp(h_raw * EMB_SCALE, -127.0, 127.0)
            mu_p = ste_round_clamp(self.sc_mu[ply] * EMB_SCALE, -4096.0, 4096.0)
            beta_p = ste_round_clamp(self.sc_beta[ply] * 100.0, -SCRELU_BETA_MAX, SCRELU_BETA_MAX)
        else:
            h = torch.clamp(h_raw * EMB_SCALE, -127.0, 127.0)
            mu_p = torch.clamp(self.sc_mu[ply] * EMB_SCALE, -4096.0, 4096.0)
            beta_p = torch.clamp(self.sc_beta[ply] * 100.0, -SCRELU_BETA_MAX, SCRELU_BETA_MAX)

        # Group 0: 4 Edge+2X + Edge+C + 2 adjacent corners
        s0 = torch.zeros((f25.shape[0], 4), dtype=f25.dtype, device=f25.device)
        for r in range(4):
            c1 = (r + 1) & 3
            u = torch.clamp(h[:, 4 + r] + h[:, 8 + r] + h[:, r] + h[:, c1] + mu_p[:, 0], 0.0, float(SCRELU_Q))
            s0 = s0 + u * u

        # Group 1: 2 Diagonals + 2 opposite corners
        s1 = torch.zeros((f25.shape[0], 4), dtype=f25.dtype, device=f25.device)
        for d in range(2):
            u = torch.clamp(h[:, 12 + d] + h[:, d] + h[:, d + 2] + mu_p[:, 1], 0.0, float(SCRELU_Q))
            s1 = s1 + u * u

        # Group 2: Global 14-pattern sum
        ug = torch.clamp(h.sum(dim=1) + mu_p[:, 2], 0.0, float(SCRELU_Q))
        s2 = ug * ug

        sc_acc = (beta_p[:, 0] * s0 + beta_p[:, 1] * s1 + beta_p[:, 2] * s2).sum(dim=1)
        if qat:
            r_key = ste_sym_rshift_int(sc_acc, SCRELU_SHIFT)
            f_key = torch.clamp(ste_floor_rshift_nonneg(s0 + s1 + s2, 8), -float(WACA_FEAT_CLAMP), float(WACA_FEAT_CLAMP))
        else:
            r_key = sc_acc / float(1 << SCRELU_SHIFT)
            f_key = torch.clamp((s0 + s1 + s2) / 256.0, -float(WACA_FEAT_CLAMP), float(WACA_FEAT_CLAMP))

        f_all = torch.cat([f25, f3_geom, f_key], dim=1)  # [B, 24]

        if qat:
            w1 = ste_round_clamp(self.mlp_w1[ply] * 64.0, -WACA_MLP_W1_MAX, WACA_MLP_W1_MAX)
            b1 = ste_round_clamp(self.mlp_b1[ply] * 16384.0, -WACA_MLP_B1_MAX, WACA_MLP_B1_MAX)
            w2 = ste_round_clamp(self.mlp_w2[ply] * 1024.0, -WACA_MLP_W2_MAX, WACA_MLP_W2_MAX)
            b2 = ste_round_clamp(self.mlp_b2[ply] * 131072.0, -WACA_MLP_B2_MAX, WACA_MLP_B2_MAX)
            z = b1 + torch.einsum("bhf,bf->bh", w1, f_all)
            z_pos = torch.clamp(z, min=0.0)
            u_mlp = torch.clamp(ste_floor_rshift_nonneg(z_pos, WACA_MLP_S1), 0.0, float(SCRELU_Q))
            acc2 = b2 + (w2 * (u_mlp * u_mlp)).sum(dim=1)
            r_mlp = ste_sym_rshift_int(acc2, WACA_MLP_S2)
        else:
            w1 = torch.clamp(self.mlp_w1[ply] * 64.0, -WACA_MLP_W1_MAX, WACA_MLP_W1_MAX)
            b1 = torch.clamp(self.mlp_b1[ply] * 16384.0, -WACA_MLP_B1_MAX, WACA_MLP_B1_MAX)
            w2 = torch.clamp(self.mlp_w2[ply] * 1024.0, -WACA_MLP_W2_MAX, WACA_MLP_W2_MAX)
            b2 = torch.clamp(self.mlp_b2[ply] * 131072.0, -WACA_MLP_B2_MAX, WACA_MLP_B2_MAX)
            z = b1 + torch.einsum("bhf,bf->bh", w1, f_all)
            u_mlp = torch.clamp(z / float(1 << WACA_MLP_S1), 0.0, float(SCRELU_Q))
            acc2 = b2 + (w2 * (u_mlp * u_mlp)).sum(dim=1)
            r_mlp = acc2 / float(1 << WACA_MLP_S2)

        return torch.clamp(r25_fixed + r_mlp + r_key, -float(WACA_MAX_RES), float(WACA_MAX_RES))

    @torch.no_grad()
    def export_quantized_to_model(
        self,
        tier1_w: np.ndarray,
        s25_w: np.ndarray,
        s25_b: np.ndarray,
        orbits: Dict[str, np.ndarray],
        score_scale: float = P2C_DEFAULT_SCALE,
    ) -> WacaG2Model:
        m = make_empty_waca_g2(mlp_hidden=self.H, score_scale=score_scale)
        m.tier1_w = tier1_w.copy()
        m.s25_w = s25_w.copy()
        m.s25_b = s25_b.copy()
        m.h_emb = np.clip(np.round(self.h_emb.detach().cpu().numpy() * EMB_SCALE), -127, 127).astype(np.int8)
        m.screlu_mu = np.clip(np.round(self.sc_mu.detach().cpu().numpy() * EMB_SCALE), -4096, 4096).astype(np.int16)
        m.screlu_beta = np.clip(np.round(self.sc_beta.detach().cpu().numpy() * 100.0), -SCRELU_BETA_MAX, SCRELU_BETA_MAX).astype(np.int16)
        m.mlp_w1 = np.clip(np.round(self.mlp_w1.detach().cpu().numpy() * 64.0), -WACA_MLP_W1_MAX, WACA_MLP_W1_MAX).astype(np.int8)
        m.mlp_b1 = np.clip(np.round(self.mlp_b1.detach().cpu().numpy() * 16384.0), -WACA_MLP_B1_MAX, WACA_MLP_B1_MAX).astype(np.int32)
        m.mlp_w2 = np.clip(np.round(self.mlp_w2.detach().cpu().numpy() * 1024.0), -WACA_MLP_W2_MAX, WACA_MLP_W2_MAX).astype(np.int16)
        m.mlp_b2 = np.clip(np.round(self.mlp_b2.detach().cpu().numpy() * 131072.0), -WACA_MLP_B2_MAX, WACA_MLP_B2_MAX).astype(np.int32)
        return symmetrize_waca_g2_model(m, orbits)


def train_stage3(
    bridge: P2C64Bridge,
    orbits: Dict[str, np.ndarray],
    tier1_w: np.ndarray,
    s25_w: np.ndarray,
    s25_b: np.ndarray,
    train_data: np.ndarray,
    val_data: np.ndarray,
    args: argparse.Namespace,
    device: torch.device,
    ckpt_path: Path,
    out_p2c: Path,
) -> Tuple[WacaG2Model, Dict[str, Dict[str, float]]]:
    total_ep = args.s3_epochs + args.s3_qat_epochs
    print(
        f"\n=== [Stage 3] Training 20-Block EO Key-14 SCReLU + 60-Ply 24->{args.mlp_hidden}->1 MLP "
        f"({args.s3_epochs} float + {args.s3_qat_epochs} QAT epochs) ==="
    )

    mod = WacaStage3Module(args.mlp_hidden, orbits["key_orb_rep"], device)
    opt = torch.optim.AdamW(
        [
            {"params": [mod.h_emb], "lr": args.s3_emb_lr, "weight_decay": 1e-4},
            {"params": [mod.sc_mu, mod.sc_beta], "lr": args.s3_mlp_lr, "weight_decay": 1e-4},
            {"params": [mod.mlp_w1, mod.mlp_b1, mod.mlp_w2, mod.mlp_b2], "lr": args.s3_mlp_lr, "weight_decay": 1e-4},
        ]
    )

    start_ep = 0
    best_rmse = 1e9
    best_model: WacaG2Model | None = None
    best_metrics: Dict[str, Dict[str, float]] | None = None

    if args.resume and ckpt_path.exists():
        ck = torch.load(ckpt_path, map_location=device, weights_only=True)
        mod.load_state_dict(ck["mod"])
        opt.load_state_dict(ck["opt"])
        start_ep = int(ck["epoch"]) + 1
        best_rmse = float(ck.get("best_rmse", 1e9))
        print(f"  Resumed Stage 3 from epoch {start_ep} (best_rmse={best_rmse:.4f})")

    n = len(train_data)
    bs = args.batch_size
    steps_per_ep = math.ceil(n / bs)
    total_steps = max(1, total_ep * steps_per_ep)
    step_idx = start_ep * steps_per_ep
    rng = np.random.default_rng(args.seed + 303)

    s25_w_i32 = s25_w.astype(np.int32)
    s25_b_i32 = s25_b.astype(np.int32)

    for ep in range(start_ep, total_ep):
        t0 = time.perf_counter()
        perm = rng.permutation(n)
        qat = (ep >= args.s3_epochs)
        loss_sum = 0.0
        n_batches = 0

        for b_idx, s in enumerate(range(0, n, bs)):
            e = min(n, s + bs)
            idx = perm[s:e]
            sl = train_data[idx]
            aug_seed = (args.seed * 1000003 + ep * 9176 + b_idx * 131 + 7) & 0xFFFFFFFFFFFFFFFF
            ext = bridge.extract_waca_g2_batch(
                sl["black"], sl["white"], sl["stm"], sl["ply"],
                tier1_w, aug_seed=aug_seed, enable_aug=True, need_stage3=True,
            )

            # Exact frozen Stage 2.5 integer residual (bit-exact with C++!)
            ply_np = sl["ply"].astype(np.int32)
            acc25 = s25_b_i32[ply_np] + (s25_w_i32[ply_np] * ext["f25"]).sum(axis=1, dtype=np.int32)
            add25 = (acc25 >> np.int32(31)) & np.int32((1 << WACA_S25_SHIFT) - 1)
            r25_np = np.clip((acc25 + add25) >> np.int32(WACA_S25_SHIFT), -WACA_MAX_RES, WACA_MAX_RES).astype(np.int32)

            ply_t = torch.from_numpy(ply_np.astype(np.int64)).to(device)
            v1_t = torch.from_numpy(ext["v1"].astype(np.float32)).to(device)
            f25_t = torch.from_numpy(ext["f25"].astype(np.float32)).to(device)
            f3g_t = torch.from_numpy(ext["f3_geom"].astype(np.float32)).to(device)
            key_t = torch.from_numpy(ext["key_ptm"].astype(np.int64)).to(device)
            r25_t = torch.from_numpy(r25_np.astype(np.float32)).to(device)
            y_t = torch.from_numpy(sl["score"].astype(np.float32)).to(device)

            # Cosine LR schedule (lower LR during QAT fine-tuning)
            prog = min(1.0, step_idx / float(total_steps))
            cos_mult = 0.08 + 0.92 * 0.5 * (1.0 + math.cos(math.pi * prog))
            if qat:
                cos_mult *= 0.35
            opt.param_groups[0]["lr"] = args.s3_emb_lr * cos_mult
            opt.param_groups[1]["lr"] = args.s3_mlp_lr * cos_mult
            opt.param_groups[2]["lr"] = args.s3_mlp_lr * cos_mult
            step_idx += 1

            opt.zero_grad(set_to_none=True)
            r_tot = mod(f25_t, f3g_t, key_t, ply_t, r25_t, qat=qat)
            pred_discs = (v1_t + r_tot) / P2C_DEFAULT_SCALE
            w_smp = compute_sample_weights(ply_t, v1_t / P2C_DEFAULT_SCALE)
            loss = weighted_huber_loss(pred_discs, y_t, w_smp, delta=4.0)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(mod.parameters(), max_norm=5.0)
            opt.step()

            # Smoothness regularization along adjacent plies & same-parity EO blocks
            apply_axis0_smoothness_(mod.sc_mu.data, args.s3_ply_smooth)
            apply_axis0_smoothness_(mod.sc_beta.data, args.s3_ply_smooth)
            apply_axis0_smoothness_(mod.mlp_w1.data, args.s3_ply_smooth)
            apply_axis0_smoothness_(mod.mlp_b1.data, args.s3_ply_smooth)
            apply_axis0_smoothness_(mod.mlp_w2.data, args.s3_ply_smooth)
            apply_axis0_smoothness_(mod.mlp_b2.data, args.s3_ply_smooth)
            if (b_idx + 1) % 8 == 0 or e == n:
                apply_eo_block_smoothness_(mod.h_emb.data, args.s3_blk_smooth)

            mod.h_emb.data.clamp_(-127.0 / EMB_SCALE, 127.0 / EMB_SCALE)
            mod.sc_mu.data.clamp_(-4096.0 / EMB_SCALE, 4096.0 / EMB_SCALE)
            mod.sc_beta.data.clamp_(-float(SCRELU_BETA_MAX) / 100.0, float(SCRELU_BETA_MAX) / 100.0)
            mod.mlp_w1.data.clamp_(-WACA_MLP_W1_MAX / 64.0, WACA_MLP_W1_MAX / 64.0)
            mod.mlp_b1.data.clamp_(-WACA_MLP_B1_MAX / 16384.0, WACA_MLP_B1_MAX / 16384.0)
            mod.mlp_w2.data.clamp_(-WACA_MLP_W2_MAX / 1024.0, WACA_MLP_W2_MAX / 1024.0)
            mod.mlp_b2.data.clamp_(-WACA_MLP_B2_MAX / 131072.0, WACA_MLP_B2_MAX / 131072.0)

            loss_sum += float(loss.item())
            n_batches += 1

        # Deployed bit-exact quantized evaluation on val_data
        q_model = mod.export_quantized_to_model(tier1_w, s25_w, s25_b, orbits)
        ev = evaluate_deployed_waca_g2(bridge, q_model, val_data, batch_size=max(bs, 16384))
        m1 = ev["stage1_raw64"]
        m25 = ev["stage25"]
        m3 = ev["stage3_full"]
        dt = time.perf_counter() - t0
        tag = "QAT" if qat else "FLT"
        print(
            f"  [S3 Ep {ep+1:2d}/{total_ep:2d} {tag}] loss={loss_sum/max(1,n_batches):.4f} | "
            f"Val RMSE: S1={m1['rmse_all']:.4f} S2.5={m25['rmse_all']:.4f} S3={m3['rmse_all']:.4f} "
            f"(d={m3['rmse_all']-m1['rmse_all']:+.4f} | "
            f"0-19:{m3['rmse_0_19']:.4f} 20-39:{m3['rmse_20_39']:.4f} 40-59:{m3['rmse_40_59']:.4f}) | {dt:.1f}s"
        )

        if m3["rmse_all"] < best_rmse:
            best_rmse = m3["rmse_all"]
            best_model = q_model
            best_metrics = ev
            save_waca_g2(out_p2c, best_model)

        torch.save(
            {"epoch": ep, "mod": mod.state_dict(), "opt": opt.state_dict(), "best_rmse": best_rmse},
            ckpt_path,
        )

    if best_model is None or best_metrics is None:
        best_model = mod.export_quantized_to_model(tier1_w, s25_w, s25_b, orbits)
        best_metrics = evaluate_deployed_waca_g2(bridge, best_model, val_data, batch_size=max(bs, 16384))
        save_waca_g2(out_p2c, best_model)

    return best_model, best_metrics


def main() -> None:
    ap = argparse.ArgumentParser(description="Train P2C-64 v3.0 (WACA-G2) on frozen raw_ntuple64.p2c")
    ap.add_argument("--train-bin", type=Path, required=True)
    ap.add_argument("--val-bin", type=Path, required=True)
    ap.add_argument("--raw-p2c", type=Path, required=True, help="Frozen Phase-1 raw_ntuple64.p2c (0-min Phase 1!)")
    ap.add_argument("--work-dir", type=Path, required=True)
    ap.add_argument("--mlp-hidden", type=int, default=16, choices=[16, 24])
    ap.add_argument("--max-train-samples", type=int, default=0, help="0 = use all train samples")
    ap.add_argument("--max-val-samples", type=int, default=0, help="0 = use all val samples")
    ap.add_argument("--batch-size", type=int, default=16384)
    ap.add_argument("--s25-epochs", type=int, default=3)
    ap.add_argument("--s25-lr", type=float, default=2.0e-2)
    ap.add_argument("--s25-smooth", type=float, default=1.0e-3)
    ap.add_argument("--s3-epochs", type=int, default=6)
    ap.add_argument("--s3-qat-epochs", type=int, default=2)
    ap.add_argument("--s3-emb-lr", type=float, default=8.0e-3)
    ap.add_argument("--s3-mlp-lr", type=float, default=4.0e-3)
    ap.add_argument("--s3-ply-smooth", type=float, default=8.0e-4)
    ap.add_argument("--s3-blk-smooth", type=float, default=4.0e-3)
    ap.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"])
    ap.add_argument("--seed", type=int, default=20261126)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if (args.device == "auto" and torch.cuda.is_available()) or args.device == "cuda" else "cpu")
    args.work_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = args.work_dir / "checkpoints_v30"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    print(f"=== P2C-64 v3.0 (WACA-G2, H={args.mlp_hidden}) Training Pipeline ===")
    print(f"  Device      : {device}")
    print(f"  Raw Tier-1  : {args.raw_p2c}")
    print(f"  Train / Val : {args.train_bin} / {args.val_bin}")
    print(f"  Work Dir    : {args.work_dir}")

    raw_model = load_p2c(args.raw_p2c)
    if not isinstance(raw_model, RawNTupleModel):
        raise ValueError(f"--raw-p2c must be a RAW_NTUPLE_64 model, got {type(raw_model).__name__}")
    tier1_w = raw_model.w_ply

    bridge = P2C64Bridge()
    orbits = bridge.get_orbit_tables()

    t_load = time.perf_counter()
    train_data = load_packed_dataset(args.train_bin, args.max_train_samples)
    val_data = load_packed_dataset(args.val_bin, args.max_val_samples)
    print(f"  Loaded train={len(train_data):,}, val={len(val_data):,} in {time.perf_counter()-t_load:.2f}s")

    # Stage 2.5
    s25_ckpt = ckpt_dir / "stage25.pt"
    s25_w, s25_b = train_stage25(bridge, tier1_w, train_data, val_data, args, device, s25_ckpt)

    # Stage 3
    s3_ckpt = ckpt_dir / f"stage3_h{args.mlp_hidden}.pt"
    out_p2c = args.work_dir / "waca_g2.p2c"
    _, metrics = train_stage3(
        bridge, orbits, tier1_w, s25_w, s25_b, train_data, val_data, args, device, s3_ckpt, out_p2c
    )

    # Save metrics_v30.json and metrics_v30.csv
    metrics_json = args.work_dir / "metrics_v30.json"
    with open(metrics_json, "w") as f:
        json.dump({"mlp_hidden": args.mlp_hidden, "n_train": len(train_data), "n_val": len(val_data), **metrics}, f, indent=2)

    metrics_csv = args.work_dir / "metrics_v30.csv"
    with open(metrics_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["stage", "rmse_all", "mae_all", "rmse_0_19", "rmse_20_39", "rmse_40_59"])
        for st in ("stage1_raw64", "stage25", "stage3_full"):
            m = metrics[st]
            w.writerow([st, f"{m['rmse_all']:.6f}", f"{m['mae_all']:.6f}", f"{m['rmse_0_19']:.6f}", f"{m['rmse_20_39']:.6f}", f"{m['rmse_40_59']:.6f}"])

    print("\n=== Final Deployed Quantized Val Metrics (Bit-Exact with C++) ===")
    for st in ("stage1_raw64", "stage25", "stage3_full"):
        m = metrics[st]
        print(
            f"  {st:<14} | RMSE_all={m['rmse_all']:.4f} MAE_all={m['mae_all']:.4f} | "
            f"0-19={m['rmse_0_19']:.4f} | 20-39={m['rmse_20_39']:.4f} | 40-59={m['rmse_40_59']:.4f}"
        )
    rs = metrics["residual_stats"]
    print(
        f"  Residual quantiles (discs): |R25| p90={rs['r25_p90_all']:.2f} p95={rs['r25_p95_all']:.2f} p99={rs['r25_p99_all']:.2f} | "
        f"|R3-R25| p95={rs['r3_delta_p95_all']:.2f} (mid={rs['r3_delta_p95_mid']:.2f}) p99={rs['r3_delta_p99_all']:.2f}"
    )
    print(f"  Saved model : {out_p2c} ({out_p2c.stat().st_size / (1024*1024):.2f} MiB)")
    print(f"  Saved stats : {metrics_json}, {metrics_csv}")


if __name__ == "__main__":
    main()
