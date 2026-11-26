#!/usr/bin/env python3
"""P2C-64 v2.0 (Full PTM) End-to-End PyTorch Training Pipeline (CUDA T4/L4/A100 & CPU).

Trains and exports all 5 models to .p2c v2.0 format:
  Phase 1 (SINGLE run -> exports bit-identical RAW_NTUPLE_64 & LINEAR_64, 0 handicap!):
    - 64-pattern x 60-ply PTM linear weights W + adjacent-ply smoothness reg
  Phase 2 (Zero-Init Residual on frozen quantized W + boundary-cliff smoothing):
    - CAND_A_PTM_SCRELU4 : 4-ch PTM SCReLU + 60-ply head (inter-ply & inter-block smoothing + QAT)
    - CAND_B_PTM_3W3B    : All 8 codes unlocked + 60-ply Macro-LUT (inter-block logit smoothing
                           -> inter-ply smoothed MM Empirical Bayes Convex Refit -> Exact EM-LUT)
    - CAND_C_PTM_HYBRID  : Phase 2-1 2-ch PTM SCReLU (frozen quantized)
                           -> Phase 2-2 60-ply 3way-3bit SCONE-D pipeline
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
    EDGE_OFF,
    EDGE_RAW_SIZE,
    KEY_RAW_ENTRIES,
    NONKEY_RAW_ENTRIES,
    NUM_BLOCKS,
    NUM_PLIES,
    PACKED_SAMPLE_DTYPE,
    PLIES_PER_BLOCK,
    TOTAL_RAW_ENTRIES,
    P2C64Bridge,
)
from p2c64_io import (
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
    raw_to_linear_p2c64,
    save_p2c64,
    save_raw_p2c,
    symmetrize_p2c64_model,
    symmetrize_raw_model,
)

EMB_SCALE = 32.0
SCRELU_QF = SCRELU_Q / EMB_SCALE  # 127 / 32 = 3.96875


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
    denom = float(1 << shift)
    y = torch.trunc(x / denom)
    return (x / denom) + (y - x / denom).detach()


@torch.no_grad()
def apply_axis0_smoothness_(w: torch.Tensor, alpha: float) -> None:
    """Unconditionally stable decoupled proximal Laplacian diffusion along axis 0 (plies or blocks)."""
    if alpha <= 0.0 or w.shape[0] < 2:
        return
    a = min(0.20, float(alpha))
    w_prev = w.clone()
    w[0].add_(w_prev[1] - w_prev[0], alpha=a)
    w[-1].add_(w_prev[-2] - w_prev[-1], alpha=a)
    if w.shape[0] > 2:
        w[1:-1].add_(w_prev[:-2] + w_prev[2:] - 2.0 * w_prev[1:-1], alpha=a)


# ============================================================================
# Macro-LUT D4 Mirror Orbit Precomputation (PTM: mirror only, all 512 active!)
# ============================================================================
def build_macro_orbit_map_v2(swap_mode: str) -> Tuple[np.ndarray, int]:
    """Returns (orb_id[512] in [0..n_orb-1], n_orb) under pure D4 mirror swap."""
    orb_id = np.full(512, -1, dtype=np.int32)
    cur = 0
    for z0 in range(8):
        for z1 in range(8):
            for z2 in range(8):
                k = (z0 << 6) | (z1 << 3) | z2
                if orb_id[k] != -1:
                    continue
                if swap_mode == "cw":
                    mk = (z0 << 6) | (z2 << 3) | z1
                elif swap_mode == "ea":
                    mk = (z1 << 6) | (z0 << 3) | z2
                else:
                    mk = (z2 << 6) | (z1 << 3) | z0
                orb_id[k] = cur
                orb_id[mk] = cur
                cur += 1
    return orb_id, cur


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
            ext = bridge.extract_batch_ptm(sl["black"], sl["white"], sl["stm"], enable_aug=False, need_p2c=False, need_raw=True)
            sc = eval_raw_ntuple64_int_batch(model, ext["raw_ptm"], sl["ply"])
        else:
            ext = bridge.extract_batch_ptm(sl["black"], sl["white"], sl["stm"], enable_aug=False, need_p2c=True, need_raw=False)
            sc = eval_p2c64_int_batch(model, ext["key_ptm"], ext["nonkey_ptm"], sl["ply"])
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
# Phase 1: SINGLE PTM 64-Pattern Training -> Bit-Identical RAW + LINEAR Export
# ============================================================================
def train_phase1_ptm_shared(
    bridge: P2C64Bridge,
    orbits: Dict[str, np.ndarray],
    train_data: np.ndarray,
    val_data: np.ndarray,
    args: argparse.Namespace,
    device: torch.device,
    raw_p2c: Path,
    lin_p2c: Path,
    ckpt_path: Path,
) -> Tuple[P2C64Model, Dict[str, float]]:
    print("\n=== [Phase 1] Training Shared 60-Ply PTM 64-Pattern Skeleton (RAW == LINEAR) ===")
    raw_rep = torch.from_numpy(orbits["raw_orb_rep"]).long().to(device)
    W = nn.Parameter(torch.zeros((NUM_PLIES, TOTAL_RAW_ENTRIES), dtype=torch.float32, device=device))
    opt = torch.optim.AdamW([W], lr=args.p1_lr, weight_decay=0.0)

    start_epoch = 0
    if args.resume and ckpt_path.exists():
        ck = torch.load(ckpt_path, map_location=device, weights_only=True)
        W.data.copy_(ck["W"])
        opt.load_state_dict(ck["opt"])
        start_epoch = int(ck["epoch"]) + 1
        print(f"  Resumed Phase 1 from epoch {start_epoch}")

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
            ext = bridge.extract_batch_ptm(
                sl["black"], sl["white"], sl["stm"],
                aug_seed=int(rng.integers(0, 2**62)), enable_aug=True,
                need_p2c=False, need_raw=True,
            )
            rp = torch.from_numpy(ext["raw_ptm"]).long().to(device, non_blocking=True)
            ply_t = torch.from_numpy(sl["ply"].astype(np.int64)).to(device, non_blocking=True)
            y_t = torch.from_numpy(sl["score"].astype(np.float32)).to(device, non_blocking=True)

            pred = W[ply_t[:, None], raw_rep[rp]].sum(dim=1)
            loss = F.smooth_l1_loss(pred, y_t, beta=args.huber_beta)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            apply_axis0_smoothness_(W.data, args.smooth_lambda * lr_now * 50.0)
            total_loss += float(loss.item())
            steps += 1

        torch.save({"epoch": epoch, "W": W.detach().cpu(), "opt": opt.state_dict()}, ckpt_path)
        print(f"  [Phase 1] Epoch {epoch+1}/{args.p1_epochs} | lr={lr_now:.4f} | huber={total_loss/max(1,steps):.4f} | {time.perf_counter()-t0:.1f}s")

    # Bake ONCE and export to BOTH raw_ntuple64.p2c and linear64.p2c!
    w_q = np.clip(np.round(W.detach().cpu().numpy() * args.score_scale), -32767, 32767).astype(np.int16)
    raw_model = symmetrize_raw_model(RawNTupleModel(w_ply=w_q, score_scale=args.score_scale), orbits)
    lin_model = raw_to_linear_p2c64(raw_model)

    save_raw_p2c(raw_p2c, raw_model)
    save_p2c64(lin_p2c, lin_model)
    metrics = evaluate_deployed_p2c(bridge, lin_model, val_data, args.val_batch_size)
    print(f"  [Phase 1] Exported IDENTICAL weights -> {raw_p2c.name} & {lin_p2c.name}")
    print(f"            Val RMSE={metrics['rmse_all']:.4f} MAE={metrics['mae_all']:.4f}")
    return lin_model, metrics


# ============================================================================
# Frozen Quantized LINEAR_64 PTM Evaluator (Exact Deployed Baseline)
# ============================================================================
class FrozenLinearPTMEvaluator:
    def __init__(self, lin_model: P2C64Model, device: torch.device) -> None:
        self.scale = float(lin_model.score_scale)
        wk = lin_model.key_blocks["w"].transpose(0, 2, 1).reshape(NUM_PLIES, KEY_RAW_ENTRIES).astype(np.float32)
        self.W_key = torch.from_numpy(wk).to(device)
        self.W_nk = torch.from_numpy(lin_model.nonkey_w.astype(np.float32)).to(device)

    @torch.no_grad()
    def eval_disc(self, key_ptm: torch.Tensor, nk_ptm: torch.Tensor, ply_t: torch.Tensor) -> torch.Tensor:
        s = self.W_key[ply_t[:, None], key_ptm].sum(dim=1)
        s = s + self.W_nk[ply_t[:, None], nk_ptm].sum(dim=1)
        return s / self.scale


# ============================================================================
# PTM SCReLU Residual Module (4-ch for CAND_A, 2-ch for CAND_C Phase 2-1)
#   - 60-ply independent smooth head (mu[60, 3, K], beta[60, 3, K])
#   - 10-block Key-14 embeddings (h[10, 91854, K]) with inter-block smoothing
# ============================================================================
class PTMSCReLUResidual(nn.Module):
    def __init__(
        self,
        k_channels: int,
        orbits: Dict[str, np.ndarray],
        mu_init: float,
        score_scale: float,
        device: torch.device,
    ) -> None:
        super().__init__()
        assert k_channels in (2, 4)
        self.K = k_channels
        self.score_scale = float(score_scale)
        self.beta_scale = self.score_scale * float(1 << SCRELU_SHIFT) / (EMB_SCALE * EMB_SCALE)
        self.register_buffer("k_rep", torch.from_numpy(orbits["key_orb_rep"]).long().to(device))

        self.h_emb = nn.Parameter(torch.randn((NUM_BLOCKS, KEY_RAW_ENTRIES, k_channels), device=device) * 0.10)
        self.mu = nn.Parameter(torch.full((NUM_PLIES, 3, k_channels), float(mu_init), dtype=torch.float32, device=device))
        self.beta = nn.Parameter(torch.zeros((NUM_PLIES, 3, k_channels), dtype=torch.float32, device=device))

    def forward(self, key_ptm: torch.Tensor, ply_t: torch.Tensor, qat: bool = False) -> torch.Tensor:
        blk_t = ply_t // PLIES_PER_BLOCK
        rep = self.k_rep[key_ptm]                  # [B, 14]
        h_raw = self.h_emb[blk_t[:, None], rep]    # [B, 14, K]
        mu_p = self.mu[ply_t]                      # [B, 3, K]
        beta_p = self.beta[ply_t]                  # [B, 3, K]

        if qat:
            h = ste_round_clamp(h_raw * EMB_SCALE, -127.0, 127.0)
            mu_u = ste_round_clamp(mu_p * EMB_SCALE, -1024.0, 1024.0)
            beta_u = ste_round_clamp(beta_p * self.beta_scale, -float(SCRELU_BETA_MAX), float(SCRELU_BETA_MAX))
            q_max = float(SCRELU_Q)
        else:
            h = h_raw
            mu_u = mu_p
            beta_u = beta_p
            q_max = SCRELU_QF

        # Group 0: 4 Edge strips
        s0 = torch.zeros((key_ptm.shape[0], self.K), dtype=h.dtype, device=h.device)
        for r in range(4):
            c1 = (r + 1) & 3
            u = torch.clamp(h[:, 4 + r] + h[:, 8 + r] + h[:, r] + h[:, c1] + mu_u[:, 0], 0.0, q_max)
            s0 = s0 + u * u

        # Group 1: 2 Diag strips
        s1 = torch.zeros((key_ptm.shape[0], self.K), dtype=h.dtype, device=h.device)
        for d in range(2):
            u = torch.clamp(h[:, 12 + d] + h[:, d] + h[:, d + 2] + mu_u[:, 1], 0.0, q_max)
            s1 = s1 + u * u

        # Group 2: 1 Global sum
        ug = torch.clamp(h.sum(dim=1) + mu_u[:, 2], 0.0, q_max)
        s2 = ug * ug

        res = (beta_u[:, 0] * s0 + beta_u[:, 1] * s1 + beta_u[:, 2] * s2).sum(dim=1)
        if qat:
            return ste_sym_rshift_int(res, SCRELU_SHIFT) / self.score_scale
        return res

    @torch.no_grad()
    def export_quantized_arrays(self) -> Dict[str, np.ndarray]:
        h_q = np.clip(np.round(self.h_emb.detach().cpu().numpy() * EMB_SCALE), -127, 127).astype(np.int8)
        mu_q = np.clip(np.round(self.mu.detach().cpu().numpy() * EMB_SCALE), -1024, 1024).astype(np.int16)
        beta_q = np.clip(
            np.round(self.beta.detach().cpu().numpy() * self.beta_scale), -SCRELU_BETA_MAX, SCRELU_BETA_MAX
        ).astype(np.int16)
        return {"h": h_q, "mu": mu_q, "beta": beta_q}


def train_ptm_screlu(
    bridge: P2C64Bridge,
    orbits: Dict[str, np.ndarray],
    frozen_lin: FrozenLinearPTMEvaluator,
    train_data: np.ndarray,
    k_channels: int,
    float_epochs: int,
    qat_epochs: int,
    lr: float,
    args: argparse.Namespace,
    device: torch.device,
    tag: str,
) -> PTMSCReLUResidual:
    mod = PTMSCReLUResidual(k_channels, orbits, args.mu_init, args.score_scale, device)
    opt = torch.optim.AdamW(mod.parameters(), lr=lr, weight_decay=1e-4)
    n = len(train_data)
    rng = np.random.default_rng(args.seed + 300 + k_channels * 19)
    total_ep = float_epochs + qat_epochs

    for ep in range(total_ep):
        is_qat = ep >= float_epochs
        t0 = time.perf_counter()
        perm = rng.permutation(n)
        lr_now = (lr * 0.25 if is_qat else lr) * 0.5 * (1.0 + math.cos(math.pi * ep / max(1, total_ep)))
        lr_now = max(lr_now, lr * 0.03)
        for pg in opt.param_groups:
            pg["lr"] = lr_now

        loss_sum = 0.0
        steps = 0
        for s in range(0, n, args.batch_size):
            idx_b = perm[s : min(n, s + args.batch_size)]
            sl = train_data[idx_b]
            ext = bridge.extract_batch_ptm(
                sl["black"], sl["white"], sl["stm"],
                aug_seed=int(rng.integers(0, 2**62)), enable_aug=True,
                need_p2c=True, need_raw=False,
            )
            kp = torch.from_numpy(ext["key_ptm"]).long().to(device, non_blocking=True)
            nkp = torch.from_numpy(ext["nonkey_ptm"]).long().to(device, non_blocking=True)
            ply_t = torch.from_numpy(sl["ply"].astype(np.int64)).to(device, non_blocking=True)
            y_t = torch.from_numpy(sl["score"].astype(np.float32)).to(device, non_blocking=True)

            with torch.no_grad():
                y_lin = frozen_lin.eval_disc(kp, nkp, ply_t)
            pred = y_lin + mod(kp, ply_t, qat=is_qat)
            loss = F.smooth_l1_loss(pred, y_t, beta=args.huber_beta)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            # Boundary-cliff elimination: inter-ply head smoothing + inter-block embedding smoothing!
            apply_axis0_smoothness_(mod.beta.data, args.head_smooth_lambda * lr_now * 200.0)
            apply_axis0_smoothness_(mod.mu.data,   args.head_smooth_lambda * lr_now * 200.0)
            apply_axis0_smoothness_(mod.h_emb.data, args.blk_smooth_lambda * lr_now * 100.0)

            loss_sum += float(loss.item())
            steps += 1

        stage = "QAT" if is_qat else "FLT"
        print(f"  [{tag}/{stage}] Epoch {ep+1}/{total_ep} | lr={lr_now:.5f} | huber={loss_sum/max(1,steps):.4f} | {time.perf_counter()-t0:.1f}s")
    return mod


# ============================================================================
# Phase 2B / 2C-2: PTM SCONE-D 3way-3bit (All 8 Codes Unlocked + 60-Ply LUT)
# ============================================================================
class SoftPTMSconeD3W3B(nn.Module):
    def __init__(self, orbits: Dict[str, np.ndarray], device: torch.device) -> None:
        super().__init__()
        self.register_buffer("k_rep", torch.from_numpy(orbits["key_orb_rep"]).long().to(device))
        e_mx = orbits["edge_mirror_x"]
        idx_arr = np.arange(EDGE_RAW_SIZE, dtype=np.int32)
        self.register_buffer("e_rep", torch.from_numpy(np.minimum(idx_arr, e_mx)).long().to(device))
        self.register_buffer("e_swapped", torch.from_numpy(idx_arr > e_mx).to(device))
        self.register_buffer("e_self", torch.from_numpy(idx_arr == e_mx).to(device))

        # All 8 codes 0..7 completely free!
        self.logits_single = nn.Parameter(torch.randn((NUM_BLOCKS, KEY_RAW_ENTRIES, 8), device=device) * 0.25)
        self.logits_edge_f = nn.Parameter(torch.randn((NUM_BLOCKS, EDGE_RAW_SIZE, 8), device=device) * 0.25)
        self.logits_edge_r = nn.Parameter(torch.randn((NUM_BLOCKS, EDGE_RAW_SIZE, 8), device=device) * 0.25)

        # 60-ply Zero-Init Macro-LUTs [60, 8, 8, 8]
        self.T_cw_raw = nn.Parameter(torch.zeros((NUM_PLIES, 8, 8, 8), dtype=torch.float32, device=device))
        self.T_ea_raw = nn.Parameter(torch.zeros((NUM_PLIES, 8, 8, 8), dtype=torch.float32, device=device))
        self.T_cd_raw = nn.Parameter(torch.zeros((NUM_PLIES, 8, 8, 8), dtype=torch.float32, device=device))

    @staticmethod
    def _sym_d4(T: torch.Tensor, a: int, b: int) -> torch.Tensor:
        return 0.5 * (T + T.transpose(a, b))

    def _edge_probs(self, blk_t: torch.Tensor, x_global: torch.Tensor, tau: float) -> Tuple[torch.Tensor, torch.Tensor]:
        x = x_global - EDGE_OFF
        rep = self.e_rep[x]
        swapped = self.e_swapped[x]
        is_self = self.e_self[x]
        pf0 = F.softmax(self.logits_edge_f[blk_t, rep] / tau, dim=-1)
        pr0 = F.softmax(self.logits_edge_r[blk_t, rep] / tau, dim=-1)
        pf = torch.where(swapped.unsqueeze(-1), pr0, pf0)
        pr = torch.where(swapped.unsqueeze(-1), pf0, pr0)
        pr = torch.where(is_self.unsqueeze(-1), pf, pr)
        return pf, pr

    def forward(self, key_ptm: torch.Tensor, ply_t: torch.Tensor, tau: float) -> torch.Tensor:
        blk_t = ply_t // PLIES_PER_BLOCK
        T_cw = self._sym_d4(self.T_cw_raw, 2, 3)[ply_t]
        T_ea = self._sym_d4(self.T_ea_raw, 1, 2)[ply_t]
        T_cd = self._sym_d4(self.T_cd_raw, 1, 3)[ply_t]

        pf = [None] * 14
        pr = [None] * 4
        for p in (0, 1, 2, 3, 8, 9, 10, 11, 12, 13):
            pf[p] = F.softmax(self.logits_single[blk_t, self.k_rep[key_ptm[:, p]]] / tau, dim=-1)
        for i in range(4):
            pf[4 + i], pr[i] = self._edge_probs(blk_t, key_ptm[:, 4 + i], tau)

        out = torch.zeros(key_ptm.shape[0], dtype=torch.float32, device=key_ptm.device)
        for i in range(4):
            out = out + torch.einsum("bi,bj,bk,bijk->b", pf[i], pf[4 + i], pr[(i + 3) & 3], T_cw)
            out = out + torch.einsum("bi,bj,bk,bijk->b", pf[4 + i], pr[i], pf[8 + i], T_ea)
        out = out + torch.einsum("bi,bj,bk,bijk->b", pf[0], pf[12], pf[2], T_cd)
        out = out + torch.einsum("bi,bj,bk,bijk->b", pf[1], pf[13], pf[3], T_cd)
        return out

    @torch.no_grad()
    def extract_hard_codes(self, orbits: Dict[str, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        k_rep = orbits["key_orb_rep"]
        e_mx = orbits["edge_mirror_x"]
        idx_arr = np.arange(EDGE_RAW_SIZE, dtype=np.int32)
        rep_e = np.minimum(idx_arr, e_mx)
        swapped = idx_arr > e_mx
        is_self = idx_arr == e_mx

        ls = self.logits_single.detach().cpu().numpy()
        lf = self.logits_edge_f.detach().cpu().numpy()
        lr = self.logits_edge_r.detach().cpu().numpy()

        z_single = np.argmax(ls, axis=-1).astype(np.int32)[:, k_rep]
        ef0 = np.argmax(lf, axis=-1).astype(np.int32)[:, rep_e]
        er0 = np.argmax(lr, axis=-1).astype(np.int32)[:, rep_e]
        ef = np.where(swapped[None, :], er0, ef0)
        er = np.where(swapped[None, :], ef0, er0)
        er = np.where(is_self[None, :], ef, er)

        z_fwd = z_single.copy()
        z_fwd[:, EDGE_OFF : EDGE_OFF + EDGE_RAW_SIZE] = ef
        return z_fwd, er


@torch.no_grad()
def eb_smoothed_60ply_convex_refit(
    bridge: P2C64Bridge,
    train_data: np.ndarray,
    base_eval_fn,
    z_fwd: np.ndarray,
    z_rev: np.ndarray,
    eb_lambda: float,
    eb_ply_lambda: float,
    eb_amax: float,
    args: argparse.Namespace,
    device: torch.device,
    passes: int = 6,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Provably Monotonic 60-Ply Inter-Ply Smoothed MM Empirical Bayes Shrinkage +
    Amplitude Clipping Convex Refit for T_cw[60], T_ea[60], T_cd[60]."""
    orb_cw, n_cw = build_macro_orbit_map_v2("cw")
    orb_ea, n_ea = build_macro_orbit_map_v2("ea")
    orb_cd, n_cd = build_macro_orbit_map_v2("cd")

    orb_maps = [
        (torch.from_numpy(orb_cw).long().to(device), n_cw),
        (torch.from_numpy(orb_ea).long().to(device), n_ea),
        (torch.from_numpy(orb_cd).long().to(device), n_cd),
    ]
    theta_orb = [
        torch.zeros((NUM_PLIES, n_cw), dtype=torch.float64, device=device),
        torch.zeros((NUM_PLIES, n_ea), dtype=torch.float64, device=device),
        torch.zeros((NUM_PLIES, n_cd), dtype=torch.float64, device=device),
    ]
    T_list = [
        torch.zeros((NUM_PLIES, 512), dtype=torch.float32, device=device),
        torch.zeros((NUM_PLIES, 512), dtype=torch.float32, device=device),
        torch.zeros((NUM_PLIES, 512), dtype=torch.float32, device=device),
    ]
    z_fwd_t = torch.from_numpy(z_fwd).long().to(device)
    z_rev_t = torch.from_numpy(z_rev).long().to(device)

    deg = torch.full((NUM_PLIES, 1), 2.0, dtype=torch.float64, device=device)
    deg[0, 0] = 1.0
    deg[-1, 0] = 1.0

    n = len(train_data)
    M_TERMS = 10.0

    for _ in range(passes):
        grad_num = [
            torch.zeros((NUM_PLIES, n_cw), dtype=torch.float64, device=device),
            torch.zeros((NUM_PLIES, n_ea), dtype=torch.float64, device=device),
            torch.zeros((NUM_PLIES, n_cd), dtype=torch.float64, device=device),
        ]
        cnt_den = [
            torch.zeros((NUM_PLIES, n_cw), dtype=torch.float64, device=device),
            torch.zeros((NUM_PLIES, n_ea), dtype=torch.float64, device=device),
            torch.zeros((NUM_PLIES, n_cd), dtype=torch.float64, device=device),
        ]
        ones_f64 = None

        for s in range(0, n, args.val_batch_size):
            e = min(n, s + args.val_batch_size)
            sl = train_data[s:e]
            ext = bridge.extract_batch_ptm(
                sl["black"], sl["white"], sl["stm"], enable_aug=False, need_p2c=True, need_raw=False
            )
            kp = torch.from_numpy(ext["key_ptm"]).long().to(device, non_blocking=True)
            nkp = torch.from_numpy(ext["nonkey_ptm"]).long().to(device, non_blocking=True)
            ply_t = torch.from_numpy(sl["ply"].astype(np.int64)).to(device, non_blocking=True)
            blk_t = ply_t // PLIES_PER_BLOCK
            y_t = torch.from_numpy(sl["score"].astype(np.float32)).to(device, non_blocking=True)

            if ones_f64 is None or ones_f64.shape[0] != kp.shape[0]:
                ones_f64 = torch.ones(kp.shape[0], dtype=torch.float64, device=device)

            r_t = y_t - base_eval_fn(kp, nkp, ply_t)
            zf = z_fwd_t[blk_t[:, None], kp]
            zr = z_rev_t[blk_t[:, None], kp[:, 4:8] - EDGE_OFF]

            all_keys = [
                [(zf[:, i] << 6) | (zf[:, 4 + i] << 3) | zr[:, (i + 3) & 3] for i in range(4)],
                [(zf[:, 4 + i] << 6) | (zr[:, i] << 3) | zf[:, 8 + i] for i in range(4)],
                [(zf[:, 0] << 6) | (zf[:, 12] << 3) | zf[:, 2], (zf[:, 1] << 6) | (zf[:, 13] << 3) | zf[:, 3]],
            ]

            pred_cur = torch.zeros_like(r_t)
            for m_idx in range(3):
                Tm = T_list[m_idx]
                for kk in all_keys[m_idx]:
                    pred_cur = pred_cur + Tm[ply_t, kk]
            err_res = (r_t - pred_cur).double()

            for m_idx in range(3):
                oid_map, n_orb = orb_maps[m_idx]
                for kk in all_keys[m_idx]:
                    flat_idx = ply_t * n_orb + oid_map[kk]
                    grad_num[m_idx].view(-1).scatter_add_(0, flat_idx, err_res)
                    cnt_den[m_idx].view(-1).scatter_add_(0, flat_idx, ones_f64)

        for m_idx in range(3):
            oid_map, _ = orb_maps[m_idx]
            th = theta_orb[m_idx]
            neigh = torch.zeros_like(th)
            neigh[0] = th[1]
            neigh[-1] = th[-2]
            neigh[1:-1] = th[:-2] + th[2:]

            mn = M_TERMS * cnt_den[m_idx]
            num = mn * th + grad_num[m_idx] + eb_ply_lambda * neigh
            den = mn + eb_lambda + eb_ply_lambda * deg
            th_new = torch.clamp(num / den, -eb_amax, eb_amax)
            theta_orb[m_idx] = th_new
            T_list[m_idx] = th_new.float()[:, oid_map]

    return T_list[0].cpu().numpy(), T_list[1].cpu().numpy(), T_list[2].cpu().numpy()


@torch.no_grad()
def exact_em_lut_v2(
    bridge: P2C64Bridge,
    orbits: Dict[str, np.ndarray],
    train_data: np.ndarray,
    base_eval_fn,
    z_fwd: np.ndarray,
    z_rev: np.ndarray,
    T_cw: np.ndarray,
    T_ea: np.ndarray,
    T_cd: np.ndarray,
    args: argparse.Namespace,
    device: torch.device,
    em_iters: int,
    tag: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    k_rep_np = orbits["key_orb_rep"]
    k_rep_t = torch.from_numpy(k_rep_np).long().to(device)
    cands = torch.arange(8, dtype=torch.long, device=device)

    n = len(train_data)
    min_hits = max(5, int(n // 5000))

    for it in range(em_iters):
        t0 = time.perf_counter()
        z_fwd_t = torch.from_numpy(z_fwd).long().to(device)
        z_rev_t = torch.from_numpy(z_rev).long().to(device)
        tcw_t = torch.from_numpy(T_cw).float().to(device)
        tea_t = torch.from_numpy(T_ea).float().to(device)
        tcd_t = torch.from_numpy(T_cd).float().to(device)

        delta_single = torch.zeros((NUM_BLOCKS, KEY_RAW_ENTRIES, 8), dtype=torch.float64, device=device)
        hits_single = torch.zeros((NUM_BLOCKS, KEY_RAW_ENTRIES), dtype=torch.int64, device=device)
        ones_i64 = None

        for s in range(0, n, args.val_batch_size):
            e = min(n, s + args.val_batch_size)
            sl = train_data[s:e]
            ext = bridge.extract_batch_ptm(
                sl["black"], sl["white"], sl["stm"], enable_aug=False, need_p2c=True, need_raw=False
            )
            kp = torch.from_numpy(ext["key_ptm"]).long().to(device, non_blocking=True)
            nkp = torch.from_numpy(ext["nonkey_ptm"]).long().to(device, non_blocking=True)
            ply_t = torch.from_numpy(sl["ply"].astype(np.int64)).to(device, non_blocking=True)
            blk_t = ply_t // PLIES_PER_BLOCK
            y_t = torch.from_numpy(sl["score"].astype(np.float32)).to(device, non_blocking=True)
            if ones_i64 is None or ones_i64.shape[0] != kp.shape[0]:
                ones_i64 = torch.ones(kp.shape[0], dtype=torch.int64, device=device)

            r_t = y_t - base_eval_fn(kp, nkp, ply_t)
            zf = z_fwd_t[blk_t[:, None], kp]
            zr = z_rev_t[blk_t[:, None], kp[:, 4:8] - EDGE_OFF]

            pred_cur = torch.zeros_like(r_t)
            for i in range(4):
                pred_cur = pred_cur + tcw_t[ply_t, (zf[:, i] << 6) | (zf[:, 4 + i] << 3) | zr[:, (i + 3) & 3]]
                pred_cur = pred_cur + tea_t[ply_t, (zf[:, 4 + i] << 6) | (zr[:, i] << 3) | zf[:, 8 + i]]
            pred_cur = pred_cur + tcd_t[ply_t, (zf[:, 0] << 6) | (zf[:, 12] << 3) | zf[:, 2]]
            pred_cur = pred_cur + tcd_t[ply_t, (zf[:, 1] << 6) | (zf[:, 13] << 3) | zf[:, 3]]
            err_cur = pred_cur - r_t
            cur_sq = err_cur * err_cur

            def accum(p_idx: int, new_term_fn, cur_term: torch.Tensor) -> None:
                rep_p = k_rep_t[kp[:, p_idx]]
                new_t = new_term_fn(cands[None, :])
                err_new = err_cur[:, None] - cur_term[:, None] + new_t
                dloss = (err_new * err_new - cur_sq[:, None]).double()
                flat1 = blk_t * KEY_RAW_ENTRIES + rep_p
                hits_single.view(-1).scatter_add_(0, flat1, ones_i64)
                delta_single.view(-1, 8).scatter_add_(0, flat1[:, None].expand(-1, 8), dloss)

            for i in range(4):
                t_cw_c = tcw_t[ply_t, (zf[:, i] << 6) | (zf[:, 4 + i] << 3) | zr[:, (i + 3) & 3]]
                if i < 2:
                    t_cd_c = tcd_t[ply_t, (zf[:, i] << 6) | (zf[:, 12 + i] << 3) | zf[:, i + 2]]
                    accum(
                        i,
                        lambda zc, ii=i: tcw_t[ply_t[:, None], (zc << 6) | (zf[:, 4 + ii, None] << 3) | zr[:, (ii + 3) & 3, None]]
                                       + tcd_t[ply_t[:, None], (zc << 6) | (zf[:, 12 + ii, None] << 3) | zf[:, ii + 2, None]],
                        t_cw_c + t_cd_c,
                    )
                else:
                    d = i - 2
                    t_cd_c = tcd_t[ply_t, (zf[:, d] << 6) | (zf[:, 12 + d] << 3) | zf[:, i]]
                    accum(
                        i,
                        lambda zc, ii=i, dd=d: tcw_t[ply_t[:, None], (zc << 6) | (zf[:, 4 + ii, None] << 3) | zr[:, (ii + 3) & 3, None]]
                                             + tcd_t[ply_t[:, None], (zf[:, dd, None] << 6) | (zf[:, 12 + dd, None] << 3) | zc],
                        t_cw_c + t_cd_c,
                    )

            for i in range(4):
                t_ea_c = tea_t[ply_t, (zf[:, 4 + i] << 6) | (zr[:, i] << 3) | zf[:, 8 + i]]
                accum(8 + i, lambda zc, ii=i: tea_t[ply_t[:, None], (zf[:, 4 + ii, None] << 6) | (zr[:, ii, None] << 3) | zc], t_ea_c)

            for d in range(2):
                t_cd_c = tcd_t[ply_t, (zf[:, d] << 6) | (zf[:, 12 + d] << 3) | zf[:, d + 2]]
                accum(12 + d, lambda zc, dd=d: tcd_t[ply_t[:, None], (zf[:, dd, None] << 6) | (zc << 3) | zf[:, dd + 2, None]], t_cd_c)

        ds_np = delta_single.cpu().numpy()
        hs_np = hits_single.cpu().numpy()
        flips = 0
        for b in range(NUM_BLOCKS):
            for x in range(KEY_RAW_ENTRIES):
                if EDGE_OFF <= x < EDGE_OFF + EDGE_RAW_SIZE:
                    continue
                if k_rep_np[x] != x or hs_np[b, x] < min_hits:
                    continue
                row = ds_np[b, x]
                best_c = int(np.argmin(row))
                if row[best_c] < -1e-3 and best_c != z_fwd[b, x]:
                    z_fwd[b, x] = best_c
                    flips += 1
            z_sym = z_fwd[b, k_rep_np]
            z_fwd[b, :EDGE_OFF] = z_sym[:EDGE_OFF]
            z_fwd[b, EDGE_OFF + EDGE_RAW_SIZE :] = z_sym[EDGE_OFF + EDGE_RAW_SIZE :]

        T_cw, T_ea, T_cd = eb_smoothed_60ply_convex_refit(
            bridge, train_data, base_eval_fn, z_fwd, z_rev,
            args.eb_lambda, args.eb_ply_lambda, args.eb_amax, args, device, passes=4,
        )
        print(f"  [{tag}/EM-LUT] Iter {it+1}/{em_iters} | updated_orbits={flips} (min_hits={min_hits}) | {time.perf_counter()-t0:.1f}s")

    return z_fwd, z_rev, T_cw, T_ea, T_cd


def train_ptm_3w3b_pipeline(
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
    mod = SoftPTMSconeD3W3B(orbits, device)
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
            ext = bridge.extract_batch_ptm(
                sl["black"], sl["white"], sl["stm"],
                aug_seed=int(rng.integers(0, 2**62)), enable_aug=True,
                need_p2c=True, need_raw=False,
            )
            kp = torch.from_numpy(ext["key_ptm"]).long().to(device, non_blocking=True)
            nkp = torch.from_numpy(ext["nonkey_ptm"]).long().to(device, non_blocking=True)
            ply_t = torch.from_numpy(sl["ply"].astype(np.int64)).to(device, non_blocking=True)
            y_t = torch.from_numpy(sl["score"].astype(np.float32)).to(device, non_blocking=True)

            frac = gstep / max(1, total_steps - 1)
            tau = args.tau_start * ((args.tau_end / args.tau_start) ** frac)
            gstep += 1

            with torch.no_grad():
                y_base = base_eval_fn(kp, nkp, ply_t)
            pred = y_base + mod(kp, ply_t, tau=tau)
            loss = F.smooth_l1_loss(pred, y_t, beta=args.huber_beta)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            # Inter-block logit smoothing + inter-ply Macro-LUT smoothing!
            a_blk = args.blk_smooth_lambda * args.b_lr * 100.0
            a_ply = args.head_smooth_lambda * args.b_lr * 100.0
            apply_axis0_smoothness_(mod.logits_single.data, a_blk)
            apply_axis0_smoothness_(mod.logits_edge_f.data, a_blk)
            apply_axis0_smoothness_(mod.logits_edge_r.data, a_blk)
            apply_axis0_smoothness_(mod.T_cw_raw.data, a_ply)
            apply_axis0_smoothness_(mod.T_ea_raw.data, a_ply)
            apply_axis0_smoothness_(mod.T_cd_raw.data, a_ply)

            loss_sum += float(loss.item())
            steps += 1

        print(f"  [{tag}/Soft] Epoch {ep+1}/{soft_epochs} | tau={tau:.3f} | huber={loss_sum/max(1,steps):.4f} | {time.perf_counter()-t0:.1f}s")

    t1 = time.perf_counter()
    z_fwd, z_rev = mod.extract_hard_codes(orbits)
    T_cw, T_ea, T_cd = eb_smoothed_60ply_convex_refit(
        bridge, train_data, base_eval_fn, z_fwd, z_rev,
        args.eb_lambda, args.eb_ply_lambda, args.eb_amax, args, device, passes=6,
    )
    print(f"  [{tag}/EB-Refit] 60-ply smoothed MM EB (lam={args.eb_lambda}, ply_lam={args.eb_ply_lambda}, Amax={args.eb_amax}) in {time.perf_counter()-t1:.1f}s")

    if em_iters > 0:
        z_fwd, z_rev, T_cw, T_ea, T_cd = exact_em_lut_v2(
            bridge, orbits, train_data, base_eval_fn, z_fwd, z_rev, T_cw, T_ea, T_cd, args, device, em_iters, tag
        )
    return z_fwd, z_rev, T_cw, T_ea, T_cd


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
    print("\n============= P2C-64 v2.0 Val RMSE / MAE Summary (Quantized .p2c) =============")
    print(f"{'Model':<22} | {'All RMSE':>8} {'All MAE':>8} | {'0-19':>8} | {'20-39':>8} | {'40-59':>8}")
    print("-" * 76)
    for name, m in all_metrics.items():
        print(
            f"{name:<22} | {m['rmse_all']:8.4f} {m['mae_all']:8.4f} | "
            f"{m['rmse_0_19']:8.4f} | {m['rmse_20_39']:8.4f} | {m['rmse_40_59']:8.4f}"
        )


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="P2C-64 v2.0 (Full PTM) Training Pipeline")
    ap.add_argument("--train-bin", type=Path, required=True)
    ap.add_argument("--val-bin", type=Path, required=True)
    ap.add_argument("--work-dir", type=Path, required=True)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--phases", type=str, default="all", help="all | phase1 | phase2 | cand_a | cand_b | cand_c")
    ap.add_argument("--batch-size", type=int, default=16384)
    ap.add_argument("--val-batch-size", type=int, default=32768)
    ap.add_argument("--score-scale", type=float, default=P2C_DEFAULT_SCALE)
    ap.add_argument("--huber-beta", type=float, default=4.0)
    ap.add_argument("--smooth-lambda", type=float, default=1e-4)
    ap.add_argument("--head-smooth-lambda", type=float, default=1e-3)
    ap.add_argument("--blk-smooth-lambda", type=float, default=5e-4)

    # Phase 1
    ap.add_argument("--p1-epochs", type=int, default=6)
    ap.add_argument("--p1-lr", type=float, default=0.05)

    # Phase 2A (CAND_A_PTM_SCRELU4)
    ap.add_argument("--a-epochs", type=int, default=4)
    ap.add_argument("--a-qat-epochs", type=int, default=2)
    ap.add_argument("--a-lr", type=float, default=0.01)
    ap.add_argument("--mu-init", type=float, default=0.5)

    # Phase 2B (CAND_B_PTM_3W3B)
    ap.add_argument("--b-soft-epochs", type=int, default=4)
    ap.add_argument("--b-lr", type=float, default=0.02)
    ap.add_argument("--tau-start", type=float, default=1.5)
    ap.add_argument("--tau-end", type=float, default=0.15)
    ap.add_argument("--eb-lambda", type=float, default=200.0)
    ap.add_argument("--eb-ply-lambda", type=float, default=100.0)
    ap.add_argument("--eb-amax", type=float, default=4.0)
    ap.add_argument("--em-iters", type=int, default=2)

    # Phase 2C (CAND_C_PTM_HYBRID)
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
            out.extend(["phase1", "cand_a", "cand_b", "cand_c"])
        elif tok in ("raw", "linear", "phase1"):
            out.append("phase1")
        elif tok == "phase2":
            out.extend(["cand_a", "cand_b", "cand_c"])
        else:
            out.append(tok)
    return list(dict.fromkeys(out))


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    torch.set_float32_matmul_precision("high")
    device = torch.device("cuda" if (args.device == "auto" and torch.cuda.is_available()) else ("cpu" if args.device == "auto" else args.device))

    args.work_dir.mkdir(parents=True, exist_ok=True)
    phases = expand_phases(args.phases)
    bridge = P2C64Bridge()
    orbits = bridge.get_orbit_tables()
    train_data = load_packed_dataset(args.train_bin)
    val_data = load_packed_dataset(args.val_bin)

    print(f"=== P2C-64 v2.0 (Full PTM) Training Pipeline | device={device} | phases={phases} ===")
    print(f"  Train={len(train_data):,} | Val={len(val_data):,} | batch={args.batch_size} | scale={args.score_scale}")

    metrics_file = args.work_dir / "metrics.json"
    all_metrics: Dict[str, Dict[str, float]] = {}
    if args.resume and metrics_file.exists():
        all_metrics = json.loads(metrics_file.read_text())

    raw_p2c = args.work_dir / "raw_ntuple64.p2c"
    lin_p2c = args.work_dir / "linear64.p2c"
    lin_model: Optional[P2C64Model] = None

    # ---- Phase 1 (Trained ONCE -> exports identical RAW_NTUPLE_64 & LINEAR_64!) ----
    if "phase1" in phases:
        if args.resume and raw_p2c.exists() and lin_p2c.exists() and "LINEAR_64" in all_metrics:
            print(f"[Skip] Phase 1 already completed ({raw_p2c.name} & {lin_p2c.name})")
            lin_model = load_p2c(lin_p2c)  # type: ignore[assignment]
        else:
            lin_model, m_p1 = train_phase1_ptm_shared(
                bridge, orbits, train_data, val_data, args, device,
                raw_p2c, lin_p2c, args.work_dir / "ckpt_phase1.pt",
            )
            all_metrics["RAW_NTUPLE_64"] = m_p1
            all_metrics["LINEAR_64"] = m_p1
            save_metrics(args.work_dir, all_metrics)

    need_p2 = any(p in phases for p in ("cand_a", "cand_b", "cand_c"))
    if need_p2:
        if lin_model is None:
            if not lin_p2c.exists():
                raise FileNotFoundError(f"Phase 2 requires {lin_p2c}")
            lin_model = load_p2c(lin_p2c)  # type: ignore[assignment]
        assert isinstance(lin_model, P2C64Model)
        frozen_lin = FrozenLinearPTMEvaluator(lin_model, device)

        # ---- Phase 2A: CAND_A_PTM_SCRELU4 ----
        cand_a_p2c = args.work_dir / "cand_a_ptm_screlu4.p2c"
        if "cand_a" in phases:
            if args.resume and cand_a_p2c.exists() and "CAND_A_PTM_SCRELU4" in all_metrics:
                print(f"[Skip] CAND_A_PTM_SCRELU4 already completed ({cand_a_p2c.name})")
            else:
                print("\n=== [Phase 2A] Training CAND_A_PTM_SCRELU4 (4-ch PTM + 60-Ply Smooth Head + QAT) ===")
                mod_a = train_ptm_screlu(
                    bridge, orbits, frozen_lin, train_data,
                    k_channels=4, float_epochs=args.a_epochs, qat_epochs=args.a_qat_epochs,
                    lr=args.a_lr, args=args, device=device, tag="CAND_A",
                )
                q = mod_a.export_quantized_arrays()
                m_a = make_empty_p2c64(EvalMode.CAND_A_PTM_SCRELU4, score_scale=args.score_scale)
                m_a.key_blocks["w"] = lin_model.key_blocks["w"].copy()
                m_a.nonkey_w = lin_model.nonkey_w.copy()
                m_a.key_blocks["aux"] = q["h"].view(np.uint8)
                m_a.screlu_mu = q["mu"]
                m_a.screlu_beta = q["beta"]
                m_a = symmetrize_p2c64_model(m_a, orbits)
                save_p2c64(cand_a_p2c, m_a)
                met_a = evaluate_deployed_p2c(bridge, m_a, val_data, args.val_batch_size)
                all_metrics["CAND_A_PTM_SCRELU4"] = met_a
                save_metrics(args.work_dir, all_metrics)
                print(f"  [CAND_A] Saved {cand_a_p2c.name} | Val RMSE={met_a['rmse_all']:.4f} MAE={met_a['mae_all']:.4f}")

        # ---- Phase 2B: CAND_B_PTM_3W3B ----
        cand_b_p2c = args.work_dir / "cand_b_ptm_3w3b.p2c"
        if "cand_b" in phases:
            if args.resume and cand_b_p2c.exists() and "CAND_B_PTM_3W3B" in all_metrics:
                print(f"[Skip] CAND_B_PTM_3W3B already completed ({cand_b_p2c.name})")
            else:
                print("\n=== [Phase 2B] Training CAND_B_PTM_3W3B (8 Codes Unlocked + 60-Ply Smoothed Macro-LUT) ===")
                z_fwd, z_rev, T_cw, T_ea, T_cd = train_ptm_3w3b_pipeline(
                    bridge, orbits, train_data, frozen_lin.eval_disc,
                    soft_epochs=args.b_soft_epochs, em_iters=args.em_iters,
                    args=args, device=device, tag="CAND_B",
                )
                m_b = make_empty_p2c64(EvalMode.CAND_B_PTM_3W3B, score_scale=args.score_scale)
                m_b.key_blocks["w"] = lin_model.key_blocks["w"].copy()
                m_b.nonkey_w = lin_model.nonkey_w.copy()
                m_b.key_blocks["aux"][:, :, 0] = z_fwd.astype(np.uint8)
                m_b.key_blocks["aux"][:, EDGE_OFF : EDGE_OFF + EDGE_RAW_SIZE, 1] = z_rev.astype(np.uint8)
                m_b.macro_cw = np.clip(np.round(T_cw * args.score_scale), -32767, 32767).astype(np.int16)
                m_b.macro_ea = np.clip(np.round(T_ea * args.score_scale), -32767, 32767).astype(np.int16)
                m_b.macro_cd = np.clip(np.round(T_cd * args.score_scale), -32767, 32767).astype(np.int16)
                m_b = symmetrize_p2c64_model(m_b, orbits)
                save_p2c64(cand_b_p2c, m_b)
                met_b = evaluate_deployed_p2c(bridge, m_b, val_data, args.val_batch_size)
                all_metrics["CAND_B_PTM_3W3B"] = met_b
                save_metrics(args.work_dir, all_metrics)
                print(f"  [CAND_B] Saved {cand_b_p2c.name} | Val RMSE={met_b['rmse_all']:.4f} MAE={met_b['mae_all']:.4f}")

        # ---- Phase 2C: CAND_C_PTM_HYBRID (2-ch PTM SCReLU + 60-Ply 3way-3bit) ----
        cand_c_p2c = args.work_dir / "cand_c_ptm_hybrid.p2c"
        if "cand_c" in phases:
            if args.resume and cand_c_p2c.exists() and "CAND_C_PTM_HYBRID" in all_metrics:
                print(f"[Skip] CAND_C_PTM_HYBRID already completed ({cand_c_p2c.name})")
            else:
                print("\n=== [Phase 2C-1] Training CAND_C 2-ch PTM SCReLU Residual ===")
                mod_c1 = train_ptm_screlu(
                    bridge, orbits, frozen_lin, train_data,
                    k_channels=2, float_epochs=args.c_screlu_epochs, qat_epochs=args.c_qat_epochs,
                    lr=args.a_lr, args=args, device=device, tag="CAND_C-SC2",
                )
                qc1 = mod_c1.export_quantized_arrays()

                k_rep_t = torch.from_numpy(orbits["key_orb_rep"]).long().to(device)
                h_q = torch.from_numpy(qc1["h"].astype(np.float32)).to(device)       # [10, 91854, 2]
                mu_q = torch.from_numpy(qc1["mu"].astype(np.float32)).to(device)     # [60, 3, 2]
                beta_q = torch.from_numpy(qc1["beta"].astype(np.float32)).to(device) # [60, 3, 2]

                @torch.no_grad()
                def eval_lin_plus_sc2(kp, nkp, ply_t):
                    y_lin = frozen_lin.eval_disc(kp, nkp, ply_t)
                    blk_t = ply_t // PLIES_PER_BLOCK
                    h = h_q[blk_t[:, None], k_rep_t[kp]]  # [B, 14, 2]
                    mp = mu_q[ply_t]
                    bp = beta_q[ply_t]
                    s0 = torch.zeros((kp.shape[0], 2), dtype=torch.float32, device=device)
                    for r in range(4):
                        c1 = (r + 1) & 3
                        u = torch.clamp(h[:, 4 + r] + h[:, 8 + r] + h[:, r] + h[:, c1] + mp[:, 0], 0.0, float(SCRELU_Q))
                        s0 = s0 + u * u
                    s1 = torch.zeros((kp.shape[0], 2), dtype=torch.float32, device=device)
                    for d in range(2):
                        u = torch.clamp(h[:, 12 + d] + h[:, d] + h[:, d + 2] + mp[:, 1], 0.0, float(SCRELU_Q))
                        s1 = s1 + u * u
                    ug = torch.clamp(h.sum(dim=1) + mp[:, 2], 0.0, float(SCRELU_Q))
                    acc = (bp[:, 0] * s0 + bp[:, 1] * s1 + bp[:, 2] * (ug * ug)).sum(dim=1)
                    return y_lin + torch.trunc(acc / float(1 << SCRELU_SHIFT)) / float(args.score_scale)

                print("\n=== [Phase 2C-2] Training CAND_C 60-Ply 3way-3bit Residual on (LINEAR + SC2) ===")
                z_fwd, z_rev, T_cw, T_ea, T_cd = train_ptm_3w3b_pipeline(
                    bridge, orbits, train_data, eval_lin_plus_sc2,
                    soft_epochs=args.c_soft_epochs, em_iters=args.c_em_iters,
                    args=args, device=device, tag="CAND_C-LUT",
                )
                m_c = make_empty_p2c64(EvalMode.CAND_C_PTM_HYBRID, score_scale=args.score_scale)
                m_c.key_blocks["w"] = lin_model.key_blocks["w"].copy()
                m_c.nonkey_w = lin_model.nonkey_w.copy()
                m_c.key_blocks["aux"][:, :, :2] = qc1["h"].view(np.uint8)
                packed = z_fwd.copy()
                packed[:, EDGE_OFF : EDGE_OFF + EDGE_RAW_SIZE] |= (z_rev << 3)
                m_c.key_blocks["aux"][:, :, 2] = packed.astype(np.uint8)
                m_c.key_blocks["aux"][:, :, 3] = 0
                m_c.screlu_mu[:, :, :2] = qc1["mu"]
                m_c.screlu_beta[:, :, :2] = qc1["beta"]
                m_c.macro_cw = np.clip(np.round(T_cw * args.score_scale), -32767, 32767).astype(np.int16)
                m_c.macro_ea = np.clip(np.round(T_ea * args.score_scale), -32767, 32767).astype(np.int16)
                m_c.macro_cd = np.clip(np.round(T_cd * args.score_scale), -32767, 32767).astype(np.int16)
                m_c = symmetrize_p2c64_model(m_c, orbits)
                save_p2c64(cand_c_p2c, m_c)
                met_c = evaluate_deployed_p2c(bridge, m_c, val_data, args.val_batch_size)
                all_metrics["CAND_C_PTM_HYBRID"] = met_c
                save_metrics(args.work_dir, all_metrics)
                print(f"  [CAND_C] Saved {cand_c_p2c.name} | Val RMSE={met_c['rmse_all']:.4f} MAE={met_c['mae_all']:.4f}")

    if all_metrics:
        print_summary_table(all_metrics)


if __name__ == "__main__":
    main()
