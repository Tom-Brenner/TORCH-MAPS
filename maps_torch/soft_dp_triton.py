"""Triton time-wavefront kernel for duration-aware soft forward DP."""

from __future__ import annotations

import torch
from torch import Tensor

import triton
import triton.language as tl


@triton.jit
def soft_fwd_timestep_kernel(
    log_a_ptr,
    ll_ptr,
    phone_ids_ptr,
    lengths_ptr,
    phone_lens_ptr,
    dur_neglog_ptr,
    floor_ptr,
    bigram_ptr,
    t,
    B,
    N,
    T,
    C,
    D,
    lambda_d,
    lambda_bg,
    tau,
    inv_tau,
    ni,
    stride_a_b,
    stride_a_n,
    stride_a_t,
    stride_a_d,
    stride_ll_b,
    stride_ll_t,
    stride_ll_c,
    stride_ph_b,
    stride_ph_n,
    stride_dur_c,
    stride_dur_d,
    stride_bg_r,
    stride_bg_c,
    BLOCK_D: tl.constexpr,
):
    pid = tl.program_id(0)
    b = pid // N
    n = pid % N
    if b >= B:
        return
    Tb = tl.load(lengths_ptr + b)
    Nb = tl.load(phone_lens_ptr + b)
    if t >= Tb or n >= Nb:
        return

    pn = tl.load(phone_ids_ptr + b * stride_ph_b + n * stride_ph_n)
    ll_t = tl.load(ll_ptr + b * stride_ll_b + t * stride_ll_t + pn * stride_ll_c)

    if t > 0:
        offs = tl.arange(0, BLOCK_D)
        for start in range(0, D - 1, BLOCK_D):
            src = start + offs
            mask = src < (D - 1)
            prev_val = tl.load(
                log_a_ptr
                + b * stride_a_b
                + n * stride_a_n
                + (t - 1) * stride_a_t
                + src * stride_a_d,
                mask=mask,
                other=ni,
            )
            out = prev_val + ll_t
            tl.store(
                log_a_ptr
                + b * stride_a_b
                + n * stride_a_n
                + t * stride_a_t
                + (src + 1) * stride_a_d,
                out,
                mask=mask,
            )

        if n >= 1:
            prev_n = n - 1
            prev_p = tl.load(phone_ids_ptr + b * stride_ph_b + prev_n * stride_ph_n)
            fl = tl.load(floor_ptr + prev_p)
            bg = lambda_bg * tl.load(bigram_ptr + prev_p * stride_bg_r + pn * stride_bg_c)
            max_v = ni
            for start in range(0, D, BLOCK_D):
                dd = start + offs
                mask = dd < D
                valid = mask & (dd >= (fl - 1))
                aval = tl.load(
                    log_a_ptr
                    + b * stride_a_b
                    + prev_n * stride_a_n
                    + (t - 1) * stride_a_t
                    + dd * stride_a_d,
                    mask=valid,
                    other=ni,
                )
                dcost = tl.load(
                    dur_neglog_ptr + prev_p * stride_dur_c + dd * stride_dur_d,
                    mask=valid,
                    other=0.0,
                )
                scored = aval - lambda_d * dcost - bg
                block_max = tl.max(tl.where(valid, scored, ni), axis=0)
                max_v = tl.maximum(max_v, block_max)
            acc = 0.0
            for start in range(0, D, BLOCK_D):
                dd = start + offs
                mask = dd < D
                valid = mask & (dd >= (fl - 1))
                aval = tl.load(
                    log_a_ptr
                    + b * stride_a_b
                    + prev_n * stride_a_n
                    + (t - 1) * stride_a_t
                    + dd * stride_a_d,
                    mask=valid,
                    other=ni,
                )
                dcost = tl.load(
                    dur_neglog_ptr + prev_p * stride_dur_c + dd * stride_dur_d,
                    mask=valid,
                    other=0.0,
                )
                scored = aval - lambda_d * dcost - bg
                acc += tl.sum(
                    tl.where(valid, tl.exp((scored - max_v) * inv_tau), 0.0), axis=0
                )
            lse = max_v + tau * tl.log(acc + 1e-20)
            tl.store(
                log_a_ptr
                + b * stride_a_b
                + n * stride_a_n
                + t * stride_a_t
                + 0 * stride_a_d,
                ll_t + lse,
            )


def soft_forward_duration_triton(
    cost: Tensor,
    phone_ids: Tensor,
    lengths: Tensor,
    phone_lens: Tensor,
    *,
    dur_neglog: Tensor,
    floor_frames: Tensor,
    bigram_neglog: Tensor,
    lambda_d: float,
    lambda_bg: float,
    tau: float,
) -> Tensor:
    """Triton time-wavefront forward; returns log_alpha [B,N,T,Dmax]."""
    B, T, C = cost.shape
    N = phone_ids.shape[1]
    Dmax = dur_neglog.shape[1]
    device = cost.device
    dtype = torch.float32
    ni = float(torch.finfo(dtype).min / 4)
    ll = (-cost).contiguous().to(dtype)
    phone_ids = phone_ids.contiguous().long()
    lengths = lengths.contiguous().long()
    phone_lens = phone_lens.contiguous().long()
    dur_neglog = dur_neglog.contiguous().to(dtype)
    floor_frames = floor_frames.contiguous().long()
    bigram_neglog = bigram_neglog.contiguous().to(dtype)

    log_a = torch.full((B, N, T, Dmax), ni, device=device, dtype=dtype)
    b_idx = torch.arange(B, device=device)
    p0 = phone_ids[:, 0].clamp(min=0)
    ll0 = ll[b_idx[:, None], torch.arange(T, device=device)[None, :], p0[:, None]].squeeze(-1)
    ccum = torch.cumsum(ll0, dim=1)
    for t0 in range(min(T, Dmax)):
        valid = lengths > t0
        log_a[valid, 0, t0, t0] = ccum[valid, t0]

    tau_s = max(float(tau), 1e-4)
    inv_tau = 1.0 / tau_s
    BLOCK_D = 64 if Dmax <= 64 else 128
    grid = (B * N,)

    sa = log_a.stride()
    sll = ll.stride()
    sph = phone_ids.stride()
    sdur = dur_neglog.stride()
    sbg = bigram_neglog.stride()

    for t in range(1, T):
        soft_fwd_timestep_kernel[grid](
            log_a,
            ll,
            phone_ids,
            lengths,
            phone_lens,
            dur_neglog,
            floor_frames,
            bigram_neglog,
            t,
            B,
            N,
            T,
            C,
            Dmax,
            float(lambda_d),
            float(lambda_bg),
            float(tau_s),
            float(inv_tau),
            float(ni),
            sa[0],
            sa[1],
            sa[2],
            sa[3],
            sll[0],
            sll[1],
            sll[2],
            sph[0],
            sph[1],
            sdur[0],
            sdur[1],
            sbg[0],
            sbg[1],
            BLOCK_D=BLOCK_D,
        )
    return log_a
