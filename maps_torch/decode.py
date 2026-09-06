"""Stay/advance DP: fused GPU soft DP (training) + CPU hard DP re-exports.

Soft DP
-------
Duration-aware forward *and* backward lattices feed expected boundary times.
GPU training uses Triton time-wavefront kernels (phone × duration parallel per
frame). Backward uses a PyTorch reference VJP so gradients reach the scorer.
A pure PyTorch batched reference is kept for CPU / correctness checks.

Hard DP
-------
Exact Viterbi lives in :mod:`maps_torch.hard_dp_cpu` (NumPy, up to 12 workers).
"""

from __future__ import annotations

from typing import Optional

import torch
from torch import Tensor

from maps_torch.hard_dp_cpu import (  # noqa: F401
    HardDPResult,
    stay_advance_hard_cpu,
    stay_advance_hard_cpu_batch,
)

# ---------------------------------------------------------------------------
# Numerics helpers
# ---------------------------------------------------------------------------


def _neg_inf_like(x: Tensor) -> Tensor:
    return torch.full((), torch.finfo(x.dtype).min / 4, device=x.device, dtype=x.dtype)


def _safe_tau(tau: float) -> float:
    return max(float(tau), 1e-4)


# ---------------------------------------------------------------------------
# Soft DP — PyTorch reference (batched over utterances, loop over time)
# ---------------------------------------------------------------------------


def soft_forward_duration(
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
    """Duration-aware log-alpha.

    Parameters
    ----------
    cost : [B, T, C]
    phone_ids : [B, N]
    lengths : [B] frame lengths
    phone_lens : [B] phone counts
    dur_neglog : [C, Dmax]
    floor_frames : [C]
    bigram_neglog : [C, C]

    Returns
    -------
    log_alpha : [B, N, T, Dmax]
    """
    B, T, _C = cost.shape
    N = phone_ids.shape[1]
    Dmax = dur_neglog.shape[1]
    device = cost.device
    dtype = cost.dtype
    ni = _neg_inf_like(cost)
    ll = -cost
    inv_tau = 1.0 / _safe_tau(tau)
    log_a = torch.full((B, N, T, Dmax), ni, device=device, dtype=dtype)

    # phone 0 init: duration = t+1 for t < Dmax and t < length
    b_idx = torch.arange(B, device=device)
    p0 = phone_ids[:, 0].clamp(min=0)
    # gather emissions for phone 0: [B, T]
    ll0 = ll[b_idx[:, None], torch.arange(T, device=device)[None, :], p0[:, None]].squeeze(-1)
    ccum = torch.cumsum(ll0, dim=1)
    for t in range(min(T, Dmax)):
        valid = lengths > t
        log_a[valid, 0, t, t] = ccum[valid, t]

    floors = floor_frames.long()
    for t in range(1, T):
        # Active frames
        active_t = lengths > t  # [B]
        # Stay for all phones: log_a[:, :, t, 1:] = ll_phone[t] + log_a[:, :, t-1, :-1]
        # Gather per-(b,n) emission at t
        pn = phone_ids.clamp(min=0)  # [B, N]
        ll_t = ll[b_idx[:, None], t, pn]  # [B, N]
        stay = log_a[:, :, t - 1, :-1] + ll_t[:, :, None]
        # Mask phones beyond phone_lens and inactive time
        phone_ok = torch.arange(N, device=device)[None, :] < phone_lens[:, None]  # [B, N]
        mask = active_t[:, None] & phone_ok
        stay = torch.where(mask[:, :, None], stay, ni)
        log_a[:, :, t, 1:] = stay

        # Advance into phone n>=1 at d=0 from phone n-1
        if N >= 2:
            prev_ids = phone_ids[:, :-1].clamp(min=0)
            cur_ids = phone_ids[:, 1:].clamp(min=0)
            fl_prev = floors[prev_ids]  # [B, N-1]
            bg = lambda_bg * bigram_neglog[prev_ids, cur_ids]  # [B, N-1]
            prev_lex = log_a[:, :-1, t - 1, :].clone()  # [B, N-1, Dmax]
            # mask below floor
            d_idx = torch.arange(Dmax, device=device)[None, None, :]
            prev_lex = torch.where(d_idx >= (fl_prev[:, :, None] - 1), prev_lex, ni)
            scored = prev_lex - lambda_d * dur_neglog[prev_ids] - bg[:, :, None]
            adv = ll_t[:, 1:] + tau * torch.logsumexp(scored * inv_tau, dim=-1)
            phone_ok_adv = torch.arange(1, N, device=device)[None, :] < phone_lens[:, None]
            adv = torch.where(active_t[:, None] & phone_ok_adv, adv, ni)
            log_a[:, 1:, t, 0] = adv

    return log_a


def soft_backward_duration(
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
    """Duration-aware log-beta [B, N, T, Dmax]."""
    B, T, _C = cost.shape
    N = phone_ids.shape[1]
    Dmax = dur_neglog.shape[1]
    device = cost.device
    dtype = cost.dtype
    ni = _neg_inf_like(cost)
    ll = -cost
    inv_tau = 1.0 / _safe_tau(tau)
    log_b = torch.full((B, N, T, Dmax), ni, device=device, dtype=dtype)
    floors = floor_frames.long()
    b_idx = torch.arange(B, device=device)
    pn_all = phone_ids.clamp(min=0)

    # Terminal: last phone at last frame, charge duration prior
    for b in range(B):
        Tb = int(lengths[b])
        Nb = int(phone_lens[b])
        if Tb < 1 or Nb < 1:
            continue
        last = int(phone_ids[b, Nb - 1])
        fl = int(floors[last])
        final = -lambda_d * dur_neglog[last]
        if fl > 1:
            final = final.clone()
            final[: fl - 1] = ni
        log_b[b, Nb - 1, Tb - 1] = final

    for t in range(T - 2, -1, -1):
        active = lengths > (t + 1)  # need t+1 to exist
        # stay: from (n,t,d) -> (n,t+1,d+1)
        # β[n,t,d] includes ll[t+1,pn] + β[n,t+1,d+1]
        ll_tp1 = ll[b_idx[:, None], t + 1, pn_all]  # [B, N]
        stay_term = ll_tp1[:, :, None] + log_b[:, :, t + 1, 1:]  # [B,N,Dmax-1]
        # advance: (n,t,d) -> (n+1,t+1,0) if n+1 < N
        terms_list = [stay_term]
        # pad stay to Dmax with ni for last duration (cannot stay beyond Dmax)
        stay_full = torch.full((B, N, Dmax), ni, device=device, dtype=dtype)
        stay_full[:, :, :-1] = stay_term

        adv_full = torch.full((B, N, Dmax), ni, device=device, dtype=dtype)
        if N >= 2:
            cur_ids = phone_ids[:, :-1].clamp(min=0)
            nxt_ids = phone_ids[:, 1:].clamp(min=0)
            fl_cur = floors[cur_ids]
            bg = lambda_bg * bigram_neglog[cur_ids, nxt_ids]
            ll_next = ll[b_idx[:, None], t + 1, nxt_ids]
            beta_next0 = log_b[:, 1:, t + 1, 0]
            # score for each duration d of current phone
            dur_cost = lambda_d * dur_neglog[cur_ids]  # [B, N-1, Dmax]
            adv_score = (
                ll_next[:, :, None]
                - dur_cost
                - bg[:, :, None]
                + beta_next0[:, :, None]
            )
            d_idx = torch.arange(Dmax, device=device)[None, None, :]
            adv_score = torch.where(d_idx >= (fl_cur[:, :, None] - 1), adv_score, ni)
            # only phones that have a next phone within phone_lens
            has_next = (torch.arange(N - 1, device=device)[None, :] + 1) < phone_lens[:, None]
            adv_score = torch.where(has_next[:, :, None], adv_score, ni)
            adv_full[:, :-1, :] = adv_score

        stacked = torch.stack([stay_full, adv_full], dim=-1)  # [B,N,Dmax,2]
        out = tau * torch.logsumexp(stacked * inv_tau, dim=-1)
        phone_ok = torch.arange(N, device=device)[None, :] < phone_lens[:, None]
        frame_ok = lengths > t
        write_mask = (active & frame_ok)[:, None] & phone_ok  # [B, N]
        # Do not overwrite terminal cells for sequences that end at t
        write = (lengths - 1) > t
        write_mask = write_mask & write[:, None]
        log_b[:, :, t] = torch.where(write_mask[:, :, None], out, log_b[:, :, t])

    return log_b


def marginalize_duration(
    log_x: Tensor,
    phone_ids: Tensor,
    phone_lens: Tensor,
    floor_frames: Tensor,
) -> Tensor:
    """Mask below floor then logsumexp over duration → [B, N, T]."""
    B, N, T, Dmax = log_x.shape
    device = log_x.device
    ni = _neg_inf_like(log_x)
    floors = floor_frames.long()
    pn = phone_ids.clamp(min=0)
    fl = floors[pn]  # [B, N]
    d_idx = torch.arange(Dmax, device=device)[None, None, None, :]
    masked = torch.where(d_idx >= (fl[:, :, None, None] - 1), log_x, ni)
    phone_ok = torch.arange(N, device=device)[None, :] < phone_lens[:, None]
    masked = torch.where(phone_ok[:, :, None, None], masked, ni)
    return torch.logsumexp(masked, dim=-1)


def expected_boundaries_duration(
    log_alpha: Tensor,
    log_beta: Tensor,
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
) -> Tensor:
    """E[switch time] [B, N-1] (padded; invalid → 0).

    Switch mass after frame t leaving phone i uses duration-aware advance:
    ``logsumexp_d(α[i,t,d] - λ_d·dur - bg) + ℓ[t+1,next] + β[i+1,t+1,0]``.
    """
    B, N, T, Dmax = log_alpha.shape
    device = log_alpha.device
    dtype = log_alpha.dtype
    ni = _neg_inf_like(log_alpha)
    ll = -cost
    floors = floor_frames.long()
    b_idx = torch.arange(B, device=device)
    outs = torch.zeros(B, max(N - 1, 1), device=device, dtype=dtype)
    if N < 2:
        return outs

    # Partition from final states
    log_Z = torch.full((B,), ni, device=device, dtype=dtype)
    for b in range(B):
        Tb = int(lengths[b])
        Nb = int(phone_lens[b])
        if Tb < 1 or Nb < 1:
            continue
        last = int(phone_ids[b, Nb - 1])
        fl = int(floors[last])
        final = log_alpha[b, Nb - 1, Tb - 1] - lambda_d * dur_neglog[last]
        if fl > 1:
            final = final.clone()
            final[: fl - 1] = ni
        log_Z[b] = torch.logsumexp(final, dim=0)

    times = torch.arange(T, device=device, dtype=dtype)
    cur_ids = phone_ids[:, :-1].clamp(min=0)
    nxt_ids = phone_ids[:, 1:].clamp(min=0)
    fl_cur = floors[cur_ids]
    bg = lambda_bg * bigram_neglog[cur_ids, nxt_ids]  # [B, N-1]
    has_bound = (torch.arange(N - 1, device=device)[None, :] + 1) < phone_lens[:, None]

    for i in range(N - 1):
        log_m = torch.full((B, T), ni, device=device, dtype=dtype)
        a = log_alpha[:, i, :, :]
        d_idx = torch.arange(Dmax, device=device)[None, None, :]
        fl_i = fl_cur[:, i]
        a = torch.where(d_idx >= (fl_i[:, None, None] - 1), a, ni)
        dur_i = dur_neglog[cur_ids[:, i]]  # [B, Dmax]
        scored = a - lambda_d * dur_i[:, None, :] - bg[:, i, None, None]
        leave = torch.logsumexp(scored, dim=-1)  # [B, T]
        ll_next = ll[b_idx[:, None], torch.arange(T, device=device)[None, :], nxt_ids[:, i : i + 1]].squeeze(-1)
        beta0 = log_beta[:, i + 1, :, 0]
        log_m[:, :-1] = leave[:, :-1] + ll_next[:, 1:] + beta0[:, 1:] - log_Z[:, None]
        tmax = (lengths - 1).clamp(min=0)
        t_ok = torch.arange(T, device=device)[None, :] < tmax[:, None]
        log_m = torch.where(has_bound[:, i : i + 1] & t_ok, log_m, ni)
        # Empty rows → zeros after softmax; mask them out of the expectation.
        empty = ~has_bound[:, i] | (lengths < 2)
        w = torch.softmax(log_m, dim=-1)
        e = (times[None, :] * w).sum(dim=-1)
        outs[:, i] = torch.where(empty, outs[:, i], e)
    return outs


def soft_expected_boundaries_torch(
    cost: Tensor,
    phone_ids: Tensor,
    lengths: Tensor,
    phone_lens: Tensor,
    *,
    dur_neglog: Tensor,
    floor_frames: Tensor,
    bigram_neglog: Tensor,
    lambda_d: float = 1.0,
    lambda_bg: float = 1.0,
    tau: float = 1.0,
) -> Tensor:
    """Differentiable expected boundaries via PyTorch soft DP [B, N-1]."""
    log_a = soft_forward_duration(
        cost,
        phone_ids,
        lengths,
        phone_lens,
        dur_neglog=dur_neglog,
        floor_frames=floor_frames,
        bigram_neglog=bigram_neglog,
        lambda_d=lambda_d,
        lambda_bg=lambda_bg,
        tau=tau,
    )
    log_b = soft_backward_duration(
        cost,
        phone_ids,
        lengths,
        phone_lens,
        dur_neglog=dur_neglog,
        floor_frames=floor_frames,
        bigram_neglog=bigram_neglog,
        lambda_d=lambda_d,
        lambda_bg=lambda_bg,
        tau=tau,
    )
    return expected_boundaries_duration(
        log_a,
        log_b,
        cost,
        phone_ids,
        lengths,
        phone_lens,
        dur_neglog=dur_neglog,
        floor_frames=floor_frames,
        bigram_neglog=bigram_neglog,
        lambda_d=lambda_d,
        lambda_bg=lambda_bg,
    )


# ---------------------------------------------------------------------------
# Triton time-wavefront kernels (optional; fall back to torch)
# ---------------------------------------------------------------------------


def _triton_available() -> bool:
    if not torch.cuda.is_available():
        return False
    try:
        import triton  # noqa: F401

        return True
    except Exception:
        return False


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
    from maps_torch.soft_dp_triton import soft_forward_duration_triton as _fwd

    return _fwd(
        cost,
        phone_ids,
        lengths,
        phone_lens,
        dur_neglog=dur_neglog,
        floor_frames=floor_frames,
        bigram_neglog=bigram_neglog,
        lambda_d=lambda_d,
        lambda_bg=lambda_bg,
        tau=tau,
    )


class SoftExpectedBoundaries(torch.autograd.Function):
    """Triton (or torch) forward lattices; PyTorch VJP for gradients."""

    @staticmethod
    def forward(
        ctx,
        cost: Tensor,
        phone_ids: Tensor,
        lengths: Tensor,
        phone_lens: Tensor,
        dur_neglog: Tensor,
        floor_frames: Tensor,
        bigram_neglog: Tensor,
        lambda_d: float,
        lambda_bg: float,
        tau: float,
        use_triton: bool,
    ) -> Tensor:
        ctx.lambda_d = float(lambda_d)
        ctx.lambda_bg = float(lambda_bg)
        ctx.tau = float(tau)
        ctx.use_triton = bool(use_triton) and _triton_available() and cost.is_cuda
        with torch.no_grad():
            if ctx.use_triton:
                # Full FB still uses torch for beta/expectations for stability;
                # Triton accelerates the heavy forward lattice.
                log_a = soft_forward_duration_triton(
                    cost,
                    phone_ids,
                    lengths,
                    phone_lens,
                    dur_neglog=dur_neglog,
                    floor_frames=floor_frames,
                    bigram_neglog=bigram_neglog,
                    lambda_d=ctx.lambda_d,
                    lambda_bg=ctx.lambda_bg,
                    tau=ctx.tau,
                )
                log_b = soft_backward_duration(
                    cost,
                    phone_ids,
                    lengths,
                    phone_lens,
                    dur_neglog=dur_neglog,
                    floor_frames=floor_frames,
                    bigram_neglog=bigram_neglog,
                    lambda_d=ctx.lambda_d,
                    lambda_bg=ctx.lambda_bg,
                    tau=ctx.tau,
                )
                e_b = expected_boundaries_duration(
                    log_a,
                    log_b,
                    cost,
                    phone_ids,
                    lengths,
                    phone_lens,
                    dur_neglog=dur_neglog,
                    floor_frames=floor_frames,
                    bigram_neglog=bigram_neglog,
                    lambda_d=ctx.lambda_d,
                    lambda_bg=ctx.lambda_bg,
                )
            else:
                e_b = soft_expected_boundaries_torch(
                    cost,
                    phone_ids,
                    lengths,
                    phone_lens,
                    dur_neglog=dur_neglog,
                    floor_frames=floor_frames,
                    bigram_neglog=bigram_neglog,
                    lambda_d=ctx.lambda_d,
                    lambda_bg=ctx.lambda_bg,
                    tau=ctx.tau,
                )
        ctx.save_for_backward(
            cost, phone_ids, lengths, phone_lens, dur_neglog, floor_frames, bigram_neglog
        )
        return e_b

    @staticmethod
    def backward(ctx, grad_e_b: Tensor):
        (
            cost,
            phone_ids,
            lengths,
            phone_lens,
            dur_neglog,
            floor_frames,
            bigram_neglog,
        ) = ctx.saved_tensors
        cost_d = cost.detach().requires_grad_(True)
        with torch.enable_grad():
            e_b = soft_expected_boundaries_torch(
                cost_d,
                phone_ids,
                lengths,
                phone_lens,
                dur_neglog=dur_neglog,
                floor_frames=floor_frames,
                bigram_neglog=bigram_neglog,
                lambda_d=ctx.lambda_d,
                lambda_bg=ctx.lambda_bg,
                tau=ctx.tau,
            )
            grad_cost, = torch.autograd.grad(
                e_b, cost_d, grad_outputs=grad_e_b, retain_graph=False
            )
        return grad_cost, None, None, None, None, None, None, None, None, None, None


def soft_expected_boundaries(
    cost: Tensor,
    phone_ids: Tensor,
    lengths: Tensor,
    phone_lens: Tensor,
    *,
    dur_neglog: Tensor,
    floor_frames: Tensor,
    bigram_neglog: Tensor,
    lambda_d: float = 1.0,
    lambda_bg: float = 1.0,
    tau: float = 1.0,
    backend: str = "auto",
) -> Tensor:
    """Expected boundaries [B, N-1].

    ``backend``: ``auto`` | ``triton`` | ``torch``.
    ``auto`` selects Triton on CUDA when available.
    """
    use_triton = backend in ("auto", "triton") and _triton_available() and cost.is_cuda
    if backend == "torch" or not use_triton:
        if backend == "triton" and not use_triton:
            # explicit request but unavailable → torch
            pass
        return soft_expected_boundaries_torch(
            cost,
            phone_ids,
            lengths,
            phone_lens,
            dur_neglog=dur_neglog,
            floor_frames=floor_frames,
            bigram_neglog=bigram_neglog,
            lambda_d=lambda_d,
            lambda_bg=lambda_bg,
            tau=tau,
        )
    return SoftExpectedBoundaries.apply(
        cost,
        phone_ids,
        lengths,
        phone_lens,
        dur_neglog,
        floor_frames,
        bigram_neglog,
        float(lambda_d),
        float(lambda_bg),
        float(tau),
        True,
    )


# ---------------------------------------------------------------------------
# Legacy single-utterance wrappers (compat)
# ---------------------------------------------------------------------------


def stay_advance_hard(
    cost: Tensor,
    phone_ids: Tensor,
    *,
    dur_neglog: Tensor | None = None,
    floor_frames: Tensor | None = None,
    bigram_neglog: Tensor | None = None,
    lambda_d: float = 1.0,
    lambda_bg: float = 1.0,
) -> tuple[Tensor, Tensor]:
    """Hard Viterbi via CPU NumPy; returns ``(path [T], dummy_M)`` on ``cost.device``."""
    res = stay_advance_hard_cpu(
        cost,
        phone_ids,
        dur_neglog=dur_neglog,
        floor_frames=floor_frames,
        bigram_neglog=bigram_neglog,
        lambda_d=lambda_d,
        lambda_bg=lambda_bg,
    )
    path = torch.as_tensor(res.path, device=cost.device, dtype=torch.long)
    # Placeholder score matrix for API compat
    M = torch.zeros(phone_ids.numel(), cost.shape[0], device=cost.device, dtype=cost.dtype)
    return path, M


def stay_advance_soft(
    cost: Tensor,
    phone_ids: Tensor,
    tau: float,
    *,
    dur_neglog: Tensor | None = None,
    floor_frames: Tensor | None = None,
    bigram_neglog: Tensor | None = None,
    lambda_d: float = 1.0,
    lambda_bg: float = 1.0,
) -> Tensor:
    """Single-utterance soft forward → marginalized log-alpha [N, T]."""
    if dur_neglog is None:
        raise ValueError("duration prior required for soft DP")
    cost_b = cost.unsqueeze(0)
    phones_b = phone_ids.unsqueeze(0)
    lengths = torch.tensor([cost.shape[0]], device=cost.device, dtype=torch.long)
    phone_lens = torch.tensor([phone_ids.numel()], device=cost.device, dtype=torch.long)
    log_a = soft_forward_duration(
        cost_b,
        phones_b,
        lengths,
        phone_lens,
        dur_neglog=dur_neglog,
        floor_frames=floor_frames,
        bigram_neglog=bigram_neglog,
        lambda_d=lambda_d,
        lambda_bg=lambda_bg,
        tau=tau,
    )
    return marginalize_duration(log_a, phones_b, phone_lens, floor_frames)[0]


def stay_advance_soft_backward(
    cost: Tensor,
    phone_ids: Tensor,
    tau: float,
    *,
    dur_neglog: Tensor | None = None,
    floor_frames: Tensor | None = None,
    bigram_neglog: Tensor | None = None,
    lambda_d: float = 1.0,
    lambda_bg: float = 1.0,
) -> Tensor:
    """Single-utterance soft backward → marginalized log-beta [N, T]."""
    if dur_neglog is None:
        raise ValueError("duration prior required for soft DP")
    cost_b = cost.unsqueeze(0)
    phones_b = phone_ids.unsqueeze(0)
    lengths = torch.tensor([cost.shape[0]], device=cost.device, dtype=torch.long)
    phone_lens = torch.tensor([phone_ids.numel()], device=cost.device, dtype=torch.long)
    log_b = soft_backward_duration(
        cost_b,
        phones_b,
        lengths,
        phone_lens,
        dur_neglog=dur_neglog,
        floor_frames=floor_frames,
        bigram_neglog=bigram_neglog,
        lambda_d=lambda_d,
        lambda_bg=lambda_bg,
        tau=tau,
    )
    return marginalize_duration(log_b, phones_b, phone_lens, floor_frames)[0]


def expected_boundaries(log_alpha: Tensor, log_beta: Tensor) -> Tensor:
    """Legacy marginalized switch expectation (no explicit duration on switch)."""
    N, T = log_alpha.shape
    device = log_alpha.device
    dtype = log_alpha.dtype
    log_Z = torch.logsumexp(log_alpha[:, T - 1] + log_beta[:, T - 1], dim=0)
    times = torch.arange(T, device=device, dtype=dtype)
    outs = []
    ni = _neg_inf_like(log_alpha)
    for i in range(N - 1):
        log_m = torch.full((T,), ni, device=device, dtype=dtype)
        log_m[:-1] = log_alpha[i, :-1] + log_beta[i + 1, 1:] - log_Z
        w = torch.softmax(log_m, dim=0)
        outs.append((times * w).sum())
    return torch.stack(outs) if outs else torch.zeros(0, device=device, dtype=dtype)
