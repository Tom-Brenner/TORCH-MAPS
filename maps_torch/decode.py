"""GPU stay/advance DP with duration floors/priors + bigram transition costs."""

from __future__ import annotations

import torch
from torch import Tensor


def _neg_inf_like(x: Tensor) -> Tensor:
    return torch.tensor(torch.finfo(x.dtype).min / 4, device=x.device, dtype=x.dtype)


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
    """Hard Viterbi stay/advance.

    ``cost``: [T, C] = -log p
    ``phone_ids``: [N]

    If ``dur_neglog`` is set ([C, Dmax], column d-1 → duration d), uses state
    (n, d) with hard floor constraints. Bigram cost is added on advance.
    Returns ``(path_phone_ids [T], M_best [N, T])``.
    """
    T, _C = cost.shape
    N = int(phone_ids.numel())
    phone_ids = phone_ids.long()
    device = cost.device
    dtype = cost.dtype

    if dur_neglog is None:
        return _hard_no_duration(
            cost, phone_ids, bigram_neglog=bigram_neglog, lambda_bg=lambda_bg
        )

    Dmax = int(dur_neglog.shape[1])
    neg_inf = float("inf")
    M = torch.full((N, T, Dmax), neg_inf, device=device, dtype=dtype)
    bp_n = torch.full((N, T, Dmax), -1, device=device, dtype=torch.long)  # 0 stay / 1 adv
    bp_d = torch.full((N, T, Dmax), -1, device=device, dtype=torch.long)

    p0 = int(phone_ids[0])
    fl0 = int(floor_frames[p0]) if floor_frames is not None else 1
    # cumulative frame cost on phone 0
    ccum = torch.cumsum(cost[:, p0], dim=0)
    for t in range(min(T, Dmax)):
        d = t + 1
        M[0, t, d - 1] = ccum[t]
        bp_n[0, t, d - 1] = 0

    for n in range(1, N):
        pn = int(phone_ids[n])
        prev = int(phone_ids[n - 1])
        fl = int(floor_frames[pn]) if floor_frames is not None else 1
        fl_prev = int(floor_frames[prev]) if floor_frames is not None else 1
        bg = (
            lambda_bg * bigram_neglog[prev, pn]
            if bigram_neglog is not None
            else torch.zeros((), device=device, dtype=dtype)
        )
        cn = cost[:, pn]

        for t in range(1, T):
            # stay: d >= 2
            # from (n, t-1, d-1)
            stay_src = M[n, t - 1, :-1]
            stay_val = stay_src + cn[t]
            better_stay = stay_val < M[n, t, 1:]
            M[n, t, 1:] = torch.where(better_stay, stay_val, M[n, t, 1:])
            bp_n[n, t, 1:] = torch.where(
                better_stay, torch.zeros_like(bp_n[n, t, 1:]), bp_n[n, t, 1:]
            )
            bp_d[n, t, 1:] = torch.where(
                better_stay,
                torch.arange(0, Dmax - 1, device=device),
                bp_d[n, t, 1:],
            )

            # advance → d=1: finish prev with duration d_prev >= floor
            prev_costs = M[n - 1, t - 1].clone()
            prev_costs[: fl_prev - 1] = neg_inf
            cand = prev_costs + lambda_d * dur_neglog[prev] + bg + cn[t]
            best_d = int(torch.argmin(cand).item())
            best_val = cand[best_d]
            if best_val < M[n, t, 0]:
                M[n, t, 0] = best_val
                bp_n[n, t, 0] = 1
                bp_d[n, t, 0] = best_d

    # terminate: last phone must have d >= floor; also charge its duration prior
    last = int(phone_ids[N - 1])
    fl_last = int(floor_frames[last]) if floor_frames is not None else 1
    final = M[N - 1, T - 1].clone()
    final[: fl_last - 1] = neg_inf
    final = final + lambda_d * dur_neglog[last]
    d_end = int(torch.argmin(final).item())

    # traceback
    path = torch.empty(T, dtype=torch.long, device=device)
    n, t, d_idx = N - 1, T - 1, d_end
    while t >= 0 and n >= 0:
        path[t] = phone_ids[n]
        kind = int(bp_n[n, t, d_idx].item())
        prev_d = int(bp_d[n, t, d_idx].item())
        if t == 0:
            path[: t + 1] = phone_ids[0]
            break
        if kind == 1:  # advance into this cell
            n = n - 1
            d_idx = prev_d
            t = t - 1
        else:  # stay
            d_idx = d_idx - 1 if d_idx > 0 else 0
            t = t - 1
            if d_idx < 0:
                d_idx = 0

    # fill any holes
    for tt in range(T):
        if path[tt] < 0:
            path[tt] = phone_ids[0]

    M_best = M.min(dim=-1).values
    return path, M_best


def _hard_no_duration(
    cost: Tensor,
    phone_ids: Tensor,
    *,
    bigram_neglog: Tensor | None,
    lambda_bg: float,
) -> tuple[Tensor, Tensor]:
    T = cost.shape[0]
    N = int(phone_ids.numel())
    device = cost.device
    dtype = cost.dtype
    M = torch.full((N, T), float("inf"), device=device, dtype=dtype)
    bt = torch.zeros((N, T), dtype=torch.uint8, device=device)  # 0 stay 1 adv

    p0 = int(phone_ids[0])
    M[0, 0] = cost[0, p0]
    for t in range(1, T):
        M[0, t] = M[0, t - 1] + cost[t, p0]

    for n in range(1, N):
        pn = int(phone_ids[n])
        prev = int(phone_ids[n - 1])
        bg = (
            lambda_bg * bigram_neglog[prev, pn]
            if bigram_neglog is not None
            else 0.0
        )
        for t in range(1, T):
            stay = M[n, t - 1]
            adv = M[n - 1, t - 1] + bg
            if stay <= adv:
                M[n, t] = cost[t, pn] + stay
                bt[n, t] = 0
            else:
                M[n, t] = cost[t, pn] + adv
                bt[n, t] = 1

    path = torch.empty(T, dtype=torch.long, device=device)
    n = N - 1
    for t in range(T - 1, -1, -1):
        path[t] = phone_ids[n]
        if t == 0 or n == 0:
            continue
        if bt[n, t] == 1:
            n = n - 1
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
    """Log-alpha [N, T] (duration-marginalized if duration prior given)."""
    T = cost.shape[0]
    N = int(phone_ids.numel())
    phone_ids = phone_ids.long()
    device = cost.device
    dtype = cost.dtype
    ni = _neg_inf_like(cost)
    ll = -cost
    inv_tau = 1.0 / max(float(tau), 1e-4)

    if dur_neglog is None:
        log_a = torch.full((N, T), ni, device=device, dtype=dtype)
        p0 = int(phone_ids[0])
        log_a[0, 0] = ll[0, p0]
        for t in range(1, T):
            log_a[0, t] = log_a[0, t - 1] + ll[t, p0]
        for n in range(1, N):
            pn = int(phone_ids[n])
            prev = int(phone_ids[n - 1])
            bg = (
                lambda_bg * float(bigram_neglog[prev, pn])
                if bigram_neglog is not None
                else 0.0
            )
            for t in range(1, T):
                stay = log_a[n, t - 1]
                adv = log_a[n - 1, t - 1] - bg
                log_a[n, t] = ll[t, pn] + tau * torch.logsumexp(
                    torch.stack([stay, adv]) * inv_tau, dim=0
                )
        return log_a

    Dmax = int(dur_neglog.shape[1])
    log_a = torch.full((N, T, Dmax), ni, device=device, dtype=dtype)
    p0 = int(phone_ids[0])
    ccum = torch.cumsum(ll[:, p0], dim=0)
    for t in range(min(T, Dmax)):
        log_a[0, t, t] = ccum[t]

    for n in range(1, N):
        pn = int(phone_ids[n])
        prev = int(phone_ids[n - 1])
        fl_prev = int(floor_frames[prev]) if floor_frames is not None else 1
        bg = (
            lambda_bg * float(bigram_neglog[prev, pn])
            if bigram_neglog is not None
            else 0.0
        )
        for t in range(1, T):
            # stay
            log_a[n, t, 1:] = ll[t, pn] + log_a[n, t - 1, :-1]
            # advance
            prev_lex = log_a[n - 1, t - 1].clone()
            prev_lex[: fl_prev - 1] = ni
            scored = prev_lex - lambda_d * dur_neglog[prev] - bg
            log_a[n, t, 0] = ll[t, pn] + tau * torch.logsumexp(scored * inv_tau, dim=0)

    # marginalize d (mask below floor)
    out = torch.full((N, T), ni, device=device, dtype=dtype)
    for n in range(N):
        fl = int(floor_frames[phone_ids[n]]) if floor_frames is not None else 1
        row = log_a[n].clone()
        row[:, : fl - 1] = ni
        out[n] = torch.logsumexp(row, dim=-1)
    return out


def stay_advance_soft_backward(
    cost: Tensor,
    phone_ids: Tensor,
    tau: float,
    *,
    bigram_neglog: Tensor | None = None,
    lambda_bg: float = 1.0,
) -> Tensor:
    """Log-beta [N, T] for the no-duration / marginalized topology."""
    T = cost.shape[0]
    N = int(phone_ids.numel())
    phone_ids = phone_ids.long()
    device = cost.device
    dtype = cost.dtype
    ni = _neg_inf_like(cost)
    ll = -cost
    log_b = torch.full((N, T), ni, device=device, dtype=dtype)
    log_b[N - 1, T - 1] = 0.0
    inv_tau = 1.0 / max(float(tau), 1e-4)

    for t in range(T - 2, -1, -1):
        for n in range(N - 1, -1, -1):
            pn = int(phone_ids[n])
            terms = [log_b[n, t + 1] + ll[t + 1, pn]]  # stay
            if n + 1 < N:
                nxt = int(phone_ids[n + 1])
                bg = (
                    lambda_bg * float(bigram_neglog[pn, nxt])
                    if bigram_neglog is not None
                    else 0.0
                )
                terms.append(log_b[n + 1, t + 1] + ll[t + 1, nxt] - bg)
            log_b[n, t] = tau * torch.logsumexp(torch.stack(terms) * inv_tau, dim=0)
    return log_b


def expected_boundaries(log_alpha: Tensor, log_beta: Tensor) -> Tensor:
    """E[switch time] in frames for each boundary i=0..N-2.

    Uses forward-backward switch posteriors:
    at time t, leaving phone i for i+1 is scored by α[i,t] + β[i+1,t+1] mass.
    """
    N, T = log_alpha.shape
    device = log_alpha.device
    dtype = log_alpha.dtype
    log_Z = torch.logsumexp(log_alpha[:, T - 1] + log_beta[:, T - 1], dim=0)
    times = torch.arange(T, device=device, dtype=dtype)
    outs = []
    ni = _neg_inf_like(log_alpha)
    for i in range(N - 1):
        # switch after frame t (t=0..T-2): mass ∝ α[i,t] + β[i+1,t+1]
        log_m = torch.full((T,), ni, device=device, dtype=dtype)
        log_m[:-1] = log_alpha[i, :-1] + log_beta[i + 1, 1:] - log_Z
        w = torch.softmax(log_m, dim=0)
        outs.append((times * w).sum())
    return torch.stack(outs)
