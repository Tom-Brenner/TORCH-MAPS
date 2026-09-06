"""CPU duration-aware hard stay/advance Viterbi (NumPy).

Intended for validation and inference. Acoustic scoring stays on GPU; costs are
copied once per utterance. Parallelize independent utterances with
``workers`` (capped at 12).
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Optional, Sequence, Tuple, Union

import numpy as np

ArrayLike = Union[np.ndarray, "torch.Tensor"]  # noqa: F821


def _as_numpy(x: ArrayLike, *, dtype=None) -> np.ndarray:
    if hasattr(x, "detach"):
        x = x.detach().cpu().numpy()
    arr = np.asarray(x)
    if dtype is not None:
        arr = arr.astype(dtype, copy=False)
    return np.ascontiguousarray(arr)


@dataclass
class HardDPResult:
    path: np.ndarray  # [T] phone ids
    switches: np.ndarray  # [N-1] switch times in frames (float)
    best_cost: float


def stay_advance_hard_cpu(
    cost: ArrayLike,
    phone_ids: ArrayLike,
    *,
    dur_neglog: Optional[ArrayLike] = None,
    floor_frames: Optional[ArrayLike] = None,
    bigram_neglog: Optional[ArrayLike] = None,
    lambda_d: float = 1.0,
    lambda_bg: float = 1.0,
) -> HardDPResult:
    """Exact duration-aware hard Viterbi on CPU.

    Parameters
    ----------
    cost : [T, C]
        Frame costs ``-log p``.
    phone_ids : [N]
        Target phone sequence (CMUdict ids).
    dur_neglog : [C, Dmax], optional
        Duration table; column ``d-1`` is duration ``d``.
    floor_frames : [C], optional
        Minimum duration in frames per phone.
    """
    cost = _as_numpy(cost, dtype=np.float64)
    phone_ids = _as_numpy(phone_ids, dtype=np.int64).reshape(-1)
    T, C = cost.shape
    N = int(phone_ids.size)
    if N < 1 or T < 1:
        raise ValueError("empty cost or phone sequence")

    if dur_neglog is None:
        path, switches, best = _hard_no_duration(
            cost,
            phone_ids,
            bigram_neglog=None
            if bigram_neglog is None
            else _as_numpy(bigram_neglog, dtype=np.float64),
            lambda_bg=lambda_bg,
        )
        return HardDPResult(path=path, switches=switches, best_cost=best)

    dur_neglog = _as_numpy(dur_neglog, dtype=np.float64)
    Dmax = int(dur_neglog.shape[1])
    floors = (
        np.ones(C, dtype=np.int64)
        if floor_frames is None
        else _as_numpy(floor_frames, dtype=np.int64)
    )
    bg_table = (
        None
        if bigram_neglog is None
        else _as_numpy(bigram_neglog, dtype=np.float64)
    )

    NEG = np.inf
    M = np.full((N, T, Dmax), NEG, dtype=np.float64)
    bp_kind = np.full((N, T, Dmax), -1, dtype=np.int8)  # 0 stay, 1 adv
    bp_d = np.full((N, T, Dmax), -1, dtype=np.int32)

    p0 = int(phone_ids[0])
    ccum = np.cumsum(cost[:, p0])
    for t in range(min(T, Dmax)):
        M[0, t, t] = ccum[t]
        bp_kind[0, t, t] = 0
        bp_d[0, t, t] = -1

    for n in range(1, N):
        pn = int(phone_ids[n])
        prev = int(phone_ids[n - 1])
        fl_prev = int(floors[prev])
        bg = (
            float(lambda_bg) * float(bg_table[prev, pn])
            if bg_table is not None
            else 0.0
        )
        cn = cost[:, pn]
        dur_prev = dur_neglog[prev]

        for t in range(1, T):
            # stay → duration >= 2 (indices 1..)
            stay_src = M[n, t - 1, :-1]
            stay_val = stay_src + cn[t]
            better = stay_val < M[n, t, 1:]
            M[n, t, 1:] = np.where(better, stay_val, M[n, t, 1:])
            bp_kind[n, t, 1:] = np.where(better, 0, bp_kind[n, t, 1:])
            bp_d[n, t, 1:] = np.where(
                better, np.arange(0, Dmax - 1, dtype=np.int32), bp_d[n, t, 1:]
            )

            # advance → duration 1 (index 0)
            prev_costs = M[n - 1, t - 1].copy()
            if fl_prev > 1:
                prev_costs[: fl_prev - 1] = NEG
            cand = prev_costs + float(lambda_d) * dur_prev + bg + cn[t]
            best_d = int(np.argmin(cand))
            best_val = float(cand[best_d])
            if best_val < M[n, t, 0]:
                M[n, t, 0] = best_val
                bp_kind[n, t, 0] = 1
                bp_d[n, t, 0] = best_d

    last = int(phone_ids[N - 1])
    fl_last = int(floors[last])
    final = M[N - 1, T - 1].copy()
    if fl_last > 1:
        final[: fl_last - 1] = NEG
    final = final + float(lambda_d) * dur_neglog[last]
    d_end = int(np.argmin(final))
    best_cost = float(final[d_end])

    path = np.empty(T, dtype=np.int64)
    n, t, d_idx = N - 1, T - 1, d_end
    while t >= 0 and n >= 0:
        path[t] = phone_ids[n]
        kind = int(bp_kind[n, t, d_idx])
        prev_d = int(bp_d[n, t, d_idx])
        if t == 0:
            path[: t + 1] = phone_ids[0]
            break
        if kind == 1:
            n = n - 1
            d_idx = prev_d
            t = t - 1
        else:
            d_idx = d_idx - 1 if d_idx > 0 else 0
            t = t - 1

    switches = _switches_from_path(path, N)
    return HardDPResult(path=path, switches=switches, best_cost=best_cost)


def _hard_no_duration(
    cost: np.ndarray,
    phone_ids: np.ndarray,
    *,
    bigram_neglog: Optional[np.ndarray],
    lambda_bg: float,
) -> Tuple[np.ndarray, np.ndarray, float]:
    T = cost.shape[0]
    N = int(phone_ids.size)
    M = np.full((N, T), np.inf, dtype=np.float64)
    bt = np.zeros((N, T), dtype=np.uint8)

    p0 = int(phone_ids[0])
    M[0, 0] = cost[0, p0]
    for t in range(1, T):
        M[0, t] = M[0, t - 1] + cost[t, p0]

    for n in range(1, N):
        pn = int(phone_ids[n])
        prev = int(phone_ids[n - 1])
        bg = (
            float(lambda_bg) * float(bigram_neglog[prev, pn])
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

    path = np.empty(T, dtype=np.int64)
    n = N - 1
    for t in range(T - 1, -1, -1):
        path[t] = phone_ids[n]
        if t == 0 or n == 0:
            continue
        if bt[n, t] == 1:
            n = n - 1
    switches = _switches_from_path(path, N)
    return path, switches, float(M[N - 1, T - 1])


def _switches_from_path(path: np.ndarray, n_phones: int) -> np.ndarray:
    switches = []
    cur = path[0]
    for t in range(1, path.shape[0]):
        if path[t] != cur:
            switches.append(float(t))
            cur = path[t]
    out = np.asarray(switches, dtype=np.float64)
    if out.size > max(n_phones - 1, 0):
        out = out[: n_phones - 1]
    return out


def _decode_one(args: tuple) -> HardDPResult:
    return stay_advance_hard_cpu(*args[0], **args[1])


def stay_advance_hard_cpu_batch(
    costs: Sequence[ArrayLike],
    phone_ids_list: Sequence[ArrayLike],
    *,
    dur_neglog: Optional[ArrayLike] = None,
    floor_frames: Optional[ArrayLike] = None,
    bigram_neglog: Optional[ArrayLike] = None,
    lambda_d: float = 1.0,
    lambda_bg: float = 1.0,
    workers: int = 12,
) -> list[HardDPResult]:
    """Decode many utterances on CPU; ``workers`` capped at 12."""
    import os

    n = len(costs)
    if n != len(phone_ids_list):
        raise ValueError("costs and phone_ids_list length mismatch")
    workers = max(1, min(int(workers), 12, n, os.cpu_count() or 1))

    # Materialize shared tables once (thread-safe read-only).
    shared = {
        "dur_neglog": None if dur_neglog is None else _as_numpy(dur_neglog, dtype=np.float64),
        "floor_frames": None
        if floor_frames is None
        else _as_numpy(floor_frames, dtype=np.int64),
        "bigram_neglog": None
        if bigram_neglog is None
        else _as_numpy(bigram_neglog, dtype=np.float64),
        "lambda_d": float(lambda_d),
        "lambda_bg": float(lambda_bg),
    }
    jobs = []
    for cost, phones in zip(costs, phone_ids_list):
        jobs.append(
            (
                (_as_numpy(cost, dtype=np.float64), _as_numpy(phones, dtype=np.int64)),
                shared,
            )
        )

    if workers == 1 or n == 1:
        return [_decode_one(j) for j in jobs]

    with ThreadPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(_decode_one, jobs))
