#!/usr/bin/env python3
"""Estimate CMUdict-39 duration priors + bigram transitions from train.npz only.

Duration model
--------------
* Durations come from human ``boundaries_sec`` on the collapsed phone sequence
  (``phone_ids``), converted to frames at 10 ms.
* Hard floors (enforced by zeroing mass below the floor, then renormalizing):
    - S, SH, Z, CH → 40 ms (4 frames)
    - all other phones → 20 ms (2 frames)
* Per-phone Poisson rate ``λ`` = mean training duration (frames, after floor
  clipping of observations used for the mean: max(d, floor)).
* Discrete probs on ``d = 1..Dmax`` from Poisson PMF; ``d = Dmax`` is an
  overflow bucket holding the Poisson tail ``P(D >= Dmax)``.

Bigram model
------------
* Counts of consecutive ``phone_ids[i] → phone_ids[i+1]`` on the training set
  only, Laplace (+1) smoothed.
* Stored as ``bigram_neglog[prev, next] = -log P(next|prev)``.

Writes ``duration_prior.pt`` with:
  dur_neglog [39, Dmax], floor_frames [39], poisson_lambda [39],
  bigram_neglog [39, 39], phones, frame_interval_sec, dmax
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import torch

import sys

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from maps_torch.phones_cmudict import CMUDICT_39, CMUDICT_TO_ID, N_CMUDICT  # noqa: E402

FRAME_INTERVAL = 0.01
LONG_FRICATIVES = {"S", "SH", "Z", "CH"}


def floor_frames_for_phone(phone: str) -> int:
    """40 ms for sibilants/CH; 20 ms otherwise (at 10 ms hop)."""
    if phone in LONG_FRICATIVES:
        return 4
    return 2


def poisson_pmf(k: np.ndarray, lam: float) -> np.ndarray:
    # P(K=k) = e^{-λ} λ^k / k!
    logp = -lam + k * math.log(max(lam, 1e-12)) - np.array([math.lgamma(int(x) + 1) for x in k])
    return np.exp(logp)


def poisson_sf_ge(k: int, lam: float) -> float:
    """P(K >= k) via survival sum (k..k+200) for stability."""
    if k <= 0:
        return 1.0
    ks = np.arange(k, k + 400)
    return float(np.clip(poisson_pmf(ks, lam).sum(), 0.0, 1.0))


def collect_train_stats(train_npz: Path):
    data = np.load(train_npz, allow_pickle=True)
    durs = {i: [] for i in range(N_CMUDICT)}
    bigram = np.zeros((N_CMUDICT, N_CMUDICT), dtype=np.float64)

    phone_ids_all = data["phone_ids"]
    bounds_all = data["boundaries_sec"]
    for phone_ids, bounds in zip(phone_ids_all, bounds_all):
        phone_ids = np.asarray(phone_ids, dtype=np.int64)
        bounds = np.asarray(bounds, dtype=np.float64)
        if phone_ids.size == 0 or bounds.size != phone_ids.size + 1:
            continue
        for i, p in enumerate(phone_ids):
            dur_sec = float(bounds[i + 1] - bounds[i])
            d = max(1, int(round(dur_sec / FRAME_INTERVAL)))
            durs[int(p)].append(d)
            if i + 1 < phone_ids.size:
                bigram[int(phone_ids[i]), int(phone_ids[i + 1])] += 1.0
    return durs, bigram


def build_duration_tables(durs, dmax: int):
    floors = np.array([floor_frames_for_phone(p) for p in CMUDICT_39], dtype=np.int64)
    lambdas = np.zeros(N_CMUDICT, dtype=np.float64)
    probs = np.zeros((N_CMUDICT, dmax), dtype=np.float64)  # index 0 unused conceptually; use d=1..dmax → cols 0..dmax-1 as d

    for i, phone in enumerate(CMUDICT_39):
        fl = int(floors[i])
        samples = durs[i]
        if samples:
            # Fit λ on floor-clipped observations so the mean respects the floor.
            clipped = [max(fl, int(d)) for d in samples]
            lam = float(np.mean(clipped))
        else:
            lam = float(fl)
        lam = max(lam, float(fl))
        lambdas[i] = lam

        # PMF on d = 1 .. Dmax (stored at index d-1)
        ds = np.arange(1, dmax + 1)
        pmf = poisson_pmf(ds, lam)
        pmf[: fl - 1] = 0.0  # hard floor: zero mass below floor
        # Overflow: put tail P(D >= Dmax) into last bin (replace point mass at Dmax)
        tail = poisson_sf_ge(dmax, lam)
        body = pmf.copy()
        body[dmax - 1] = 0.0
        body[dmax - 1] = tail + poisson_pmf(np.array([dmax]), lam)[0]
        # Re-zero below floor after overflow assign
        body[: fl - 1] = 0.0
        s = body.sum()
        if s <= 0:
            body[:] = 0.0
            body[fl - 1 :] = 1.0 / (dmax - fl + 1)
        else:
            body /= s
        probs[i] = body

    dur_neglog = -np.log(np.clip(probs, 1e-12, 1.0))
    return dur_neglog.astype(np.float32), floors, lambdas.astype(np.float32)


def build_bigram_neglog(counts: np.ndarray) -> np.ndarray:
    # Laplace +1
    smoothed = counts + 1.0
    row_sum = smoothed.sum(axis=1, keepdims=True)
    probs = smoothed / row_sum
    return (-np.log(np.clip(probs, 1e-12, 1.0))).astype(np.float32)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--train-npz", type=Path, default=Path("/media/tom/SATAM/maps_datasets/train.npz"))
    ap.add_argument("--out", type=Path, default=Path("/media/tom/SATAM/maps_datasets/duration_prior.pt"))
    ap.add_argument("--dmax", type=int, default=80, help="Max duration frames (overflow bucket)")
    args = ap.parse_args()

    durs, bigram_counts = collect_train_stats(args.train_npz)
    dur_neglog, floors, lambdas = build_duration_tables(durs, args.dmax)
    bigram_neglog = build_bigram_neglog(bigram_counts)

    payload = {
        "dur_neglog": torch.from_numpy(dur_neglog),  # [39, Dmax], d=1..Dmax
        "floor_frames": torch.from_numpy(floors),
        "poisson_lambda": torch.from_numpy(lambdas),
        "bigram_neglog": torch.from_numpy(bigram_neglog),
        "phones": CMUDICT_39,
        "frame_interval_sec": FRAME_INTERVAL,
        "dmax": args.dmax,
        "long_fricatives": sorted(LONG_FRICATIVES),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, args.out)
    print(f"Wrote {args.out}")
    for i, p in enumerate(CMUDICT_39):
        n = len(durs[i])
        print(f"  {p:3s} n={n:6d} floor={int(floors[i])} λ={lambdas[i]:5.2f}")


if __name__ == "__main__":
    main()
