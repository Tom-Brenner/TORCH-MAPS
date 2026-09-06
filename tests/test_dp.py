"""Tests for CPU hard DP and soft expected-boundary DP."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from maps_torch.decode import (
    soft_expected_boundaries,
    soft_expected_boundaries_torch,
    soft_forward_duration,
    soft_forward_duration_triton,
    _triton_available,
)
from maps_torch.hard_dp_cpu import stay_advance_hard_cpu, stay_advance_hard_cpu_batch


def _toy_prior(C: int = 5, Dmax: int = 8, device="cpu"):
    # Mild duration preference around d=3
    d = torch.arange(1, Dmax + 1, dtype=torch.float32, device=device)
    lam = torch.full((C,), 3.0, device=device)
    # -log Poisson-ish surrogate
    dur = (d[None, :] - lam[:, None]).abs() * 0.5 + 0.1
    floors = torch.full((C,), 2, dtype=torch.long, device=device)
    floors[0] = 1
    bg = torch.ones(C, C, device=device) * 0.1
    return dur, floors, bg


def test_hard_dp_cpu_simple_path():
    # Strong preference: phone 0 then phone 1
    T, C = 6, 4
    cost = torch.ones(T, C) * 5.0
    cost[:3, 0] = 0.1
    cost[3:, 1] = 0.1
    phones = torch.tensor([0, 1])
    dur, floors, bg = _toy_prior(C)
    res = stay_advance_hard_cpu(
        cost, phones, dur_neglog=dur, floor_frames=floors, bigram_neglog=bg
    )
    assert res.path.shape == (T,)
    assert set(res.path.tolist()) <= {0, 1}
    assert res.path[0] == 0
    assert res.path[-1] == 1
    assert res.switches.size == 1
    assert 1 <= res.switches[0] <= T - 1


def test_hard_dp_cpu_batch_matches_single():
    rng = np.random.default_rng(0)
    C, Dmax = 6, 10
    dur = torch.rand(C, Dmax) + 0.2
    floors = torch.full((C,), 2, dtype=torch.long)
    bg = torch.rand(C, C) * 0.2
    costs, phones = [], []
    for _ in range(4):
        T = int(rng.integers(8, 20))
        N = int(rng.integers(2, 5))
        cost = torch.rand(T, C)
        ph = torch.randint(0, C, (N,))
        costs.append(cost)
        phones.append(ph)
    batch = stay_advance_hard_cpu_batch(
        costs,
        phones,
        dur_neglog=dur,
        floor_frames=floors,
        bigram_neglog=bg,
        workers=2,
    )
    singles = [
        stay_advance_hard_cpu(
            c, p, dur_neglog=dur, floor_frames=floors, bigram_neglog=bg
        )
        for c, p in zip(costs, phones)
    ]
    for a, b in zip(batch, singles):
        np.testing.assert_array_equal(a.path, b.path)
        np.testing.assert_allclose(a.switches, b.switches, rtol=0, atol=0)
        assert abs(a.best_cost - b.best_cost) < 1e-6


def test_soft_boundaries_finite_and_grad():
    B, T, C, N = 2, 12, 5, 3
    cost = torch.randn(B, T, C, requires_grad=True)
    phone_ids = torch.tensor([[0, 1, 2], [1, 0, 2]])
    lengths = torch.tensor([12, 10])
    phone_lens = torch.tensor([3, 3])
    dur, floors, bg = _toy_prior(C)
    e = soft_expected_boundaries_torch(
        cost,
        phone_ids,
        lengths,
        phone_lens,
        dur_neglog=dur,
        floor_frames=floors,
        bigram_neglog=bg,
        tau=1.0,
    )
    assert e.shape == (B, N - 1)
    assert torch.isfinite(e).all()
    loss = e.sum()
    loss.backward()
    assert cost.grad is not None
    assert torch.isfinite(cost.grad).all()
    assert cost.grad.abs().sum() > 0


@pytest.mark.skipif(not torch.cuda.is_available() or not _triton_available(), reason="CUDA+Triton")
def test_triton_forward_close_to_torch():
    device = "cuda"
    B, T, C, N = 2, 16, 5, 3
    torch.manual_seed(1)
    cost = torch.randn(B, T, C, device=device)
    phone_ids = torch.tensor([[0, 1, 2], [2, 1, 0]], device=device)
    lengths = torch.tensor([16, 12], device=device)
    phone_lens = torch.tensor([3, 3], device=device)
    dur, floors, bg = _toy_prior(C, device=device)
    a_t = soft_forward_duration(
        cost,
        phone_ids,
        lengths,
        phone_lens,
        dur_neglog=dur,
        floor_frames=floors,
        bigram_neglog=bg,
        lambda_d=1.0,
        lambda_bg=0.1,
        tau=1.0,
    )
    a_k = soft_forward_duration_triton(
        cost,
        phone_ids,
        lengths,
        phone_lens,
        dur_neglog=dur,
        floor_frames=floors,
        bigram_neglog=bg,
        lambda_d=1.0,
        lambda_bg=0.1,
        tau=1.0,
    )
    # Compare finite entries
    ni = torch.finfo(a_t.dtype).min / 4
    mask = (a_t > ni / 2) & (a_k > ni / 2)
    if mask.any():
        err = (a_t[mask] - a_k[mask]).abs().max().item()
        assert err < 5e-2, err


@pytest.mark.skipif(not torch.cuda.is_available() or not _triton_available(), reason="CUDA+Triton")
def test_soft_auto_backend_grad():
    device = "cuda"
    B, T, C, N = 2, 10, 5, 3
    cost = torch.randn(B, T, C, device=device, requires_grad=True)
    phone_ids = torch.tensor([[0, 1, 2], [1, 2, 0]], device=device)
    lengths = torch.tensor([10, 8], device=device)
    phone_lens = torch.tensor([3, 3], device=device)
    dur, floors, bg = _toy_prior(C, device=device)
    e = soft_expected_boundaries(
        cost,
        phone_ids,
        lengths,
        phone_lens,
        dur_neglog=dur,
        floor_frames=floors,
        bigram_neglog=bg,
        backend="auto",
    )
    e.sum().backward()
    assert cost.grad is not None
    assert torch.isfinite(cost.grad).all()
