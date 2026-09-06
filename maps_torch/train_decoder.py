#!/usr/bin/env python3
"""Train a MAPS-adapted CMUdict-39 aligner with duration + bigram DP losses.

Loads pretrained TORCH-MAPS BiLSTMs, replaces the 61-way head with a 39-way
classifier, and optimizes:

* ``L_nll`` — frame CE on human majority labels (masked)
* ``L_b`` — Huber on Soft-DP expected boundaries vs human ``b_true``
* Soft DP boundary loss on GPU (Triton time-wavefront; ``--soft-dp-backend``)
* Hard DP (duration + bigram) on CPU for validation / inference (``--hard-dp-workers``)

Only the **best** checkpoint (lowest val boundary MAE) is kept locally and
optionally uploaded to GCS (``--gcs-bucket``). Use ``--resume`` / ``--resume-best``
to continue.

Duration priors (train set only)::

    python tools/estimate_duration_prior.py \\
        --train-npz /media/tom/SATAM/maps_datasets/train.npz \\
        --out /media/tom/SATAM/maps_datasets/duration_prior.pt

Example::

    python maps_torch/train_decoder.py \\
        --train-npz /data/maps_datasets/train.npz \\
        --val-npz /data/maps_datasets/val.npz \\
        --prior /data/maps_datasets/duration_prior.pt \\
        --checkpoint /data/maps_datasets/timbuck_eng.pt \\
        --out-dir logs/maps_decoder39 \\
        --device cuda --bf16 --epochs 20 \\
        --gcs-bucket gs://glen-train-data
"""

from __future__ import annotations

import argparse
import gc
import json
import signal
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from maps_torch.dataset import MapsNpzDataset, collate_maps  # noqa: E402
from maps_torch.decode import soft_expected_boundaries  # noqa: E402
from maps_torch.hard_dp_cpu import stay_advance_hard_cpu_batch  # noqa: E402
from maps_torch.model39 import MapsAcousticModel39, load_adapted_maps39  # noqa: E402


def release_cuda_memory(*, synchronize: bool = False) -> None:
    """Drop Python refs and return unused blocks to the CUDA caching allocator."""
    gc.collect()
    if torch.cuda.is_available():
        if synchronize:
            torch.cuda.synchronize()
        torch.cuda.empty_cache()


def _install_cuda_cleanup_handlers() -> None:
    """On SIGINT/SIGTERM, free CUDA cache before default termination handling."""

    def _handler(signum, frame):  # noqa: ANN001
        try:
            release_cuda_memory(synchronize=True)
        except Exception:
            pass
        # Re-raise as KeyboardInterrupt for SIGINT so ``finally`` blocks still run.
        if signum == signal.SIGINT:
            raise KeyboardInterrupt
        raise SystemExit(128 + int(signum))

    signal.signal(signal.SIGINT, _handler)
    signal.signal(signal.SIGTERM, _handler)


def _filter_indices(ds: MapsNpzDataset, max_frames: int | None) -> list[int]:
    idx = []
    for i in range(len(ds)):
        t = int(np.asarray(ds.feats[i]).shape[0])
        n = int(np.asarray(ds.phone_ids[i]).shape[0])
        if t < 3 or n < 1:
            continue
        if max_frames is not None and t > max_frames:
            continue
        idx.append(i)
    return idx


def load_prior(path: Path, device: torch.device):
    prior = torch.load(path, map_location="cpu", weights_only=False)
    return {
        "dur_neglog": prior["dur_neglog"].to(device),
        "floor_frames": prior["floor_frames"].to(device),
        "bigram_neglog": prior["bigram_neglog"].to(device),
        "dmax": int(prior["dmax"]),
    }


def maps_nll(logits39: torch.Tensor, frame_ids: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if mask.sum() == 0:
        return logits39.sum() * 0.0
    return F.cross_entropy(logits39[mask], frame_ids[mask])


def boundary_huber(e_b: torch.Tensor, b_true: torch.Tensor) -> torch.Tensor:
    return F.huber_loss(e_b, b_true, delta=1.0, reduction="mean")


def upload_to_gcs(local_path: Path, bucket: str, *, object_prefix: str | None = None) -> None:
    """Copy ``local_path`` to ``gs://bucket/<prefix>/<name>`` (or cwd-relative path)."""
    local_path = Path(local_path)
    if not local_path.exists():
        return
    if not bucket.startswith("gs://"):
        bucket = "gs://" + bucket
    if object_prefix:
        uri = f"{bucket.rstrip('/')}/{object_prefix.strip('/')}/{local_path.name}"
    else:
        try:
            rel = local_path.resolve().relative_to(Path.cwd().resolve())
        except ValueError:
            rel = Path(local_path.name)
        uri = f"{bucket.rstrip('/')}/{rel.as_posix()}"
    try:
        subprocess.run(
            ["gcloud", "storage", "cp", str(local_path), uri],
            check=True,
            capture_output=True,
            text=True,
        )
        print(f"Uploaded {local_path} -> {uri}", flush=True)
    except FileNotFoundError:
        print(f"warning: gcloud not found; skipped GCS upload for {local_path}", flush=True)
    except subprocess.CalledProcessError as exc:
        err = (exc.stderr or exc.stdout or "").strip()
        print(f"warning: GCS upload failed for {local_path}: {err}", flush=True)


def save_checkpoint(
    path: Path,
    *,
    model: torch.nn.Module,
    opt: torch.optim.Optimizer,
    epoch: int,
    best_mae: float,
    args: argparse.Namespace,
    history: list,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": model.state_dict(),
        "optimizer": opt.state_dict(),
        "epoch": epoch,
        "best_val_boundary_mae_frames": best_mae,
        "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "history": history,
    }
    torch.save(payload, path)


def step_batch(
    model,
    batch,
    prior,
    *,
    lambda_nll: float,
    lambda_b: float,
    lambda_d: float,
    lambda_bg: float,
    tau: float,
    use_boundary_loss: bool,
    autocast_dtype: torch.dtype | None,
    soft_dp_backend: str = "auto",
):
    device = next(model.parameters()).device
    feats = batch["feats"].to(device)
    use_amp = autocast_dtype is not None and device.type == "cuda"
    with torch.autocast(device_type="cuda", dtype=autocast_dtype or torch.float32, enabled=use_amp):
        logits = model.logits(feats)
        # Keep DP math in fp32 for stability
        logits_f = logits.float()
        logp = logits_f.log_softmax(-1)
        cost_b = -logp
        L_n = maps_nll(
            logits_f,
            batch["frame_ids"].to(device),
            batch["frame_mask"].to(device),
        )

        L_b = logits_f.new_zeros(())
        if use_boundary_loss and lambda_b > 0:
            lengths = batch["lengths"].to(device)
            phone_mask = batch["phone_mask"].to(device)
            phone_lens = phone_mask.sum(dim=1).long()
            phone_ids = batch["phone_ids"].to(device)
            b_true = batch["b_true"].to(device)
            b_mask = batch["b_mask"].to(device)
            e_b = soft_expected_boundaries(
                cost_b,
                phone_ids,
                lengths,
                phone_lens,
                dur_neglog=prior["dur_neglog"],
                floor_frames=prior["floor_frames"],
                bigram_neglog=prior["bigram_neglog"],
                lambda_d=lambda_d,
                lambda_bg=lambda_bg,
                tau=tau,
                backend=soft_dp_backend,
            )
            # Huber only on valid boundaries (b_mask); average over masked entries
            nb = min(e_b.shape[1], b_true.shape[1], b_mask.shape[1])
            if nb > 0 and b_mask[:, :nb].any():
                pred = e_b[:, :nb][b_mask[:, :nb]]
                tgt = b_true[:, :nb][b_mask[:, :nb]]
                L_b = boundary_huber(pred, tgt)

        loss = lambda_nll * L_n + lambda_b * L_b

    stats = {
        "L_nll": float(L_n.detach()),
        "L_b": float(L_b.detach()),
        "loss": float(loss.detach()),
    }
    return loss, stats


@torch.inference_mode()
def eval_boundary_mae(
    model,
    loader,
    prior,
    *,
    lambda_d,
    lambda_bg,
    device,
    max_batches=50,
    hard_dp_workers: int = 12,
):
    """Hard-DP validation on CPU (duration-aware Viterbi)."""
    model.eval()
    total = 0.0
    count = 0
    # Priors on CPU for hard DP
    prior_cpu = {
        "dur_neglog": prior["dur_neglog"].detach().cpu(),
        "floor_frames": prior["floor_frames"].detach().cpu(),
        "bigram_neglog": prior["bigram_neglog"].detach().cpu(),
    }
    try:
        for bi, batch in enumerate(loader):
            if bi >= max_batches:
                break
            feats = batch["feats"].to(device)
            logits = model.logits(feats).float()
            cost_b = (-logits.log_softmax(-1)).cpu()
            # Free GPU activations before CPU hard DP.
            del feats, logits
            B = cost_b.shape[0]
            costs = []
            phones = []
            b_trues = []
            for i in range(B):
                T = int(batch["lengths"][i])
                n = int(batch["phone_mask"][i].sum())
                if n < 2 or T < 2:
                    continue
                costs.append(cost_b[i, :T])
                phones.append(batch["phone_ids"][i, :n])
                b_trues.append(batch["b_true"][i, : n - 1])
            if not costs:
                continue
            results = stay_advance_hard_cpu_batch(
                costs,
                phones,
                dur_neglog=prior_cpu["dur_neglog"],
                floor_frames=prior_cpu["floor_frames"],
                bigram_neglog=prior_cpu["bigram_neglog"],
                lambda_d=lambda_d,
                lambda_bg=lambda_bg,
                workers=hard_dp_workers,
            )
            for res, b_true in zip(results, b_trues):
                b_true = np.asarray(b_true, dtype=np.float64)
                sw = res.switches
                nb = min(sw.size, b_true.size)
                if nb == 0:
                    continue
                total += float(np.abs(sw[:nb] - b_true[:nb]).mean())
                count += 1
            if (bi + 1) % 10 == 0 and device.type == "cuda":
                release_cuda_memory()
    finally:
        if device.type == "cuda":
            release_cuda_memory()
        model.train()
    return total / max(count, 1)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--train-npz", type=Path, default=Path("/media/tom/SATAM/maps_datasets/train.npz"))
    ap.add_argument("--val-npz", type=Path, default=Path("/media/tom/SATAM/maps_datasets/val.npz"))
    ap.add_argument("--prior", type=Path, default=Path("/media/tom/SATAM/maps_datasets/duration_prior.pt"))
    ap.add_argument("--checkpoint", type=Path, default=REPO / "torch_models" / "timbuck_eng.pt")
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=Path("logs/maps_decoder39"),
        help="Relative path recommended so GCS keys stay short",
    )
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--lambda-nll", type=float, default=1.0)
    ap.add_argument("--lambda-b", type=float, default=1.0)
    ap.add_argument("--lambda-d", type=float, default=1.0)
    ap.add_argument("--lambda-bg", type=float, default=0.1)
    ap.add_argument("--tau", type=float, default=1.0)
    ap.add_argument("--max-frames", type=int, default=2000)
    ap.add_argument("--num-workers", type=int, default=2)
    freeze = ap.add_mutually_exclusive_group()
    freeze.add_argument(
        "--freeze-backbone",
        action="store_true",
        help="Freeze all BiLSTMs/LNs (incl. last); train only the 39-way head",
    )
    freeze.add_argument(
        "--freeze-last-lstm",
        action="store_true",
        help="Freeze only the last BiLSTM (+ LN); train earlier layers + head",
    )
    ap.add_argument("--no-boundary-loss", action="store_true")
    ap.add_argument("--bf16", action="store_true", help="Autocast forward in bfloat16 (CUDA)")
    ap.add_argument(
        "--soft-dp-backend",
        choices=("auto", "triton", "torch"),
        default="auto",
        help="Soft DP backend for boundary loss (auto→Triton on CUDA)",
    )
    ap.add_argument(
        "--hard-dp-workers",
        type=int,
        default=12,
        help="CPU threads for hard-DP validation (capped at 12)",
    )
    ap.add_argument(
        "--cuda-release-every",
        type=int,
        default=50,
        help="Call empty_cache every N train batches (0 disables periodic release)",
    )
    ap.add_argument(
        "--resume",
        type=Path,
        default=None,
        help="Resume from this checkpoint (model+opt+epoch+best_mae)",
    )
    ap.add_argument(
        "--resume-best",
        action="store_true",
        help="Resume from ``<out-dir>/best.pt`` if present",
    )
    ap.add_argument(
        "--gcs-bucket",
        default=None,
        help="e.g. gs://glen-train-data — upload best.pt + history.json on improvement",
    )
    ap.add_argument(
        "--gcs-prefix",
        default="logs/maps_decoder39",
        help="Object prefix under the bucket for best.pt / history.json",
    )
    args = ap.parse_args()

    _install_cuda_cleanup_handlers()

    device = torch.device(
        args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu"
    )
    if args.bf16 and device.type != "cuda":
        print("warning: --bf16 ignored on non-CUDA device")
        args.bf16 = False
    autocast_dtype = torch.bfloat16 if args.bf16 else None

    if not args.prior.is_file():
        raise SystemExit(
            f"Missing {args.prior}. Run tools/estimate_duration_prior.py first."
        )

    prior = load_prior(args.prior, device)
    model = load_adapted_maps39(
        args.checkpoint, device=device, freeze_backbone=args.freeze_backbone
    )
    if args.freeze_last_lstm:
        model.freeze_last_lstm()
    if args.freeze_backbone:
        freeze_mode = "backbone (head only)"
    elif args.freeze_last_lstm:
        freeze_mode = "last BiLSTM only"
    else:
        freeze_mode = "none (all trainable)"

    train_ds = MapsNpzDataset(args.train_npz, max_frames=args.max_frames)
    val_ds = MapsNpzDataset(args.val_npz, max_frames=args.max_frames)
    train_ds = Subset(train_ds, _filter_indices(train_ds, args.max_frames))
    val_ds = Subset(val_ds, _filter_indices(val_ds, args.max_frames))
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_all = sum(p.numel() for p in model.parameters())
    print(
        f"train utts={len(train_ds)} val utts={len(val_ds)} "
        f"bf16={args.bf16} freeze={freeze_mode} "
        f"trainable_params={n_train}/{n_all}"
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_maps,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_maps,
    )

    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.Adam(params, lr=args.lr)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    start_epoch = 1
    best_mae = float("inf")
    history: list = []

    resume_path = args.resume
    if args.resume_best:
        cand = args.out_dir / "best.pt"
        if cand.is_file():
            resume_path = cand
    if resume_path is not None:
        if not resume_path.is_file():
            raise SystemExit(f"--resume file not found: {resume_path}")
        ckpt = torch.load(resume_path, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["model"])
        if "optimizer" in ckpt:
            opt.load_state_dict(ckpt["optimizer"])
        start_epoch = int(ckpt.get("epoch", 0)) + 1
        best_mae = float(ckpt.get("best_val_boundary_mae_frames", best_mae))
        history = list(ckpt.get("history", []))
        # Drop CPU checkpoint payload before training.
        del ckpt
        release_cuda_memory()
        print(
            f"Resumed from {resume_path} at epoch {start_epoch} "
            f"(best_mae={best_mae:.4f})"
        )

    if args.gcs_bucket:
        print(f"GCS uploads enabled -> {args.gcs_bucket}/{args.gcs_prefix}")

    use_b = not args.no_boundary_loss
    release_every = max(0, int(args.cuda_release_every))
    try:
        for epoch in range(start_epoch, args.epochs + 1):
            model.train()
            running = {"L_nll": 0.0, "L_b": 0.0, "loss": 0.0}
            n_batches = 0
            for batch in train_loader:
                if int(batch["lengths"].max()) > args.max_frames:
                    continue
                opt.zero_grad(set_to_none=True)
                loss, stats = step_batch(
                    model,
                    batch,
                    prior,
                    lambda_nll=args.lambda_nll,
                    lambda_b=args.lambda_b,
                    lambda_d=args.lambda_d,
                    lambda_bg=args.lambda_bg,
                    tau=args.tau,
                    use_boundary_loss=use_b,
                    autocast_dtype=autocast_dtype,
                    soft_dp_backend=args.soft_dp_backend,
                )
                loss.backward()
                torch.nn.utils.clip_grad_norm_(params, 5.0)
                opt.step()
                for k in running:
                    running[k] += stats[k]
                n_batches += 1
                # Drop graph roots so soft-DP lattices can be collected.
                del loss
                if n_batches % 1== 0:
                    print(
                        f"epoch {epoch} batch {n_batches} "
                        f"loss={stats['loss']:.4f} L_nll={stats['L_nll']:.4f} "
                        f"L_b={stats['L_b']:.4f}"
                    )
                if (
                    device.type == "cuda"
                    and release_every > 0
                    and n_batches % release_every == 0
                ):
                    release_cuda_memory()

            for k in running:
                running[k] /= max(n_batches, 1)
            mae = eval_boundary_mae(
                model,
                val_loader,
                prior,
                lambda_d=args.lambda_d,
                lambda_bg=args.lambda_bg,
                device=device,
                hard_dp_workers=args.hard_dp_workers,
            )
            row = {
                "epoch": epoch,
                "train_loss": running["loss"],
                "L_nll": running["L_nll"],
                "L_b": running["L_b"],
                "val_boundary_mae_frames": mae,
            }
            history.append(row)
            print(
                f"EPOCH {epoch}: train loss={running['loss']:.4f} "
                f"L_nll={running['L_nll']:.4f} L_b={running['L_b']:.4f} "
                f"val_boundary_MAE_frames={mae:.3f}"
            )

            improved = mae < best_mae
            if improved:
                best_mae = mae
                best_path = args.out_dir / "best.pt"
                save_checkpoint(
                    best_path,
                    model=model,
                    opt=opt,
                    epoch=epoch,
                    best_mae=best_mae,
                    args=args,
                    history=history,
                )
                hist_path = args.out_dir / "history.json"
                hist_path.write_text(json.dumps(history, indent=2))
                print(f"  new best -> {best_path} (mae={best_mae:.4f})")
                if args.gcs_bucket:
                    upload_to_gcs(best_path, args.gcs_bucket, object_prefix=args.gcs_prefix)
                    upload_to_gcs(hist_path, args.gcs_bucket, object_prefix=args.gcs_prefix)
            else:
                # Still refresh local history for resume bookkeeping (not uploaded unless best)
                (args.out_dir / "history.json").write_text(json.dumps(history, indent=2))
            if device.type == "cuda":
                release_cuda_memory()
    except KeyboardInterrupt:
        print("\nInterrupted — releasing CUDA memory…", flush=True)
        raise
    finally:
        release_cuda_memory(synchronize=device.type == "cuda")


if __name__ == "__main__":
    main()
