#!/usr/bin/env python3
"""Build ``.pt`` checkpoints from portable ``exported_tf/`` weight exports.

Optional maintenance tooling. The ``exported_tf/`` tree is not shipped in
torch-maps; ``torch_models/`` contains the runtime checkpoints.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from maps_torch.model import MapsAcousticModel, state_dict_from_npz
from maps_torch.phones import N_CLASSES, N_FEATURES, PHONES


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def npz_to_dict(npz_path: Path) -> dict[str, np.ndarray]:
    data = np.load(npz_path)
    return {k: data[k] for k in data.files}


def build_one(export_manifest: dict, export_dir: Path, out_path: Path, probe_dir: Path | None) -> dict:
    weights_rel = export_manifest["weights_npz"]
    weights_path = export_dir / weights_rel
    layers_path = export_dir / export_manifest["layers"]
    weights = npz_to_dict(weights_path)
    state = state_dict_from_npz(weights, layers_path)

    model = MapsAcousticModel()
    model.load_state_dict(state, strict=True)
    model.eval()

    max_abs = 0.0
    if probe_dir is not None:
        for probe_file in sorted(probe_dir.glob("probe_*.npz")):
            probe = np.load(probe_file)
            x = probe["input"].astype(np.float32)
            tf_y = probe["output"].astype(np.float32)
            with torch.inference_mode():
                torch_y = model(torch.from_numpy(x)).numpy()
            diff = np.max(np.abs(tf_y - torch_y))
            max_abs = max(max_abs, float(diff))

    metadata = {
        "phones": PHONES,
        "n_features": N_FEATURES,
        "n_classes": N_CLASSES,
        "source_path": export_manifest["source_path"],
        "source_hashes": export_manifest["source_hashes"],
        "weights_npz_hash": export_manifest["weights_npz_hash"],
        "max_probe_abs_error": max_abs,
    }
    payload = {"state_dict": model.state_dict(), "metadata": metadata}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, out_path)
    metadata["checkpoint_path"] = str(out_path)
    metadata["checkpoint_hash"] = sha256_file(out_path)
    return metadata


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build Torch checkpoints from exported TF weights.")
    parser.add_argument("--export-dir", type=Path, default=REPO_ROOT / "exported_tf")
    parser.add_argument("--out-dir", type=Path, default=REPO_ROOT / "torch_models")
    args = parser.parse_args(argv)

    master_path = args.export_dir / "manifest.json"
    master = json.loads(master_path.read_text())
    built: dict[str, dict] = {}

    for key, manifest in master["models"].items():
        stem = Path(key).name.removesuffix(".tf")
        if key.startswith("ensemble_model"):
            out_path = args.out_dir / "ensemble_model" / f"{stem}.pt"
            probe_dir = args.export_dir / "ensemble_model" / stem
        else:
            out_path = args.out_dir / f"{stem}.pt"
            probe_dir = args.export_dir / stem
        built[key] = build_one(manifest, args.export_dir, out_path, probe_dir)
        print(f"Built {out_path} (max probe |Δ|={built[key]['max_probe_abs_error']:.3e})")

    out_manifest = {
        "phones": PHONES,
        "n_features": N_FEATURES,
        "n_classes": N_CLASSES,
        "export_manifest_hash": sha256_file(master_path),
        "checkpoints": built,
    }
    manifest_out = args.out_dir / "manifest.json"
    manifest_out.write_text(json.dumps(out_manifest, indent=2))
    print(f"Wrote {manifest_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
