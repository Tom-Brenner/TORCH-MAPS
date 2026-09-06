# Conda environment — MAPS PyTorch alignment training

Use the existing **`torch-maps`** environment for all new CMUdict-39 dataset,
duration-prior, decoding, and end-to-end training code:

```bash
conda activate torch-maps
cd ~/MAPS
```

`TF_maps` is the legacy TensorFlow MAPS environment. It contains TensorFlow
2.12.1 but **does not contain PyTorch**, so it cannot run
`tools/estimate_duration_prior.py` or `maps_torch/train_decoder.py`.

## Verified environment

The local `torch-maps` env has:

```text
Python:  /home/tom/miniconda3/envs/torch-maps/bin/python
NumPy:   2.0.2
PyTorch: 2.5.1+cu124
Triton:  3.1.0   (bundled with the CUDA PyTorch wheel — do not pip-pin separately)
CUDA:    available
```

To recreate or repair it, use Python 3.10+ and install the repository
requirements plus the CUDA-compatible PyTorch wheel appropriate for the
machine:

```bash
conda create -n torch-maps python=3.10 -y
conda activate torch-maps

# Install PyTorch for the target GPU / CUDA runtime. Example: CUDA 12.4.
pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
# Optional: duration histogram + prior overlay plots (Torch-MFA scripts/prepare_plot.py)
pip install pillow

python -c "import torch, triton; print(torch.__version__, torch.cuda.is_available(), triton.__version__)"
```

On a new GCP VM, choose the PyTorch wheel to match the installed CUDA/driver
rather than assuming `cu124` is the correct build. **Do not** `pip install triton`
separately; use the version that ships with that PyTorch build.

## DP backends (training vs inference)

| Path | Where | Flag |
|---|---|---|
| Soft DP (boundary loss) | GPU Triton time-wavefront (fallback: batched PyTorch) | `--soft-dp-backend auto\|triton\|torch` |
| Hard DP (val / inference) | CPU NumPy, up to 12 workers | `--hard-dp-workers 12` |

Acoustic scoring always stays on GPU when `--device cuda`. Hard decoding copies
`[T, C]` costs to CPU once per utterance.

## Build the train-only duration / bigram prior

```bash
conda activate torch-maps
cd ~/MAPS

python tools/estimate_duration_prior.py \
  --train-npz /media/tom/SATAM/maps_datasets/train.npz \
  --out /media/tom/SATAM/maps_datasets/duration_prior.pt
```

This needs only NumPy + PyTorch. It reads `train.npz` only; it does not use
validation data.

## Train

Full end-to-end fine-tuning is the default:

```bash
conda activate torch-maps
cd ~/MAPS

python maps_torch/train_decoder.py \
  --train-npz /media/tom/SATAM/maps_datasets/train.npz \
  --val-npz /media/tom/SATAM/maps_datasets/val.npz \
  --prior /media/tom/SATAM/maps_datasets/duration_prior.pt \
  --checkpoint torch_models/timbuck_eng.pt \
  --out-dir logs/maps_decoder39 \
  --device cuda --bf16 \
  --soft-dp-backend auto \
  --hard-dp-workers 12
```

Optional freeze ablations:

```bash
# Train only the new CMUdict-39 head:
... --freeze-backbone

# Train all except final BiLSTM + its LayerNorm:
... --freeze-last-lstm
```

The trainer needs `torch`, `numpy`, and the dependencies in
`requirements.txt`; its data comes from the precomputed NPZs, so
`python_speech_features` is not needed for the training loop itself.
