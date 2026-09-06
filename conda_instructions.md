# Conda environment — MAPS PyTorch alignment training

Use the **`torch-maps`** environment for all CMUdict-39 dataset, duration-prior,
decoding, and end-to-end training code:

```bash
conda activate torch-maps
cd ~/MAPS
```

`TF_maps` is the legacy TensorFlow MAPS environment. It contains TensorFlow
2.12.1 but **does not contain PyTorch**, so it cannot run
`tools/estimate_duration_prior.py` or `maps_torch/train_decoder.py`.

Spec file: [`environment.yml`](environment.yml) (Python + pip deps).  
**PyTorch is installed in a second step** so the CUDA wheel can match the machine.

## Verified environment (local)

```text
Python:  /home/tom/miniconda3/envs/torch-maps/bin/python
NumPy:   2.0.2
PyTorch: 2.5.1+cu124
Triton:  3.1.0   (bundled with the CUDA PyTorch wheel — do not pip-pin separately)
CUDA:    available
```

## GCP VM (fresh image — no conda)

Deep Learning “common CUDA” images often have drivers/CUDA but **no conda**.
Install Miniconda, then create `torch-maps` from the yml:

```bash
# --- Miniconda (one-time) ---
cd ~
curl -fsSL https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -o miniconda.sh
bash miniconda.sh -b -p $HOME/miniconda3
rm miniconda.sh
eval "$($HOME/miniconda3/bin/conda shell.bash hook)"
conda init bash
# open a new shell, or: source ~/.bashrc

# --- torch-maps from repo ---
cd ~/MAPS
conda env create -f environment.yml
conda activate torch-maps

# GPU PyTorch. cu124 wheels work with newer drivers (e.g. cu129 DL images).
pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu124

python -c "import torch, triton; print(torch.__version__, torch.cuda.is_available(), triton.__version__)"
nvidia-smi
```

If `conda` is already on `PATH` (some DL images ship `/opt/conda`), skip Miniconda
and run only the `conda env create …` block.

**Do not** `pip install triton` separately.

## Recreate / repair (laptop or VM)

```bash
cd ~/MAPS
conda env create -f environment.yml          # or: conda env update -f environment.yml --prune
conda activate torch-maps
pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu124
# Optional plots: pip install pillow

python -c "import torch, triton; print(torch.__version__, torch.cuda.is_available(), triton.__version__)"
```

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
