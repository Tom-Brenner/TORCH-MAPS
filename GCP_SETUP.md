# GCP training setup — MAPS CMUdict-39 decoder (boundary / duration DP)

Project: `glen-503819` (number `398729092111`)  
gcloud config: `glen` (`source ~/vits/switch-gcloud.sh glen`)  
Bucket: `gs://glen-train-data`  
Repo: [TORCH-MAPS](https://github.com/Tom-Brenner/TORCH-MAPS) (local `~/MAPS`)

**VM: not assigned yet.** Create a GPU VM when ready (§3). Until then you can still
upload the dataset + priors to the existing glen bucket and develop locally.

Local ↔ GCS ↔ VM paths:

| Local | GCS | VM (planned) |
|---|---|---|
| `/media/tom/SATAM/maps_datasets/train.npz` | `gs://glen-train-data/maps_datasets/train.npz` | `/data/maps_datasets/train.npz` |
| `/media/tom/SATAM/maps_datasets/val.npz` | `gs://glen-train-data/maps_datasets/val.npz` | `/data/maps_datasets/val.npz` |
| `/media/tom/SATAM/maps_datasets/duration_prior.pt` | `gs://glen-train-data/maps_datasets/duration_prior.pt` | `/data/maps_datasets/duration_prior.pt` |
| `~/MAPS/torch_models/timbuck_eng.pt` | `gs://glen-train-data/maps_datasets/timbuck_eng.pt` | `/data/maps_datasets/timbuck_eng.pt` |
| `logs/maps_decoder39/best.pt` | `gs://glen-train-data/logs/maps_decoder39/best.pt` | `~/MAPS/logs/maps_decoder39/best.pt` |

Trainer: `maps_torch/train_decoder.py`  
- Saves **only** `best.pt` when val boundary MAE improves (no per-epoch weights).  
- `--bf16` · `--resume` / `--resume-best` · `--gcs-bucket gs://glen-train-data`  
- Soft DP (boundary loss): GPU Triton (`--soft-dp-backend auto`); hard DP (val): CPU (`--hard-dp-workers 12`).  
- Triton comes with the CUDA PyTorch wheel — do not pin a separate `triton` package.

---

## Status

### Done / available

| Item | Notes |
|---|---|
| Project + gcloud `glen` config | Same as GLEN / vits (`glen-503819`). |
| Bucket `gs://glen-train-data` | Exists (`us-west4`). |
| Local MAPS datasets | `train.npz` / `val.npz` (CMUdict-39) under `/media/tom/SATAM/maps_datasets/`. |
| Duration + bigram prior | `duration_prior.pt` (train-only Poisson + floors). |
| Trainer flags | best-only save, `--bf16`, `--resume` / `--resume-best`, `--gcs-bucket`, `--soft-dp-backend`, `--hard-dp-workers`. |

### Not done

| Item | Notes |
|---|---|
| **GPU VM** | **Not assigned yet** — create when ready (§3). |
| Dataset upload to GCS | Run §2 rsync before VM training. |
| VM env / clone | After VM exists (§6). |

---

## 0 — Activate glen credentials (local)

```bash
source ~/vits/switch-gcloud.sh glen
gcloud config get-value project   # glen-503819
gcloud config get-value account
```

---

## 1 — Install & authenticate gcloud (local, one-time)

Skip if `gcloud` already works with the `glen` configuration (see `~/vits/GCP_SETUP.md`).

```bash
gcloud auth login
gcloud config configurations activate glen
gcloud config set project glen-503819
gcloud auth application-default login
gcloud auth application-default set-quota-project glen-503819
```

---

## 2 — Upload MAPS datasets + prior + base checkpoint (local)

```bash
source ~/vits/switch-gcloud.sh glen
cd ~/MAPS

# Ensure prior exists
python tools/estimate_duration_prior.py \
  --train-npz /media/tom/SATAM/maps_datasets/train.npz \
  --out /media/tom/SATAM/maps_datasets/duration_prior.pt

gcloud storage cp \
  /media/tom/SATAM/maps_datasets/train.npz \
  /media/tom/SATAM/maps_datasets/val.npz \
  /media/tom/SATAM/maps_datasets/duration_prior.pt \
  ~/MAPS/torch_models/timbuck_eng.pt \
  gs://glen-train-data/maps_datasets/
```

Optional recursive sync if you add more artifacts under that folder:

```bash
gcloud storage rsync \
  /media/tom/SATAM/maps_datasets \
  gs://glen-train-data/maps_datasets --recursive
```

Do **not** upload huge unrelated SATAM trees.

---

## 3 — Create GPU VM (local) — **pending**

No VM has been assigned for this MAPS run yet. When ready, example (adjust zone/type):

```bash
export VM=maps-decoder-train   # choose a name
export ZONE=us-west4-a

gcloud compute instances create $VM \
  --zone=$ZONE \
  --machine-type=g2-standard-8 \
  --image-family=common-cu129-ubuntu-2204-nvidia-580 \
  --image-project=deeplearning-platform-release \
  --boot-disk-size=200GB \
  --boot-disk-type=pd-ssd \
  --maintenance-policy=TERMINATE \
  --metadata=install-nvidia-driver=True \
  --scopes=cloud-platform
```

Spot (preemptible) is optional; add your usual Spot flags if desired.

```bash
gcloud compute instances stop  $VM --zone=$ZONE
gcloud compute instances start $VM --zone=$ZONE
```

---

## 4 — Grant VM access to bucket (after VM exists)

```bash
# Project number:
gcloud projects describe glen-503819 --format="value(projectNumber)"

gcloud storage buckets add-iam-policy-binding gs://glen-train-data \
  --member="serviceAccount:398729092111-compute@developer.gserviceaccount.com" \
  --role="roles/storage.objectAdmin"
```

(Reuse the same compute SA as other glen VMs if that binding already exists.)

---
gcloud compute instances list
## 5 — SSH

```bash
export VM=maps-decoder-train ZONE=us-west4-a
gcloud compute instances start $VM --zone=$ZONE   # if stopped
gcloud compute ssh $VM --zone=$ZONE
```
ssh-keygen -t ed25519 -C "maps-vm" -f ~/.ssh/id_ed25519 -N ""
cat ~/.ssh/id_ed25519.pub
---

## 6 — VM setup (on VM, one-time)

```bash
sudo mkdir -p /data/maps_datasets
sudo chown -R $USER:$USER /data

gcloud storage cp \
  gs://glen-train-data/maps_datasets/train.npz \
  gs://glen-train-data/maps_datasets/val.npz \
  gs://glen-train-data/maps_datasets/duration_prior.pt \
  gs://glen-train-data/maps_datasets/timbuck_eng.pt \
  /data/maps_datasets/

ssh-keygen -t ed25519 -C "maps-vm" -f ~/.ssh/id_ed25519 -N ""
cat ~/.ssh/id_ed25519.pub

# Clone TORCH-MAPS (deploy key or HTTPS as you prefer)
git clone git@github.com:Tom-Brenner/TORCH-MAPS.git ~/MAPS
cd ~/MAPS
git checkout training   # training branch has decoder + DP backends

# Conda is usually missing on "common CUDA" DL images — see conda_instructions.md
# Quick path:
curl -fsSL https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -o ~/miniconda.sh
bash ~/miniconda.sh -b -p $HOME/miniconda3 && rm ~/miniconda.sh
eval "$($HOME/miniconda3/bin/conda shell.bash hook)"
conda init bash
source ~/.bashrc

cd ~/MAPS
# Accept Anaconda ToS if conda env create complains about channels:
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r

conda env create -f environment.yml
conda activate torch-maps
python -V   # expect 3.10.x — do not pip-install torch in (base)
pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu124

python -c "import torch, triton; print(torch.__version__, torch.cuda.is_available(), triton.__version__)"
nvidia-smi
```

---

## 7 — Training with tmux (on VM)

```bash
tmux new -s maps39

cd ~/MAPS
# activate env

python maps_torch/train_decoder.py \
  --train-npz /data/maps_datasets/train.npz \
  --val-npz /data/maps_datasets/val.npz \
  --prior /data/maps_datasets/duration_prior.pt \
  --checkpoint /data/maps_datasets/timbuck_eng.pt \
  --out-dir logs/maps_decoder39 \
  --device cuda \
  --bf16 \
  --soft-dp-backend auto \
  --hard-dp-workers 12 \
  --epochs 20 \
  --batch-size 8 \
  --gcs-bucket gs://glen-train-data \
  --gcs-prefix logs/maps_decoder39 \
  2>&1 | tee ~/maps39_train.log
```

Resume after preemption / restart:

```bash
python maps_torch/train_decoder.py \
  ...same data flags... \
  --out-dir logs/maps_decoder39 \
  --resume-best \
  --bf16 \
  --soft-dp-backend auto \
  --hard-dp-workers 12 \
  --gcs-bucket gs://glen-train-data \
  --gcs-prefix logs/maps_decoder39
```

Or `--resume logs/maps_decoder39/best.pt`.

On each **val improvement**, the trainer writes locally and uploads:

- `gs://glen-train-data/logs/maps_decoder39/best.pt`
- `gs://glen-train-data/logs/maps_decoder39/history.json`

- **Detach:** `Ctrl-b d`
- **Reattach:** `tmux attach -t maps39`

---

## 8 — VM lifecycle

```bash
export VM=maps-decoder-train ZONE=us-west4-a

gcloud compute instances stop   $VM --zone=$ZONE
gcloud compute instances start  $VM --zone=$ZONE
gcloud compute ssh $VM --zone=$ZONE
gcloud compute instances delete $VM --zone=$ZONE   # when finished
```

---

## 9 — Pull best checkpoint locally

```bash
source ~/vits/switch-gcloud.sh glen
mkdir -p ~/MAPS/logs/maps_decoder39_gcp
gcloud storage cp \
  gs://glen-train-data/logs/maps_decoder39/best.pt \
  ~/MAPS/logs/maps_decoder39_gcp/best.pt
gcloud storage cp \
  gs://glen-train-data/logs/maps_decoder39/history.json \
  ~/MAPS/logs/maps_decoder39_gcp/history.json
```
