# torch-maps

PyTorch inference port of the **Mason-Alberta Phonetic Segmenter (MAPS)**.

**Original project (TensorFlow runtime, SavedModels, paper, training context):**  
[https://github.com/MasonPhonLab/MAPS](https://github.com/MasonPhonLab/MAPS)

This repository is a **standalone Torch implementation**. It does not import TensorFlow at
runtime. Acoustic weights ship as `.pt` checkpoints in `torch_models/`, converted from the
upstream MAPS SavedModels.

Model limitations and bias notes: [timbuck_eng_model_card.md](timbuck_eng_model_card.md).

## Requirements

- Python 3.10
- PyTorch 2.5.x (CPU or CUDA)
- Packages in [requirements.txt](requirements.txt)

```bash
conda create -y -n torch-maps python=3.10 pip
conda activate torch-maps
pip install torch==2.5.1   # platform-appropriate wheel
pip install -r requirements.txt
```

## Usage

**Single model:**

```bash
python maps.py \
  --audio demo_files/dark_suit_sentence_16khz.wav \
  --text demo_files/dark_suit_sentence_16khz.txt \
  --model torch_models/timbuck_eng.pt \
  --dict /path/to/cmudict \
  --device auto \
  --overwrite
```

**Ensemble** (10 checkpoints, median boundaries + confidence intervals):

```bash
python maps.py ... --model torch_models/ensemble_model/
```

Use a CMU Pronouncing Dictionary–style lexicon (uppercase words). The demo
`demo_files/sample_dictionary.txt` covers only the bundled sentence.

Audio should be **16 kHz** mono (stereo: left channel). `--resample` resamples in a
temporary directory without overwriting sources.

## Tests

```bash
python -m pytest tests/ -v
```

## Citation

Please cite the MAPS paper (Kelley, Perry & Tucker, 2024, *Phonetica*). See the
[upstream README](https://github.com/MasonPhonLab/MAPS) for DOI and details.

## License

MIT — see [LICENSE](LICENSE). MAPS model and method attribution remain with the
original authors.
