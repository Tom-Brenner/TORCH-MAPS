"""MAPS PyTorch test suite (TensorFlow-free)."""

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from scipy.io import wavfile
from textgrid import textgrid

REPO = Path(__file__).resolve().parents[1]
DEMO_WAV = REPO / "demo_files" / "dark_suit_sentence_16khz.wav"
DEMO_REF = REPO / "demo_files" / "dark_suit_sentence_16khz.TextGrid"
SINGLE_CKPT = REPO / "torch_models" / "timbuck_eng.pt"
ENSEMBLE_DIR = REPO / "torch_models" / "ensemble_model"
TORCH_MANIFEST = REPO / "torch_models" / "manifest.json"
EXPORT_MANIFEST = REPO / "exported_tf" / "manifest.json"


@pytest.fixture(scope="module")
def model():
    from maps_torch.model import load_checkpoint

    return load_checkpoint(SINGLE_CKPT, device="cpu")


def test_no_tensorflow_installed():
    try:
        import tensorflow  # noqa: F401
    except ImportError:
        return
    pytest.fail("TensorFlow must not be installed in the Torch runtime environment")


def test_all_checkpoints_load():
    from maps_torch.model import load_checkpoint

    paths = [SINGLE_CKPT] + sorted(ENSEMBLE_DIR.glob("*.pt"))
    assert len(paths) == 11
    for p in paths:
        m = load_checkpoint(p, device="cpu")
        assert m is not None


@pytest.mark.parametrize("batch,time", [(1, 5), (2, 17), (1, 100)])
def test_model_output_shape(model, batch, time):
    x = torch.randn(batch, time, 39)
    with torch.inference_mode():
        y = model(x)
    assert y.shape == (batch, time, 61)


def test_model_probabilities(model):
    x = torch.randn(1, 20, 39)
    with torch.inference_mode():
        y = model(x).numpy()
    assert np.all(np.isfinite(y))
    assert np.all(y >= 0)
    sums = y.sum(axis=-1)
    np.testing.assert_allclose(sums, 1.0, atol=1e-5)


def test_model_deterministic(model):
    x = torch.randn(1, 15, 39)
    with torch.inference_mode():
        y1 = model(x).numpy()
        y2 = model(x).numpy()
    np.testing.assert_array_equal(y1, y2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_model_cuda():
    from maps_torch.model import load_checkpoint

    m = load_checkpoint(SINGLE_CKPT, device="cuda")
    x = torch.randn(1, 8, 39, device="cuda")
    with torch.inference_mode():
        y = m(x)
    assert y.device.type == "cuda"


def test_manifest_phones():
    manifest = json.loads(TORCH_MANIFEST.read_text())
    from maps_torch.phones import PHONES

    assert manifest["phones"] == PHONES
    assert len(manifest["checkpoints"]) == 11


def test_conversion_probes():
    from maps_torch.model import MapsAcousticModel, load_checkpoint

    model = load_checkpoint(SINGLE_CKPT, device="cpu")
    probe = np.load(REPO / "exported_tf" / "timbuck_eng" / "probe_random_T37.npz")
    with torch.inference_mode():
        torch_y = model(torch.from_numpy(probe["input"])).numpy()
    np.testing.assert_allclose(torch_y, probe["output"], rtol=1e-5, atol=1e-5)


def test_features_demo_wav():
    from maps_torch.features import extract_features, read_mono_wav

    sr, samples = read_mono_wav(DEMO_WAV)
    feats = extract_features(samples, sr)
    assert feats.shape[1] == 39
    assert feats.dtype == np.float32
    assert np.all(np.isfinite(feats))
    assert feats.shape[0] > 100


def test_features_stereo_left_channel(tmp_path):
    from maps_torch.features import read_mono_wav

    stereo = np.column_stack([np.ones(1600, dtype=np.int16), np.zeros(1600, dtype=np.int16) * 1000])
    path = tmp_path / "stereo.wav"
    wavfile.write(path, 16000, stereo)
    _, mono = read_mono_wav(path)
    assert np.all(mono == 1)


def test_nltk_cmudict_variant_index_skipped(tmp_path):
    import re

    from maps_torch.alignment import load_dictionary
    from maps_torch.cli_types import WordString

    lex_path = tmp_path / "cmudict_sample"
    lex_path.write_text("DARK 1 D AA1 R K\nSUIT 1 S UW1 T\n")
    lex = load_dictionary(lex_path)
    ws = WordString(["DARK"], lex["DARK"])
    phones = [re.sub(r"[0-9]", "", x) for x in ws.phone_string]
    assert "" not in phones
    assert phones[0] == "D"


def test_interpolated_part_no_seq_bug():
    from maps_torch.textgrid_io import interpolated_part

    probs = np.random.rand(10, 5).astype(np.float64)
    val = interpolated_part(3, 1, probs)
    assert isinstance(val, float)


def test_alignment_known_path():
    from maps_torch.alignment import force_align

    # Two-frame path with obvious minimum.
    yhat = np.zeros((1, 4, 61), dtype=np.float32)
    yhat[0, :, 0] = [0.9, 0.1, 0.1, 0.1]
    yhat[0, :, 1] = [0.1, 0.9, 0.9, 0.9]
    seq, _ = force_align(["h#", "q"], yhat)
    assert len(seq) == 2


def test_demo_cli_boundary_accuracy(tmp_path):
    out_tg = tmp_path / "out.TextGrid"
    subprocess.run(
        [
            sys.executable,
            str(REPO / "maps.py"),
            "--audio",
            str(DEMO_WAV),
            "--text",
            str(REPO / "demo_files" / "dark_suit_sentence_16khz.txt"),
            "--model",
            str(SINGLE_CKPT),
            "--dict",
            str(REPO / "demo_files" / "sample_dictionary.txt"),
            "--overwrite",
            "--device",
            "cpu",
        ],
        check=True,
        cwd=REPO,
    )
    produced = DEMO_WAV.with_suffix(".TextGrid")
    assert produced.is_file()
    ref = textgrid.TextGrid()
    ref.read(DEMO_REF)
    hyp = textgrid.TextGrid()
    hyp.read(produced)

    ref_phones = ref.tiers[1]
    hyp_phones = hyp.tiers[1]
    assert len(ref_phones) == len(hyp_phones)
    for r, h in zip(ref_phones, hyp_phones):
        assert r.mark == h.mark

    diffs = []
    for r, h in zip(ref_phones, hyp_phones):
        diffs.append(abs(r.minTime - h.minTime))
        diffs.append(abs(r.maxTime - h.maxTime))
    median = float(np.median(diffs))
    maximum = float(np.max(diffs))
    assert median <= 0.002, f"median boundary diff {median:.4f}s"
    assert maximum <= 0.010, f"max boundary diff {maximum:.4f}s"
