"""CMUdict-39 inventory and maps from TIMIT / Buckeye / MAPS-61.

Training and stay/advance DP in the boundary-decoder experiment use **39**
CMUdict phones (no stress, no silence class). The frozen TORCH-MAPS encoder
still emits 61-way softmax; collapse with ``collapse_maps61_posteriors``.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from maps_torch.phones import PHONES as MAPS_PHONES

CMUDICT_39: Tuple[str, ...] = (
    "AA",
    "AE",
    "AH",
    "AO",
    "AW",
    "AY",
    "B",
    "CH",
    "D",
    "DH",
    "EH",
    "ER",
    "EY",
    "F",
    "G",
    "HH",
    "IH",
    "IY",
    "JH",
    "K",
    "L",
    "M",
    "N",
    "NG",
    "OW",
    "OY",
    "P",
    "R",
    "S",
    "SH",
    "T",
    "TH",
    "UH",
    "UW",
    "V",
    "W",
    "Y",
    "Z",
    "ZH",
)

N_CMUDICT = len(CMUDICT_39)
CMUDICT_TO_ID = {p: i for i, p in enumerate(CMUDICT_39)}
ID_TO_CMUDICT = {i: p for i, p in enumerate(CMUDICT_39)}

# TIMIT / MAPS-style lowercase → CMUdict; None = drop (silence / glottal / etc.).
TIMIT_TO_CMUDICT: Dict[str, Optional[str]] = {
    "aa": "AA",
    "ae": "AE",
    "ah": "AH",
    "ao": "AO",
    "aw": "AW",
    "ax": "AH",
    "ax-h": "AH",
    "ay": "AY",
    "eh": "EH",
    "er": "ER",
    "axr": "ER",
    "ey": "EY",
    "ih": "IH",
    "ix": "IH",
    "iy": "IY",
    "ow": "OW",
    "oy": "OY",
    "uh": "UH",
    "uw": "UW",
    "ux": "UW",
    "b": "B",
    "ch": "CH",
    "d": "D",
    "dh": "DH",
    "dx": "D",
    "el": "L",
    "em": "M",
    "en": "N",
    "eng": "NG",
    "f": "F",
    "g": "G",
    "hh": "HH",
    "hv": "HH",
    "jh": "JH",
    "k": "K",
    "l": "L",
    "m": "M",
    "n": "N",
    "ng": "NG",
    "nx": "N",
    "p": "P",
    "r": "R",
    "s": "S",
    "sh": "SH",
    "t": "T",
    "th": "TH",
    "v": "V",
    "w": "W",
    "y": "Y",
    "z": "Z",
    "zh": "ZH",
    # closures: fold into release for posterior collapse; tier merge handled separately
    "bcl": "B",
    "dcl": "D",
    "gcl": "G",
    "pcl": "P",
    "tcl": "T",
    "kcl": "K",
    "q": None,
    "epi": None,
    "pau": None,
    "h#": None,
    "sil": None,
}

BUCKEYE_TO_CMUDICT: Dict[str, Optional[str]] = {
    **{p.lower(): p for p in CMUDICT_39},
    "aan": "AA",
    "aen": "AE",
    "ahn": "AH",
    "aon": "AO",
    "awn": "AW",
    "ayn": "AY",
    "ehn": "EH",
    "eyn": "EY",
    "ihn": "IH",
    "iyn": "IY",
    "own": "OW",
    "oyn": "OY",
    "uhn": "UH",
    "uwn": "UW",
    "ern": "ER",
    "el": "L",
    "em": "M",
    "en": "N",
    "eng": "NG",
    "dx": "D",
    "nx": "N",
    "tq": None,
    "sil": None,
    "noise": None,
    "vocnoise": None,
    "iver": None,
    "laugh": None,
    "unknown": None,
    "{b_trans}": None,
    "{e_trans}": None,
}

# MAPS-61 label → CMUdict (closures → release; silence → None).
MAPS61_TO_CMUDICT: Dict[str, Optional[str]] = {
    p: TIMIT_TO_CMUDICT.get(p, p.upper() if p.upper() in CMUDICT_TO_ID else None)
    for p in MAPS_PHONES
}


def _collapse_matrix() -> np.ndarray:
    """Return float32 [61, 39] that sums MAPS posterior mass into CMUdict bins."""
    w = np.zeros((len(MAPS_PHONES), N_CMUDICT), dtype=np.float32)
    for i, p in enumerate(MAPS_PHONES):
        dest = MAPS61_TO_CMUDICT.get(p)
        if dest is None:
            continue
        w[i, CMUDICT_TO_ID[dest]] = 1.0
    return w


MAPS61_TO_CMUDICT39_MATRIX = _collapse_matrix()


def to_cmudict_id(label: str, *, source: str) -> Optional[int]:
    """Map a raw corpus label to a CMUdict-39 id, or None to drop."""
    key = label.strip().lower()
    if source == "timit":
        dest = TIMIT_TO_CMUDICT.get(key)
    elif source == "buckeye":
        dest = BUCKEYE_TO_CMUDICT.get(key)
        if dest is None and key.endswith("n") and key[:-1] in BUCKEYE_TO_CMUDICT:
            dest = BUCKEYE_TO_CMUDICT[key[:-1]]
    else:
        raise ValueError(source)
    if dest is None:
        return None
    return CMUDICT_TO_ID[dest]


def is_timit_closure(label: str) -> bool:
    return label.strip().lower() in {"bcl", "dcl", "gcl", "pcl", "tcl", "kcl"}


def collapse_maps61_posteriors(post61: np.ndarray) -> np.ndarray:
    """Collapse ``[..., 61]`` MAPS softmax to ``[..., 39]`` CMUdict, renormalized."""
    collapsed = post61 @ MAPS61_TO_CMUDICT39_MATRIX
    denom = collapsed.sum(axis=-1, keepdims=True)
    denom = np.maximum(denom, 1e-12)
    return (collapsed / denom).astype(np.float32, copy=False)
