# Demo (torch-maps)

See the [upstream MAPS demo](https://github.com/MasonPhonLab/MAPS/tree/main/demo_files) for
background. This folder is the same example audio and transcription.

```bash
python maps.py \
  --audio=demo_files/dark_suit_sentence_16khz.wav \
  --text=demo_files/dark_suit_sentence_16khz.txt \
  --dict=demo_files/sample_dictionary.txt \
  --model=torch_models/timbuck_eng.pt \
  --overwrite
```

Use a full CMUdict for real data, not `sample_dictionary.txt` (11 words only).
