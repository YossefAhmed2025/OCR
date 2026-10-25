# Models

This directory holds the OCR model files used for **offline** inference.
It starts empty. Populate it once, with network access:

```bat
python scripts\download_models.py
```

## What gets downloaded

| Role | Model name | Used by |
|---|---|---|
| Text detection | `PP-OCRv5_mobile_det` | `ar`, `en`, `mixed` |
| Arabic-script recognition | `arabic_PP-OCRv5_mobile_rec` | `ar`, `mixed` |
| English recognition | `en_PP-OCRv5_mobile_rec` | `en`, `mixed` |

All three are official PaddleOCR PP-OCRv5 models. They are resolved by
the installed `paddleocr`/`paddlex` release from PaddlePaddle's own
model hosting, then copied here.

Reference:
<https://www.paddleocr.ai/main/en/version3.x/algorithm/PP-OCRv5/PP-OCRv5_multi_languages.html>

## Important: what the Arabic model actually is

`arabic_PP-OCRv5_mobile_rec` is an **Arabic-script** recognizer, not an
Arabic-language one. Its character set also covers Persian, Urdu,
Uyghur, Pashto, Kurdish, Sindhi and Balochi, plus Latin digits. Whether
its charset includes the full set of tashkeel, and how well it
recognises them, is an empirical question this project exists to
measure -- see `docs/arabic_diacritics.md`.

## manifest.json

`download_models.py` writes `manifest.json`, recording for every model:

- the local directory,
- each file name and byte size,
- a SHA-256 checksum per file,
- the `paddleocr` version that resolved it.

Those checksums pin **what this machine downloaded**. PaddleOCR does not
publish official per-file checksums, so they are a reproducibility
record for this install, not an upstream integrity check.

## Verifying

```bat
python scripts\download_models.py --verify
python scripts\verify_offline.py
```

## Re-downloading

```bat
python scripts\download_models.py --force
```
