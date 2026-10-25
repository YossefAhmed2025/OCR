# Sample images

This directory is for small, shareable test images.

**It ships empty.** No sample page images are bundled: printed book
pages are usually copyrighted, and a synthetic image of rendered font
glyphs is not a substitute for a photographed page when the question is
"does this work on real books?".

`scripts/verify_offline.py` writes one file here, `offline_selftest.png`,
which it generates on the fly from a system font. That image is a smoke
test for the offline code path only. It is clean, flat, perfectly lit
rendered text — easy mode. **Do not quote accuracy numbers from it.**

## Getting real test images

For a baseline worth trusting, capture images the way Phonics Prism
eventually will: a camera photographing a physical page.

Cover at least these conditions, a handful of pages each:

| Condition | Why it matters |
|---|---|
| Arabic with full tashkeel | The core requirement. Diacritics are small and easy to lose. |
| Arabic without tashkeel | The baseline the vowelled score is compared against. |
| English | A separate model and a separate failure profile. |
| Mixed Arabic + English | Experimental; see `docs/architecture.md`. |
| Clean flat scan | The upper bound on what the model can do. |
| Photographed at an angle | What a camera rig actually produces. |
| Uneven lighting / shadow | Page curl and room lighting. |
| Slight motion blur | A child moving the book. |
| Small print | Diacritics may fall below the resolvable size. |

## Practical capture notes

- Shoot at the highest resolution available. Diacritics need pixels:
  below roughly 600px on the short side, `run_ocr.py` will warn you.
- Avoid flash directly on glossy paper — the specular highlight wipes
  out a whole region.
- Keep the camera roughly parallel to the page. `--deskew` corrects
  small in-plane rotation; it does **not** correct perspective.
- Do not pre-sharpen or denoise in your phone's editor. Feed the
  original, and use the pipeline's own flags so the preprocessing is
  recorded in the JSON.

## Then

1. Copy images into `data/input/`.
2. Label them — see `data/ground_truth/README.md`.
3. Evaluate — see `docs/evaluation_protocol.md`.
