"""Derive every logo asset the project ships from one master image.

The master is a square, white-background render of the JAMICA mark. Everything
downstream -- README, Sphinx navbar, browser tab, GitHub social card -- wants a
different crop, background, or palette, and hand-editing four files in a photo
editor is how they drift apart. This script regenerates all of them, so the
master is the only thing anyone has to keep.

Run it whenever the master changes:

    python scripts/make_logo_assets.py path/to/master.png

Six outputs, and the reason each one exists:

  logo.png         Trimmed, white knocked out to transparency. Transparency
                   matters because the Sphinx navbar and the README are not
                   both white -- a baked-in white background shows up as a
                   bright rectangle on any dark surface.

  logo-dark.png    Same, but with the wordmark lightened. The mark's text is
                   near-black, so once the white background is gone it becomes
                   invisible against a dark page rather than merely ugly. Only
                   low-saturation pixels are touched: the brain, the players
                   and the JAX glyph are chromatic and must survive untouched,
                   which is why this inverts by luminance under a saturation
                   guard rather than inverting the image.

  logo-mark.png    The emblem alone. The Sphinx navbar scales its logo to about
                   40px tall, at which height the stacked wordmark and its
                   subtitle are unreadable smears.

  logo-mark-dark.png
                   The emblem is very nearly all chromatic and so survives
                   either theme almost unchanged -- but not quite. The pianist's
                   stool is drawn in black, 0.25% of the opaque pixels, and it
                   is the one part that would silently vanish against a dark
                   navbar. Cheap enough to fix that there is no reason to ship
                   the emblem without a dark counterpart.

  favicon.ico      The emblem again, at tab sizes. No dark variant: the ICO is
                   composited against the browser's tab strip, not the page,
                   and browsers do not theme-switch favicons.

  logo-social.png  1280x640 on white, for GitHub's social preview card. That
                   card is cropped to a fixed aspect and composited on an
                   unknown background by every chat client that unfurls it,
                   which is the one place a flat white background is correct.

Pillow is the only dependency and is not a project dependency -- this is a
maintenance script, run by hand, not part of the build.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parent.parent
STATIC_DIR = REPO_ROOT / "docs" / "_static"

# Coverage below this is treated as nothing there. The master is a JPEG, and
# JPEG ringing puts a faint skirt of near-white pixels around every stroke;
# left in, it shows up as speckle once the background is gone. Genuine
# anti-aliasing this faint is not visually missed.
NOISE_FLOOR = 0.06

# Above this much chroma -- plain max(rgb) - min(rgb) -- a pixel counts as
# coloured and the dark-mode pass leaves it alone.
#
# Chroma rather than HLS saturation, which this used to use and which is the
# wrong tool here: saturation divides by lightness, so it is unstable exactly
# where the wordmark lives. A near-black (10, 10, 30) scores about 0.5
# saturation and reads as vividly coloured despite being visually black. JPEG
# chroma subsampling scatters pixels like that through black text, so the
# saturation test skipped a speckling of them and left holes in the glyphs,
# most visibly in the M, while half-inverting JAX's dark navy. Absolute chroma
# does not blow up in the dark and separates black text from the purple, blue
# and teal artwork cleanly.
CHROMA_GUARD = 0.20


def _load(master: Path) -> Image.Image:
    img = Image.open(master).convert("RGBA")
    if img.width != img.height:
        print(f"  note: master is {img.width}x{img.height}, not square")
    return img


def _unmix_from_white(img: Image.Image) -> Image.Image:
    """Recover the artwork's own colour and coverage from its white backing.

    The master is flat artwork composited onto white, so each pixel is
    ``observed = colour * a + 1 * (1 - a)``. Thresholding alpha and keeping the
    observed colour -- the obvious approach, and the one this used to take --
    is wrong: it leaves every anti-aliased edge fully opaque in a colour that
    was mixed with the background. Against white nobody notices, because the
    mixed colour matches what is behind it. Against anything darker the same
    pixels read as a pale rim around every stroke, and the dark-mode pass then
    inverts that rim into a bright halo.

    Solving the equation instead gives back both unknowns. Taking the darkest
    channel as full coverage, ``a = 1 - min(rgb)``, and the colour follows.
    Edges then carry the stroke's true colour at partial alpha and composite
    correctly onto any background.
    """
    rgb = np.asarray(img.convert("RGB"), dtype=np.float64) / 255.0
    a = 1.0 - rgb.min(axis=2)
    a[a < NOISE_FLOOR] = 0.0

    nonzero = np.where(a > 0, a, 1.0)[..., None]
    colour = np.clip((rgb - (1.0 - a)[..., None]) / nonzero, 0.0, 1.0)

    out = np.concatenate([colour, a[..., None]], axis=2)
    return Image.fromarray((out * 255.0).round().astype(np.uint8), "RGBA")


def _lighten_greyscale(img: Image.Image) -> Image.Image:
    """Invert the lightness of near-grey pixels, leaving coloured ones alone.

    Runs on unmixed colour, so a half-covered black glyph edge is black at
    alpha 0.5 rather than opaque grey. It inverts to white at alpha 0.5, which
    is the correct anti-aliasing for white text, instead of a grey fringe.
    """
    arr = np.asarray(img, dtype=np.float64) / 255.0
    rgb, a = arr[..., :3], arr[..., 3]

    hi, lo = rgb.max(axis=2), rgb.min(axis=2)
    light = (hi + lo) / 2.0
    chroma = hi - lo

    grey = (chroma <= CHROMA_GUARD) & (light < 0.6) & (a > 0)
    out = rgb.copy()
    out[grey] = (1.0 - light)[grey][..., None]

    merged = np.concatenate([out, a[..., None]], axis=2)
    return Image.fromarray((merged * 255.0).round().astype(np.uint8), "RGBA")


def _emblem(img: Image.Image) -> Image.Image:
    """Crop to the circular mark, dropping the wordmark below it.

    Found by scanning rows for the horizontal gap that separates artwork from
    text: the widest run of fully transparent rows in the upper two-thirds is
    that gap. Falling back to a fixed fraction if no gap is found keeps this
    from failing outright on a redesign.
    """
    bbox = img.getbbox()
    if bbox is None:
        raise SystemExit("master image is entirely transparent")
    art = img.crop(bbox)

    alpha = art.getchannel("A")
    limit = int(art.height * 0.75)
    blank = [alpha.crop((0, y, art.width, y + 1)).getbbox() is None for y in range(limit)]

    # Longest consecutive run of blank rows above the limit.
    best_start, best_len, run_start = None, 0, None
    for y in range(limit + 1):
        if y < limit and blank[y]:
            if run_start is None:
                run_start = y
        elif run_start is not None:
            if y - run_start > best_len:
                best_start, best_len = run_start, y - run_start
            run_start = None

    cut = best_start if best_len > 4 else int(art.height * 0.62)
    emblem = art.crop((0, 0, art.width, cut))
    emblem = emblem.crop(emblem.getbbox())

    side = max(emblem.width, emblem.height)
    square = Image.new("RGBA", (side, side), (0, 0, 0, 0))
    square.paste(emblem, ((side - emblem.width) // 2, (side - emblem.height) // 2), emblem)
    return square


def _social_card(img: Image.Image) -> Image.Image:
    """Letterbox the mark onto GitHub's 1280x640 card, on white."""
    card = Image.new("RGBA", (1280, 640), (255, 255, 255, 255))
    art = img.crop(img.getbbox())
    scale = min(1120 / art.width, 520 / art.height)
    art = art.resize(
        (max(1, int(art.width * scale)), max(1, int(art.height * scale))),
        Image.LANCZOS,
    )
    card.paste(art, ((1280 - art.width) // 2, (640 - art.height) // 2), art)
    return card.convert("RGB")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("master", type=Path, help="square master render of the mark")
    args = parser.parse_args()

    if not args.master.is_file():
        raise SystemExit(f"no such file: {args.master}")

    STATIC_DIR.mkdir(parents=True, exist_ok=True)
    print(f"master: {args.master}")

    raw = _load(args.master)
    light = _unmix_from_white(raw)
    light = light.crop(light.getbbox())

    def save(image: Image.Image, name: str, **kw: object) -> None:
        path = STATIC_DIR / name
        image.save(path, **kw)
        size = path.stat().st_size
        print(f"  {name:<18} {image.width}x{image.height}  {size / 1024:.0f} KiB")

    save(light, "logo.png")
    save(_lighten_greyscale(light), "logo-dark.png")

    emblem = _emblem(light)
    mark = emblem.copy()
    # Cap rather than resize to a fixed size: downscaling a 700px emblem to 512
    # threw away detail that retina navbars can show, and upscaling a small one
    # would invent detail that is not in the master. Only shrink, and only when
    # the emblem is genuinely larger than anything that will be displayed.
    if mark.width > 1024:
        mark = mark.resize((1024, 1024), Image.LANCZOS)
    save(mark, "logo-mark.png")
    save(_lighten_greyscale(mark), "logo-mark-dark.png")
    save(
        emblem,
        "favicon.ico",
        format="ICO",
        sizes=[(16, 16), (32, 32), (48, 48), (64, 64)],
    )
    save(_social_card(light), "logo-social.png", quality=95)

    print("\nSocial preview is not settable over the API -- upload")
    print("docs/_static/logo-social.png by hand under Settings > General.")


if __name__ == "__main__":
    main()
