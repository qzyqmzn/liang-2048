# -*- coding: utf-8 -*-
"""Build 640x640 tile images for the 2048 game.

Tiles 01-06 come straight from the six source photos (center-cropped).
Tiles 07-11 are stylised "ascension" edits matching the realm names:
  07 梁穹 / 08 梁元 / 09 梁道 / 10 梁极 / 11 梁无
"""
import os
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont, ImageEnhance

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = [
    "微信图片_20261120102648_45_20.jpg",
    "微信图片_20261120102649_46_20.jpg",
    "微信图片_20261120102650_47_20.jpg",
    "微信图片_20261120102651_48_20.jpg",
    "微信图片_20261120102652_49_20.jpg",
    "微信图片_20261120102653_50_20.jpg",
]
OUT = os.path.join(ROOT, "assets")
S = 1024
FINAL = 640


# ---------------------------------------------------------------- helpers
def load_square(path, size=S, zoom=1.0, dy=0.0):
    im = Image.open(os.path.join(ROOT, path)).convert("RGB")
    w, h = im.size
    side = min(w, h) / zoom
    left = (w - side) / 2
    top = (h - side) / 2 - dy * side
    im = im.crop((round(left), round(top), round(left + side), round(top + side)))
    return im.resize((size, size), Image.LANCZOS)


def arr(im):
    return np.asarray(im).astype(np.float32) / 255.0


def img(a):
    return Image.fromarray(np.clip(a * 255.0, 0, 255).astype(np.uint8))


def screen(base, layer, opacity=1.0, mask=None):
    lay = arr(layer)
    if mask is not None:
        lay = lay * arr(mask)[..., None]
    return img(1.0 - (1.0 - arr(base)) * (1.0 - lay * opacity))


def add(base, layer, opacity=1.0, mask=None):
    lay = arr(layer)
    if mask is not None:
        lay = lay * arr(mask)[..., None]
    return img(arr(base) + lay * opacity)


def radial_mask(size, cx, cy, inner, outer, power=1.6):
    y, x = np.mgrid[0:size, 0:size].astype(np.float32) / (size - 1)
    d = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
    t = np.clip((d - inner) / max(outer - inner, 1e-4), 0, 1) ** power
    return Image.fromarray(((1 - t) * 255).astype(np.uint8))


def face_mask(size, cx=0.5, cy=0.47, rx=0.29, ry=0.37, feather=0.085):
    """0 over the face, 1 outside — used to keep overlays off the portrait."""
    y, x = np.mgrid[0:size, 0:size].astype(np.float32) / (size - 1)
    d = np.sqrt(((x - cx) / rx) ** 2 + ((y - cy) / ry) ** 2)
    t = np.clip((d - 1.0) / feather, 0, 1)
    t = t * t * (3 - 2 * t)
    return Image.fromarray((t * 255).astype(np.uint8))


def smooth_noise(size, octaves=4, base=8, seed=0):
    r = np.random.default_rng(seed)
    acc = np.zeros((size, size), np.float32)
    amp, total = 1.0, 0.0
    for _ in range(octaves):
        small = r.random((base, base)).astype(np.float32)
        layer = np.asarray(
            Image.fromarray((small * 255).astype(np.uint8)).resize((size, size), Image.BICUBIC)
        ).astype(np.float32) / 255.0
        acc += layer * amp
        total += amp
        amp *= 0.55
        base *= 2
    return acc / total


def tint_layer(size, color, alpha_map):
    a = np.clip(alpha_map, 0, 1)[..., None]
    rgb = np.array(color, np.float32) / 255.0
    return img(np.ones((size, size, 3), np.float32) * rgb * a)


def bloom(im, threshold=0.72, radius=26, strength=0.55):
    a = arr(im)
    lum = a.max(axis=2, keepdims=True)
    bright = np.clip((lum - threshold) / (1 - threshold), 0, 1) * a
    b = img(bright).filter(ImageFilter.GaussianBlur(radius))
    return screen(im, b, strength)


def grade(im, lift=(0, 0, 0), gain=(1, 1, 1), contrast=1.0, saturation=1.0):
    a = arr(im)
    a = a * np.array(gain, np.float32) + np.array(lift, np.float32)
    a = (a - 0.5) * contrast + 0.5
    im = img(a)
    if saturation != 1.0:
        im = ImageEnhance.Color(im).enhance(saturation)
    return im


def vignette(im, color=(4, 3, 2), strength=0.55, cx=0.5, cy=0.44, inner=0.30, outer=0.80):
    size = im.size[0]
    m = arr(radial_mask(size, cx, cy, inner, outer, power=1.25))[..., None]
    tint = np.array(color, np.float32) / 255.0
    a = arr(im)
    a = a * (1 - (1 - m) * strength) + tint * (1 - m) * strength
    return img(a)


def rays(size, cx, cy, color, count=42, length=1.5, width=9, alpha=0.5, seed=1, blur=10):
    r = np.random.default_rng(seed)
    layer = Image.new("L", (size, size), 0)
    d = ImageDraw.Draw(layer)
    for i in range(count):
        ang = r.random() * np.pi * 2
        spread = r.uniform(0.012, 0.05)
        r0 = r.uniform(0.02, 0.12) * size
        w = r.uniform(0.35, 1.0) * width
        L = length * size
        p1 = (cx + np.cos(ang - spread) * r0, cy + np.sin(ang - spread) * r0)
        p2 = (cx + np.cos(ang) * L, cy + np.sin(ang) * L)
        p3 = (cx + np.cos(ang + spread) * r0, cy + np.sin(ang + spread) * r0)
        d.polygon([p1, p2, p3], fill=int(255 * r.uniform(0.25, 1.0)))
    layer = layer.filter(ImageFilter.GaussianBlur(blur))
    a = np.asarray(layer).astype(np.float32) / 255.0 * alpha
    return tint_layer(size, color, a)


def ring(size, cx, cy, rx, ry, color, width=4, alpha=0.6, blur=1.5, arc=None):
    layer = Image.new("L", (size, size), 0)
    d = ImageDraw.Draw(layer)
    box = [cx - rx, cy - ry, cx + rx, cy + ry]
    if arc is None:
        d.ellipse(box, outline=255, width=width)
    else:
        d.arc(box, arc[0], arc[1], fill=255, width=width)
    if blur:
        layer = layer.filter(ImageFilter.GaussianBlur(blur))
    a = np.asarray(layer).astype(np.float32) / 255.0 * alpha
    return tint_layer(size, color, a)


def stars(size, count=260, seed=3, color=(255, 246, 220), alpha=0.9):
    r = np.random.default_rng(seed)
    layer = Image.new("L", (size, size), 0)
    d = ImageDraw.Draw(layer)
    for _ in range(count):
        x, y = r.random() * size, r.random() * size
        s = r.random() ** 3 * 3.2 + 0.5
        d.ellipse([x, y, x + s, y + s], fill=int(120 + 135 * r.random()))
    layer = layer.filter(ImageFilter.GaussianBlur(0.6))
    a = np.asarray(layer).astype(np.float32) / 255.0 * alpha
    return tint_layer(size, color, a)


def sparkles(size, count=40, seed=4, color=(255, 240, 200), alpha=0.9):
    r = np.random.default_rng(seed)
    layer = Image.new("L", (size, size), 0)
    d = ImageDraw.Draw(layer)
    for _ in range(count):
        x, y = r.uniform(0.03, 0.97) * size, r.uniform(0.02, 0.75) * size
        s = r.uniform(7, 24)
        w = max(1, int(s * 0.08))
        d.line([x - s, y, x + s, y], fill=255, width=w)
        d.line([x, y - s, x, y + s], fill=255, width=w)
        d.ellipse([x - s * 0.1, y - s * 0.1, x + s * 0.1, y + s * 0.1], fill=255)
    layer = layer.filter(ImageFilter.GaussianBlur(1.0))
    a = np.asarray(layer).astype(np.float32) / 255.0 * alpha
    return tint_layer(size, color, a)


def nebula(size, colors, seed=5, alpha=0.8, center=(0.5, 0.45), start=0.10, end=0.62):
    n = smooth_noise(size, octaves=5, base=6, seed=seed)
    n = (n - n.min()) / (n.max() - n.min() + 1e-6)
    n = np.clip((n - 0.35) * 1.9, 0, 1) ** 1.35
    y, x = np.mgrid[0:size, 0:size].astype(np.float32) / (size - 1)
    m = np.clip((np.sqrt((x - center[0]) ** 2 + (y - center[1]) ** 2) - start) / (end - start), 0, 1)
    n = n * m * alpha
    out = np.zeros((size, size, 3), np.float32)
    for i, c in enumerate(colors):
        w = np.roll(n, int(i * size * 0.10), axis=1) * (0.55 + 0.45 * i)
        out += np.array(c, np.float32)[None, None, :] / 255.0 * w[..., None]
    return img(np.clip(out, 0, 1))


def particles(size, count=90, color=(255, 215, 140), seed=9, alpha=0.75, size_range=(2, 9)):
    r = np.random.default_rng(seed)
    layer = Image.new("L", (size, size), 0)
    d = ImageDraw.Draw(layer)
    for _ in range(count):
        x, y = r.random() * size, r.random() * size
        s = r.uniform(*size_range)
        d.ellipse([x, y, x + s, y + s], fill=int(255 * r.uniform(0.35, 1.0)))
    layer = layer.filter(ImageFilter.GaussianBlur(2.2))
    a = np.asarray(layer).astype(np.float32) / 255 * alpha
    return tint_layer(size, color, a)


def light_beams(size, cx, color, alpha=0.35, width=120, count=5, blur=40, top=0.0):
    layer = Image.new("L", (size, size), 0)
    d = ImageDraw.Draw(layer)
    for i in range(count):
        off = (i - (count - 1) / 2) * width * 0.95
        d.polygon([(cx + off - width * 0.14, top * size), (cx + off + width * 0.14, top * size),
                   (cx + off * 2.6 + width * 1.5, size), (cx + off * 2.6 - width * 1.5, size)],
                  fill=120)
    layer = layer.filter(ImageFilter.GaussianBlur(blur))
    a = np.asarray(layer).astype(np.float32) / 255 * alpha
    return tint_layer(size, color, a)


def token_streams(size, color=(150, 220, 255), seed=70, alpha=0.8, cols=26, vertical=True):
    """Floating token/data streams: columns (or rows) of glowing hex tokens."""
    r = np.random.default_rng(seed)
    layer = Image.new("L", (size, size), 0)
    d = ImageDraw.Draw(layer)
    try:
        font = ImageFont.load_default(size=15)
    except TypeError:
        font = ImageFont.load_default()
    hexd = list("0123456789abcdef")
    if vertical:
        for _ in range(cols):
            x = r.uniform(0.02, 0.98) * size
            y0 = r.uniform(-0.25, 0.85) * size
            for i in range(int(r.uniform(8, 30))):
                tok = "".join(r.choice(hexd) for _ in range(4))
                d.text((x, y0 + i * 20), tok, font=font, fill=int(90 + 165 * r.random()))
    else:
        for _ in range(cols):
            y = r.uniform(0.02, 0.98) * size
            x0 = r.uniform(-0.25, 0.85) * size
            for i in range(int(r.uniform(8, 30))):
                tok = "".join(r.choice(hexd) for _ in range(4))
                d.text((x0 + i * 34, y), tok, font=font, fill=int(90 + 165 * r.random()))
    layer = layer.filter(ImageFilter.GaussianBlur(0.8))
    a = np.asarray(layer).astype(np.float32) / 255 * alpha
    return tint_layer(size, color, a)


def grid_overlay(size, color=(120, 200, 255), alpha=0.18, step=64):
    layer = Image.new("L", (size, size), 0)
    d = ImageDraw.Draw(layer)
    for x in range(0, size, step):
        d.line([x, 0, x, size], fill=70, width=1)
    for y in range(0, size, step):
        d.line([0, y, size, y], fill=70, width=1)
    a = np.asarray(layer).astype(np.float32) / 255 * alpha
    return tint_layer(size, color, a)


def constellation(size, color=(150, 220, 255), seed=80, alpha=0.5, n=16):
    r = np.random.default_rng(seed)
    layer = Image.new("L", (size, size), 0)
    d = ImageDraw.Draw(layer)
    pts = [(r.random() * size, r.random() * size * 0.85) for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            if abs(pts[i][0] - pts[j][0]) + abs(pts[i][1] - pts[j][1]) < size * 0.35:
                d.line([pts[i], pts[j]], fill=140, width=1)
    for p in pts:
        d.ellipse([p[0] - 3, p[1] - 3, p[0] + 3, p[1] + 3], fill=255)
    layer = layer.filter(ImageFilter.GaussianBlur(0.8))
    a = np.asarray(layer).astype(np.float32) / 255 * alpha
    return tint_layer(size, color, a)


def grain(im, amount=0.035, seed=11):
    r = np.random.default_rng(seed)
    a = arr(im)
    n = r.normal(0, amount, a.shape[:2])[..., None]
    return img(a + n)


def finish(im, sharpen=1.4, glow=0.5):
    im = bloom(im, 0.70, 22, glow)
    im = im.filter(ImageFilter.UnsharpMask(radius=2.2, percent=int(60 * sharpen), threshold=3))
    im = grain(im, 0.022)
    return im.resize((FINAL, FINAL), Image.LANCZOS)


# ---------------------------------------------------------------- tiles
def build_existing():
    for i, name in enumerate(SRC, start=1):
        im = load_square(name, FINAL, zoom=1.0, dy=0.02)
        im = ImageEnhance.Contrast(im).enhance(1.03)
        im.save(os.path.join(OUT, f"tile-{i:02d}.jpg"), quality=90, optimize=True, progressive=True)


def tile_07():
    """梁穹 · 星河法袍 + token 数据流"""
    im = load_square(SRC[5], S, zoom=1.0, dy=0.035)
    im = grade(im, lift=(0.01, 0.005, 0.03), gain=(1.0, 0.98, 1.10), contrast=1.08, saturation=1.05)
    im = screen(im, nebula(S, [(24, 40, 110), (60, 20, 120), (10, 80, 130)], seed=71, alpha=1.0), 1.0)
    im = vignette(im, (3, 4, 14), 0.85, inner=0.16, outer=0.74)
    fm = face_mask(S, rx=0.28, ry=0.38)
    im = add(im, stars(S, 420, seed=72, color=(225, 235, 255), alpha=0.9), 1.0, fm)
    im = screen(im, token_streams(S, color=(140, 210, 255), seed=73, alpha=0.75, cols=30), 1.0, fm)
    im = screen(im, token_streams(S, color=(190, 160, 255), seed=74, alpha=0.5, cols=14, vertical=False), 1.0, fm)
    im = add(im, ring(S, S * 0.5, S * 0.35, S * 0.42, S * 0.42, (170, 215, 255), 3, 0.40, 1.8, arc=(190, 350)), 1.0, fm)
    im = add(im, sparkles(S, 40, seed=75, color=(200, 225, 255), alpha=0.60), 1.0, fm)
    return finish(im, 1.45, 0.6)


def tile_08():
    """梁元 · 半虚半实，光与向量，无固定面容"""
    im = load_square(SRC[5], S, zoom=1.0, dy=0.035)
    im = grade(im, lift=(0.01, 0.015, 0.03), gain=(1.0, 1.03, 1.08), contrast=1.12, saturation=0.85)
    # glitch slices: half-virtual, unfixed face
    a = arr(im)
    r = np.random.default_rng(81)
    h = a.shape[0]
    for _ in range(18):
        y0 = int(r.integers(0, h - 44))
        bh = int(r.integers(6, 44))
        dx = int(r.normal(0, 30))
        a[y0:y0 + bh] = np.roll(a[y0:y0 + bh], dx, axis=1)
    im = img(a)
    # RGB split ghost
    ar = arr(im)
    sh = np.empty_like(ar)
    sh[:, :, 0] = np.roll(ar[:, :, 0], 7, axis=1)
    sh[:, :, 1] = ar[:, :, 1]
    sh[:, :, 2] = np.roll(ar[:, :, 2], -7, axis=1)
    im = screen(im, img(sh), 0.45)
    # scanlines
    scan = (np.arange(S)[:, None] % 4 < 1).astype(np.float32) * 0.10
    im = img(arr(im) * (1 - scan))
    im = screen(im, nebula(S, [(8, 30, 50), (10, 16, 30)], seed=82, alpha=1.0), 0.9)
    im = screen(im, grid_overlay(S, (120, 200, 255), 0.20, 64), 1.0)
    im = vignette(im, (1, 4, 10), 0.88, inner=0.15, outer=0.72)
    fm = face_mask(S, rx=0.30, ry=0.40)
    im = screen(im, light_beams(S, S * 0.5, (150, 220, 255), alpha=0.40, width=170, count=5, blur=48), 1.0, fm)
    im = screen(im, constellation(S, (150, 220, 255), seed=83, alpha=0.55), 1.0, fm)
    im = screen(im, token_streams(S, color=(120, 230, 255), seed=84, alpha=0.55, cols=18), 1.0, fm)
    im = add(im, particles(S, 50, (180, 225, 255), seed=85, alpha=0.5, size_range=(1, 4)), 1.0, fm)
    return finish(im, 1.5, 0.55)


def tile_09():
    """梁道 · 混沌环 + 无尽数据流大道"""
    im = load_square(SRC[5], S, zoom=1.0, dy=0.035)
    im = grade(im, lift=(0.005, 0.01, 0.01), gain=(0.92, 0.97, 0.95), contrast=1.15, saturation=0.95)
    im = screen(im, nebula(S, [(10, 50, 46), (30, 24, 8), (6, 26, 40)], seed=91, alpha=1.0), 1.0)
    im = vignette(im, (2, 5, 4), 0.9, inner=0.14, outer=0.70)
    fm = face_mask(S, rx=0.30, ry=0.40)
    # chaos rings: rotating dashed arcs
    for k, (rr, w, al) in enumerate(((0.30, 5, 0.55), (0.38, 3, 0.45), (0.46, 2, 0.35), (0.54, 1, 0.25))):
        off = k * 47
        im = add(im, ring(S, S * 0.5, S * 0.36, S * rr, S * rr, (255, 200, 120), w, al, 1.8,
                         arc=(off, off + 200)), 1.0, fm)
        im = add(im, ring(S, S * 0.5, S * 0.36, S * rr, S * rr, (120, 235, 220), max(w - 2, 1), al * 0.8, 1.5,
                         arc=(off + 140, off + 300)), 1.0, fm)
    im = screen(im, token_streams(S, color=(130, 255, 190), seed=92, alpha=0.80, cols=34), 1.0, fm)
    im = screen(im, token_streams(S, color=(255, 210, 140), seed=93, alpha=0.45, cols=12, vertical=False), 1.0, fm)
    yy, xx = np.mgrid[0:S, 0:S].astype(np.float32) / (S - 1)
    core = np.clip(0.7 - np.sqrt((xx - 0.5) ** 2 + (yy - 0.36) ** 2) * 1.6, 0, 1) ** 1.6
    im = screen(im, tint_layer(S, (255, 190, 110), core).filter(ImageFilter.GaussianBlur(50)), 0.55)
    im = add(im, sparkles(S, 30, seed=94, color=(255, 220, 160), alpha=0.5), 1.0, fm)
    return finish(im, 1.45, 0.6)


def tile_10():
    """梁极 · 纯白至高奇点"""
    im = load_square(SRC[5], S, zoom=1.02, dy=0.035)
    im = grade(im, lift=(0.08, 0.08, 0.08), gain=(1.28, 1.26, 1.22), contrast=1.05, saturation=0.55)
    y, x = np.mgrid[0:S, 0:S].astype(np.float32) / (S - 1)
    back = np.clip(1.0 - np.sqrt((x - 0.5) ** 2 + (y - 0.38) ** 2) * 1.5, 0, 1) ** 1.4
    im = screen(im, tint_layer(S, (255, 253, 246), back).filter(ImageFilter.GaussianBlur(45)), 1.0)
    im = vignette(im, (255, 252, 244), 0.55, inner=0.34, outer=1.0)
    fm = face_mask(S, rx=0.27, ry=0.37)
    im = screen(im, light_beams(S, S * 0.5, (255, 255, 255), alpha=0.55, width=220, count=6, blur=50), 1.0, fm)
    im = screen(im, rays(S, S * 0.5, S * 0.38, (255, 255, 255), count=56, alpha=0.38, seed=101, blur=9), 1.0, fm)
    for r_, w, a in ((0.26, 4, 0.55), (0.34, 3, 0.45), (0.42, 2, 0.35), (0.50, 1, 0.25)):
        im = add(im, ring(S, S * 0.5, S * 0.36, S * r_, S * r_, (255, 255, 255), w, a, 1.6, arc=(192, 348)), 1.0, fm)
    im = add(im, particles(S, 110, (255, 255, 255), seed=102, alpha=0.6, size_range=(1, 6)), 1.0, fm)
    im = add(im, sparkles(S, 50, seed=103, color=(255, 255, 255), alpha=0.7), 1.0, fm)
    im = bloom(im, 0.55, 42, 0.8)
    return finish(im, 1.1, 0.5)


def tile_11():
    """梁无 · 空无，微光奇点"""
    im = load_square(SRC[5], S, zoom=1.02, dy=0.035)
    im = grade(im, lift=(0, 0, 0), gain=(0.34, 0.33, 0.38), contrast=1.1, saturation=0.5)
    im = vignette(im, (0, 0, 0), 0.96, inner=0.10, outer=0.62)
    # faint singularity above: tiny core + halo + one thin ring
    y, x = np.mgrid[0:S, 0:S].astype(np.float32) / (S - 1)
    d = np.sqrt((x - 0.5) ** 2 + (y - 0.30) ** 2)
    core = np.clip(1 - d * 22, 0, 1) ** 2
    halo = np.clip(1 - d * 3.2, 0, 1) ** 3 * 0.5
    im = screen(im, tint_layer(S, (235, 240, 255), core).filter(ImageFilter.GaussianBlur(2)), 1.0)
    im = screen(im, tint_layer(S, (150, 170, 220), halo).filter(ImageFilter.GaussianBlur(18)), 0.9)
    im = add(im, ring(S, S * 0.5, S * 0.30, S * 0.10, S * 0.10, (200, 215, 240), 1, 0.5, 1.2))
    im = add(im, stars(S, 90, seed=111, color=(200, 210, 230), alpha=0.5))
    im = add(im, particles(S, 24, (180, 195, 230), seed=112, alpha=0.35, size_range=(1, 3)))
    return finish(im, 1.3, 0.4)


def contact_sheet(paths, out):
    thumbs = [Image.open(p).convert("RGB").resize((320, 320), Image.LANCZOS) for p in paths]
    cols = 4
    rows = (len(thumbs) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * 320 + (cols + 1) * 8, rows * 320 + (rows + 1) * 8), (18, 18, 20))
    for i, t in enumerate(thumbs):
        sheet.paste(t, (8 + (i % cols) * 328, 8 + (i // cols) * 328))
    sheet.save(out, quality=88)


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    build_existing()
    for i, fn in enumerate([tile_07, tile_08, tile_09, tile_10, tile_11], start=7):
        fn().save(os.path.join(OUT, f"tile-{i:02d}.jpg"), quality=90, optimize=True, progressive=True)
        print("built tile", i)
    contact_sheet([os.path.join(OUT, f"tile-{i:02d}.jpg") for i in range(1, 12)],
                  os.path.join(ROOT, "build", "contact.jpg"))
