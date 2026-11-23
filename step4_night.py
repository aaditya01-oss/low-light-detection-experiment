from pathlib import Path
import cv2
import numpy as np
from ultralytics import YOLO

PHOTOS = Path("photos")
OUT = Path("out_night")
OUT.mkdir(exist_ok=True)

model = YOLO("yolov8n.pt")
rng = np.random.default_rng(42)          # fixed seed so results repeat


def darken(img, factor=0.25):
    return (img.astype(np.float32) * factor).clip(0, 255).astype(np.uint8)


def add_noise(img, sigma=12):
    noise = rng.normal(0, sigma, img.shape)
    return (img.astype(np.float32) + noise).clip(0, 255).astype(np.uint8)


def to_gray3(img):
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)      # back to 3 channels so the model accepts it


def enhance(img):
    # 1) brighten with gamma, 2) boost local contrast (CLAHE)
    table = np.array([(i / 255.0) ** 0.5 * 255 for i in range(256)]).astype("uint8")
    bright = cv2.LUT(img, table)
    lab = cv2.cvtColor(bright, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)).apply(l)
    return cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)


def detect(img):
    r = model(img, conf=0.25, verbose=False)[0]
    found = [(model.names[int(b.cls)], float(b.conf)) for b in r.boxes]
    return found, r.plot()


files = [p for p in PHOTOS.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png")]

for path in files:
    original = cv2.imread(str(path))
    dark = darken(original)
    night = to_gray3(add_noise(dark))
    fixed = enhance(night)

    variants = [
        ("1 original", original),
        ("2 dark", dark),
        ("3 dark+noise+gray (fake night)", night),
        ("4 night then enhanced", fixed),
    ]

    print(f"\n=== {path.name} ===")
    for label, img in variants:
        found, annotated = detect(img)
        summary = ", ".join(f"{n} {c:.0%}" for n, c in found) or "nothing"
        print(f"{label:34s}: {len(found)} found -> {summary}")
        cv2.imwrite(str(OUT / f"{path.stem}_{label[0]}.jpg"), annotated)

print("\nDone. Open the 'out_night' folder and compare _1, _2, _3, _4 for each photo.")