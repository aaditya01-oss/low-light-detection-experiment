import cv2
from pathlib import Path

PHOTOS = Path("photos")
OUT = Path("out")
OUT.mkdir(exist_ok=True)

files = [p for p in PHOTOS.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png")]
if not files:
    print("No photos found. Put 3 or 4 of your own photos in the 'photos' folder.")

for path in files:
    img = cv2.imread(str(path))            # a grid of pixels: height x width x 3 colours (BGR)
    if img is None:
        print("Could not read", path.name)
        continue

    h, w, c = img.shape
    print(f"{path.name}: {w}x{h} pixels, {c} colour values per pixel")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)     # 1 number per pixel (0 black to 255 white)
    blur = cv2.GaussianBlur(gray, (7, 7), 0)         # smooths noise
    edges = cv2.Canny(blur, 80, 160)                 # finds sharp changes = edges

    stem = path.stem
    cv2.imwrite(str(OUT / f"{stem}_1_gray.jpg"), gray)
    cv2.imwrite(str(OUT / f"{stem}_2_blur.jpg"), blur)
    cv2.imwrite(str(OUT / f"{stem}_3_edges.jpg"), edges)

print("Done. Open the 'out' folder to see the results.")