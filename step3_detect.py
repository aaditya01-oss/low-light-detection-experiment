from pathlib import Path
import cv2
from ultralytics import YOLO

PHOTOS = Path("photos")
OUT = Path("out")
OUT.mkdir(exist_ok=True)

model = YOLO("yolov8n.pt")   # small pretrained model; downloads about 6 MB the first time

files = [p for p in PHOTOS.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png")]

for path in files:
    result = model(str(path), conf=0.25, verbose=False)[0]   # keep guesses at least 25% confident

    print(f"\n{path.name}: {len(result.boxes)} objects found")
    for box in result.boxes:
        name = model.names[int(box.cls)]
        conf = float(box.conf)
        print(f"   {name}  ({conf:.0%} sure)")

    annotated = result.plot()                      # photo with boxes and labels drawn on it
    cv2.imwrite(str(OUT / f"{path.stem}_detected.jpg"), annotated)

print("\nDone. Open the 'out' folder and look at the *_detected.jpg files.")