# Low-light object detection experiment

A small experiment to see how a pretrained YOLOv8-nano detector behaves when
photos look like night footage. Built with OpenCV and Ultralytics.

## What it does
- `step2_basics.py` - OpenCV basics: grayscale, blur, edge detection
- `step3_detect.py` - runs a pretrained YOLOv8n detector on photos
- `step4_night.py` - makes fake night versions (darkened, noisy, grayscale),
  re-runs the detector, and tests a simple enhancement (gamma + CLAHE)

## Findings (4 personal photos, one run)
- In normal light it found people, a chair and a bench (48-69% confidence).
- Darkening alone barely changed results.
- Noise plus grayscale cut detections (4 objects down to 1 in one photo)
  and lowered confidence.
- Simple brightening and contrast enhancement did not reliably help: it lost a
  person in one photo and produced doubtful labels in others.

## Limits
Only 4 photos, one run, three changes applied together, judged by eye.
This is a learning experiment, not a benchmark. No model was trained.

## Next steps
Change one thing at a time, use a labelled low-light dataset, measure
precision and recall, try better denoising, and fine-tune on night images.

## Run it
    pip install opencv-python ultralytics
    put your own photos in a folder named photos
    python step2_basics.py
    python step3_detect.py
    python step4_night.py