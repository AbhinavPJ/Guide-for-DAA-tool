# Frame Extraction and Annotation Setup Guide

## Step 1: Extract Frames from Video

Run the following Python script to extract frames from your video:

```python
import cv2
import os
import numpy as np

video_path = "Top 20 Movies With The Best Gun Action [Z_2C1FJWfME].f137.mp4"
output_dir = "frames"

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS)
frame_skip = int(fps)  # Number of frames to skip = 1 second
frame_id = 0
saved_count = 0
threshold = 4 * 1e7  # New threshold

# Read the first frame to initialize prev_gray
ret, frame = cap.read()
if not ret:
    print("Failed to read the first frame.")
    exit(1)

prev_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

while True:
    # Skip to the next frame at +1 second
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
    ret, frame = cap.read()
    if not ret:
        break
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    diff = cv2.absdiff(gray, prev_gray)
    diff_score = np.sum(diff)
    
    print(f"Frame {frame_id}: diff score = {diff_score:.2f}")
    
    if diff_score > threshold:
        filename = f"{output_dir}/frame_{frame_id:05d}.jpg"
        cv2.imwrite(filename, frame)
        saved_count += 1
        print(f"  → Saved: {filename}")
    
    prev_gray = gray
    frame_id += frame_skip  # Advance by 1 second

cap.release()
print(f"\nSaved {saved_count} distinct frames.")
```

## Step 2: Setup Annotation Environment

### 2.1 Create Virtual Environment
```bash
python -m venv annotation_env
source annotation_env/bin/activate  # On Windows: annotation_env\Scripts\activate
```

### 2.2 Clone LabelImg Repository
```bash
git clone https://github.com/ajeetkr-7/labelImg.git
cd labelImg
```

### 2.3 Install Dependencies
Install the requirements, but be prepared for missing modules:
```bash
pip install -r requirements.txt
```

**Note**: The requirements file may be incomplete. When you encounter "module not found X" errors, install them individually:
```bash
pip install X
```

### 2.4 Fix Dependency Conflicts
If you encounter dependency conflicts, downgrade numpy:
```bash
pip install "numpy<2.0.0"
```

### 2.5 Fix MoviePy Issues
If you get this error: `ERROR: No matching distribution found for moviepy.editor`

Fix it with:
```bash
pip uninstall moviepy
pip install moviepy==1.0.3
```

## Step 3: Fix LabelImg Issues

### 3.1 Fix Import Issue
Add this line to `labelimg.py`:
```python
from PyQt5 import QtWidgets
```

### 3.2 Fix Canvas Issues
In `libs/canvas.py`, make the following changes:

**Line 224**: Replace with:
```python
combined_flags = QDockWidget.DockWidgetFeature(
    int(self.dock.features()) ^ int(self.dock_features)
)
self.dock.setFeatures(combined_flags)
```

**Line 168**: Replace with:
```python
if Qt.RightButton & ev.buttons():
```

## Step 4: Start Annotation

### 4.1 Run LabelImg
```bash
python labelimg.py
```

### 4.2 Configure for YOLO
1. Open the annotation tool
2. Change the format from PascalVOC to YOLO
3. Load your frames directory
4. Start annotating!

## Notes
- The frame extraction script saves frames at 1-second intervals only when there's significant change between frames
- The threshold value (4 * 1e7) can be adjusted based on your video content
- Make sure to activate your virtual environment before running the annotation tool
- The fixes address common issues with resizing and clicking buttons on the canvas due to version conflicts

Good luck with your annotation project!
