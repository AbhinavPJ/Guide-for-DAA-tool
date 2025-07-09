to extract frames:
run->


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
threshold =4* 1e7  # New threshold

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




once this is done,we can start annotating,for this:


 create a virtual environment
then run this on terminal:
git clone https://github.com/ajeetkr-7/labelImg.git
cd labelImg



however all the requirements are not in the text file so whenever the program says module not found X, then run pip install X

once this is done,you might get a dependency conflict
all you need to do is downgrade to numpy <2.0.0

then when you run python labelimg.py, you might get issues like
ERROR: No matching distribution found for moviepy.editor

for this the fix is:
pip uninstall moviepy
pip install moviepy==1.0.3

after this,you may run into issues while resizing/clicking buttons on canvas this is because of some issues with the old version of modules,
the fixes are:
add this line in labelimg.py: from PyQt5 import QtWidgets
then in libs/canvas.py change line 224 to:
combined_flags = QDockWidget.DockWidgetFeature(
    int(self.dock.features()) ^ int(self.dock_features)
)
self.dock.setFeatures(combined_flags)

and line 168 to:
if Qt.RightButton & ev.buttons():


after this the annotation tool is ready,change Pascalvoc to YOLO and start annotating! All the best

create a .md file
