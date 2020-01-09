import cv2
import numpy as np
from os.path import join
import glob

frames_path = "data/frames/test"
# masks_path = "data/masks/val"

frames = glob.glob(join(frames_path, "*.jpg"))
# masks = glob.glob(join(masks_path, "*.jpg"))

for frame in frames:
    img = cv2.imread(frame)
    cv2.imwrite(
        frame, cv2.resize(
            img, (256, 256)
        )
    )
'''
for mask in masks:
    img = cv2.imread(mask)
    cv2.imwrite(
        mask, cv2.resize(
            img, (256, 256)
        )
    )
'''
