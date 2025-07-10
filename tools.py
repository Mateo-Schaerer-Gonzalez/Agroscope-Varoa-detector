import os
import cv2
import numpy as np


def get_frames(folder_path):
    frames = []

    for fname in os.listdir(folder_path):
        img_path = os.path.join(folder_path, fname)

        # Skip if it's a directory
        if os.path.isdir(img_path):
            print(f"Skipped directory: {fname}")
            continue

        img = cv2.imread(img_path)
        if img is None:
            print(f"Skipped (not image or unreadable): {fname}")
            continue

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        frames.append(img)

    if len(frames) == 0:
        raise ValueError("No images found in folder or none could be loaded.")

    return np.stack(frames)



import tensorflow as tf

def preprocess_frames(frames):
    # Ensure dtype is float32 and range is [0, 1]
    frames = tf.convert_to_tensor(frames, dtype=tf.float32)
    if tf.reduce_max(frames) > 1.0:
        frames = frames / 255.0


    # Reduce contrast
    mean_intensity = 0.5
    frames = mean_intensity + 0.5 * (frames - mean_intensity)
    frames = tf.clip_by_value(frames, 0.0, 1.0)

    output = (frames * 255.0)


    
    return np.asarray(tf.cast(output, tf.uint8))  # (N, H, W, C)
