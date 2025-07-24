import os
import cv2
import numpy as np


def get_frames(folder_path, discobox_run=True):
    frames = []
    for root, dirs, files in os.walk(folder_path, followlinks=True):
        

        for fname in files:
            if not fname.lower().endswith(".bmp"):
                continue

            img_path = os.path.join(root, fname)
            img = cv2.imread(img_path)
            if img is None:
                print(f"Skipped (not image or unreadable): {img_path}")
                continue

            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            frames.append(img)

    if len(frames) == 0:
        raise ValueError("No images found in folder or none could be loaded.")

    return np.stack(frames)


def convert_yolo_to_coords(input_file, output_file, image_path):
    """Convert YOLO polygon format bounding boxes to pixel coordinates (x1, y1, x2, y2).
    
    Args:
        input_file (str): Path to the input file with YOLO format bounding boxes.
        output_file (str): Path to save the output file with pixel coordinates.
        image_path (str): Path to the image to get dimensions for conversion.
    
    Output format:
        class_id x1 y1 x2 y2  (pixel coordinates)
    """
    # Load image using cv2
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Image not found or unable to open: {image_path}")
    
    img_height, img_width = img.shape[:2]
    print(f"Image size: {img_width}x{img_height}")

    with open(input_file, 'r') as f_in, open(output_file, 'w') as f_out:
        for line in f_in:
            parts = line.strip().split()
            if len(parts) != 9:
                print(f"Skipping invalid line (expected 9 elements): {line.strip()}")
                continue

            class_id = parts[0]
            coords = list(map(float, parts[1:]))

            xs = coords[0::2]
            ys = coords[1::2]

            min_x, max_x = min(xs), max(xs)
            min_y, max_y = min(ys), max(ys)

            # Convert normalized coordinates to pixel coordinates
            x1_px = min_x * img_width
            y1_px = min_y * img_height
            x2_px = max_x * img_width
            y2_px = max_y * img_height

            f_out.write(f"{class_id} {x1_px:.2f} {y1_px:.2f} {x2_px:.2f} {y2_px:.2f}\n")

    print(f"Conversion complete! Output saved to {output_file}")
