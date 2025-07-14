import os
import random
import shutil


destination_folder = "Data"
"""
source_root = "D:\DataVaroaMateo/"  # Change this to your source directory
# Walk through all subdirectories
for root, dirs, files in os.walk(source_root):
    for dir_name in dirs:
        # Check if the folder name is numeric
        if dir_name.isdigit():
            folder_path = os.path.join(root, dir_name)
            images = [f for f in os.listdir(folder_path)
                      if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]

            if images:
                chosen_img = random.choice(images)
                src_path = os.path.join(folder_path, chosen_img)

                # Use folder name in output filename to avoid collisions
                out_name = f"{dir_name}_{chosen_img}"
                dst_path = os.path.join(destination_folder, out_name)

                shutil.copy(src_path, dst_path)
                print(f"✔ Copied: {src_path} → {dst_path}")
            else:
                print(f"⚠ No images in: {folder_path}")
"""
print(f"Current number of images in '{destination_folder}': {len(os.listdir(destination_folder))}")