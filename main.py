from detector import Detector
from mite import  get_mites_from_bboxes, draw_mite_boxes, Mite
import os
import cv2

#run the detector:
detector = Detector()
detector.run_detection(image_folder="Single_image")
mites = []


#extract the bounding boxes from the results
for img in detector.results:
    mites = get_mites_from_bboxes(img)

# Suppose you have a list of Mite objects called mites_list
image_path = "Single_image/test1.jpg"
draw_mite_boxes(image_path, mites, color=(0, 0, 255), thickness=1, show=True, save_path="output.jpg")



