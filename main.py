from detector import Detector
from mite import  get_mites_from_bboxes, draw_mite_boxes, Mite
import gradio as gr
import os
import cv2


def predict(image):
    detector = Detector()
    # image: numpy array (H, W, 3), RGB from Gradio upload
   

    detector.run_detection(image)


    # get the mites from the image:
    mites = get_mites_from_bboxes(detector.result)
    print("got mites:", len(mites))


   

    #add first frame to each mite ROI:
    for mite in mites:
        mite.add_image(image)


     # draw the bounding boxes on the image:
    return draw_mite_boxes(image, mites, thickness=2)


iface = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="numpy"),
    outputs=gr.Image(type="numpy"),
    title="Varroa Mite Detector",
    description="Upload an image to detect varroa mites using YOLOv8."
)

iface.launch()