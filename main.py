from detector import Detector
from mite import  get_mites_from_bboxes, draw_mite_boxes
from tools import get_frames, preprocess_frames


#get images from usb folder:
usb_image_folder = "Data"  # Change this to your USB absolute path


def predict(folder_path):
    #get the image as np array from the folder path
    frames = get_frames(usb_image_folder)

    #pre process the frames:
    processed_frames = preprocess_frames(frames)  # Assuming preprocessing is done in get_frames    

    detector = Detector()

   
    # get the bounding boxes from the first frame:
    detector.run_detection(processed_frames[0]) 


    # get the mites from the image:
    mites = get_mites_from_bboxes(detector.result, frames)
    print("got mites:", len(mites))

 

    draw_mite_boxes(frames[0], mites, thickness=2, show=False, save_path="output.jpg")

    for mites in mites:
        print("Mite variability:", mites.variability)
    


predict(folder_path="Sample images/")