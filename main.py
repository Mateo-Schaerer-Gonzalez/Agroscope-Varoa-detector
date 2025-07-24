from classes.detector import Detector
from utils.tools import get_frames, convert_yolo_to_coords  #, preprocess_frames
from classes.MiteManager import MiteManager



#get images from usb folder:
usb_image_folder = "Datasets/Data"


num_per_plate = 1 # assays per plate
name = "test2"



def predict(folder_path):
    #get the image as np array from the folder path
    frames = get_frames(usb_image_folder)

    #processed_frames = preprocess_frames(frames)  # Assuming preprocessing is done in get_frames    
    detector = Detector()

    # get the bounding boxes from the first frame:
    detector.run_detection(frames[5]) 

    # get the mites from the image:
    stage = MiteManager(coordinate_file=f"Zoning/coordinates{num_per_plate}.txt",
                        mites_detection=detector.result, 
                        frames=frames,
                        name = name)
   


    stage.mite_variability()

    stage.draw(frames[5], thickness=2, draw_zones=True)

    stage.Excelsummary()


    
predict(usb_image_folder)

# get coordinats from file

"""convert_yolo_to_coords(image_path=r"Data\4_0_2021-8-20_19-5-34-922.bmp",
                     input_file="yolo_label.txt", 
                     output_file="coords.txt")
"""
