import sys
import os
# Ensure root of the project is in Python path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


from classes.detector import Detector
from utils.tools import get_frames, convert_yolo_to_coords  #, preprocess_frames
from classes.MiteManager import MiteManager


def predict(folder_path, name, num_per_plate):
    print("getting frames")
    #get the image as np array from the folder path
    frames = get_frames(folder_path)

    print("got frames")

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

predict("Datasets/Data/", "test3", 2)
