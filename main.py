import sys
import os
# Ensure root of the project is in Python path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


from classes.detector import Detector
from utils.tools import get_frames, convert_yolo_to_coords  #, preprocess_frames
from classes.MiteManager import MiteManager


def predict(folder_path, name, num_per_plate, reanalyze=True):
    print("getting frames")

    frames = get_frames(folder_path)

    print("got frames")

    #processed_frames = preprocess_frames(frames)  # Assuming preprocessing is done in get_frames    
    detector = Detector()

    # get the bounding boxes from the first frame:
    detector.run_detection(frames[0]) 

    #get the reanalyze folder:
    if reanalyze:
        # Create the base reanalyzed folder
        base_dir = folder_path
        i = 1
        while True:
            results_folder = os.path.join(base_dir, f"reanalysis{i}")
            if not os.path.exists(results_folder):
                os.makedirs(results_folder)
                break
            i += 1

        print( f"reanalysis{i} created...")

       
    else:
        # Create the base results folder
        results_base = os.path.join(folder_path, "results")
        os.makedirs(results_base, exist_ok=True)

        # Find the next available recording subfolder (e.g., recording1, recording2, ...)
        i = 1
        while True:
            results_folder = os.path.join(results_base, f"recording{i}")
            if not os.path.exists(results_folder):
                os.makedirs(results_folder)
                break
            i += 1

    
        


    # get the mites from the image:
    stage = MiteManager(coordinate_file=f"Zoning/coordinates{num_per_plate}.txt",
                        mites_detection=detector.result, 
                        frames=frames,
                        name = name,
                        output_folder = results_folder,
                        reanalyze = i if reanalyze else 0)
   


    stage.mite_variability()

    stage.draw(frames[0], thickness=2)

    stage.Excelsummary()
