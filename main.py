import sys
import os

# Ensure root of the project is in Python path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from classes.detector import Detector
from utils.tools import get_frames, convert_yolo_to_coords
from classes.MiteManager import MiteManager


def reanalyze_recording(results_base, num_per_plate, detector, frames_by_recording, discobox_run, name):
    i = 1
    while True:
        reanalyze_path = os.path.join(results_base, f"reanalysis{i}")
        if not os.path.exists(reanalyze_path):
            os.makedirs(reanalyze_path)
            print(f"reanalysis{i} created...")
            break
        i += 1

    # analysis
    for i, frames in enumerate(frames_by_recording):
        results_folder = os.path.join(reanalyze_path, f"recording{i+1}")
        os.makedirs(results_folder)

       
        detector.run_detection(frames[0])

        stage = MiteManager(
        coordinate_file=f"Zoning/coordinates{num_per_plate}.txt",
        mites_detection=detector.result,
        frames=frames,
        name=name,
        output_folder=results_folder,
        reanalyze=0,
        discobox_run=discobox_run)

        stage.mite_variability()
        stage.draw(frames[0], thickness=2)
        stage.Excelsummary()


def analyze_recording(results_base, num_per_plate, detector, frames, discobox_run, name):
    i = 1
    results_base = os.path.join(results_base, "results")
    os.makedirs(results_base, exist_ok=True)

    while True:
        results_folder = os.path.join(results_base, f"recording{i}")
        if not os.path.exists(results_folder):
            os.makedirs(results_folder)
            break
        i += 1

    # analysis
    detector.run_detection(frames[0])


    stage = MiteManager(
        coordinate_file=f"Zoning/coordinates{num_per_plate}.txt",
        mites_detection=detector.result,
        frames=frames,
        name=name,
        output_folder=results_folder,
        reanalyze=0,
        discobox_run=discobox_run
    )

    stage.mite_variability()
    stage.draw(frames[0], thickness=2)
    stage.Excelsummary()

    

    

def predict(folder_path, name, num_per_plate, reanalyze=False, discobox_run=False):
    detector = Detector()
    frames = get_frames(folder_path, discobox_run, reanalyze)
   

     # Run detection on first frame

    if discobox_run:
        results_base = folder_path
    else:
        results_base = "outputs"

 
    if reanalyze:
        reanalyze_recording(results_base, num_per_plate, detector, frames, discobox_run, name)

    else:
        analyze_recording(results_base, num_per_plate, detector, frames, discobox_run, name)
        

   

    


predict("Datasets/writing_test2/", "test", 1, reanalyze=False)