import sys
import os

# Ensure root of the project is in Python path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from classes.detector import Detector
from utils.tools import get_frames, convert_yolo_to_coords
from classes.MiteManager import MiteManager


def predict(folder_path, name, num_per_plate, reanalyze=False, discobox_run=False):
    print("getting frames")
    frames = get_frames(folder_path, discobox_run)
    print("got frames")

    detector = Detector()
    detector.run_detection(frames[0])  # Run detection on first frame

    if discobox_run:
        results_base = folder_path
    else:
        results_base = "outputs"

    i = 1
    if reanalyze:
        print("reanalysis is on")
        while True:
            reanalyze_path = os.path.join(results_base, f"reanalysis{i}")
            if not os.path.exists(reanalyze_path):
                os.makedirs(reanalyze_path)
                results_folder = os.path.join(reanalyze_path, f"recording{i}")
                os.makedirs(results_folder)
                print(f"reanalysis{i} created...")
                break
            i += 1
    else:
        results_base = os.path.join(results_base, "results")
        os.makedirs(results_base, exist_ok=True)

        while True:
            results_folder = os.path.join(results_base, f"recording{i}")
            if not os.path.exists(results_folder):
                os.makedirs(results_folder)
                break
            i += 1

    print("Output folder will be:", results_folder)
    print("i =", i)

    stage = MiteManager(
        coordinate_file=f"Zoning/coordinates{num_per_plate}.txt",
        mites_detection=detector.result,
        frames=frames,
        name=name,
        output_folder=results_folder,
        reanalyze=i if reanalyze else 0,
        discobox_run=discobox_run
    )

    stage.mite_variability()
    stage.draw(frames[0], thickness=2)
    stage.Excelsummary()


predict("Datasets/writing_test2/", "test", 1, reanalyze=True)