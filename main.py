import sys
import os

# Ensure root of the project is in Python path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from classes.detector import Detector
from utils.tools import get_frames, convert_yolo_to_coords
from classes.MiteManager import MiteManager
from classes.Plotter import Plotter
from utils.tools import read_counter, reset_counter, write_counter



def reanalyze_recording(results_base, num_per_plate, detector, frames_by_recording, discobox_run, name, Ground_truth):
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
        name=name)

        stage.update_mite_status(Ground_truth)

        stage.save_data(recording_count= i+1)


        plotter = Plotter(stage=stage,
                      output_folder=results_folder,
                      discobox_run=discobox_run)
        
        plotter.save_frame0_detection(frames[0], thickness=2)


        plotter.make_survival_graph( recording_number = i + 1)

        # save them to pdf
        plotter.create_recording_pdf(recording_count= i + 1)

        stage.save()

    # general summary:
    plotter.make_survival_time_graph()
    #plotter.distribution_graph()
    plotter.plot_variability_by_mite()
    
    plotter.excel_summary_mites()
    plotter.excel_summary_recordings()


    
    stage.reset()

   


def analyze_recording(results_base, num_per_plate, detector, frames, discobox_run, name, num_recordings, Ground_truth, count, time_between_recording):


    results_folder = os.path.join(results_base, "results", f"recording{count}")
    os.makedirs(results_folder, exist_ok=True)

    # analysis
    detector.run_detection(frames[0])


    stage = MiteManager(
        coordinate_file=f"Zoning/coordinates{num_per_plate}.txt",
        mites_detection=detector.result,
        frames=frames,
        name=name
    )

    stage.update_mite_status(Ground_truth)
    

    stage.save_data(recording_count=count)


    #make a plotter
    plotter = Plotter(stage=stage,
                      output_folder=results_folder,
                      discobox_run=discobox_run,
                      time_between_recordings = time_between_recording)
    
    
    plotter.save_frame0_detection(frames[0], thickness=2)


    plotter.make_survival_graph(recording_number = count)

        # save them to pdf
    plotter.create_recording_pdf(recording_count= count)

        
    stage.save()

    if count >= num_recordings:
        plotter.make_survival_time_graph()
        plotter.plot_variability_by_mite()
        plotter.excel_summary_mites()
        plotter.excel_summary_recordings()
        stage.reset()

       
def predict(folder_path, name, num_per_plate, reanalyze=False, discobox_run=False, num_recordings=1, count=1, time_between_rec=5):
    detector = Detector()
    frames = get_frames(folder_path, discobox_run, reanalyze)

    Ground_truth = "" # alive or dead
   

     # Run detection on first frame

    if discobox_run:
        results_base = folder_path
    else:
        results_base = "outputs"

 
    if reanalyze:
        reanalyze_recording(results_base, num_per_plate, detector, frames, discobox_run, name, Ground_truth)

    else:
        analyze_recording(results_base, num_per_plate, detector, frames, discobox_run, name, num_recordings, Ground_truth, count, time_between_rec)

    
    # plot variablitiy distribution collected so far
    # Load the CSV file

predict("Datasets/long_run_alive_5_recordings/", "test", num_per_plate=2, reanalyze=False)