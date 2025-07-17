from detector import Detector
from tools import get_frames, convert_yolo_to_coords  #, preprocess_frames
from MiteManager import MiteManager



#get images from usb folder:
usb_image_folder = "Data"  # Change this to your USB absolute path


def predict(folder_path):
    #get the image as np array from the folder path
    frames = get_frames(usb_image_folder)

    #processed_frames = preprocess_frames(frames)  # Assuming preprocessing is done in get_frames    
    detector = Detector()

    # get the bounding boxes from the first frame:
    detector.run_detection(frames[2]) 

    # get the mites from the image:
    stage = MiteManager(coordinate_file="coords.txt",
                        mites_detection=detector.result, 
                        frames=frames)
   


    stage.print_mite_variability()

    stage.draw(frames[2], thickness=2, save_path="output.jpg", draw_zones=True)


    
predict(usb_image_folder)

# get coordinats from file

"""convert_yolo_to_coords(image_path=r"Data\4_0_2021-8-20_19-5-34-922.bmp",
                     input_file="yolo_label.txt", 
                     output_file="coords.txt")
"""
