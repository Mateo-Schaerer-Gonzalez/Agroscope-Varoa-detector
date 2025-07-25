import cv2
from classes.mite import Mite
from classes.Rect import TextZone, MiteZone
from classes.TextReader import TextReader
from PIL import Image
from classes.TextReader import has_text
import pandas as pd
from openpyxl import load_workbook
from openpyxl.drawing.image import Image as OpenpyxlImage

import matplotlib
matplotlib.use('Agg')  # Must be set before importing pyplot



import matplotlib.pyplot as plt
import os

class MiteManager:
    def __init__(self, coordinate_file, mites_detection, frames,  name, output_folder):

        if not os.path.isabs(coordinate_file):
            coordinate_file = os.path.abspath(
                os.path.join(
                    os.path.dirname(__file__),
                    "..",
                    coordinate_file  # assuming relative path is relative to the folder above this script
                )
            )

        # get the output path
        base_dir = os.path.dirname(os.path.abspath(__file__))
        self.output_path = os.path.abspath(os.path.join(base_dir, "..",  "..", output_folder))


        self.coordinate_file = coordinate_file
        self.zones = []
        self.name = name
        self.frames = frames
        self.zone_map = {
            0: "text_zone",
            1: "mite_zone"
        }
        self.get_zones(coordinate_file) # get the zones from the coordinate file
        self.getMites(mites_detection, self.frames, self.zones)  # get the mites from the detection results and frames

    def draw(self, image, thickness=2):
        results_folder = os.path.join(self.output_path, "results")
        os.makedirs(results_folder, exist_ok=True)

        for zone in self.zones:
            zone.draw(image, thickness=thickness)


        filename = os.path.join(results_folder, "frame_0.jpg")
      
        cv2.imwrite(filename, image)
        print("image saved")
           

    def get_zones(self, coordinate_file):
        textReader = TextReader() #load the text reader
        print("textReader loaded...")

        with open(coordinate_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) != 5:
                    print(f"Skipping invalid line: {line.strip()}")
                    continue

                class_id = int(parts[0])
                x1, y1, x2, y2 = map(float, parts[1:])
                zone_id = self.zone_map[class_id]

                if zone_id == "mite_zone":
                     self.zones.append(MiteZone(int(x1), int(y1), int(x2), int(y2)))
                     
                if zone_id == "text_zone":
                    textZone = TextZone(int(x1), int(y1), int(x2), int(y2))
                    #find the mitebox it belongs to:
                    for miteZone in self.zones:
                        if textZone in miteZone:
                            textZone.parent_rect = miteZone
                            img = textZone.get_ROI(self.frames)[0]
                            if has_text(img):
                                img_PIL = Image.fromarray(img).convert("RGB")  # Get the image from the ROI
                                textZone.text = textReader.read(img_PIL)
                            else:
                                textZone.text = "EMPTY"

                            miteZone.add_text_zone(textZone)
                            break

               


    def getMites(self, result, frames, zones):
        """
        Extract mites from detection results and frames.
        
        Parameters:
            result: Detection result containing bounding boxes.
            frames (np.ndarray): A 4D tensor of shape (num_frames, H, W, C)
        
        Returns:
            List of Mite objects.
        """
        assigned = 0
        
        boxes = result.boxes.xyxy.cpu().numpy().astype(int)  # numpy array
       
        worked = False
        for i, box in enumerate(boxes):
            mite = Mite(box, frames)
            
            for zone in zones:
                if mite.bbox in zone:
                    worked = zone.assign_mites(mite)
                    break
            if worked:
                assigned += 1
                worked = False
     
        print("got mites:", len(boxes))
        print("assigned mites:", assigned)

    def mite_variability(self):
        for zone in self.zones:
            print(f"Zone {zone.zone_id} has {len(zone.mites)} mites.")
            
            for mite in zone.mites:
                mite.checkAlive()
                
         


    def Excelsummary(self):
        # Ensure the results output folder exists
        os.makedirs(results_folder, exist_ok=True)
        results_folder = os.path.join(self.output_path, "results")


        # Update file paths to use the results folder
        filename = os.path.join(results_folder, f"{self.name}_summary.xlsx")
        image_path = os.path.join(results_folder, f"{self.name}.jpg")
        hist_path = os.path.join(results_folder, f"{self.name}_variability_histogram.png")
        survival_path = os.path.join(results_folder, f"{self.name}_survival_path.png")
        

        # Step 1: Prepare the data
        summary_data = []
        all_variabilities = []

        for zone in self.zones:
            if zone.zone_id == "EMPTY":
                continue

            total = len(zone.mites)
            alive = sum(1 for mite in zone.mites if mite.alive)
            dead = total - alive
            survival_pct = (alive / total * 100) if total > 0 else 0.0

            # Collect variability values
            for mite in zone.mites:
                if hasattr(mite, "variability"):
                    all_variabilities.append(mite.variability)

            summary_data.append({
                "Zone ID": zone.zone_id,
                "Total Mites": total,
                "Alive Mites": alive,
                "Dead Mites": dead,
                "Survival %": round(survival_pct, 2)
            })

        # Step 2: Export summary to Excel
        df = pd.DataFrame(summary_data)
        df.to_excel(filename, index=False)

        # Step 3: Add first image if it exists
        wb = load_workbook(filename)
        ws = wb.active

        try:
            img = OpenpyxlImage(image_path)
            img.width = img.width * 0.5  # scale down to 50%
            img.height = img.height * 0.5
            img.anchor = "G2"
            ws.add_image(img)
        except FileNotFoundError:
            print(f"Image not found at {image_path}, skipping image insertion.")

        # Step 4: Create and insert variability histogram
        if all_variabilities:
            plt.figure(figsize=(6, 4))
            plt.hist(all_variabilities, bins=20, color="steelblue", edgecolor="black")
            plt.title("Distribution of Mite Variability")
            plt.xlabel("Variability")
            plt.ylabel("Frequency")
            plt.tight_layout()
            plt.savefig(hist_path)
            plt.close()

            # Insert histogram image
            try:
                hist_img = OpenpyxlImage(hist_path)
                

                hist_img.anchor = "W1"  # Placed lower to avoid overlap
                ws.add_image(hist_img)
            except FileNotFoundError:
                print(f"Histogram image not found at {hist_path}, skipping.")


        # alive % by zone 
         
        

        zone_labels = [row["Zone ID"] for row in summary_data]
        alive_percentages = [row["Survival %"] for row in summary_data]

        plt.figure(figsize=(8, 4))
        plt.bar(zone_labels, alive_percentages, color="mediumseagreen", edgecolor="black")
        plt.title("Survival Rate by Zone")
        plt.xlabel("Zone ID")
        plt.ylabel("Survival %")
        plt.ylim(0, 100)
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.savefig(survival_path)
        plt.close()

        # Insert bar chart into Excel
        try:
            alive_img = OpenpyxlImage(survival_path)
            alive_img.width = alive_img.width * 0.7
            alive_img.height = alive_img.height * 0.7
            alive_img.anchor = "w21"  
            ws.add_image(alive_img)
        except FileNotFoundError:
            print(f"Alive % chart not found at {survival_path}, skipping.")

        # Final save
        wb.save(filename)
        print("summary completed sucessfully ...")



