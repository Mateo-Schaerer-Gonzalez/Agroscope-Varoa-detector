import cv2
from classes.mite import Mite
from classes.Rect import TextZone, MiteZone
from classes.TextReader import TextReader
from PIL import Image
from utils.tools import read_counter
import pandas as pd
from openpyxl import load_workbook
from openpyxl.drawing.image import Image as OpenpyxlImage
import re
import pickle

import matplotlib
matplotlib.use('Agg')  # Must be set before importing pyplot


from matplotlib.backends.backend_pdf import PdfPages


import matplotlib.pyplot as plt
import os

class MiteManager:
    

    def __init__(self, coordinate_file, mites_detection, 
                 frames,  name, output_folder, reanalyze=0, discobox_run=False, recording_count=1):
        
        print("initializing stage..")
        base_dir = os.path.dirname(os.path.abspath(__file__))
        self.save_path = os.path.join(base_dir, "mite_manager.plk")
        

        if not os.path.isabs(coordinate_file):
            coordinate_file = os.path.abspath(
                os.path.join(
                    os.path.dirname(__file__),
                    "..",
                    coordinate_file  # assuming relative path is relative to the folder above this script
                )
            )
    
       

        if discobox_run:
            # get the output path
            self.output_path = os.path.abspath(os.path.join(base_dir, "..",  "..", output_folder))
        else:
            self.output_path = os.path.abspath(os.path.join(base_dir,"..",  output_folder))


        self.coordinate_file = coordinate_file
        self.zones = []
        self.name = name
        self.frames = frames
        self.zone_map = {
            0: "text_zone",
            1: "mite_zone"
        }

       
        self.get_zones(coordinate_file, recording_count) # get the zones from the coordinate file
        

        self.getMites(mites_detection, self.frames, self.zones)  # get the mites from the detection results and frames
        self.reanalyze = reanalyze
        self.img_size = (15,10)
        
        #check if there is a reanalyze folder:

    def save(self):
        with open(self.save_path, 'wb') as f:
            pickle.dump(self, f)


    def reset(self):
        """Delete save file and reset object to default state."""
        if os.path.exists(self.save_path):
            os.remove(self.save_path)
            print(f"Deleted save file: {self.save_path}")
        


    def draw(self, image, thickness=2):
        # Draw zones on the image
        for zone in self.zones:
            zone.draw(image, thickness=thickness)

        # Save image with a unique name
        filename = os.path.join(self.output_path, "frame_0.jpg")

        cv2.imwrite(filename, image)
        print(f"Image saved to: {filename}")
           

    def get_zones(self, coordinate_file, recording_count):
        zones_file = os.path.join(os.path.dirname(coordinate_file), "zones.pkl")
      
        if recording_count <= 1:
            textReader = TextReader() #load the text reader
            print("textReader loaded...")

            with open(coordinate_file, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) != 5:
                        #print(f"Skipping invalid line: {line.strip()}")
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
                                
                                img_PIL = Image.fromarray(img).convert("RGB")  # Get the image from the ROI
                                textZone.text = textReader.read(img_PIL)
                                

                                miteZone.add_text_zone(textZone)
                                break

            with open(zones_file, "wb") as f:
                pickle.dump(self.zones, f)
            print(f"Zones saved to {zones_file}")
            
        else:
            if os.path.exists(zones_file):
                with open(zones_file, "rb") as f:
                    self.zones = pickle.load(f)
                print(f"Zones loaded from {zones_file}")
            else:
                raise FileNotFoundError(f"No saved zone file found at {zones_file}")
     


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

    def mite_variability(self, Ground_truth):
        for zone in self.zones:
            print(f"Zone {zone.zone_id} has {len(zone.mites)} mites.")
            
            for mite in zone.mites:
                mite.checkAlive(Ground_truth)

    


    def get_summary_data(self):

        csv_path = os.path.join(self.output_path, "summary.csv")

        # Step 1: Prepare the data
        summary_data = []
        all_maxdiff = []

        for zone in self.zones:
            if zone.zone_id == "EMPTY":
                continue

            total = len(zone.mites)
            alive = sum(1 for mite in zone.mites if mite.alive)
            dead = total - alive
            survival_pct = (alive / total * 100) if total > 0 else 0.0

            # Collect variability values
            for mite in zone.mites:
                all_maxdiff.append(mite.max_diff)

            summary_data.append({
                "Zone ID": zone.zone_id,
                "Total Mites": total,
                "Alive Mites": alive,
                "Dead Mites": dead,
                "Survival %": round(survival_pct, 2)
            })

            df = pd.DataFrame(summary_data)


            df = df.groupby('Zone ID', as_index=False).agg({
            'Total Mites': 'sum',     # replace with actual column names
            'Alive Mites': 'sum',
            'Dead Mites': 'sum',
            'Survival %': 'mean'
            })

           

            df.to_csv(csv_path, index=False)

        return df, all_maxdiff
    

    def make_survival_graph(self,summary_data, all_maxdiff):
        out_path = os.path.join(self.output_path, "survival.png")

        zone_labels = summary_data['Zone ID']
       
        survival_rates = summary_data['Survival %']

        # Create a figure with 2 subplots side by side
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))

        # Subplot 1: Histogram of max pixel difference
        axes[0].hist(all_maxdiff, bins=20, color="steelblue", edgecolor="black")
        axes[0].set_title("Distribution of Mite Max Pixel Difference")
        axes[0].set_xlabel("Max Pixel Difference")
        axes[0].set_ylabel("Frequency")
        axes[0].grid(True)

        # Subplot 2: Bar chart of survival rates
        axes[1].bar(zone_labels, survival_rates, color="mediumseagreen", edgecolor="black")
        axes[1].set_title("Survival Rate by Zone")
        axes[1].set_xlabel("Zone ID")
        axes[1].set_ylabel("Survival %")
        axes[1].set_ylim(0, 100)
        axes[1].tick_params(axis='x', rotation=45)

        plt.tight_layout()
        plt.savefig(out_path)
        plt.close()


    def create_recording_pdf(self):
        pdf_path = os.path.join(self.output_path, "recording.pdf")
        csv_path = os.path.join(self.output_path, "summary.csv")
        frame_path = os.path.join(self.output_path, "frame_0.jpg")
        survival_path = os.path.join(self.output_path, "survival.png")

        with PdfPages(pdf_path) as pdf:
            # A4 size in inches: 8.27 x 11.69
            fig = plt.figure(figsize=(8.27, 11.69))

            # Adjust height_ratios to make the frame image larger
            # Example: table smaller, frame bigger, survival moderate
            n_rows = 3
            gs = fig.add_gridspec(n_rows, 1, height_ratios=[1, 3, 1.5])

            # --- Section 1: Summary Table ---
            if os.path.exists(csv_path):
                df = pd.read_csv(csv_path)
                ax1 = fig.add_subplot(gs[0])
                ax1.axis('off')
                table = ax1.table(cellText=df.values, colLabels=df.columns, loc='center', cellLoc='center')
                table.auto_set_font_size(False)
                table.set_fontsize(8)
                table.scale(1.0, 1.0)
                ax1.set_title("Zone Survival Summary", fontsize=12, pad=10)

            # --- Section 2: Frame Image (larger) ---
            if os.path.exists(frame_path):
                img = plt.imread(frame_path)
                ax2 = fig.add_subplot(gs[1])
                ax2.imshow(img)
                ax2.axis('off')
                ax2.set_title("Detection output", fontsize=12, pad=10)

            # --- Section 3: Survival Plot ---
            if os.path.exists(survival_path):
                img2 = plt.imread(survival_path)
                ax3 = fig.add_subplot(gs[2])
                ax3.imshow(img2)
                ax3.axis('off')
                ax3.set_title("Survival Rate by Zone", fontsize=12, pad=10)

            plt.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)

        print(f"PDF successfully saved to: {pdf_path}")


    def load_data(self, root_folder=None):
        if not root_folder:
            # Find the parent folder (results folder)
            parent_folder = os.path.dirname(self.output_path)

        pattern = re.compile(r"recording(\d+)", re.IGNORECASE)
        all_summaries = []

        for entry in os.listdir(parent_folder):
            full_path = os.path.join(parent_folder, entry)
            if os.path.isdir(full_path):
                match = pattern.match(entry)
                if match:
                    recording_number = int(match.group(1))
                    csv_path = os.path.join(full_path, "summary.csv")

                    if os.path.exists(csv_path):
                        df = pd.read_csv(csv_path)
                        df["Recording Number"] = recording_number
                        all_summaries.append(df)
                    else:
                        print(f"No summary.csv found in {csv_path}")

        if all_summaries:
            combined_df = pd.concat(all_summaries, ignore_index=True)
            return combined_df
        else:
            print("No CSV files loaded.")
            return pd.DataFrame()
        

    def make_survival_time_graph(self):
        save_path = os.path.abspath(os.path.join(self.output_path, os.pardir))
        survival_path = os.path.join(save_path, "surivival.png")

        df = self.load_data()
        
        # SURVIVAL BY ZONE VS TIME

        plt.figure(figsize=self.img_size)
        # Loop through each zone and plot survival across recordings
        for zone_id, zone_data in df.groupby('Zone ID'):
            plt.plot(zone_data['Recording Number'], zone_data['Survival %'], marker='o', label=f'Zone {zone_id}')

        plt.xlabel('Recording Number')
        plt.ylabel('Survival Rate (%)')
        plt.title('Survival Rate over Time by Zone')
        plt.ylim(0, 100) 
        plt.legend(title='Zone ID')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(survival_path)
        plt.close()

        print("survival time saved to", survival_path)

    
       
    def distribution_graph(self):
        script_dir = os.path.dirname(os.path.abspath(__file__))

        filename = os.path.join(script_dir, "variabilites.csv")


        save_path = os.path.abspath(os.path.join(self.output_path, os.pardir))
        local_diff = os.path.join(save_path, "local_diff.png")
        max_diff = os.path.join(save_path, "max_diff.png")
       
        df = pd.read_csv(filename)

        # Plot histogram for each group (Alive True/False)
        plt.figure(figsize=self.img_size)

        for alive_status, group in df.groupby('Ground_truth'):
            plt.hist(group['max_diff'], bins=30, alpha=0.6, label=alive_status)

        
        plt.xlabel('max difference')
        plt.ylabel('Frequency')
        plt.title('Histogram of local difference')
        plt.legend()
        plt.savefig(max_diff)
        plt.close()


        for alive_status, group in df.groupby('Ground_truth'):
            plt.hist(group['local_diff'], bins=30, alpha=0.6, label=alive_status)

        

        plt.xlabel('local difference')
        plt.ylabel('Frequency')
        plt.title('Histogram of local difference')
        plt.legend()
        plt.savefig(local_diff)
        plt.close()


       


       