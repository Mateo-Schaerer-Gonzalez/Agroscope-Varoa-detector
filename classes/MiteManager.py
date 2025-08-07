import cv2
from classes.mite import Mite
from classes.Rect import TextZone, MiteZone
from classes.TextReader import TextReader
from PIL import Image
import pandas as pd
import pickle
import os

class MiteManager:
    

    def __init__(self, mites_detection, 
                 frames, coordinate_file, name):
        
        print("initializing stage..")
        base_dir = os.path.dirname(os.path.abspath(__file__))
        self.save_path = os.path.join(base_dir, "mite_manager.plk")

        

        if os.path.exists(self.save_path):
            self.load_miteManager(frames)
    
        else:
            self.zones = []
            self.name = name
            self.frames = frames
            self.zone_map = {
                0: "text_zone",
                1: "mite_zone"
            }

            self.get_zones(coordinate_file) # get the zones from the coordinate file
            self.getMites(mites_detection)  # get the mites from the detection results and frames
            self.img_size = (15,10)
            self.frame0 = None
            self.data = pd.DataFrame()
            self.mite_data = pd.DataFrame()
            self.reloaded = False
        

    def save(self):
        with open(self.save_path, 'wb') as f:
            pickle.dump(self, f)


    def load_miteManager(self,frames):
        with open(self.save_path, 'rb') as f:
            loaded = pickle.load(f)
            self.__dict__.update(loaded.__dict__)
            self.reloaded = True
            self.update_mites(frames)


    def load_coordinate_file(self,coordinate_file):
        if not os.path.isabs(coordinate_file):
            coordinate_file = os.path.abspath(
                os.path.join(
                    os.path.dirname(__file__),
                    "..",
                    coordinate_file  # assuming relative path is relative to the folder above this script
                )
            )
            self.coordinate_file = coordinate_file
        else:
            self.coordinate_file = None
            raise NameError("coordinate file not found")


    def reset(self):
        """Delete save file and reset object to default state."""
        if os.path.exists(self.save_path):
            os.remove(self.save_path)
            print(f"Deleted save file: {self.save_path}")
        
           

    def get_zones(self,coordinate_file):
        self.load_coordinate_file(coordinate_file)
    
        

        with open(self.coordinate_file, 'r') as f:
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
                            miteZone.add_text_zone(textZone)
                            break


    def getMites(self, result):
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
            mite = Mite(box, self.frames)
            
            for zone in self.zones:
                if mite.bbox in zone and all(mite.bbox not in text_zone for text_zone in zone.text_zones):
                    worked = zone.assign_mites(mite)
                    break
            if worked:
                assigned += 1
                worked = False
     
        print("got mites:", len(boxes))
        print("assigned mites:", assigned)


        # get the labels of the mites
        textReader = TextReader() #load the text reader
        print("textReader loaded...")
        for zone in self.zones:
            if zone.mites: #if it has mites read the label
                for text_zone in zone.text_zones:
                    print(text_zone)
                    img = text_zone.get_ROI(self.frames)[0]
                            
                    img_PIL = Image.fromarray(img).convert("RGB")  # Get the image from the ROI
                    text_zone.text = textReader.read(img_PIL)

                    #update the zone id aswell
                    zone.zone_id = text_zone.text
        




    def update_mites(self, frames):
        for zone in self.zones:
            for mite in zone.mites:
                mite.update_ROI(frames)


    def update_mite_status(self, Ground_truth):
        save = Ground_truth == 'alive' or Ground_truth == 'dead'

        for zone in self.zones:
            print(f"Zone {zone.zone_id} has {len(zone.mites)} mites.")
            
            for mite in zone.mites:
                mite.update_status()
                mite.update_status_severin()
    
                if save:
                    mite.save_with_ground_truth(Ground_truth)
        

    def save_data(self, recording_count):
        # Step 1: Prepare the data
        summary_data = []
        mite_data = []

        for zone in self.zones:
            if zone.zone_id == "EMPTY":
                continue

            total = len(zone.mites)
            alive = sum(1 for mite in zone.mites if mite.alive)
            dead = total - alive
            survival_pct = (alive / total * 100) if total > 0 else 0.0

          
            summary_data.append({
                "Zone ID": zone.zone_id,
                "Total Mites": total,
                "Alive Mites": alive,
                "Dead Mites": dead,
                "Survival %": round(survival_pct, 2),
                "recording": recording_count
            })

            # collect individual mite data
            for mite in zone.mites:
                mite_data.append(mite.to_dict(recording_count))


        if summary_data:
            df_summary = pd.DataFrame(summary_data)
            self.data = pd.concat([df_summary, self.data], ignore_index=True)
             #merge identicall labels
            self.data = (
                self.data
                .groupby(['Zone ID', 'recording'], as_index=False)
                .agg({
                    'Total Mites': 'sum',
                    'Alive Mites': 'sum',
                    'Dead Mites': 'sum',
                })
            )

            #recalculate survival rate after merge
            self.data['Survival %'] = ((self.data['Alive Mites'] / self.data['Total Mites']) * 100).round(2)


            # Sort by recording and Zone
            self.data = self.data.sort_values(by=['Zone ID', 'recording'])
           

        else:
            print("found no mites..")
        if mite_data:


            df_mites = pd.DataFrame(mite_data)

        
            self.mite_data = pd.concat([df_mites, self.mite_data], ignore_index=True)
            self.mite_data = self.mite_data.sort_values(by=['mite ID', 'recording'])
        else:
            print("NO Mite data found")


       

        
                
        return self.data, self.mite_data



       


       