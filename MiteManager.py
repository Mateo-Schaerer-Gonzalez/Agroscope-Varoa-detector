import cv2
from mite import Mite
from Rect import TextZone, MiteZone
from TextReader import TextReader
from PIL import Image
from TextReader import has_text

class MiteManager:
    def __init__(self, coordinate_file, mites_detection, frames):
        self.mites = []
        self.zones = []
        self.frames = frames
        self.zone_map = {
            1: "mite_zone",
            0: "text_zone"
        }
        self.get_zones(coordinate_file) # get the zones from the coordinate file
        self.getMites(mites_detection, self.frames, self.zones)  # get the mites from the detection results and frames

    def draw(self, image, thickness=2, save_path=None, draw_zones=False):
        for zone in self.zones:
            zone.draw(image, thickness=thickness)

        if save_path is not None:
            cv2.imwrite(save_path, image)
           

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
                            img = textZone.get_ROI(self.frames)[2]
                            if has_text(img):
                                img_PIL = Image.fromarray(img).convert("RGB")  # Get the image from the ROI
                                textZone.text = textReader.read(img_PIL)
                            else:
                                textZone.text = "No mites in this zone"

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

    def print_mite_variability(self):
        for zone in self.zones:
            print(f"Zone {zone.zone_id} has {len(zone.mites)} mites.")
            
            """for mite in zone.mites:
                print(f"  Mite {mite.bbox} has variability: {mite.checkAlive()}")"""
                
         
