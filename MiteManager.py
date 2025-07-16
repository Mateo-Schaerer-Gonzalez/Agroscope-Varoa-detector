import cv2
from mite import Mite
from Rect import TextZone, MiteZone

class MiteManager:
    def __init__(self):
        self.mites = []
        self.zones = []
        self.zone_map = {
            1: "mite_zone",
            0: "text_zone"
        }

    def draw_mites(self, image, thickness=2, save_path=None, draw_zones=False):
        for mite in self.mites:
            mite.draw(image, thickness=thickness)
       

        if draw_zones:
            for zone in self.zones:
                zone.draw(image, thickness=thickness)

        if save_path:
            cv2.imwrite(save_path, image)
           

    def get_zones(self, coordinate_file):
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
                            miteZone.add_text_zone(textZone)
                            break

               


    def getMites(self, result, frames):
        """
        Extract mites from detection results and frames.
        
        Parameters:
            result: Detection result containing bounding boxes.
            frames (np.ndarray): A 4D tensor of shape (num_frames, H, W, C)
        
        Returns:
            List of Mite objects.
        """
        
        boxes = result.boxes.xyxy.cpu().numpy().astype(int)  # numpy array
        for box in boxes:
            self.mites.append(Mite(box, frames))
        
        print("got mites:", len(self.mites))

    def print_mite_variability(self):
        for mite in self.mites:
            print("Mite alive, variability:", mite.checkAlive())


    def assign_mites(self):
        """
        Assigns mites to rectangles based on their bounding boxes.
        
        Parameters:
            rectangles (list of Rect): List of Rect objects to assign mites to.
        """
        
        for zone in self.zones:
            zone.assign_mites(self.mites)

                #update mite colors based on their assigned rectangle
