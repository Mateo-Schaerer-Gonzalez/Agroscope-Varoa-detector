import cv2

class Rect:
    def __init__(self, x1, y1, x2, y2, color=(0, 255, 0)):
        # Ensure coordinates are in correct order (x1,y1) top-left, (x2,y2) bottom-right
        self.x1, self.y1 = min(x1, x2), min(y1, y2)
        self.x2, self.y2 = max(x1, x2), max(y1, y2)
        self.color = color
       
    def draw(self, image, thickness=2):
        cv2.rectangle(image, (self.x1, self.y1), (self.x2, self.y2), self.color, thickness)

    def __contains__(self, other):
        # Return True if other Rect is fully inside this Rect
        return (self.x1 <= other.x1 <= other.x2 <= self.x2 and
                self.y1 <= other.y1 <= other.y2 <= self.y2)

    def __repr__(self):
        return f"Rect({self.x1}, {self.y1}, {self.x2}, {self.y2})"
    
    def __iter__(self):
        # This makes unpacking possible
        yield self.x1
        yield self.y1
        yield self.x2
        yield self.y2

    def get_ROI(self, frames):
        """
        Extract ROI from all frames and append to self.roi_series.
        
        Parameters:
            frames (np.ndarray): A 4D tensor of shape (num_frames, H, W, C)
        """
        # Bounding box coordinates
        roi = frames[:, self.y1:self.y2, self.x1:self.x2, :]  # Slice all frames with the bbox

        return roi
    

class TextZone(Rect):
    def __init__(self, x1, y1, x2, y2, text="EMPTY", parent_rect=None, color=(255, 0, 0)):
        super().__init__(x1, y1, x2, y2)
        self.text = text
        self.parent_rect = parent_rect
        self.color = color

        if parent_rect and not (self in parent_rect):
            raise ValueError("TextZone must be fully inside the parent Rect.")

    def draw(self, image, thickness=2, font_scale=1, font=cv2.FONT_HERSHEY_SIMPLEX):
        # Draw rectangle using Rect's method
        super().draw(image, thickness=thickness)


        # Add text
        cv2.putText(image, self.text, (self.x1, max(self.y1 - 10, 0)),
                    font, font_scale, (0,0,255), thickness)
        
    def __repr__(self):
        return f"TextZone({self.x1}, {self.y1}, {self.x2}, {self.y2}, text='{self.text}')"
    

    




class MiteZone(Rect):
    def __init__(self, x1, y1, x2, y2, zone_id=None):
        super().__init__(x1, y1, x2, y2)
        self.zone_id = zone_id
        self.text_zones = []
        self.mites = []
        self.color = (255, 0, 0)

    def add_text_zone(self, tz):
        assert tz in self, "Text zone must be inside the MiteRegion"
        self.text_zones.append(tz)
        self.zone_id = tz.text

    def assign_mites(self, mite):
        if mite.bbox in self:
            self.mites.append(mite)
            mite.assigned_rect = self
            #mite.bbox.color = self.color  # Assign mite to this big MiteRegion

            return True
        return False

    def draw(self, image, thickness=2):
        # Draw the bounding box
        super().draw(image, thickness=thickness)

        # Draw text zones
        for tz in self.text_zones:
            tz.draw(image, thickness=1)

        #draw mites
        for mite in self.mites:
            mite.bbox.draw(image, thickness=thickness)

        