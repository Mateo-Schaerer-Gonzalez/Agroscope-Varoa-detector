from ultralytics import YOLO

class Detector:
    def __init__(self):
        self.model_path = "runs/detect/fine_tuned_varro_model12/weights/best.pt"  # Path to your fine-tuned model
        #self.model_path = "best.pt"  # pretrained model without fine-tuning
        self.model = YOLO(self.model_path, verbose=False)
        self.result = None

    def run_detection(self, image):
        # Run the model on a single image
        result = self.model(
            image,
            imgsz=1024,  #fine tuned uses 1024 / 6016
            max_det=2000,
            conf=0.1,
            iou=0.5,
            save=False,
            show_labels=False,
            line_width=2,
            save_txt=False,
            save_conf=False,
            verbose=False,
            batch=1,
            exist_ok=True,
        )

        self.result = result[0] # only for one image

    



# Train (fine-tune) on your dataset

if __name__ == '__main__':
    if False:
        # Load a pretrained model (or your existing one)
        model = YOLO("varrodetector/model/weights/best.pt")  # or 'varrodetector/model/weights/best.pt' if you want to continue training
        

        #freeze first 100 layers of the model
        layers = list(model.model.children())

        for i, layer in enumerate(layers[:100]):
            for param in layer.parameters():
                param.requires_grad = False


        model.train(
            data="yolo_data/data.yaml",  # path to your dataset YAML
            epochs=50,
            imgsz=1024,
            batch=16,
            name="fine_tuned_varro_model",
            resume=False  # True if you're continuing from a checkpoint
        )


