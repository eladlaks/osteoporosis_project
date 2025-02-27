import os
import torch
import cv2
import numpy as np
from PIL import Image
from torchvision import transforms
import torchvision.transforms.functional as TF
from ultralytics import YOLO

class PreprocessingPipeline:
    def __init__(self, data_folder, processed_folder, yolo_model_path):
        self.data_folder = data_folder
        self.processed_folder = processed_folder
        self.yolo_model = YOLO(yolo_model_path)
        
        self.transforms = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])

    def crop_roi_with_yolo(self, image_path):
        image = Image.open(image_path).convert("RGB")
        results = self.yolo_model(image)
        
        if results and results[0].boxes:
            box = results[0].boxes.xyxy[0].cpu().numpy()
            x1, y1, x2, y2 = map(int, box)
            image = image.crop((x1, y1, x2, y2))

        return image

    def augment_image(self, image):
        if np.random.rand() > 0.5:
            image = TF.hflip(image)
        if np.random.rand() > 0.5:
            image = TF.adjust_brightness(image, brightness_factor=1.2)
        return image

    def process_images(self):
        os.makedirs(self.processed_folder, exist_ok=True)
        for class_folder in os.listdir(self.data_folder):
            class_path = os.path.join(self.data_folder, class_folder)
            output_class_path = os.path.join(self.processed_folder, class_folder)
            os.makedirs(output_class_path, exist_ok=True)

            if not os.path.isdir(class_path):
                continue

            for image_name in os.listdir(class_path):
                image_path = os.path.join(class_path, image_name)
                output_path = os.path.join(output_class_path, image_name)

                try:
                    image = self.crop_roi_with_yolo(image_path)
                    image = self.augment_image(image)
                    image = self.transforms(image)

                    image = TF.to_pil_image(image)
                    image.save(output_path)
                    print(f"Processed: {image_path} -> {output_path}")
                except Exception as e:
                    print(f"Error processing {image_path}: {e}")
