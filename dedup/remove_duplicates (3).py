import os
import cv2
import imagehash
import numpy as np
import tkinter as tk
from tkinter import Toplevel, Label, Button, Checkbutton, IntVar
from PIL import Image, ImageTk
from collections import defaultdict
from skimage.metrics import structural_similarity as ssim

class DuplicateRemover:
    def __init__(self, data_folder, unknown_class_folder, log_file):
        self.data_folder = data_folder
        self.unknown_class_folder = unknown_class_folder
        self.log_file = log_file
        os.makedirs(self.unknown_class_folder, exist_ok=True)

    def get_image_hash(self, image_path, size=(224, 224)):
        image = Image.open(image_path).convert('L').resize(size)
        return imagehash.phash(image)

    def detect_duplicates(self):
        image_hashes = defaultdict(list)

        for class_folder in os.listdir(self.data_folder):
            class_path = os.path.join(self.data_folder, class_folder)
            if not os.path.isdir(class_path):
                continue

            for image_name in os.listdir(class_path):
                image_path = os.path.join(class_path, image_name)
                try:
                    img_hash = self.get_image_hash(image_path)
                    image_hashes[img_hash].append(image_path)
                except Exception as e:
                    print(f"Error processing {image_path}: {e}")

        duplicates = []
        with open(self.log_file, "w") as log:
            log.write("Possible Duplicates:\n")
            for img_list in image_hashes.values():
                if len(img_list) > 1:
                    duplicates.append(img_list)
                    log.write(f"{img_list}\n")

        print(f"Detection complete. {len(duplicates)} groups of possible duplicates found.")
        return duplicates

    def review_duplicates(self, duplicates):
        """Review duplicates one group at a time using a Tkinter GUI."""
        for duplicate_group in duplicates:
            if len(duplicate_group) > 1:
                self.display_image_group(duplicate_group)

    def display_image_group(self, img_paths):
        """Open a Tkinter GUI to manually review a group of duplicate images at once."""
        self.root = tk.Tk()
        self.root.title("Duplicate Review")

        self.img_vars = []  # Stores checkbox variables

        frame = tk.Frame(self.root)
        frame.pack()

        for img_path in img_paths:
            img = Image.open(img_path).resize((150, 150))  # Resize for display
            img_tk = ImageTk.PhotoImage(img)

            img_label = Label(frame, image=img_tk)
            img_label.image = img_tk
            img_label.pack(side=tk.LEFT)

            var = IntVar(value=0)  # Checkbox for each image
            chk = Checkbutton(frame, text=img_path, variable=var)
            chk.pack(side=tk.LEFT)
            self.img_vars.append((var, img_path))

        button_frame = tk.Frame(self.root)
        button_frame.pack()

        Button(button_frame, text="Delete Selected", command=self.approve_callback).pack(side=tk.LEFT)
        Button(button_frame, text="Keep All", command=self.reject_callback).pack(side=tk.RIGHT)

        self.root.mainloop()

    def approve_callback(self):
        """Delete selected images."""
        for var, img_path in self.img_vars:
            if var.get() == 1:  # If checkbox is checked, delete image
                print(f"Deleting {img_path}")
                os.remove(img_path)
        self.root.destroy()

    def reject_callback(self):
        """Keep all images and close window."""
        print("Keeping all images.")
        self.root.destroy()


if __name__ == "__main__":
    DATA_FOLDER = "data/images"
    UNKNOWN_CLASS_FOLDER = "unknown_class"
    LOG_FILE = "duplicates_log.txt"

    remover = DuplicateRemover(DATA_FOLDER, UNKNOWN_CLASS_FOLDER, LOG_FILE)
    duplicates = remover.detect_duplicates()
    remover.review_duplicates(duplicates)
