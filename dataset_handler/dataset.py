import os
import glob
from PIL import Image
from torch.utils.data import Dataset
import hashlib
import config
import wandb

# from preprocessing.preprocess import preprocess_image


class ImageDataset(Dataset):
    def __init__(self, root_dir, data_kind, transform=None):
        """
        Args:
            root_dir (string): Directory with subdirectories for each class.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []
        if data_kind == "train":
            self._load_train_dataset()
        else:
            self._load_test_dataset

    def _load_train_dataset(self):
        # Remove duplicate images based on file hash
        seen_hashes = set()
        class_folders = [
            d
            for d in os.listdir(self.root_dir)
            if os.path.isdir(os.path.join(self.root_dir, d))
        ]
        class_folders.sort()  # ensure consistent ordering
        label = 0
        for class_name in class_folders:
            if wandb.config.USE_OSTEOPENIA or class_name != "Osteopenia":
                folder_path = os.path.join(self.root_dir, class_name)
                for img_path in glob.glob(os.path.join(folder_path, "*.*")):
                    try:
                        with open(img_path, "rb") as f:
                            img_bytes = f.read()
                            img_hash = hashlib.md5(img_bytes).hexdigest()
                        if img_hash in seen_hashes and wandb.config.SKIP_DUP_DATA:
                            continue  # skip duplicate image
                        seen_hashes.add(img_hash)
                        self.image_paths.append(img_path)
                        self.labels.append(label)
                    except Exception as e:
                        print(f"Error processing {img_path}: {e}")
                label += 1

    def _load_test_dataset(self):
        # Remove duplicate images based on file hash
        seen_hashes = set()
        class_folders = [
            d
            for d in os.listdir(self.root_dir)
            if os.path.isdir(os.path.join(self.root_dir, d))
        ]
        class_folders.sort()  # ensure consistent ordering
        label = 0
        for class_name in class_folders:
            folder_path = os.path.join(self.root_dir, class_name)
            for img_path in glob.glob(os.path.join(folder_path, "*.*")):
                try:
                    with open(img_path, "rb") as f:
                        img_bytes = f.read()
                        img_hash = hashlib.md5(img_bytes).hexdigest()
                    if img_hash in seen_hashes and wandb.config.SKIP_DUP_DATA:
                        continue  # skip duplicate image
                    seen_hashes.add(img_hash)
                    self.image_paths.append(img_path)
                    self.labels.append(label)
                except Exception as e:
                    print(f"Error processing {img_path}: {e}")
                label += 1

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        image = Image.open(img_path).convert("RGB")
        # Preprocess image (crop ROI and augment)
        # image = preprocess_image(image)
        if self.transform:
            image = self.transform(image)
        return image, label
