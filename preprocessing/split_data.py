import os
import shutil
import random
from pathlib import Path


def split_dataset(src_dir, train_dir, test_dir, test_ratio=0.2, seed=42):
    random.seed(seed)
    src_dir = Path(src_dir)
    train_path = Path(src_dir, train_dir)
    test_path = Path(src_dir, test_dir)
    classes_list = os.listdir(src_dir)
    train_path.mkdir(parents=True, exist_ok=True)
    test_path.mkdir(parents=True, exist_ok=True)

    for class_folder in classes_list:
        if (
            class_folder == train_dir
            or class_folder == test_dir
            or class_folder not in ["Normal", "Osteopenia", "Osteoporosis"]
        ):
            pass
        class_path = src_dir / class_folder
        if not class_path.is_dir():
            continue

        images = list(class_path.glob("*"))
        if not images:
            continue

        random.shuffle(images)
        num_test = int(len(images) * test_ratio)

        test_images = images[:num_test]
        train_images = images[num_test:]

        # Create target folders
        train_class_dir = train_path / class_folder
        test_class_dir = test_path / class_folder
        train_class_dir.mkdir(parents=True, exist_ok=True)
        test_class_dir.mkdir(parents=True, exist_ok=True)

        # Move images
        for img_path in train_images:
            shutil.copy2(str(img_path), str(train_class_dir / img_path.name))

        for img_path in test_images:

            shutil.copy2(str(img_path), str(test_class_dir / img_path.name))

        print(f"{class_folder}: {len(train_images)} train, {len(test_images)} test")
