from imagededup.methods import PHash
from imagededup.utils import plot_duplicates
import os
import shutil


def dedup_images_from_folder(image_dir, th=5, only_create_duplicates=False):

    processed_folder = copy_to_two_folders(image_dir)
    image_dir = processed_folder
    phasher = PHash()
    # Generate encodings for all images in an image directory
    encodings = phasher.encode_images(image_dir, recursive=True)

    # Find duplicates using the generated encodings
    duplicates = phasher.find_duplicates(
        encoding_map=encodings, max_distance_threshold=th, recursive=True
    )
    if not only_create_duplicates:
        non_empty_count_1 = sum(
            1 for v in duplicates.values() if len(v) > 0
        )  # “if v” is True when the list is non‑empty
        print(
            f"there are {len(duplicates) - non_empty_count_1 } unique images from {len(duplicates)} images"
        )
        most_dups = 0
        for i in range(1, 10):
            non_empty_count = sum(
                1 for v in duplicates.values() if len(v) == i
            )  # “if v” is True when the list is non‑empty
            print(f"there are {round(non_empty_count/i+1)-1} images with {i+1} shows")
            if non_empty_count > 0:
                most_dups += 1
        non_empty_count = sum(
            1 for v in duplicates.values() if len(v) > 10
        )  # “if v” is True when the list is non‑empty
        print(f"there are {non_empty_count} images with more than {10} shows")
        filtered_dict = {k: v for k, v in duplicates.items() if len(v) > most_dups - 1}
        print(most_dups)
        duplicates_list = []
        if most_dups > 0:
            first_key = next(iter(filtered_dict))  # ➜ 'a'
            first_value = filtered_dict[first_key][0]

            plot_duplicates(
                image_dir=image_dir, duplicate_map=duplicates, filename=first_value
            )
            duplicates_list = phasher.find_duplicates_to_remove(
                encoding_map=encodings, max_distance_threshold=th, recursive=True
            )
        print(f"there are {duplicates_list.__len__()} to remove")
        os.path.join(image_dir, "train")
        images_to_remove = duplicates_list

        # Paths
        source_folder = os.path.join(image_dir)  # Folder containing the images

        # del files
        for image_name in images_to_remove:
            source_path = os.path.join(source_folder, image_name)

            try:
                # Ensure the destination subfolder exists
                if os.path.exists(source_path):
                    os.remove(source_path)
                    print(f"removed: {source_path}")
                else:
                    print(f"File not found: {image_name}")
            except Exception as e:
                print(f"Error moving {image_name}: {e}")
    return duplicates


def copy_to_two_folders(source_folder):
    """
    Copies all files and subfolders from source_folder into BOTH
    source_folder/original_data and source_folder/processed_data.
    """
    # ✅ Create target folders INSIDE source_folder
    # original_folder = os.path.join(source_folder, "original_data")
    processed_folder = os.path.join(
        os.path.split(source_folder)[:-1][0],
        os.path.split(source_folder)[-1] + "_no_dups_data",
    )

    # os.makedirs(original_folder, exist_ok=True)
    os.makedirs(processed_folder, exist_ok=True)

    # ✅ Iterate through items in source_folder
    for item in os.listdir(source_folder):
        src_path = os.path.join(source_folder, item)

        # ✅ Skip copying into itself
        if item in ["no_dups_data"]:
            continue

        # dst_original = os.path.join(original_folder, item)
        dst_processed = os.path.join(processed_folder, item)

        if os.path.isdir(src_path):
            # shutil.copytree(src_path, dst_original, dirs_exist_ok=True)
            shutil.copytree(src_path, dst_processed, dirs_exist_ok=True)
        else:
            # shutil.copy2(src_path, dst_original)
            shutil.copy2(src_path, dst_processed)

    print(f"✅ All files & folders copied to '{processed_folder}'")
    return processed_folder


def find_multiclass_images(image_dict):
    multiclass_images = []

    for key_path, value_paths in image_dict.items():
        if not value_paths:
            continue  # Skip if no values

        # Extract class of the key
        key_class = os.path.dirname(key_path).split("\\")[
            -1
        ]  # Handles Windows paths like 'Normal\\file.jpg'

        # Extract classes of all values
        value_classes = [os.path.dirname(val).split("\\")[-1] for val in value_paths]

        # If ANY value has a different class than the key, mark it
        if any(val_class != key_class for val_class in value_classes):
            component = [key_path]
            component.extend(value_paths)
            multiclass_images.append(component)

    return multiclass_images


import re


def natural_key(path):
    # Split into parts: text vs numbers for natural sorting
    return [
        int(text) if text.isdigit() else text.lower()
        for text in re.split(r"(\d+)", path)
    ]


import os
import cv2


def collect_duplicates_to_delete(image_dict, data_dir):
    duplicates_to_delete = []
    duplicates_to_keep = []

    for key_path, value_paths in image_dict.items():
        if len(value_paths) == 0:
            continue  # skip keys with no values

        # Extract class of key and all values
        key_class = os.path.dirname(key_path).split("\\")[-1]
        value_classes = [os.path.dirname(v).split("\\")[-1] for v in value_paths]

        # Only proceed if all values are from the same class as the key
        if all(val_class == key_class for val_class in value_classes):
            all_images = [key_path] + value_paths
            all_images = sorted(all_images, key=natural_key)
            # Measure resolutions
            resolutions = {}
            for img_path in all_images:
                img = cv2.imread(os.path.join(data_dir, img_path))
                if img is None:
                    continue  # skip unreadable files
                h, w = img.shape[:2]
                resolutions[img_path] = w * h  # resolution = width x height

            if not resolutions:
                continue  # skip if no valid images were read

            # Find image with max resolution
            keep_image = max(resolutions, key=resolutions.get)
            duplicates_to_keep.append(keep_image)
            # Add everything except the one to keep
            for img in all_images:
                if img != keep_image:
                    duplicates_to_delete.append(img)

    return duplicates_to_delete, duplicates_to_keep


def unique_list(lst):
    seen = set()
    unique = []
    for item in lst:
        if item not in seen:
            seen.add(item)
            unique.append(item)
    return unique
