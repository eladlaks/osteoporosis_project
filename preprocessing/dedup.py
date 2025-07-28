from imagededup.methods import PHash
from imagededup.utils import plot_duplicates
import os
import shutil


def dedup_images_from_folder(image_dir, th=5):

    copy_to_two_folders(image_dir)
    image_dir = os.path.join(image_dir, "no_dups_data")
    phasher = PHash()
    # Generate encodings for all images in an image directory
    encodings = phasher.encode_images(image_dir, recursive=True)

    # Find duplicates using the generated encodings
    duplicates = phasher.find_duplicates(
        encoding_map=encodings, max_distance_threshold=th, recursive=True
    )
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
        print(f"there are {non_empty_count/i} images with {i+1} shows")
        if non_empty_count > 0:
            most_dups += 1
    non_empty_count = sum(
        1 for v in duplicates.values() if len(v) > 10
    )  # “if v” is True when the list is non‑empty
    print(f"there are {non_empty_count} images with more than {10} shows")
    filtered_dict = {k: v for k, v in duplicates.items() if len(v) > most_dups}
    first_key = next(iter(filtered_dict))  # ➜ 'a'
    first_value = filtered_dict[first_key][0]

    plot_duplicates(image_dir=image_dir, duplicate_map=duplicates, filename=first_value)
    duplicates_list = phasher.find_duplicates_to_remove(
        encoding_map=encodings, max_distance_threshold=th, recursive=True
    )
    print(f"there are {duplicates_list.__len__()} to remove")
    os.path.join(image_dir, "train")
    images_to_remove = duplicates_list

    # Paths
    source_folder = os.path.join(image_dir)  # Folder containing the images
    destination_folder = os.path.join(
        source_folder, "deleted_images"
    )  # Folder for moved images

    # Ensure the destination folder exists
    os.makedirs(destination_folder, exist_ok=True)

    # del files
    for image_name in images_to_remove:
        source_path = os.path.join(source_folder, image_name)

        try:
            # Ensure the destination subfolder exists
            if os.path.exists(source_path):
                os.remove(source_path)
            else:
                print(f"File not found: {image_name}")
        except Exception as e:
            print(f"Error moving {image_name}: {e}")


def copy_to_two_folders(source_folder):
    """
    Copies all files and subfolders from source_folder into BOTH
    source_folder/original_data and source_folder/processed_data.
    """
    # ✅ Create target folders INSIDE source_folder
    # original_folder = os.path.join(source_folder, "original_data")
    processed_folder = os.path.join(source_folder, "no_dups_data")

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
