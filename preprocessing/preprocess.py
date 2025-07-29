import os
import matplotlib.pyplot as plt
import random
import glob
import matplotlib.pyplot as plt
from ultralytics import YOLO
from PIL import Image
import cv2
from tqdm import tqdm


# Function to print dataset statistics
def dataset_statistics(dataset_path, step_name, categories):
    """Prints the total number of images in each category."""
    print(f"\n📊 **Dataset Statistics After {step_name}:**")
    counts = {}
    for category in categories:
        category_path = os.path.join(dataset_path, category)
        num_images = (
            len(os.listdir(category_path)) if os.path.exists(category_path) else 0
        )
        counts[category] = num_images
        print(f"  📂 {category}: {num_images} images")

    # Plot bar chart
    plt.figure(figsize=(8, 5))
    plt.bar(counts.keys(), counts.values(), color=["blue", "orange", "green"])
    plt.title(f"Data Distribution - {step_name}")
    plt.ylabel("Number of Images")
    plt.xlabel("Category")
    plt.show()


def adaptive_centered_square_crop(image, x1, y1, x2, y2, output_size=224):
    """Crop a square around the center of the YOLO box using avg(width, height), then resize to output_size."""
    h, w, _ = image.shape
    x_center = (x1 + x2) // 2
    y_center = (y1 + y2) // 2

    box_w = x2 - x1
    box_h = y2 - y1
    crop_size = int((box_w + box_h) / 2)
    half_crop = crop_size // 2

    x1_crop = max(0, x_center - half_crop)
    y1_crop = max(0, y_center - half_crop)
    x2_crop = min(w, x_center + half_crop)
    y2_crop = min(h, y_center + half_crop)

    cropped = image[y1_crop:y2_crop, x1_crop:x2_crop]

    # Pad if smaller than crop_size (due to borders)
    pad_top = (crop_size - cropped.shape[0]) // 2
    pad_bottom = (crop_size - cropped.shape[0] + 1) // 2
    pad_left = (crop_size - cropped.shape[1]) // 2
    pad_right = (crop_size - cropped.shape[1] + 1) // 2

    padded = cv2.copyMakeBorder(
        cropped,
        pad_top,
        pad_bottom,
        pad_left,
        pad_right,
        borderType=cv2.BORDER_CONSTANT,
        value=[0, 0, 0],
    )

    # Resize to model input size
    resized = cv2.resize(padded, (output_size, output_size))
    return resized


def crop_and_save_knees(input_path, output_path, model, categories):
    """Uses YOLO to detect and save square knee crops using avg(w,h) with resize to 224x224."""
    for category in categories:
        category_path = os.path.join(input_path, category)
        output_category_path = os.path.join(output_path, category)

        # Get all images
        image_extensions = ["jpg", "jpeg", "png", "bmp", "tiff"]
        image_paths = []
        for ext in image_extensions:
            image_paths.extend(glob.glob(os.path.join(category_path, f"*.{ext}")))

        print(f"📂 Processing {category} - {len(image_paths)} images")

        for img_path in tqdm(image_paths, desc=f"Cropping {category}"):
            image = cv2.imread(img_path)
            if image is None:
                print(f"⚠️ Skipping unreadable image: {img_path}")
                continue

            results = model(image, verbose=False)
            cropped_knees = []
            width_threshold = image.shape[1] // 2

            for result in results:
                for box in result.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    confidence = box.conf[0].item()

                    cropped = adaptive_centered_square_crop(image, x1, y1, x2, y2)

                    if cropped.size != 0:
                        cropped_knees.append((cropped, confidence, x1, x2))

            # Separate left and right knees
            left_knee, right_knee = None, None
            left_conf, right_conf = 0, 0

            for knee, conf, x1, x2 in cropped_knees:
                if x2 < width_threshold:
                    if conf > left_conf:
                        left_knee, left_conf = knee, conf
                else:
                    if conf > right_conf:
                        right_knee, right_conf = knee, conf

            # Save cropped knees
            base_filename = os.path.basename(img_path)
            filename_no_ext, ext = os.path.splitext(base_filename)
            if left_knee is not None and right_knee is not None:
                # Concatenate side-by-side (left, right)
                both_knees = cv2.hconcat([left_knee, right_knee])
                both_output_path = os.path.join(
                    output_category_path, f"{filename_no_ext}_both{ext}"
                )
                cv2.imwrite(both_output_path, both_knees)
            else:
                if left_knee is not None:
                    # left_knee = cv2.flip(left_knee,1)
                    # both_knees = cv2.hconcat([left_knee, left_knee])
                    # both_output_path = os.path.join(output_category_path, f"{filename_no_ext}_left{ext}")
                    # cv2.imwrite(both_output_path, both_knees)
                    left_output_path = os.path.join(
                        output_category_path, f"{filename_no_ext}_left{ext}"
                    )
                    cv2.imwrite(left_output_path, left_knee)

                if right_knee is not None:
                    # left_knee = cv2.flip(right_knee,1)
                    # both_knees = cv2.hconcat([right_knee, right_knee])
                    # both_output_path = os.path.join(output_category_path, f"{filename_no_ext}_right{ext}")
                    # cv2.imwrite(both_output_path, right_knee)
                    right_output_path = os.path.join(
                        output_category_path, f"{filename_no_ext}_left{ext}"
                    )
                    cv2.imwrite(right_output_path, right_knee)

    print("\n✅ Step 2 Complete: Cropped knees saved with adaptive square logic.")


def show_yolo_performance(input_path, model, categories):
    # Get all image file paths from Step 1 (after duplicate removal)
    image_paths = glob.glob(input_path + "/*/*.png")  # Adjust for multiple categories

    # Ensure there are enough images in the folder
    if len(image_paths) == 0:
        print(
            "❌ No images found in the specified folder. Please check the path or file types."
        )
    else:
        # Select 10 random images (or less if there are fewer than 10)
        random_images = random.sample(image_paths, min(10, len(image_paths)))

        # Plot YOLO Predictions
        fig, axes = plt.subplots(
            2, 5, figsize=(20, 8)
        )  # 2 rows, 5 columns for 10 images

        for ax, image_path in zip(axes.flatten(), random_images):
            results = model(image_path, verbose=False)  # Run YOLO on the image
            result_img = results[0].plot()  # Get the annotated image

            ax.imshow(result_img)  # Display the image with bounding boxes
            ax.set_title(image_path.split("/")[-1])  # Show image filename
            ax.axis("off")  # Hide axes

        plt.tight_layout()
        plt.show()
