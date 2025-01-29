import cv2
import numpy as np
import os

def detect_padding_color(image):
    """
    Detects the dominant color of the padding from the borders of the image.

    Args:
        image (numpy.ndarray): The input image.

    Returns:
        tuple: The RGB color of the padding.
    """
    # Take samples from the borders of the image
    top = image[0, :, :]        # Top border
    bottom = image[-1, :, :]    # Bottom border
    left = image[:, 0, :]       # Left border
    right = image[:, -1, :]     # Right border

    # Combine all border pixels into one array
    border_pixels = np.vstack((top, bottom, left, right))

    # Calculate the most common color
    unique_colors, counts = np.unique(border_pixels.reshape(-1, 3), axis=0, return_counts=True)
    dominant_color = unique_colors[np.argmax(counts)]

    return dominant_color


def remove_padding(image_path, output_path):
    """
    Removes padding of any color from an image.

    Args:
        image_path (str): Path to the input image.
        output_path (str): Path to save the cropped image.
    """
    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Failed to load image: {image_path}")
        return

    # Detect the padding color
    padding_color = detect_padding_color(image)

    # Create a mask where the padding color is excluded
    lower_bound = np.array(padding_color, dtype=np.uint8) - 5  # Add small tolerance
    upper_bound = np.array(padding_color, dtype=np.uint8) + 5
    mask = cv2.inRange(image, lower_bound, upper_bound)
    mask = cv2.bitwise_not(mask)  # Invert mask to keep non-padding regions

    # Find the bounding box of the content
    coords = cv2.findNonZero(mask)
    if coords is None:
        print(f"No content found in image: {image_path}")
        return

    x, y, w, h = cv2.boundingRect(coords)

    # Crop the image to the bounding box
    cropped = image[y:y+h, x:x+w]

    # Save the cropped image
    cv2.imwrite(output_path, cropped)
    print(f"Saved cropped image to: {output_path}")


# Example usage
if __name__ == "__main__":
    input_folder = "path_to_input_images"  # Folder with input images
    output_folder = "path_to_output_images"  # Folder to save cropped images

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for filename in os.listdir(input_folder):
        input_path = os.path.join(input_folder, filename)
        output_path = os.path.join(output_folder, filename)
        remove_padding(input_path, output_path)
