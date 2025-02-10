import fiftyone as fo
import fiftyone.utils.data as foud
import torch
import torchvision.transforms as T
from torchvision.models import resnet50
from PIL import Image
import numpy as np
from tqdm import tqdm

# Path to your folder with images (replace with your folder's path)
base_path = r"data\multi-class-knee-osteoporosis-x-ray-dataset\archive"

# Delete existing dataset if it exists
if "image_classification_dataset" in fo.list_datasets():
    print("Deleting existing dataset...")
    fo.delete_dataset("image_classification_dataset")

# Load the dataset from subfolders
dataset = fo.Dataset.from_dir(
    dataset_dir=base_path,
    dataset_type=fo.types.ImageClassificationDirectoryTree,
    name="image_classification_dataset",
)

# # Launch the FiftyOne App
session = fo.launch_app(dataset)

# Load a pre-trained ResNet50 model
model = resnet50(pretrained=True)
model.eval()

# Remove the final classification layer to use the embeddings
model = torch.nn.Sequential(*list(model.children())[:-1])

# Define the image transformation pipeline
transform = T.Compose(
    [
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


# Function to compute embeddings
def compute_embedding(filepath):
    # Load the image using Pillow
    image = Image.open(filepath).convert("RGB")
    input_tensor = transform(image).unsqueeze(0)  # Add batch dimension
    with torch.no_grad():
        embedding = model(input_tensor)
    return embedding.squeeze().numpy()


# Generate embeddings for all samples
print("Generating embeddings...")
embeddings = []
for sample in tqdm(dataset):
    embedding = compute_embedding(sample.filepath)
    sample["embedding"] = embedding.tolist()  # Convert to list for JSON serialization
    embeddings.append(embedding)
    sample.save()

# Add embeddings to the dataset
embeddings = np.vstack(embeddings)  # Stack all embeddings into a 2D array
labels = list(dataset.values("ground_truth.label"))  # Get labels for all samples

# Apply dimensionality reduction (t-SNE) for visualization
from sklearn.manifold import TSNE

print("Performing t-SNE...")
tsne = TSNE(n_components=2, random_state=42)
reduced_embeddings = tsne.fit_transform(embeddings)

# Add the reduced embeddings as sample fields
for sample, coords in zip(dataset, reduced_embeddings):
    sample["tsne_x"] = coords[0]
    sample["tsne_y"] = coords[1]
    sample.save()

# Visualize the embeddings in the FiftyOne App
session.dataset = dataset
session.view = dataset.sort_by("tsne_x")
# session.wait()

import fiftyone.brain as fob


# Image embeddings
fob.compute_visualization(dataset, brain_key="img_viz_22")

# # Object patch embeddings
fob.compute_visualization(dataset, patches_field="ground_truth", brain_key="gt_viz")

session = fo.launch_app(dataset)
