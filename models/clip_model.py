import torch
import clip
from torch import nn

class CLIPModel(nn.Module):
    def __init__(self, device=None):
        super(CLIPModel, self).__init__()
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        # Load the pre-trained CLIP model with a ViT backbone
        self.model, self.preprocess = clip.load("ViT-B/32", device=self.device)

    def encode_image(self, image):
        """
        Encodes an image tensor using the CLIP image encoder.
        :param image: Preprocessed image tensor.
        :return: Encoded image features.
        """
        with torch.no_grad():
            image_features = self.model.encode_image(image.to(self.device))
        return image_features

    def encode_text(self, text):
        """
        Encodes a text tensor using the CLIP text encoder.
        :param text: Tokenized text tensor.
        :return: Encoded text features.
        """
        with torch.no_grad():
            text_features = self.model.encode_text(text.to(self.device))
        return text_features

    def forward(self, image, text):
        """
        Performs a forward pass computing similarity scores between image and text features.
        :param image: Preprocessed image tensor.
        :param text: Tokenized text tensor.
        :return: Similarity score matrix.
        """
        image_features = self.encode_image(image)
        text_features = self.encode_text(text)
        # Compute cosine similarity between image and text features
        similarity = (image_features @ text_features.T)
        return similarity

if __name__ == "__main__":
    # Basic test: instantiate the model and print initialization message.
    model = CLIPModel()
    print("CLIP model initialized on device:", model.device)
