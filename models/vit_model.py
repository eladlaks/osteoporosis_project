import timm
from config import NUM_CLASSES

def get_vit_model():
    # Create a Vision Transformer model (example: vit_base_patch16_224)
    model = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=NUM_CLASSES)
    return model
