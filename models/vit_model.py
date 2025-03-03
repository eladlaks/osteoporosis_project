import timm
import wandb


def get_vit_model():
    # Create a Vision Transformer model (example: vit_base_patch16_224)
    model = timm.create_model(
        "vit_base_patch16_224", pretrained=True, num_classes=wandb.config.NUM_CLASSES
    )
    return model
