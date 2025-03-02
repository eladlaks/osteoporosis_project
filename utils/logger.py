import argparse
import wandb
import config


def parse_args():
    parser = argparse.ArgumentParser(description="Train an image classification model")
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=config.LEARNING_RATE,
        help="Learning rate",
    )
    parser.add_argument(
        "--batch_size", type=int, default=config.BATCH_SIZE, help="Batch size"
    )
    parser.add_argument(
        "--num_epochs", type=int, default=config.NUM_EPOCHS, help="Number of epochs"
    )
    return parser.parse_args()


def init_wandb(project_name="image_classification_project", args={}):
    # Convert config.py module attributes to dictionary
    config_dict = {k: v for k, v in vars(config).items() if k.isupper()}

    # Override config parameters with any command-line arguments
    arg_dict = vars(args)
    config_dict.update(arg_dict)

    # Initialize wandb with the full configuration
    wandb.init(project="image_classification_project", reinit=True, config=config_dict)

    # Print out the final configuration to verify
    print("Wandb configuration:")
    for key, value in wandb.config.items():
        print(f"{key}: {value}")
