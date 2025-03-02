import argparse
import wandb
import config
from train.train import run_training


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


if __name__ == "__main__":
    args = parse_args()

    run_training(args)
