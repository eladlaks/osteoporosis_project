import argparse
import wandb
import config
from train.train import run_training


def parse_args():
    parser = argparse.ArgumentParser(description="Train an image classification model")
    parser.add_argument(
        "--LEARNING_RATE",
        type=float,
        default=config.LEARNING_RATE,
        help="LEARNING RATE",
    )
    parser.add_argument(
        "--BATCH_SIZE", type=int, default=config.BATCH_SIZE, help="BATCH SIZE"
    )
    parser.add_argument(
        "--NUM_EPOCHS", type=int, default=config.NUM_EPOCHS, help="NUMBER OF EPOCHS"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    run_training(args)
