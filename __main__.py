import argparse
import wandb
import config
from models.dino_model import get_dinov2_model
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
    parser.add_argument(
        "--USE_CLAHE", type=bool, default=config.USE_CLAHE, help="USE CLAHE"
    )
    parser.add_argument(
        "--MODEL_NAME", type=str, default=config.MODEL_NAME, help="MODEL NAME"
    )
    parser.add_argument(
        "--USE_OSTEOPENIA", type=bool, default=config.USE_OSTEOPENIA, help="USE OSTEOPENIA"
    )
    parser.add_argument(
        "--TRAIN_WEIGHTED_RANDOM_SAMPLER", type=int, default=config.TRAIN_WEIGHTED_RANDOM_SAMPLER, help="TRAIN_WEIGHTED_RANDOM_SAMPLER"
    )
    parser.add_argument(
        "--NUM_WORKERS", type=int, default=config.NUM_WORKERS, help="NUM_WORKERS"
    )
    parser.add_argument(
        "--DATA_DIR", type=str, default=config.DATA_DIR, help="Path to dataset directory"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    run_training(args)
