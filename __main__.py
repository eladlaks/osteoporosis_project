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
        "--USE_OSTEOPENIA",
        type=bool,
        default=config.USE_OSTEOPENIA,
        help="USE OSTEOPENIA",
    )
    parser.add_argument(
        "--USE_METABOLIC_FOR_TEST",
        type=bool,
        default=config.USE_METABOLIC_FOR_TEST,
        help="USE_METABOLIC_FOR_TEST",
    )
    parser.add_argument(
        "--USE_SCHEDULER",
        type=bool,
        default=config.USE_SCHEDULER,
        help="USE_SCHEDULER",
    )
    parser.add_argument(
        "--TRAIN_WEIGHTED_RANDOM_SAMPLER",
        type=int,
        default=config.TRAIN_WEIGHTED_RANDOM_SAMPLER,
        help="TRAIN_WEIGHTED_RANDOM_SAMPLER",
    )
    parser.add_argument(
        "--NUM_WORKERS", type=int, default=config.NUM_WORKERS, help="NUM_WORKERS"
    )
    parser.add_argument(
        "--DATA_DIR",
        type=str,
        default=config.DATA_DIR,
        help="Path to dataset directory",
    )
    parser.add_argument(
        "--TEST_DATA_DIR",
        type=str,
        default=config.TEST_DATA_DIR,
        help="Path to test_data directory",
    )
    parser.add_argument(
        "--USE_LABEL_SMOOTHING",
        type=bool,
        default=config.USE_LABEL_SMOOTHING,
        help="Use Label Smoothing"
    )
    parser.add_argument(
        "--USE_HARD_SAMPLING",
        type=bool,
        default=config.USE_HARD_SAMPLING,
        help="Use confidence-based hard sampling"
    )
    parser.add_argument(
        "--USE_CONFIDENCE_WEIGHTED_LOSS",
        type=bool,
        default=config.USE_CONFIDENCE_WEIGHTED_LOSS,
        help="Use confidence-weighted loss"
    )
    parser.add_argument(
        "--CONFIDENCE_PENALTY_WEIGHT",
        type=float,
        default=config.CONFIDENCE_PENALTY_WEIGHT,
        help="Penalty weight for high-confidence mistakes",
    )
    parser.add_argument(
        "--CONFIDENCE_THRESHOLD",
        type=float,
        default=config.CONFIDENCE_THRESHOLD,
        help="Threshold under which predictions are considered low-confidence",
    )
    parser.add_argument(
        "--LABEL_SMOOTHING_EPSILON",
        type=float,
        default=config.LABEL_SMOOTHING_EPSILON,
        help="Epsilon value for label smoothing",
    )
    parser.add_argument(
        "--USE_TRANSFORM_AUGMENTATION_IN_TRAINING",
        type=bool,
        default=config.USE_TRANSFORM_AUGMENTATION_IN_TRAINING,
        help="Use data augmentation during training",
    )
    return parser, parser.parse_args()


if __name__ == "__main__":
    parser, args = parse_args()

    # Override args with wandb.config if running from a sweep
    if wandb.run is not None:
        for action in parser._actions:
            key = action.dest
            if key in wandb.config:
                value = wandb.config[key]
                if action.type:
                    value = action.type(value)
                setattr(args, key, value)

    run_training(args)