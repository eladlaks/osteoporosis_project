import wandb
import config
import random
import string


def init_wandb(project_name="final_project", args={}):
    # Convert config.py module attributes to dictionary
    config_dict = {k: v for k, v in vars(config).items() if k.isupper()}

    # Override config parameters with any command-line arguments
    arg_dict = vars(args)
    arg_dict = {k.upper(): v for k, v in arg_dict.items()}

    config_dict.update(arg_dict)

    # Create run name with learning rate and batch size
    lr = config_dict.get("LEARNING_RATE", config_dict.get("LR", "unknown"))
    batch_size = config_dict.get("BATCH_SIZE", "unknown")
    random_suffix = "".join(random.choices(string.ascii_lowercase + string.digits, k=6))
    run_name = f"lr_{lr}_bs_{batch_size}_{random_suffix}"

    # Initialize wandb with the full configuration and run name
    wandb.init(project=project_name, name=run_name, reinit=True, config=config_dict)

    # Print out the final configuration to verify
    print("Wandb configuration:")
    for key, value in wandb.config.items():
        print(f"{key}: {value}")
    print(f"Run name: {run_name}")
