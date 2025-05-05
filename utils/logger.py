import wandb
import config


def init_wandb(project_name="osteoporosis_project", args={}):
    # Convert config.py module attributes to dictionary
    config_dict = {k: v for k, v in vars(config).items() if k.isupper()}

    # Override config parameters with any command-line arguments
    arg_dict = vars(args)
    arg_dict = {k.upper(): v for k, v in arg_dict.items()}

    config_dict.update(arg_dict)

    # Initialize wandb with the full configuration
    wandb.init(project="osteoporosis_project", reinit=True, config=config_dict)

    # Print out the final configuration to verify
    print("Wandb configuration:")
    for key, value in wandb.config.items():
        print(f"{key}: {value}")
