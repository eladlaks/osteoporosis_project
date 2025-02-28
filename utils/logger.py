import wandb

def init_wandb(project_name="image_classification_project"):
    wandb.init(project=project_name)
