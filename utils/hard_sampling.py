import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import wandb

def get_low_confidence_samples(model, dataloader, threshold=0.75, device='cuda'):
    model.eval()
    low_conf_samples = []
    all_confidences = []

    with torch.no_grad():
        for images, labels, paths in dataloader:
            images = images.to(device)
            outputs = model(images)
            probs = F.softmax(outputs, dim=1)
            confidences, _ = torch.max(probs, dim=1)

            for path, conf in zip(paths, confidences):
                conf_value = conf.item()
                all_confidences.append(conf_value)
                if conf_value < threshold:
                    low_conf_samples.append(path)

    # Plot and log confidence histogram
    if len(all_confidences) > 0:
        plt.figure(figsize=(8, 5))
        plt.hist(all_confidences, bins=30, color='skyblue', edgecolor='black')
        plt.title("Confidence Distribution on Dataset")
        plt.xlabel("Max Softmax Probability")
        plt.ylabel("Number of Samples")
        plt.tight_layout()

        # Save to file and log to W&B
        hist_path = "confidence_distribution.png"
        plt.savefig(hist_path)
        wandb.log({"confidence_distribution": wandb.Image(hist_path)})
        plt.close()

    return low_conf_samples