import torch
import torch.nn.functional as F

def get_low_confidence_samples(model, dataloader, threshold=0.75, device='cuda'):
    model.eval()
    low_conf_samples = []

    with torch.no_grad():
        for images, labels, paths in dataloader:
            images = images.to(device)
            outputs = model(images)
            probs = F.softmax(outputs, dim=1)
            confidences, _ = torch.max(probs, dim=1)

            for path, conf in zip(paths, confidences):
                if conf.item() < threshold:
                    low_conf_samples.append(path)

    return low_conf_samples