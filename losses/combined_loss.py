import torch
import torch.nn as nn
import torch.nn.functional as F

class CombinedLabelSmoothingConfidenceWeightedLoss(nn.Module):
    def __init__(self, epsilon=0.1, threshold=0.85, penalty_factor=2.0, reduction='mean'):
        super().__init__()
        self.epsilon = epsilon
        self.threshold = threshold
        self.penalty_factor = penalty_factor
        self.reduction = reduction

    def forward(self, outputs, targets):
        num_classes = outputs.size(1)
        log_probs = F.log_softmax(outputs, dim=1)
        probs = F.softmax(outputs, dim=1)
        confidences, predictions = torch.max(probs, dim=1)
        correct = predictions.eq(targets)

        targets_one_hot = F.one_hot(targets, num_classes).float()
        smoothed_targets = (1 - self.epsilon) * targets_one_hot + self.epsilon / num_classes
        loss_per_sample = -torch.sum(smoothed_targets * log_probs, dim=1)

        weights = torch.ones_like(loss_per_sample)
        high_conf_mistakes = (~correct) & (confidences > self.threshold)
        weights[high_conf_mistakes] = self.penalty_factor

        weighted_loss = loss_per_sample * weights

        if self.reduction == 'mean':
            return weighted_loss.mean()
        elif self.reduction == 'sum':
            return weighted_loss.sum()
        else:
            return weighted_loss