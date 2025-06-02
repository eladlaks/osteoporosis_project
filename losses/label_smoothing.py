import torch
import torch.nn as nn
import torch.nn.functional as F

class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, epsilon=0.1, reduction='mean'):
        super(LabelSmoothingCrossEntropy, self).__init__()
        self.epsilon = epsilon
        self.reduction = reduction

    def forward(self, outputs, targets):
        num_classes = outputs.size(1)
        log_probs = F.log_softmax(outputs, dim=1)
        targets_one_hot = F.one_hot(targets, num_classes).float()
        smoothed_targets = (1 - self.epsilon) * targets_one_hot + self.epsilon / num_classes
        loss = -torch.sum(smoothed_targets * log_probs, dim=1)

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss