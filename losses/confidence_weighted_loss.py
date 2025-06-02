import torch
import torch.nn as nn
import torch.nn.functional as F

class ConfidenceWeightedCrossEntropy(nn.Module):
    def __init__(self, threshold=0.85, penalty_factor=2.0, reduction='mean'):
        super(ConfidenceWeightedCrossEntropy, self).__init__()
        self.threshold = threshold
        self.penalty_factor = penalty_factor
        self.reduction = reduction

    def forward(self, outputs, targets):
        probs = F.softmax(outputs, dim=1)
        confidences, predictions = torch.max(probs, dim=1)
        correct = predictions.eq(targets)

        base_loss = F.cross_entropy(outputs, targets, reduction='none')

        # זיהוי טעויות עם ביטחון גבוה
        high_conf_mistakes = (~correct) & (confidences > self.threshold)
        weights = torch.ones_like(base_loss)
        weights[high_conf_mistakes] = self.penalty_factor

        weighted_loss = base_loss * weights

        if self.reduction == 'mean':
            return weighted_loss.mean()
        elif self.reduction == 'sum':
            return weighted_loss.sum()
        else:
            return weighted_loss