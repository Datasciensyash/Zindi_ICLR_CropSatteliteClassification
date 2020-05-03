import torch
from torch.nn.functional import one_hot

def multiclass_focal_loss(labels, logits, gamma=2.0, alpha=4., eps=1e-9):
    num_classes = logits.shape[1]
    model_out = logits + eps
    onehot_labels = one_hot(labels.long(), num_classes)
    ce = onehot_labels * (-1) * torch.log(model_out)
    weight = onehot_labels * (1 - model_out) ** gamma
    fl = alpha * weight * ce
    return fl