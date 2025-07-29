import torch
import torch.nn as nn

def _flatten(tensor):
    """Flattens a given tensor such that the channel axis is first.
    The shapes are transformed as follows:
       (N, C, D, H, W) -> (C, N * D * H * W)
    """
    # number of channels
    C = tensor.size(1)
    # new axis order
    axis_order = (1, 0) + tuple(range(2, tensor.dim()))
    # Transpose: (N, C, D, H, W) -> (C, N, D, H, W)
    transposed = tensor.permute(axis_order)
    # Flatten: (C, N, D, H, W) -> (C, N * D * H * W)
    return transposed.contiguous().view(C, -1)

def dice_score_function(model_output, tagret, epsilon=0.0001):
    """
    Computes the dice score between a model output and the target. Model output should be in [0, 1]
    """
    input = _flatten(model_output)
    target = _flatten(tagret)
    target = target.float()

    intersect = (input * target).sum(-1)
    # here we can use standard dice (input + target).sum(-1) or extension (see V-Net) (input^2 + target^2).sum(-1)
    denominator = (input * input).sum(-1) + (target * target).sum(-1) + epsilon
    dice_score = 2 * (intersect / denominator)
    return dice_score

def bce_loss_function(model_output, target, logits, reduction):
    input = _flatten(model_output)
    target = _flatten(target)
    if logits:
        return nn.BCEWithLogitsLoss(reduction=reduction)(input, target)
    else:
        return nn.BCELoss(reduction=reduction)(input, target)

def kld_loss_function(z_mean, z_sigma):
    """
    Computes the KL loss of a gaussian distribution defined by its mean and variance.
    """
    kld_loss = torch.mean(-0.5 * torch.sum(1 + z_sigma - z_mean ** 2 - z_sigma.exp(), dim = 1), dim = 0)
    return kld_loss
