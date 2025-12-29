import torch
import torch.nn as nn


def Entropy(input_):
    """
    Compute entropy
    Args:
        input_: Input tensor
    Returns:
        entropy: Computed entropy
    """
    epsilon = 1e-5
    entropy = -input_ * torch.log(input_ + epsilon)
    entropy = torch.sum(entropy, dim=1)
    return entropy


def get_annealing_down_params(warm_up_factor, current_epoch, end_epoch):
    """
    Compute parameters that decay from warm_up_factor to 0
    Args:
        warm_up_factor: Initial factor
        current_epoch: Current training epoch
        end_epoch: Total training epochs
    Returns:
        param_factor: Decay factor for the current epoch
    """
    gamma = 10
    power = 0.75
    decay = (1 + gamma * current_epoch / end_epoch) ** (-power)
    param_factor = warm_up_factor * decay
    return param_factor


def Demix(feat1, feat2, lam, text_features=None, image_features=None, logit_scale=None):
    """
    Compute demixing loss for features
    Args:
        feat1: First feature vector (features processed by the fixed model)
        feat2: Second feature vector (features processed by the custom model)
        lam: Mixing parameter
        text_features: Text features, optional
        image_features: Image features, optional
        logit_scale: Scaling factor
    Returns:
        entropy_desc: Entropy loss
    """
    epsilon = 1e-3
    nume = feat1 - feat2 * lam
    denomi = 1 - lam
    if denomi < epsilon:
        denomi += epsilon
    if 1 - denomi < epsilon:
        denomi -= epsilon

    feat_desc = nume / denomi

    if text_features is None:
        output_desc = logit_scale * (image_features @ feat_desc.t())
    else:
        output_desc = logit_scale * (feat_desc @ text_features.t())

    softmax_desc = nn.Softmax(dim=1)(output_desc)
    entropy_desc = torch.mean(Entropy(softmax_desc))
    return entropy_desc