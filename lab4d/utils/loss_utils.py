# Copyright (c) 2025 Jeff Tan, Gengshan Yang, Carnegie Mellon University.
import torch
import numpy as np
import torch.nn.functional as F


def get_mask_balance_wt(mask, vis2d, is_detected):
    """Balance contribution of positive and negative pixels in mask.

    Args:
        mask: (M,N,1) Object segmentation mask
        vis2d: (M,N,1) Whether each pixel is visible in the video frame
        is_detected: (M,) Whether there is segmentation mask in the frame
    Returns:
        mask_balance_wt: (M,N,1) Balanced mask
    """
    # all the positive labels
    mask = mask.float()
    # all the labels
    vis2d = vis2d.float() * is_detected.float()[..., None, None]
    if mask.sum() > 0 and (1 - mask).sum() > 0:
        pos_wt = vis2d.sum() / mask[vis2d > 0].sum()
        neg_wt = vis2d.sum() / (1 - mask[vis2d > 0]).sum()
        mask_balance_wt = 0.5 * pos_wt * mask + 0.5 * neg_wt * (1 - mask)
    else:
        mask_balance_wt = 1
    return mask_balance_wt


def entropy_loss(prob, dim=-1):
    """Compute entropy of a probability distribution
    In the case of skinning weights, each column is a distribution over assignment to B bones.
    We want to encourage low entropy, i.e. each point is assigned to fewer bones.

    Args:
        prob: (..., B) Probability distribution
    Returns:
        entropy (...,) Entropy of each distribution
    """
    entropy = -(prob * (prob + 1e-9).log()).sum(dim)
    return entropy


def cross_entropy_skin_loss(skin):
    """Compute entropy of a probability distribution
    In the case of skinning weights, each column is a distribution over assignment to B bones.
    We want to encourage low entropy, i.e. each point is assigned to fewer bones.

    Args:
        skin: (..., B) un-normalized skinning weights
    """
    shape = skin.shape
    nbones = shape[-1]
    full_skin = skin.clone()

    # find the most likely bone assignment
    score, indices = skin.max(-1, keepdim=True)
    skin = torch.zeros_like(skin).fill_(0)
    skin = skin.scatter(-1, indices, torch.ones_like(score))

    cross_entropy = F.cross_entropy(
        full_skin.view(-1, nbones), skin.view(-1, nbones), reduction="none"
    )
    cross_entropy = cross_entropy.view(shape[:-1])
    return cross_entropy


def align_tensors(v1, v2, dim=None):
    """Return the scale that best aligns v1 to v2 in the L2 sense:
    min || kv1-v2 ||^2

    Args:
        v1: (...,) Source vector
        v2: (...,) Target vector
        dim: Dimension to align. If None, return a scalar
    Returns:
        scale_fac (1,): Scale factor
    """
    if dim is None:
        scale = (v1 * v2).sum() / (v1 * v1).sum()
        if scale < 0:
            scale = torch.tensor([1.0], device=scale.device)
        return scale
    else:
        scale = (v1 * v2).sum(dim, keepdim=True) / (v1 * v1).sum(dim, keepdim=True)
        scale[scale < 0] = 1.0
        return scale
