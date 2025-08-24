import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def compute_class_balanced_weights(samples_per_cls, beta=0.999, device=None):
    """Compute class-balanced weights once and return as tensor.

    Args:
        samples_per_cls: array of sample counts per class
        beta: hyperparameter for class balancing (default 0.999)
        device: torch device to place the weights on

    Returns:
        torch.Tensor: class weights shaped (1, num_classes, 1, 1) for broadcasting
    """
    effective_num = 1.0 - np.power(beta, samples_per_cls)
    weights = (1.0 - beta) / np.array(effective_num)
    weights = weights / np.sum(weights) * len(samples_per_cls)

    # Convert to tensor and reshape for broadcasting: (1, num_classes, 1, 1)
    class_weights = torch.FloatTensor(weights).view(1, -1, 1, 1)

    if device is not None:
        class_weights = class_weights.to(device)

    return class_weights


def _neg_loss_slow(preds, targets):
    pos_inds = targets == 1  # todo targets > 1-epsilon ?
    neg_inds = targets < 1  # todo targets < 1-epsilon ?

    neg_weights = torch.pow(1 - targets[neg_inds], 4)

    loss = 0
    for pred in preds:
        pred = torch.clamp(torch.sigmoid(pred), min=1e-4, max=1 - 1e-4)
        pos_pred = pred[pos_inds]
        neg_pred = pred[neg_inds]

        pos_loss = torch.log(pos_pred) * torch.pow(1 - pos_pred, 2)
        neg_loss = torch.log(1 - neg_pred) * torch.pow(neg_pred, 2) * neg_weights

        num_pos = pos_inds.float().sum()
        pos_loss = pos_loss.sum()
        neg_loss = neg_loss.sum()

        if pos_pred.nelement() == 0:
            loss = loss - neg_loss
        else:
            loss = loss - (pos_loss + neg_loss) / num_pos
    return loss


def _neg_loss(preds, targets, class_weights=None):
    """
    Modified focal loss with optional class balancing. Based on CornerNet focal loss.
    Arguments:
    preds ((multiple outputs) x B x c x h x w)
    targets (B x c x h x w)
    class_weights: pre-computed class-balanced weights tensor (1, num_classes, 1, 1) or None
    """
    pos_inds = targets.eq(1).float()
    neg_inds = targets.lt(1).float()

    neg_weights = torch.pow(1 - targets, 4)

    loss = 0
    final_loss = 0
    for pred in preds:
        pred = torch.clamp(torch.sigmoid(pred), min=1e-4, max=1 - 1e-4)

        # Standard focal loss components
        pos_loss = torch.log(pred) * torch.pow(1 - pred, 2) * pos_inds
        neg_loss = torch.log(1 - pred) * torch.pow(pred, 2) * neg_weights * neg_inds

        # Apply class balancing to positive loss if weights are provided
        if class_weights is not None:
            # Ensure weights are on the same device
            if class_weights.device != pred.device:
                class_weights = class_weights.to(pred.device)
            pos_loss = pos_loss * class_weights

        num_pos = pos_inds.float().sum()
        pos_loss = pos_loss.sum()
        neg_loss = neg_loss.sum()

        if num_pos == 0:
            loss += -neg_loss
            final_loss = -neg_loss
        else:
            loss += -(pos_loss + neg_loss) / num_pos
            final_loss = -(pos_loss + neg_loss) / num_pos

    return (loss / len(preds), final_loss)


def _reg_loss(regs, gt_regs, mask):
    mask = mask[:, :, None].expand_as(gt_regs).float()
    loss = sum(
        F.l1_loss(r * mask, gt_regs * mask, reduction="sum") / (mask.sum() + 1e-4)
        for r in regs
    )
    return loss / len(regs)
