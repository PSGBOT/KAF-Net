import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import os


def gaussian_focal_loss(pred, gaussian_target, alpha=2.0, gamma=4.0):
    """`Focal Loss <https://arxiv.org/abs/1708.02002>`_ for targets in gaussian
    distribution.
    From mmdetection:
    hhttps://github.com/open-mmlab/mmdetection/blob/master/mmdet/models/losses/gaussian_focal_loss.py
    Args:
        pred (torch.Tensor): The prediction.
        gaussian_target (torch.Tensor): The learning target of the prediction
            in gaussian distribution.
        alpha (float, optional): A balanced form for Focal Loss.
            Defaults to 2.0.
        gamma (float, optional): The gamma for calculating the modulating
            factor. Defaults to 4.0.
    """
    eps = 1e-12
    pos_weights = gaussian_target.eq(1)
    neg_weights = (1 - gaussian_target).pow(gamma)
    pos_loss = -(pred + eps).log() * (1 - pred).pow(alpha) * pos_weights
    neg_loss = -(1 - pred + eps).log() * pred.pow(alpha) * neg_weights
    pos_num = pos_weights.sum().clamp(min=1)
    return (pos_loss + neg_loss).sum() / pos_num


def CB_loss_weights(samples_per_cls, beta):
    """Compute the Class Balanced Loss between `logits` and the ground truth `labels`.

    Class Balanced Loss: ((1-beta)/(1-beta^n))*Loss(labels, logits)
    where Loss is one of the standard losses used for Neural Networks.

    Args:
      labels: A int tensor of size [batch].
      logits: A float tensor of size [batch, no_of_classes].
      samples_per_cls: A python list of size [no_of_classes].
      no_of_classes: total number of classes. int
      loss_type: string. One of "sigmoid", "focal", "softmax".
      beta: float. Hyperparameter for Class balanced loss.
      gamma: float. Hyperparameter for Focal loss.

    Returns:
      cb_loss: A float tensor representing class balanced loss
    """
    effective_num = 1.0 - np.power(beta, samples_per_cls)
    no_of_classes = len(samples_per_cls)
    weights = (1.0 - beta) / np.array(effective_num)
    weights = weights / np.sum(weights) * no_of_classes

    return weights


class RAFLoss(nn.Module):
    def __init__(
        self,
        cos_similar=False,
        beta=0.999,
        reduction="mean",
        loss_weight=1.0,
        use_class_balanced=True,
        samples_per_cls=None,
    ):
        super(RAFLoss, self).__init__()
        self.raf_type = "vector"  # 'vector' or 'point'
        self.cos_similar = cos_similar
        self.beta = beta
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.neg_loss_weight = 0
        self.loss_func = F.l1_loss
        self.use_class_balanced = use_class_balanced
        # Compute class-balanced weights if enabled
        if self.use_class_balanced and samples_per_cls is not None:
            self.cb_weights = self._compute_cb_weights(samples_per_cls)
        else:
            self.cb_weights = None

    def _compute_cb_weights(self, samples_per_cls):
        """Compute class-balanced weights using effective number of samples"""
        effective_num = 1.0 - np.power(self.beta, samples_per_cls)
        weights = (1.0 - self.beta) / np.array(effective_num)
        weights = weights / np.sum(weights) * len(samples_per_cls)
        return torch.FloatTensor(weights)

    def _apply_class_weights(self, loss, gt_raf_weights, device):
        """Apply class-balanced weights to the loss"""
        if self.cb_weights is None:
            return loss

        # Move weights to device if needed
        if self.cb_weights.device != device:
            self.cb_weights = self.cb_weights.to(device)

        # gt_raf_weights shape: (B, R*2, H, W)
        # Reshape to (B, R, 2, H, W) to separate relations
        B, RC, H, W = gt_raf_weights.shape
        R = RC // 2
        weights_reshaped = gt_raf_weights.view(B, R, 2, H, W)

        # Create class weight map - broadcast cb_weights to match spatial dimensions
        # cb_weights shape: (R,) -> (1, R, 1, 1, 1) for broadcasting
        class_weight_map = self.cb_weights.view(1, R, 1, 1, 1).expand(B, R, 2, H, W)
        class_weight_map = class_weight_map.reshape(B, RC, H, W)

        # Apply class weights only where we have valid relations (weights > 0)
        valid_mask = (gt_raf_weights > 0).float()
        weighted_loss = loss * (class_weight_map * valid_mask + (1 - valid_mask))

        return weighted_loss

    def _reg_loss(self, preds, targets, weights, num=None, weighted=True):
        # in case we have no relation, the loss will still be zero since gt_raf_weights are all zero
        # num = weights.gt(0).sum().clamp(min=1)
        num = weights.eq(1).sum().clamp(min=1)
        if weighted:
            loss = self.loss_func(preds, targets, reduction="none") * weights
        else:
            loss = (
                self.loss_func(preds, targets, reduction="none") * (weights > 0).float()
            )
        if self.cos_similar:
            angle = torch.pow(-F.cosine_similarity(preds, targets, dim=2) + 2.0, 2)
            loss = loss * angle[:, :, None, ...]

        if self.use_class_balanced:
            loss = self._apply_class_weights(loss, weights, preds.device)
        loss = loss.sum() / num
        return loss

    def forward(self, preds, targets):
        gt_rafs = torch.stack([x["gt_relations"] for x in targets], dim=0)
        gt_raf_weights = torch.stack(
            [x["gt_relations_weights"] for x in targets], dim=0
        )
        total_loss = 0
        # Ensure the prediction tensor has the same shape as the ground truth for view_as
        pred = preds.view_as(gt_rafs)
        if self.raf_type == "vector":
            loss = self._reg_loss(pred, gt_rafs, gt_raf_weights, num=None)
            loss = loss * self.loss_weight

            # spatial weights (class-agnostic) on gt paths
            # self.reg_area == "neg"
            spatial_mask = (
                gt_raf_weights.max(dim=1, keepdim=True).values != gt_raf_weights
            )
            spatial_loss = F.l1_loss(pred, gt_rafs, reduction="none") * spatial_mask

            if self.use_class_balanced:
                spatial_loss = self._apply_class_weights(
                    spatial_loss, gt_raf_weights, pred.device
                )

            loss += spatial_loss.mean() * self.neg_loss_weight

        elif self.raf_type == "point":
            loss = gaussian_focal_loss(pred, gt_rafs)
        else:
            raise NotImplementedError()
        total_loss += loss
        return total_loss


def _kaf_loss(rafs, gt_rafs, gt_raf_weights, samples_per_cls=None):
    """Raf loss. Exactly the same as fcsgg.
    Arguments:
      preds: (B x r*2 x h x w)
    Returns:
      raf_loss: a scalar tensor
    """
    if samples_per_cls is None:
        use_cb = False
    else:
        use_cb = True
    raf_loss_evaluator = RAFLoss(
        cos_similar=False,
        beta=0.999,
        reduction="mean",
        loss_weight=1.0,
        use_class_balanced=use_cb,
        samples_per_cls=samples_per_cls,
    )

    # gt_rafs:[B, r, 2, 128,128] => [B, r*2, 128,128]
    # Reshape to combine the predicate and vector component dimensions
    B, R, C, H, W = gt_rafs.shape
    gt_rafs = gt_rafs.reshape(B, R * C, H, W)

    # gt_raf_weights:[B, r, 2, 128,128] => [B, r*2, 128,128]
    # Reshape to combine the predicate and vector component dimensions
    gt_raf_weights = gt_raf_weights.reshape(B, R * C, H, W)

    """
    gt_rafs: a List[Dict] with B dicts:
    {
        gt_relations: tensor (r*2 x h x w)
        gt_relations_weights: tensor (r*2 x h x w)
    }
    """
    gt = [
        {"gt_relations": gt_raf, "gt_relations_weights": gt_weight}
        for gt_raf, gt_weight in zip(gt_rafs, gt_raf_weights)
    ]

    # Call the visualization function
    # visualize_raf_features(gt_rafs, gt_raf_weights)

    raf_loss = raf_loss_evaluator(rafs, gt)
    return raf_loss
