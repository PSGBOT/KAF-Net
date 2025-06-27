import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


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
        self, cos_similar=False, beta=0.999, reduction="mean", loss_weight=1.0
    ):
        super(RAFLoss, self).__init__()
        self.raf_type = "vector"  # 'vector' or 'point'
        self.cos_similar = cos_similar
        self.beta = beta
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.neg_loss_weight = 1
        self.loss_func = F.l1_loss
        # elif self.reg_type == "l2":
        #     self.loss_func = F.mse_loss
        # elif self.reg_type == "smooth_l1":
        #     self.loss_func = F.smooth_l1_loss
        # else:
        #     raise NotImplemented()

    def _cosine_similarity_loss(self, preds, targets, weights, num=None):
        weights = weights[:, :, 0, ...]
        # num = weights.gt(0).sum().clamp(min=1)
        # batched_num_valid = weights.gt(0).sum(dim=[1, 2, 3]).clamp(min=1)  # shape (b, )
        cosine_sim = F.cosine_similarity(preds, targets, dim=2)
        # no gt position will be zero
        loss = (1 - cosine_sim) * weights
        # loss = (loss.sum(dim=[1, 2, 3]) / batched_num_valid).sum()
        # valid = (weights > 0).any(dim=2)
        loss = loss.sum() / num
        # cosine_sim = F.cosine_similarity(preds, targets, dim=2)[valid]
        # loss = F.l1_loss(cosine_sim, torch.ones_like(cosine_sim), reduction='none')
        return loss

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

        loss = loss.sum() / num
        return loss

    def forward(self, preds, targets):
        gt_rafs = torch.stack([x["gt_relations"] for x in targets], dim=0)
        gt_raf_weights = torch.stack(
            [x["gt_relations_weights"] for x in targets], dim=0
        )
        total_loss = 0
        for pred in preds:
            # preds may be of (B, P*2, h, w)
            pred = pred.view_as(gt_rafs)
            if self.raf_type == "vector":
                loss = self._reg_loss(pred, gt_rafs, gt_raf_weights, num=None)
                loss = loss * self.loss_weight

                # spatial weights (class-agnostic) on gt paths
                # self.reg_area == "neg"
                spatial_mask = (
                    gt_raf_weights.max(dim=1, keepdim=True).values != gt_raf_weights
                )
                loss += (
                    F.l1_loss(pred, gt_rafs, reduction="none") * spatial_mask
                ).mean() * self.neg_loss_weight

            elif self.raf_type == "point":
                loss = gaussian_focal_loss(pred, gt_rafs)
            else:
                raise NotImplementedError()
            total_loss += loss
        return total_loss / len(preds)


def _raf_loss(rafs, gt_rafs, gt_raf_weights):
    """Raf loss. Exactly the same as fcsgg.
    Arguments:
      preds: (B x r*2 x h x w)
    Returns:
      raf_loss: a scalar tensor
    """
    raf_loss_evaluator = RAFLoss(
        cos_similar=False, beta=0.999, reduction="mean", loss_weight=1.0
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

    raf_loss = raf_loss_evaluator(rafs, gt)
    return raf_loss
