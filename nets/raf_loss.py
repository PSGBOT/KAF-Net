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
            loss += (
                F.l1_loss(pred, gt_rafs, reduction="none") * spatial_mask
            ).mean() * self.neg_loss_weight

        elif self.raf_type == "point":
            loss = gaussian_focal_loss(pred, gt_rafs)
        else:
            raise NotImplementedError()
        total_loss += loss
        return total_loss


def visualize_raf_features(gt_rafs, gt_raf_weights):
    """
    Visualizes the Ground Truth Relation Affinity Features (RAF) and saves them as images.

    Args:
        gt_rafs (torch.Tensor): Ground truth relation affinity features (B, R*2, H, W).
        gt_raf_weights (torch.Tensor): Weights for the ground truth relation affinity features (B, R*2, H, W).
    """
    debug_viz_dir = "debug_viz"
    os.makedirs(debug_viz_dir, exist_ok=True)

    for i in range(len(gt_rafs)):
        # Get the ground truth relation features and weights for the current sample
        gt_raf_sample = gt_rafs[i]  # shape (R*2, H, W)
        gt_weight_sample = gt_raf_weights[i]  # shape (R*2, H, W)

        # Reshape to (R, 2, H, W) for easier access to individual relations
        R_actual = gt_raf_sample.shape[0] // 2
        H, W = gt_raf_sample.shape[1], gt_raf_sample.shape[2]

        gt_raf_sample_reshaped = gt_raf_sample.view(R_actual, 2, H, W)
        gt_weight_sample_reshaped = gt_weight_sample.view(R_actual, 2, H, W)

        sample_weight_images = []
        sample_magnitude_images = []
        sample_color_images = []
        sample_weight_direction_color_images = []

        for r_idx in range(R_actual):
            # Extract dx, dy components and their corresponding weights for the current relation
            dx = gt_raf_sample_reshaped[r_idx, 0, :, :].cpu().numpy()
            dy = gt_raf_sample_reshaped[r_idx, 1, :, :].cpu().numpy()
            weight_x = gt_weight_sample_reshaped[r_idx, 0, :, :].cpu().numpy()
            weight_y = gt_weight_sample_reshaped[r_idx, 1, :, :].cpu().numpy()

            # Calculate angle and magnitude
            angle = np.arctan2(dy, dx)  # Angle in radians (-pi to pi)
            magnitude = np.sqrt(dx**2 + dy**2)

            # Combined weights for saturation
            combined_weights = (weight_x + weight_y) / 2.0

            # Append grayscale images for combined visualization (for vertical concatenation later)
            combined_weights_normalized_gray = np.zeros_like(
                combined_weights, dtype=np.float32
            )
            cv2.normalize(
                combined_weights,
                combined_weights_normalized_gray,
                0,
                255,
                cv2.NORM_MINMAX,
            )
            sample_weight_images.append(
                combined_weights_normalized_gray.astype(np.uint8)
            )

            magnitude_normalized_gray = np.zeros_like(magnitude, dtype=np.float32)
            cv2.normalize(magnitude, magnitude_normalized_gray, 0, 255, cv2.NORM_MINMAX)
            sample_magnitude_images.append(magnitude_normalized_gray.astype(np.uint8))

            # --- HSV Visualization ---
            # Normalize angle to [0, 179] for Hue (OpenCV's H range for 8-bit images)
            # Add pi to angle to shift range to [0, 2*pi], then normalize to [0, 179]
            hue = ((angle + np.pi) / (2 * np.pi) * 179).astype(np.uint8)

            # Normalize combined_weights to [0, 255] for Saturation
            saturation_full = np.full_like(combined_weights, 255, dtype=np.uint8)

            # Normalize magnitude to [0, 255] for Value
            value = np.zeros_like(magnitude, dtype=np.float32)
            cv2.normalize(magnitude, value, 0, 255, cv2.NORM_MINMAX)
            value = value.astype(np.uint8)

            # Create HSV image
            hsv_image = cv2.merge([hue, saturation_full, value])

            # Convert HSV to BGR
            bgr_image = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR)
            sample_color_images.append(bgr_image)

            # --- Weight-focused HSV Visualization (Hue: angle, Saturation: 255, Value: normalized combined weights) ---
            # Hue is already calculated: hue
            # Saturation is fixed to 255 for full color
            saturation_full = np.full_like(combined_weights, 255, dtype=np.uint8)

            # Value is normalized combined_weights
            value_weights = np.zeros_like(combined_weights, dtype=np.float32)
            cv2.normalize(combined_weights, value_weights, 0, 255, cv2.NORM_MINMAX)
            value_weights = value_weights.astype(np.uint8)

            # Create HSV image for weight direction
            hsv_weight_direction_image = cv2.merge(
                [hue, saturation_full, value_weights]
            )

            # Convert HSV to BGR
            bgr_weight_direction_image = cv2.cvtColor(
                hsv_weight_direction_image, cv2.COLOR_HSV2BGR
            )
            sample_weight_direction_color_images.append(bgr_weight_direction_image)

        # Concatenate all relation images horizontally for the current sample
        if sample_weight_images and sample_magnitude_images:
            concatenated_weights = np.concatenate(sample_weight_images, axis=1)
            concatenated_magnitudes = np.concatenate(sample_magnitude_images, axis=1)
            combined_image = np.concatenate(
                (concatenated_weights, concatenated_magnitudes), axis=0
            )
            combined_filename = os.path.join(
                debug_viz_dir, f"sample_{i}_combined_raf_viz.png"
            )
            cv2.imwrite(combined_filename, combined_image)

        # Concatenate and save RAF color images
        if sample_color_images:
            concatenated_raf_color_images = np.concatenate(sample_color_images, axis=1)
            raf_color_filename = os.path.join(
                debug_viz_dir, f"sample_{i}_raf_color_viz.png"
            )
            cv2.imwrite(raf_color_filename, concatenated_raf_color_images)

        # Concatenate and save Weight-Direction color images
        if sample_weight_direction_color_images:
            concatenated_weight_direction_color_images = np.concatenate(
                sample_weight_direction_color_images, axis=1
            )
            weight_direction_color_filename = os.path.join(
                debug_viz_dir, f"sample_{i}_weight_direction_color_viz.png"
            )
            cv2.imwrite(
                weight_direction_color_filename,
                concatenated_weight_direction_color_images,
            )

        # Vertically concatenate the horizontally combined RAF color image and the new horizontally combined weight-direction color image
        if sample_color_images and sample_weight_direction_color_images:
            concatenated_raf_color_images = np.concatenate(sample_color_images, axis=1)
            concatenated_weight_direction_color_images = np.concatenate(
                sample_weight_direction_color_images, axis=1
            )
            final_combined_color_image = np.concatenate(
                (
                    concatenated_raf_color_images,
                    concatenated_weight_direction_color_images,
                ),
                axis=0,
            )
            final_combined_color_filename = os.path.join(
                debug_viz_dir, f"sample_{i}_final_combined_color_viz.png"
            )
            cv2.imwrite(final_combined_color_filename, final_combined_color_image)


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

    # Call the visualization function
    # visualize_raf_features(gt_rafs, gt_raf_weights)

    raf_loss = raf_loss_evaluator(rafs, gt)
    return raf_loss
