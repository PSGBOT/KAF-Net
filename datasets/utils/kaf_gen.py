import logging
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import torch
import torch.nn.functional as F
from fvcore.common.file_io import PathManager
from PIL import Image
import math


@torch.no_grad()
def get_oval_gaussian_radius(wh_tensor, min_overlap=0.7):
    """
    Return the two axis radius of the gaussian based on IOU min_overlap.
    Note this returns long tensors
    """
    factor = (1 - np.sqrt(min_overlap)) / np.sqrt(2)  # > 0
    radius_a_b = wh_tensor * factor + 1
    return radius_a_b.long()


@torch.no_grad()
def gaussian2D(diameters, sigma_factor=6):
    num_instances = diameters.size(0)
    sigmas_x_y = diameters.float() / sigma_factor
    starts, ends = -diameters // 2, (diameters + 1) // 2
    guassian_masks = []
    # different gauss kernels have different range, had to use loop
    for i in range(num_instances):
        y, x = torch.meshgrid(
            torch.arange(starts[i][1], ends[i][1]),
            torch.arange(starts[i][0], ends[i][0]),
        )
        x = x.to(diameters.device)
        y = y.to(diameters.device)
        # range (0, 1]
        guassian_masks.append(
            torch.exp(
                -(
                    x**2 / (2 * sigmas_x_y[i, 0] ** 2)
                    + y**2 / (2 * sigmas_x_y[i, 1] ** 2)
                )
            )
        )
    return guassian_masks


@torch.no_grad()
def get_kaf(
    gt_relations,
    gt_boxes,
    num_predicates,
    output_stride,
    output_size,
    sigma=0,
    range_wh=None,
    min_overlap=0.5,
    weight_by_length=False,
):
    """
    kr structure: [
    ["mask0", "mask1", relation_idx],
    ["mask0", "mask2", relation_idx],
    ...
    ]

    masks_bbox structure:
    {
    "mask0": {
        "center": [x, y],
        "scale": [w, h]
    },
    ...
    }
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # convert gt_relation to tensor
    gt_relations = torch.tensor(gt_relations, device=device)

    # (P, 2, h, w)
    h, w = output_size
    rafs = torch.zeros(
        (
            num_predicates,
            2,
        )
        + output_size,
        device=device,
    )
    # weight each relation equally or not, if no weight, then long and big relation field
    # will dominant the loss
    rafs_weights = torch.zeros(
        (
            num_predicates,
            2,
        )
        + output_size,
        device=device,
    )

    # if no relations
    num_rels = gt_relations.size(0)
    if num_rels == 0:
        return rafs, rafs_weights

    # gt_centers = gt_boxes.get_centers()  # in image scale
    gt_centers = np.zeros((len(gt_boxes), 2))
    gt_wh = np.zeros((len(gt_boxes), 2))
    for mask_idx in gt_boxes.keys():
        gt_centers[mask_idx] = gt_boxes[mask_idx]["center"]
        gt_wh[mask_idx] = gt_boxes[mask_idx]["scale"]
    gt_centers = torch.tensor(gt_centers, device=device)
    gt_wh = torch.tensor(gt_wh, device=device)
    # shape of (m, 2)
    # !important: objects can collapse into the same center
    leaf_centers = gt_centers[gt_relations[:, 0]]
    root_centers = gt_centers[gt_relations[:, 1]]
    mid_centers = (leaf_centers + root_centers) / 2
    true_m2o_vectors = mid_centers - leaf_centers
    true_m2s_vectors = mid_centers - root_centers
    # if true vector collapse, we define vector [1e-6, 1e-6] for it
    true_m2o_vectors[true_m2o_vectors.eq(0).all(dim=1)] += 1e-6
    true_m2o_vectors_norms = torch.norm(true_m2o_vectors, dim=1, keepdim=False)
    true_m2s_vectors[true_m2s_vectors.eq(0).all(dim=1)] += 1e-6
    true_m2s_vectors_norms = torch.norm(true_m2s_vectors, dim=1, keepdim=False)
    # vector in feature

    leaf_centers = leaf_centers // output_stride
    root_centers = root_centers // output_stride
    mid_centers = mid_centers // output_stride
    m2o_vectors = mid_centers - leaf_centers
    m2s_vectors = mid_centers - root_centers
    # if two centers are collapsed, we use true vector
    zero_vec_mask = m2o_vectors.eq(0).all(dim=1)
    m2o_vectors[zero_vec_mask] = true_m2o_vectors[zero_vec_mask]
    zero_vec_mask = m2s_vectors.eq(0).all(dim=1)
    m2s_vectors[zero_vec_mask] = true_m2s_vectors[zero_vec_mask]
    # right now it is a (1, 0) vector, not a random unit vector
    # rand_vec = torch.rand(2)
    # rand_vec /= rand_vec.norm()
    # s2o_vectors[zero_vec_mask] = torch.tensor([1.,0.], device=device)
    # shape (m, 1)
    m2o_vector_norms = torch.norm(m2o_vectors, dim=1, keepdim=False)
    m2s_vector_norms = torch.norm(m2s_vectors, dim=1, keepdim=False)

    if range_wh is not None:
        # check the vector norm, if it is in the range
        # norm_range = torch.norm(range_wh, dim=0)
        # valid_rel_mask = torch.logical_and(
        #     true_s2o_vectors_norms > norm_range[0],
        #     true_s2o_vectors_norms <= norm_range[1],
        # )

        valid_rel_mask = torch.logical_and(
            true_m2o_vectors_norms > range_wh[0],
            true_m2o_vectors_norms <= range_wh[1],
        )
        # or either axis-aligned side
        # s2o_vectors_abs = torch.abs(s2o_vectors)
        # valid_rel_mask = torch.logical_and(s2o_vectors_abs > range_wh[..., 0],
        #                                    s2o_vectors_abs <= range_wh[..., 1]).any(dim=-1)
        gt_relations = gt_relations[valid_rel_mask]
        m2o_vectors = m2o_vectors[valid_rel_mask]
        m2s_vectors = m2s_vectors[valid_rel_mask]
        m2o_vector_norms = m2o_vector_norms[valid_rel_mask][..., None]
        m2s_vector_norms = m2s_vector_norms[valid_rel_mask][..., None]
        leaf_centers = leaf_centers[valid_rel_mask]
        root_centers = root_centers[valid_rel_mask]
        mid_centers = mid_centers[valid_rel_mask]
        num_rels = gt_relations.size(0)
        # for some scales, there could be no relations
        if num_rels == 0:
            return rafs, rafs_weights
    else:
        m2o_vector_norms = m2o_vector_norms[..., None]
        m2s_vector_norms = m2s_vector_norms[..., None]

    # count the number of raf overlap at each pixel location
    cross_raf_counts = torch.zeros((num_predicates,) + output_size, device=device)

    if sigma == 0:
        radius_a_b = get_oval_gaussian_radius(
            gt_wh // output_stride, min_overlap=min_overlap
        )
        # raf width dependent on the radius r_s, r_o
        subject_radius = radius_a_b[gt_relations[:, 0]]
        object_radius = radius_a_b[gt_relations[:, 1]]
        sigma = (
            torch.cat((subject_radius, object_radius), dim=1)
            .min(dim=1, keepdim=True)[0]
            .unsqueeze(-1)
        )
    else:
        # sigma = torch.ones((num_rels, 1, 1), device=device) * sigma
        sigma = torch.ones((num_rels, 1, 1), device=device) * np.sqrt(
            128 / output_stride
        )
    relation_unit_vecs_m2s = m2s_vectors / m2s_vector_norms  # List[shape (m, 2)]
    relation_unit_vecs_m2o = m2o_vectors / m2o_vector_norms  # List[shape (m, 2)]
    # for assign weights, longer relation has smaller weights
    relation_distance_weights = F.normalize(
        1 / (m2o_vector_norms + m2s_vector_norms), p=2, dim=0
    )
    # shape (m, w, h)
    m, y, x = torch.meshgrid(
        torch.arange(num_rels, device=device),
        torch.arange(h, device=device),
        torch.arange(w, device=device),
    )
    # equation: v = (v_x, v_y) unit vector along s->o, a center point c = (c_x, c_y), any point p = (x, y)
    # <v, p - c> = v_x * (x - c_x) + v_y * (y - c_y) gives distance along the relation vector
    # <v⊥, p - c> gives distance orthogonal to the relation vector
    dist_along_rel_m2s = torch.abs(
        relation_unit_vecs_m2s[:, 0:1, None]
        * (x - (mid_centers[:, 0:1, None] + root_centers[:, 0:1, None]) / 2)
        + relation_unit_vecs_m2s[:, 1:2, None]
        * (y - (mid_centers[:, 1:2, None] + root_centers[:, 1:2, None]) / 2)
    )
    dist_along_rel_m2o = torch.abs(
        relation_unit_vecs_m2o[:, 0:1, None]
        * (x - (mid_centers[:, 0:1, None] + leaf_centers[:, 0:1, None]) / 2)
        + relation_unit_vecs_m2o[:, 1:2, None]
        * (y - (mid_centers[:, 1:2, None] + leaf_centers[:, 1:2, None]) / 2)
    )
    dist_ortho_rel = torch.abs(
        relation_unit_vecs_m2o[:, 1:2, None] * (x - leaf_centers[:, 0:1, None])
        - relation_unit_vecs_m2o[:, 0:1, None] * (y - leaf_centers[:, 1:2, None])
    )
    # valid = (dist_along_rel >= 0) \
    #         * (dist_along_rel <= s2o_vector_norms[..., None]) \
    #         * (dist_ortho_rel <= sigma)
    # print(dist_along_rel_m2s.shape, m2s_vector_norms.shape)
    valid_m2s = (dist_along_rel_m2s <= m2s_vector_norms[..., None] // 2) * (
        dist_ortho_rel <= sigma
    )
    valid_m2o = (dist_along_rel_m2o <= m2o_vector_norms[..., None] // 2) * (
        dist_ortho_rel <= sigma
    )
    # (m, w, h) <-- (m, 2) (m, w, h)
    rafs_x = (
        relation_unit_vecs_m2s[:, 0:1, None] * valid_m2s
        + relation_unit_vecs_m2o[:, 0:1, None] * valid_m2o
    )
    rafs_y = (
        relation_unit_vecs_m2s[:, 1:2, None] * valid_m2s
        + relation_unit_vecs_m2o[:, 1:2, None] * valid_m2o
    )
    valid = torch.max(valid_m2s, valid_m2o).float()  # for computing the weights
    rafs_weights_ortho_rel = (
        torch.min(
            torch.max(
                torch.exp(
                    -torch.clamp(
                        dist_along_rel_m2o - m2o_vector_norms[..., None] / 2, min=0
                    ).round()
                    / 1
                ),
                torch.exp(
                    -torch.clamp(
                        dist_along_rel_m2s - m2s_vector_norms[..., None] / 2, min=0
                    ).round()
                    / 1
                ),
            ),
            torch.exp(-torch.round(dist_ortho_rel) / 1),
        )
        * valid
    )  # [0, 1]
    # rafs_weights = rafs_weights_ortho_rel.max(dim=0).values
    # gather by predicate class (not sure it is fast enough) to shape (50, 2, h, w)
    gt_predicates = gt_relations[:, 2]
    for i, gt_predicate in enumerate(gt_predicates):
        cross_raf_counts[gt_predicate, ...] += torch.logical_or(rafs_x[i], rafs_y[i])
        rafs[gt_predicate, 0] += rafs_x[i]
        rafs[gt_predicate, 1] += rafs_y[i]
        # OPTIONAL: if overlapped, the weight is higher too; normalize by relation area/length
        # rafs_weights_ortho_rel[i].eq(1).sum()
        if weight_by_length:
            rafs_weights[gt_predicate] = torch.max(
                rafs_weights[gt_predicate],
                rafs_weights_ortho_rel[i] * relation_distance_weights[i],
            )
        else:
            rafs_weights[gt_predicate] = torch.max(
                rafs_weights[gt_predicate], rafs_weights_ortho_rel[i]
            )
    # divide by number of intersections
    rafs = torch.where(
        cross_raf_counts[:, None, ...] > 1, rafs / cross_raf_counts[:, None, ...], rafs
    )
    # for weights, if there is one object with many relations, then its weight should be low.
    # Why? At the object location, the raf will have many vectors superposition, it is impossible to
    # predict such vector
    counts_sum = cross_raf_counts[:, None, :, :]
    rafs_weights = torch.where(counts_sum > 1, rafs_weights / counts_sum, rafs_weights)

    # [14, 2, 128, 128], [14, 2, 128, 128]
    return rafs, rafs_weights


# =============================================================================
# CLASSIFICATION BASELINE: Gaussian heatmap at relation midpoints
# =============================================================================

def _gaussian_radius(det_size, min_overlap=0.7):
    """Compute Gaussian radius given (height, width) of a detection box."""
    height, width = det_size
    a1 = 1
    b1 = (height + width)
    c1 = width * height * (1 - min_overlap) / (1 + min_overlap)
    sq1 = np.sqrt(b1 ** 2 - 4 * a1 * c1)
    r1 = (b1 + sq1) / 2

    a2 = 4
    b2 = 2 * (height + width)
    c2 = (1 - min_overlap) * width * height
    sq2 = np.sqrt(b2 ** 2 - 4 * a2 * c2)
    r2 = (b2 + sq2) / 2

    a3 = 4 * min_overlap
    b3 = -2 * min_overlap * (height + width)
    c3 = (min_overlap - 1) * width * height
    sq3 = np.sqrt(b3 ** 2 - 4 * a3 * c3)
    r3 = (b3 + sq3) / 2
    return min(r1, r2, r3)


def _draw_umich_gaussian(heatmap, center_int, radius, k=1):
    """
    Draw a 2D Gaussian on a numpy heatmap at center_int with given radius.
    Same logic as utils.image.draw_umich_gaussian.
    """
    diameter = 2 * radius + 1
    sigma = diameter / 6.0
    # generate gaussian kernel
    x_range = np.arange(0, diameter, 1, np.float32)
    y_range = x_range[:, np.newaxis]
    x0 = y0 = diameter // 2
    gaussian = np.exp(-((x_range - x0) ** 2 + (y_range - y0) ** 2) / (2 * sigma ** 2))

    height, width = heatmap.shape
    x, y = int(center_int[0]), int(center_int[1])

    left, right = min(x, radius), min(width - x, radius + 1)
    top, bottom = min(y, radius), min(height - y, radius + 1)

    masked_heatmap = heatmap[y - top:y + bottom, x - left:x + right]
    masked_gaussian = gaussian[radius - top:radius + bottom, radius - left:radius + right]

    if min(masked_gaussian.shape) > 0 and min(masked_heatmap.shape) > 0:
        np.maximum(masked_heatmap, masked_gaussian * k, out=masked_heatmap)
    return heatmap


@torch.no_grad()
def get_rel_cls_gt(
    gt_relations,
    gt_boxes,
    num_predicates,
    output_stride,
    output_size,
    range_wh=None,
    gaussian_iou=0.7,
):
    """
    Classification baseline ground truth: Gaussian heatmaps at relation midpoints.

    For each relation (part_i, part_j, predicate_type), we compute the midpoint
    between part_i and part_j centers, then draw a Gaussian peak at that midpoint
    on the heatmap channel corresponding to predicate_type.

    Uses the SAME FPN-level assignment logic as get_kaf():
    the distance between two part centers determines which FPN level this
    relation is assigned to, controlled by range_wh.

    Args:
        gt_relations: list of [subject_idx, object_idx, relation_idx]
        gt_boxes: dict {mask_idx: {"center": (x,y), "scale": [w,h]}}
        num_predicates: number of relation types (14)
        output_stride: downsampling stride for this FPN level
        output_size: (H, W) of the feature map
        range_wh: [min_dist, max_dist] for FPN-level filtering (same as KAF)
        gaussian_iou: IoU threshold for Gaussian radius computation

    Returns:
        rel_hmap: numpy array (P, H, W) — Gaussian heatmap per predicate
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    h, w = output_size
    # Output is a numpy array (P, H, W), matching what psr.py expects for hmap
    rel_hmap = np.zeros((num_predicates, h, w), dtype=np.float32)

    if len(gt_relations) == 0:
        return rel_hmap

    gt_relations_t = torch.tensor(gt_relations, device=device)

    # Extract centers and sizes
    gt_centers = np.zeros((len(gt_boxes), 2))
    gt_wh_np = np.zeros((len(gt_boxes), 2))
    for mask_idx in gt_boxes.keys():
        gt_centers[mask_idx] = gt_boxes[mask_idx]["center"]
        gt_wh_np[mask_idx] = gt_boxes[mask_idx]["scale"]

    gt_centers_t = torch.tensor(gt_centers, device=device)
    gt_wh_t = torch.tensor(gt_wh_np, device=device)

    # Compute midpoints and distances (same as KAF)
    leaf_centers = gt_centers_t[gt_relations_t[:, 0]]
    root_centers = gt_centers_t[gt_relations_t[:, 1]]
    mid_centers = (leaf_centers + root_centers) / 2.0

    # Distance for FPN assignment (same as KAF: uses true_m2o_vectors_norms)
    true_m2o_vectors = mid_centers - leaf_centers
    true_m2o_vectors[true_m2o_vectors.eq(0).all(dim=1)] += 1e-6
    true_m2o_vectors_norms = torch.norm(true_m2o_vectors, dim=1, keepdim=False)

    # FPN-level filtering (exact same logic as KAF)
    if range_wh is not None:
        valid_rel_mask = torch.logical_and(
            true_m2o_vectors_norms > range_wh[0],
            true_m2o_vectors_norms <= range_wh[1],
        )
        gt_relations_t = gt_relations_t[valid_rel_mask]
        mid_centers = mid_centers[valid_rel_mask]
        leaf_centers = leaf_centers[valid_rel_mask]
        root_centers = root_centers[valid_rel_mask]

    num_rels = gt_relations_t.size(0)
    if num_rels == 0:
        return rel_hmap

    # For each relation, draw a Gaussian at the midpoint
    for i in range(num_rels):
        predicate = int(gt_relations_t[i, 2].item())
        sub_idx = int(gt_relations_t[i, 0].item())
        obj_idx = int(gt_relations_t[i, 1].item())

        # Midpoint in feature map coordinates
        mid_x = float(mid_centers[i, 0].item()) / output_stride
        mid_y = float(mid_centers[i, 1].item()) / output_stride
        mid_int = np.array([int(mid_x), int(mid_y)], dtype=np.int32)

        # Clamp to valid range
        mid_int[0] = np.clip(mid_int[0], 0, w - 1)
        mid_int[1] = np.clip(mid_int[1], 0, h - 1)

        # Compute Gaussian radius based on the smaller of the two part bboxes
        # (same philosophy as KAF's sigma computation)
        sub_w, sub_h = gt_wh_np[sub_idx]
        obj_w, obj_h = gt_wh_np[obj_idx]
        # Use the smaller bbox area to determine radius
        sub_area = (sub_w / output_stride) * (sub_h / output_stride)
        obj_area = (obj_w / output_stride) * (obj_h / output_stride)
        if min(sub_area, obj_area) < 1:
            radius = 0
        else:
            # Use the smaller part's dimensions for Gaussian radius
            if sub_area < obj_area:
                det_h = math.ceil(sub_h / output_stride)
                det_w = math.ceil(sub_w / output_stride)
            else:
                det_h = math.ceil(obj_h / output_stride)
                det_w = math.ceil(obj_w / output_stride)
            radius = max(
                0,
                int(_gaussian_radius((det_h, det_w), gaussian_iou))
            )

        _draw_umich_gaussian(rel_hmap[predicate], mid_int, radius)

    return rel_hmap
