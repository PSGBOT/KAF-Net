import torch
import torch.nn.functional as F
import numpy as np
from scipy.ndimage import maximum_filter

from utils.post_process import ctdet_decode
import math
from datasets.psr import PSR_FUNC_CAT, PSR_KR_CAT
import cv2
import matplotlib.pyplot as plt


@torch.no_grad()
def get_kaf_path(objects, sigma_factor=2.0, stride=4):
    """
    Generate inference weights for KAF relation extraction.

    Args:
        objects: list of detected objects with positions
        sigma_factor: controls the width of integration region

    Returns:
        relation_weights: tensor [num_objects, num_objects, H, W]
                         representing integration weights for each object pair
    """
    if len(objects) == 0:
        return torch.empty(0, 0, 0, 0)

    # Get device from first object's position (assuming they're tensors)
    # If objects store positions as lists, we'll use CPU and move to GPU later
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Extract object centers
    centers = torch.tensor(
        [[obj["yx"][0] / stride, obj["yx"][1] / stride] for obj in objects],
        device=device,
        dtype=torch.float32,
    )  # [N, 2] (y, x)

    # Get typical feature map size from the calling context
    # We'll assume a reasonable default size and let it be overridden if needed
    H, W = (
        int(512 / stride),
        int(512 / stride),
    )  # Default size, should match the KAF feature map

    num_objects = len(objects)

    # Create coordinate grids
    y_coords = (
        torch.arange(H, device=device, dtype=torch.float32).view(H, 1).expand(H, W)
    )
    x_coords = (
        torch.arange(W, device=device, dtype=torch.float32).view(1, W).expand(H, W)
    )
    coords = torch.stack([y_coords, x_coords], dim=0)  # [2, H, W]

    # Initialize weights tensor
    relation_weights = torch.zeros(num_objects, num_objects, H, W, device=device)

    # Compute integration widths for each object
    widths = torch.tensor(
        [min(obj["width"], obj["height"]) / 4 for obj in objects], device=device
    )
    widths = torch.clamp(widths, min=1.0)

    # For each object pair
    for i in range(num_objects):
        for j in range(num_objects):
            if i == j:
                continue

            # Get line endpoints
            start = centers[i]  # [2] (y, x)
            end = centers[j]  # [2] (y, x)
            mid = (start + end) / 2

            # Compute line direction and length
            line_vec = mid - start  # [2]
            line_length = torch.norm(line_vec)

            if line_length < 1e-6:
                continue

            # Unit vector along the line
            line_unit = line_vec / line_length

            # Perpendicular unit vector
            perp_unit = torch.tensor([-line_unit[1], line_unit[0]], device=device)

            # Integration width for this pair
            sigma = max(min(widths[i], widths[j]) / stride * sigma_factor, 0.5)

            # For each pixel, compute distance to the line
            # Vector from start point to each pixel
            pixel_vecs = coords - start.view(2, 1, 1)  # [2, H, W]

            # Project onto line direction to get position along line
            along_line = torch.sum(
                pixel_vecs * line_unit.view(2, 1, 1), dim=0
            )  # [H, W]

            # Project onto perpendicular direction to get distance from line
            perp_dist = torch.abs(
                torch.sum(pixel_vecs * perp_unit.view(2, 1, 1), dim=0)
            )  # [H, W]

            # Only consider pixels within the line segment (0 <= t <= 1)
            t = along_line / line_length
            on_segment = (t >= 0) & (t <= 1)

            # Gaussian weight based on perpendicular distance
            weights = torch.exp(-0.5 * (perp_dist / sigma) ** 2)

            # Zero out weights outside the line segment
            weights = weights * on_segment.float()

            # Normalize weights so they sum to 1 along the integration path
            weight_sum = weights.sum()
            if weight_sum > 1e-6:
                weights = weights / weight_sum

            relation_weights[i, j] = weights

            # visualize the weights use matplotlib
            # import matplotlib.pyplot as plt
            #
            # plt.imshow(weights.cpu().numpy())
            # plt.show()

    return relation_weights


def extract_relations(kaf_img, objects, rel_thresh=0.2):
    """
    Fast relation extraction using pre-computed weights and vectorized operations.

    Args:
        kaf_img: tensor of shape [num_relations*2, H, W]
        objects: list of detected objects
        rel_thresh: confidence threshold
        use_weights: whether to use weighted integration

    Returns:
        relations: list of detected relations
    """
    if len(objects) == 0:
        return []
    fpn_range = {
        0: [128, 512],  # stride 32
        1: [64, 128],  # stride 16
        2: [32, 64],  # stride 8
        3: [0, 32],  # stride 4
        #   ^not conluded
    }

    device = kaf_img[0].device
    num_rel = kaf_img[0].shape[0] // 2
    relations = []

    # Pre-compute integration weights for all object pairs
    weights = []
    for fpn_level in range(len(fpn_range)):
        weights.append(
            get_kaf_path(objects, sigma_factor=0.5, stride=int(32 / pow(2, fpn_level)))
        )

    # Extract object scores for vectorized computation
    scores = torch.tensor([obj["score"] for obj in objects], device=device)

    # Vectorized computation for all relations
    for rel_type in range(num_rel):
        confidences = torch.zeros(len(objects), len(objects), device=device)
        for i, subj in enumerate(objects):
            for j, obj in enumerate(objects):
                if i < j:
                    continue

                # Get relation unit vector
                y0, x0 = subj["yx"]
                y1, x1 = obj["yx"]
                y_m = (y0 + y1) / 2
                x_m = (x0 + x1) / 2
                rel_vec_m20 = torch.tensor(
                    [x_m - x0, y_m - y0], device=device, dtype=torch.float32
                )
                rel_vec_m21 = torch.tensor(
                    [x_m - x1, y_m - y1], device=device, dtype=torch.float32
                )
                rel_length = torch.norm(rel_vec_m20)

                fpn_level = -1
                if rel_length < 1e-6:
                    continue
                else:
                    for fpn_idx in range(len(fpn_range)):
                        if fpn_range[fpn_idx][0] < rel_length <= fpn_range[fpn_idx][1]:
                            fpn_level = fpn_idx
                            # print(
                            #     f"length {rel_length} ,use {fpn_level} level for relation"
                            # )
                if fpn_level == -1:
                    continue

                rel_unit_m20 = rel_vec_m20 / rel_length
                rel_unit_m21 = rel_vec_m21 / rel_length
                # Get relation vector field [2, H, W]
                kaf_vec = kaf_img[fpn_level][2 * rel_type : 2 * rel_type + 2]
                rel_weights_m20 = weights[fpn_level][
                    i, j
                ]  # the line area between middle and object0
                rel_weights_m21 = weights[fpn_level][
                    j, i
                ]  # the line area between middle and object1
                dot_product_m20 = (
                    kaf_vec[0] * rel_unit_m20[0] + kaf_vec[1] * rel_unit_m20[1]
                )
                dot_product_m21 = (
                    kaf_vec[0] * rel_unit_m21[0] + kaf_vec[1] * rel_unit_m21[1]
                )
                confidence_m20 = (rel_weights_m20 * dot_product_m20).sum()
                confidence_m21 = (rel_weights_m21 * dot_product_m21).sum()
                # print(confidence_m20, confidence_m21)
                confidences[i, j] = confidence_m20 + confidence_m21

        # Apply score weighting and threshold
        score_matrix = scores[:, None] * scores[None, :]  # [num_obj, num_obj]
        final_confidences = confidences  # * score_matrix
        # print(final_confidences)

        # Find relations above threshold
        valid_rels = final_confidences > rel_thresh
        subj_ids, obj_ids = torch.where(valid_rels)
        # print(valid_rels)
        # print(subj_ids, obj_ids)

        for subj_id, obj_id in zip(subj_ids.cpu(), obj_ids.cpu()):
            if subj_id != obj_id:
                relations.append(
                    {
                        "subject_id": int(subj_id),
                        "object_id": int(obj_id),
                        "relation": rel_type,
                        "confidence": float(final_confidences[subj_id, obj_id]),
                        "subject_category": objects[subj_id]["category"],
                        "object_category": objects[obj_id]["category"],
                    }
                )

    return relations


def visualize_detections(dets, image):
    """
    Visualize object detections on an image.

    Args:
        dets: numpy array of detections with shape [N, 6] where each row is [x1, y1, x2, y2, score, class_id]
        image: numpy array of image with shape [H, W, 3]

    Returns:
        annotated_image: numpy array of annotated image with detections
    """
    annotated_image = image.copy()
    for det in dets:
        x1, y1, x2, y2, score_class = det
        cv2.rectangle(
            annotated_image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2
        )
        for s_c in score_class:
            score, class_id = s_c
            cv2.putText(
                annotated_image,
                f"{class_id}: {score:.2f}",
                (int(x1 + 50), int(y1)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 0, 0),
                2,
            )

    plt.imshow(annotated_image)
    plt.show()
    return annotated_image


def get_dets_using_mask_bbox(hmaps, regs, w_h_s, gt_bbox, thresh=0.1):
    """
    Refine object detections using mask-derived bounding boxes.

    Args:
        dets: numpy array of detections with shape [N, 6] where each row is [x1, y1, x2, y2, score, class_id]
        gt_bbox: dict mapping object_id to {'center': (x, y), 'scale': [width, height]}

    Returns:
        refined_dets: numpy array of refined detections
    """
    fpn_range = {
        0: [128, 512],  # stride 32
        1: [64, 128],  # stride 16
        2: [32, 64],  # stride 8
        3: [0, 32],  # stride 4
        #   ^not conluded
    }
    final_dets = [0] * len(gt_bbox)
    # print(gt_bbox)

    for mask_id, mask_info in gt_bbox.items():
        mask_center_x, mask_center_y = mask_info["center"]
        mask_scale_x, mask_scale_y = mask_info["scale"]
        mask_scale = mask_scale_x * mask_scale_y

        useful_level = 0

        raw_dets = []
        for fpn_level, range in fpn_range.items():
            if mask_scale <= range[1] ** 2 and mask_scale > range[0] ** 2:
                useful_level = fpn_level
                hmap = hmaps[fpn_level][0].unsqueeze(0)  # [1, 13, H, W]
                reg = regs[fpn_level][0].unsqueeze(0)  # [1, 2, H, W]
                w_h_ = w_h_s[fpn_level][0].unsqueeze(0)  # [1, 2, H, W]
                dets = ctdet_decode(hmap, reg, w_h_, int(32 / pow(2, fpn_level)))
                dets = dets.detach().cpu().numpy().reshape(1, -1, dets.shape[2])[0]

                # Apply score threshold and add to all_detections
                valid_dets = dets[dets[:, 4] > thresh]  # Filter by score > 0.1
                if len(valid_dets) > 0:
                    raw_dets = valid_dets

        best_dist = int(32 / pow(2, useful_level)) * 1.5
        best_det = [
            mask_center_x - mask_scale_x / 2,
            mask_center_y - mask_scale_y / 2,
            mask_center_x + mask_scale_x / 2,
            mask_center_y + mask_scale_y / 2,
            [],
        ]
        for det in raw_dets:
            x1, y1, x2, y2, _, _ = det
            det_center_x, det_center_y = (x1 + x2) / 2, (y1 + y2) / 2
            dist = math.sqrt(
                (det_center_x - mask_center_x) ** 2
                + (det_center_y - mask_center_y) ** 2
            )
            if dist < best_dist:
                best_dist = dist
                best_det[4].append([det[4], det[5]])

        if best_det is None:
            print(f"No matching detection for mask {mask_id}")
            return []
        else:
            final_dets[mask_id] = best_det

    return final_dets


def get_scene_graph(
    hmaps, regs, w_h_s, kafs, bbox, top_K=100, thresh=0.1, inp_image=None
):
    # [hmap(List[tensor[B,13,H,W]]), reg(List[tensor[B,2,H,W]]), w_h_(List[tensor[B,2,H,W]]), kaf(List[tensor[B,28,H,W]])]
    fpn_num = len(hmaps)
    print(f"Receiving fmaps with {fpn_num} FPNs")

    # Combine detections from all FPN levels
    # Refine detections using mask bounding boxes if available
    if bbox:
        refined_dets = get_dets_using_mask_bbox(hmaps, regs, w_h_s, bbox, thresh)
    else:
        print("fail")
        return None

    if inp_image is not None:
        visualize_detections(
            refined_dets, (inp_image[0, 3:6, :, :].permute(1, 2, 0)).cpu().numpy()
        )

    # Convert detections to object format
    objects = []
    for idx, det in enumerate(refined_dets):
        x1, y1, x2, y2, scores_classes = det

        # Filter out low confidence predictions
        valid_scores_classes = [
            [score, class_id] for score, class_id in scores_classes if score > thresh
        ]

        if valid_scores_classes:
            # Use the highest scoring class as the primary category
            primary_score, primary_class = max(valid_scores_classes, key=lambda x: x[0])

            width = x2 - x1
            height = y2 - y1
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2

            objects.append(
                {
                    "id": idx,
                    "category": int(primary_class),
                    "center": [center_x, center_y],
                    "width": width,
                    "height": height,
                    "score": float(primary_score),
                    "yx": [center_y, center_x],
                    "all_classes": [
                        [float(score), int(class_id)]
                        for score, class_id in valid_scores_classes
                    ],
                }
            )

    # Extract relations using the highest resolution KAF map
    relations = []
    if objects and len(kafs) > 0:
        kaf = [k[0] for k in kafs]
        relations = extract_relations(kaf, objects, rel_thresh=0.2)

    scene_graph = {"objects": objects, "relations": relations}
    # print_scene_graph(scene_graph)
    return scene_graph


def print_scene_graph(scene_graph):
    objects = scene_graph["objects"]
    relations = scene_graph["relations"]

    # Visualize objects
    for obj in objects:
        print(
            f"Object ID: {obj['id']}, Category: {PSR_FUNC_CAT[obj['category']]}, Center: {obj['center']}, Width: {obj['width']}, Height: {obj['height']}, Score: {obj['score']}, YX: {obj['yx']}"
        )

    # Visualize relations
    if relations:
        for rel in relations:
            print(
                f"Relation Between Object {rel['subject_id']} and Object {rel['object_id']} of type {PSR_KR_CAT[rel['relation']]}, confidence: {rel['confidence']}"
            )
