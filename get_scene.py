import torch
import numpy as np

from utils.post_process import ctdet_decode
from datasets.psr_labels import PSR_FUNC_CAT, PSR_KR_CAT


@torch.no_grad()
def get_kaf_path(objects, sigma_factor=2.0, stride=4, *, shape=None, device=None):
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

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Extract object centers
    centers = torch.tensor(
        [[obj["yx"][0] / stride, obj["yx"][1] / stride] for obj in objects],
        device=device,
        dtype=torch.float32,
    )  # [N, 2] (y, x)

    H, W = shape if shape is not None else (512 // stride, 512 // stride)

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


def _fpn_levels(count):
    if count == 1:
        return ((4, 0, float("inf")),)
    if count == 4:
        return ((32, 128, 512), (16, 64, 128), (8, 32, 64), (4, 0, 32))
    raise ValueError(f"Expected one or four KAF feature levels, got {count}.")


def extract_relations(kaf_img, objects, rel_thresh=0.2, debug=False):
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
    levels = _fpn_levels(len(kaf_img))

    device = kaf_img[0].device
    num_rel = kaf_img[0].shape[0] // 2
    relations = []

    # Pre-compute integration weights for all object pairs
    weights = []
    for field, (stride, _, _) in zip(kaf_img, levels):
        weights.append(
            get_kaf_path(objects, sigma_factor=0.5, stride=stride, shape=field.shape[-2:], device=device)
        )

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
                    for fpn_idx, (_, lower, upper) in enumerate(levels):
                        if lower < rel_length <= upper:
                            fpn_level = fpn_idx
                            if debug:
                                print(
                                    f"length {rel_length} ,use {fpn_level} level for relation"
                                )
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

        # Find relations above threshold
        valid_rels = confidences > rel_thresh
        subj_ids, obj_ids = torch.where(valid_rels)
        scores = confidences[subj_ids, obj_ids].sigmoid().tolist()
        for subj_id, obj_id, score in zip(subj_ids.tolist(), obj_ids.tolist(), scores):
            if subj_id != obj_id:
                relations.append(
                    {
                        "subject_id": subj_id,
                        "subject_bbox": objects[subj_id]["bbox"],
                        "object_id": obj_id,
                        "object_bbox": objects[obj_id]["bbox"],
                        "relation": rel_type,
                        "confidence": score,
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
    import cv2
    import matplotlib.pyplot as plt

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


def get_dets_using_mask_bbox(hmaps, regs, w_h_s, gt_bbox, thresh=0.1, topk=100):
    """Associate class proposals with supplied masks at their assigned FPN level."""
    mask_ids = sorted(gt_bbox)
    if not mask_ids:
        return []
    centers = np.array([gt_bbox[mask]["center"] for mask in mask_ids])
    sizes = np.array([gt_bbox[mask]["scale"] for mask in mask_ids])
    area = sizes.prod(axis=1)
    scores_by_mask = [{} for _ in mask_ids]
    for level, (stride, lower, upper) in enumerate(_fpn_levels(len(hmaps))):
        eligible = np.flatnonzero((lower ** 2 < area) & (area <= upper ** 2))
        if not len(eligible):
            continue
        hmap = hmaps[level]
        h, w = hmap.shape[-2:]
        proposals = ctdet_decode(hmap, regs[level], w_h_s[level], stride, K=min(topk, h * w))
        proposals = proposals[0].detach().cpu().numpy()
        proposals = proposals[proposals[:, 4] > thresh]
        proposal_centers = (proposals[:, :2] + proposals[:, 2:4]) / 2
        distances = ((proposal_centers[:, None] - centers[eligible]) ** 2).sum(axis=2)
        closest = distances.argmin(axis=1)
        for index, (candidate, owner) in enumerate(zip(proposals, closest)):
            if distances[index, owner] < (stride * 1.5) ** 2:
                score, category = candidate[4:]
                scores_by_mask[eligible[owner]].setdefault(int(category), float(score))
    return [
        [*(center - size / 2).tolist(), *(center + size / 2).tolist(),
         [(score, category) for category, score in classes.items()]]
        for center, size, classes in zip(centers, sizes, scores_by_mask)
    ]


def get_scene_graph(
    hmaps, regs, w_h_s, kafs, bbox, top_K=100, thresh=0.1, inp_image=None, debug=False
):
    # [hmap(List[tensor[B,13,H,W]]), reg(List[tensor[B,2,H,W]]), w_h_(List[tensor[B,2,H,W]]), kaf(List[tensor[B,28,H,W]])]
    # Combine detections from all FPN levels
    # Refine detections using mask bounding boxes if available
    if bbox:
        refined_dets = get_dets_using_mask_bbox(hmaps, regs, w_h_s, bbox, thresh, top_K)
    else:
        return {"objects": [], "relations": []}

    if inp_image is not None:
        visualize_detections(
            refined_dets, (inp_image[0, 3:6, :, :].permute(1, 2, 0)).cpu().numpy()
        )

    # Convert detections to object format
    objects = []
    for idx, det in zip(sorted(bbox), refined_dets):
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
                    "bbox": [x1, y1, x2 - x1, y2 - y1],  # x1, y1, w, h
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
        relations = extract_relations(kaf, objects, rel_thresh=0.2, debug=debug)

    scene_graph = {"objects": objects, "relations": relations}
    if debug:
        print_scene_graph(scene_graph)
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
