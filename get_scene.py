import torch
import torch.nn.functional as F
import numpy as np
from scipy.ndimage import maximum_filter

from utils.post_process import ctdet_decode
import math


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

            # Compute line direction and length
            line_vec = end - start  # [2]
            line_length = torch.norm(line_vec)

            if line_length < 1e-6:
                continue

            # Unit vector along the line
            line_unit = line_vec / line_length

            # Perpendicular unit vector
            perp_unit = torch.tensor([-line_unit[1], line_unit[0]], device=device)

            # Integration width for this pair
            sigma = min(widths[i], widths[j]) / stride * sigma_factor

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


def fast_extract_relations(kaf_img, objects, rel_thresh=0.2, use_weights=True):
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

    if use_weights:
        # Pre-compute integration weights for all object pairs
        weights = []
        for fpn_level in range(len(fpn_range)):
            weights.append(
                get_kaf_path(
                    objects, sigma_factor=0.5, stride=int(32 / pow(2, fpn_level))
                )
            )

        # Extract object scores for vectorized computation
        scores = torch.tensor([obj["score"] for obj in objects], device=device)

        # Vectorized computation for all relations
        for rel_type in range(num_rel):
            confidences = torch.zeros(len(objects), len(objects), device=device)
            for i, subj in enumerate(objects):
                for j, obj in enumerate(objects):
                    if i == j:
                        continue

                    # Get relation unit vector
                    y0, x0 = subj["yx"]
                    y1, x1 = obj["yx"]
                    rel_vec = torch.tensor(
                        [y1 - y0, x1 - x0], device=device, dtype=torch.float32
                    )
                    rel_length = torch.norm(rel_vec)

                    fpn_level = -1
                    if rel_length < 1e-6:
                        continue
                    else:
                        for fpn_idx in range(len(fpn_range)):
                            if (
                                fpn_range[fpn_idx][0]
                                < rel_length
                                <= fpn_range[fpn_idx][1]
                            ):
                                fpn_level = fpn_idx
                    if fpn_level == -1:
                        continue

                    rel_unit = rel_vec / rel_length
                    # Get relation vector field [2, H, W]
                    kaf_vec = kaf_img[fpn_level][2 * rel_type : 2 * rel_type + 2]

                    # Compute confidence for all object pairs at once
                    # weights: [num_obj, num_obj, num_rel, H, W]
                    # kaf_vec: [2, H, W]
                    rel_weights = weights[fpn_level]  # [num_obj, num_obj, H, W]

                    # Compute dot product between vector field and relation direction

                    # Compute weighted dot product
                    dot_product = kaf_vec[0] * rel_unit[0] + kaf_vec[1] * rel_unit[1]

                    # Weighted integration
                    confidence = (rel_weights[i, j] * dot_product).sum()
                    confidences[i, j] = confidence

            # Apply score weighting and threshold
            score_matrix = scores[:, None] * scores[None, :]  # [num_obj, num_obj]
            final_confidences = confidences * score_matrix
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

    else:
        # Fallback to original method if weights disabled
        relations = extract_relations(kaf_img, objects, rel_thresh)

    return relations


def apply_nms(dets, iou_threshold=0.5):
    """
    Apply Non-Maximum Suppression to remove duplicate detections while preserving
    detections with different classes but high IoU.

    Args:
        dets: numpy array of detections [N, 6] where each row is [x1, y1, x2, y2, score, class_id]
        iou_threshold: IoU threshold for NMS

    Returns:
        filtered_dets: list of detections in format [x1, y1, x2, y2, [[score1, cls1], [score2, cls2], ...]]
    """
    if len(dets) == 0:
        return []

    # Sort by score (descending)
    sorted_indices = np.argsort(dets[:, 4])[::-1]
    sorted_dets = dets[sorted_indices]

    merged_dets = []

    while len(sorted_dets) > 0:
        # Start with the detection with highest score
        current_det = sorted_dets[0]
        current_bbox = current_det[:4]  # [x1, y1, x2, y2]
        current_score = current_det[4]
        current_class = int(current_det[5])

        # Initialize the merged detection with current detection's bbox and first score/class
        merged_detection = {
            "bbox": current_bbox.copy(),
            "scores_classes": [[current_score, current_class]],
        }

        if len(sorted_dets) == 1:
            merged_dets.append(
                [*merged_detection["bbox"], merged_detection["scores_classes"]]
            )
            break

        # Calculate IoU with remaining detections
        remaining_dets = sorted_dets[1:]

        # Calculate intersection
        x1_inter = np.maximum(current_bbox[0], remaining_dets[:, 0])
        y1_inter = np.maximum(current_bbox[1], remaining_dets[:, 1])
        x2_inter = np.minimum(current_bbox[2], remaining_dets[:, 2])
        y2_inter = np.minimum(current_bbox[3], remaining_dets[:, 3])

        inter_area = np.maximum(0, x2_inter - x1_inter) * np.maximum(
            0, y2_inter - y1_inter
        )

        # Calculate union
        current_area = (current_bbox[2] - current_bbox[0]) * (
            current_bbox[3] - current_bbox[1]
        )
        remaining_areas = (remaining_dets[:, 2] - remaining_dets[:, 0]) * (
            remaining_dets[:, 3] - remaining_dets[:, 1]
        )
        union_area = current_area + remaining_areas - inter_area

        # Calculate IoU
        ious = inter_area / (union_area + 1e-8)

        # Find detections with high IoU but different classes
        high_iou_mask = ious >= iou_threshold
        high_iou_dets = remaining_dets[high_iou_mask]

        # Check for different classes among high IoU detections
        for det in high_iou_dets:
            det_class = int(det[5])
            det_score = det[4]

            # If it's a different class, add it to the merged detection
            if det_class != current_class:
                # Check if this class already exists in merged detection
                existing_classes = [sc[1] for sc in merged_detection["scores_classes"]]
                if det_class not in existing_classes:
                    merged_detection["scores_classes"].append([det_score, det_class])
                else:
                    # Update score if this detection has higher score for the same class
                    for i, (score, cls) in enumerate(
                        merged_detection["scores_classes"]
                    ):
                        if cls == det_class and det_score > score:
                            merged_detection["scores_classes"][i][0] = det_score

        # Keep detections with IoU below threshold or same class
        keep_mask = (ious < iou_threshold) | (remaining_dets[:, 5] == current_class)
        # For same class detections with high IoU, we remove them (standard NMS behavior)
        keep_mask = keep_mask & ~(
            (ious >= iou_threshold) & (remaining_dets[:, 5] == current_class)
        )
        sorted_dets = remaining_dets[keep_mask]

        # Add the merged detection to results
        merged_dets.append(
            [*merged_detection["bbox"], merged_detection["scores_classes"]]
        )

    return merged_dets


def extract_objects(hmaps, reg_maps, w_h_maps, down_ratio=4, top_K=100):
    # given the maps output from FPN model (stride 32, 16, 8, 4 => size 16, 32, 64, 128), extract valid objects with score
    # map_shape: List[fmap_16(16, 16, c), fmap_32(32, 32, c), fmap_64(64, 64, c), fmap_128(128, 128, c)]
    # Add a batch dimension of 1 to inputs, as ctdet_decode expects batched inputs
    hmaps = hmaps.unsqueeze(0)
    reg_maps = reg_maps.unsqueeze(0)
    w_h_maps = w_h_maps.unsqueeze(0)

    detections = ctdet_decode(hmaps, reg_maps, w_h_maps, down_ratio, K=top_K)

    objects = []
    # Detections are [x1, y1, x2, y2, score, class_id]
    # We assume batch size is 1 here as extract_objects processes a single image's outputs
    for k in range(detections.shape[1]):  # Iterate over K objects
        det = detections[0, k]  # Get the k-th detection from the first (and only) batch
        score = det[4].item()
        if score < 0.1:  # Threshold to filter out low-confidence detections
            continue

        x1, y1, x2, y2 = det[0].item(), det[1].item(), det[2].item(), det[3].item()
        category_id = int(det[5].item())

        width = x2 - x1
        height = y2 - y1
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2

        objects.append(
            {
                "id": k,
                "category": category_id,
                "center": [center_x, center_y],
                "width": width,
                "height": height,
                "score": score,
                "yx": [center_y, center_x],  # Assuming yx means [y, x]
            }
        )
    return objects


def integrate_along_line(kaf_map, y0, x0, y1, x1, inte_width, debug=False):
    # Bresenham's line algorithm or linear interpolation
    num_points = int(np.hypot(y1 - y0, x1 - x0)) + 1
    if num_points <= 0:
        return 0
    ys = np.linspace(y0, y1, num_points)
    xs = np.linspace(x0, x1, num_points)
    unitv = np.array([y1 - y0, x1 - x0])
    norm = np.linalg.norm(unitv)
    if norm == 0:
        return 0
    unitv = unitv / norm  # shape (2,)
    # Perpendicular unit vector
    perp = np.array([-unitv[1], unitv[0]])

    dot_sum = 0
    count = 0
    visited = set()
    # For each point along the line
    for y, x in zip(ys, xs):
        # For each offset along the perpendicular direction
        for w in np.linspace(-inte_width, inte_width, int(2 * inte_width) + 1):
            yy = y + perp[0] * w
            xx = x + perp[1] * w
            y_int, x_int = int(round(yy)), int(round(xx))
            if (0 <= y_int < kaf_map.shape[1]) and (0 <= x_int < kaf_map.shape[2]):
                key = (y_int, x_int)
                if key in visited and debug:
                    print(f"Repeated pixel: {key}")
                    continue
                visited.add(key)
                vec = kaf_map[:, y_int, x_int].cpu().numpy()
                dot_sum += np.dot(vec, unitv)
                count += 1
    if count == 0:
        return 0
    return dot_sum / count


def extract_relations(kaf_img, objects, rel_thresh=0.3):
    """
    Extract relations between objects using Keypoint Affinity Fields (KAF).

    Args:
        kaf_img: tensor of shape [num_relations*2, H, W] containing vector fields for each relation type
        objects: list of detected objects with their positions and scores
        rel_thresh: confidence threshold for relation detection

    Returns:
        relations: list of detected relations with subject, object, relation type and confidence
    """
    relations = []
    num_rel = (
        kaf_img.shape[0] // 2
    )  # Each relation has 2 channels (x, y vector components)

    for i, subj in enumerate(objects):
        for j, obj in enumerate(objects):
            if i == j:
                continue

            # Get subject and object centers in feature map coordinates
            subj_center = subj["yx"]  # [y, x] format
            obj_center = obj["yx"]  # [y, x] format

            y0, x0 = subj_center
            y1, x1 = obj_center

            # Process each relation type
            for rel_type in range(num_rel):
                # Get the 2D vector field for this relation type
                kaf_vec = kaf_img[2 * rel_type : 2 * rel_type + 2]  # shape [2, H, W]
                print(kaf_vec.shape)

                # Calculate integration width based on object sizes
                inte_width = (
                    min(
                        abs(subj["width"]),
                        abs(obj["width"]),
                        abs(subj["height"]),
                        abs(obj["height"]),
                    )
                    / 4
                )  # Use quarter of minimum dimension
                inte_width = max(1, inte_width)  # Ensure minimum width of 1

                # Integrate along the line connecting subject to object
                line_confidence = integrate_along_line(
                    kaf_vec, y0, x0, y1, x1, inte_width, debug=False
                )

                # Weight by object detection scores
                final_confidence = line_confidence * subj["score"] * obj["score"]
                print(final_confidence)

                # Add relation if confidence exceeds threshold
                if final_confidence > rel_thresh:
                    relations.append(
                        {
                            "subject_id": subj["id"],
                            "object_id": obj["id"],
                            "relation": rel_type,
                            "confidence": float(final_confidence),
                            "subject_category": subj["category"],
                            "object_category": obj["category"],
                        }
                    )

    return relations


def refine_dets_using_mask_bbox(dets, gt_bbox):
    """
    Refine object detections using mask-derived bounding boxes.

    Args:
        dets: numpy array of detections with shape [N, 6] where each row is [x1, y1, x2, y2, score, class_id]
        gt_bbox: dict mapping object_id to {'center': (x, y), 'scale': [width, height]}

    Returns:
        refined_dets: numpy array of refined detections
    """
    # Convert multi-class format back to single detections for refinement
    single_dets = []
    for det in dets:
        x1, y1, x2, y2, scores_classes = det
        for score, cls in scores_classes:
            single_dets.append([x1, y1, x2, y2, score, cls])
    dets = np.array(single_dets) if single_dets else np.empty((0, 6))
    if len(dets) == 0 or len(gt_bbox) == 0:
        print("No detections or ground truth bounding boxes found.")
        return dets

    refined_dets = []

    for det in dets:
        x1, y1, x2, y2, score, class_id = det

        # Skip low confidence detections
        if score < 0.1:
            continue

        det_center_x = (x1 + x2) / 2
        det_center_y = (y1 + y2) / 2
        det_width = 1
        det_height = 1

        best_match_id = None
        best_dist = 512

        # Find the best matching mask bbox
        for mask_id, mask_info in gt_bbox.items():
            mask_center_x, mask_center_y = mask_info["center"]
            mask_width, mask_height = mask_info["scale"]

            # Convert mask bbox to x1, y1, x2, y2 format
            mask_x1 = mask_center_x - mask_width / 2
            mask_y1 = mask_center_y - mask_height / 2
            mask_x2 = mask_center_x + mask_width / 2
            mask_y2 = mask_center_y + mask_height / 2

            if (
                mask_x1 <= det_center_x <= mask_x2
                and mask_y1 <= det_center_y <= mask_y2
            ):
                # calculate dist between det center and mask bbox center
                dist = math.sqrt(
                    (det_center_x - mask_center_x) ** 2
                    + (det_center_y - mask_center_y) ** 2
                )
                # print(dist)

                if dist < best_dist and dist < 20:  # Minimum IoU threshold
                    best_dist = dist
                    best_match_id = mask_id

        # Refine detection using best matching mask bbox
        if best_match_id is not None:
            mask_info = gt_bbox[best_match_id]
            mask_center_x, mask_center_y = mask_info["center"]
            mask_width, mask_height = mask_info["scale"]

            # Use weighted average of detection and mask bbox
            weight_det = score
            weight_mask = 2.5  # Fixed weight for mask information
            total_weight = weight_det + weight_mask

            # Weighted center
            refined_center_x = (
                det_center_x * weight_det + mask_center_x * weight_mask
            ) / total_weight
            refined_center_y = (
                det_center_y * weight_det + mask_center_y * weight_mask
            ) / total_weight

            # Weighted size
            refined_width = (
                det_width * weight_det + mask_width * weight_mask
            ) / total_weight
            refined_height = (
                det_height * weight_det + mask_height * weight_mask
            ) / total_weight

            # Convert back to x1, y1, x2, y2 format
            refined_x1 = refined_center_x - refined_width / 2
            refined_y1 = refined_center_y - refined_height / 2
            refined_x2 = refined_center_x + refined_width / 2
            refined_y2 = refined_center_y + refined_height / 2

            refined_dets.append(
                [refined_x1, refined_y1, refined_x2, refined_y2, score, class_id]
            )
        else:
            # Keep original detection if no good match found
            refined_dets.append(det)

    # Convert back to multi-class format after refinement
    refined_dets = apply_nms(
        np.array(refined_dets) if refined_dets else np.empty((0, 6)), iou_threshold=0.5
    )

    return refined_dets


def get_scene_graph(hmaps, regs, w_h_s, kafs, bbox, top_K=100):
    # [hmap(List[tensor[B,13,H,W]]), reg(List[tensor[B,2,H,W]]), w_h_(List[tensor[B,2,H,W]]), kaf(List[tensor[B,28,H,W]])]
    fpn_num = len(hmaps)
    print(f"Receiving fmaps with {fpn_num} FPNs")

    all_detections = []

    # Process detections from each FPN level
    # for fpn_level in range(fpn_num):
    for fpn_level in range(fpn_num):
        hmap = hmaps[fpn_level][0].unsqueeze(0)  # [1, 13, H, W]
        reg = regs[fpn_level][0].unsqueeze(0)  # [1, 2, H, W]
        w_h_ = w_h_s[fpn_level][0].unsqueeze(0)  # [1, 2, H, W]

        # Extract detections from current FPN level
        dets = ctdet_decode(hmap, reg, w_h_, int(32 / pow(2, fpn_level)), K=top_K)
        dets = dets.detach().cpu().numpy().reshape(1, -1, dets.shape[2])[0]

        # Apply score threshold and add to all_detections
        valid_dets = dets[dets[:, 4] > 0.1]  # Filter by score > 0.1
        if len(valid_dets) > 0:
            all_detections.append(valid_dets)

    # Combine detections from all FPN levels
    if all_detections:
        combined_dets = np.vstack(all_detections)

        # Apply Non-Maximum Suppression to remove duplicate detections
        nms_dets = apply_nms(combined_dets, iou_threshold=0.5)
        # refined_dets = combined_dets

        # Refine detections using mask bounding boxes if available
        if bbox:
            refined_dets = refine_dets_using_mask_bbox(nms_dets, bbox)
        else:
            refined_dets = nms_dets

        print(refined_dets)

        # Convert detections to object format
        objects = []
        for idx, det in enumerate(refined_dets):
            x1, y1, x2, y2, scores_classes = det

            # Filter out low confidence predictions
            valid_scores_classes = [
                [score, class_id] for score, class_id in scores_classes if score > 0.1
            ]

            if valid_scores_classes:
                # Use the highest scoring class as the primary category
                primary_score, primary_class = max(
                    valid_scores_classes, key=lambda x: x[0]
                )

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
    else:
        objects = []

    # Extract relations using the highest resolution KAF map
    relations = []
    if objects and len(kafs) > 0:
        kaf = [k[0] for k in kafs]
        relations = fast_extract_relations(
            kaf, objects, rel_thresh=0.15, use_weights=True
        )

    scene_graph = {"objects": objects, "relations": relations}
    print(scene_graph)
    return scene_graph


def visualize_scene_graph(scene_graph):
    objects = scene_graph["objects"]
    relations = scene_graph["relations"]

    # Visualize objects
    for obj in objects:
        print(
            f"Object ID: {obj['id']}, Category: {obj['category']}, Center: {obj['center']}, Width: {obj['width']}, Height: {obj['height']}, Score: {obj['score']}, YX: {obj['yx']}"
        )

    # Visualize relations
    if relations:
        for rel in relations:
            print(
                f"Relation from Object {rel['subject_id']} to Object {rel['object_id']} of type {rel['relation']}, confidence: {rel['confidence']}"
            )


if __name__ == "__main__":
    # Example usage
    # Assuming `kaf` is the output from the KAF-Net model
    kaf = [
        [
            torch.zeros(1, 13, 32, 32),  # hmap
            torch.randn(1, 2, 32, 32),  # regs
            torch.randn(1, 2, 32, 32),  # w_h_
            torch.randn(1, 28, 32, 32),  # kaf
        ]
    ]
    kaf[-1][0][0][0][23][12] = 1
    kaf[-1][0][0][0][13][20] = 1
    kaf[-1][0][0][3][23][12] = 1
    kaf[-1][0][0][3][13][20] = 1
    print("Extracting scene graphs...")

    for scene_graph in get_scene_graph(kaf):
        visualize_scene_graph(scene_graph)
# This code defines a function to extract scene graphs from the KAF-Net outputs
# and visualize them. The actual extraction logic will depend on the specific dataset and task.
