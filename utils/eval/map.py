import numpy as np


def calculate_iou(box1, box2):
    # This is the same IoU function from the previous response.
    # It calculates the Intersection over Union of two bounding boxes.
    # Assumes boxes are in the format [x, y, w, h].
    box1_xmin, box1_ymin, box1_xmax, box1_ymax = (
        box1[0],
        box1[1],
        box1[0] + box1[2],
        box1[1] + box1[3],
    )
    box2_xmin, box2_ymin, box2_xmax, box2_ymax = (
        box2[0],
        box2[1],
        box2[0] + box2[2],
        box2[1] + box2[3],
    )

    inter_xmin = max(box1_xmin, box2_xmin)
    inter_ymin = max(box1_ymin, box2_ymin)
    inter_xmax = min(box1_xmax, box2_xmax)
    inter_ymax = min(box1_ymax, box2_ymax)

    inter_width = max(0, inter_xmax - inter_xmin)
    inter_height = max(0, inter_ymax - inter_ymin)
    inter_area = inter_width * inter_height

    box1_area = box1[2] * box1[3]
    box2_area = box2[2] * box2[3]
    union_area = box1_area + box2_area - inter_area

    if union_area == 0:
        return 0.0

    return inter_area / union_area


def compute_ap50_for_class(predictions, ground_truths, iou_thresh=0.5):
    # This is the single-class AP50 function from the previous response.
    # It takes predictions and ground truths for a single class.
    # If there are no ground truths, AP is 0.
    if len(ground_truths) == 0:
        return -1

    # Sort predictions by confidence score in descending order.
    predictions.sort(key=lambda x: x["score"], reverse=True)

    tp = np.zeros(len(predictions))
    fp = np.zeros(len(predictions))
    matched_ground_truths = set()

    for i, pred in enumerate(predictions):
        best_iou = 0.0
        best_gt_idx = -1

        for j, gt in enumerate(ground_truths):
            iou = calculate_iou(pred["bbox"], gt["bbox"])
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = j

        if best_iou >= iou_thresh and best_gt_idx not in matched_ground_truths:
            tp[i] = 1
            matched_ground_truths.add(best_gt_idx)
        else:
            fp[i] = 1

    tp_cumsum = np.cumsum(tp)
    fp_cumsum = np.cumsum(fp)
    precision = tp_cumsum / (tp_cumsum + fp_cumsum)
    recall = tp_cumsum / len(ground_truths)

    # Compute the average precision using interpolated precision.
    ap = 0.0
    for t in np.arange(0.0, 1.1, 0.1):
        if np.sum(recall >= t) == 0:
            p = 0
        else:
            p = np.max(precision[recall >= t])
        ap += p / 11.0
    return ap


def compute_map50(all_predictions, all_ground_truths, class_list):
    """
    Computes mAP50 by averaging AP50 across all classes.

    Args:
        all_predictions (list of dicts): Predictions for the entire dataset.
            Each dict: {'bbox': [...], 'score': ..., 'class_id': ...}
        all_ground_truths (list of dicts): Ground truths for the entire dataset.
            Each dict: {'bbox': [...], 'class_id': ...}
        class_list (list): A list of all class IDs in the dataset.

    Returns:
        float: The mAP50 value.
    """
    class_aps = []

    # expand all predictions
    expanded_preds = []
    for pred in all_predictions:
        top_score = pred["score"]
        rang = 0.1
        for s_c in pred["classes"]:
            if s_c[0] > top_score - rang:
                expanded_preds.append(
                    {"bbox": pred["bbox"], "score": s_c[0], "class_id": s_c[1]}
                )

    for class_id in class_list:
        print(f"compute map for func {class_id}")
        # Filter predictions and ground truths for the current class.
        class_preds = [
            p for p in expanded_preds if class_id == p["class_id"]
        ]  # pred is single class
        class_gts = [
            gt for gt in all_ground_truths if class_id in gt["class_id"]
        ]  # gt is multi class

        print(
            f"Class ID: {class_id}, Number of predictions: {len(class_preds)}, Number of ground truths: {len(class_gts)}"
        )

        # Compute AP50 for the current class.
        ap = compute_ap50_for_class(class_preds, class_gts)
        if ap != -1:
            class_aps.append(ap)

    # Return the mean of all class APs.
    if not class_aps:
        return 0.0

    print(f"APs for each class:\n {class_aps}")
    return np.mean(class_aps)


if __name__ == "__main__":
    # --- Example Usage ---
    # Dummy data for a two-class problem (class_id 1 and 2).
    all_ground_truths = [
        {"bbox": [50, 50, 100, 100], "class_id": 1},
        {"bbox": [200, 200, 50, 50], "class_id": 1},
        {"bbox": [100, 100, 30, 30], "class_id": 2},
    ]

    all_predictions = [
        # Predictions for class 1
        {"bbox": [55, 55, 95, 95], "score": 0.95, "class_id": 1},
        {"bbox": [190, 190, 60, 60], "score": 0.80, "class_id": 1},
        {"bbox": [10, 10, 20, 20], "score": 0.70, "class_id": 1},
        # Predictions for class 2
        {"bbox": [105, 105, 25, 25], "score": 0.90, "class_id": 2},
        {"bbox": [50, 50, 10, 10], "score": 0.60, "class_id": 2},
    ]

    class_list = [1, 2]
    mAP50_value = compute_map50(all_predictions, all_ground_truths, class_list)
    print(f"The calculated mAP50 is: {mAP50_value:.4f}")
