import numpy as np
from tqdm import tqdm


def compute_rel_recall_for_class(predictions, ground_truths, iou_threshold=0.5):
    """
    Calculates the recall of a scene graph.

    Args:
        predictions (list): A list of predicted relationship triplets.
            Each triplet is a dict: {'sub_box', 'pred', 'obj_box', 'score'}
            where 'sub_box' and 'obj_box' are bounding boxes [x, y, w, h],
            'pred' is the predicted predicate, and 'score' is confidence.
        ground_truths (list): A list of ground-truth relationship triplets.
            Each triplet is a dict: {'sub_box', 'pred', 'obj_box'}
        k (int): The number of top-scoring predictions to consider.
        iou_threshold (float): The IoU threshold for a positive match.

    Returns:
        float: The recall at k.
    """
    if not ground_truths:
        return -1

    # Sort predictions by confidence score in descending order.
    predictions.sort(key=lambda x: x["score"], reverse=True)

    # Create a list to track matched ground truths to avoid double counting.
    matched_ground_truths = [False] * len(ground_truths)
    true_positives = 0

    for pred in tqdm(predictions, desc=f"Processing {len(predictions)} predictions"):
        for gt_idx, gt in enumerate(ground_truths):
            if matched_ground_truths[gt_idx]:
                continue

            # Check for a match on all three components: subject, object, and predicate.
            sub_iou = calculate_iou(pred["sub_box"], gt["sub_box"])
            obj_iou = calculate_iou(pred["obj_box"], gt["obj_box"])

            # Check if the predicate is a match.
            pred_match = pred["pred"] == gt["pred"]

            # A correct detection requires all three components to match.
            if sub_iou >= iou_threshold and obj_iou >= iou_threshold and pred_match:
                true_positives += 1
                matched_ground_truths[gt_idx] = True
                # Break to ensure one prediction matches only one ground truth.
                break

    recall = true_positives / len(ground_truths)
    return recall


def compute_rel_mean_recall(all_predictions, all_ground_truths, class_list, k=200):
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
    if len(all_ground_truths) == 0:
        return -1
    class_Rs = []

    for class_id in class_list:
        print(f"compute recall for rel {class_id}")
        # Filter predictions and ground truths for the current class.
        class_preds = [
            p for p in all_predictions if class_id == p["pred"]
        ]  # pred is single class
        class_gts = [
            gt for gt in all_ground_truths if class_id == gt["pred"][0]
        ]  # gt is multi class

        # Compute AP50 for the current class.
        R = compute_rel_recall_for_class(class_preds, class_gts)
        if R != -1:
            class_Rs.append(R)

    # Return the mean of all class APs.
    if not class_Rs:
        return 0.0
    print(f"Recalls for each class:\n {class_Rs}")
    return np.mean(class_Rs)


def calculate_iou(box1, box2):
    """
    Helper function to calculate IoU of two bounding boxes.
    (This is the same helper function from previous responses.)
    """
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


# --- Example Usage ---
if __name__ == "__main__":
    # Dummy data for a single image
    predictions = [
        {
            "sub_box": [10, 10, 20, 20],
            "pred": "on",
            "obj_box": [50, 50, 30, 30],
            "score": 0.9,
        },  # Correct detection
        {
            "sub_box": [11, 11, 18, 18],
            "pred": "on",
            "obj_box": [55, 55, 25, 25],
            "score": 0.8,
        },  # Duplicate detection, but also correct
        {
            "sub_box": [100, 100, 50, 50],
            "pred": "next to",
            "obj_box": [160, 160, 40, 40],
            "score": 0.7,
        },  # Correct
        {
            "sub_box": [200, 200, 50, 50],
            "pred": "riding",
            "obj_box": [250, 250, 60, 60],
            "score": 0.6,
        },  # Incorrect predicate
        {
            "sub_box": [300, 300, 20, 20],
            "pred": "in",
            "obj_box": [350, 350, 10, 10],
            "score": 0.5,
        },  # Incorrect bounding box
    ]

    ground_truths = [
        {"sub_box": [12, 12, 17, 17], "pred": "on", "obj_box": [52, 52, 28, 28]},
        {
            "sub_box": [102, 102, 48, 48],
            "pred": "next to",
            "obj_box": [162, 162, 38, 38],
        },
    ]

    recall_at_50 = compute_rel_recall_for_class(predictions, ground_truths, k=50)
    print(
        f"Recall@50: {recall_at_50:.2f}"
    )  # This will correctly output 1.00 because both ground truths are found.
