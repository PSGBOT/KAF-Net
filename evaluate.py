import os
import sys
import argparse
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
import torch.distributed as dist

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "lib"))

from datasets.psr import PSRDataset_eval, PSR_FUNC_CAT
from nets.kaf.kaf_resdcn import get_kaf_resdcn
from nets.kaf.kaf_swint import get_kaf_swint
from nets.kaf.hourglass import get_kaf_hourglass
from nets.kaf.HRnet import get_kaf_hrnet
from nets.kaf.resnet_pretrain import get_resnet50_fpn
from utils.utils import load_model
from utils.post_process import ctdet_decode
from utils.eval.map import compute_map50
from utils.summary import create_logger, DisablePrint
from get_scene import get_scene_graph


def parse_args():
    parser = argparse.ArgumentParser(description="KAF-Net Evaluation")
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument("--dist", action="store_true")

    parser.add_argument("--root_dir", type=str, default="./")
    parser.add_argument("--data_dir", type=str, default="./data")
    parser.add_argument("--log_name", type=str, default="test_eval")
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the trained model checkpoint",
    )

    parser.add_argument("--dataset", type=str, default="psr")
    parser.add_argument("--arch", type=str, default="large_hourglass")
    parser.add_argument("--img_size", type=int, default=512)
    parser.add_argument("--split_ratio", type=float, default=1.0)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--test_topk", type=int, default=100)
    parser.add_argument(
        "--conf_thresh",
        type=float,
        default=0.1,
        help="Confidence threshold for detections",
    )
    parser.add_argument(
        "--nms_thresh",
        type=float,
        default=0.5,
        help="NMS threshold for post-processing",
    )

    return parser.parse_args()


def to_device(batch, device):
    if isinstance(batch, torch.Tensor):
        return batch.to(device=device, non_blocking=True)
    elif isinstance(batch, dict):
        return {k: to_device(v, device) for k, v in batch.items()}
    elif isinstance(batch, list):
        return [to_device(v, device) for v in batch]
    else:
        return batch


def decode_detections(outputs, bbox, conf_thresh=0.1, topk=100):
    """
    Decode model outputs into detection format
    Returns: list of detections with format [x, y, w, h, score, class_id]
    """
    hmaps, regs, w_h_s, rafs = outputs
    topk_relations = 50  # Top K relations to consider
    scene_graph = get_scene_graph(
        hmaps,
        regs,
        w_h_s,
        rafs,
        bbox,
        topk_relations,
        conf_thresh,
        inp_image=None,
    )
    all_detections = []
    for key, bbox in enumerate(scene_graph["objects"]):
        all_detections.append(
            {
                "bbox": [
                    bbox["center"][0],
                    bbox["center"][1],
                    bbox["width"],
                    bbox["width"],
                ],
                "score": float(bbox["score"]),
                "class_id": int(bbox["category"]),
            }
        )
    print(all_detections)

    return all_detections


def extract_ground_truth(sample):
    """
    Extract ground truth annotations from batch
    Returns: list of ground truth with format [x, y, w, h, class_id]
    bbox: {1: {'center': (265.0, 72.0), 'scale': [36, 18]}, 0: {'center': (267.0, 95.5), 'scale': [26, 37]}}
    """
    ground_truth = []
    bbox = {}

    # Extract from metadata if available
    ind_mask = sample["gt_ind_mask"][0]
    centers = sample["masks_bbox_center"][0][: sum(ind_mask)]
    whs = sample["masks_bbox_wh"][0][: sum(ind_mask)]
    cat_values = list(sample["masks_cat"].values())[: sum(ind_mask)]
    for obj_idx in range(len(centers)):
        valid_cat = []
        for index, value in cat_values[obj_idx].items():
            if value > 0 and index not in valid_cat:
                valid_cat.append(index)

        ground_truth.append(
            {
                "bbox": [
                    centers[obj_idx][0],
                    centers[obj_idx][1],
                    whs[obj_idx][0],
                    whs[obj_idx][1],
                ],
                "class_id": valid_cat,
            }
        )
        bbox[obj_idx] = {
            "center": (centers[obj_idx][0], centers[obj_idx][1]),
            "scale": [whs[obj_idx][0], whs[obj_idx][1]],
        }
    print(ground_truth)
    return ground_truth, bbox


@torch.no_grad()
def evaluate_model(model, test_loader, device, down_ratio, args, logger):
    """
    Evaluate model on test dataset and compute mAP50
    """
    model.eval()
    all_predictions = []
    all_ground_truths = []

    logger.info("Starting evaluation...")

    map50s = []
    for batch_idx, batch in enumerate(test_loader):
        # Move batch to device
        for k in batch:
            if k != "meta":
                batch[k] = to_device(batch[k], device)

        # Forward pass
        outputs = model(batch["masked_img"])

        batch_size = outputs[0][0].shape[0]
        for img_idx in range(batch_size):
            # Extract ground truth
            ground_truths, bbox_for_infer = extract_ground_truth(batch)
            all_ground_truths.extend(ground_truths)
            # Decode predictions

            predictions = decode_detections(outputs, bbox_for_infer)
            all_predictions.extend(predictions)
        if batch_idx % 10 == 0:
            logger.info(f"Processed {batch_idx}/{len(test_loader)} batches")

    class_list = list(range(len(PSR_FUNC_CAT)))
    map50 = compute_map50(all_predictions, all_ground_truths, class_list)
    print(map50)
    return map50


def main():
    args = parse_args()

    os.chdir(args.root_dir)
    args.log_dir = os.path.join(args.root_dir, "logs", args.log_name)
    os.makedirs(args.log_dir, exist_ok=True)

    # Setup distributed training if needed
    num_gpus = torch.cuda.device_count()
    if args.dist:
        args.device = torch.device("cuda:%d" % args.local_rank)
        torch.cuda.set_device(args.local_rank)
        dist.init_process_group(
            backend="nccl",
            init_method="env://",
            world_size=num_gpus,
            rank=args.local_rank,
        )
    else:
        args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Setup logger
    logger = create_logger(args.local_rank, save_dir=args.log_dir)
    print = logger.info
    print(args)

    # Setup down_ratio based on architecture
    if "hourglass" in args.arch:
        down_ratio = {"p2": 4}
    elif "resdcn" in args.arch:
        down_ratio = {"p5": 32, "p4": 16, "p3": 8, "p2": 4}
    elif "swin" in args.arch:
        down_ratio = {"p5": 32, "p4": 16, "p3": 8, "p2": 4}
    elif "hrnet" in args.arch:
        down_ratio = {"p2": 4}
    elif "resnet" in args.arch:
        down_ratio = {"p5": 32, "p4": 16, "p3": 8, "p2": 4}
    else:
        raise NotImplementedError(f"Architecture {args.arch} not implemented")

    print("Setting up test dataset...")
    test_dataset = PSRDataset_eval(
        os.path.join(args.data_dir, "test"),
        "test",
        split_ratio=args.split_ratio,
        down_ratio=down_ratio,
        img_size=args.img_size,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size // num_gpus if args.dist else args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    print("Loading model...")
    # Create model based on architecture
    if "hourglass" in args.arch:
        model = get_kaf_hourglass[args.arch]
    elif "resdcn" in args.arch:
        model = get_kaf_resdcn(
            num_layers=int(args.arch.split("_")[-1]),
            num_classes=test_dataset.num_func_cat,
            num_rel=test_dataset.num_kr_cat,
        )
    elif "swin" in args.arch:
        model = get_kaf_swint(
            head_conv=64,
            num_classes=test_dataset.num_func_cat,
            num_rel=test_dataset.num_kr_cat,
        )
    elif "hrnet" in args.arch:
        model = get_kaf_hrnet(
            num_classes=test_dataset.num_func_cat,
            num_relations=test_dataset.num_kr_cat,
        )
    elif "resnet" in args.arch:
        model = get_resnet50_fpn(
            pretrained=True, input_channels=4, num_classes=13, num_rel=14, head_conv=64
        )
    else:
        raise NotImplementedError(f"Architecture {args.arch} not implemented")

    # Load trained weights
    if not os.path.isfile(args.model_path):
        raise FileNotFoundError(f"Model checkpoint not found: {args.model_path}")

    model, _ = load_model(model, args.model_path)

    if args.dist:
        model = model.to(args.device)
        model = nn.parallel.DistributedDataParallel(
            model,
            device_ids=[args.local_rank],
            output_device=args.local_rank,
        )
    else:
        model = nn.DataParallel(model).to(args.device)

    # Run evaluation
    map50 = evaluate_model(model, test_loader, args.device, down_ratio, args, logger)

    print(f"\nEvaluation completed!")
    print(f"mAP50: {map50:.4f}")

    # Save results
    results = {
        "mAP50": map50,
    }

    import json

    results_path = os.path.join(args.log_dir, "evaluation_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Results saved to: {results_path}")


if __name__ == "__main__":
    args = parse_args()
    with DisablePrint(local_rank=args.local_rank):
        main()
