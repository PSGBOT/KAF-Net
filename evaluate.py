if __name__ == "__main__":
    from utils.model_routing import route_model

    route_model("evaluate")

import argparse
from contextlib import closing
import os
from pathlib import Path
import sys

import torch
from torch import distributed as dist

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "lib"))

from utils.eval.kaf import evaluate_kaf, kaf_eval_loader
from datasets.psr_pairs import read_manifest
from utils.eval.reporting import atomic_json, write_evaluation
from utils.model_routing import kaf_architecture
from utils.summary import create_logger, create_tensorboard_summary
from utils.utils import load_model_for_inference


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate KAF-Net on PSR mask and relation labels.")
    parser.add_argument("--local_rank", "--local-rank", type=int, default=int(os.environ.get("LOCAL_RANK", "0")))
    parser.add_argument("--dist", action="store_true")
    parser.add_argument("--root_dir", type=Path, default=Path("."))
    parser.add_argument("--data_dir", "--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--log_name", default="test_eval")
    parser.add_argument("--model_path", "--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset", default="psr")
    parser.add_argument("--arch", default="large_hourglass")
    parser.add_argument("--img_size", type=int, default=512)
    parser.add_argument("--split", choices=("val", "test"), default="test")
    parser.add_argument("--manifest", type=Path, help="Optional grouped PSR split manifest.")
    parser.add_argument("--split_ratio", type=float, default=1.0)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--test_topk", type=int, default=100)
    parser.add_argument("--conf_thresh", type=float, default=0.1)
    parser.add_argument("--nms_thresh", type=float, default=0.5, help=argparse.SUPPRESS)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--tensorboard-dir", type=Path)
    return parser.parse_args()


def main(args):
    os.chdir(args.root_dir)
    distributed = args.dist or int(os.environ.get("WORLD_SIZE", "1")) > 1
    device = torch.device("cuda", args.local_rank) if torch.cuda.is_available() else torch.device("cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)
    if distributed:
        dist.init_process_group("nccl" if device.type == "cuda" else "gloo")
    rank, world_size = (dist.get_rank(), dist.get_world_size()) if distributed else (0, 1)
    log_dir = Path("logs") / args.log_name
    if rank == 0:
        log_dir.mkdir(parents=True, exist_ok=True)
    logger = create_logger(rank, save_dir=log_dir)

    manifest, manifest_hash = None, None
    if args.manifest:
        manifest, manifest_hash = read_manifest(args.manifest)
    model, down_ratio, channels = kaf_architecture(args.arch, image_size=args.img_size, pretrained=False)
    model, epoch = load_model_for_inference(model, args.model_path, strict=True)
    loader = kaf_eval_loader(
        args.data_dir, args.split, down_ratio, args.img_size, max(1, args.batch_size // world_size),
        args.num_workers, rank=rank, world_size=world_size, manifest=manifest, split_ratio=args.split_ratio,
    )
    metrics = evaluate_kaf(
        model.to(device), loader, device, rank=rank, world_size=world_size, channels=channels,
        conf_thresh=args.conf_thresh, topk=args.test_topk,
    )
    if rank == 0:
        output = args.output or log_dir / "evaluation_results.json"
        with closing(create_tensorboard_summary(log_dir=str(args.tensorboard_dir or output.parent / "tensorboard"))) as writer:
            atomic_json(output, metrics | {"checkpoint_epoch": epoch, "manifest_sha256": manifest_hash})
            write_evaluation(writer, args.split, metrics, epoch)
        logger.info(
            "%s | relation macro AP=%s | function macro AP=%s | R=%s | mR=%s",
            output, metrics["macro_AP"], metrics["functions"]["macro_AP"], metrics["R"], metrics["mR"],
        )


if __name__ == "__main__":
    try:
        main(parse_args())
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()
