"""Prepare, train, evaluate, or run DSFlash on supplied PSR part masks."""

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import random

import numpy as np
import torch
from torch import distributed as dist
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from datasets.psr_labels import PSR_KR_CAT
from datasets.psr_pairs import PSRPairs, collate_pairs, create_manifest, load_sample
from experiments.dsflash_metrics import relation_metrics
from nets.kaf.dsflash import DSFlashSwin

DEFAULT_CONFIG = Path(__file__).with_name("dsflash_swin_psr.json")


def atomic_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def update_best_checkpoint(latest):
    best_path = latest.with_name("best.pt")
    best_temporary = best_path.with_suffix(".tmp")
    best_temporary.unlink(missing_ok=True)
    os.link(latest, best_temporary)
    best_temporary.replace(best_path)


def save_checkpoint(path, checkpoint):
    temporary = path.with_suffix(".tmp")
    torch.save(checkpoint, temporary)
    temporary.replace(path)
    if checkpoint["epoch"] == checkpoint["best_epoch"]:
        update_best_checkpoint(path)


def init_device():
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    device = torch.device("cuda", local_rank) if torch.cuda.is_available() else torch.device("cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)
    if world_size > 1:
        dist.init_process_group("nccl" if device.type == "cuda" else "gloo")
    return device, rank, world_size


def load_model(config, device, *, pretrained):
    return DSFlashSwin(
        len(PSR_KR_CAT), config["image_size"], config["width"], config["depth"], config["heads"],
        config["patch_grid"], config["pair_chunk"], pretrained=pretrained,
    ).to(device)


def model_inputs(batch, device):
    return tuple(batch[key].to(device, non_blocking=True) for key in ("images", "masks", "pairs"))


def make_loader(config, data_dir, samples, device, training, rank, world_size):
    dataset = PSRPairs(data_dir, samples, config["image_size"], training)
    batch_size = config["global_batch_size"]
    if batch_size < world_size or batch_size % world_size:
        raise ValueError("Global batch size must be divisible by the number of training ranks.")
    sampler = DistributedSampler(
        dataset, num_replicas=world_size, rank=rank, shuffle=True, seed=config["seed"], drop_last=True,
    ) if training else range(rank, len(dataset), world_size)
    return DataLoader(
        dataset, batch_size=batch_size // world_size, sampler=sampler,
        num_workers=config["workers_per_gpu"], collate_fn=collate_pairs,
        pin_memory=device.type == "cuda", drop_last=training, worker_init_fn=configure_worker,
        multiprocessing_context="spawn" if config["workers_per_gpu"] else None,
    )


def configure_worker(_):
    import cv2

    cv2.setNumThreads(1)


@torch.no_grad()
def evaluate(model, loader, device, rank, world_size):
    model.eval()
    samples = []
    for batch in loader:
        output = model(*model_inputs(batch, device))
        scores = output.logits.float().cpu().numpy()
        pairs = batch["pairs"][:, 0].numpy()
        targets = batch["targets"].numpy()
        labeled = batch["labeled"].numpy()
        for index, unavailable in enumerate(batch["unavailable"].numpy()):
            selected = pairs == index
            samples.append((scores[selected], targets[selected], unavailable, labeled[selected]))

    if world_size > 1:
        shards = [None] * world_size
        dist.all_gather_object(shards, samples)
        if rank != 0:
            return None
        samples = [sample for shard in shards for sample in shard]
    return relation_metrics(samples)


def seed_epoch(seed, rank, epoch):
    seed = seed + 1009 * epoch + rank
    random.seed(seed)
    np.random.seed(seed % (2 ** 32))
    torch.manual_seed(seed)


def train_epoch(model, loader, optimizer, scaler, config, pos_weight, device, rank, world_size, epoch):
    model.train()
    totals = torch.zeros(2, device=device)
    for step, batch in enumerate(loader, 1):
        labeled = batch["labeled"]
        support = labeled.sum().to(device, non_blocking=True)
        global_support = support.clone()
        if world_size > 1:
            dist.all_reduce(global_support)
        if not global_support.item():
            continue
        batch["pairs"] = batch["pairs"][labeled]
        inputs = model_inputs(batch, device)
        targets = batch["targets"][labeled].to(device, non_blocking=True).float()
        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=scaler.is_enabled()):
            output = model(*inputs)
        both_orders = torch.stack((output.logits.float(), output.swapped_logits.float()))
        prediction_loss = F.binary_cross_entropy_with_logits(
            both_orders, targets.expand_as(both_orders), pos_weight=pos_weight, reduction="none",
        ).mean(dim=(0, 2))
        per_pair = prediction_loss + config["consistency_weight"] * output.consistency
        numerator = per_pair.sum()
        loss = numerator * world_size / global_support.clamp_min(1)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config["max_grad_norm"])
        scaler.step(optimizer)
        scaler.update()
        totals += torch.stack((numerator.detach(), support.float()))

        if rank == 0 and (step == 1 or step % config["log_interval"] == 0):
            print(
                f"epoch={epoch} step={step}/{len(loader)} loss={float(loss):.6f} "
                f"grad_norm={float(grad_norm):.4f} labeled_pairs={int(global_support)}",
                flush=True,
            )

    if world_size > 1:
        dist.all_reduce(totals)
    mean_loss = float(totals[0] / totals[1].clamp_min(1))
    if not math.isfinite(mean_loss) or int(totals[1]) == 0:
        raise RuntimeError(f"Invalid training epoch {epoch}: loss {mean_loss}, annotated pairs {int(totals[1])}")
    return mean_loss


def read_manifest(path):
    content = path.read_bytes()
    manifest = json.loads(content)
    if manifest.get("version") != 1:
        raise ValueError("Unsupported PSR split manifest version.")
    grouped = [
        {manifest["source_hashes"][sample] for sample in manifest["splits"][split]}
        for split in ("train", "val", "test")
    ]
    if not all(grouped) or len(set.union(*grouped)) != sum(map(len, grouped)):
        raise ValueError("PSR training, validation, and test must have nonempty and disjoint source-image groups.")
    return manifest, hashlib.sha256(content).hexdigest()


def train(args):
    if args.manifest is None or args.data_dir is None or args.run_dir is None:
        raise ValueError("Training requires --manifest, --data-dir, and --run-dir.")
    config = json.loads(args.config.read_text())
    manifest, manifest_hash = read_manifest(args.manifest)
    args.run_dir.mkdir(parents=True, exist_ok=True)
    latest = args.run_dir / "latest.pt"
    if latest.is_file() and not args.resume:
        raise FileExistsError(f"{latest} already exists. Use --resume or another run directory.")
    checkpoint = torch.load(latest, map_location="cpu") if args.resume else None
    if checkpoint and (checkpoint["config"] != config or checkpoint["manifest_sha256"] != manifest_hash):
        raise ValueError("Resume requires the same training configuration and split manifest.")

    device, rank, world_size = init_device()
    configure_worker(None)
    seed_epoch(config["seed"], rank, 0)
    model = load_model(config, device, pretrained=not args.no_pretrained and checkpoint is None)
    if checkpoint:
        model.load_state_dict(checkpoint["model"])
    train_model = DistributedDataParallel(
        model, device_ids=[device.index] if device.type == "cuda" else None,
    ) if world_size > 1 else model

    optimizer = torch.optim.AdamW([
        {"params": model.encoder.parameters(), "lr": config["backbone_lr"]},
        {"params": list(model.fpn.parameters()) + list(model.head.parameters()), "lr": config["head_lr"]},
    ], weight_decay=config["weight_decay"])

    def schedule(epoch):
        warmup = config["warmup_epochs"]
        if epoch < warmup:
            return (epoch + 1) / (warmup + 1)
        progress = min(1.0, (epoch - warmup) / max(1, config["epochs"] - warmup))
        return 0.1 + 0.9 * (1.0 + math.cos(math.pi * progress)) / 2.0

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)
    scaler = torch.cuda.amp.GradScaler(enabled=config["amp"] and device.type == "cuda")
    start_epoch, best_epoch, best_ap = 1, 0, -1.0
    if checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        scaler.load_state_dict(checkpoint["scaler"])
        best_ap = checkpoint["best_val_macro_ap"]
        best_epoch = checkpoint["best_epoch"]
        start_epoch = checkpoint["epoch"] + 1
        if rank == 0 and checkpoint["epoch"] == best_epoch:
            # Recover the best link if the previous run stopped between the two atomic renames.
            update_best_checkpoint(latest)
    del checkpoint

    positive = manifest["support"]["train"]
    support = torch.tensor([positive["labels"][name] for name in PSR_KR_CAT], device=device).float()
    pos_weight = ((positive["labeled_pairs"] - support) / support.clamp_min(1)).sqrt()
    pos_weight = torch.where(support > 0, pos_weight.clamp(1, config["max_positive_weight"]), 1)
    training, validation = [
        make_loader(config, args.data_dir, manifest["splits"][name], device, name == "train", rank, world_size)
        for name in ("train", "val")
    ]
    if not len(training):
        raise ValueError("Training split does not contain one complete global batch.")

    if rank == 0:
        atomic_json(args.run_dir / "config.json", config)
        atomic_json(args.run_dir / "run.json", {
            "manifest": str(args.manifest.resolve()),
            "manifest_sha256": manifest_hash,
            "dataset": str(args.data_dir.resolve()),
            "world_size": world_size,
            "device": str(device),
            "torch": str(torch.__version__),
            "label_support": manifest["support"],
        })
        print(f"DSFlash training ranks={world_size} batch={config['global_batch_size']} "
              f"train={len(training.dataset)} val={len(validation.dataset)} "
              f"starting_epoch={start_epoch} trainable_parameters={sum(p.numel() for p in model.parameters())}",
              flush=True)

    end_epoch = min(args.stop_after_epoch or config["epochs"], config["epochs"])
    for epoch in range(start_epoch, end_epoch + 1):
        seed_epoch(config["seed"], rank, epoch)
        training.sampler.set_epoch(epoch)
        loss = train_epoch(train_model, training, optimizer, scaler, config, pos_weight, device, rank, world_size, epoch)
        if world_size > 1:
            # Validation uses a shard of the dataset per rank, with no padded duplicates.
            for buffer in model.buffers():
                dist.broadcast(buffer, src=0)
        metrics = evaluate(model, validation, device, rank, world_size)
        scheduler.step()
        if rank == 0:
            ap = metrics["macro_AP"]
            if ap is None:
                raise ValueError("Validation has no positive labels.")
            best = ap > best_ap
            if best:
                best_ap = ap
                best_epoch = epoch
            atomic_json(args.run_dir / f"val-{epoch:03d}.json", metrics | {"epoch": epoch, "train_loss": loss})
            save_checkpoint(latest, {
                "config": config, "manifest_sha256": manifest_hash,
                "epoch": epoch, "best_epoch": best_epoch, "best_val_macro_ap": best_ap,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                "scaler": scaler.state_dict(),
            })
            print(f"epoch={epoch} train_loss={loss:.6f} val_macro_AP={ap:.6f} best={best}", flush=True)

    if end_epoch == config["epochs"]:
        if world_size > 1:
            dist.barrier()
        best = torch.load(args.run_dir / "best.pt", map_location="cpu")
        model.load_state_dict(best["model"])
        test = make_loader(config, args.data_dir, manifest["splits"]["test"], device, False, rank, world_size)
        metrics = evaluate(model, test, device, rank, world_size)
        if rank == 0:
            atomic_json(args.run_dir / "test.json", metrics | {
                "checkpoint_epoch": best["epoch"], "manifest_sha256": manifest_hash,
            })
            print(f"test_macro_AP={metrics['macro_AP']:.6f} checkpoint_epoch={best['epoch']}", flush=True)


def run_evaluation(args):
    manifest, manifest_hash = read_manifest(args.manifest)
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    if checkpoint["manifest_sha256"] != manifest_hash:
        raise ValueError("Evaluation manifest differs from the one used for training.")
    config = checkpoint["config"]
    device, rank, world_size = init_device()
    configure_worker(None)
    model = load_model(config, device, pretrained=False)
    model.load_state_dict(checkpoint["model"])
    loader = make_loader(config, args.data_dir, manifest["splits"][args.split], device, False, rank, world_size)
    metrics = evaluate(model, loader, device, rank, world_size)
    if rank == 0:
        atomic_json(args.output, metrics | {"checkpoint_epoch": checkpoint["epoch"], "manifest_sha256": manifest_hash})


@torch.no_grad()
def infer(args):
    device, _, _ = init_device()
    configure_worker(None)
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    config = checkpoint["config"]
    model = load_model(config, device, pretrained=False)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    batch = collate_pairs([load_sample(args.sample_dir, config["image_size"])])
    output = model(*model_inputs(batch, device))
    relations = []
    ids = batch["mask_ids"][0]
    for (_, first, second), scores in zip(batch["pairs"].tolist(), output.logits.sigmoid().tolist()):
        relations.append(dict(mask_ids=[ids[first], ids[second]], scores=dict(zip(PSR_KR_CAT, scores))))
    atomic_json(args.output, {"mask_ids": ids, "relations": relations})


def prepare(args):
    config = json.loads(args.config.read_text())
    manifest = create_manifest(args.data_dir, config["seed"], config["validation_fraction"], config["test_fraction"])
    atomic_json(args.manifest, manifest)
    print(json.dumps(manifest["support"], indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare_parser = commands.add_parser("prepare")
    prepare_parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    prepare_parser.add_argument("--data-dir", required=True, type=Path)
    prepare_parser.add_argument("--manifest", required=True, type=Path)
    prepare_parser.set_defaults(run=prepare)

    train_parser = commands.add_parser("train")
    train_parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    train_parser.add_argument("--manifest", required=True, type=Path)
    train_parser.add_argument("--data-dir", required=True, type=Path)
    train_parser.add_argument("--run-dir", required=True, type=Path)
    train_parser.add_argument("--resume", action="store_true")
    train_parser.add_argument("--stop-after-epoch", type=int)
    train_parser.add_argument("--no-pretrained", action="store_true")
    train_parser.set_defaults(run=train)

    eval_parser = commands.add_parser("evaluate")
    eval_parser.add_argument("--manifest", required=True, type=Path)
    eval_parser.add_argument("--data-dir", required=True, type=Path)
    eval_parser.add_argument("--checkpoint", required=True, type=Path)
    eval_parser.add_argument("--split", choices=("val", "test"), default="test")
    eval_parser.add_argument("--output", required=True, type=Path)
    eval_parser.set_defaults(run=run_evaluation)

    infer_parser = commands.add_parser("infer")
    infer_parser.add_argument("--sample-dir", required=True, type=Path, help="Folder with src_img.png and mask<ID>.png files.")
    infer_parser.add_argument("--checkpoint", required=True, type=Path)
    infer_parser.add_argument("--output", required=True, type=Path)
    infer_parser.set_defaults(run=infer)

    args = parser.parse_args()
    try:
        args.run(args)
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
