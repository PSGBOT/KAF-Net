"""Evaluate the original KAF scene decoder on PSR mask and pair identities."""

from itertools import combinations
from pathlib import Path

import numpy as np
import torch
from torch import distributed as dist
from torch.nn import functional as F
from torch.utils.data import DataLoader

from datasets.psr_labels import PSR_FUNC_CAT, PSR_KR_CAT
from experiments.dsflash_metrics import function_metrics, relation_metrics


def score_sample(sample, graph):
    graph = graph or {"objects": (), "relations": ()}
    boxes = sample.get("mask_boxes")
    mask_ids = sorted(sample["masks_cat"]) if boxes is None else sorted(
        mask for mask, box in boxes.items() if all(box["scale"])
    )
    mask_positions = {mask: i for i, mask in enumerate(mask_ids)}
    pair_positions = {pair: i for i, pair in enumerate(combinations(mask_ids, 2))}
    part_scores = np.full((len(mask_positions), len(PSR_FUNC_CAT)), -np.inf)
    edge_scores = np.full((len(pair_positions), len(PSR_KR_CAT)), -np.inf)
    part_labels = np.zeros_like(part_scores, dtype=bool)
    edge_labels = np.zeros_like(edge_scores, dtype=bool)
    missed_parts = np.zeros(len(PSR_FUNC_CAT), dtype=int)
    missed_edges = np.zeros(len(PSR_KR_CAT), dtype=int)
    for mask, votes in sample["masks_cat"].items():
        for category, count in votes.items():
            if count:
                if mask in mask_positions:
                    part_labels[mask_positions[mask], category] = True
                else:
                    missed_parts[category] += 1
    for first, second, category in sample["relations"]:
        pair = tuple(sorted((first, second)))
        if pair in pair_positions:
            edge_labels[pair_positions[pair], category] = True
        else:
            missed_edges[category] += 1

    objects = graph["objects"]
    for obj in objects:
        row = part_scores[mask_positions[obj["id"]]]
        for score, category in obj["all_classes"]:
            row[category] = max(row[category], score)
    for relation in graph["relations"]:
        first = objects[int(relation["subject_id"])]["id"]
        second = objects[int(relation["object_id"])]["id"]
        index = pair_positions[tuple(sorted((first, second)))], int(relation["relation"])
        edge_scores[index] = max(edge_scores[index], float(relation["confidence"]))
    edges = (edge_scores, edge_labels, missed_edges, edge_labels.any(axis=1))
    parts = (part_scores, part_labels, missed_parts)
    return edges, parts


def _collate(samples):
    return samples


def kaf_eval_loader(data_dir, split, down_ratio, img_size, batch_size, num_workers, *,
                    prune=False, rank=0, world_size=1, manifest=None, split_ratio=1.0):
    from datasets.psr import PSRDataset

    dataset = PSRDataset(
        Path(data_dir) / split if manifest is None else data_dir,
        split, split_ratio=split_ratio, down_ratio=down_ratio, img_size=img_size,
        prune=prune, labels_only=True, sample_paths=None if manifest is None else manifest["splits"][split],
    )
    if not len(dataset):
        raise ValueError(f"The PSR {split} split is empty.")
    return DataLoader(
        dataset, batch_size=batch_size, sampler=range(rank, len(dataset), world_size),
        num_workers=num_workers, collate_fn=_collate,
        multiprocessing_context="spawn" if num_workers else None,
    )


def kaf_fields(outputs):
    """Return four heads with level lists from either FPN or stacked outputs."""
    if len(outputs) != 4:
        *detections, relations = outputs[-1]
        shape = relations.shape[-2:]
        return tuple(
            (F.interpolate(field, size=shape, mode="bilinear", align_corners=False),)
            if field.shape[-2:] != shape else (field,)
            for field in (*detections, relations)
        )
    return outputs


def kaf_input(images, channels):
    if channels == 4 and images.shape[1] == 6:
        return torch.cat((images[:, :3], images[:, 3:].amax(dim=1, keepdim=True)), dim=1)
    return images


@torch.no_grad()
def evaluate_kaf(model, loader, device, *, rank=0, world_size=1, conf_thresh=0.1, topk=100, channels=6):
    from get_scene import get_scene_graph

    model.eval()
    records = []
    for batch in loader:
        images = torch.stack([torch.as_tensor(sample["masked_img"]) for sample in batch])
        fields = kaf_fields(model(kaf_input(images.to(device), channels)))
        for index, sample in enumerate(batch):
            prediction = [[level[index:index + 1] for level in head] for head in fields]
            graph = get_scene_graph(*prediction, sample["mask_boxes"], top_K=topk, thresh=conf_thresh)
            records.append(score_sample(sample, graph))
    if world_size > 1:
        shards = [None] * world_size
        dist.all_gather_object(shards, records)
        if rank != 0:
            return None
        records = [result for shard in shards for result in shard]
    metrics = relation_metrics([edges for edges, _ in records])
    metrics["functions"] = function_metrics([parts for _, parts in records])
    return metrics
