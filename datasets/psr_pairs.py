"""Read independent PSR masks and their unordered relation annotations."""

from collections import defaultdict
import hashlib
import json
from pathlib import Path
import random
import re

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset

from datasets.psr_labels import PSR_FUNC_CAT, PSR_FUNC_CAT_IDX, PSR_KR_CAT, PSR_MEAN, PSR_STD, relation_index


_MASK_FILE = re.compile(r"mask(\d+)\.png\Z")
_SPLITS = ("train", "val", "test")


def pair_indices(valid):
    """Return (image, first_mask, second_mask) for each valid unordered pair."""
    first, second = torch.triu_indices(valid.size(1), valid.size(1), offset=1, device=valid.device)
    images, pairs = (valid[:, first] & valid[:, second]).nonzero(as_tuple=True)
    return torch.stack((images, first[pairs], second[pairs]), dim=1)


def annotation_labels(annotation):
    labels = set()
    for first, second, relation in annotation["kinematic relation"]:
        i, j = sorted((int(first.removeprefix("mask")), int(second.removeprefix("mask"))))
        if i != j:
            labels.update((i, j, relation_index(joint)) for joint in relation["kinematic_joints"])
    return labels


def build_targets(annotation, mask_ids):
    """Encode observed labels and keep unannotated pairs out of the loss."""
    positions = {mask_id: index for index, mask_id in enumerate(mask_ids)}
    labels = torch.zeros(len(mask_ids), len(mask_ids), len(PSR_KR_CAT), dtype=torch.bool)
    labeled = torch.zeros(len(mask_ids), len(mask_ids), dtype=torch.bool)
    for first, second, relation in annotation_labels(annotation):
        if first not in positions or second not in positions:
            raise ValueError(f"Annotation references missing mask IDs in pair ({first}, {second}).")
        i, j = sorted((positions[first], positions[second]))
        labels[i, j, relation] = True
        labeled[i, j] = True
    return labels, labeled


def annotation_functions(annotation):
    functions = defaultdict(set)
    for first, second, relation in annotation["kinematic relation"]:
        for mask, key in ((first, "part0_function"), (second, "part1_function")):
            classes = {PSR_FUNC_CAT_IDX[name] for name in (relation.get(key) or ()) if name in PSR_FUNC_CAT_IDX}
            if classes:
                functions[int(mask.removeprefix("mask"))].update(classes)
    return functions


def build_function_targets(annotation, mask_ids):
    positions = {mask_id: index for index, mask_id in enumerate(mask_ids)}
    labels = torch.zeros(len(mask_ids), len(PSR_FUNC_CAT), dtype=torch.bool)
    for mask_id, classes in annotation_functions(annotation).items():
        if mask_id not in positions:
            raise ValueError(f"Function annotation references missing mask ID {mask_id}.")
        labels[positions[mask_id], list(classes)] = True
    return labels


def group_source_images(records, seed, val_fraction, test_fraction, *, source_labels=None):
    """Assign source images, rather than part samples, to one split each."""
    if not (0 < val_fraction < 1 and 0 < test_fraction < 1 and val_fraction + test_fraction < 1):
        raise ValueError("Validation and test fractions must leave room for training.")

    by_source = defaultdict(list)
    for sample, source_hash in records:
        by_source[source_hash].append(sample)

    source_ids = sorted(by_source)
    if len(source_ids) < 3:
        raise ValueError("The dataset needs at least three distinct source images.")
    random.Random(seed).shuffle(source_ids)
    val_count = max(1, round(len(source_ids) * val_fraction))
    test_count = max(1, round(len(source_ids) * test_fraction))
    if val_count + test_count >= len(source_ids):
        raise ValueError("Cannot allocate a nonempty training split.")

    limits = {"train": len(source_ids) - val_count - test_count, "val": val_count, "test": test_count}
    selected = {name: set() for name in _SPLITS}
    remaining = set(source_ids)
    if source_labels:
        sources_by_label = defaultdict(list)
        for source in source_ids:
            for label in source_labels.get(source, ()):
                sources_by_label[label].append(source)

        # Reserve support for rare labels before allocating the recurring classes.
        for label in sorted(sources_by_label, key=lambda label: (len(sources_by_label[label]), label)):
            candidates = sources_by_label[label]
            priority = ("train", "test", "val")[:len(candidates)]
            for split in priority:
                if len(selected[split]) == limits[split] or selected[split].intersection(candidates):
                    continue
                source = next((source for source in candidates if source in remaining), None)
                if source is not None:
                    selected[split].add(source)
                    remaining.remove(source)

    unallocated = iter(source for source in source_ids if source in remaining)
    for split in ("val", "test", "train"):
        selected[split].update(next(unallocated) for _ in range(limits[split] - len(selected[split])))

    return {split: sorted(path for source in selected[split] for path in by_source[source])
            for split in _SPLITS}


def _sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sample_fingerprint(sample):
    files = [sample / "config.json", *sorted(path for path in sample.iterdir() if _MASK_FILE.fullmatch(path.name))]
    return tuple((path.name, _sha256(path)) for path in files)


def create_manifest(data_dir, seed, val_fraction, test_fraction):
    data_dir = Path(data_dir)
    source_hashes, labels_by_sample = {}, {}
    source_labels = defaultdict(set)
    duplicate_candidates = defaultdict(list)
    removed_copies = 0
    for split in _SPLITS:
        for config_path in sorted((data_dir / split).glob("*/config.json")):
            sample = config_path.parent
            source = _sha256(sample / "src_img.png")
            candidates = duplicate_candidates[source, sample.name]
            if candidates:
                fingerprint = _sample_fingerprint(sample)
                if any(fingerprint == _sample_fingerprint(prior) for prior in candidates):
                    removed_copies += 1
                    continue
            candidates.append(sample)
            relative_path = sample.relative_to(data_dir).as_posix()
            labels = annotation_labels(json.loads(config_path.read_text()))
            source_hashes[relative_path] = source
            labels_by_sample[relative_path] = labels
            source_labels[source].update(label for _, _, label in labels)

    splits = group_source_images(
        source_hashes.items(), seed, val_fraction, test_fraction, source_labels=source_labels,
    )
    support = {}
    for split, paths in splits.items():
        counts = [0] * len(PSR_KR_CAT)
        labeled_pairs = 0
        for path in paths:
            labels = labels_by_sample[path]
            for _, _, label in labels:
                counts[label] += 1
            labeled_pairs += len({(first, second) for first, second, _ in labels})
        support[split] = dict(
            samples=len(paths),
            source_images=len({source_hashes[path] for path in paths}),
            labeled_pairs=labeled_pairs,
            labels=dict(zip(PSR_KR_CAT, counts)),
        )
    return dict(
        version=1, seed=seed, val_fraction=val_fraction, test_fraction=test_fraction,
        duplicates_removed=removed_copies, source_hashes=source_hashes, splits=splits, support=support,
    )


def load_sample(sample_dir, image_size, training=False):
    """Apply the same affine transform to the RGB image and every raw mask."""
    import cv2
    from datasets.utils.augimg import generate_mask_colors
    from utils.image import get_affine_transform

    sample_dir = Path(sample_dir)
    with Image.open(sample_dir / "src_img.png") as image:
        rgb = np.asarray(image.convert("RGB"))
    height, width = rgb.shape[:2]
    extent = max(height, width)
    flip = training and np.random.random() < 0.5
    if training:
        extent *= np.random.choice((1.0, 1.1, 1.2, 1.3))
    size = (image_size, image_size)
    affine = get_affine_transform(np.array([width / 2, height / 2]), float(extent), 0, size)

    def warp(array, interpolation):
        return cv2.warpAffine(array[:, ::-1] if flip else array, affine, size, flags=interpolation)

    rgb = warp(rgb, cv2.INTER_LINEAR).astype(np.float32) / 255
    rgb -= np.asarray(PSR_MEAN, dtype=np.float32)
    rgb /= np.asarray(PSR_STD, dtype=np.float32)
    mask_paths = {}
    for path in sample_dir.iterdir():
        if match := _MASK_FILE.fullmatch(path.name):
            mask_id = int(match.group(1))
            if mask_id in mask_paths:
                raise ValueError(f"Multiple files for mask{mask_id} in {sample_dir}")
            mask_paths[mask_id] = path

    mask_ids = tuple(sorted(mask_paths))
    masks = np.zeros((len(mask_ids), image_size, image_size), dtype=bool)
    mcm = np.zeros((image_size, image_size, 3), dtype=np.float32)
    covered = np.zeros((image_size, image_size), dtype=bool)
    colors = generate_mask_colors(len(mask_ids), len(mask_ids))
    for index, mask_id in enumerate(mask_ids):
        with Image.open(mask_paths[mask_id]) as image:
            binary = np.asarray(image.convert("L").resize((width, height), Image.Resampling.NEAREST)) > 0
        mask = warp(binary.astype(np.uint8), cv2.INTER_NEAREST).astype(bool)
        masks[index] = mask
        mcm[mask & ~covered] = colors[index + 1]
        covered |= mask

    image = np.concatenate((rgb, mcm), axis=2).transpose(2, 0, 1).copy()
    return {"image": torch.from_numpy(image), "masks": torch.from_numpy(masks), "mask_ids": mask_ids}


class PSRPairs(Dataset):
    def __init__(self, data_dir, samples, image_size, training=False, functions=False):
        self.data_dir = Path(data_dir)
        self.samples = samples
        self.image_size = image_size
        self.training = training
        self.functions = functions

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path = self.data_dir / self.samples[index]
        sample = load_sample(path, self.image_size, self.training)
        annotation = json.loads((path / "config.json").read_text())
        try:
            sample["targets"], sample["labeled"] = build_targets(annotation, sample["mask_ids"])
            if self.functions:
                sample["function_targets"] = build_function_targets(annotation, sample["mask_ids"])
        except ValueError as error:
            raise ValueError(f"{path}: {error}") from error
        return sample


def collate_pairs(samples):
    images = torch.stack([sample["image"] for sample in samples])
    masks = torch.zeros(len(samples), max(len(sample["masks"]) for sample in samples),
                        *images.shape[-2:], dtype=torch.bool)
    valid = torch.zeros(masks.shape[:2], dtype=torch.bool)
    for index, sample in enumerate(samples):
        parts = sample["masks"]
        masks[index, :len(parts)] = parts
        valid[index, :len(parts)] = parts.flatten(1).any(dim=1)
    pairs = pair_indices(valid)
    batch = dict(images=images, masks=masks, pairs=pairs, mask_ids=[sample["mask_ids"] for sample in samples])
    batch["parts"] = valid.nonzero()
    if "function_targets" in samples[0]:
        targets = torch.zeros(*valid.shape, len(PSR_FUNC_CAT), dtype=torch.bool)
        for index, sample in enumerate(samples):
            labels = sample["function_targets"]
            targets[index, :len(labels)] = labels
        batch["function_targets"] = targets[valid]
        batch["function_unavailable"] = targets[~valid].sum(dim=0)
    if "targets" in samples[0]:
        batch["targets"] = torch.stack([samples[b]["targets"][i, j] for b, i, j in pairs]) if len(pairs) else (
            torch.empty(0, len(PSR_KR_CAT), dtype=torch.bool)
        )
        batch["labeled"] = torch.tensor([bool(samples[b]["labeled"][i, j]) for b, i, j in pairs], dtype=torch.bool)
        all_support = torch.stack([sample["targets"].sum(dim=(0, 1)) for sample in samples])
        all_support.index_add_(0, pairs[:, 0], -batch["targets"].long())
        batch["unavailable"] = all_support
    return batch
