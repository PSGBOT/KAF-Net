import torch

from nets.kaf.dsflash import DSFlashHead
from datasets.psr_pairs import build_targets, group_source_images, pair_indices


def test_pairs_only_join_valid_masks_from_the_same_image():
    valid = torch.tensor([[True, False, True], [True, False, False], [True, True, True]])
    assert pair_indices(valid).tolist() == [[0, 0, 2], [2, 0, 1], [2, 0, 2], [2, 1, 2]]


def test_dsflash_reuses_features_and_trains_on_overlapping_masks():
    torch.manual_seed(317)
    head = DSFlashHead(num_relations=14, feature_channels=8, width=16, depth=1, heads=4, patch_grid=4)
    features = torch.randn(2, 8, 4, 4, requires_grad=True)
    masks = torch.zeros(2, 3, 16, 16, dtype=torch.bool)
    masks[:, 0, :10, :10] = True
    masks[:, 1, 6:14, 6:14] = True
    masks[1, 2, 3:8, 1:5] = True
    pairs = pair_indices(masks.flatten(2).any(dim=2))
    logits, swapped, consistency = head(features, masks, pairs)
    assert swapped.shape == (4, 14)
    assert logits.shape == (4, 14)
    assert consistency.shape == (4,)
    (logits.square().mean() + consistency.mean()).backward()
    assert features.grad is not None and torch.isfinite(features.grad).all()
    assert head.gate[0].weight.grad.abs().sum() > 0
    assert head.mask_role_embedding.grad.abs().sum() > 0

    head.eval()
    with torch.no_grad():
        assert head(features, masks, pairs)[0].shape == (4, 14)
        assert head(features, masks, pairs[:0])[0].shape == (0, 14)


def test_pair_targets_preserve_label_multiplicity_and_unknown_pairs():
    annotation = {"kinematic relation": [
        ["mask8", "mask2", {"kinematic_joints": [
            {"joint_type": "revolute", "controllable": "controlled", "root": 0},
            {"joint_type": "unrelated", "root": 1},
        ]}],
        ["mask2", "mask8", {"kinematic_joints": [{"joint_type": "unrelated", "root": 0}]}],
    ]}
    targets, labeled = build_targets(annotation, (2, 5, 8))
    assert labeled[0, 2]
    assert targets[0, 2].nonzero().flatten().tolist() == [3, 13]
    assert not labeled[0, 1]


def test_images_with_the_same_digest_cannot_cross_splits():
    records = [
        ("train/one", "digest-1"), ("val/two", "digest-1"),
        *((f"train/{i}", f"digest-{i}") for i in range(3, 23)),
    ]
    split = group_source_images(records, seed=317, val_fraction=0.1, test_fraction=0.1)
    flattened = [path for paths in split.values() for path in paths]
    assert sorted(flattened) == sorted(path for path, _ in records)
    assert all(split.values())
    assert any({"train/one", "val/two"} <= set(paths) for paths in split.values())
    assert split == group_source_images(list(reversed(records)), seed=317, val_fraction=0.1, test_fraction=0.1)


def test_collate_preserves_ids_and_excludes_empty_masks():
    from datasets.psr_pairs import collate_pairs

    masks = torch.zeros(3, 8, 8, dtype=torch.bool)
    masks[0, :5, :5] = True
    masks[2, 3:, 3:] = True
    batch = collate_pairs([
        dict(image=torch.zeros(6, 8, 8), masks=masks, mask_ids=(1, 3, 7)),
        dict(image=torch.zeros(6, 8, 8), masks=masks[:1], mask_ids=(9,)),
    ])
    assert batch["pairs"].tolist() == [[0, 0, 2]]
    assert batch["mask_ids"] == [(1, 3, 7), (9,)]
    assert (batch["masks"][0, 0] & batch["masks"][0, 2]).sum() == 4


def test_relation_metrics_count_unavailable_labels_in_recall_and_ap():
    import numpy as np
    from experiments.dsflash_metrics import relation_metrics

    scores = np.zeros((2, 14))
    scores[:, 1] = [1.0, 0.5]
    targets = np.zeros((2, 14), dtype=bool)
    targets[0, 1] = True
    missed = np.zeros(14, dtype=int)
    missed[1] = 1
    result = relation_metrics([(scores, targets, missed, np.array([True, True]))])
    assert result["R"] == [0.5, 0.5, 0.5]
    assert result["macro_AP"] == 0.5
    assert result["per_class"]["fixed"]["support"] == 2
    assert result["per_class"]["unknown"]["AP"] is None


def test_grouped_split_prioritizes_support_for_rare_labels():
    records = [(f"train/{i}", f"digest-{i}") for i in range(30)]
    labels = {"digest-0": {0, 1}, "digest-1": {0, 1}, "digest-2": {1}}
    split = group_source_images(records, 317, 0.1, 0.1, source_labels=labels)
    assert "train/2" in split["val"]
    assert any(path in split["train"] for path in ["train/0", "train/1"])
    assert any(path in split["test"] for path in ["train/0", "train/1"])
    assert [len(split[name]) for name in ("train", "val", "test")] == [24, 3, 3]


def test_train_and_inference_use_the_same_primary_pair_ordering():
    head = DSFlashHead(num_relations=14, feature_channels=8, width=16, depth=1, heads=4, patch_grid=2)
    features = torch.randn(1, 8, 4, 4)
    masks = torch.zeros(1, 2, 8, 8, dtype=torch.bool)
    masks[0, 0, :5, :5] = True
    masks[0, 1, 3:, 3:] = True
    pairs = torch.tensor([[0, 0, 1]])
    training = head(features, masks, pairs)
    swapped = head(features, masks, pairs[:, [0, 2, 1]])
    torch.testing.assert_close(training.logits, swapped.swapped_logits)
    torch.testing.assert_close(training.consistency, swapped.consistency)
    head.eval()
    with torch.no_grad():
        prediction = head(features, masks, pairs)
    torch.testing.assert_close(training.logits, prediction.logits)
    assert prediction.swapped_logits is None


def test_unannotated_candidates_can_affect_recall_but_do_not_affect_ap():
    import numpy as np
    from experiments.dsflash_metrics import relation_metrics

    scores = np.zeros((2, 14))
    scores[1, :5] = 2.0
    scores[0, 1] = 1.0
    targets = np.zeros((2, 14), dtype=bool)
    targets[0, 1] = True
    result = relation_metrics([(scores, targets, np.zeros(14, dtype=int), np.array([True, False]))])
    assert result["R"] == [0.0, 1.0, 1.0]
    assert result["macro_AP"] == 1.0


def test_checkpointed_pair_chunks_backpropagate():
    head = DSFlashHead(num_relations=14, feature_channels=8, width=16, depth=1, heads=4, patch_grid=2, pair_chunk=1)
    features = torch.randn(1, 8, 4, 4, requires_grad=True)
    masks = torch.ones(1, 3, 8, 8, dtype=torch.bool)
    output = head(features, masks, pair_indices(masks.flatten(2).any(dim=2)))
    output.logits.square().mean().backward()
    assert features.grad is not None and torch.isfinite(features.grad).all()
    assert head.classifier[0].weight.grad.abs().sum() > 0


def test_missing_annotated_mask_is_not_silently_dropped():
    import pytest

    annotation = {"kinematic relation": [["mask0", "mask1", {"kinematic_joints": [{"joint_type": "fixed"}]}]]}
    with pytest.raises(ValueError, match="missing mask"):
        build_targets(annotation, (0,))
