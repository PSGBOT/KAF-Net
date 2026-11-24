import numpy as np
import pytest


def test_model_routes_use_exact_dsflash_names():
    from utils.model_routing import dsflash_command

    assert dsflash_command("train", ["--arch", "swin", "--data_dir", "samples"]) is None
    command = dsflash_command("train", ["--arch=dsflash_swin_functions", "--data_dir", "samples"])
    assert command[:4] == ["-m", "experiments.dsflash_psr", "train", "--data_dir"]
    assert command[-1].endswith("dsflash_swin_psr_functions.json")
    command = dsflash_command("infer", ["--arch", "dsflash_swin", "--model_weights", "best.pt"])
    assert command == ["-m", "experiments.dsflash_psr", "infer", "--model_weights", "best.pt"]
    command = dsflash_command("evaluate", ["--arch", "dsflash_swin_functions"])
    assert command[-1] == "--require-functions"
    with pytest.raises(SystemExit):
        dsflash_command("train", ["--arch", "dsflash_swin_typo"])


def test_checkpoint_interval_includes_explicit_final_epoch():
    from utils.eval.reporting import checkpoint_due

    assert [i for i in range(1, 13) if checkpoint_due(i, 12, 5)] == [5, 10, 12]
    assert [i for i in range(1, 7) if checkpoint_due(i, 6, 0)] == [6]


def test_tensorboard_records_split_cutoffs_and_per_class_metrics():
    from utils.eval.reporting import write_evaluation

    class Writer:
        def __init__(self):
            self.scalars = {}
            self.flushed = False

        def add_scalar(self, tag, value, epoch):
            self.scalars[tag] = (value, epoch)

        def flush(self):
            self.flushed = True

    writer = Writer()
    write_evaluation(writer, "val", {
        "macro_AP": 0.3, "cutoffs": [5, 10, 20], "R": [0.2, 0.4, 0.8], "mR": [0.1, 0.2, 0.3],
        "per_class": {"fixed": {"AP": 0.6, "R": [0.4, 0.5, 0.6], "support": 17}},
        "functions": {"macro_AP": 0.5, "per_class": {"handle": {"AP": 0.4, "support": 12}}},
    }, 5)
    assert writer.scalars["val/relations/R_10"] == (0.4, 5)
    assert writer.scalars["val/relations/per_class/fixed/R_5"] == (0.4, 5)
    assert writer.scalars["val/functions/per_class/handle/AP"] == (0.4, 5)
    assert writer.flushed


def test_missing_kaf_candidates_do_not_turn_into_true_positives():
    from experiments.dsflash_metrics import relation_metrics

    logits = np.full((1, 14), -np.inf)
    targets = np.zeros_like(logits, dtype=bool)
    targets[0, 1] = True
    result = relation_metrics([(logits, targets, np.zeros(14, dtype=int), np.ones(1, dtype=bool))])
    assert result["macro_AP"] == 0
    assert result["R"] == [0.0, 0.0, 0.0]


def test_kaf_graph_evaluation_resolves_proposals_to_input_mask_ids():
    from utils.eval.kaf import score_sample

    masks = {0: {1: 1}, 1: {4: 1}, 2: {11: 1}}
    sample = {"masks_cat": masks, "relations": [(2, 0, 3), (0, 1, 1)]}
    graph = {
        "objects": [{"id": 0, "all_classes": [[0.8, 1]]}, {"id": 2, "all_classes": [[0.7, 11]]}],
        "relations": [{"subject_id": 1, "object_id": 0, "relation": 3, "confidence": 0.9}],
    }
    (edge_scores, edge_labels, *_), (part_scores, part_labels, _) = score_sample(sample, graph)
    assert part_scores[2, 11] == 0.7 and part_labels[2, 11]
    assert np.isneginf(part_scores[1, 4]) and part_labels[1, 4]
    assert edge_scores[1, 3] == 0.9 and edge_labels[1, 3]
    assert np.isneginf(edge_scores[0, 1]) and edge_labels[0, 1]


def test_kaf_field_layout_preserves_fpn_heads_and_uses_the_last_stack():
    import torch
    from utils.eval.kaf import kaf_fields

    stack = [torch.zeros(2, channels, 8, 8) for channels in (13, 2, 2, 28)]
    fpn = [[field] * 4 for field in stack]
    assert kaf_fields(fpn) is fpn
    assert all(levels[0] is field for levels, field in zip(kaf_fields([stack, stack]), stack))


def test_kaf_decoder_assigns_proposals_once_and_keeps_sparse_mask_ids():
    import torch
    from get_scene import get_scene_graph

    heatmap = torch.full((1, 13, 8, 8), -20.0)
    heatmap[0, 1, 2, 2], heatmap[0, 4, 2, 2] = 6, 5
    regs = torch.zeros(1, 2, 8, 8)
    boxes = {7: {"center": (10, 8), "scale": (4, 4)}, 2: {"center": (8, 8), "scale": (4, 4)}}
    graph = get_scene_graph([heatmap], [regs], [regs + 4], [], boxes)
    assert [entry["id"] for entry in graph["objects"]] == [2]
    assert {category for _, category in graph["objects"][0]["all_classes"]} == {1, 4}


def test_numbered_checkpoints_keep_earlier_best_and_repoint_latest(tmp_path):
    import torch
    from utils.summary import link_checkpoint, publish_checkpoint

    first = tmp_path / "epoch-005.pt"
    publish_checkpoint({"epoch": 5}, first, aliases=("latest.pt", "best.pt"))
    publish_checkpoint({"epoch": 10}, tmp_path / "epoch-010.pt", aliases=("latest.pt",))
    assert first.samefile(tmp_path / "best.pt")
    link_checkpoint(first, tmp_path / "best.pt")
    assert not list(tmp_path.glob("*.tmp"))
    assert torch.load(tmp_path / "latest.pt") == {"epoch": 10}
    assert torch.load(tmp_path / "best.pt") == {"epoch": 5}
