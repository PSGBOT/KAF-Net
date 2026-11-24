# DSFlash relations on PSR

This configuration reuses the KAF-Net SwinV2 and FPN, replacing the dense relation field with patch coverage embeddings, a pair-conditioned Transformer, and DSFlash's complementary gate. The head predicts one 14-label score vector for each unordered pair of supplied masks. Training supervises both mask orders and adds a feature consistency loss. Evaluation uses a single pass in canonical mask ID order. The KAF backbone still sees its RGB and MCM inputs. The relation head uses the independent masks, including overlaps. [DSFlash §3](https://arxiv.org/html/2603.10538v1#S3)

## Prepare and run

Point `dataset` to the existing PSR_v2 directory. Set `run_dir` to a new experiment directory and `source_dir` to this checkout. On labserver0 the launcher uses the existing `kafnet` environment and cached ImageNet Swin weights.

```bash
config="$source_dir/experiments/dsflash_swin_psr.json"
manifest="$run_dir/split.json"
python -m experiments.dsflash_psr prepare --config "$config" --data-dir "$dataset" --manifest "$manifest"
sbatch --open-mode=append --output="$run_dir/train.log" experiments/dsflash_psr.sbatch \
    "$source_dir" "$run_dir" "$dataset" "$manifest" "$config"
```

The split groups samples by the SHA-256 of `src_img.png`, removes identical copies with the same sample name, and prioritizes coverage of rare labels across disjoint groups. Append `--resume` to the launcher for an interrupted run.

## Model routing and evaluation

Use the same `train.py`, `evaluate.py`, and `infer.py` entry points for all three Swin models.

| `--arch` | Prediction | Weight format |
| --- | --- | --- |
| `swin` | Original KAF-Swin scene graph | KAF `.pth` |
| `dsflash_swin` | DSFlash relations | DSFlash `.pt` |
| `dsflash_swin_functions` | DSFlash relations and part functions | DSFlash `.pt` with the function head |

```bash
python train.py --arch dsflash_swin --data-dir "$dataset" --manifest "$manifest" --run-dir "$run_dir"
python evaluate.py --arch dsflash_swin --data-dir "$dataset" --manifest "$manifest" \
    --checkpoint "$run_dir/best.pt" --output "$run_dir/standalone-test.json"
python evaluate.py --arch swin --data-dir "$dataset" --manifest "$manifest" \
    --checkpoint "$kaf_checkpoint" --output "$kaf_report"
```

KAF training and evaluation also accept `--manifest` for the grouped split. For KAF without a manifest, point `--data-dir` to a directory containing the existing split folders. Both evaluators default to the test split and accept `--split val`. TensorBoard event files for standalone evaluation go in `tensorboard/` alongside the JSON report. Set `--tensorboard-dir` to use another location.

Training saves a numbered checkpoint and evaluates the complete validation split every five epochs and at the final epoch. Set `checkpoint_interval` in the DSFlash JSON or `--val_interval` for KAF to change the interval. DSFlash also saves at an explicitly requested `--stop-after-epoch`. DSFlash keeps `latest.pt` and `best.pt` in the run directory. KAF keeps `checkpoint.pth` and `best.pth` in its checkpoint directory. The default selection metric is validation relation macro AP, with function macro AP used for the frozen DSFlash function-head run.

The final test evaluates the best validation checkpoint on the held-out split. Legacy KAF datasets with a `test/` partition use that partition. KAF reports use the full annotations. The KAF `--prune` setting applies to training and validation loss. Each evaluation saves a JSON report and writes aggregate metrics, class scores and support to TensorBoard. The training environment needs `tensorboardX` or `tensorboard` to write events. Install `tensorboard` to run the dashboard. Use `$run_dir/tensorboard` when training with `--run-dir`, or `logs/<log_name>` for legacy KAF runs.

```bash
tensorboard --logdir "$run_dir/tensorboard"
```

## Predict from supplied masks

Use a directory with `src_img.png` and numbered `mask<ID>.png` files. An annotation file is optional for inference. The JSON output contains per-label probabilities for each valid unordered pair of mask IDs.

```bash
python infer.py --arch dsflash_swin --checkpoint "$run_dir/best.pt" \
    --sample-dir "$sample_dir" --output "$run_dir/prediction.json"
```

Validation and test report relation macro AP, R@5/10/20, mR@5/10/20 and per-class support. AP uses annotated pairs, and recall ranks scored predictions per image. KAF associates its proposals with input masks before evaluation. Labels for unavailable masks or missing predictions remain in the recall and AP denominators. Mask-function AP uses annotated masks and is reported under `functions`. Use the same grouped manifest for a KAF and DSFlash comparison.

## Train a part-function head

Use the paired dataset's existing manifest and a relation-only checkpoint trained on that same manifest. Training freezes SwinV2, the FPN, and the relationship head, including their running statistics. The function head pools P2 and P3 with each independent mask's patch coverage in FP32. It predicts 13 sigmoid scores and is selected by validation function macro AP.

```bash
sbatch --open-mode=append --output="$function_run/train.log" experiments/dsflash_psr.sbatch \
    "$source_dir" "$function_run" "$dataset" "$manifest" \
    "$source_dir/experiments/dsflash_swin_psr_functions.json" --init-from "$relation_run/best.pt"
```

On resume, replace `--init-from ...` with `--resume`. Evaluation adds mask-level AP and class supports under `functions`. Inference adds `parts`, one entry per valid mask ID with 13 `function_scores`. A single mask produces a function prediction and an empty relation list.
