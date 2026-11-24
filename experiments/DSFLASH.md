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

The split groups samples by the SHA-256 of `src_img.png`, removes identical copies with the same sample name, and prioritizes coverage of rare labels across disjoint groups. Training saves `latest.pt` and `best.pt` and selects the latter by validation macro AP. Append `--resume` to the launcher for an interrupted run. The best checkpoint is evaluated on test after the configured final epoch.

## Predict from supplied masks

Use a directory with `src_img.png` and numbered `mask<ID>.png` files. An annotation file is optional for inference. The JSON output contains per-label probabilities for each valid unordered pair of mask IDs.

```bash
python -m experiments.dsflash_psr infer --checkpoint "$run_dir/best.pt" \
    --sample-dir "$sample_dir" --output "$run_dir/prediction.json"
```

Validation and test use exact mask-pair identity and report relation macro AP, R@5/10/20, mR@5/10/20 and the support for each relation class. AP uses annotated pairs. Recall ranks every valid pair. Use the same grouped manifest for any new KAF comparison.
