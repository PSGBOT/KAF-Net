"""Shared PSR checkpoint cadence and TensorBoard metric names."""

import json
from numbers import Real


def atomic_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def checkpoint_due(epoch, last_epoch, interval):
    return epoch == last_epoch or (interval > 0 and epoch % interval == 0)


def _write_scalars(writer, prefix, values, cutoffs, epoch):
    for key, value in values.items():
        if isinstance(value, Real):
            writer.add_scalar(f"{prefix}/{key}", value, epoch)
        elif key in {"R", "mR"} and value is not None:
            for k, score in zip(cutoffs, value):
                writer.add_scalar(f"{prefix}/{key}_{k}", score, epoch)
        elif isinstance(value, dict):
            _write_scalars(writer, f"{prefix}/{key}", value, cutoffs, epoch)


def write_evaluation(writer, split, metrics, epoch):
    cutoffs = metrics["cutoffs"]
    relations = {key: value for key, value in metrics.items() if key not in {"functions", "cutoffs"}}
    _write_scalars(writer, f"{split}/relations", relations, cutoffs, epoch)
    if "functions" in metrics:
        _write_scalars(writer, f"{split}/functions", metrics["functions"], (), epoch)
    writer.flush()
