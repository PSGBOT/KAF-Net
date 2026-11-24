"""PSR relation metrics using exact unordered mask-pair identities."""

import numpy as np

from datasets.psr_labels import PSR_KR_CAT

CUTOFFS = (5, 10, 20)


def _average_precision(scores, labels, support):
    if not support:
        return None
    if not len(scores):
        return 0.0
    order = np.argsort(-scores, kind="stable")
    by_score = scores[order]
    last_at_score = np.r_[np.flatnonzero(np.diff(by_score)), len(scores) - 1]
    true_positives = np.cumsum(labels[order])[last_at_score]
    precision = true_positives / (last_at_score + 1)
    recall_increments = np.diff(true_positives, prepend=0) / support
    return float(precision @ recall_increments)


def relation_metrics(samples):
    """Rank all valid pairs for recall and use annotated pairs for precision."""
    classes = len(PSR_KR_CAT)
    support, unavailable = np.zeros(classes, dtype=int), np.zeros(classes, dtype=int)
    hits = np.zeros((len(CUTOFFS), classes), dtype=int)
    all_scores, all_labels = [], []
    candidate_count = 0

    for scores, targets, missed, labeled in samples:
        support += targets.sum(axis=0) + missed
        unavailable += missed
        candidate_count += len(scores)
        all_scores.append(scores[labeled])
        all_labels.append(targets[labeled])
        flat_ranks = np.argsort(-scores.ravel(), kind="stable")[:max(CUTOFFS)]
        for cutoff_index, cutoff in enumerate(CUTOFFS):
            selected = flat_ranks[:cutoff]
            hit_labels = selected[targets.ravel()[selected]] % classes
            hits[cutoff_index] += np.bincount(hit_labels, minlength=classes)

    scores = np.concatenate(all_scores) if all_scores else np.empty((0, classes))
    labels = np.concatenate(all_labels) if all_labels else np.empty((0, classes), dtype=bool)
    aps = [_average_precision(scores[:, c], labels[:, c], support[c]) for c in range(classes)]
    observed = support > 0
    recalls = hits[:, observed] / support[observed]
    return {
        "samples": len(samples),
        "scored_pairs": candidate_count,
        "annotated_scored_pairs": len(scores),
        "cutoffs": CUTOFFS,
        "macro_AP": float(np.mean([ap for ap in aps if ap is not None])) if observed.any() else None,
        "R": (hits.sum(axis=1) / support.sum()).tolist() if observed.any() else None,
        "mR": recalls.mean(axis=1).tolist() if observed.any() else None,
        "per_class": {
            label: dict(support=int(support[c]), unavailable=int(unavailable[c]), AP=aps[c],
                        R=(hits[:, c] / support[c]).tolist() if support[c] else None)
            for c, label in enumerate(PSR_KR_CAT)
        },
    }
