"""Selector-anchor source cross-fit utilities for frozen target-drug-OOF scores."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from mc_tv_pu_q_tau_semantics_audit import source_sets_by_drug
from mc_tv_pu_selector_evaluation import (
    TAU_GRID,
    fraction_text,
    midrank_loss_numerators,
)

PRIMARY_BASELINES = (
    "rna_prior_only",
    "prior_preserving_pairwise_residual",
)
COMPATIBLE_SELECTION_RULE = "compatible"
SELECTION_RULE_ABLATIONS = ("matched_rectangular", "empirical_center")
METRICS = ("average_precision", "pairwise_auc", "recall_at_10", "recall_at_50")
BOOTSTRAP_ITERATIONS = 20_000
BOOTSTRAP_SEED = 20_261_021
CROSS_FIT_FOLD_COUNT = 5
CROSS_FIT_SEED = 20_261_021


@dataclass(frozen=True)
class SourceEpisode:
    """One leave-one-PMID-out episode for a single drug."""

    drug_id: int
    heldout_source: str
    anchor_sources: tuple[str, ...]
    anchor_rnas: tuple[int, ...]
    heldout_novel_rnas: tuple[int, ...]

    @property
    def evaluable(self) -> bool:
        return bool(self.heldout_novel_rnas)


@dataclass(frozen=True)
class SourceCrossFitEpisode:
    """One selector-anchor-source-disjoint episode for a drug and fold."""

    fold: int
    drug_id: int
    heldout_sources: tuple[str, ...]
    anchor_sources: tuple[str, ...]
    anchor_rnas: tuple[int, ...]
    heldout_novel_rnas: tuple[int, ...]

    @property
    def evaluable(self) -> bool:
        return bool(self.heldout_novel_rnas)


def assign_sources_to_folds(
    sources: Sequence[str],
    *,
    fold_count: int = CROSS_FIT_FOLD_COUNT,
    seed: int = CROSS_FIT_SEED,
) -> dict[str, int]:
    """Assign every PMID-like source to one deterministic global fold."""

    if type(fold_count) is not int or fold_count < 2:
        raise ValueError("fold_count must be an integer of at least two")
    if type(seed) is not int or isinstance(seed, bool):
        raise ValueError("seed must be an integer")
    source_values = tuple(str(source) for source in sources)
    if any(not source or source != source.strip() for source in source_values):
        raise ValueError("sources must be nonempty normalized strings")
    source_ids = tuple(sorted(source_values))
    if len(set(source_ids)) != len(source_ids):
        raise ValueError("sources must be unique")
    assignments: dict[str, int] = {}
    for source in source_ids:
        digest = hashlib.sha256(f"{seed}\0{source}".encode()).digest()
        assignments[source] = int.from_bytes(digest[:8], "big") % fold_count
    return assignments


def build_source_crossfit_episodes(
    pair_sources: Mapping[tuple[int, int], set[str]],
    *,
    fold_count: int = CROSS_FIT_FOLD_COUNT,
    seed: int = CROSS_FIT_SEED,
) -> tuple[dict[str, int], dict[int, tuple[SourceCrossFitEpisode, ...]]]:
    """Create folds where no held-out source appears in a same-fold selector anchor."""

    sources_by_drug = source_sets_by_drug(dict(pair_sources))
    all_sources = sorted(
        {source for sources in sources_by_drug.values() for source in sources}
    )
    assignments = assign_sources_to_folds(
        all_sources, fold_count=fold_count, seed=seed
    )
    episodes_by_fold: dict[int, tuple[SourceCrossFitEpisode, ...]] = {}
    for fold in range(fold_count):
        fold_episodes: list[SourceCrossFitEpisode] = []
        for drug_id, sources in sorted(sources_by_drug.items()):
            heldout_sources = tuple(
                sorted(source for source in sources if assignments[source] == fold)
            )
            anchor_sources = tuple(
                sorted(source for source in sources if assignments[source] != fold)
            )
            if not heldout_sources or not anchor_sources:
                continue
            anchor_rnas = set().union(*(sources[source] for source in anchor_sources))
            heldout_rnas = set().union(*(sources[source] for source in heldout_sources))
            fold_episodes.append(
                SourceCrossFitEpisode(
                    fold=fold,
                    drug_id=int(drug_id),
                    heldout_sources=heldout_sources,
                    anchor_sources=anchor_sources,
                    anchor_rnas=tuple(sorted(anchor_rnas)),
                    heldout_novel_rnas=tuple(sorted(heldout_rnas - anchor_rnas)),
                )
            )
        episodes_by_fold[fold] = tuple(fold_episodes)

    for fold, episodes in episodes_by_fold.items():
        heldout = {
            source
            for source, assigned_fold in assignments.items()
            if assigned_fold == fold
        }
        anchors = {source for episode in episodes for source in episode.anchor_sources}
        if heldout & anchors:
            raise AssertionError("held-out source leaked into a same-fold anchor")
    return assignments, episodes_by_fold


def source_crossfit_coverage_summary(
    episodes_by_fold: Mapping[int, Sequence[SourceCrossFitEpisode]],
) -> dict[str, int | bool | str]:
    """Apply the amended publication coverage gate without score access."""

    all_episodes = [episode for episodes in episodes_by_fold.values() for episode in episodes]
    evaluable = [episode for episode in all_episodes if episode.evaluable]
    evaluable_drugs = {episode.drug_id for episode in evaluable}
    evaluable_folds = {episode.fold for episode in evaluable}
    heldout_records = sum(len(episode.heldout_novel_rnas) for episode in evaluable)
    passed = (
        len(evaluable_drugs) >= 100
        and len(evaluable) >= 200
        and heldout_records >= 500
        and len(evaluable_folds) >= 4
    )
    return {
        "eligible_drug_fold_count": len(all_episodes),
        "evaluable_drug_count": len(evaluable_drugs),
        "evaluable_drug_fold_count": len(evaluable),
        "empty_drug_fold_count": len(all_episodes) - len(evaluable),
        "heldout_record_count": heldout_records,
        "evaluable_fold_count": len(evaluable_folds),
        "publication_coverage_passed": passed,
        "validation_tier": "publication" if passed else "pilot",
    }


def build_source_episodes(
    pair_sources: Mapping[tuple[int, int], set[str]],
) -> dict[int, tuple[SourceEpisode, ...]]:
    """Create deterministic leave-one-source-out episodes without score access."""

    episodes: dict[int, tuple[SourceEpisode, ...]] = {}
    for drug_id, sources in sorted(source_sets_by_drug(dict(pair_sources)).items()):
        if len(sources) < 2:
            continue
        source_ids = sorted(sources)
        drug_episodes: list[SourceEpisode] = []
        for heldout in source_ids:
            anchor_sources = tuple(source for source in source_ids if source != heldout)
            anchor_rnas = set().union(*(sources[source] for source in anchor_sources))
            heldout_novel = sources[heldout] - anchor_rnas
            drug_episodes.append(
                SourceEpisode(
                    drug_id=int(drug_id),
                    heldout_source=heldout,
                    anchor_sources=anchor_sources,
                    anchor_rnas=tuple(sorted(anchor_rnas)),
                    heldout_novel_rnas=tuple(sorted(heldout_novel)),
                )
            )
        episodes[int(drug_id)] = tuple(drug_episodes)
    return episodes


def episode_coverage_summary(
    episodes_by_drug: Mapping[int, Sequence[SourceEpisode]],
) -> dict[str, int | bool | str]:
    """Apply the frozen publication coverage gate before reading performance."""

    all_episodes = [episode for episodes in episodes_by_drug.values() for episode in episodes]
    evaluable = [episode for episode in all_episodes if episode.evaluable]
    evaluable_drugs = {episode.drug_id for episode in evaluable}
    heldout_records = sum(len(episode.heldout_novel_rnas) for episode in evaluable)
    passed = (
        len(evaluable_drugs) >= 100
        and len(evaluable) >= 200
        and heldout_records >= 500
    )
    return {
        "multi_source_drug_count": len(episodes_by_drug),
        "episode_count": len(all_episodes),
        "evaluable_drug_count": len(evaluable_drugs),
        "evaluable_episode_count": len(evaluable),
        "empty_heldout_episode_count": len(all_episodes) - len(evaluable),
        "heldout_record_count": heldout_records,
        "publication_coverage_passed": passed,
        "validation_tier": "publication" if passed else "pilot",
    }


def expected_recall_at_k(
    labels: Sequence[int] | np.ndarray,
    scores: Sequence[float] | np.ndarray,
    k: int,
) -> tuple[float, int]:
    """Recall@K with fractional allocation for a tie crossing the K boundary."""

    label_array = np.asarray(labels, dtype=np.int8)
    score_array = np.asarray(scores, dtype=np.float64)
    if label_array.ndim != 1 or score_array.ndim != 1:
        raise ValueError("labels and scores must be one-dimensional")
    if len(label_array) == 0 or len(label_array) != len(score_array):
        raise ValueError("labels and scores must have equal nonzero length")
    if not np.isin(label_array, (0, 1)).all() or not np.isfinite(score_array).all():
        raise ValueError("labels must be binary and scores must be finite")
    positive_count = int(label_array.sum())
    if positive_count <= 0:
        raise ValueError("recall requires at least one held-out positive")
    if not isinstance(k, int) or isinstance(k, bool) or k <= 0:
        raise ValueError("k must be a positive integer")
    actual_k = min(k, len(score_array))
    threshold = np.partition(score_array, len(score_array) - actual_k)[
        len(score_array) - actual_k
    ]
    above = score_array > threshold
    tied = score_array == threshold
    remaining_slots = actual_k - int(above.sum())
    expected_hits = float(label_array[above].sum())
    expected_hits += (
        remaining_slots / int(tied.sum()) * float(label_array[tied].sum())
    )
    return expected_hits / positive_count, actual_k


def tie_aware_pairwise_auc(
    labels: Sequence[int] | np.ndarray,
    scores: Sequence[float] | np.ndarray,
) -> float:
    """Return AUC using the selector's exact 0/0.5/1 midrank tie rule.

    For ``P`` positives and ``N`` unlabeled candidates, the sum of selector
    midrank numerators over positives is ``P^2 + 2PN(1-AUC)``.  This computes
    the objective-aligned empirical AUC without a quadratic pair expansion.
    """

    label_array = np.asarray(labels, dtype=np.int8)
    score_array = np.asarray(scores, dtype=np.float64)
    if label_array.ndim != 1 or score_array.ndim != 1:
        raise ValueError("labels and scores must be one-dimensional")
    if len(label_array) == 0 or len(label_array) != len(score_array):
        raise ValueError("labels and scores must have equal nonzero length")
    if not np.isin(label_array, (0, 1)).all() or not np.isfinite(score_array).all():
        raise ValueError("labels must be binary and scores must be finite")
    positive_count = int(label_array.sum())
    negative_count = len(label_array) - positive_count
    if positive_count <= 0 or negative_count <= 0:
        raise ValueError("pairwise AUC requires positive and unlabeled candidates")
    numerator_sum = int(midrank_loss_numerators(score_array)[label_array == 1].sum())
    pairwise_error_numerator = numerator_sum - positive_count**2
    pairwise_error_denominator = 2 * positive_count * negative_count
    if not 0 <= pairwise_error_numerator <= pairwise_error_denominator:
        raise AssertionError("derived pairwise error must lie in [0, 1]")
    return 1.0 - pairwise_error_numerator / pairwise_error_denominator


def evaluate_episode_scores(
    episode: SourceEpisode,
    *,
    scores: np.ndarray,
    candidate_ids: Sequence[str],
) -> pd.DataFrame:
    """Evaluate every frozen ranker on one anchor-isolated held-out source."""

    values = np.asarray(scores, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] != len(candidate_ids):
        raise ValueError("scores must have shape (candidate, RNA)")
    if not np.isfinite(values).all():
        raise ValueError("scores must be finite")
    if len(set(candidate_ids)) != len(candidate_ids):
        raise ValueError("candidate ids must be unique")
    n_rna = values.shape[1]
    anchors = set(episode.anchor_rnas)
    heldout = set(episode.heldout_novel_rnas)
    if anchors & heldout:
        raise ValueError("held-out novel RNA set must exclude all anchors")
    if any(index < 0 or index >= n_rna for index in anchors | heldout):
        raise ValueError("episode contains an RNA outside the score universe")
    if not heldout:
        return pd.DataFrame(
            columns=(
                "drug_id",
                "heldout_source",
                "candidate",
                "candidate_count",
                "positive_count",
                "average_precision",
                "pairwise_auc",
                "recall_at_10",
                "recall_at_50",
                "actual_k_10",
                "actual_k_50",
            )
        )

    include = np.ones(n_rna, dtype=bool)
    if anchors:
        include[np.fromiter(sorted(anchors), dtype=np.int64)] = False
    labels = np.zeros(n_rna, dtype=np.int8)
    labels[np.fromiter(sorted(heldout), dtype=np.int64)] = 1
    labels = labels[include]
    rows: list[dict[str, object]] = []
    for model_index, candidate in enumerate(candidate_ids):
        candidate_scores = values[model_index, include]
        pairwise_auc = tie_aware_pairwise_auc(labels, candidate_scores)
        recall_10, actual_10 = expected_recall_at_k(labels, candidate_scores, 10)
        recall_50, actual_50 = expected_recall_at_k(labels, candidate_scores, 50)
        rows.append(
            {
                "drug_id": episode.drug_id,
                "heldout_source": episode.heldout_source,
                "candidate": candidate,
                "candidate_count": len(labels),
                "positive_count": int(labels.sum()),
                "average_precision": float(
                    average_precision_score(labels, candidate_scores)
                ),
                "pairwise_auc": pairwise_auc,
                "recall_at_10": recall_10,
                "recall_at_50": recall_50,
                "actual_k_10": actual_10,
                "actual_k_50": actual_50,
            }
        )
    return pd.DataFrame(rows)


def evaluate_crossfit_episode_scores(
    episode: SourceCrossFitEpisode,
    *,
    scores: np.ndarray,
    candidate_ids: Sequence[str],
) -> pd.DataFrame:
    """Evaluate one selector-anchor-source-disjoint drug--fold episode."""

    proxy = SourceEpisode(
        drug_id=episode.drug_id,
        heldout_source=f"fold={episode.fold}",
        anchor_sources=episode.anchor_sources,
        anchor_rnas=episode.anchor_rnas,
        heldout_novel_rnas=episode.heldout_novel_rnas,
    )
    result = evaluate_episode_scores(proxy, scores=scores, candidate_ids=candidate_ids)
    if result.empty:
        return pd.DataFrame(
            columns=("fold", *[column for column in result.columns if column != "heldout_source"])
        )
    result = result.drop(columns="heldout_source")
    result.insert(0, "fold", episode.fold)
    return result


def aggregate_episode_metrics(episode_metrics: pd.DataFrame) -> pd.DataFrame:
    """Average episodes within drug before returning drug-level candidate metrics."""

    required = {"drug_id", "heldout_source", "candidate", *METRICS}
    missing = required - set(episode_metrics.columns)
    if missing:
        raise ValueError(f"episode metrics lack columns: {sorted(missing)}")
    if episode_metrics.empty:
        raise ValueError("at least one evaluable episode is required")
    if episode_metrics.duplicated(["drug_id", "heldout_source", "candidate"]).any():
        raise ValueError("episode metrics contain duplicate candidate rows")
    return (
        episode_metrics.groupby(["drug_id", "candidate"], as_index=False, sort=True)[
            list(METRICS)
        ]
        .mean()
        .sort_values(["drug_id", "candidate"], kind="mergesort")
        .reset_index(drop=True)
    )


def _metric_matrix(
    drug_metrics: pd.DataFrame,
    *,
    metric: str,
    candidate_ids: Sequence[str],
) -> tuple[np.ndarray, np.ndarray]:
    pivot = drug_metrics.pivot(index="drug_id", columns="candidate", values=metric)
    missing = set(candidate_ids) - set(pivot.columns)
    extra = set(pivot.columns) - set(candidate_ids)
    if missing or extra:
        raise ValueError(
            f"drug metrics candidate mismatch: missing={sorted(missing)}, extra={sorted(extra)}"
        )
    pivot = pivot.loc[:, list(candidate_ids)]
    if pivot.isna().any().any():
        raise ValueError("every evaluable drug must have every frozen candidate metric")
    return pivot.index.to_numpy(dtype=np.int64), pivot.to_numpy(dtype=np.float64)


def bootstrap_set_valued_performance(
    drug_metrics: pd.DataFrame,
    *,
    selector_cells: Sequence[Mapping[str, object]],
    candidate_ids: Sequence[str],
    baselines: Sequence[str] = PRIMARY_BASELINES,
    iterations: int = BOOTSTRAP_ITERATIONS,
    seed: int = BOOTSTRAP_SEED,
    batch_size: int = 500,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Paired drug bootstrap for the frozen set-valued selector path."""

    ordered_candidates = tuple(candidate_ids)
    candidate_index = {candidate: index for index, candidate in enumerate(ordered_candidates)}
    if len(candidate_index) != len(ordered_candidates):
        raise ValueError("candidate_ids must be unique")
    if any(baseline not in candidate_index for baseline in baselines):
        raise ValueError("every baseline must belong to the frozen candidate library")
    if iterations <= 0 or batch_size <= 0:
        raise ValueError("iterations and batch_size must be positive")

    identities: list[tuple[str, str, tuple[str, ...], str]] = []
    for cell in selector_cells:
        center = str(cell["center"])
        tau = str(cell["tau"])
        selected = tuple(str(value) for value in cell["possibly_optimal"])
        if not selected or any(value not in candidate_index for value in selected):
            raise ValueError("selector cell has an invalid possibly_optimal set")
        for baseline in baselines:
            identities.append((center, tau, selected, baseline))

    matrices: dict[str, np.ndarray] = {}
    drug_ids: np.ndarray | None = None
    for metric in METRICS:
        current_drugs, matrix = _metric_matrix(
            drug_metrics, metric=metric, candidate_ids=ordered_candidates
        )
        if drug_ids is None:
            drug_ids = current_drugs
        elif not np.array_equal(drug_ids, current_drugs):
            raise AssertionError("metric pivots produced different drug orders")
        matrices[metric] = matrix
    assert drug_ids is not None
    drug_count = len(drug_ids)
    if drug_count == 0:
        raise ValueError("at least one drug is required")

    points: dict[str, np.ndarray] = {}
    bootstraps: dict[str, np.ndarray] = {}
    for metric, matrix in matrices.items():
        points[metric] = np.array(
            [
                min(
                    float(np.mean(matrix[:, candidate_index[candidate]]))
                    - float(np.mean(matrix[:, candidate_index[baseline]]))
                    for candidate in selected
                )
                for _, _, selected, baseline in identities
            ],
            dtype=np.float64,
        )
        bootstraps[metric] = np.empty((iterations, len(identities)), dtype=np.float64)

    rng = np.random.default_rng(seed)
    offset = 0
    while offset < iterations:
        size = min(batch_size, iterations - offset)
        sampled = rng.integers(0, drug_count, size=(size, drug_count))
        for metric, matrix in matrices.items():
            sampled_means = matrix[sampled].mean(axis=1)
            for column, (_, _, selected, baseline) in enumerate(identities):
                selected_values = sampled_means[
                    :, [candidate_index[candidate] for candidate in selected]
                ]
                baseline_values = sampled_means[:, candidate_index[baseline]]
                bootstraps[metric][offset : offset + size, column] = (
                    selected_values.min(axis=1) - baseline_values
                )
        offset += size

    ap_deviations = bootstraps["average_precision"] - points["average_precision"]
    simultaneous_shift = float(np.quantile(ap_deviations.min(axis=1), 0.05))
    rows: list[dict[str, object]] = []
    for column, (center, tau, selected, baseline) in enumerate(identities):
        row: dict[str, object] = {
            "center": center,
            "tau": tau,
            "baseline": baseline,
            "possibly_optimal": list(selected),
            "drug_count": drug_count,
            "average_precision_difference": points["average_precision"][column],
            "average_precision_simultaneous_lower_95": (
                points["average_precision"][column] + simultaneous_shift
            ),
        }
        for metric in ("pairwise_auc", "recall_at_10", "recall_at_50"):
            row[f"{metric}_difference"] = points[metric][column]
            lower, upper = np.quantile(
                bootstraps[metric][:, column], [0.025, 0.975]
            )
            row[f"{metric}_lower_95"] = float(lower)
            row[f"{metric}_upper_95"] = float(upper)
        rows.append(row)
    return pd.DataFrame(rows), {
        "bootstrap_unit": "drug",
        "iterations": iterations,
        "seed": seed,
        "primary_family_size": len(identities),
        "simultaneous_lower_shift": simultaneous_shift,
    }


def bootstrap_crossfit_set_valued_performance(
    episode_metrics: pd.DataFrame,
    *,
    selector_cells_by_fold: Mapping[int, Sequence[Mapping[str, object]]],
    candidate_ids: Sequence[str],
    baselines: Sequence[str] = PRIMARY_BASELINES,
    iterations: int = BOOTSTRAP_ITERATIONS,
    seed: int = BOOTSTRAP_SEED,
    batch_size: int = 500,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Drug bootstrap with a separately selected set for every source fold."""

    required = {"fold", "drug_id", "candidate", *METRICS}
    missing = required - set(episode_metrics.columns)
    if missing:
        raise ValueError(f"cross-fit metrics lack columns: {sorted(missing)}")
    if episode_metrics.empty:
        raise ValueError("at least one evaluable drug--fold is required")
    if episode_metrics.duplicated(["fold", "drug_id", "candidate"]).any():
        raise ValueError("cross-fit metrics contain duplicate candidate rows")
    ordered_candidates = tuple(candidate_ids)
    candidate_index = {candidate: index for index, candidate in enumerate(ordered_candidates)}
    if len(candidate_index) != len(ordered_candidates):
        raise ValueError("candidate_ids must be unique")
    if any(baseline not in candidate_index for baseline in baselines):
        raise ValueError("every baseline must belong to the frozen candidate library")
    if iterations <= 0 or batch_size <= 0:
        raise ValueError("iterations and batch_size must be positive")

    selections: dict[tuple[int, str, str], tuple[str, ...]] = {}
    cell_keys_by_fold: dict[int, set[tuple[str, str]]] = {}
    for fold, cells in selector_cells_by_fold.items():
        current: set[tuple[str, str]] = set()
        for cell in cells:
            center = str(cell["center"])
            tau = str(cell["tau"])
            selected = tuple(str(value) for value in cell["possibly_optimal"])
            if not selected or any(value not in candidate_index for value in selected):
                raise ValueError("cross-fit selector cell has an invalid set")
            key = (int(fold), center, tau)
            if key in selections:
                raise ValueError("cross-fit selector contains a duplicate cell")
            selections[key] = selected
            current.add((center, tau))
        cell_keys_by_fold[int(fold)] = current
    observed_folds = set(episode_metrics["fold"].astype(int))
    if observed_folds != set(cell_keys_by_fold):
        raise ValueError("metrics and selector summaries use different folds")
    if len(set(map(frozenset, cell_keys_by_fold.values()))) != 1:
        raise ValueError("source folds do not share the same center/tau cells")
    cell_keys = sorted(next(iter(cell_keys_by_fold.values())))
    identities = [
        (center, tau, baseline)
        for center, tau in cell_keys
        for baseline in baselines
    ]

    indexed = episode_metrics.set_index(["drug_id", "fold", "candidate"]).sort_index()
    episode_keys = sorted(
        {(int(drug), int(fold)) for drug, fold, _ in indexed.index}
    )
    drug_ids = np.array(sorted({drug for drug, _ in episode_keys}), dtype=np.int64)
    drug_index = {drug: index for index, drug in enumerate(drug_ids)}
    per_metric = {
        metric: np.zeros((len(drug_ids), len(identities)), dtype=np.float64)
        for metric in METRICS
    }
    fold_counts = np.zeros(len(drug_ids), dtype=np.int64)
    for drug, fold in episode_keys:
        try:
            block = indexed.xs((drug, fold), level=("drug_id", "fold"))
        except KeyError as error:
            raise ValueError("cross-fit metric block is missing") from error
        if set(block.index) != set(ordered_candidates):
            raise ValueError("every drug--fold must contain every candidate")
        row = drug_index[drug]
        fold_counts[row] += 1
        for column, (center, tau, baseline) in enumerate(identities):
            selected = selections[(fold, center, tau)]
            for metric in METRICS:
                selected_value = min(float(block.loc[candidate, metric]) for candidate in selected)
                per_metric[metric][row, column] += selected_value - float(
                    block.loc[baseline, metric]
                )
    if (fold_counts <= 0).any():
        raise AssertionError("every bootstrap drug must have an evaluable fold")
    for matrix in per_metric.values():
        matrix /= fold_counts[:, None]

    points = {metric: matrix.mean(axis=0) for metric, matrix in per_metric.items()}
    bootstraps = {
        metric: np.empty((iterations, len(identities)), dtype=np.float64)
        for metric in METRICS
    }
    rng = np.random.default_rng(seed)
    offset = 0
    while offset < iterations:
        size = min(batch_size, iterations - offset)
        sampled = rng.integers(0, len(drug_ids), size=(size, len(drug_ids)))
        for metric, matrix in per_metric.items():
            bootstraps[metric][offset : offset + size] = matrix[sampled].mean(axis=1)
        offset += size

    deviations = bootstraps["average_precision"] - points["average_precision"]
    simultaneous_shift = float(np.quantile(deviations.min(axis=1), 0.05))
    rows: list[dict[str, object]] = []
    for column, (center, tau, baseline) in enumerate(identities):
        row: dict[str, object] = {
            "center": center,
            "tau": tau,
            "baseline": baseline,
            "drug_count": len(drug_ids),
            "average_precision_difference": points["average_precision"][column],
            "average_precision_simultaneous_lower_95": (
                points["average_precision"][column] + simultaneous_shift
            ),
        }
        for metric in ("pairwise_auc", "recall_at_10", "recall_at_50"):
            row[f"{metric}_difference"] = points[metric][column]
            lower, upper = np.quantile(bootstraps[metric][:, column], [0.025, 0.975])
            row[f"{metric}_lower_95"] = float(lower)
            row[f"{metric}_upper_95"] = float(upper)
        rows.append(row)
    return pd.DataFrame(rows), {
        "bootstrap_unit": "drug",
        "fold_handling": "worst_selected_within_fold_then_mean_folds_within_drug",
        "iterations": iterations,
        "seed": seed,
        "primary_family_size": len(identities),
        "simultaneous_lower_shift": simultaneous_shift,
    }


def _selection_rule_candidates(
    cell: Mapping[str, object], rule: str
) -> tuple[str, ...]:
    if rule == COMPATIBLE_SELECTION_RULE:
        raw = cell.get("possibly_optimal")
    else:
        section = cell.get(rule)
        if not isinstance(section, Mapping):
            raise ValueError(f"selector cell lacks selection rule {rule!r}")
        raw = section.get("optimal")
    if isinstance(raw, (str, bytes)) or not isinstance(raw, Sequence):
        raise ValueError(f"selector rule {rule!r} must provide a candidate sequence")
    candidates = tuple(str(candidate) for candidate in raw)
    if not candidates or len(set(candidates)) != len(candidates):
        raise ValueError(f"selector rule {rule!r} has an invalid candidate set")
    return candidates


def bootstrap_crossfit_rule_ablation_performance(
    episode_metrics: pd.DataFrame,
    *,
    selector_cells_by_fold: Mapping[int, Sequence[Mapping[str, object]]],
    candidate_ids: Sequence[str],
    comparison_rules: Sequence[str] = SELECTION_RULE_ABLATIONS,
    iterations: int = BOOTSTRAP_ITERATIONS,
    seed: int = BOOTSTRAP_SEED,
    batch_size: int = 500,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Compare compatible selection with matched rule ablations by drug bootstrap."""

    required = {"fold", "drug_id", "candidate", *METRICS}
    missing = required - set(episode_metrics.columns)
    if missing:
        raise ValueError(f"cross-fit rule-ablation metrics lack columns: {sorted(missing)}")
    if episode_metrics.empty:
        raise ValueError("at least one evaluable drug--fold is required")
    if episode_metrics.duplicated(["fold", "drug_id", "candidate"]).any():
        raise ValueError("cross-fit rule-ablation metrics contain duplicate rows")
    ordered_candidates = tuple(candidate_ids)
    candidate_index = {candidate: index for index, candidate in enumerate(ordered_candidates)}
    if not ordered_candidates or len(candidate_index) != len(ordered_candidates):
        raise ValueError("candidate_ids must be a nonempty unique sequence")
    rules = tuple(str(rule) for rule in comparison_rules)
    if (
        not rules
        or len(set(rules)) != len(rules)
        or COMPATIBLE_SELECTION_RULE in rules
        or any(rule not in SELECTION_RULE_ABLATIONS for rule in rules)
    ):
        raise ValueError("comparison_rules must be unique frozen selector ablations")
    if iterations <= 0 or batch_size <= 0:
        raise ValueError("iterations and batch_size must be positive")

    selections: dict[tuple[int, str, str, str], tuple[str, ...]] = {}
    cell_keys_by_fold: dict[int, set[tuple[str, str]]] = {}
    all_rules = (COMPATIBLE_SELECTION_RULE, *rules)
    for fold, cells in selector_cells_by_fold.items():
        current: set[tuple[str, str]] = set()
        for cell in cells:
            center = str(cell["center"])
            tau = str(cell["tau"])
            key = (center, tau)
            if key in current:
                raise ValueError("cross-fit rule-ablation selector has a duplicate cell")
            current.add(key)
            for rule in all_rules:
                selected = _selection_rule_candidates(cell, rule)
                if any(candidate not in candidate_index for candidate in selected):
                    raise ValueError(
                        f"selector rule {rule!r} contains a candidate outside the library"
                    )
                selections[(int(fold), center, tau, rule)] = selected
        cell_keys_by_fold[int(fold)] = current
    observed_folds = set(episode_metrics["fold"].astype(int))
    if observed_folds != set(cell_keys_by_fold):
        raise ValueError("metrics and rule-ablation selectors use different folds")
    if len(set(map(frozenset, cell_keys_by_fold.values()))) != 1:
        raise ValueError("source folds do not share the same rule-ablation cells")
    tau_rank = {fraction_text(tau): index for index, tau in enumerate(TAU_GRID)}
    observed_taus = {
        tau for cells in cell_keys_by_fold.values() for _, tau in cells
    }
    unknown_taus = observed_taus - set(tau_rank)
    if unknown_taus:
        raise ValueError(
            f"rule-ablation selectors have unknown tau values: {sorted(unknown_taus)}"
        )
    cell_keys = sorted(
        next(iter(cell_keys_by_fold.values())),
        key=lambda item: (item[0], tau_rank[item[1]]),
    )
    if not cell_keys:
        raise ValueError("rule-ablation selectors must contain at least one cell")
    identities = [
        (center, tau, rule)
        for center, tau in cell_keys
        for rule in rules
    ]

    indexed = episode_metrics.set_index(["drug_id", "fold", "candidate"]).sort_index()
    episode_keys = sorted({(int(drug), int(fold)) for drug, fold, _ in indexed.index})
    drug_ids = np.array(sorted({drug for drug, _ in episode_keys}), dtype=np.int64)
    drug_index = {drug: index for index, drug in enumerate(drug_ids)}
    per_metric = {
        metric: np.zeros((len(drug_ids), len(identities)), dtype=np.float64)
        for metric in METRICS
    }
    fold_counts = np.zeros(len(drug_ids), dtype=np.int64)
    for drug, fold in episode_keys:
        block = indexed.xs((drug, fold), level=("drug_id", "fold"))
        if set(block.index) != set(ordered_candidates):
            raise ValueError("every drug--fold must contain every candidate")
        row = drug_index[drug]
        fold_counts[row] += 1
        for column, (center, tau, rule) in enumerate(identities):
            compatible = selections[
                (fold, center, tau, COMPATIBLE_SELECTION_RULE)
            ]
            ablation = selections[(fold, center, tau, rule)]
            for metric in METRICS:
                compatible_worst = min(
                    float(block.loc[candidate, metric]) for candidate in compatible
                )
                ablation_worst = min(
                    float(block.loc[candidate, metric]) for candidate in ablation
                )
                per_metric[metric][row, column] += (
                    compatible_worst - ablation_worst
                )
    if (fold_counts <= 0).any():
        raise AssertionError("every rule-ablation bootstrap drug needs an evaluable fold")
    for matrix in per_metric.values():
        matrix /= fold_counts[:, None]

    points = {metric: matrix.mean(axis=0) for metric, matrix in per_metric.items()}
    bootstraps = {
        metric: np.empty((iterations, len(identities)), dtype=np.float64)
        for metric in METRICS
    }
    rng = np.random.default_rng(seed)
    offset = 0
    while offset < iterations:
        size = min(batch_size, iterations - offset)
        sampled = rng.integers(0, len(drug_ids), size=(size, len(drug_ids)))
        for metric, matrix in per_metric.items():
            bootstraps[metric][offset : offset + size] = matrix[sampled].mean(axis=1)
        offset += size

    ap_deviations = bootstraps["average_precision"] - points["average_precision"]
    simultaneous_lower_shift = float(
        np.quantile(ap_deviations.min(axis=1), 0.025)
    )
    simultaneous_upper_shift = float(
        np.quantile(ap_deviations.max(axis=1), 0.975)
    )
    rows: list[dict[str, object]] = []
    for column, (center, tau, rule) in enumerate(identities):
        row: dict[str, object] = {
            "center": center,
            "tau": tau,
            "reference_rule": COMPATIBLE_SELECTION_RULE,
            "ablation_rule": rule,
            "drug_count": len(drug_ids),
            "average_precision_difference": points["average_precision"][column],
            "average_precision_simultaneous_lower_95": (
                points["average_precision"][column] + simultaneous_lower_shift
            ),
            "average_precision_simultaneous_upper_95": (
                points["average_precision"][column] + simultaneous_upper_shift
            ),
        }
        for metric in ("pairwise_auc", "recall_at_10", "recall_at_50"):
            row[f"{metric}_difference"] = points[metric][column]
            lower, upper = np.quantile(bootstraps[metric][:, column], [0.025, 0.975])
            row[f"{metric}_lower_95"] = float(lower)
            row[f"{metric}_upper_95"] = float(upper)
        rows.append(row)
    return pd.DataFrame(rows), {
        "bootstrap_unit": "drug",
        "fold_handling": "worst_member_per_rule_within_fold_then_mean_folds_within_drug",
        "reference_rule": COMPATIBLE_SELECTION_RULE,
        "comparison_rules": list(rules),
        "iterations": iterations,
        "seed": seed,
        "ap_simultaneous_family_size": len(identities),
        "simultaneous_interval": "two_sided_familywise_95",
        "simultaneous_lower_tail_probability": 0.025,
        "simultaneous_upper_tail_probability": 0.975,
        "simultaneous_lower_shift": simultaneous_lower_shift,
        "simultaneous_upper_shift": simultaneous_upper_shift,
        "affects_primary_success_gate": False,
    }


def _consecutive_windows(
    labels: Sequence[str], values: Sequence[bool], length: int = 3
) -> list[list[str]]:
    """Return every labeled consecutive all-true window."""

    if len(labels) != len(values):
        raise ValueError("consecutive-window labels and values must align")
    if length <= 0:
        raise ValueError("consecutive-window length must be positive")
    return [
        list(labels[start : start + length])
        for start in range(len(values) - length + 1)
        if all(values[start : start + length])
    ]


def evaluate_objective_alignment_diagnostic(
    inference: pd.DataFrame,
) -> dict[str, object]:
    """Classify pairwise-AUC evidence without changing the AP success gate."""

    required = {
        "center",
        "tau",
        "baseline",
        "pairwise_auc_difference",
        "pairwise_auc_lower_95",
        "pairwise_auc_upper_95",
    }
    missing = required - set(inference.columns)
    if missing:
        raise ValueError(
            f"objective-alignment inference lacks columns: {sorted(missing)}"
        )
    statuses = (
        "ORDINARY_95_SUPPORTED_AGAINST_BOTH_BASELINES",
        "POINT_SUPPORTED_ORDINARY_95_INCONCLUSIVE",
        "ORDINARY_95_CONTRADICTED_AGAINST_AT_LEAST_ONE_BASELINE",
        "MIXED_OR_INCONCLUSIVE",
    )
    counts = {status: 0 for status in statuses}
    tau_order = [fraction_text(tau) for tau in TAU_GRID]
    tau_rank = {tau: index for index, tau in enumerate(tau_order)}
    observed_taus = set(inference["tau"].astype(str))
    unknown_taus = observed_taus - set(tau_rank)
    if unknown_taus:
        raise ValueError(
            f"objective-alignment inference has unknown tau values: {sorted(unknown_taus)}"
        )

    cells: list[dict[str, object]] = []
    identities = sorted(
        {
            (str(center), str(tau))
            for center, tau in inference[["center", "tau"]].itertuples(index=False)
        },
        key=lambda item: (item[0], tau_rank[item[1]]),
    )
    for center, tau in identities:
        cell = inference[
            (inference["center"].astype(str) == center)
            & (inference["tau"].astype(str) == tau)
        ]
        if set(cell["baseline"].astype(str)) != set(PRIMARY_BASELINES):
            raise ValueError(
                f"objective-alignment inference is incomplete for center={center}, tau={tau}"
            )
        values = cell[
            [
                "pairwise_auc_difference",
                "pairwise_auc_lower_95",
                "pairwise_auc_upper_95",
            ]
        ].to_numpy(dtype=np.float64)
        if not np.isfinite(values).all():
            raise ValueError("objective-alignment inference contains non-finite values")
        if (values[:, 1] > values[:, 2]).any():
            raise ValueError("objective-alignment AUC interval has lower > upper")

        differences = values[:, 0]
        lowers = values[:, 1]
        uppers = values[:, 2]
        if (lowers > 0.0).all():
            status = "ORDINARY_95_SUPPORTED_AGAINST_BOTH_BASELINES"
        elif (uppers < 0.0).any():
            status = "ORDINARY_95_CONTRADICTED_AGAINST_AT_LEAST_ONE_BASELINE"
        elif (differences > 0.0).all():
            status = "POINT_SUPPORTED_ORDINARY_95_INCONCLUSIVE"
        else:
            status = "MIXED_OR_INCONCLUSIVE"
        counts[status] += 1
        cells.append({"center": center, "tau": tau, "status": status})

    return {
        "affects_primary_success_gate": False,
        "comparison_class": "heldout_recorded_positive_vs_unlabeled",
        "interval_type": "ordinary_paired_drug_bootstrap_95",
        "multiplicity_adjusted": False,
        "classification_counts": counts,
        "cells": cells,
    }


def _route_objective_alignment_claim(
    diagnostic: dict[str, object],
    *,
    center_strong_windows: Mapping[str, Sequence[Sequence[str]]],
    primary_passed: bool,
) -> None:
    """Bind mechanism language to the union of preregistered AP-strong windows."""

    cells = diagnostic["cells"]
    if not isinstance(cells, list):
        raise TypeError("objective-alignment diagnostic cells must be a list")
    status_by_cell = {
        (str(cell["center"]), str(cell["tau"])): str(cell["status"])
        for cell in cells
    }
    tau_rank = {
        fraction_text(tau): index for index, tau in enumerate(TAU_GRID)
    }
    region_keys = sorted(
        {
            (str(center), str(tau))
            for center, windows in center_strong_windows.items()
            for window in windows
            for tau in window
        },
        key=lambda item: (item[0], tau_rank[item[1]]),
    )
    region_cells = [
        {"center": center, "tau": tau, "status": status_by_cell[(center, tau)]}
        for center, tau in region_keys
    ]
    statuses = [str(cell["status"]) for cell in region_cells]
    supported = "ORDINARY_95_SUPPORTED_AGAINST_BOTH_BASELINES"
    point_supported = "POINT_SUPPORTED_ORDINARY_95_INCONCLUSIVE"
    contradicted = "ORDINARY_95_CONTRADICTED_AGAINST_AT_LEAST_ONE_BASELINE"
    if not primary_passed:
        routing = "PRIMARY_GATE_FAILED_NO_MECHANISM_CLAIM"
    elif not statuses:
        raise AssertionError("a passing primary gate must have an AP-strong region")
    elif all(status == supported for status in statuses):
        routing = "ORDINARY_95_CONSISTENT_ACROSS_AP_STRONG_REGION"
    elif any(status == contradicted for status in statuses):
        routing = "ORDINARY_95_CONTRADICTION_WITHIN_AP_STRONG_REGION"
    elif all(status in {supported, point_supported} for status in statuses):
        routing = "POINT_CONSISTENT_ORDINARY_95_INCONCLUSIVE_ACROSS_AP_STRONG_REGION"
    else:
        routing = "MIXED_OR_INCONCLUSIVE_WITHIN_AP_STRONG_REGION"
    diagnostic["ap_strong_region"] = {
        "definition": "union_of_all_center_strong_windows",
        "cells": region_cells,
        "claim_routing": routing,
    }


def evaluate_rule_ablation_claim_routing(
    inference: pd.DataFrame,
    *,
    primary_success: Mapping[str, object],
) -> dict[str, object]:
    """Route N2 claims over the full preregistered AP-strong region."""

    required = {
        "center",
        "tau",
        "reference_rule",
        "ablation_rule",
        "average_precision_difference",
        "average_precision_simultaneous_lower_95",
        "average_precision_simultaneous_upper_95",
    }
    missing = required - set(inference.columns)
    if missing:
        raise ValueError(f"rule-ablation inference lacks columns: {sorted(missing)}")
    if inference.empty:
        raise ValueError("rule-ablation inference must not be empty")
    if set(inference["reference_rule"].astype(str)) != {
        COMPATIBLE_SELECTION_RULE
    }:
        raise ValueError("rule-ablation inference has an unexpected reference rule")
    observed_rules = set(inference["ablation_rule"].astype(str))
    if observed_rules != set(SELECTION_RULE_ABLATIONS):
        raise ValueError("rule-ablation inference does not contain both frozen rules")

    windows = primary_success.get("center_strong_windows")
    if not isinstance(windows, Mapping):
        raise ValueError("primary success summary lacks center_strong_windows")
    tau_rank = {fraction_text(tau): index for index, tau in enumerate(TAU_GRID)}
    region_keys = sorted(
        {
            (str(center), str(tau))
            for center, center_windows in windows.items()
            for window in center_windows
            for tau in window
        },
        key=lambda item: (item[0], tau_rank[item[1]]),
    )
    primary_passed = bool(primary_success.get("practical_selection_passed"))
    if primary_passed and not region_keys:
        raise AssertionError("a passing primary gate must have an AP-strong region")

    by_rule: dict[str, dict[str, object]] = {}
    for rule in SELECTION_RULE_ABLATIONS:
        cells: list[dict[str, object]] = []
        for center, tau in region_keys:
            row = inference[
                (inference["center"].astype(str) == center)
                & (inference["tau"].astype(str) == tau)
                & (inference["ablation_rule"].astype(str) == rule)
            ]
            if len(row) != 1:
                raise ValueError(
                    "rule-ablation inference must contain exactly one row for "
                    f"center={center}, tau={tau}, rule={rule}"
                )
            values = row.iloc[0]
            difference = float(values["average_precision_difference"])
            lower = float(values["average_precision_simultaneous_lower_95"])
            upper = float(values["average_precision_simultaneous_upper_95"])
            if not np.isfinite([difference, lower, upper]).all():
                raise ValueError("rule-ablation inference contains non-finite AP values")
            if lower > upper:
                raise ValueError("rule-ablation simultaneous AP interval has lower > upper")
            cells.append(
                {
                    "center": center,
                    "tau": tau,
                    "difference": difference,
                    "simultaneous_lower_95": lower,
                    "simultaneous_upper_95": upper,
                }
            )
        if not primary_passed:
            routing = "PRIMARY_GATE_FAILED_NO_RULE_ABLATION_CLAIM"
        elif all(float(cell["simultaneous_lower_95"]) > 0.0 for cell in cells):
            routing = "SIMULTANEOUS_95_DOMINATES_ACROSS_AP_STRONG_REGION"
        elif any(float(cell["simultaneous_upper_95"]) < 0.0 for cell in cells):
            routing = "SIMULTANEOUS_95_WORSE_WITHIN_AP_STRONG_REGION"
        elif all(float(cell["difference"]) >= 0.0 for cell in cells):
            routing = "POINT_NONNEGATIVE_SIMULTANEOUS_95_INCONCLUSIVE"
        else:
            routing = "MIXED_OR_INCONCLUSIVE_WITHIN_AP_STRONG_REGION"
        by_rule[rule] = {"claim_routing": routing, "cells": cells}

    return {
        "affects_primary_success_gate": False,
        "interval_type": "paired_drug_bootstrap_two_sided_familywise_95",
        "region_definition": "union_of_all_center_strong_windows",
        "primary_gate_passed": primary_passed,
        "by_ablation_rule": by_rule,
        "matched_rectangular_claim_routing": by_rule["matched_rectangular"][
            "claim_routing"
        ],
    }


def evaluate_success_gate(
    inference: pd.DataFrame,
    *,
    selector_cells: Sequence[Mapping[str, object]],
    coverage: Mapping[str, object],
    candidate_count: int,
    main_selector_narrowed: bool,
) -> dict[str, object]:
    """Apply the frozen selector-anchor source-crossfit success gate."""

    required = {
        "center",
        "tau",
        "baseline",
        "average_precision_difference",
        "average_precision_simultaneous_lower_95",
        "pairwise_auc_difference",
        "pairwise_auc_lower_95",
        "pairwise_auc_upper_95",
        "recall_at_10_difference",
        "recall_at_50_difference",
    }
    missing = required - set(inference.columns)
    if missing:
        raise ValueError(f"inference lacks columns: {sorted(missing)}")
    tau_order = [fraction_text(tau) for tau in TAU_GRID]
    centers = sorted({str(cell["center"]) for cell in selector_cells})
    narrowed_cells = [
        {
            "source_fold": cell.get("source_fold"),
            "center": str(cell["center"]),
            "tau": str(cell["tau"]),
            "selected_count": len(cell["possibly_optimal"]),
        }
        for cell in selector_cells
        if len(cell["possibly_optimal"]) < candidate_count
    ]
    narrowed = bool(narrowed_cells)
    center_point_runs: dict[str, bool] = {}
    center_strong_runs: dict[str, bool] = {}
    center_point_windows: dict[str, list[list[str]]] = {}
    center_strong_windows: dict[str, list[list[str]]] = {}
    center_recall_blocked_windows: dict[str, list[list[str]]] = {}
    for center in centers:
        point_flags: list[bool] = []
        ap_strong_flags: list[bool] = []
        for tau in tau_order:
            cell = inference[(inference["center"] == center) & (inference["tau"] == tau)]
            if (
                len(cell) != len(PRIMARY_BASELINES)
                or cell["baseline"].duplicated().any()
                or set(cell["baseline"]) != set(PRIMARY_BASELINES)
            ):
                raise ValueError(f"inference is incomplete for center={center}, tau={tau}")
            point_nonnegative = bool((cell["average_precision_difference"] >= 0).all())
            prior_row = cell[cell["baseline"] == "rna_prior_only"].iloc[0]
            simultaneous_positive = (
                prior_row["average_precision_simultaneous_lower_95"] > 0
            )
            point_flags.append(point_nonnegative)
            ap_strong_flags.append(point_nonnegative and simultaneous_positive)
        center_point_windows[center] = _consecutive_windows(tau_order, point_flags)
        candidate_strong_windows = _consecutive_windows(tau_order, ap_strong_flags)
        blocked_windows: list[list[str]] = []
        accepted_windows: list[list[str]] = []
        for window in candidate_strong_windows:
            window_rows = inference[
                (inference["center"] == center)
                & (inference["tau"].isin(window))
            ]
            systematic_degradation = any(
                bool(
                    (
                        window_rows[window_rows["baseline"] == baseline][metric]
                        < -0.01
                    ).all()
                )
                for baseline in PRIMARY_BASELINES
                for metric in (
                    "recall_at_10_difference",
                    "recall_at_50_difference",
                )
            )
            if systematic_degradation:
                blocked_windows.append(window)
            else:
                accepted_windows.append(window)
        center_recall_blocked_windows[center] = blocked_windows
        center_strong_windows[center] = accepted_windows
        center_point_runs[center] = bool(center_point_windows[center])
        center_strong_runs[center] = bool(center_strong_windows[center])

    passed = bool(
        coverage.get("publication_coverage_passed")
        and main_selector_narrowed
        and narrowed
        and centers
        and all(center_point_runs.values())
        and any(center_strong_runs.values())
    )
    objective_alignment = evaluate_objective_alignment_diagnostic(inference)
    _route_objective_alignment_claim(
        objective_alignment,
        center_strong_windows=center_strong_windows,
        primary_passed=passed,
    )
    return {
        "publication_coverage_passed": bool(
            coverage.get("publication_coverage_passed")
        ),
        "main_selector_narrowed_at_least_one_cell": bool(main_selector_narrowed),
        "selector_narrowed_at_least_one_cell": narrowed,
        "selector_narrowed_cells": narrowed_cells,
        "center_point_runs": center_point_runs,
        "center_point_windows": center_point_windows,
        "center_strong_runs": center_strong_runs,
        "center_strong_windows": center_strong_windows,
        "center_recall_blocked_windows": center_recall_blocked_windows,
        "objective_alignment_diagnostic": objective_alignment,
        "practical_selection_passed": passed,
        "decision": "PASS" if passed else "FAIL_RETAIN_AS_BOUNDARY_RESULT",
    }
