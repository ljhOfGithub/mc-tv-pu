"""Run the superseded leave-one-PMID-out diagnostic.

This runner does not provide globally source-disjoint external validation.  It is
retained only to reproduce the pre-amendment source-rotation sensitivity check.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from fractions import Fraction
from pathlib import Path

import numpy as np
import pandas as pd

from mc_tv_pu_q_tau_semantics_audit import (
    restrict_pair_sources_to_positive_matrix,
    source_sets_by_drug,
)
from mc_tv_pu_score_input_gate import (
    audit_score_partition_directory,
    validate_candidate_manifest,
)
from mc_tv_pu_selector_evaluation import (
    CENTER_MODES,
    CHECKPOINT_COLUMNS,
    TAU_GRID,
    evaluate_drug_center,
    fraction_text,
    iter_fold_score_matrices,
    parse_fraction,
    run_configuration,
    source_center_profile,
    write_or_validate_run_configuration,
    write_or_validate_text,
)
from mc_tv_pu_source_isolated_validation import (
    BOOTSTRAP_ITERATIONS,
    BOOTSTRAP_SEED,
    PRIMARY_BASELINES,
    SourceEpisode,
    aggregate_episode_metrics,
    bootstrap_set_valued_performance,
    build_source_episodes,
    episode_coverage_summary,
    evaluate_episode_scores,
    evaluate_success_gate,
)
from source_balanced_pairwise_pilot import build_pair_sources

SOURCE_ORACLE_COLUMNS = (
    "heldout_source",
    "anchor_source_count",
    "heldout_novel_count",
    *CHECKPOINT_COLUMNS,
)
SOURCE_METRIC_COLUMNS = (
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


def source_oracle_checkpoint_path(output_dir: Path, drug_id: int) -> Path:
    return output_dir / "source_oracle" / f"drug={drug_id}.parquet"


def source_metric_checkpoint_path(output_dir: Path, drug_id: int) -> Path:
    return output_dir / "source_metrics" / f"drug={drug_id}.parquet"


def _atomic_parquet(frame: pd.DataFrame, path: Path) -> None:
    if path.exists():
        raise FileExistsError(f"checkpoint already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp.parquet")
    if temporary.exists():
        raise FileExistsError(f"unfinished checkpoint requires manual review: {temporary}")
    frame.to_parquet(temporary, index=False, compression="zstd")
    temporary.replace(path)


def validate_source_oracle_checkpoint(
    path: Path,
    *,
    episodes: Sequence[SourceEpisode],
    candidate_ids: Sequence[str],
) -> None:
    frame = pd.read_parquet(path)
    if tuple(frame.columns) != SOURCE_ORACLE_COLUMNS:
        raise ValueError(f"source-oracle checkpoint schema mismatch: {path}")
    expected_rows = (
        len(episodes)
        * len(CENTER_MODES)
        * len(TAU_GRID)
        * len(candidate_ids) ** 2
    )
    if len(frame) != expected_rows:
        raise ValueError(f"source-oracle checkpoint row count mismatch: {path}")
    identity = ["heldout_source", "center", "tau", "candidate", "comparator"]
    if frame.duplicated(identity).any():
        raise ValueError(f"source-oracle checkpoint contains duplicate cells: {path}")
    expected = {
        (
            episode.heldout_source,
            center,
            fraction_text(tau),
            candidate,
            comparator,
        )
        for episode in episodes
        for center in CENTER_MODES
        for tau in TAU_GRID
        for candidate in candidate_ids
        for comparator in candidate_ids
    }
    observed = set(map(tuple, frame[identity].itertuples(index=False, name=None)))
    if observed != expected:
        raise ValueError(f"source-oracle checkpoint cell coverage mismatch: {path}")
    episode_map = {episode.heldout_source: episode for episode in episodes}
    for source, group in frame.groupby("heldout_source", sort=False):
        episode = episode_map[str(source)]
        if set(group["drug_id"]) != {episode.drug_id}:
            raise ValueError(f"source-oracle checkpoint drug mismatch: {path}")
        if set(group["anchor_source_count"]) != {len(episode.anchor_sources)}:
            raise ValueError(f"source-oracle checkpoint anchor count mismatch: {path}")
        if set(group["heldout_novel_count"]) != {len(episode.heldout_novel_rnas)}:
            raise ValueError(f"source-oracle checkpoint held-out count mismatch: {path}")


def validate_source_metric_checkpoint(
    path: Path,
    *,
    episodes: Sequence[SourceEpisode],
    candidate_ids: Sequence[str],
) -> None:
    frame = pd.read_parquet(path)
    if tuple(frame.columns) != SOURCE_METRIC_COLUMNS:
        raise ValueError(f"source-metric checkpoint schema mismatch: {path}")
    evaluable_sources = {episode.heldout_source for episode in episodes if episode.evaluable}
    expected_rows = len(evaluable_sources) * len(candidate_ids)
    if len(frame) != expected_rows:
        raise ValueError(f"source-metric checkpoint row count mismatch: {path}")
    if frame.duplicated(["heldout_source", "candidate"]).any():
        raise ValueError(f"source-metric checkpoint contains duplicate rows: {path}")
    if set(frame["heldout_source"]) != evaluable_sources:
        raise ValueError(f"source-metric checkpoint source coverage mismatch: {path}")
    expected_candidates = set(candidate_ids) if evaluable_sources else set()
    if set(frame["candidate"]) != expected_candidates:
        raise ValueError(f"source-metric checkpoint candidate coverage mismatch: {path}")
    if not frame.empty and set(frame["drug_id"]) != {episodes[0].drug_id}:
        raise ValueError(f"source-metric checkpoint drug mismatch: {path}")
    numeric = frame[
        ["average_precision", "pairwise_auc", "recall_at_10", "recall_at_50"]
    ]
    if not np.isfinite(numeric.to_numpy(dtype=np.float64)).all():
        raise ValueError(f"source-metric checkpoint contains non-finite values: {path}")


def evaluate_source_drug_outputs(
    *,
    drug_id: int,
    score_matrix: np.ndarray,
    candidate_ids: tuple[str, ...],
    source_sets: dict[str, set[int]],
    episodes: tuple[SourceEpisode, ...],
    n_rna: int,
    output_dir: Path,
) -> tuple[int, int]:
    """Create or validate exact-oracle and held-out metric checkpoints for one drug."""

    oracle_path = source_oracle_checkpoint_path(output_dir, drug_id)
    metric_path = source_metric_checkpoint_path(output_dir, drug_id)
    oracle_exists = oracle_path.exists()
    metric_exists = metric_path.exists()
    if oracle_exists:
        validate_source_oracle_checkpoint(
            oracle_path, episodes=episodes, candidate_ids=candidate_ids
        )
    if metric_exists:
        validate_source_metric_checkpoint(
            metric_path, episodes=episodes, candidate_ids=candidate_ids
        )
    if oracle_exists and metric_exists:
        return drug_id, 0

    core_by_profile: dict[tuple[Fraction, ...], pd.DataFrame] = {}
    oracle_frames: list[pd.DataFrame] = []
    metric_frames: list[pd.DataFrame] = []
    for episode in episodes:
        anchors = {
            source: source_sets[source] for source in episode.anchor_sources
        }
        for center in CENTER_MODES:
            p0 = source_center_profile(
                anchors, n_rna_candidates=n_rna, mode=center
            )
            if p0 in core_by_profile:
                core = core_by_profile[p0].copy()
                core["center"] = center
            else:
                core = evaluate_drug_center(
                    drug_id=drug_id,
                    center=center,
                    scores=score_matrix,
                    candidate_ids=candidate_ids,
                    p0=p0,
                    taus=TAU_GRID,
                )
                core_by_profile[p0] = core
            annotated = core.copy()
            annotated.insert(0, "heldout_novel_count", len(episode.heldout_novel_rnas))
            annotated.insert(0, "anchor_source_count", len(episode.anchor_sources))
            annotated.insert(0, "heldout_source", episode.heldout_source)
            oracle_frames.append(annotated)
        metrics = evaluate_episode_scores(
            episode, scores=score_matrix, candidate_ids=candidate_ids
        )
        if not metrics.empty:
            metric_frames.append(metrics)

    oracle_frame = pd.concat(oracle_frames, ignore_index=True)[list(SOURCE_ORACLE_COLUMNS)]
    if metric_frames:
        metric_frame = pd.concat(metric_frames, ignore_index=True)[list(SOURCE_METRIC_COLUMNS)]
    else:
        metric_frame = pd.DataFrame(columns=SOURCE_METRIC_COLUMNS)
    created = 0
    if not metric_exists:
        _atomic_parquet(metric_frame, metric_path)
        created += 1
    if not oracle_exists:
        _atomic_parquet(oracle_frame, oracle_path)
        created += 1
    validate_source_oracle_checkpoint(
        oracle_path, episodes=episodes, candidate_ids=candidate_ids
    )
    validate_source_metric_checkpoint(
        metric_path, episodes=episodes, candidate_ids=candidate_ids
    )
    return drug_id, created


def evaluate_source_drugs(
    score_items: Iterable[tuple[int, np.ndarray]],
    *,
    candidate_ids: tuple[str, ...],
    source_sets: Mapping[int, dict[str, set[int]]],
    episodes_by_drug: Mapping[int, tuple[SourceEpisode, ...]],
    n_rna: int,
    output_dir: Path,
    workers: int,
) -> set[int]:
    if workers <= 0:
        raise ValueError("workers must be a positive integer")

    def task(drug_id: int, score_matrix: np.ndarray):
        return {
            "drug_id": drug_id,
            "score_matrix": score_matrix,
            "candidate_ids": candidate_ids,
            "source_sets": source_sets[drug_id],
            "episodes": episodes_by_drug[drug_id],
            "n_rna": n_rna,
            "output_dir": output_dir,
        }

    observed: set[int] = set()
    eligible_items = (
        (drug_id, matrix)
        for drug_id, matrix in score_items
        if drug_id in episodes_by_drug
    )
    if workers == 1:
        for drug_id, matrix in eligible_items:
            observed.add(drug_id)
            completed, created = evaluate_source_drug_outputs(**task(drug_id, matrix))
            print(
                f"[mc-tv-pu-source] completed drug={completed} checkpoints={created}",
                flush=True,
            )
        return observed

    with ProcessPoolExecutor(max_workers=workers) as executor:
        pending = set()
        for drug_id, matrix in eligible_items:
            observed.add(drug_id)
            pending.add(executor.submit(evaluate_source_drug_outputs, **task(drug_id, matrix)))
            if len(pending) >= 2 * workers:
                done, pending = wait(pending, return_when=FIRST_COMPLETED)
                for future in done:
                    completed, created = future.result()
                    print(
                        f"[mc-tv-pu-source] completed drug={completed} "
                        f"checkpoints={created}",
                        flush=True,
                    )
        while pending:
            done, pending = wait(pending, return_when=FIRST_COMPLETED)
            for future in done:
                completed, created = future.result()
                print(
                    f"[mc-tv-pu-source] completed drug={completed} checkpoints={created}",
                    flush=True,
                )
    return observed


def aggregate_source_oracle_intervals(
    frames: Iterable[pd.DataFrame], *, candidate_ids: Sequence[str]
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Aggregate episode intervals within drug, then drugs, then comparator max."""

    candidates = tuple(candidate_ids)
    episode_values: dict[
        tuple[str, str, str, str, int, str], tuple[Fraction, Fraction]
    ] = {}
    for frame in frames:
        if tuple(frame.columns) != SOURCE_ORACLE_COLUMNS:
            raise ValueError("source-oracle checkpoint has an unexpected schema")
        for row in frame.itertuples(index=False):
            if not bool(row.certified) or row.status != "gap_met":
                raise ValueError(
                    "uncertified source oracle blocks aggregation: "
                    f"drug={row.drug_id}, source={row.heldout_source}"
                )
            key = (
                row.center,
                row.tau,
                row.candidate,
                row.comparator,
                int(row.drug_id),
                str(row.heldout_source),
            )
            if key in episode_values:
                raise ValueError(f"duplicate source-oracle cell {key!r}")
            lower = parse_fraction(row.lower_bound)
            upper = parse_fraction(row.upper_bound)
            if lower > upper:
                raise ValueError("source-oracle checkpoint has an inverted interval")
            episode_values[key] = (lower, upper)

    center_tau = sorted({(key[0], key[1]) for key in episode_values})
    interval_rows: list[dict[str, object]] = []
    summary_cells: list[dict[str, object]] = []
    for center, tau in center_tau:
        pair_values: dict[tuple[str, str], tuple[Fraction, Fraction, int]] = {}
        reference_drugs: set[int] | None = None
        for candidate in candidates:
            for comparator in candidates:
                by_drug: dict[int, list[tuple[Fraction, Fraction]]] = defaultdict(list)
                for key, value in episode_values.items():
                    if key[:4] == (center, tau, candidate, comparator):
                        by_drug[key[4]].append(value)
                if not by_drug:
                    raise ValueError("source-oracle aggregation cell is missing")
                drugs = set(by_drug)
                if reference_drugs is None:
                    reference_drugs = drugs
                elif drugs != reference_drugs:
                    raise ValueError("source-oracle cells use different drug sets")
                drug_means = [
                    (
                        sum((value[0] for value in values), Fraction(0)) / len(values),
                        sum((value[1] for value in values), Fraction(0)) / len(values),
                    )
                    for values in by_drug.values()
                ]
                pair_values[(candidate, comparator)] = (
                    sum((value[0] for value in drug_means), Fraction(0))
                    / len(drug_means),
                    sum((value[1] for value in drug_means), Fraction(0))
                    / len(drug_means),
                    len(drug_means),
                )

        candidate_intervals: dict[str, tuple[Fraction, Fraction]] = {}
        rows_by_candidate: dict[str, dict[str, object]] = {}
        for candidate in candidates:
            comparisons = {
                comparator: pair_values[(candidate, comparator)]
                for comparator in candidates
            }
            lower = max(value[0] for value in comparisons.values())
            upper = max(value[1] for value in comparisons.values())
            candidate_intervals[candidate] = (lower, upper)
            row = {
                "center": center,
                "tau": tau,
                "candidate": candidate,
                "drug_count": comparisons[candidates[0]][2],
                "lower_V": fraction_text(lower),
                "upper_V": fraction_text(upper),
                "lower_V_float": float(lower),
                "upper_V_float": float(upper),
                "worst_comparator_lower": json.dumps(
                    [
                        comparator
                        for comparator, value in comparisons.items()
                        if value[0] == lower
                    ]
                ),
                "worst_comparator_upper": json.dumps(
                    [
                        comparator
                        for comparator, value in comparisons.items()
                        if value[1] == upper
                    ]
                ),
            }
            interval_rows.append(row)
            rows_by_candidate[candidate] = row
        minimum_upper = min(value[1] for value in candidate_intervals.values())
        possibly = [
            candidate
            for candidate in candidates
            if candidate_intervals[candidate][0] <= minimum_upper
        ]
        unique = [
            candidate
            for candidate in candidates
            if candidate_intervals[candidate][1]
            < min(
                candidate_intervals[other][0]
                for other in candidates
                if other != candidate
            )
        ]
        for candidate, row in rows_by_candidate.items():
            row["possibly_optimal"] = candidate in possibly
            row["certifiably_unique"] = candidate in unique
        summary_cells.append(
            {
                "center": center,
                "tau": tau,
                "possibly_optimal": possibly,
                "certifiably_unique": unique,
                "minimum_upper": fraction_text(minimum_upper),
            }
        )
    all_uninformative = bool(summary_cells) and all(
        len(cell["possibly_optimal"]) == len(candidates) for cell in summary_cells
    )
    return pd.DataFrame(interval_rows), {
        "status": "PASS",
        "weighting": "episode-equal-within-drug_then_drug-equal",
        "practical_selection_gate": (
            "NO_SELECTION_VALUE"
            if all_uninformative
            else "AT_LEAST_ONE_CELL_NARROWS_LIBRARY"
        ),
        "cells": summary_cells,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--score-dir", type=Path, required=True)
    parser.add_argument("--pair-matrix", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--source-table", type=Path, required=True)
    parser.add_argument("--dataset-dir", type=Path, default=Path("Dataset"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument(
        "--acknowledge-diagnostic-only",
        action="store_true",
        help=(
            "acknowledge that this runner permits cross-episode source reuse and "
            "cannot support a held-out-source performance claim"
        ),
    )
    args = parser.parse_args()
    if not args.acknowledge_diagnostic_only:
        raise ValueError(
            "this superseded runner is diagnostic-only; pass "
            "--acknowledge-diagnostic-only to reproduce it explicitly"
        )

    manifest = json.loads(args.manifest.read_text())
    candidate_ids, manifest_issues = validate_candidate_manifest(
        manifest, root=args.root, pair_matrix_path=args.pair_matrix
    )
    if manifest_issues:
        raise ValueError("candidate manifest invalid: " + "; ".join(manifest_issues))
    pair_matrix = np.load(args.pair_matrix, allow_pickle=False)
    active_drugs = tuple(np.flatnonzero((pair_matrix > 0).any(axis=0)).tolist())
    n_rna = int(pair_matrix.shape[0])
    fold_count = int(manifest["dataset"]["fold_count"])
    audit = audit_score_partition_directory(
        args.score_dir,
        expected_drug_ids=active_drugs,
        expected_rna_ids=range(n_rna),
        expected_models=candidate_ids,
        expected_fold_count=fold_count,
    )
    if audit.status != "PASS":
        raise ValueError("score input gate blocked: " + "; ".join(audit.issues))

    pair_sources = restrict_pair_sources_to_positive_matrix(
        build_pair_sources(args.source_table, args.dataset_dir), pair_matrix
    )
    sources = source_sets_by_drug(pair_sources)
    episodes = build_source_episodes(pair_sources)
    coverage = episode_coverage_summary(episodes)
    config = run_configuration(
        root=args.root,
        score_dir=args.score_dir,
        pair_matrix_path=args.pair_matrix,
        manifest_path=args.manifest,
        source_table=args.source_table,
    )
    config.update(
        {
            "analysis": "leave_one_pmid_out_source_rotation_diagnostic",
            "claim_validity": "diagnostic_only_not_globally_source_disjoint",
            "candidate_ids": list(candidate_ids),
            "score_input_audit": audit.to_dict(),
            "coverage": coverage,
            "bootstrap_iterations": BOOTSTRAP_ITERATIONS,
            "bootstrap_seed": BOOTSTRAP_SEED,
            "primary_baselines": list(PRIMARY_BASELINES),
        }
    )
    write_or_validate_run_configuration(args.output_dir, config)

    observed = evaluate_source_drugs(
        iter_fold_score_matrices(
            args.score_dir,
            candidate_ids=candidate_ids,
            fold_count=fold_count,
            n_rna_candidates=n_rna,
        ),
        candidate_ids=tuple(candidate_ids),
        source_sets=sources,
        episodes_by_drug=episodes,
        n_rna=n_rna,
        output_dir=args.output_dir,
        workers=args.workers,
    )
    if observed != set(episodes):
        raise ValueError("score partitions did not yield every multi-source drug")

    drug_order = sorted(episodes)
    oracle_paths = [source_oracle_checkpoint_path(args.output_dir, drug) for drug in drug_order]
    intervals, selector_summary = aggregate_source_oracle_intervals(
        (pd.read_parquet(path) for path in oracle_paths), candidate_ids=candidate_ids
    )
    metric_paths = [source_metric_checkpoint_path(args.output_dir, drug) for drug in drug_order]
    episode_metrics = pd.concat(
        (pd.read_parquet(path) for path in metric_paths), ignore_index=True
    )
    drug_metrics = aggregate_episode_metrics(episode_metrics)
    inference, bootstrap_config = bootstrap_set_valued_performance(
        drug_metrics,
        selector_cells=selector_summary["cells"],
        candidate_ids=candidate_ids,
    )
    success = evaluate_success_gate(
        inference,
        selector_cells=selector_summary["cells"],
        coverage=coverage,
        candidate_count=len(candidate_ids),
        main_selector_narrowed=True,
    )
    summary = {
        "coverage": coverage,
        "selector": selector_summary,
        "bootstrap": bootstrap_config,
        "success_gate": success,
    }
    write_or_validate_text(
        args.output_dir / "source_candidate_intervals.csv", intervals.to_csv(index=False)
    )
    write_or_validate_text(
        args.output_dir / "source_episode_metrics.csv", episode_metrics.to_csv(index=False)
    )
    write_or_validate_text(
        args.output_dir / "source_drug_metrics.csv", drug_metrics.to_csv(index=False)
    )
    write_or_validate_text(
        args.output_dir / "source_inference.csv", inference.to_csv(index=False)
    )
    write_or_validate_text(
        args.output_dir / "source_validation_summary.json",
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
