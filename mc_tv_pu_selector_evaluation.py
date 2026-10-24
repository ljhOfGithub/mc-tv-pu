"""Publication-tier evaluation for the frozen MC-TV-PU selector.

The module implements the analysis frozen in
``docs/mc-tv-pu-selector-preanalysis.md``.  It keeps score validation, exact
rank-loss construction, per-drug oracle checkpoints, and interval aggregation
separate so that no uncertified point estimate can silently enter a selector.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from fractions import Fraction
from pathlib import Path

import numpy as np
import pandas as pd

from mc_tv_pu_oracle import (
    ExactOracleGeometry,
    ExactOracleResult,
    prepare_mc_tv_pu_oracle_exact,
    solve_fixed_t_exact,
    solve_mc_tv_pu_oracle_exact,
    solve_mc_tv_pu_oracle_exact_prepared,
)
from mc_tv_pu_q_tau_semantics_audit import (
    restrict_pair_sources_to_positive_matrix,
    source_sets_by_drug,
)
from mc_tv_pu_score_input_gate import (
    audit_score_partition_directory,
    file_sha256,
    score_partition_path,
    validate_candidate_manifest,
)
from source_balanced_pairwise_pilot import build_pair_sources

CENTER_MODES = ("pmid_equal", "positive_pair_equal")
TAU_GRID = (
    Fraction(0),
    Fraction(1, 20),
    Fraction(1, 10),
    Fraction(1, 5),
    Fraction(3, 10),
    Fraction(2, 5),
    Fraction(1, 2),
    Fraction(3, 5),
    Fraction(7, 10),
    Fraction(4, 5),
    Fraction(9, 10),
    Fraction(1),
)
EXACT_GAP = Fraction(1, 1000)
MAX_EXACT_EVALUATIONS = 4096
CHECKPOINT_COLUMNS = (
    "center",
    "tau",
    "drug_id",
    "candidate",
    "comparator",
    "lower_bound",
    "upper_bound",
    "certified",
    "status",
    "evaluations",
    "prior_supremum",
    "rectangular_value",
    "rectangular_status",
    "empirical_value",
)


def fraction_text(value: Fraction) -> str:
    """Serialize a rational without losing its denominator."""

    return f"{value.numerator}/{value.denominator}"


def parse_fraction(value: object) -> Fraction:
    """Parse an exact value from a checkpoint cell."""

    if not isinstance(value, (str, int)):
        raise TypeError(f"expected an exact fraction string, observed {value!r}")
    return Fraction(value)


def matched_rectangular_value_exact(
    geometry: ExactOracleGeometry,
    score_difference: Sequence[Fraction],
) -> tuple[Fraction | None, str]:
    """Solve the prior-clipped TV rectangle after removing coordinate caps."""

    tv_value = solve_fixed_t_exact(
        geometry.q,
        geometry.p0,
        tuple(score_difference),
        geometry.tau,
        geometry.saturation,
    ).value
    if tv_value == 0:
        return Fraction(0), "finite"
    if tv_value > 0 and geometry.prior_supremum == 1:
        return None, "unbounded"
    if tv_value > 0:
        value = tv_value / (1 - geometry.prior_supremum)
    else:
        value = tv_value / (1 - geometry.prior_lower)
    return value, "finite"


def midrank_loss_numerators(scores: Sequence[float] | np.ndarray) -> np.ndarray:
    """Return integer numerators for the frozen rank loss with denominator ``2n``.

    A score tied with ``r`` items below ``a`` strictly greater items receives
    ``2a + r``.  Exact equality after Float64 loading defines a tie.
    """

    values = np.asarray(scores, dtype=np.float64)
    if values.ndim != 1 or len(values) == 0:
        raise ValueError("scores must be a nonempty one-dimensional vector")
    if not np.isfinite(values).all():
        raise ValueError("scores must all be finite")
    order = np.argsort(-values, kind="mergesort")
    result = np.empty(len(values), dtype=np.int64)
    start = 0
    while start < len(values):
        end = start + 1
        reference = values[order[start]]
        while end < len(values) and values[order[end]] == reference:
            end += 1
        result[order[start:end]] = 2 * start + (end - start)
        start = end
    if int(result.sum()) != len(values) ** 2:
        raise AssertionError("midrank loss numerators must average to one half")
    return result


def loss_difference_numerators(
    candidate_scores: Sequence[float] | np.ndarray,
    comparator_scores: Sequence[float] | np.ndarray,
) -> np.ndarray:
    """Return the centered exact loss-profile difference numerator."""

    candidate = midrank_loss_numerators(candidate_scores)
    comparator = midrank_loss_numerators(comparator_scores)
    if candidate.shape != comparator.shape:
        raise ValueError("candidate and comparator score vectors must have equal length")
    difference = candidate - comparator
    if int(difference.sum()) != 0:
        raise AssertionError("uniform-query loss difference must be exactly centered")
    return difference


def source_center_profile(
    sources: Mapping[str, set[int]],
    *,
    n_rna_candidates: int,
    mode: str,
) -> tuple[Fraction, ...]:
    """Construct one frozen exact source center."""

    if mode not in CENTER_MODES:
        raise ValueError(f"unknown source-center mode {mode!r}")
    if n_rna_candidates <= 0:
        raise ValueError("n_rna_candidates must be positive")
    if not sources or any(not indices for indices in sources.values()):
        raise ValueError("every drug must have at least one nonempty PMID source")
    for indices in sources.values():
        if any(index < 0 or index >= n_rna_candidates for index in indices):
            raise ValueError("source center contains an RNA outside the candidate universe")

    mass = [Fraction(0) for _ in range(n_rna_candidates)]
    if mode == "pmid_equal":
        source_weight = Fraction(1, len(sources))
        for source in sorted(sources):
            indices = sources[source]
            within_source = source_weight / len(indices)
            for index in indices:
                mass[index] += within_source
    else:
        union = set().union(*sources.values())
        within_union = Fraction(1, len(union))
        for index in union:
            mass[index] = within_union
    if sum(mass, Fraction(0)) != 1:
        raise AssertionError("source center must have exact unit mass")
    return tuple(mass)


def evaluate_drug_center(
    *,
    drug_id: int,
    center: str,
    scores: np.ndarray,
    candidate_ids: Sequence[str],
    p0: Sequence[Fraction],
    taus: Sequence[Fraction] = TAU_GRID,
    oracle: Callable[..., ExactOracleResult] = solve_mc_tv_pu_oracle_exact,
) -> pd.DataFrame:
    """Evaluate every directed candidate comparison for one drug and center."""

    values = np.asarray(scores, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError("scores must have shape (candidate, RNA)")
    if values.shape[0] != len(candidate_ids):
        raise ValueError("score rows must match the frozen candidate list")
    if len(set(candidate_ids)) != len(candidate_ids):
        raise ValueError("candidate ids must be unique")
    n_rna = values.shape[1]
    if len(p0) != n_rna or sum(p0, Fraction(0)) != 1:
        raise ValueError("p0 must be an exact probability vector over all RNAs")
    if not np.isfinite(values).all():
        raise ValueError("score matrix contains non-finite values")

    q = tuple(Fraction(1, n_rna) for _ in range(n_rna))
    denominator = 2 * n_rna
    losses = [midrank_loss_numerators(row) for row in values]
    differences = {
        (left, right): tuple(
            Fraction(int(value), denominator)
            for value in (losses[left] - losses[right])
        )
        for left in range(len(candidate_ids))
        for right in range(len(candidate_ids))
        if left != right
    }
    rows: list[dict[str, object]] = []
    cache: dict[tuple[Fraction, tuple[Fraction, ...]], ExactOracleResult] = {}
    rectangular_cache: dict[
        tuple[Fraction, tuple[Fraction, ...]], tuple[Fraction | None, str]
    ] = {}
    for tau in taus:
        if not isinstance(tau, Fraction) or not 0 <= tau <= 1:
            raise ValueError("taus must be exact Fractions in [0, 1]")
        geometry = prepare_mc_tv_pu_oracle_exact(q, tuple(p0), tau, 0, 1)
        for left, candidate in enumerate(candidate_ids):
            for right, comparator in enumerate(candidate_ids):
                if left == right:
                    lower = upper = Fraction(0)
                    certified = True
                    status = "gap_met"
                    evaluations = 0
                    prior_supremum = fraction_text(geometry.prior_supremum)
                    rectangular_value: Fraction | None = Fraction(0)
                    rectangular_status = "finite"
                    empirical_value = Fraction(0)
                else:
                    difference = differences[(left, right)]
                    key = (tau, difference)
                    if key not in cache:
                        if oracle is solve_mc_tv_pu_oracle_exact:
                            cache[key] = solve_mc_tv_pu_oracle_exact_prepared(
                                geometry,
                                difference,
                                absolute_gap=EXACT_GAP,
                                max_evaluations=MAX_EXACT_EVALUATIONS,
                            )
                        else:
                            cache[key] = oracle(
                                q,
                                tuple(p0),
                                difference,
                                tau,
                                0,
                                1,
                                absolute_gap=EXACT_GAP,
                                max_evaluations=MAX_EXACT_EVALUATIONS,
                            )
                    result = cache[key]
                    lower = result.lower_bound
                    upper = result.upper_bound
                    certified = bool(result.enclosure_valid and result.certified)
                    status = str(result.diagnostics.get("status", "missing_status"))
                    evaluations = int(result.evaluations)
                    prior_supremum = fraction_text(result.prior_supremum)
                    if key not in rectangular_cache:
                        rectangular_cache[key] = matched_rectangular_value_exact(
                            geometry, difference
                        )
                    rectangular_value, rectangular_status = rectangular_cache[key]
                    empirical_value = sum(
                        (
                            mass * value
                            for mass, value in zip(p0, difference, strict=True)
                        ),
                        Fraction(0),
                    )
                rows.append(
                    {
                        "center": center,
                        "tau": fraction_text(tau),
                        "drug_id": int(drug_id),
                        "candidate": candidate,
                        "comparator": comparator,
                        "lower_bound": fraction_text(lower),
                        "upper_bound": fraction_text(upper),
                        "certified": certified,
                        "status": status,
                        "evaluations": evaluations,
                        "prior_supremum": prior_supremum,
                        "rectangular_value": (
                            None
                            if rectangular_value is None
                            else fraction_text(rectangular_value)
                        ),
                        "rectangular_status": rectangular_status,
                        "empirical_value": fraction_text(empirical_value),
                    }
                )
    return pd.DataFrame(rows, columns=CHECKPOINT_COLUMNS)


def _empirical_center_values(
    scores: np.ndarray,
    candidate_ids: Sequence[str],
    p0: Sequence[Fraction],
) -> dict[tuple[str, str], str]:
    """Return exact empirical risks for every directed candidate pair."""

    values = np.asarray(scores, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] != len(candidate_ids):
        raise ValueError("scores must match the frozen candidate list")
    if values.shape[1] != len(p0) or sum(p0, Fraction(0)) != 1:
        raise ValueError("p0 must be an exact probability vector over all RNAs")
    losses = [midrank_loss_numerators(row) for row in values]
    denominator = 2 * values.shape[1]
    result: dict[tuple[str, str], str] = {}
    for left, candidate in enumerate(candidate_ids):
        for right, comparator in enumerate(candidate_ids):
            if left == right:
                value = Fraction(0)
            else:
                value = sum(
                    (
                        mass * Fraction(int(delta), denominator)
                        for mass, delta in zip(
                            p0, losses[left] - losses[right], strict=True
                        )
                    ),
                    Fraction(0),
                )
            result[(candidate, comparator)] = fraction_text(value)
    return result


def _validate_checkpoint_empirical_values(
    frame: pd.DataFrame,
    *,
    path: Path,
    scores: np.ndarray,
    candidate_ids: Sequence[str],
    p0: Sequence[Fraction],
) -> None:
    """Reject a checkpoint whose empirical ablation uses another center."""

    expected = _empirical_center_values(scores, candidate_ids, p0)
    expected_column = [
        expected[(candidate, comparator)]
        for candidate, comparator in zip(
            frame["candidate"], frame["comparator"], strict=True
        )
    ]
    mismatches = sum(
        observed != wanted
        for observed, wanted in zip(
            frame["empirical_value"], expected_column, strict=True
        )
    )
    if mismatches:
        raise ValueError(
            "checkpoint empirical values do not match its source center: "
            f"{path} ({mismatches} mismatches)"
        )


def iter_fold_score_matrices(
    score_dir: Path,
    *,
    candidate_ids: Sequence[str],
    fold_count: int,
    n_rna_candidates: int,
) -> Iterator[tuple[int, np.ndarray]]:
    """Load one fold at a time and yield model-by-RNA scores for each drug."""

    for fold in range(fold_count):
        reference_coordinates: np.ndarray | None = None
        fold_scores: list[np.ndarray] = []
        for model in candidate_ids:
            path = score_partition_path(score_dir, model=model, fold=fold)
            frame = pd.read_parquet(
                path, columns=["fold", "drug_id", "rna_id", "score"]
            ).sort_values(["drug_id", "rna_id"], kind="mergesort")
            coordinates = frame[["drug_id", "rna_id"]].to_numpy(dtype=np.int64)
            if reference_coordinates is None:
                reference_coordinates = coordinates
            elif not np.array_equal(reference_coordinates, coordinates):
                raise ValueError(f"partition coordinate mismatch for model {model!r}, fold {fold}")
            if not (frame["fold"].to_numpy(dtype=np.int64) == fold).all():
                raise ValueError(f"partition fold mismatch for model {model!r}, fold {fold}")
            fold_scores.append(frame["score"].to_numpy(dtype=np.float64))
        if reference_coordinates is None:
            raise AssertionError("frozen candidate list unexpectedly empty")
        drugs, counts = np.unique(reference_coordinates[:, 0], return_counts=True)
        if not (counts == n_rna_candidates).all():
            raise ValueError(f"fold {fold} does not contain a full RNA grid per drug")
        expected_rnas = np.tile(np.arange(n_rna_candidates, dtype=np.int64), len(drugs))
        if not np.array_equal(reference_coordinates[:, 1], expected_rnas):
            raise ValueError(f"fold {fold} RNA order or coverage is invalid")
        stacked = np.stack(fold_scores, axis=0).reshape(
            len(candidate_ids), len(drugs), n_rna_candidates
        )
        for drug_index, drug_id in enumerate(drugs):
            yield int(drug_id), stacked[:, drug_index, :]


def checkpoint_path(output_dir: Path, *, center: str, drug_id: int) -> Path:
    """Return the immutable per-drug checkpoint path."""

    if center not in CENTER_MODES:
        raise ValueError(f"unknown center mode {center!r}")
    return output_dir / "oracle" / f"center={center}" / f"drug={drug_id}.parquet"


def write_checkpoint(frame: pd.DataFrame, path: Path) -> None:
    """Atomically create one checkpoint without overwriting prior evidence."""

    if tuple(frame.columns) != CHECKPOINT_COLUMNS:
        raise ValueError("checkpoint frame does not satisfy the frozen schema")
    if path.exists():
        raise FileExistsError(f"checkpoint already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp.parquet")
    if temporary.exists():
        raise FileExistsError(f"unfinished checkpoint requires manual review: {temporary}")
    frame.to_parquet(temporary, index=False, compression="zstd")
    temporary.replace(path)


def validate_checkpoint(
    path: Path,
    *,
    center: str,
    drug_id: int,
    candidate_ids: Sequence[str],
    taus: Sequence[Fraction] = TAU_GRID,
) -> None:
    """Validate an existing checkpoint before a resumed run skips it."""

    frame = pd.read_parquet(path)
    if tuple(frame.columns) != CHECKPOINT_COLUMNS:
        raise ValueError(f"checkpoint schema mismatch: {path}")
    expected_rows = len(candidate_ids) ** 2 * len(taus)
    if len(frame) != expected_rows:
        raise ValueError(f"checkpoint row count mismatch: {path}")
    if set(frame["center"]) != {center} or set(frame["drug_id"]) != {drug_id}:
        raise ValueError(f"checkpoint identity mismatch: {path}")
    if set(frame["candidate"]) != set(candidate_ids):
        raise ValueError(f"checkpoint candidate coverage mismatch: {path}")
    if set(frame["comparator"]) != set(candidate_ids):
        raise ValueError(f"checkpoint comparator coverage mismatch: {path}")
    if set(frame["tau"]) != {fraction_text(tau) for tau in taus}:
        raise ValueError(f"checkpoint tau coverage mismatch: {path}")
    identity_columns = ["tau", "candidate", "comparator"]
    if frame.duplicated(identity_columns).any():
        raise ValueError(f"checkpoint contains duplicate selector cells: {path}")
    observed = set(map(tuple, frame[identity_columns].itertuples(index=False, name=None)))
    expected = {
        (fraction_text(tau), candidate, comparator)
        for tau in taus
        for candidate in candidate_ids
        for comparator in candidate_ids
    }
    if observed != expected:
        raise ValueError(f"checkpoint selector-cell coverage mismatch: {path}")
    for row in frame.itertuples(index=False):
        lower = parse_fraction(row.lower_bound)
        upper = parse_fraction(row.upper_bound)
        if lower > upper:
            raise ValueError(f"checkpoint contains an inverted interval: {path}")
        if upper - lower > EXACT_GAP:
            raise ValueError(f"checkpoint interval exceeds the frozen exact gap: {path}")


def aggregate_selector_intervals(
    frames: Iterable[pd.DataFrame], *, candidate_ids: Sequence[str]
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Aggregate certified drug intervals using one shared comparator."""

    ordered_candidates = tuple(candidate_ids)
    if not ordered_candidates or len(set(ordered_candidates)) != len(ordered_candidates):
        raise ValueError("candidate_ids must be a nonempty unique sequence")
    sums: dict[tuple[str, str, str, str], list[Fraction | int]] = {}
    rectangular_sums: dict[tuple[str, str, str, str], Fraction] = defaultdict(Fraction)
    rectangular_unbounded: set[tuple[str, str, str, str]] = set()
    empirical_sums: dict[tuple[str, str, str, str], Fraction] = defaultdict(Fraction)
    drugs_by_cell: dict[tuple[str, str, str, str], set[int]] = defaultdict(set)
    for frame in frames:
        if tuple(frame.columns) != CHECKPOINT_COLUMNS:
            raise ValueError("oracle checkpoint has an unexpected schema")
        for row in frame.itertuples(index=False):
            if not bool(row.certified) or row.status != "gap_met":
                raise ValueError(
                    "uncertified oracle result blocks selector aggregation: "
                    f"drug={row.drug_id}, candidate={row.candidate}, "
                    f"comparator={row.comparator}, tau={row.tau}"
                )
            if row.candidate not in ordered_candidates or row.comparator not in ordered_candidates:
                raise ValueError("checkpoint contains a candidate outside the frozen library")
            lower = parse_fraction(row.lower_bound)
            upper = parse_fraction(row.upper_bound)
            if lower > upper:
                raise ValueError("oracle checkpoint has an inverted interval")
            key = (row.center, row.tau, row.candidate, row.comparator)
            if key not in sums:
                sums[key] = [Fraction(0), Fraction(0), 0]
            sums[key][0] += lower
            sums[key][1] += upper
            sums[key][2] += 1
            if row.rectangular_status == "unbounded":
                if row.rectangular_value is not None:
                    raise ValueError("unbounded rectangular result must not have a value")
                rectangular_unbounded.add(key)
            elif row.rectangular_status == "finite":
                rectangular_sums[key] += parse_fraction(row.rectangular_value)
            else:
                raise ValueError("rectangular result has an unknown status")
            empirical_sums[key] += parse_fraction(row.empirical_value)
            drug = int(row.drug_id)
            if drug in drugs_by_cell[key]:
                raise ValueError(f"duplicate drug result for selector cell {key!r}")
            drugs_by_cell[key].add(drug)

    center_tau = sorted({(key[0], key[1]) for key in sums})
    interval_rows: list[dict[str, object]] = []
    summary_cells: list[dict[str, object]] = []
    for center, tau in center_tau:
        pair_intervals: dict[tuple[str, str], tuple[Fraction, Fraction, int]] = {}
        drug_sets: list[frozenset[int]] = []
        for candidate in ordered_candidates:
            for comparator in ordered_candidates:
                key = (center, tau, candidate, comparator)
                if key not in sums:
                    raise ValueError(f"missing selector cell {key!r}")
                lower_sum, upper_sum, count = sums[key]
                assert isinstance(lower_sum, Fraction)
                assert isinstance(upper_sum, Fraction)
                assert isinstance(count, int)
                pair_intervals[(candidate, comparator)] = (
                    lower_sum / count,
                    upper_sum / count,
                    count,
                )
                drug_sets.append(frozenset(drugs_by_cell[key]))
        if len(set(drug_sets)) != 1:
            raise ValueError(f"selector cells use different drug sets for {center}, tau={tau}")

        candidate_intervals: dict[str, tuple[Fraction, Fraction]] = {}
        rectangular_values: dict[str, Fraction | None] = {}
        empirical_values: dict[str, Fraction] = {}
        row_by_candidate: dict[str, dict[str, object]] = {}
        for candidate in ordered_candidates:
            comparisons = {
                comparator: pair_intervals[(candidate, comparator)]
                for comparator in ordered_candidates
            }
            lower = max(value[0] for value in comparisons.values())
            upper = max(value[1] for value in comparisons.values())
            worst_lower = [
                comparator
                for comparator, value in comparisons.items()
                if value[0] == lower
            ]
            worst_upper = [
                comparator
                for comparator, value in comparisons.items()
                if value[1] == upper
            ]
            candidate_intervals[candidate] = (lower, upper)
            rectangular_comparisons: list[Fraction | None] = []
            empirical_comparisons: list[Fraction] = []
            for comparator in ordered_candidates:
                key = (center, tau, candidate, comparator)
                count = pair_intervals[(candidate, comparator)][2]
                rectangular_comparisons.append(
                    None
                    if key in rectangular_unbounded
                    else rectangular_sums[key] / count
                )
                empirical_comparisons.append(empirical_sums[key] / count)
            rectangular_values[candidate] = (
                None
                if any(value is None for value in rectangular_comparisons)
                else max(
                    value
                    for value in rectangular_comparisons
                    if value is not None
                )
            )
            empirical_values[candidate] = max(empirical_comparisons)
            row = {
                "center": center,
                "tau": tau,
                "candidate": candidate,
                "drug_count": comparisons[ordered_candidates[0]][2],
                "lower_V": fraction_text(lower),
                "upper_V": fraction_text(upper),
                "lower_V_float": float(lower),
                "upper_V_float": float(upper),
                "worst_comparator_lower": json.dumps(worst_lower),
                "worst_comparator_upper": json.dumps(worst_upper),
                "rectangular_V": (
                    "infinity"
                    if rectangular_values[candidate] is None
                    else fraction_text(rectangular_values[candidate])
                ),
                "empirical_V": fraction_text(empirical_values[candidate]),
            }
            interval_rows.append(row)
            row_by_candidate[candidate] = row

        minimum_upper = min(value[1] for value in candidate_intervals.values())
        possibly_optimal = [
            candidate
            for candidate in ordered_candidates
            if candidate_intervals[candidate][0] <= minimum_upper
        ]
        certifiably_unique = []
        for candidate in ordered_candidates:
            other_lower = min(
                candidate_intervals[other][0]
                for other in ordered_candidates
                if other != candidate
            )
            if candidate_intervals[candidate][1] < other_lower:
                certifiably_unique.append(candidate)
        finite_rectangular = {
            candidate: value
            for candidate, value in rectangular_values.items()
            if value is not None
        }
        if finite_rectangular:
            minimum_rectangular = min(finite_rectangular.values())
            rectangular_optimal = [
                candidate
                for candidate in ordered_candidates
                if rectangular_values[candidate] == minimum_rectangular
            ]
        else:
            minimum_rectangular = None
            rectangular_optimal = list(ordered_candidates)
        minimum_empirical = min(empirical_values.values())
        empirical_optimal = [
            candidate
            for candidate in ordered_candidates
            if empirical_values[candidate] == minimum_empirical
        ]
        for candidate, row in row_by_candidate.items():
            row["possibly_optimal"] = candidate in possibly_optimal
            row["certifiably_unique"] = candidate in certifiably_unique
            row["rectangular_optimal"] = candidate in rectangular_optimal
            row["empirical_optimal"] = candidate in empirical_optimal
        summary_cells.append(
            {
                "center": center,
                "tau": tau,
                "possibly_optimal": possibly_optimal,
                "certifiably_unique": certifiably_unique,
                "minimum_upper": fraction_text(minimum_upper),
                "matched_rectangular": {
                    "optimal": rectangular_optimal,
                    "minimum": (
                        "infinity"
                        if minimum_rectangular is None
                        else fraction_text(minimum_rectangular)
                    ),
                },
                "empirical_center": {
                    "optimal": empirical_optimal,
                    "minimum": fraction_text(minimum_empirical),
                },
            }
        )
    all_cells_uninformative = bool(summary_cells) and all(
        len(cell["possibly_optimal"]) == len(ordered_candidates)
        for cell in summary_cells
    )
    return pd.DataFrame(interval_rows), {
        "status": "PASS",
        "selection_rule": "shared-comparator compatible minimax regret",
        "practical_selection_gate": (
            "NO_SELECTION_VALUE"
            if all_cells_uninformative
            else "AT_LEAST_ONE_CELL_NARROWS_LIBRARY"
        ),
        "cells": summary_cells,
    }


def run_configuration(
    *,
    root: Path,
    score_dir: Path,
    pair_matrix_path: Path,
    manifest_path: Path,
    source_table: Path,
) -> dict[str, object]:
    """Build the immutable configuration record for one formal run."""

    score_files = sorted(score_dir.glob("model=*/fold=*.parquet"))
    return {
        "schema_version": 3,
        "checkpoint_semantics": "center_specific_empirical_v1",
        "analysis_status": "FROZEN_BEFORE_SELECTOR_RESULTS",
        "ablations": [
            "prior_clipped_rectangular_without_coordinate_compatibility",
            "empirical_center",
        ],
        "center_modes": list(CENTER_MODES),
        "tau_grid": [fraction_text(tau) for tau in TAU_GRID],
        "loss": "exact_midrank_pairwise_0_0.5_1",
        "prior_interval": ["0/1", "1/1"],
        "exact_gap": fraction_text(EXACT_GAP),
        "max_exact_evaluations": MAX_EXACT_EVALUATIONS,
        "root": str(root.resolve()),
        "pair_matrix": str(pair_matrix_path.resolve()),
        "pair_matrix_sha256": file_sha256(pair_matrix_path),
        "candidate_manifest": str(manifest_path.resolve()),
        "candidate_manifest_sha256": file_sha256(manifest_path),
        "source_table": str(source_table.resolve()),
        "source_table_sha256": file_sha256(source_table),
        "score_dir": str(score_dir.resolve()),
        "score_partitions": [
            {
                "path": str(path.relative_to(score_dir)),
                "sha256": file_sha256(path),
            }
            for path in score_files
        ],
    }


def write_or_validate_run_configuration(output_dir: Path, config: dict[str, object]) -> None:
    """Create a run record or require byte-equivalent semantics on resume."""

    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "run_config.json"
    serialized = json.dumps(config, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if path.exists():
        if path.read_text() != serialized:
            raise ValueError("existing output directory belongs to a different frozen run")
        return
    first_checkpoint = next(
        (output_dir / "oracle").glob("center=*/drug=*.parquet"), None
    )
    if first_checkpoint is not None:
        raise ValueError(
            "checkpoint files exist without run_config.json; explicit migration required"
        )
    path.write_text(serialized)


def validate_checkpoint_migration_source(
    source_dir: Path, expected_config: dict[str, object]
) -> dict[str, object]:
    """Require a schema-v2 source with otherwise identical frozen semantics."""

    path = source_dir / "run_config.json"
    if not path.is_file():
        raise ValueError("checkpoint migration source lacks run_config.json")
    source_config = json.loads(path.read_text())
    expected_legacy = dict(expected_config)
    expected_legacy.pop("checkpoint_semantics", None)
    expected_legacy["schema_version"] = 2
    if source_config != expected_legacy:
        raise ValueError("checkpoint migration source has incompatible frozen semantics")
    unfinished = sorted(source_dir.rglob("*.tmp.parquet")) + sorted(
        source_dir.rglob("*.partial")
    )
    if unfinished:
        raise ValueError("checkpoint migration source contains unfinished files")
    return source_config


def write_or_validate_text(path: Path, content: str) -> None:
    """Create a final text artifact, or verify an identical prior artifact."""

    if path.exists():
        if path.read_text() != content:
            raise ValueError(f"existing final artifact differs from recomputation: {path}")
        return
    path.write_text(content)


def _migrated_checkpoint_frame(
    frame: pd.DataFrame,
    *,
    path: Path,
    center: str,
    scores: np.ndarray,
    candidate_ids: Sequence[str],
    p0: Sequence[Fraction],
) -> tuple[pd.DataFrame, int]:
    """Correct the known v2 empirical reuse bug and reject any other drift."""

    expected = _empirical_center_values(scores, candidate_ids, p0)
    expected_column = [
        expected[(candidate, comparator)]
        for candidate, comparator in zip(
            frame["candidate"], frame["comparator"], strict=True
        )
    ]
    mismatch = np.asarray(
        [
            observed != wanted
            for observed, wanted in zip(
                frame["empirical_value"], expected_column, strict=True
            )
        ],
        dtype=bool,
    )
    allowed = (
        (center == "positive_pair_equal")
        & (frame["tau"].to_numpy() == "1/1")
        & (frame["candidate"].to_numpy() != frame["comparator"].to_numpy())
    )
    if bool(np.any(mismatch & ~allowed)):
        raise ValueError(
            "checkpoint migration observed drift outside the known v2 empirical bug: "
            f"{path}"
        )
    migrated = frame.copy()
    migrated["empirical_value"] = expected_column
    _validate_checkpoint_empirical_values(
        migrated,
        path=path,
        scores=scores,
        candidate_ids=candidate_ids,
        p0=p0,
    )
    return migrated, int(mismatch.sum())


def migrate_checkpoint_directory(
    score_items: Iterable[tuple[int, np.ndarray]],
    *,
    candidate_ids: tuple[str, ...],
    sources: Mapping[int, dict[str, set[int]]],
    expected_drug_ids: Sequence[int],
    n_rna: int,
    source_dir: Path,
    output_dir: Path,
) -> dict[str, object]:
    """Import validated schema-v2 checkpoints into a new immutable v3 directory."""

    source_resolved = source_dir.resolve()
    output_resolved = output_dir.resolve()
    if (
        source_resolved == output_resolved
        or output_resolved.is_relative_to(source_resolved)
        or source_resolved.is_relative_to(output_resolved)
    ):
        raise ValueError("checkpoint migration requires isolated source and output trees")
    items = list(score_items)
    observed_drug_ids = [drug_id for drug_id, _ in items]
    if len(observed_drug_ids) != len(set(observed_drug_ids)):
        raise ValueError("checkpoint migration score items contain duplicate drugs")
    if set(observed_drug_ids) != set(expected_drug_ids):
        raise ValueError("checkpoint migration score items do not cover expected drugs")
    items.sort(key=lambda item: item[0])
    expected_paths = {
        checkpoint_path(source_dir, center=center, drug_id=drug_id).resolve()
        for center in CENTER_MODES
        for drug_id in expected_drug_ids
    }
    oracle_root = source_dir / "oracle"
    oracle_files = sorted(path for path in oracle_root.rglob("*") if path.is_file())
    unexpected = [path for path in oracle_files if path.resolve() not in expected_paths]
    if unexpected:
        raise ValueError("checkpoint migration source contains unexpected oracle files")
    source_paths = oracle_files

    seen: set[Path] = set()
    corrected_cells = 0
    file_records: list[dict[str, object]] = []
    for drug_id, score_matrix in items:
        if drug_id not in sources:
            raise ValueError(f"checkpoint migration lacks sources for drug {drug_id}")
        profiles = {
            center: source_center_profile(
                sources[drug_id], n_rna_candidates=n_rna, mode=center
            )
            for center in CENTER_MODES
        }
        proposals: list[tuple[Path, Path, pd.DataFrame, int]] = []
        for center in CENTER_MODES:
            source_path = checkpoint_path(
                source_dir, center=center, drug_id=drug_id
            )
            if not source_path.exists():
                continue
            seen.add(source_path.resolve())
            validate_checkpoint(
                source_path,
                center=center,
                drug_id=drug_id,
                candidate_ids=candidate_ids,
            )
            migrated, changes = _migrated_checkpoint_frame(
                pd.read_parquet(source_path),
                path=source_path,
                center=center,
                scores=score_matrix,
                candidate_ids=candidate_ids,
                p0=profiles[center],
            )
            proposals.append(
                (
                    source_path,
                    checkpoint_path(output_dir, center=center, drug_id=drug_id),
                    migrated,
                    changes,
                )
            )

        for _, target_path, migrated, _ in proposals:
            if target_path.exists():
                validate_checkpoint(
                    target_path,
                    center=str(migrated["center"].iloc[0]),
                    drug_id=drug_id,
                    candidate_ids=candidate_ids,
                )
                pd.testing.assert_frame_equal(
                    pd.read_parquet(target_path),
                    migrated,
                    check_dtype=True,
                    check_exact=True,
                    check_like=False,
                )

        for source_path, target_path, migrated, changes in proposals:
            if not target_path.exists():
                write_checkpoint(migrated, target_path)
            corrected_cells += changes
            file_records.append(
                {
                    "path": str(source_path.relative_to(source_dir)),
                    "source_sha256": file_sha256(source_path),
                    "target_sha256": file_sha256(target_path),
                    "corrected_empirical_cells": changes,
                }
            )

    if seen != {path.resolve() for path in source_paths}:
        raise ValueError("checkpoint migration did not visit every source checkpoint")
    file_records.sort(key=lambda record: str(record["path"]))
    canonical_records = json.dumps(
        file_records, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode()
    return {
        "schema_version": 1,
        "policy": "preserve_non_empirical_cells_and_recompute_center_empirical",
        "source_dir": str(source_dir.resolve()),
        "output_dir": str(output_dir.resolve()),
        "source_checkpoint_count": len(source_paths),
        "target_checkpoint_count": len(file_records),
        "corrected_empirical_cells": corrected_cells,
        "non_empirical_columns_exact": True,
        "file_manifest_sha256": hashlib.sha256(canonical_records).hexdigest(),
    }


def evaluate_drug_checkpoints(
    *,
    drug_id: int,
    score_matrix: np.ndarray,
    candidate_ids: tuple[str, ...],
    source_sets: dict[str, set[int]],
    n_rna: int,
    output_dir: Path,
) -> tuple[int, int]:
    """Compute or validate both immutable center checkpoints for one drug."""

    profiles = {
        center: source_center_profile(
            source_sets, n_rna_candidates=n_rna, mode=center
        )
        for center in CENTER_MODES
    }
    existing_frames: dict[str, pd.DataFrame] = {}
    for center in CENTER_MODES:
        path = checkpoint_path(output_dir, center=center, drug_id=drug_id)
        if not path.exists():
            continue
        validate_checkpoint(
            path,
            center=center,
            drug_id=drug_id,
            candidate_ids=candidate_ids,
        )
        frame = pd.read_parquet(path)
        _validate_checkpoint_empirical_values(
            frame,
            path=path,
            scores=score_matrix,
            candidate_ids=candidate_ids,
            p0=profiles[center],
        )
        existing_frames[center] = frame

    frames_by_profile: dict[tuple[Fraction, ...], pd.DataFrame] = {}
    tau_one_frame: pd.DataFrame | None = None
    created = 0
    for center in CENTER_MODES:
        p0 = profiles[center]
        path = checkpoint_path(output_dir, center=center, drug_id=drug_id)
        if center in existing_frames:
            frame = existing_frames[center]
            frames_by_profile[p0] = frame
            if tau_one_frame is None:
                tau_one_frame = frame.loc[frame["tau"] == "1/1"].copy()
            continue
        if p0 in frames_by_profile:
            frame = frames_by_profile[p0].copy()
            frame["center"] = center
        else:
            taus = TAU_GRID if tau_one_frame is None else TAU_GRID[:-1]
            frame = evaluate_drug_center(
                drug_id=drug_id,
                center=center,
                scores=score_matrix,
                candidate_ids=candidate_ids,
                p0=p0,
                taus=taus,
            )
            if tau_one_frame is None:
                tau_one_frame = frame.loc[frame["tau"] == "1/1"].copy()
            else:
                shared_tau_one = tau_one_frame.copy()
                shared_tau_one["center"] = center
                empirical_values = _empirical_center_values(
                    score_matrix, candidate_ids, p0
                )
                shared_tau_one["empirical_value"] = [
                    empirical_values[(candidate, comparator)]
                    for candidate, comparator in zip(
                        shared_tau_one["candidate"],
                        shared_tau_one["comparator"],
                        strict=True,
                    )
                ]
                frame = pd.concat([frame, shared_tau_one], ignore_index=True)
            frames_by_profile[p0] = frame
        write_checkpoint(frame, path)
        created += 1
    return drug_id, created


def evaluate_all_drugs(
    score_items: Iterable[tuple[int, np.ndarray]],
    *,
    candidate_ids: tuple[str, ...],
    sources: Mapping[int, dict[str, set[int]]],
    n_rna: int,
    output_dir: Path,
    workers: int,
) -> set[int]:
    """Run independent drug checkpoints serially or in bounded processes."""

    if workers <= 0:
        raise ValueError("workers must be a positive integer")
    observed: set[int] = set()
    if workers == 1:
        for drug_id, score_matrix in score_items:
            observed.add(drug_id)
            completed, created = evaluate_drug_checkpoints(
                drug_id=drug_id,
                score_matrix=score_matrix,
                candidate_ids=candidate_ids,
                source_sets=sources[drug_id],
                n_rna=n_rna,
                output_dir=output_dir,
            )
            print(
                f"[mc-tv-pu] completed drug={completed} new_checkpoints={created}",
                flush=True,
            )
        return observed

    with ProcessPoolExecutor(max_workers=workers) as executor:
        pending = set()
        for drug_id, score_matrix in score_items:
            observed.add(drug_id)
            pending.add(
                executor.submit(
                    evaluate_drug_checkpoints,
                    drug_id=drug_id,
                    score_matrix=score_matrix,
                    candidate_ids=candidate_ids,
                    source_sets=sources[drug_id],
                    n_rna=n_rna,
                    output_dir=output_dir,
                )
            )
            if len(pending) >= 2 * workers:
                done, pending = wait(pending, return_when=FIRST_COMPLETED)
                for future in done:
                    completed, created = future.result()
                    print(
                        f"[mc-tv-pu] completed drug={completed} "
                        f"new_checkpoints={created}",
                        flush=True,
                    )
        while pending:
            done, pending = wait(pending, return_when=FIRST_COMPLETED)
            for future in done:
                completed, created = future.result()
                print(
                    f"[mc-tv-pu] completed drug={completed} new_checkpoints={created}",
                    flush=True,
                )
    return observed


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--score-dir", type=Path, required=True)
    parser.add_argument("--pair-matrix", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--source-table", type=Path, required=True)
    parser.add_argument("--dataset-dir", type=Path, default=Path("Dataset"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--checkpoint-source-dir", type=Path)
    parser.add_argument("--migration-only", action="store_true")
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    if args.migration_only and args.checkpoint_source_dir is None:
        parser.error("--migration-only requires --checkpoint-source-dir")

    manifest = json.loads(args.manifest.read_text())
    candidate_ids, manifest_issues = validate_candidate_manifest(
        manifest, root=args.root, pair_matrix_path=args.pair_matrix
    )
    if manifest_issues:
        raise ValueError("candidate manifest invalid: " + "; ".join(manifest_issues))
    dataset = manifest["dataset"]
    fold_count = int(dataset["fold_count"])
    pair_matrix = np.load(args.pair_matrix, allow_pickle=False)
    active_drugs = tuple(np.flatnonzero((pair_matrix > 0).any(axis=0)).tolist())
    n_rna = int(pair_matrix.shape[0])
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
    missing_sources = sorted(set(active_drugs) - set(sources))
    if missing_sources:
        raise ValueError(
            "active drugs lack source-backed anchors: "
            + ", ".join(map(str, missing_sources))
        )

    config = run_configuration(
        root=args.root,
        score_dir=args.score_dir,
        pair_matrix_path=args.pair_matrix,
        manifest_path=args.manifest,
        source_table=args.source_table,
    )
    config["candidate_ids"] = list(candidate_ids)
    config["score_input_audit"] = audit.to_dict()
    if args.checkpoint_source_dir is not None:
        source_resolved = args.checkpoint_source_dir.resolve()
        output_resolved = args.output_dir.resolve()
        if (
            source_resolved == output_resolved
            or output_resolved.is_relative_to(source_resolved)
            or source_resolved.is_relative_to(output_resolved)
        ):
            raise ValueError(
                "checkpoint migration requires isolated source and output trees"
            )
        validate_checkpoint_migration_source(args.checkpoint_source_dir, config)
        source_config_path = args.checkpoint_source_dir / "run_config.json"
        config["checkpoint_import"] = {
            "source_dir": str(args.checkpoint_source_dir.resolve()),
            "source_schema_version": 2,
            "source_run_config_sha256": file_sha256(source_config_path),
            "policy": "preserve_non_empirical_cells_and_recompute_center_empirical",
        }
    write_or_validate_run_configuration(args.output_dir, config)

    if args.checkpoint_source_dir is not None:
        migration_report = migrate_checkpoint_directory(
            iter_fold_score_matrices(
                args.score_dir,
                candidate_ids=candidate_ids,
                fold_count=fold_count,
                n_rna_candidates=n_rna,
            ),
            candidate_ids=tuple(candidate_ids),
            sources=sources,
            expected_drug_ids=active_drugs,
            n_rna=n_rna,
            source_dir=args.checkpoint_source_dir,
            output_dir=args.output_dir,
        )
        write_or_validate_text(
            args.output_dir / "checkpoint_migration_report.json",
            json.dumps(migration_report, ensure_ascii=False, indent=2) + "\n",
        )
        if args.migration_only:
            print(json.dumps(migration_report, ensure_ascii=False, indent=2))
            return 0

    observed_drugs = evaluate_all_drugs(
        iter_fold_score_matrices(
            args.score_dir,
            candidate_ids=candidate_ids,
            fold_count=fold_count,
            n_rna_candidates=n_rna,
        ),
        candidate_ids=tuple(candidate_ids),
        sources=sources,
        n_rna=n_rna,
        output_dir=args.output_dir,
        workers=args.workers,
    )
    if observed_drugs != set(active_drugs):
        raise ValueError("score partitions did not yield every active drug")

    paths = [
        checkpoint_path(args.output_dir, center=center, drug_id=drug_id)
        for center in CENTER_MODES
        for drug_id in active_drugs
    ]
    intervals, summary = aggregate_selector_intervals(
        (pd.read_parquet(path) for path in paths), candidate_ids=candidate_ids
    )
    write_or_validate_text(
        args.output_dir / "candidate_intervals.csv", intervals.to_csv(index=False)
    )
    write_or_validate_text(
        args.output_dir / "selector_summary.json",
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
