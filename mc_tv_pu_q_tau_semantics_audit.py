"""Audit source heterogeneity for the MC-TV-PU Q/p0/tau semantics gate."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

from generalization_audit import load_bundle
from source_balanced_pairwise_pilot import build_pair_sources


def probability_profile(indices: set[int]) -> dict[int, float]:
    """Return the uniform probability profile on a nonempty index set."""

    if not indices:
        raise ValueError("a source profile must contain at least one RNA")
    mass = 1.0 / len(indices)
    return {index: mass for index in indices}


def mixture_profile(profiles: list[dict[int, float]]) -> dict[int, float]:
    """Return the equal-weight mixture of nonempty probability profiles."""

    if not profiles:
        raise ValueError("at least one source profile is required")
    result: dict[int, float] = defaultdict(float)
    weight = 1.0 / len(profiles)
    for profile in profiles:
        for index, mass in profile.items():
            result[index] += weight * mass
    return dict(result)


def tv_distance(left: dict[int, float], right: dict[int, float]) -> float:
    """Compute total variation on the union of two sparse supports."""

    support = left.keys() | right.keys()
    value = 0.5 * sum(
        abs(left.get(i, 0.0) - right.get(i, 0.0)) for i in support
    )
    return min(1.0, max(0.0, value))


def source_sets_by_drug(
    pair_sources: dict[tuple[int, int], set[str]],
) -> dict[int, dict[str, set[int]]]:
    """Invert pair-to-PMID evidence into drug-to-PMID RNA supports."""

    result: dict[int, dict[str, set[int]]] = defaultdict(lambda: defaultdict(set))
    for (rna_id, drug_id), sources in pair_sources.items():
        for source in sources:
            result[drug_id][source].add(rna_id)
    return {drug: dict(sources) for drug, sources in result.items()}


def restrict_pair_sources_to_positive_matrix(
    pair_sources: dict[tuple[int, int], set[str]], positive_matrix: np.ndarray
) -> dict[tuple[int, int], set[str]]:
    """Keep only evidence pairs present in the frozen positive matrix."""

    if positive_matrix.ndim != 2:
        raise ValueError("positive_matrix must be two-dimensional")
    restricted: dict[tuple[int, int], set[str]] = {}
    for pair, sources in pair_sources.items():
        rna_id, drug_id = pair
        if not (0 <= rna_id < positive_matrix.shape[0]):
            raise ValueError(f"RNA index {rna_id} lies outside the frozen matrix")
        if not (0 <= drug_id < positive_matrix.shape[1]):
            raise ValueError(f"drug index {drug_id} lies outside the frozen matrix")
        if positive_matrix[pair] > 0:
            restricted[pair] = sources
    return restricted


def audit_source_heterogeneity(
    pair_sources: dict[tuple[int, int], set[str]], n_rna_candidates: int
) -> pd.DataFrame:
    """Compute per-drug observed-source TV diagnostics without model scores."""

    if n_rna_candidates <= 0:
        raise ValueError("n_rna_candidates must be positive")
    rows: list[dict[str, float | int]] = []
    for drug_id, source_sets in sorted(source_sets_by_drug(pair_sources).items()):
        if len(source_sets) < 2:
            continue
        source_ids = sorted(source_sets)
        profiles = [probability_profile(source_sets[source]) for source in source_ids]
        center = mixture_profile(profiles)
        source_to_center = [tv_distance(profile, center) for profile in profiles]
        pairwise = [
            tv_distance(profiles[left], profiles[right])
            for left, right in combinations(range(len(profiles)), 2)
        ]
        source_to_loso = []
        for held_out, profile in enumerate(profiles):
            remainder = mixture_profile(
                [candidate for index, candidate in enumerate(profiles) if index != held_out]
            )
            source_to_loso.append(tv_distance(profile, remainder))
        maximum_center_mass = max(center.values())
        rows.append(
            {
                "drug_id": int(drug_id),
                "source_count": len(profiles),
                "source_union_size": len(center),
                "min_source_to_center_tv": min(source_to_center),
                "mean_source_to_center_tv": float(np.mean(source_to_center)),
                "max_source_to_center_tv": max(source_to_center),
                "min_pairwise_source_tv": min(pairwise),
                "mean_pairwise_source_tv": float(np.mean(pairwise)),
                "max_pairwise_source_tv": max(pairwise),
                "min_source_to_loso_tv": min(source_to_loso),
                "mean_source_to_loso_tv": float(np.mean(source_to_loso)),
                "max_source_to_loso_tv": max(source_to_loso),
                "uniform_q_prior_supremum_tau0": 1.0
                / (n_rna_candidates * maximum_center_mass),
            }
        )
    return pd.DataFrame(rows)


def summarize_audit(
    per_drug: pd.DataFrame,
    *,
    source_table_mapped_pair_count: int,
    source_table_mapped_drug_count: int,
    frozen_positive_pair_count: int,
    used_pair_count: int,
    used_drug_count: int,
    n_rna_candidates: int,
) -> dict[str, object]:
    """Summarize the preregistered semantics gate."""

    if per_drug.empty:
        raise ValueError("at least one drug with two nonempty PMID sources is required")
    metric = per_drug["max_source_to_center_tv"].to_numpy(dtype=np.float64)
    loso = per_drug["max_source_to_loso_tv"].to_numpy(dtype=np.float64)
    quantiles = [0.0, 0.25, 0.5, 0.75, 1.0]
    metric_quantiles = np.quantile(metric, quantiles)
    loso_quantiles = np.quantile(loso, quantiles)
    median = float(metric_quantiles[2])
    return {
        "estimand": "uniform query over the fixed RNA candidate universe",
        "q_definition": {
            "type": "uniform_finite_candidate_design",
            "rna_candidate_count": n_rna_candidates,
            "mass_per_rna": 1.0 / n_rna_candidates,
            "not_claimed": "natural biological population distribution",
        },
        "p0_definition": "equal PMID mass, then equal unique-RNA mass within PMID",
        "tau_interpretation": "sensitivity axis; observed-source TV is diagnostic, not a coverage bound",
        "mapping": {
            "source_table_mapped_pairs": source_table_mapped_pair_count,
            "source_table_mapped_drugs": source_table_mapped_drug_count,
            "frozen_positive_pairs": frozen_positive_pair_count,
            "used_intersection_pairs": used_pair_count,
            "used_intersection_drugs": used_drug_count,
            "excluded_source_only_pairs": source_table_mapped_pair_count
            - used_pair_count,
            "excluded_source_only_drugs": source_table_mapped_drug_count
            - used_drug_count,
            "drugs_with_at_least_two_sources": int(len(per_drug)),
        },
        "max_source_to_center_tv_quantiles": dict(
            zip(["min", "q25", "median", "q75", "max"], map(float, metric_quantiles))
        ),
        "max_source_to_loso_tv_quantiles": dict(
            zip(["min", "q25", "median", "q75", "max"], map(float, loso_quantiles))
        ),
        "fraction_with_max_source_to_center_tv_at_most": {
            str(threshold): float(np.mean(metric <= threshold))
            for threshold in (0.25, 0.5, 0.75)
        },
        "uniform_q_prior_supremum_tau0_quantiles": dict(
            zip(
                ["min", "q25", "median", "q75", "max"],
                map(
                    float,
                    np.quantile(
                        per_drug["uniform_q_prior_supremum_tau0"].to_numpy(
                            dtype=np.float64
                        ),
                        quantiles,
                    ),
                ),
            )
        ),
        "gate": {
            "threshold": 0.5,
            "observed_median": median,
            "decision": (
                "tau_is_sensitivity_axis_only"
                if median > 0.5
                else "external_calibration_still_required"
            ),
            "population_coverage_certified": False,
        },
        "model_scores_read": False,
        "model_training_run": False,
    }


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", type=Path, default=Path("Dataset"))
    parser.add_argument(
        "--source-table",
        type=Path,
        default=Path("/Users/jackie/Downloads/DT_Curated.txt"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/mc-tv-pu-q-tau-semantics-audit-20261021"),
    )
    args = parser.parse_args()

    source_table_pair_sources = build_pair_sources(args.source_table, args.dataset_dir)
    frozen_bundle = load_bundle(args.dataset_dir, "Mi")
    pair_sources = restrict_pair_sources_to_positive_matrix(
        source_table_pair_sources, frozen_bundle.pair
    )
    rna_table = pd.read_excel(args.dataset_dir / "miRNA_with_sequence.xlsx")
    per_drug = audit_source_heterogeneity(pair_sources, len(rna_table))
    summary = summarize_audit(
        per_drug,
        source_table_mapped_pair_count=len(source_table_pair_sources),
        source_table_mapped_drug_count=len(
            {drug for _, drug in source_table_pair_sources}
        ),
        frozen_positive_pair_count=int(np.count_nonzero(frozen_bundle.pair > 0)),
        used_pair_count=len(pair_sources),
        used_drug_count=len({drug for _, drug in pair_sources}),
        n_rna_candidates=len(rna_table),
    )
    summary["source_table_sha256"] = file_sha256(args.source_table)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    per_drug.to_csv(args.output_dir / "per_drug.csv", index=False)
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
