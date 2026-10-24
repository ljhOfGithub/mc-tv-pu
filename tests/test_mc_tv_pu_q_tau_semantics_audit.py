from __future__ import annotations

import pytest
import numpy as np

from mc_tv_pu_q_tau_semantics_audit import (
    audit_source_heterogeneity,
    mixture_profile,
    probability_profile,
    restrict_pair_sources_to_positive_matrix,
    summarize_audit,
    tv_distance,
)


def test_sparse_profiles_and_tv_are_exact_on_overlap_example() -> None:
    first = probability_profile({0, 1})
    second = probability_profile({1, 2})
    center = mixture_profile([first, second])

    assert center == {0: 0.25, 1: 0.5, 2: 0.25}
    assert tv_distance(first, second) == pytest.approx(0.5)
    assert tv_distance(first, center) == pytest.approx(0.25)


def test_per_drug_audit_deduplicates_rna_within_each_source() -> None:
    pair_sources = {
        (0, 7): {"p1"},
        (1, 7): {"p1", "p2"},
        (2, 7): {"p2"},
        (0, 8): {"only"},
    }

    result = audit_source_heterogeneity(pair_sources, n_rna_candidates=4)

    assert result["drug_id"].tolist() == [7]
    row = result.iloc[0]
    assert row["source_count"] == 2
    assert row["source_union_size"] == 3
    assert row["max_source_to_center_tv"] == pytest.approx(0.25)
    assert row["max_pairwise_source_tv"] == pytest.approx(0.5)
    assert row["max_source_to_loso_tv"] == pytest.approx(0.5)
    assert row["uniform_q_prior_supremum_tau0"] == pytest.approx(0.5)


def test_source_mapping_is_restricted_to_frozen_positive_matrix() -> None:
    pair_sources = {
        (0, 0): {"kept"},
        (1, 0): {"source-only"},
        (1, 1): {"kept-too"},
    }
    positive_matrix = np.array([[1, 0], [0, 1]], dtype=np.float32)

    restricted = restrict_pair_sources_to_positive_matrix(
        pair_sources, positive_matrix
    )

    assert restricted == {(0, 0): {"kept"}, (1, 1): {"kept-too"}}


def test_summary_applies_preregistered_high_heterogeneity_gate() -> None:
    pair_sources = {
        (0, 9): {"p1"},
        (1, 9): {"p2"},
        (2, 9): {"p3"},
    }
    per_drug = audit_source_heterogeneity(pair_sources, n_rna_candidates=4)

    summary = summarize_audit(
        per_drug,
        source_table_mapped_pair_count=4,
        source_table_mapped_drug_count=2,
        frozen_positive_pair_count=3,
        used_pair_count=3,
        used_drug_count=1,
        n_rna_candidates=4,
    )

    assert summary["max_source_to_center_tv_quantiles"]["median"] == pytest.approx(
        2 / 3
    )
    assert summary["gate"]["decision"] == "tau_is_sensitivity_axis_only"
    assert summary["gate"]["population_coverage_certified"] is False
    assert summary["model_scores_read"] is False
    assert summary["model_training_run"] is False
    assert summary["max_source_to_loso_tv_quantiles"]["max"] <= 1.0
    assert summary["mapping"]["excluded_source_only_pairs"] == 1
    assert summary["mapping"]["excluded_source_only_drugs"] == 1
