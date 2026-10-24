from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import roc_auc_score

from mc_tv_pu_selector_evaluation import midrank_loss_numerators
from mc_tv_pu_source_isolated_validation import (
    TAU_GRID,
    SourceCrossFitEpisode,
    aggregate_episode_metrics,
    assign_sources_to_folds,
    bootstrap_crossfit_rule_ablation_performance,
    bootstrap_crossfit_set_valued_performance,
    bootstrap_set_valued_performance,
    build_source_crossfit_episodes,
    build_source_episodes,
    episode_coverage_summary,
    evaluate_crossfit_episode_scores,
    evaluate_episode_scores,
    evaluate_rule_ablation_claim_routing,
    evaluate_success_gate,
    expected_recall_at_k,
    source_crossfit_coverage_summary,
    tie_aware_pairwise_auc,
)


def test_source_fold_assignment_is_deterministic_and_global() -> None:
    first = assign_sources_to_folds(("p3", "p1", "p2"), fold_count=3, seed=17)
    second = assign_sources_to_folds(("p2", "p3", "p1"), fold_count=3, seed=17)

    assert first == second
    assert set(first) == {"p1", "p2", "p3"}
    assert all(0 <= fold < 3 for fold in first.values())


@pytest.mark.parametrize(
    "sources",
    (("", "p1"), (" p1", "p2"), ("p1 ", "p2")),
)
def test_source_fold_assignment_rejects_unnormalized_ids(
    sources: tuple[str, str]
) -> None:
    with pytest.raises(ValueError, match="nonempty normalized"):
        assign_sources_to_folds(sources, fold_count=3, seed=17)


def test_source_crossfit_excludes_heldout_sources_from_every_same_fold_anchor() -> None:
    pair_sources = {
        (0, 7): {"shared"},
        (1, 7): {"drug7-only"},
        (2, 8): {"shared"},
        (3, 8): {"drug8-only"},
        (4, 8): {"third"},
    }

    assignments, episodes_by_fold = build_source_crossfit_episodes(
        pair_sources, fold_count=3, seed=23
    )

    for fold, episodes in episodes_by_fold.items():
        heldout = {source for source, value in assignments.items() if value == fold}
        assert all(heldout.isdisjoint(episode.anchor_sources) for episode in episodes)
        assert all(set(episode.heldout_sources) <= heldout for episode in episodes)

    shared_fold = assignments["shared"]
    assert all(
        "shared" not in episode.anchor_sources
        for episode in episodes_by_fold[shared_fold]
    )


def test_source_crossfit_coverage_counts_drug_folds_and_unique_drugs() -> None:
    _, episodes_by_fold = build_source_crossfit_episodes(
        {
            (0, 7): {"p1"},
            (1, 7): {"p2"},
            (2, 8): {"p1"},
            (3, 8): {"p3"},
        },
        fold_count=3,
        seed=11,
    )

    summary = source_crossfit_coverage_summary(episodes_by_fold)

    assert summary["eligible_drug_fold_count"] >= summary["evaluable_drug_fold_count"]
    assert summary["evaluable_drug_count"] <= 2
    assert summary["validation_tier"] == "pilot"


def test_source_episodes_remove_anchor_overlap_from_heldout() -> None:
    pair_sources = {
        (0, 7): {"p1"},
        (1, 7): {"p1", "p2"},
        (2, 7): {"p2"},
        (0, 8): {"single"},
    }

    episodes = build_source_episodes(pair_sources)

    assert set(episodes) == {7}
    first, second = episodes[7]
    assert first.heldout_source == "p1"
    assert first.anchor_rnas == (1, 2)
    assert first.heldout_novel_rnas == (0,)
    assert second.heldout_source == "p2"
    assert second.anchor_rnas == (0, 1)
    assert second.heldout_novel_rnas == (2,)


def test_coverage_counts_episodes_without_double_counting_candidates() -> None:
    episodes = build_source_episodes(
        {
            (0, 7): {"p1"},
            (1, 7): {"p2"},
            (0, 8): {"p1", "p2"},
        }
    )

    summary = episode_coverage_summary(episodes)

    assert summary["multi_source_drug_count"] == 2
    assert summary["episode_count"] == 4
    assert summary["evaluable_episode_count"] == 2
    assert summary["empty_heldout_episode_count"] == 2
    assert summary["heldout_record_count"] == 2
    assert summary["validation_tier"] == "pilot"


def test_expected_recall_fractionally_allocates_boundary_ties() -> None:
    labels = np.array([0, 1, 1, 0])
    scores = np.array([1.0, 0.5, 0.5, 0.0])

    recall, actual_k = expected_recall_at_k(labels, scores, 2)

    assert actual_k == 2
    assert recall == pytest.approx(0.5)


def test_pairwise_auc_is_exactly_aligned_with_positive_midrank_loss() -> None:
    labels = np.array([1, 1, 0, 0])
    scores = np.array([0.9, 0.5, 0.5, 0.1])

    pairwise_auc = tie_aware_pairwise_auc(labels, scores)
    positive_count = int(labels.sum())
    negative_count = len(labels) - positive_count
    mean_positive_midrank_loss = (
        midrank_loss_numerators(scores)[labels == 1].mean() / (2 * len(labels))
    )
    identity_value = (
        negative_count * (1.0 - pairwise_auc) + positive_count / 2
    ) / len(labels)

    assert pairwise_auc == pytest.approx(0.875)
    assert mean_positive_midrank_loss == pytest.approx(identity_value)


@pytest.mark.parametrize(
    ("scores", "expected"),
    (
        ([0.9, 0.8, 0.2, 0.1], 1.0),
        ([0.2, 0.1, 0.9, 0.8], 0.0),
        ([0.5, 0.5, 0.5, 0.5], 0.5),
    ),
)
def test_pairwise_auc_handles_order_extremes_and_ties(
    scores: list[float], expected: float
) -> None:
    assert tie_aware_pairwise_auc([1, 1, 0, 0], scores) == expected


@pytest.mark.parametrize("labels", ([1, 1], [0, 0]))
def test_pairwise_auc_requires_both_recorded_and_unlabeled_candidates(
    labels: list[int],
) -> None:
    with pytest.raises(
        ValueError, match="requires positive and unlabeled candidates"
    ):
        tie_aware_pairwise_auc(labels, [0.9, 0.1])


def test_pairwise_auc_matches_standard_auc_across_exact_ties() -> None:
    generator = np.random.default_rng(20_261_023)
    for _ in range(100):
        labels = generator.integers(0, 2, size=17)
        if labels.min() == labels.max():
            labels[0] = 1 - labels[0]
        scores = generator.integers(-2, 3, size=17).astype(np.float64)

        assert tie_aware_pairwise_auc(labels, scores) == pytest.approx(
            roc_auc_score(labels, scores)
        )


def test_episode_metrics_exclude_anchors_and_keep_all_candidates() -> None:
    episode = build_source_episodes(
        {(0, 7): {"p1"}, (1, 7): {"p2"}}
    )[7][0]
    scores = np.array(
        [
            [0.9, 0.1, 0.2, 0.0],
            [0.1, 0.9, 0.2, 0.0],
        ]
    )

    metrics = evaluate_episode_scores(
        episode, scores=scores, candidate_ids=("good", "bad")
    )

    assert metrics["candidate"].tolist() == ["good", "bad"]
    assert set(metrics["candidate_count"]) == {3}
    assert set(metrics["positive_count"]) == {1}
    assert metrics.loc[0, "average_precision"] > metrics.loc[1, "average_precision"]
    assert metrics.loc[0, "pairwise_auc"] > metrics.loc[1, "pairwise_auc"]


def test_crossfit_episode_metrics_preserve_fold_identity() -> None:
    episode = SourceCrossFitEpisode(
        fold=1,
        drug_id=7,
        heldout_sources=("p2",),
        anchor_sources=("p1",),
        anchor_rnas=(0,),
        heldout_novel_rnas=(1,),
    )
    scores = np.array([[0.9, 0.1, 0.0], [0.1, 0.9, 0.0]])

    metrics = evaluate_crossfit_episode_scores(
        episode, scores=scores, candidate_ids=("forward", "reverse")
    )

    assert set(metrics["fold"]) == {episode.fold}
    assert "heldout_source" not in metrics
    assert set(metrics["candidate"]) == {"forward", "reverse"}


def test_episode_aggregation_weights_drugs_not_source_count() -> None:
    frame = pd.DataFrame(
        [
            {"drug_id": 1, "heldout_source": "a", "candidate": "m", "average_precision": 0.0, "pairwise_auc": 0.0, "recall_at_10": 0.0, "recall_at_50": 0.0},
            {"drug_id": 1, "heldout_source": "b", "candidate": "m", "average_precision": 1.0, "pairwise_auc": 1.0, "recall_at_10": 1.0, "recall_at_50": 1.0},
            {"drug_id": 2, "heldout_source": "c", "candidate": "m", "average_precision": 1.0, "pairwise_auc": 1.0, "recall_at_10": 1.0, "recall_at_50": 1.0},
        ]
    )

    result = aggregate_episode_metrics(frame)

    assert result.loc[result["drug_id"] == 1, "average_precision"].item() == 0.5
    assert result.loc[result["drug_id"] == 2, "average_precision"].item() == 1.0
    assert result["average_precision"].mean() == 0.75
    assert result["pairwise_auc"].mean() == 0.75


def test_set_valued_bootstrap_uses_worst_selected_candidate_and_is_reproducible() -> None:
    rows = []
    values = {
        0: {"a": 0.8, "b": 0.6, "base": 0.5, "history": 0.55},
        1: {"a": 0.7, "b": 0.65, "base": 0.5, "history": 0.60},
        2: {"a": 0.9, "b": 0.7, "base": 0.5, "history": 0.65},
    }
    for drug_id, candidates in values.items():
        for candidate, value in candidates.items():
            rows.append(
                {
                    "drug_id": drug_id,
                    "candidate": candidate,
                    "average_precision": value,
                    "pairwise_auc": value,
                    "recall_at_10": value,
                    "recall_at_50": value,
                }
            )
    frame = pd.DataFrame(rows)
    cells = [
        {
            "center": "pmid_equal",
            "tau": "0/1",
            "possibly_optimal": ["a", "b"],
        }
    ]

    first, first_config = bootstrap_set_valued_performance(
        frame,
        selector_cells=cells,
        candidate_ids=("a", "b", "base", "history"),
        baselines=("base", "history"),
        iterations=200,
        seed=17,
        batch_size=50,
    )
    second, second_config = bootstrap_set_valued_performance(
        frame,
        selector_cells=cells,
        candidate_ids=("a", "b", "base", "history"),
        baselines=("base", "history"),
        iterations=200,
        seed=17,
        batch_size=50,
    )

    against_base = first[first["baseline"] == "base"].iloc[0]
    assert against_base["average_precision_difference"] == pytest.approx(0.15)
    assert against_base["pairwise_auc_difference"] == pytest.approx(0.15)
    pd.testing.assert_frame_equal(first, second)
    assert first_config == second_config


def test_crossfit_bootstrap_uses_fold_specific_sets_before_drug_averaging() -> None:
    values = {
        (1, 0): {"a": 0.8, "b": 0.6, "base": 0.5, "history": 0.55},
        (1, 1): {"a": 0.9, "b": 0.7, "base": 0.5, "history": 0.60},
        (2, 0): {"a": 0.7, "b": 0.65, "base": 0.5, "history": 0.60},
    }
    rows = []
    for (drug_id, fold), candidates in values.items():
        for candidate, value in candidates.items():
            rows.append(
                {
                    "drug_id": drug_id,
                    "fold": fold,
                    "candidate": candidate,
                    "average_precision": value,
                    "pairwise_auc": value,
                    "recall_at_10": value,
                    "recall_at_50": value,
                }
            )
    cells = {
        0: [{"center": "pmid_equal", "tau": "0/1", "possibly_optimal": ["a", "b"]}],
        1: [{"center": "pmid_equal", "tau": "0/1", "possibly_optimal": ["a"]}],
    }

    result, config = bootstrap_crossfit_set_valued_performance(
        pd.DataFrame(rows),
        selector_cells_by_fold=cells,
        candidate_ids=("a", "b", "base", "history"),
        baselines=("base", "history"),
        iterations=200,
        seed=19,
        batch_size=50,
    )

    against_base = result[result["baseline"] == "base"].iloc[0]
    assert against_base["average_precision_difference"] == pytest.approx(0.2)
    assert against_base["pairwise_auc_difference"] == pytest.approx(0.2)
    assert config["fold_handling"] == (
        "worst_selected_within_fold_then_mean_folds_within_drug"
    )


def test_crossfit_rule_ablation_compares_worst_members_with_paired_drugs() -> None:
    values = {
        (1, 0): {"a": 0.9, "b": 0.4},
        (1, 1): {"a": 0.2, "b": 0.8},
        (2, 0): {"a": 0.8, "b": 0.3},
        (2, 1): {"a": 0.1, "b": 0.7},
    }
    rows = [
        {
            "drug_id": drug,
            "fold": fold,
            "candidate": candidate,
            "average_precision": value,
            "pairwise_auc": value,
            "recall_at_10": value,
            "recall_at_50": value,
        }
        for (drug, fold), candidates in values.items()
        for candidate, value in candidates.items()
    ]
    cells = {
        0: [
            {
                "center": "pmid_equal",
                "tau": "0/1",
                "possibly_optimal": ["a"],
                "matched_rectangular": {"optimal": ["b"]},
                "empirical_center": {"optimal": ["a"]},
            }
        ],
        1: [
            {
                "center": "pmid_equal",
                "tau": "0/1",
                "possibly_optimal": ["b"],
                "matched_rectangular": {"optimal": ["a"]},
                "empirical_center": {"optimal": ["a"]},
            }
        ],
    }

    result, config = bootstrap_crossfit_rule_ablation_performance(
        pd.DataFrame(rows),
        selector_cells_by_fold=cells,
        candidate_ids=("a", "b"),
        iterations=200,
        seed=23,
        batch_size=50,
    )

    indexed = result.set_index("ablation_rule")
    assert indexed.loc[
        "matched_rectangular", "average_precision_difference"
    ] == pytest.approx(0.55)
    assert indexed.loc[
        "matched_rectangular", "average_precision_simultaneous_lower_95"
    ] == pytest.approx(0.55)
    assert indexed.loc[
        "matched_rectangular", "average_precision_simultaneous_upper_95"
    ] == pytest.approx(0.55)
    assert indexed.loc[
        "empirical_center", "average_precision_difference"
    ] == pytest.approx(0.30)
    assert indexed.loc[
        "matched_rectangular", "pairwise_auc_difference"
    ] == pytest.approx(0.55)
    assert config["ap_simultaneous_family_size"] == 2
    assert config["simultaneous_interval"] == "two_sided_familywise_95"
    assert config["simultaneous_lower_tail_probability"] == pytest.approx(0.025)
    assert config["simultaneous_upper_tail_probability"] == pytest.approx(0.975)
    assert config["simultaneous_upper_shift"] == pytest.approx(0.0)
    assert not config["affects_primary_success_gate"]


def test_crossfit_rule_ablation_requires_every_frozen_rule() -> None:
    metrics = pd.DataFrame(
        [
            {
                "drug_id": 1,
                "fold": 0,
                "candidate": candidate,
                "average_precision": value,
                "pairwise_auc": value,
                "recall_at_10": value,
                "recall_at_50": value,
            }
            for candidate, value in (("a", 0.8), ("b", 0.4))
        ]
    )
    incomplete = {
        0: [
            {
                "center": "pmid_equal",
                "tau": "0/1",
                "possibly_optimal": ["a"],
                "empirical_center": {"optimal": ["b"]},
            }
        ]
    }

    with pytest.raises(ValueError, match="lacks selection rule"):
        bootstrap_crossfit_rule_ablation_performance(
            metrics,
            selector_cells_by_fold=incomplete,
            candidate_ids=("a", "b"),
            iterations=10,
        )

    unknown_tau = {
        0: [
            {
                "center": "pmid_equal",
                "tau": "1/3",
                "possibly_optimal": ["a"],
                "matched_rectangular": {"optimal": ["b"]},
                "empirical_center": {"optimal": ["b"]},
            }
        ]
    }
    with pytest.raises(ValueError, match="unknown tau"):
        bootstrap_crossfit_rule_ablation_performance(
            metrics,
            selector_cells_by_fold=unknown_tau,
            candidate_ids=("a", "b"),
            iterations=10,
        )


def rule_ablation_claim_fixture() -> tuple[pd.DataFrame, dict[str, object]]:
    tau_values = ("0/1", "1/20", "1/10")
    rows = []
    for center in ("pmid_equal", "positive_pair_equal"):
        for tau in tau_values:
            rows.extend(
                [
                    {
                        "center": center,
                        "tau": tau,
                        "reference_rule": "compatible",
                        "ablation_rule": "matched_rectangular",
                        "average_precision_difference": 0.10,
                        "average_precision_simultaneous_lower_95": 0.01,
                        "average_precision_simultaneous_upper_95": 0.20,
                    },
                    {
                        "center": center,
                        "tau": tau,
                        "reference_rule": "compatible",
                        "ablation_rule": "empirical_center",
                        "average_precision_difference": 0.05,
                        "average_precision_simultaneous_lower_95": -0.02,
                        "average_precision_simultaneous_upper_95": 0.12,
                    },
                ]
            )
    primary = {
        "practical_selection_passed": True,
        "center_strong_windows": {
            "pmid_equal": [list(tau_values)],
            "positive_pair_equal": [list(tau_values)],
        },
    }
    return pd.DataFrame(rows), primary


def test_rule_ablation_claim_routing_uses_entire_ap_strong_region() -> None:
    inference, primary = rule_ablation_claim_fixture()

    result = evaluate_rule_ablation_claim_routing(
        inference, primary_success=primary
    )

    assert result["matched_rectangular_claim_routing"] == (
        "SIMULTANEOUS_95_DOMINATES_ACROSS_AP_STRONG_REGION"
    )
    assert result["by_ablation_rule"]["empirical_center"]["claim_routing"] == (
        "POINT_NONNEGATIVE_SIMULTANEOUS_95_INCONCLUSIVE"
    )
    assert len(result["by_ablation_rule"]["matched_rectangular"]["cells"]) == 6
    assert result["interval_type"] == (
        "paired_drug_bootstrap_two_sided_familywise_95"
    )
    assert not result["affects_primary_success_gate"]


def test_rule_ablation_claim_routing_prioritizes_familywise_contradiction() -> None:
    inference, primary = rule_ablation_claim_fixture()
    row = (
        (inference["center"] == "positive_pair_equal")
        &
        (inference["tau"] == "1/20")
        & (inference["ablation_rule"] == "matched_rectangular")
    )
    inference.loc[
        row,
        [
            "average_precision_difference",
            "average_precision_simultaneous_lower_95",
            "average_precision_simultaneous_upper_95",
        ],
    ] = (-0.10, -0.20, -0.01)

    result = evaluate_rule_ablation_claim_routing(
        inference, primary_success=primary
    )

    assert result["matched_rectangular_claim_routing"] == (
        "SIMULTANEOUS_95_WORSE_WITHIN_AP_STRONG_REGION"
    )


def test_rule_ablation_claim_routing_keeps_mixed_region_inconclusive() -> None:
    inference, primary = rule_ablation_claim_fixture()
    row = (
        (inference["center"] == "positive_pair_equal")
        & (inference["tau"] == "1/20")
        & (inference["ablation_rule"] == "empirical_center")
    )
    inference.loc[
        row,
        [
            "average_precision_difference",
            "average_precision_simultaneous_lower_95",
            "average_precision_simultaneous_upper_95",
        ],
    ] = (-0.01, -0.10, 0.08)

    result = evaluate_rule_ablation_claim_routing(
        inference, primary_success=primary
    )

    assert result["by_ablation_rule"]["empirical_center"]["claim_routing"] == (
        "MIXED_OR_INCONCLUSIVE_WITHIN_AP_STRONG_REGION"
    )


def test_rule_ablation_claim_routing_rejects_inverted_simultaneous_band() -> None:
    inference, primary = rule_ablation_claim_fixture()
    inference.loc[0, "average_precision_simultaneous_lower_95"] = 0.30

    with pytest.raises(ValueError, match="lower > upper"):
        evaluate_rule_ablation_claim_routing(inference, primary_success=primary)


def test_rule_ablation_claim_routing_blocks_claim_when_primary_gate_fails() -> None:
    inference, primary = rule_ablation_claim_fixture()
    primary["practical_selection_passed"] = False

    result = evaluate_rule_ablation_claim_routing(
        inference, primary_success=primary
    )

    assert {
        value["claim_routing"]
        for value in result["by_ablation_rule"].values()
    } == {"PRIMARY_GATE_FAILED_NO_RULE_ABLATION_CLAIM"}


def gate_fixture(*, recall_difference: float) -> tuple[list[dict], pd.DataFrame]:
    cells = []
    rows = []
    for center in ("pmid_equal", "positive_pair_equal"):
        for tau in TAU_GRID:
            tau_text = f"{tau.numerator}/{tau.denominator}"
            cells.append(
                {
                    "center": center,
                    "tau": tau_text,
                    "possibly_optimal": ["selected"],
                }
            )
            for baseline in (
                "rna_prior_only",
                "prior_preserving_pairwise_residual",
            ):
                rows.append(
                    {
                        "center": center,
                        "tau": tau_text,
                        "baseline": baseline,
                        "average_precision_difference": 0.02,
                        "average_precision_simultaneous_lower_95": 0.01,
                        "pairwise_auc_difference": 0.01,
                        "pairwise_auc_lower_95": -0.01,
                        "pairwise_auc_upper_95": 0.03,
                        "recall_at_10_difference": recall_difference,
                        "recall_at_50_difference": recall_difference,
                    }
                )
    return cells, pd.DataFrame(rows)


def test_success_gate_requires_dual_center_consecutive_paths() -> None:
    cells, inference = gate_fixture(recall_difference=0.0)
    cells = [
        {"source_fold": index % 5, **cell}
        for index, cell in enumerate(cells)
    ]

    result = evaluate_success_gate(
        inference,
        selector_cells=cells,
        coverage={"publication_coverage_passed": True},
        candidate_count=15,
        main_selector_narrowed=True,
    )

    assert result["practical_selection_passed"]
    assert result["decision"] == "PASS"
    assert result["center_point_windows"]["pmid_equal"][0] == [
        "0/1",
        "1/20",
        "1/10",
    ]
    assert result["center_strong_windows"]["positive_pair_equal"][0] == [
        "0/1",
        "1/20",
        "1/10",
    ]
    assert {cell["source_fold"] for cell in result["selector_narrowed_cells"]} == {
        0,
        1,
        2,
        3,
        4,
    }
    assert result["objective_alignment_diagnostic"]["classification_counts"] == {
        "ORDINARY_95_SUPPORTED_AGAINST_BOTH_BASELINES": 0,
        "POINT_SUPPORTED_ORDINARY_95_INCONCLUSIVE": 24,
        "ORDINARY_95_CONTRADICTED_AGAINST_AT_LEAST_ONE_BASELINE": 0,
        "MIXED_OR_INCONCLUSIVE": 0,
    }
    assert not result["objective_alignment_diagnostic"][
        "affects_primary_success_gate"
    ]
    assert not result["objective_alignment_diagnostic"]["multiplicity_adjusted"]
    assert result["objective_alignment_diagnostic"]["ap_strong_region"][
        "claim_routing"
    ] == "POINT_CONSISTENT_ORDINARY_95_INCONCLUSIVE_ACROSS_AP_STRONG_REGION"


def test_success_gate_rejects_sustained_recall_degradation() -> None:
    cells, inference = gate_fixture(recall_difference=-0.02)

    result = evaluate_success_gate(
        inference,
        selector_cells=cells,
        coverage={"publication_coverage_passed": True},
        candidate_count=15,
        main_selector_narrowed=True,
    )

    assert not result["practical_selection_passed"]
    assert not any(result["center_strong_runs"].values())
    assert result["center_recall_blocked_windows"]["pmid_equal"]
    assert result["objective_alignment_diagnostic"]["ap_strong_region"][
        "claim_routing"
    ] == "PRIMARY_GATE_FAILED_NO_MECHANISM_CLAIM"


def test_success_gate_allows_isolated_recall_degradation() -> None:
    cells, inference = gate_fixture(recall_difference=0.0)
    isolated = (
        (inference["center"] == "pmid_equal")
        & (inference["tau"] == "0/1")
        & (inference["baseline"] == "rna_prior_only")
    )
    inference.loc[isolated, "recall_at_10_difference"] = -0.02

    result = evaluate_success_gate(
        inference,
        selector_cells=cells,
        coverage={"publication_coverage_passed": True},
        candidate_count=15,
        main_selector_narrowed=True,
    )

    assert result["practical_selection_passed"]
    assert result["center_strong_runs"]["pmid_equal"]
    assert result["center_recall_blocked_windows"]["pmid_equal"] == []


def test_success_gate_requires_main_selector_to_narrow() -> None:
    cells, inference = gate_fixture(recall_difference=0.0)

    result = evaluate_success_gate(
        inference,
        selector_cells=cells,
        coverage={"publication_coverage_passed": True},
        candidate_count=15,
        main_selector_narrowed=False,
    )

    assert result["selector_narrowed_at_least_one_cell"]
    assert not result["main_selector_narrowed_at_least_one_cell"]
    assert not result["practical_selection_passed"]
    assert result["decision"] == "FAIL_RETAIN_AS_BOUNDARY_RESULT"


def test_success_gate_rejects_duplicate_baseline_rows() -> None:
    cells, inference = gate_fixture(recall_difference=0.0)
    inference = pd.concat([inference, inference.iloc[[0]]], ignore_index=True)

    with pytest.raises(ValueError, match="inference is incomplete"):
        evaluate_success_gate(
            inference,
            selector_cells=cells,
            coverage={"publication_coverage_passed": True},
            candidate_count=15,
            main_selector_narrowed=True,
        )


def test_auc_mechanism_diagnostic_classifies_all_cells_without_changing_ap_gate() -> None:
    cells, inference = gate_fixture(recall_difference=0.0)
    original = evaluate_success_gate(
        inference,
        selector_cells=cells,
        coverage={"publication_coverage_passed": True},
        candidate_count=15,
        main_selector_narrowed=True,
    )

    supported = (inference["center"] == "pmid_equal") & (inference["tau"] == "0/1")
    inference.loc[supported, "pairwise_auc_lower_95"] = 0.001
    contradicted = (
        (inference["center"] == "pmid_equal")
        & (inference["tau"] == "1/20")
        & (inference["baseline"] == "rna_prior_only")
    )
    inference.loc[contradicted, [
        "pairwise_auc_difference",
        "pairwise_auc_lower_95",
        "pairwise_auc_upper_95",
    ]] = (-0.02, -0.04, -0.01)
    mixed = (
        (inference["center"] == "pmid_equal")
        & (inference["tau"] == "1/10")
        & (inference["baseline"] == "rna_prior_only")
    )
    inference.loc[mixed, "pairwise_auc_difference"] = -0.01

    result = evaluate_success_gate(
        inference,
        selector_cells=cells,
        coverage={"publication_coverage_passed": True},
        candidate_count=15,
        main_selector_narrowed=True,
    )

    assert result["decision"] == original["decision"] == "PASS"
    assert result["objective_alignment_diagnostic"]["classification_counts"] == {
        "ORDINARY_95_SUPPORTED_AGAINST_BOTH_BASELINES": 1,
        "POINT_SUPPORTED_ORDINARY_95_INCONCLUSIVE": 21,
        "ORDINARY_95_CONTRADICTED_AGAINST_AT_LEAST_ONE_BASELINE": 1,
        "MIXED_OR_INCONCLUSIVE": 1,
    }
    assert result["objective_alignment_diagnostic"]["ap_strong_region"][
        "claim_routing"
    ] == "ORDINARY_95_CONTRADICTION_WITHIN_AP_STRONG_REGION"


def test_auc_claim_routing_requires_support_across_entire_ap_strong_region() -> None:
    cells, inference = gate_fixture(recall_difference=0.0)
    inference["pairwise_auc_lower_95"] = 0.001

    result = evaluate_success_gate(
        inference,
        selector_cells=cells,
        coverage={"publication_coverage_passed": True},
        candidate_count=15,
        main_selector_narrowed=True,
    )

    region = result["objective_alignment_diagnostic"]["ap_strong_region"]
    assert len(region["cells"]) == 24
    assert region["claim_routing"] == (
        "ORDINARY_95_CONSISTENT_ACROSS_AP_STRONG_REGION"
    )


def test_auc_mechanism_diagnostic_rejects_inverted_interval() -> None:
    cells, inference = gate_fixture(recall_difference=0.0)
    inference.loc[0, "pairwise_auc_lower_95"] = 0.2
    inference.loc[0, "pairwise_auc_upper_95"] = 0.1

    with pytest.raises(ValueError, match="lower > upper"):
        evaluate_success_gate(
            inference,
            selector_cells=cells,
            coverage={"publication_coverage_passed": True},
            candidate_count=15,
            main_selector_narrowed=True,
        )
