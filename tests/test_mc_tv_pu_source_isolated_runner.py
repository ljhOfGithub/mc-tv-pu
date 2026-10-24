from __future__ import annotations

import numpy as np
import pandas as pd

from mc_tv_pu_q_tau_semantics_audit import source_sets_by_drug
from mc_tv_pu_source_isolated_runner import (
    SOURCE_METRIC_COLUMNS,
    SOURCE_ORACLE_COLUMNS,
    aggregate_source_oracle_intervals,
    evaluate_source_drug_outputs,
    source_metric_checkpoint_path,
    source_oracle_checkpoint_path,
    validate_source_metric_checkpoint,
    validate_source_oracle_checkpoint,
)
from mc_tv_pu_source_isolated_validation import build_source_episodes


def test_source_drug_outputs_are_complete_and_resumable(tmp_path) -> None:
    pair_sources = {(0, 7): {"p1"}, (2, 7): {"p2"}}
    episodes = build_source_episodes(pair_sources)[7]
    sources = source_sets_by_drug(pair_sources)[7]
    scores = np.array([[0.9, 0.5, 0.1], [0.1, 0.5, 0.9]])
    candidates = ("forward", "reverse")

    completed, created = evaluate_source_drug_outputs(
        drug_id=7,
        score_matrix=scores,
        candidate_ids=candidates,
        source_sets=sources,
        episodes=episodes,
        n_rna=3,
        output_dir=tmp_path,
    )

    assert completed == 7
    assert created == 2
    oracle_path = source_oracle_checkpoint_path(tmp_path, 7)
    metric_path = source_metric_checkpoint_path(tmp_path, 7)
    validate_source_oracle_checkpoint(
        oracle_path, episodes=episodes, candidate_ids=candidates
    )
    validate_source_metric_checkpoint(
        metric_path, episodes=episodes, candidate_ids=candidates
    )
    assert len(pd.read_parquet(oracle_path)) == 2 * 2 * 12 * 4
    metric_frame = pd.read_parquet(metric_path)
    assert len(metric_frame) == 2 * 2
    assert "pairwise_auc" in metric_frame.columns
    assert metric_frame["pairwise_auc"].between(0.0, 1.0).all()

    _, resumed_created = evaluate_source_drug_outputs(
        drug_id=7,
        score_matrix=scores,
        candidate_ids=candidates,
        source_sets=sources,
        episodes=episodes,
        n_rna=3,
        output_dir=tmp_path,
    )
    assert resumed_created == 0


def test_empty_metric_checkpoint_is_valid_for_overlap_only_sources(tmp_path) -> None:
    pair_sources = {(0, 7): {"p1", "p2"}}
    episodes = build_source_episodes(pair_sources)[7]
    path = source_metric_checkpoint_path(tmp_path, 7)
    path.parent.mkdir(parents=True)
    pd.DataFrame(columns=SOURCE_METRIC_COLUMNS).to_parquet(path, index=False)

    validate_source_metric_checkpoint(
        path, episodes=episodes, candidate_ids=("a", "b")
    )


def oracle_frame() -> pd.DataFrame:
    rows = []
    episode_values = {
        (1, "s1"): 3,
        (1, "s2"): -1,
        (2, "s3"): 3,
    }
    for (drug_id, source), a_vs_b in episode_values.items():
        for candidate, comparator, value in (
            ("a", "a", 0),
            ("a", "b", a_vs_b),
            ("b", "a", 0),
            ("b", "b", 0),
        ):
            core = {
                "center": "pmid_equal",
                "tau": "0/1",
                "drug_id": drug_id,
                "candidate": candidate,
                "comparator": comparator,
                "lower_bound": f"{value}/1",
                "upper_bound": f"{value}/1",
                "certified": True,
                "status": "gap_met",
                "evaluations": 2,
                "prior_supremum": "1/2",
            }
            rows.append(
                {
                    "heldout_source": source,
                    "anchor_source_count": 1,
                    "heldout_novel_count": 1,
                    **core,
                }
            )
    return pd.DataFrame(rows, columns=SOURCE_ORACLE_COLUMNS)


def test_source_oracle_aggregation_averages_episodes_then_drugs() -> None:
    intervals, summary = aggregate_source_oracle_intervals(
        [oracle_frame()], candidate_ids=("a", "b")
    )

    a = intervals[intervals["candidate"] == "a"].iloc[0]
    b = intervals[intervals["candidate"] == "b"].iloc[0]
    assert a["lower_V"] == "2/1"
    assert b["lower_V"] == "0/1"
    assert summary["cells"][0]["possibly_optimal"] == ["b"]
