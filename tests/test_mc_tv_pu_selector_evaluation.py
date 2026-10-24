from __future__ import annotations

import json
import sys
from fractions import Fraction

import numpy as np
import pandas as pd
import pytest

from mc_tv_pu_oracle import prepare_mc_tv_pu_oracle_exact
from mc_tv_pu_selector_evaluation import (
    CHECKPOINT_COLUMNS,
    EXACT_GAP,
    TAU_GRID,
    aggregate_selector_intervals,
    checkpoint_path,
    evaluate_all_drugs,
    evaluate_drug_center,
    evaluate_drug_checkpoints,
    loss_difference_numerators,
    main,
    matched_rectangular_value_exact,
    midrank_loss_numerators,
    migrate_checkpoint_directory,
    source_center_profile,
    validate_checkpoint,
    validate_checkpoint_migration_source,
    write_checkpoint,
    write_or_validate_run_configuration,
    write_or_validate_text,
)


def test_midrank_loss_is_exact_and_preserves_ties() -> None:
    scores = np.array([3.0, 1.0, 1.0, 2.0])

    numerators = midrank_loss_numerators(scores)

    assert numerators.tolist() == [1, 6, 6, 3]
    assert int(numerators.sum()) == len(scores) ** 2


def test_rank_loss_is_invariant_to_strict_monotone_transform() -> None:
    scores = np.array([-2.0, 0.5, 0.5, 8.0])

    original = midrank_loss_numerators(scores)
    transformed = midrank_loss_numerators(3.0 * scores + 11.0)

    assert np.array_equal(original, transformed)


def test_loss_difference_is_exactly_centered() -> None:
    left = np.array([0.9, 0.8, 0.1, 0.0])
    right = np.array([0.1, 0.8, 0.9, 0.0])

    difference = loss_difference_numerators(left, right)

    assert int(difference.sum()) == 0


def test_source_centers_match_frozen_definitions() -> None:
    sources = {"p1": {0, 1}, "p2": {1, 2}}

    pmid_equal = source_center_profile(
        sources, n_rna_candidates=4, mode="pmid_equal"
    )
    pair_equal = source_center_profile(
        sources, n_rna_candidates=4, mode="positive_pair_equal"
    )

    assert pmid_equal == (
        Fraction(1, 4),
        Fraction(1, 2),
        Fraction(1, 4),
        Fraction(0),
    )
    assert pair_equal == (
        Fraction(1, 3),
        Fraction(1, 3),
        Fraction(1, 3),
        Fraction(0),
    )
    one_source = {"only": {0, 2}}
    assert source_center_profile(
        one_source, n_rna_candidates=4, mode="pmid_equal"
    ) == source_center_profile(
        one_source, n_rna_candidates=4, mode="positive_pair_equal"
    )


def test_drug_center_uses_certified_exact_oracle_on_small_problem() -> None:
    scores = np.array([[0.9, 0.5, 0.1], [0.1, 0.5, 0.9]])
    center = (Fraction(1, 2), Fraction(1, 2), Fraction(0))

    result = evaluate_drug_center(
        drug_id=7,
        center="pmid_equal",
        scores=scores,
        candidate_ids=("forward", "reverse"),
        p0=center,
        taus=(Fraction(0),),
    )

    assert len(result) == 4
    assert result["certified"].all()
    assert set(result["status"]) == {"gap_met"}
    diagonal = result[result["candidate"] == result["comparator"]]
    assert set(diagonal["lower_bound"]) == {"0/1"}
    assert set(diagonal["upper_bound"]) == {"0/1"}


def test_drug_center_records_prior_clipped_rectangular_ablation() -> None:
    scores = np.array(
        [
            [4.0, 3.0, 2.0, 1.0],
            [3.0, 2.0, 1.0, 4.0],
        ]
    )

    result = evaluate_drug_center(
        drug_id=7,
        center="pmid_equal",
        scores=scores,
        candidate_ids=("a", "b"),
        p0=(Fraction(1), Fraction(0), Fraction(0), Fraction(0)),
        taus=(Fraction(3, 5),),
    )

    a_vs_b = result[
        (result["candidate"] == "a") & (result["comparator"] == "b")
    ].iloc[0]
    b_vs_a = result[
        (result["candidate"] == "b") & (result["comparator"] == "a")
    ].iloc[0]
    assert a_vs_b["prior_supremum"] == "5/8"
    assert a_vs_b["rectangular_status"] == "finite"
    assert a_vs_b["rectangular_value"] == "14/15"
    assert Fraction(a_vs_b["upper_bound"]) < Fraction(2, 3)
    assert b_vs_a["rectangular_value"] == "2/3"
    assert b_vs_a["lower_bound"] == b_vs_a["upper_bound"] == "2/3"


def test_identical_non_named_rankers_have_zero_rectangular_value_at_unit_prior() -> None:
    scores = np.array([[3.0, 2.0, 1.0], [3.0, 2.0, 1.0]])

    result = evaluate_drug_center(
        drug_id=7,
        center="pmid_equal",
        scores=scores,
        candidate_ids=("a", "b"),
        p0=(Fraction(1, 3),) * 3,
        taus=(Fraction(0),),
    )

    off_diagonal = result[result["candidate"] != result["comparator"]]
    assert set(off_diagonal["prior_supremum"]) == {"1/1"}
    assert set(off_diagonal["rectangular_status"]) == {"finite"}
    assert set(off_diagonal["rectangular_value"]) == {"0/1"}


def test_matched_rectangle_scales_negative_direction_at_prior_lower_bound() -> None:
    geometry = prepare_mc_tv_pu_oracle_exact(
        q=(Fraction(1, 2), Fraction(1, 2)),
        p0=(Fraction(1), Fraction(0)),
        tau=Fraction(0),
        prior_lower=Fraction(1, 4),
        prior_upper=Fraction(1, 2),
    )

    value, status = matched_rectangular_value_exact(
        geometry, (Fraction(-1), Fraction(1))
    )

    assert status == "finite"
    assert value == Fraction(-4, 3)


def test_tau_one_oracle_result_is_independent_of_center() -> None:
    scores = np.array([[0.9, 0.5, 0.1], [0.1, 0.5, 0.9]])
    common = {
        "drug_id": 7,
        "scores": scores,
        "candidate_ids": ("forward", "reverse"),
        "taus": (Fraction(1),),
    }

    left = evaluate_drug_center(
        center="pmid_equal",
        p0=(Fraction(1), Fraction(0), Fraction(0)),
        **common,
    )
    right = evaluate_drug_center(
        center="positive_pair_equal",
        p0=(Fraction(0), Fraction(0), Fraction(1)),
        **common,
    )

    columns = ["candidate", "comparator", "lower_bound", "upper_bound"]
    assert left[columns].to_dict("records") == right[columns].to_dict("records")


def test_tau_one_reuse_keeps_empirical_value_center_specific(tmp_path) -> None:
    scores = np.array([[3.0, 2.0, 1.0], [1.0, 2.0, 3.0]])
    candidate_ids = ("forward", "reverse")
    sources = {"p1": {0}, "p2": {0, 1}}

    evaluate_drug_checkpoints(
        drug_id=7,
        score_matrix=scores,
        candidate_ids=candidate_ids,
        source_sets=sources,
        n_rna=3,
        output_dir=tmp_path,
    )

    observed = {}
    expected = {}
    for center in ("pmid_equal", "positive_pair_equal"):
        observed[center] = pd.read_parquet(
            checkpoint_path(tmp_path, center=center, drug_id=7)
        ).loc[lambda frame: frame["tau"] == "1/1"]
        expected[center] = evaluate_drug_center(
            drug_id=7,
            center=center,
            scores=scores,
            candidate_ids=candidate_ids,
            p0=source_center_profile(
                sources, n_rna_candidates=3, mode=center
            ),
            taus=(Fraction(1),),
        )
        pd.testing.assert_frame_equal(
            observed[center].reset_index(drop=True), expected[center]
        )

    oracle_columns = ["candidate", "comparator", "lower_bound", "upper_bound"]
    assert observed["pmid_equal"][oracle_columns].to_dict("records") == observed[
        "positive_pair_equal"
    ][oracle_columns].to_dict("records")
    assert observed["pmid_equal"]["empirical_value"].tolist() != observed[
        "positive_pair_equal"
    ]["empirical_value"].tolist()


def legacy_tau_one_frame(
    *,
    scores: np.ndarray,
    candidate_ids: tuple[str, ...],
    sources: dict[str, set[int]],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    frames = {}
    for center in ("pmid_equal", "positive_pair_equal"):
        frames[center] = evaluate_drug_center(
            drug_id=7,
            center=center,
            scores=scores,
            candidate_ids=candidate_ids,
            p0=source_center_profile(
                sources, n_rna_candidates=scores.shape[1], mode=center
            ),
            taus=TAU_GRID,
        )
    pmid_tau_one = frames["pmid_equal"].loc[
        lambda frame: frame["tau"] == "1/1",
        ["candidate", "comparator", "empirical_value"],
    ].set_index(["candidate", "comparator"])["empirical_value"]
    pair_frame = frames["positive_pair_equal"].copy()
    tau_one = pair_frame["tau"] == "1/1"
    pair_frame.loc[tau_one, "empirical_value"] = [
        pmid_tau_one[(candidate, comparator)]
        for candidate, comparator in zip(
            pair_frame.loc[tau_one, "candidate"],
            pair_frame.loc[tau_one, "comparator"],
            strict=True,
        )
    ]
    return frames["pmid_equal"], pair_frame


@pytest.mark.parametrize("include_pmid_checkpoint", [True, False])
def test_resume_rejects_legacy_cross_center_empirical_values_before_writing(
    tmp_path, include_pmid_checkpoint
) -> None:
    scores = np.array([[3.0, 2.0, 1.0], [1.0, 2.0, 3.0]])
    candidate_ids = ("forward", "reverse")
    sources = {"p1": {0}, "p2": {0, 1}}
    pmid_frame, pair_frame = legacy_tau_one_frame(
        scores=scores, candidate_ids=candidate_ids, sources=sources
    )
    pmid_path = checkpoint_path(tmp_path, center="pmid_equal", drug_id=7)
    if include_pmid_checkpoint:
        write_checkpoint(pmid_frame, pmid_path)
    write_checkpoint(
        pair_frame,
        checkpoint_path(tmp_path, center="positive_pair_equal", drug_id=7),
    )

    with pytest.raises(ValueError, match="empirical values do not match"):
        evaluate_drug_checkpoints(
            drug_id=7,
            score_matrix=scores,
            candidate_ids=candidate_ids,
            source_sets=sources,
            n_rna=3,
            output_dir=tmp_path,
        )

    assert pmid_path.exists() is include_pmid_checkpoint


def test_run_configuration_refuses_unowned_checkpoints(tmp_path) -> None:
    checkpoint = checkpoint_path(tmp_path, center="pmid_equal", drug_id=7)
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_text("unowned legacy checkpoint")

    with pytest.raises(ValueError, match="explicit migration required"):
        write_or_validate_run_configuration(tmp_path, {"schema_version": 3})

    assert not (tmp_path / "run_config.json").exists()


def test_migration_source_requires_exact_legacy_configuration(tmp_path) -> None:
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    expected = {
        "schema_version": 3,
        "checkpoint_semantics": "center_specific_empirical_v1",
        "frozen": "same",
    }
    legacy = {"schema_version": 2, "frozen": "same"}
    (source_dir / "run_config.json").write_text(json.dumps(legacy))

    assert validate_checkpoint_migration_source(source_dir, expected) == legacy

    (source_dir / "run_config.json").write_text(
        json.dumps({"schema_version": 2, "frozen": "different"})
    )
    with pytest.raises(ValueError, match="incompatible frozen semantics"):
        validate_checkpoint_migration_source(source_dir, expected)


def test_migration_only_requires_checkpoint_source_dir(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "mc_tv_pu_selector_evaluation.py",
            "--score-dir",
            str(tmp_path / "scores"),
            "--pair-matrix",
            str(tmp_path / "pairs.npy"),
            "--manifest",
            str(tmp_path / "manifest.json"),
            "--source-table",
            str(tmp_path / "sources.tsv"),
            "--output-dir",
            str(tmp_path / "output"),
            "--migration-only",
        ],
    )

    with pytest.raises(SystemExit) as error:
        main()

    assert error.value.code == 2


def test_checkpoint_migration_is_exact_resumable_and_preserves_source(tmp_path) -> None:
    scores = np.array([[3.0, 2.0, 1.0], [1.0, 2.0, 3.0]])
    candidate_ids = ("forward", "reverse")
    sources = {7: {"p1": {0}, "p2": {0, 1}}}
    pmid_frame, pair_frame = legacy_tau_one_frame(
        scores=scores, candidate_ids=candidate_ids, sources=sources[7]
    )
    source_dir = tmp_path / "source"
    output_dir = tmp_path / "target"
    write_checkpoint(
        pmid_frame,
        checkpoint_path(source_dir, center="pmid_equal", drug_id=7),
    )
    source_pair_path = checkpoint_path(
        source_dir, center="positive_pair_equal", drug_id=7
    )
    write_checkpoint(pair_frame, source_pair_path)

    first = migrate_checkpoint_directory(
        iter([(7, scores)]),
        candidate_ids=candidate_ids,
        sources=sources,
        expected_drug_ids=(7,),
        n_rna=3,
        source_dir=source_dir,
        output_dir=output_dir,
    )
    second = migrate_checkpoint_directory(
        iter([(7, scores)]),
        candidate_ids=candidate_ids,
        sources=sources,
        expected_drug_ids=(7,),
        n_rna=3,
        source_dir=source_dir,
        output_dir=output_dir,
    )

    assert first == second
    assert first["source_checkpoint_count"] == 2
    assert first["target_checkpoint_count"] == 2
    assert first["corrected_empirical_cells"] == 2
    target_pair = pd.read_parquet(
        checkpoint_path(output_dir, center="positive_pair_equal", drug_id=7)
    )
    expected_pair = evaluate_drug_center(
        drug_id=7,
        center="positive_pair_equal",
        scores=scores,
        candidate_ids=candidate_ids,
        p0=source_center_profile(
            sources[7], n_rna_candidates=3, mode="positive_pair_equal"
        ),
        taus=TAU_GRID,
    )
    pd.testing.assert_frame_equal(target_pair, expected_pair)
    assert not pd.read_parquet(source_pair_path).equals(target_pair)


def test_checkpoint_migration_rejects_drift_outside_known_bug_before_writing(
    tmp_path,
) -> None:
    scores = np.array([[3.0, 2.0, 1.0], [1.0, 2.0, 3.0]])
    candidate_ids = ("forward", "reverse")
    sources = {7: {"p1": {0}, "p2": {0, 1}}}
    _, pair_frame = legacy_tau_one_frame(
        scores=scores, candidate_ids=candidate_ids, sources=sources[7]
    )
    outside = (
        (pair_frame["tau"] == "0/1")
        & (pair_frame["candidate"] == "forward")
        & (pair_frame["comparator"] == "reverse")
    )
    pair_frame.loc[outside, "empirical_value"] = "999/1"
    source_dir = tmp_path / "source"
    output_dir = tmp_path / "target"
    write_checkpoint(
        pair_frame,
        checkpoint_path(source_dir, center="positive_pair_equal", drug_id=7),
    )

    with pytest.raises(ValueError, match="drift outside the known v2 empirical bug"):
        migrate_checkpoint_directory(
            iter([(7, scores)]),
            candidate_ids=candidate_ids,
            sources=sources,
            expected_drug_ids=(7,),
            n_rna=3,
            source_dir=source_dir,
            output_dir=output_dir,
        )

    assert not (output_dir / "oracle").exists()


@pytest.mark.parametrize(
    "unexpected_relative_path",
    (
        "oracle/center=pmid_equal/orphan.parquet",
        "oracle/nested/drug=7.parquet",
    ),
)
def test_checkpoint_migration_rejects_unknown_oracle_files(
    tmp_path, unexpected_relative_path
) -> None:
    scores = np.array([[3.0, 2.0, 1.0], [1.0, 2.0, 3.0]])
    candidate_ids = ("forward", "reverse")
    sources = {7: {"p1": {0}, "p2": {0, 1}}}
    pmid_frame, pair_frame = legacy_tau_one_frame(
        scores=scores, candidate_ids=candidate_ids, sources=sources[7]
    )
    source_dir = tmp_path / "source"
    output_dir = tmp_path / "target"
    write_checkpoint(
        pmid_frame,
        checkpoint_path(source_dir, center="pmid_equal", drug_id=7),
    )
    write_checkpoint(
        pair_frame,
        checkpoint_path(source_dir, center="positive_pair_equal", drug_id=7),
    )
    unexpected = source_dir / unexpected_relative_path
    unexpected.parent.mkdir(parents=True, exist_ok=True)
    unexpected.write_text("unexpected")

    with pytest.raises(ValueError, match="unexpected oracle files"):
        migrate_checkpoint_directory(
            iter([(7, scores)]),
            candidate_ids=candidate_ids,
            sources=sources,
            expected_drug_ids=(7,),
            n_rna=3,
            source_dir=source_dir,
            output_dir=output_dir,
        )

    assert not (output_dir / "oracle").exists()


def test_checkpoint_migration_preflights_all_existing_targets_before_writing(
    tmp_path,
) -> None:
    scores = np.array([[3.0, 2.0, 1.0], [1.0, 2.0, 3.0]])
    candidate_ids = ("forward", "reverse")
    sources = {7: {"p1": {0}, "p2": {0, 1}}}
    pmid_frame, pair_frame = legacy_tau_one_frame(
        scores=scores, candidate_ids=candidate_ids, sources=sources[7]
    )
    source_dir = tmp_path / "source"
    output_dir = tmp_path / "target"
    write_checkpoint(
        pmid_frame,
        checkpoint_path(source_dir, center="pmid_equal", drug_id=7),
    )
    write_checkpoint(
        pair_frame,
        checkpoint_path(source_dir, center="positive_pair_equal", drug_id=7),
    )
    write_checkpoint(
        pair_frame,
        checkpoint_path(output_dir, center="positive_pair_equal", drug_id=7),
    )

    with pytest.raises(AssertionError):
        migrate_checkpoint_directory(
            iter([(7, scores)]),
            candidate_ids=candidate_ids,
            sources=sources,
            expected_drug_ids=(7,),
            n_rna=3,
            source_dir=source_dir,
            output_dir=output_dir,
        )

    assert not checkpoint_path(
        output_dir, center="pmid_equal", drug_id=7
    ).exists()


def test_checkpoint_migration_report_is_stable_under_score_item_reordering(
    tmp_path,
) -> None:
    scores = np.array([[3.0, 2.0, 1.0], [1.0, 2.0, 3.0]])
    candidate_ids = ("forward", "reverse")
    sources = {
        7: {"p1": {0}, "p2": {0, 1}},
        8: {"p1": {0}, "p2": {0, 1}},
    }
    source_dir = tmp_path / "source"
    output_dir = tmp_path / "target"
    for drug_id in (7, 8):
        pmid_frame, pair_frame = legacy_tau_one_frame(
            scores=scores, candidate_ids=candidate_ids, sources=sources[drug_id]
        )
        for center, frame in (
            ("pmid_equal", pmid_frame),
            ("positive_pair_equal", pair_frame),
        ):
            write_checkpoint(
                frame.assign(drug_id=drug_id),
                checkpoint_path(source_dir, center=center, drug_id=drug_id),
            )

    first = migrate_checkpoint_directory(
        iter([(7, scores), (8, scores)]),
        candidate_ids=candidate_ids,
        sources=sources,
        expected_drug_ids=(7, 8),
        n_rna=3,
        source_dir=source_dir,
        output_dir=output_dir,
    )
    second = migrate_checkpoint_directory(
        iter([(8, scores), (7, scores)]),
        candidate_ids=candidate_ids,
        sources=sources,
        expected_drug_ids=(7, 8),
        n_rna=3,
        source_dir=source_dir,
        output_dir=output_dir,
    )

    assert first == second
    assert first["source_checkpoint_count"] == 4
    assert first["target_checkpoint_count"] == 4


def test_checkpoint_migration_rejects_duplicate_score_items(tmp_path) -> None:
    scores = np.array([[3.0, 2.0, 1.0], [1.0, 2.0, 3.0]])
    output_dir = tmp_path / "target"

    with pytest.raises(ValueError, match="duplicate drugs"):
        migrate_checkpoint_directory(
            iter([(7, scores), (7, scores)]),
            candidate_ids=("forward", "reverse"),
            sources={7: {"p1": {0}, "p2": {0, 1}}},
            expected_drug_ids=(7,),
            n_rna=3,
            source_dir=tmp_path / "source",
            output_dir=output_dir,
        )

    assert not (output_dir / "oracle").exists()


@pytest.mark.parametrize("output_is_descendant", (True, False))
def test_checkpoint_migration_rejects_nested_source_and_output_trees(
    tmp_path, output_is_descendant
) -> None:
    outer = tmp_path / "outer"
    inner = outer / "inner"
    source_dir, output_dir = (
        (outer, inner) if output_is_descendant else (inner, outer)
    )

    with pytest.raises(ValueError, match="isolated source and output trees"):
        migrate_checkpoint_directory(
            iter([]),
            candidate_ids=("forward", "reverse"),
            sources={},
            expected_drug_ids=(),
            n_rna=3,
            source_dir=source_dir,
            output_dir=output_dir,
        )


def checkpoint_frame(rows: list[tuple[int, str, str, str, str]]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "center": "pmid_equal",
                "tau": "0/1",
                "drug_id": drug,
                "candidate": candidate,
                "comparator": comparator,
                "lower_bound": lower,
                "upper_bound": upper,
                "certified": True,
                "status": "gap_met",
                "evaluations": 2,
                "prior_supremum": "1/2",
                "rectangular_value": lower,
                "rectangular_status": "finite",
                "empirical_value": lower,
            }
            for drug, candidate, comparator, lower, upper in rows
        ],
        columns=CHECKPOINT_COLUMNS,
    )


def test_selector_maximizes_after_drug_average_with_shared_comparator() -> None:
    frame = checkpoint_frame(
        [
            (0, "a", "a", "0", "0"),
            (1, "a", "a", "0", "0"),
            (0, "a", "b", "3", "3"),
            (1, "a", "b", "-1", "-1"),
            (0, "b", "a", "0", "0"),
            (1, "b", "a", "0", "0"),
            (0, "b", "b", "0", "0"),
            (1, "b", "b", "0", "0"),
        ]
    )

    intervals, summary = aggregate_selector_intervals([frame], candidate_ids=("a", "b"))

    a = intervals.loc[intervals["candidate"] == "a"].iloc[0]
    b = intervals.loc[intervals["candidate"] == "b"].iloc[0]
    assert a["lower_V"] == "1/1"
    assert b["lower_V"] == "0/1"
    assert b["possibly_optimal"]
    assert b["certifiably_unique"]
    assert summary["cells"][0]["possibly_optimal"] == ["b"]


def test_selector_retains_interval_overlap_instead_of_breaking_ties() -> None:
    frame = checkpoint_frame(
        [
            (0, "a", "a", "0", "0"),
            (0, "a", "b", "0", "1/5"),
            (0, "b", "a", "1/10", "3/10"),
            (0, "b", "b", "0", "0"),
        ]
    )

    intervals, summary = aggregate_selector_intervals([frame], candidate_ids=("a", "b"))

    assert intervals["possibly_optimal"].tolist() == [True, True]
    assert intervals["certifiably_unique"].tolist() == [False, False]
    assert summary["cells"][0]["possibly_optimal"] == ["a", "b"]
    assert summary["practical_selection_gate"] == "NO_SELECTION_VALUE"


def test_selector_aggregates_compatible_rectangular_and_empirical_rules_separately() -> None:
    frame = checkpoint_frame(
        [
            (0, "a", "a", "0", "0"),
            (0, "a", "b", "3/5", "3/5"),
            (0, "b", "a", "2/3", "2/3"),
            (0, "b", "b", "0", "0"),
        ]
    )
    frame.loc[
        (frame["candidate"] == "a") & (frame["comparator"] == "b"),
        ["rectangular_value", "empirical_value"],
    ] = ["14/15", "-1/4"]
    frame.loc[
        (frame["candidate"] == "b") & (frame["comparator"] == "a"),
        ["rectangular_value", "empirical_value"],
    ] = ["2/3", "1/4"]

    intervals, summary = aggregate_selector_intervals(
        [frame], candidate_ids=("a", "b")
    )

    cell = summary["cells"][0]
    assert cell["possibly_optimal"] == ["a"]
    assert cell["matched_rectangular"]["optimal"] == ["b"]
    assert cell["empirical_center"]["optimal"] == ["a"]
    indexed = intervals.set_index("candidate")
    assert bool(indexed.loc["a", "rectangular_optimal"]) is False
    assert bool(indexed.loc["b", "rectangular_optimal"]) is True


def test_uncertified_oracle_blocks_aggregation() -> None:
    frame = checkpoint_frame(
        [
            (0, "a", "a", "0", "0"),
            (0, "a", "b", "0", "0"),
            (0, "b", "a", "0", "0"),
            (0, "b", "b", "0", "0"),
        ]
    )
    frame.loc[1, "certified"] = False

    with pytest.raises(ValueError, match="uncertified oracle result blocks"):
        aggregate_selector_intervals([frame], candidate_ids=("a", "b"))


def test_checkpoint_is_validated_and_never_overwritten(tmp_path) -> None:
    frame = checkpoint_frame(
        [
            (0, "a", "a", "0", "0"),
            (0, "a", "b", "0", "0"),
            (0, "b", "a", "0", "0"),
            (0, "b", "b", "0", "0"),
        ]
    )
    path = tmp_path / "oracle" / "center=pmid_equal" / "drug=0.parquet"

    write_checkpoint(frame, path)
    validate_checkpoint(
        path,
        center="pmid_equal",
        drug_id=0,
        candidate_ids=("a", "b"),
        taus=(Fraction(0),),
    )

    with pytest.raises(FileExistsError, match="already exists"):
        write_checkpoint(frame, path)


def test_checkpoint_rejects_duplicate_selector_cell(tmp_path) -> None:
    frame = checkpoint_frame(
        [
            (0, "a", "a", "0", "0"),
            (0, "a", "a", "0", "0"),
            (0, "b", "a", "0", "0"),
            (0, "b", "b", "0", "0"),
        ]
    )
    path = tmp_path / "duplicate.parquet"
    frame.to_parquet(path, index=False)

    with pytest.raises(ValueError, match="duplicate selector cells"):
        validate_checkpoint(
            path,
            center="pmid_equal",
            drug_id=0,
            candidate_ids=("a", "b"),
            taus=(Fraction(0),),
        )


def test_checkpoint_rejects_interval_wider_than_frozen_exact_gap(tmp_path) -> None:
    frame = checkpoint_frame(
        [
            (0, "a", "a", "0", "0"),
            (0, "a", "b", "0", "0"),
            (0, "b", "a", "0", "0"),
            (0, "b", "b", "0", "0"),
        ]
    )
    frame.loc[0, "upper_bound"] = str(EXACT_GAP + Fraction(1, 10_000))
    path = tmp_path / "wide.parquet"
    frame.to_parquet(path, index=False)

    with pytest.raises(ValueError, match="exceeds the frozen exact gap"):
        validate_checkpoint(
            path,
            center="pmid_equal",
            drug_id=0,
            candidate_ids=("a", "b"),
            taus=(Fraction(0),),
        )


def test_final_text_artifact_can_only_be_revalidated(tmp_path) -> None:
    path = tmp_path / "summary.json"

    write_or_validate_text(path, '{"status":"PASS"}\n')
    write_or_validate_text(path, '{"status":"PASS"}\n')

    with pytest.raises(ValueError, match="differs from recomputation"):
        write_or_validate_text(path, '{"status":"CHANGED"}\n')


def test_parallel_drug_evaluation_matches_serial_checkpoints(tmp_path) -> None:
    candidate_ids = ("forward", "reverse")
    score_items = [
        (0, np.array([[0.9, 0.5, 0.1], [0.1, 0.5, 0.9]])),
        (1, np.array([[0.8, 0.2, 0.1], [0.1, 0.8, 0.2]])),
    ]
    sources = {0: {"p0": {0}}, 1: {"p1": {1}}}
    serial_dir = tmp_path / "serial"
    parallel_dir = tmp_path / "parallel"

    serial_drugs = evaluate_all_drugs(
        iter(score_items),
        candidate_ids=candidate_ids,
        sources=sources,
        n_rna=3,
        output_dir=serial_dir,
        workers=1,
    )
    parallel_drugs = evaluate_all_drugs(
        iter(score_items),
        candidate_ids=candidate_ids,
        sources=sources,
        n_rna=3,
        output_dir=parallel_dir,
        workers=2,
    )

    assert serial_drugs == parallel_drugs == {0, 1}
    for center in ("pmid_equal", "positive_pair_equal"):
        for drug_id in (0, 1):
            serial = pd.read_parquet(
                checkpoint_path(serial_dir, center=center, drug_id=drug_id)
            )
            parallel = pd.read_parquet(
                checkpoint_path(parallel_dir, center=center, drug_id=drug_id)
            )
            pd.testing.assert_frame_equal(serial, parallel)
