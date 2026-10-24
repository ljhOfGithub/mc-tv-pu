from __future__ import annotations

import json
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from generalization_audit import scaffold_fold_assignment
from mc_tv_pu_score_input_gate import (
    audit_score_partition_directory,
    write_score_partition,
)
from mc_tv_pu_selector_evaluation import (
    CENTER_MODES,
    TAU_GRID,
    aggregate_selector_intervals,
    evaluate_drug_center,
    source_center_profile,
    validate_checkpoint,
    write_checkpoint,
    write_or_validate_run_configuration,
    write_or_validate_text,
)
from mc_tv_pu_source_crossfit_runner import (
    CROSSFIT_FINAL_ARTIFACTS,
    IMPLEMENTATION_SOURCE_FILES,
    _assignment_digest,
    crossfit_final_artifact_hashes,
    crossfit_implementation_hashes,
    crossfit_metric_checkpoint_path,
    crossfit_oracle_checkpoint_path,
    evaluate_crossfit_drug_outputs,
    exclusive_output_lock,
    source_isolation_scope_metadata,
    validate_crossfit_artifact_package,
    validate_crossfit_metric_checkpoint,
    validate_dataset_dir_binding,
    validate_main_selector_completion,
    validate_target_drug_oof_assignment,
    write_or_validate_crossfit_run_configuration,
)
from mc_tv_pu_source_isolated_validation import SourceCrossFitEpisode


def _write_scaffold_fold_scores(
    output_dir,
    *,
    pair_matrix: np.ndarray,
    drug_scaffolds: np.ndarray,
    rotate_folds: bool = False,
):
    fold_count = 3
    folds = scaffold_fold_assignment(
        pair_matrix, drug_scaffolds, n_folds=fold_count, seed=17
    )
    for expected_fold, drug_ids in enumerate(folds):
        output_fold = (expected_fold + 1) % fold_count if rotate_folds else expected_fold
        pairs = np.asarray(
            [
                (rna_id, int(drug_id))
                for drug_id in drug_ids
                for rna_id in range(pair_matrix.shape[0])
            ],
            dtype=np.int64,
        )
        write_score_partition(
            output_dir,
            model="reference",
            fold=output_fold,
            pairs=pairs,
            scores=np.arange(len(pairs), dtype=np.float64),
        )
    return audit_score_partition_directory(
        output_dir,
        expected_drug_ids=np.flatnonzero((pair_matrix > 0).any(axis=0)),
        expected_rna_ids=range(pair_matrix.shape[0]),
        expected_models=("reference",),
        expected_fold_count=fold_count,
    )


def _complete_main_selector_fixture(tmp_path):
    selector_dir = tmp_path / "selector"
    exit_log = tmp_path / "selector.log"
    candidate_ids = ("forward", "reverse")
    drug_id = 7
    scores = np.array([[0.9, 0.5, 0.1], [0.1, 0.5, 0.9]])
    p0 = (1, 0, 0)
    config = {
        "schema_version": 3,
        "checkpoint_semantics": "center_specific_empirical_v1",
        "candidate_ids": list(candidate_ids),
    }
    write_or_validate_run_configuration(selector_dir, config)
    paths = []
    for center in CENTER_MODES:
        frame = evaluate_drug_center(
            drug_id=drug_id,
            center=center,
            scores=scores,
            candidate_ids=candidate_ids,
            p0=p0,
        )
        path = selector_dir / "oracle" / f"center={center}" / f"drug={drug_id}.parquet"
        write_checkpoint(frame, path)
        paths.append(path)
    intervals, summary = aggregate_selector_intervals(
        (pd.read_parquet(path) for path in paths),
        candidate_ids=candidate_ids,
    )
    write_or_validate_text(
        selector_dir / "candidate_intervals.csv", intervals.to_csv(index=False)
    )
    write_or_validate_text(
        selector_dir / "selector_summary.json",
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
    )
    exit_log.write_text("selector output\nEXIT_STATUS=0\n")
    return selector_dir, exit_log, config, candidate_ids, drug_id


def test_main_selector_completion_gate_accepts_recomputed_terminal_run(tmp_path) -> None:
    selector_dir, exit_log, config, candidate_ids, drug_id = (
        _complete_main_selector_fixture(tmp_path)
    )

    evidence = validate_main_selector_completion(
        selector_dir,
        exit_log=exit_log,
        expected_config=config,
        candidate_ids=candidate_ids,
        active_drugs=(drug_id,),
    )

    assert evidence["checkpoint_count"] == 2
    assert evidence["selector_dir"] == str(selector_dir.resolve())
    assert evidence["practical_selection_gate"] in {
        "NO_SELECTION_VALUE",
        "AT_LEAST_ONE_CELL_NARROWS_LIBRARY",
    }


def test_dataset_dir_must_be_manifest_hashed_root_dataset(tmp_path) -> None:
    expected = tmp_path / "Dataset"
    expected.mkdir()

    evidence = validate_dataset_dir_binding(tmp_path, expected)

    assert evidence == {"status": "PASS", "dataset_dir": str(expected.resolve())}
    with pytest.raises(ValueError, match="root/Dataset"):
        validate_dataset_dir_binding(tmp_path, tmp_path / "other-dataset")


def test_output_lock_blocks_a_second_process(tmp_path) -> None:
    output_dir = tmp_path / "formal-output"
    code = (
        "from pathlib import Path; "
        "from mc_tv_pu_source_crossfit_runner import exclusive_output_lock; "
        "import sys; "
        "\nwith exclusive_output_lock(Path(sys.argv[1])):\n    pass\n"
    )

    with exclusive_output_lock(output_dir) as lock_path:
        result = subprocess.run(
            [sys.executable, "-c", code, str(output_dir)],
            cwd=__import__("pathlib").Path(__file__).resolve().parent.parent,
            capture_output=True,
            text=True,
            check=False,
        )
        assert lock_path.read_text().startswith("pid=")

    assert result.returncode != 0
    assert "already locked" in result.stderr


@pytest.mark.parametrize(
    ("corruption", "message"),
    (
        ("nonzero_exit", "EXIT_STATUS=0"),
        ("missing_checkpoint", "coverage mismatch"),
        ("unfinished", "unfinished files"),
        ("changed_config", "configuration mismatch"),
        ("changed_intervals", "failed recomputation"),
        ("changed_summary", "failed recomputation"),
    ),
)
def test_main_selector_completion_gate_rejects_incomplete_or_corrupt_run(
    tmp_path, corruption: str, message: str
) -> None:
    selector_dir, exit_log, config, candidate_ids, drug_id = (
        _complete_main_selector_fixture(tmp_path)
    )
    if corruption == "nonzero_exit":
        exit_log.write_text("EXIT_STATUS=1\n")
    elif corruption == "missing_checkpoint":
        (selector_dir / "oracle/center=pmid_equal/drug=7.parquet").unlink()
    elif corruption == "unfinished":
        (selector_dir / "oracle/center=pmid_equal/drug=8.tmp.parquet").write_bytes(
            b"unfinished"
        )
    elif corruption == "changed_config":
        changed = dict(config)
        changed["candidate_ids"] = ["reverse", "forward"]
        (selector_dir / "run_config.json").write_text(
            json.dumps(changed, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
        )
    elif corruption == "changed_intervals":
        (selector_dir / "candidate_intervals.csv").write_text("changed\n")
    elif corruption == "changed_summary":
        (selector_dir / "selector_summary.json").write_text("{}\n")

    with pytest.raises(ValueError, match=message):
        validate_main_selector_completion(
            selector_dir,
            exit_log=exit_log,
            expected_config=config,
            candidate_ids=candidate_ids,
            active_drugs=(drug_id,),
        )


def test_crossfit_drug_outputs_are_complete_and_resumable(tmp_path) -> None:
    episode = SourceCrossFitEpisode(
        fold=2,
        drug_id=7,
        heldout_sources=("p2",),
        anchor_sources=("p1",),
        anchor_rnas=(0,),
        heldout_novel_rnas=(2,),
    )
    scores = np.array([[0.9, 0.5, 0.1], [0.1, 0.5, 0.9]])
    candidates = ("forward", "reverse")

    completed, created = evaluate_crossfit_drug_outputs(
        drug_id=7,
        score_matrix=scores,
        candidate_ids=candidates,
        source_sets={"p1": {0}, "p2": {2}},
        episodes=(episode,),
        n_rna=3,
        output_dir=tmp_path,
    )

    assert completed == 7
    assert created == 3
    metric_path = crossfit_metric_checkpoint_path(tmp_path, fold=2, drug_id=7)
    validate_crossfit_metric_checkpoint(
        metric_path,
        episode=episode,
        candidate_ids=candidates,
        n_rna=3,
        scores=scores,
    )
    for center in ("pmid_equal", "positive_pair_equal"):
        validate_checkpoint(
            crossfit_oracle_checkpoint_path(
                tmp_path, fold=2, center=center, drug_id=7
            ),
            center=center,
            drug_id=7,
            candidate_ids=candidates,
        )
    pmid = pd.read_parquet(
        crossfit_oracle_checkpoint_path(
            tmp_path, fold=2, center="pmid_equal", drug_id=7
        )
    )
    pair = pd.read_parquet(
        crossfit_oracle_checkpoint_path(
            tmp_path, fold=2, center="positive_pair_equal", drug_id=7
        )
    )
    pd.testing.assert_frame_equal(
        pmid.drop(columns="center"), pair.drop(columns="center")
    )

    _, resumed_created = evaluate_crossfit_drug_outputs(
        drug_id=7,
        score_matrix=scores,
        candidate_ids=candidates,
        source_sets={"p1": {0}, "p2": {2}},
        episodes=(episode,),
        n_rna=3,
        output_dir=tmp_path,
    )
    assert resumed_created == 0


@pytest.mark.parametrize(
    ("column", "bad_value"),
    (
        ("candidate_count", 3),
        ("positive_count", 2),
        ("actual_k_10", 1),
        ("actual_k_50", 1),
        ("average_precision", 1.1),
        ("pairwise_auc", 1.1),
        ("recall_at_10", -0.1),
        ("recall_at_50", 1.1),
    ),
)
def test_crossfit_metric_checkpoint_rejects_finite_semantic_corruption(
    tmp_path, column: str, bad_value: float
) -> None:
    episode = SourceCrossFitEpisode(
        fold=2,
        drug_id=7,
        heldout_sources=("p2",),
        anchor_sources=("p1",),
        anchor_rnas=(0,),
        heldout_novel_rnas=(2,),
    )
    path = crossfit_metric_checkpoint_path(tmp_path, fold=2, drug_id=7)
    frame = pd.DataFrame(
        [
            {
                "fold": 2,
                "drug_id": 7,
                "candidate": "forward",
                "candidate_count": 2,
                "positive_count": 1,
                "average_precision": 0.5,
                "pairwise_auc": 0.5,
                "recall_at_10": 1.0,
                "recall_at_50": 1.0,
                "actual_k_10": 2,
                "actual_k_50": 2,
            }
        ],
        columns=(
            "fold",
            "drug_id",
            "candidate",
            "candidate_count",
            "positive_count",
            "average_precision",
            "pairwise_auc",
            "recall_at_10",
            "recall_at_50",
            "actual_k_10",
            "actual_k_50",
        ),
    )
    frame.loc[0, column] = bad_value
    path.parent.mkdir(parents=True)
    frame.to_parquet(path, index=False)

    with pytest.raises(ValueError):
        validate_crossfit_metric_checkpoint(
            path,
            episode=episode,
            candidate_ids=("forward",),
            n_rna=3,
            scores=np.array([[0.9, 0.5, 0.1]]),
        )


def test_crossfit_metric_checkpoint_rejects_in_range_metric_drift(tmp_path) -> None:
    episode = SourceCrossFitEpisode(
        fold=2,
        drug_id=7,
        heldout_sources=("p2",),
        anchor_sources=("p1",),
        anchor_rnas=(0,),
        heldout_novel_rnas=(2,),
    )
    scores = np.array([[0.9, 0.5, 0.1], [0.1, 0.5, 0.9]])
    candidates = ("forward", "reverse")
    evaluate_crossfit_drug_outputs(
        drug_id=7,
        score_matrix=scores,
        candidate_ids=candidates,
        source_sets={"p1": {0}, "p2": {2}},
        episodes=(episode,),
        n_rna=3,
        output_dir=tmp_path,
    )
    path = crossfit_metric_checkpoint_path(tmp_path, fold=2, drug_id=7)
    frame = pd.read_parquet(path)
    frame.loc[frame["candidate"] == "forward", "average_precision"] = 0.6
    frame.to_parquet(path, index=False)

    with pytest.raises(ValueError, match="score/episode recomputation"):
        validate_crossfit_metric_checkpoint(
            path,
            episode=episode,
            candidate_ids=candidates,
            n_rna=3,
            scores=scores,
        )


def test_crossfit_metric_checkpoint_rejects_changed_heldout_identity(tmp_path) -> None:
    original = SourceCrossFitEpisode(
        fold=2,
        drug_id=7,
        heldout_sources=("p2",),
        anchor_sources=("p1",),
        anchor_rnas=(0,),
        heldout_novel_rnas=(2,),
    )
    changed = SourceCrossFitEpisode(
        fold=2,
        drug_id=7,
        heldout_sources=("p3",),
        anchor_sources=("p1",),
        anchor_rnas=(0,),
        heldout_novel_rnas=(1,),
    )
    scores = np.array([[0.9, 0.8, 0.1], [0.1, 0.2, 0.9]])
    candidates = ("forward", "reverse")
    evaluate_crossfit_drug_outputs(
        drug_id=7,
        score_matrix=scores,
        candidate_ids=candidates,
        source_sets={"p1": {0}, "p2": {2}},
        episodes=(original,),
        n_rna=3,
        output_dir=tmp_path,
    )
    path = crossfit_metric_checkpoint_path(tmp_path, fold=2, drug_id=7)

    with pytest.raises(ValueError, match="score/episode recomputation"):
        validate_crossfit_metric_checkpoint(
            path,
            episode=changed,
            candidate_ids=candidates,
            n_rna=3,
            scores=scores,
        )


def test_source_isolation_scope_metadata_prevents_end_to_end_claim() -> None:
    target_oof_evidence = {
        "status": "PASS",
        "basis": "recomputed_bemis_murcko_scaffold_test_fold",
    }

    assert source_isolation_scope_metadata(target_oof_evidence) == {
        "target_drug_oof": True,
        "target_drug_oof_audit": target_oof_evidence,
        "selector_anchor_source_disjoint": True,
        "source_isolation_scope": "selector_anchors_only",
        "base_model_source_disjoint": False,
        "episode_membership_uses_heldout_source_presence": True,
    }


def test_source_isolation_scope_requires_target_drug_oof_evidence() -> None:
    with pytest.raises(ValueError, match="target-drug OOF evidence"):
        source_isolation_scope_metadata({"status": "INPUT_BLOCKED"})


def test_target_drug_oof_assignment_matches_recomputed_scaffold_folds(
    tmp_path,
) -> None:
    pair_matrix = np.asarray([[1, 0, 1], [0, 1, 0]], dtype=np.int8)
    drug_scaffolds = np.asarray(["s0", "s1", "s2"])
    score_audit = _write_scaffold_fold_scores(
        tmp_path,
        pair_matrix=pair_matrix,
        drug_scaffolds=drug_scaffolds,
    )

    evidence = validate_target_drug_oof_assignment(
        tmp_path,
        reference_model="reference",
        pair_matrix=pair_matrix,
        drug_scaffolds=drug_scaffolds,
        fold_count=3,
        split_seed=17,
        score_input_audit=score_audit,
    )

    assert evidence["status"] == "PASS"
    assert evidence["drug_count"] == 3
    assert evidence["fold_count"] == 3
    assert evidence["split_seed"] == 17


def test_target_drug_oof_assignment_rejects_self_consistent_wrong_folds(
    tmp_path,
) -> None:
    pair_matrix = np.asarray([[1, 0, 1], [0, 1, 0]], dtype=np.int8)
    drug_scaffolds = np.asarray(["s0", "s1", "s2"])
    score_audit = _write_scaffold_fold_scores(
        tmp_path,
        pair_matrix=pair_matrix,
        drug_scaffolds=drug_scaffolds,
        rotate_folds=True,
    )
    assert score_audit.status == "PASS"

    with pytest.raises(ValueError, match="deterministic scaffold test-fold map"):
        validate_target_drug_oof_assignment(
            tmp_path,
            reference_model="reference",
            pair_matrix=pair_matrix,
            drug_scaffolds=drug_scaffolds,
            fold_count=3,
            split_seed=17,
            score_input_audit=score_audit,
        )


def test_source_assignment_digest_is_order_invariant() -> None:
    assert _assignment_digest({"p1": 0, "p2": 1}) == _assignment_digest(
        {"p2": 1, "p1": 0}
    )


def test_crossfit_resume_rejects_oracle_from_another_anchor_center(tmp_path) -> None:
    scores = np.array([[0.9, 0.5, 0.1], [0.1, 0.5, 0.9]])
    candidates = ("forward", "reverse")
    first = SourceCrossFitEpisode(
        fold=2,
        drug_id=7,
        heldout_sources=("p2",),
        anchor_sources=("p1",),
        anchor_rnas=(0,),
        heldout_novel_rnas=(2,),
    )
    evaluate_crossfit_drug_outputs(
        drug_id=7,
        score_matrix=scores,
        candidate_ids=candidates,
        source_sets={"p1": {0}, "p2": {2}, "p3": {1}},
        episodes=(first,),
        n_rna=3,
        output_dir=tmp_path,
    )
    changed_anchor = SourceCrossFitEpisode(
        fold=2,
        drug_id=7,
        heldout_sources=("p2",),
        anchor_sources=("p3",),
        anchor_rnas=(1,),
        heldout_novel_rnas=(2,),
    )

    with pytest.raises(ValueError, match="source center"):
        evaluate_crossfit_drug_outputs(
            drug_id=7,
            score_matrix=scores,
            candidate_ids=candidates,
            source_sets={"p1": {0}, "p2": {2}, "p3": {1}},
            episodes=(changed_anchor,),
            n_rna=3,
            output_dir=tmp_path,
        )


def test_crossfit_tau_one_reuse_matches_direct_fold_center_recomputation(
    tmp_path, monkeypatch
) -> None:
    scores = np.array(
        [[0.9, 0.7, 0.3, 0.1, 0.0], [0.1, 0.3, 0.7, 0.9, 0.0]]
    )
    candidates = ("forward", "reverse")
    source_sets = {
        "p1": {0},
        "p2": {0, 1},
        "p3": {2},
        "p4": {3},
    }
    episodes = (
        SourceCrossFitEpisode(
            fold=0,
            drug_id=7,
            heldout_sources=("p3",),
            anchor_sources=("p1", "p2"),
            anchor_rnas=(0, 1),
            heldout_novel_rnas=(2,),
        ),
        SourceCrossFitEpisode(
            fold=1,
            drug_id=7,
            heldout_sources=("p4",),
            anchor_sources=("p2", "p3"),
            anchor_rnas=(0, 1, 2),
            heldout_novel_rnas=(3,),
        ),
    )
    tau_counts: list[int] = []

    def counted_evaluation(**kwargs):
        tau_counts.append(len(kwargs["taus"]))
        return evaluate_drug_center(**kwargs)

    monkeypatch.setattr(
        "mc_tv_pu_source_crossfit_runner.evaluate_drug_center",
        counted_evaluation,
    )
    evaluate_crossfit_drug_outputs(
        drug_id=7,
        score_matrix=scores,
        candidate_ids=candidates,
        source_sets=source_sets,
        episodes=episodes,
        n_rna=5,
        output_dir=tmp_path,
    )

    expected_tau_counts = [
        len(TAU_GRID),
        *(len(TAU_GRID) - 1 for _ in range(3)),
    ]
    assert tau_counts == expected_tau_counts
    reversed_dir = tmp_path / "reversed"
    evaluate_crossfit_drug_outputs(
        drug_id=7,
        score_matrix=scores,
        candidate_ids=candidates,
        source_sets=source_sets,
        episodes=tuple(reversed(episodes)),
        n_rna=5,
        output_dir=reversed_dir,
    )
    assert tau_counts == [*expected_tau_counts, *expected_tau_counts]
    for episode in episodes:
        anchors = {source: source_sets[source] for source in episode.anchor_sources}
        for center in CENTER_MODES:
            p0 = source_center_profile(
                anchors, n_rna_candidates=5, mode=center
            )
            expected = evaluate_drug_center(
                drug_id=7,
                center=center,
                scores=scores,
                candidate_ids=candidates,
                p0=p0,
            ).sort_values(["tau", "candidate", "comparator"], kind="mergesort")
            observed = pd.read_parquet(
                crossfit_oracle_checkpoint_path(
                    tmp_path,
                    fold=episode.fold,
                    center=center,
                    drug_id=7,
                )
            ).sort_values(["tau", "candidate", "comparator"], kind="mergesort")
            reversed_observed = pd.read_parquet(
                crossfit_oracle_checkpoint_path(
                    reversed_dir,
                    fold=episode.fold,
                    center=center,
                    drug_id=7,
                )
            ).sort_values(["tau", "candidate", "comparator"], kind="mergesort")
            pd.testing.assert_frame_equal(
                observed.reset_index(drop=True), expected.reset_index(drop=True)
            )
            pd.testing.assert_frame_equal(
                reversed_observed.reset_index(drop=True),
                expected.reset_index(drop=True),
            )


@pytest.mark.parametrize(
    "relative_path",
    (
        "source_fold=0/oracle/center=pmid_equal/drug=7.parquet",
        "source_fold=0/metrics/drug=7.parquet",
        "source_fold=0/metrics/drug=7.partial",
    ),
)
def test_crossfit_configuration_rejects_nested_orphan_checkpoints(
    tmp_path, relative_path: str
) -> None:
    orphan = tmp_path / relative_path
    orphan.parent.mkdir(parents=True)
    orphan.write_bytes(b"orphan")

    with pytest.raises(ValueError, match="without run_config"):
        write_or_validate_crossfit_run_configuration(
            tmp_path, {"schema_version": 3}
        )


def test_crossfit_configuration_is_bound_to_implementation_sources(tmp_path) -> None:
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    for index, name in enumerate(IMPLEMENTATION_SOURCE_FILES):
        (source_dir / name).write_text(f"source-{index}\n")

    hashes = crossfit_implementation_hashes(source_dir)
    assert tuple(hashes) == IMPLEMENTATION_SOURCE_FILES
    assert all(len(digest) == 64 for digest in hashes.values())

    output_dir = tmp_path / "output"
    config = {
        "schema_version": 3,
        "implementation_source_sha256": hashes,
    }
    write_or_validate_crossfit_run_configuration(output_dir, config)
    write_or_validate_crossfit_run_configuration(output_dir, config)

    first_source = source_dir / IMPLEMENTATION_SOURCE_FILES[0]
    first_source.write_text("changed\n")
    changed = {
        **config,
        "implementation_source_sha256": crossfit_implementation_hashes(source_dir),
    }
    with pytest.raises(ValueError, match="different frozen run"):
        write_or_validate_crossfit_run_configuration(output_dir, changed)


def test_crossfit_terminal_summary_hashes_every_final_artifact(tmp_path) -> None:
    (tmp_path / "run_config.json").write_text('{"schema_version":3}\n')
    (tmp_path / "source_fold_assignments.json").write_text('{"p1":0}\n')
    table_texts = {
        "candidate_intervals_text": "candidate,lower\na,0\n",
        "episode_metrics_text": "drug_id,average_precision\n7,0.5\n",
        "inference_text": "metric,estimate\naverage_precision,0.1\n",
        "rule_ablation_inference_text": "rule,estimate\nmatched_rectangular,0\n",
    }

    observed = crossfit_final_artifact_hashes(tmp_path, **table_texts)
    assert set(observed) == {
        "run_config.json",
        "source_fold_assignments.json",
        "crossfit_candidate_intervals.csv",
        "crossfit_episode_metrics.csv",
        "crossfit_inference.csv",
        "crossfit_rule_ablation_inference.csv",
    }
    assert all(len(digest) == 64 for digest in observed.values())

    changed_table_texts = dict(table_texts)
    changed_table_texts["inference_text"] += "pairwise_auc,0.2\n"
    changed_table = crossfit_final_artifact_hashes(tmp_path, **changed_table_texts)
    assert changed_table["crossfit_inference.csv"] != observed[
        "crossfit_inference.csv"
    ]
    for name in set(observed) - {"crossfit_inference.csv"}:
        assert changed_table[name] == observed[name]

    (tmp_path / "run_config.json").write_text('{"schema_version":4}\n')
    changed_config = crossfit_final_artifact_hashes(tmp_path, **table_texts)
    assert changed_config["run_config.json"] != observed["run_config.json"]


def test_crossfit_terminal_artifact_package_rejects_drift_and_residue(tmp_path) -> None:
    (tmp_path / "run_config.json").write_text('{"schema_version":3}\n')
    (tmp_path / "source_fold_assignments.json").write_text('{"p1":0}\n')
    table_texts = {
        "candidate_intervals_text": "candidate,lower\na,0\n",
        "episode_metrics_text": "drug_id,average_precision\n7,0.5\n",
        "inference_text": "metric,estimate\naverage_precision,0.1\n",
        "rule_ablation_inference_text": "rule,estimate\nmatched_rectangular,0\n",
    }
    names_by_argument = {
        "candidate_intervals_text": "crossfit_candidate_intervals.csv",
        "episode_metrics_text": "crossfit_episode_metrics.csv",
        "inference_text": "crossfit_inference.csv",
        "rule_ablation_inference_text": "crossfit_rule_ablation_inference.csv",
    }
    for argument, name in names_by_argument.items():
        (tmp_path / name).write_text(table_texts[argument])
    hashes = crossfit_final_artifact_hashes(tmp_path, **table_texts)
    (tmp_path / "crossfit_validation_summary.json").write_text(
        json.dumps({"artifact_sha256": hashes}) + "\n"
    )

    evidence = validate_crossfit_artifact_package(tmp_path)
    assert evidence["status"] == "PASS"
    assert evidence["artifact_count"] == len(CROSSFIT_FINAL_ARTIFACTS)

    inference_path = tmp_path / "crossfit_inference.csv"
    original_inference = inference_path.read_text()
    inference_path.write_text(original_inference + "pairwise_auc,0.2\n")
    with pytest.raises(ValueError, match="artifact hash mismatch"):
        validate_crossfit_artifact_package(tmp_path)
    inference_path.write_text(original_inference)

    summary_path = tmp_path / "crossfit_validation_summary.json"
    summary = json.loads(summary_path.read_text())
    summary["artifact_sha256"].pop("crossfit_episode_metrics.csv")
    summary_path.write_text(json.dumps(summary) + "\n")
    with pytest.raises(ValueError, match="hash coverage mismatch"):
        validate_crossfit_artifact_package(tmp_path)
    summary["artifact_sha256"] = hashes
    summary_path.write_text(json.dumps(summary) + "\n")

    residue = tmp_path / "source_fold=0/oracle/drug=7.partial"
    residue.parent.mkdir(parents=True)
    residue.write_text("unfinished")
    with pytest.raises(ValueError, match="unfinished files"):
        validate_crossfit_artifact_package(tmp_path)
