from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mc_tv_pu_score_input_gate import (
    audit_score_partition_directory,
    audit_score_table,
    candidate_ids_for_source_group,
    export_frozen_score_partitions,
    file_sha256,
    validate_candidate_manifest,
    validated_candidate_ids_for_export,
    validated_seed_candidate_ids_for_export,
    write_score_partition,
)


def complete_table() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "model": model,
                "fold": 0 if drug_id == 2 else 1,
                "drug_id": drug_id,
                "rna_id": rna_id,
                "score": model_index + drug_id / 10 + rna_id / 100,
            }
            for model_index, model in enumerate(("prior", "residual"))
            for drug_id in (2, 4)
            for rna_id in (0, 1, 2)
        ]
    )


def run_audit(frame: pd.DataFrame):
    return audit_score_table(
        frame,
        expected_drug_ids=(2, 4),
        expected_rna_ids=(0, 1, 2),
        score_column="score",
        expected_models=("prior", "residual"),
        expected_fold_count=2,
    )


def test_complete_cartesian_scores_pass() -> None:
    audit = run_audit(complete_table())

    assert audit.status == "PASS"
    assert audit.issues == ()
    assert audit.model_count == 2
    assert audit.expected_rows_per_model == 6


def test_missing_coordinate_and_duplicate_are_rejected() -> None:
    frame = complete_table()
    frame.loc[0, ["drug_id", "rna_id"]] = frame.loc[1, ["drug_id", "rna_id"]]

    audit = run_audit(frame)

    assert audit.status == "INPUT_BLOCKED"
    assert any("duplicate" in issue for issue in audit.issues)
    assert any("lacks exact Cartesian coverage" in issue for issue in audit.issues)


def test_extra_id_and_non_finite_score_are_rejected() -> None:
    frame = complete_table()
    frame.loc[0, "rna_id"] = 9
    frame.loc[1, "score"] = np.inf

    audit = run_audit(frame)

    assert audit.status == "INPUT_BLOCKED"
    assert any("non-finite" in issue for issue in audit.issues)
    assert any("extra_rnas=1" in issue for issue in audit.issues)


def test_one_drug_cannot_span_outer_folds() -> None:
    frame = complete_table()
    frame.loc[
        (frame["model"] == "prior")
        & (frame["drug_id"] == 2)
        & (frame["rna_id"] == 0),
        "fold",
    ] = 99

    audit = run_audit(frame)

    assert audit.status == "INPUT_BLOCKED"
    assert any("span multiple" in issue for issue in audit.issues)


def test_frozen_model_manifest_is_required() -> None:
    frame = complete_table()

    audit = audit_score_table(
        frame,
        expected_drug_ids=(2, 4),
        expected_rna_ids=(0, 1, 2),
        score_column="score",
        expected_fold_count=2,
    )

    assert audit.status == "INPUT_BLOCKED"
    assert "frozen model manifest was not provided" in audit.issues


def test_provided_fold_cannot_be_missing() -> None:
    frame = complete_table()
    frame.loc[0, "fold"] = np.nan

    audit = run_audit(frame)

    assert audit.status == "INPUT_BLOCKED"
    assert any("fold contains" in issue and "missing" in issue for issue in audit.issues)


def test_table_rejects_missing_fold_and_label_column() -> None:
    frame = complete_table().drop(columns="fold")
    frame["label"] = 0

    audit = run_audit(frame)

    assert audit.status == "INPUT_BLOCKED"
    assert any("exact columns" in issue for issue in audit.issues)


def test_table_rejects_cross_model_fold_mapping_mismatch() -> None:
    frame = complete_table()
    frame.loc[frame["model"].eq("residual"), "fold"] = (
        1 - frame.loc[frame["model"].eq("residual"), "fold"]
    )

    audit = run_audit(frame)

    assert audit.status == "INPUT_BLOCKED"
    assert any("different outer folds across models" in issue for issue in audit.issues)


def minimal_manifest(tmp_path, pair_matrix_path):
    evidence = tmp_path / "evidence.csv"
    generator = tmp_path / "generator.py"
    evidence.write_text("model\nprior\n")
    generator.write_text("# frozen\n")
    return {
        "schema_version": 1,
        "status": "FROZEN_PRE_SCORE_EXPORT",
        "candidates": [
            {"id": "prior", "source_group": "native"},
            {"id": "residual", "source_group": "native"},
        ],
        "dataset": {
            "active_drug_count": 2,
            "rna_candidate_count": 3,
            "expected_rows_per_candidate": 6,
            "fold_count": 2,
            "pair_matrix_sha256": file_sha256(pair_matrix_path),
        },
        "required_score_artifact": {
            "expected_candidate_count": 2,
            "expected_rows_per_candidate": 6,
        },
        "export_utility": {
            "path": generator.name,
            "sha256": file_sha256(generator),
        },
        "source_groups": [
            {
                "family": "native",
                "eligibility_artifact": evidence.name,
                "eligibility_sha256": file_sha256(evidence),
                "generator_script": generator.name,
                "current_generator_sha256": file_sha256(generator),
                "export_runtime_contract": {"folds": 2, "split_seed": 7},
            }
        ],
    }


def test_candidate_manifest_validates_evidence_and_dimensions(tmp_path) -> None:
    pair_matrix_path = tmp_path / "pair.npy"
    np.save(pair_matrix_path, np.array([[1, 0], [0, 1], [0, 0]], dtype=np.int8))
    manifest = minimal_manifest(tmp_path, pair_matrix_path)

    model_ids, issues = validate_candidate_manifest(
        manifest, root=tmp_path, pair_matrix_path=pair_matrix_path
    )

    assert model_ids == ("prior", "residual")
    assert issues == ()


def test_candidate_manifest_rejects_duplicate_id_and_hash_drift(tmp_path) -> None:
    pair_matrix_path = tmp_path / "pair.npy"
    np.save(pair_matrix_path, np.array([[1, 0], [0, 1], [0, 0]], dtype=np.int8))
    manifest = minimal_manifest(tmp_path, pair_matrix_path)
    manifest["candidates"][1]["id"] = "prior"
    (tmp_path / "evidence.csv").write_text("changed\n")

    _, issues = validate_candidate_manifest(
        manifest, root=tmp_path, pair_matrix_path=pair_matrix_path
    )

    assert "candidate manifest contains duplicate model ids" in issues
    assert any("hash mismatch" in issue for issue in issues)


def test_candidate_manifest_rejects_dependency_code_hash_drift(tmp_path) -> None:
    pair_matrix_path = tmp_path / "pair.npy"
    np.save(pair_matrix_path, np.array([[1, 0], [0, 1], [0, 0]], dtype=np.int8))
    manifest = minimal_manifest(tmp_path, pair_matrix_path)
    dependency = tmp_path / "estimator.py"
    dependency.write_text("# frozen dependency\n")
    manifest["source_groups"][0]["code_artifacts"] = [
        {"path": dependency.name, "sha256": file_sha256(dependency)}
    ]
    dependency.write_text("# drifted dependency\n")

    _, issues = validate_candidate_manifest(
        manifest, root=tmp_path, pair_matrix_path=pair_matrix_path
    )

    assert any("code artifact hash mismatch" in issue for issue in issues)


def test_partitioned_scores_pass_without_loading_one_global_table(tmp_path) -> None:
    fold_pairs = {
        0: np.array([[0, 2], [1, 2], [2, 2]], dtype=np.int64),
        1: np.array([[0, 4], [1, 4], [2, 4]], dtype=np.int64),
    }
    for model_index, model in enumerate(("prior", "residual")):
        for fold, pairs in fold_pairs.items():
            write_score_partition(
                tmp_path,
                model=model,
                fold=fold,
                pairs=pairs,
                scores=np.arange(len(pairs), dtype=np.float64) + model_index,
            )

    audit = audit_score_partition_directory(
        tmp_path,
        expected_drug_ids=(2, 4),
        expected_rna_ids=(0, 1, 2),
        expected_models=("prior", "residual"),
        expected_fold_count=2,
    )

    assert audit.status == "PASS"
    assert audit.row_count == 12


@pytest.mark.parametrize(
    ("column", "bad_value", "message"),
    (
        ("model", "other", "does not match path model"),
        ("model", None, "missing or empty model"),
        ("fold", 1, "does not match path fold"),
    ),
)
def test_partition_content_identity_must_match_path(
    tmp_path, column: str, bad_value, message: str
) -> None:
    pairs = np.array([[0, 2], [1, 2], [2, 2]], dtype=np.int64)
    path = write_score_partition(
        tmp_path,
        model="prior",
        fold=0,
        pairs=pairs,
        scores=np.array([0.1, 0.2, 0.3]),
    )
    frame = pd.read_parquet(path)
    frame[column] = bad_value
    frame.to_parquet(path, index=False)

    audit = audit_score_partition_directory(
        tmp_path,
        expected_drug_ids=(2,),
        expected_rna_ids=(0, 1, 2),
        expected_models=("prior",),
        expected_fold_count=2,
    )

    assert audit.status == "INPUT_BLOCKED"
    assert any(message in issue for issue in audit.issues)


def test_partitioned_scores_reject_incomplete_rna_coverage(tmp_path) -> None:
    pairs = np.array([[0, 2], [1, 2]], dtype=np.int64)
    write_score_partition(
        tmp_path,
        model="prior",
        fold=0,
        pairs=pairs,
        scores=np.array([0.1, 0.2]),
    )

    audit = audit_score_partition_directory(
        tmp_path,
        expected_drug_ids=(2,),
        expected_rna_ids=(0, 1, 2),
        expected_models=("prior",),
        expected_fold_count=1,
    )

    assert audit.status == "INPUT_BLOCKED"
    assert any("incomplete RNA coverage" in issue for issue in audit.issues)


def test_partitioned_scores_reject_cross_model_fold_mapping_mismatch(tmp_path) -> None:
    pairs_by_drug = {
        2: np.array([[0, 2], [1, 2], [2, 2]], dtype=np.int64),
        4: np.array([[0, 4], [1, 4], [2, 4]], dtype=np.int64),
    }
    for model, assignments in {
        "prior": {2: 0, 4: 1},
        "residual": {2: 1, 4: 0},
    }.items():
        for drug_id, fold in assignments.items():
            write_score_partition(
                tmp_path,
                model=model,
                fold=fold,
                pairs=pairs_by_drug[drug_id],
                scores=np.arange(3, dtype=np.float64),
            )

    audit = audit_score_partition_directory(
        tmp_path,
        expected_drug_ids=(2, 4),
        expected_rna_ids=(0, 1, 2),
        expected_models=("prior", "residual"),
        expected_fold_count=2,
    )

    assert audit.status == "INPUT_BLOCKED"
    assert any("different outer folds across models" in issue for issue in audit.issues)


def test_export_preflight_rejects_runtime_contract_mismatch(tmp_path) -> None:
    pair_matrix_path = tmp_path / "pair.npy"
    np.save(pair_matrix_path, np.array([[1, 0], [0, 1], [0, 0]], dtype=np.int8))
    manifest = minimal_manifest(tmp_path, pair_matrix_path)
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(__import__("json").dumps(manifest))

    with pytest.raises(ValueError, match="runtime contract mismatch for split_seed"):
        validated_candidate_ids_for_export(
            manifest_path,
            root=tmp_path,
            pair_matrix_path=pair_matrix_path,
            source_group="native",
            runtime_parameters={"folds": 2, "split_seed": 8},
        )


def test_export_preflight_returns_only_validated_source_group(tmp_path) -> None:
    pair_matrix_path = tmp_path / "pair.npy"
    np.save(pair_matrix_path, np.array([[1, 0], [0, 1], [0, 0]], dtype=np.int8))
    manifest = minimal_manifest(tmp_path, pair_matrix_path)
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(__import__("json").dumps(manifest))

    model_ids = validated_candidate_ids_for_export(
        manifest_path,
        root=tmp_path,
        pair_matrix_path=pair_matrix_path,
        source_group="native",
        runtime_parameters={"folds": 2, "split_seed": 7},
    )

    assert model_ids == ("prior", "residual")


def test_seed_qualified_export_ids_are_bound_by_manifest_seed(tmp_path) -> None:
    pair_matrix_path = tmp_path / "pair.npy"
    np.save(pair_matrix_path, np.array([[1, 0], [0, 1], [0, 0]], dtype=np.int8))
    manifest = minimal_manifest(tmp_path, pair_matrix_path)
    manifest["candidates"] = [
        {"id": "model@11", "source_group": "native", "training_seed": 11},
        {"id": "model@13", "source_group": "native", "training_seed": 13},
    ]
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(__import__("json").dumps(manifest))

    mapping = validated_seed_candidate_ids_for_export(
        manifest_path,
        root=tmp_path,
        pair_matrix_path=pair_matrix_path,
        source_group="native",
        runtime_parameters={"folds": 2, "split_seed": 7},
    )

    assert mapping == {11: "model@11", 13: "model@13"}


def test_seed_qualified_export_ids_reject_duplicate_seed(tmp_path) -> None:
    pair_matrix_path = tmp_path / "pair.npy"
    np.save(pair_matrix_path, np.array([[1, 0], [0, 1], [0, 0]], dtype=np.int8))
    manifest = minimal_manifest(tmp_path, pair_matrix_path)
    manifest["candidates"] = [
        {"id": "model-a", "source_group": "native", "training_seed": 11},
        {"id": "model-b", "source_group": "native", "training_seed": 11},
    ]
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(__import__("json").dumps(manifest))

    with pytest.raises(ValueError, match="duplicate training_seed"):
        validated_seed_candidate_ids_for_export(
            manifest_path,
            root=tmp_path,
            pair_matrix_path=pair_matrix_path,
            source_group="native",
            runtime_parameters={"folds": 2, "split_seed": 7},
        )


def test_score_partition_is_immutable(tmp_path) -> None:
    pairs = np.array([[0, 2], [1, 2], [2, 2]], dtype=np.int64)
    write_score_partition(
        tmp_path,
        model="prior",
        fold=0,
        pairs=pairs,
        scores=np.array([0.1, 0.2, 0.3]),
    )

    with pytest.raises(FileExistsError, match="already exists"):
        write_score_partition(
            tmp_path,
            model="prior",
            fold=0,
            pairs=pairs,
            scores=np.array([0.3, 0.2, 0.1]),
        )


def test_manifest_source_group_drives_exact_export_set(tmp_path) -> None:
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        '{"candidates": ['
        '{"id": "prior", "source_group": "native"},'
        '{"id": "residual", "source_group": "native"},'
        '{"id": "external", "source_group": "other"}'
        ']}'
    )
    expected = candidate_ids_for_source_group(manifest_path, "native")
    pairs = np.array([[0, 2], [1, 2], [2, 2]], dtype=np.int64)

    paths = export_frozen_score_partitions(
        tmp_path / "scores",
        fold=0,
        pairs=pairs,
        predictions={
            "prior": np.array([0.1, 0.2, 0.3]),
            "residual": np.array([0.3, 0.2, 0.1]),
            "ignored": np.array([0.0, 0.0, 0.0]),
        },
        expected_models=expected,
    )

    assert expected == ("prior", "residual")
    assert [path.parent.name for path in paths] == ["model=prior", "model=residual"]
    assert not (tmp_path / "scores" / "model=ignored").exists()


def test_export_rejects_missing_manifest_model(tmp_path) -> None:
    pairs = np.array([[0, 2]], dtype=np.int64)

    with pytest.raises(ValueError, match="missing frozen models: residual"):
        export_frozen_score_partitions(
            tmp_path,
            fold=0,
            pairs=pairs,
            predictions={"prior": np.array([0.1])},
            expected_models=("prior", "residual"),
        )
