"""Validate frozen full-candidate score tables before MC-TV-PU selection."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ScoreInputAudit:
    status: str
    issues: tuple[str, ...]
    model_count: int
    row_count: int
    expected_drug_count: int
    expected_rna_count: int
    expected_rows_per_model: int

    def to_dict(self) -> dict[str, object]:
        result = asdict(self)
        result["issues"] = list(self.issues)
        return result


PARTITION_COLUMNS = ("model", "fold", "drug_id", "rna_id", "score")
SAFE_MODEL_ID = re.compile(r"^[A-Za-z0-9_.@-]+$")


def _integer_series(frame: pd.DataFrame, column: str) -> tuple[pd.Series, str | None]:
    numeric = pd.to_numeric(frame[column], errors="coerce")
    if numeric.isna().any():
        return numeric, f"{column} contains non-numeric or missing values"
    if not np.equal(numeric.to_numpy(), np.floor(numeric.to_numpy())).all():
        return numeric, f"{column} contains non-integer values"
    return numeric.astype(np.int64), None


def audit_score_table(
    frame: pd.DataFrame,
    *,
    expected_drug_ids: Iterable[int],
    expected_rna_ids: Iterable[int],
    score_column: str,
    expected_fold_count: int,
    expected_models: Iterable[str] | None = None,
) -> ScoreInputAudit:
    """Audit an in-memory copy of the strict five-column score contract."""

    expected_drugs = frozenset(int(value) for value in expected_drug_ids)
    expected_rnas = frozenset(int(value) for value in expected_rna_ids)
    if not expected_drugs or not expected_rnas:
        raise ValueError("expected drug and RNA ID sets must be nonempty")
    if expected_fold_count <= 0:
        raise ValueError("expected_fold_count must be positive")
    if score_column != "score" or tuple(frame.columns) != PARTITION_COLUMNS:
        return ScoreInputAudit(
            status="INPUT_BLOCKED",
            issues=(
                f"score table must have exact columns {PARTITION_COLUMNS!r}; "
                f"observed {tuple(frame.columns)!r}",
            ),
            model_count=0,
            row_count=len(frame),
            expected_drug_count=len(expected_drugs),
            expected_rna_count=len(expected_rnas),
            expected_rows_per_model=len(expected_drugs) * len(expected_rnas),
        )

    issues: list[str] = []
    working = frame.copy()
    for column in ("fold", "drug_id", "rna_id"):
        working[column], issue = _integer_series(working, column)
        if issue is not None:
            issues.append(issue)
    if issues:
        return ScoreInputAudit(
            status="INPUT_BLOCKED",
            issues=tuple(issues),
            model_count=int(working["model"].nunique(dropna=True)),
            row_count=len(working),
            expected_drug_count=len(expected_drugs),
            expected_rna_count=len(expected_rnas),
            expected_rows_per_model=len(expected_drugs) * len(expected_rnas),
        )

    if working["model"].isna().any() or (working["model"].astype(str).str.len() == 0).any():
        issues.append("model contains missing or empty values")
    working["model"] = working["model"].astype(str)

    scores = pd.to_numeric(working[score_column], errors="coerce").to_numpy()
    if not np.isfinite(scores).all():
        issues.append(f"{score_column} contains non-finite or non-numeric values")

    duplicate_count = int(
        working.duplicated(["model", "drug_id", "rna_id"], keep=False).sum()
    )
    if duplicate_count:
        issues.append(
            f"{duplicate_count} rows participate in duplicate model/drug/RNA coordinates"
        )

    observed_models = frozenset(working["model"].unique())
    if expected_models is None:
        issues.append("frozen model manifest was not provided")
    else:
        required_models = frozenset(str(value) for value in expected_models)
        if not required_models:
            issues.append("frozen model manifest is empty")
        missing_models = sorted(required_models - observed_models)
        extra_models = sorted(observed_models - required_models)
        if missing_models:
            issues.append(f"missing frozen models: {', '.join(missing_models)}")
        if extra_models:
            issues.append(f"unexpected models: {', '.join(extra_models)}")

    expected_rows = len(expected_drugs) * len(expected_rnas)
    for model, group in working.groupby("model", sort=True, dropna=False):
        observed_drugs = frozenset(group["drug_id"].tolist())
        observed_rnas = frozenset(group["rna_id"].tolist())
        missing_drugs = len(expected_drugs - observed_drugs)
        extra_drugs = len(observed_drugs - expected_drugs)
        missing_rnas = len(expected_rnas - observed_rnas)
        extra_rnas = len(observed_rnas - expected_rnas)
        unique_coordinates = int(group[["drug_id", "rna_id"]].drop_duplicates().shape[0])
        if (
            missing_drugs
            or extra_drugs
            or missing_rnas
            or extra_rnas
            or unique_coordinates != expected_rows
        ):
            issues.append(
                f"model {model!r} lacks exact Cartesian coverage: "
                f"unique={unique_coordinates}/{expected_rows}, "
                f"missing_drugs={missing_drugs}, extra_drugs={extra_drugs}, "
                f"missing_rnas={missing_rnas}, extra_rnas={extra_rnas}"
            )

    fold_counts = working.groupby(["model", "drug_id"], dropna=False)["fold"].nunique(
        dropna=False
    )
    inconsistent = int((fold_counts != 1).sum())
    if inconsistent:
        issues.append(f"{inconsistent} model/drug groups span multiple outer folds")
    expected_folds = frozenset(range(expected_fold_count))
    for model, group in working.groupby("model", sort=True):
        observed_folds = frozenset(int(value) for value in group["fold"].unique())
        if observed_folds != expected_folds:
            issues.append(
                f"model {model!r} uses folds {sorted(observed_folds)}, "
                f"expected {sorted(expected_folds)}"
            )
    fold_by_model_drug = working[["model", "drug_id", "fold"]].drop_duplicates()
    cross_model_fold_counts = fold_by_model_drug.groupby("drug_id")["fold"].nunique()
    mismatched_drugs = int((cross_model_fold_counts != 1).sum())
    if mismatched_drugs:
        issues.append(
            f"{mismatched_drugs} drugs use different outer folds across models"
        )

    return ScoreInputAudit(
        status="PASS" if not issues else "INPUT_BLOCKED",
        issues=tuple(issues),
        model_count=len(observed_models),
        row_count=len(working),
        expected_drug_count=len(expected_drugs),
        expected_rna_count=len(expected_rnas),
        expected_rows_per_model=expected_rows,
    )


def score_partition_path(output_dir: Path, *, model: str, fold: int) -> Path:
    """Return the only valid immutable partition path for one model/fold."""

    if not SAFE_MODEL_ID.fullmatch(model):
        raise ValueError(f"unsafe model id: {model!r}")
    if isinstance(fold, bool) or not float(fold).is_integer() or int(fold) < 0:
        raise ValueError("fold must be a nonnegative integer")
    return output_dir / f"model={model}" / f"fold={int(fold):02d}.parquet"


def write_score_partition(
    output_dir: Path,
    *,
    model: str,
    fold: int,
    pairs: np.ndarray,
    scores: np.ndarray,
) -> Path:
    """Write one immutable model/fold full-candidate score partition."""

    path = score_partition_path(output_dir, model=model, fold=fold)
    pair_array = np.asarray(pairs)
    score_array = np.asarray(scores, dtype=np.float64)
    if pair_array.ndim != 2 or pair_array.shape[1] != 2:
        raise ValueError("pairs must have shape [n, 2] in RNA/drug order")
    if len(pair_array) != len(score_array):
        raise ValueError("pairs and scores must have the same length")
    if not np.isfinite(score_array).all():
        raise ValueError("scores must be finite")
    if not np.equal(pair_array, np.floor(pair_array)).all() or (pair_array < 0).any():
        raise ValueError("pair ids must be nonnegative integers")
    frame = pd.DataFrame(
        {
            "model": model,
            "fold": int(fold),
            "drug_id": pair_array[:, 1].astype(np.int64),
            "rna_id": pair_array[:, 0].astype(np.int64),
            "score": score_array,
        }
    )
    if frame.duplicated(["drug_id", "rna_id"]).any():
        raise ValueError("score partition contains duplicate drug/RNA coordinates")
    model_dir = path.parent
    model_dir.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp.parquet")
    if path.exists() or temporary.exists():
        raise FileExistsError(f"score partition already exists: {path}")
    frame.to_parquet(temporary, index=False, compression="zstd")
    temporary.replace(path)
    return path


def candidate_ids_for_source_group(manifest_path: Path, source_group: str) -> tuple[str, ...]:
    manifest = json.loads(manifest_path.read_text())
    candidates = manifest.get("candidates")
    if not isinstance(candidates, list):
        raise ValueError("candidate manifest candidates must be a list")
    model_ids = tuple(
        str(candidate["id"])
        for candidate in candidates
        if isinstance(candidate, dict) and candidate.get("source_group") == source_group
    )
    if not model_ids:
        raise ValueError(f"candidate manifest has no models for source_group={source_group!r}")
    if len(model_ids) != len(set(model_ids)):
        raise ValueError(f"candidate manifest repeats a model in source_group={source_group!r}")
    return model_ids


def export_frozen_score_partitions(
    output_dir: Path,
    *,
    fold: int,
    pairs: np.ndarray,
    predictions: dict[str, np.ndarray],
    expected_models: Iterable[str],
) -> tuple[Path, ...]:
    """Export exactly the manifest-selected models from an in-memory prediction map."""

    ordered_models = tuple(expected_models)
    missing = [model for model in ordered_models if model not in predictions]
    if missing:
        raise ValueError(f"full predictions are missing frozen models: {', '.join(missing)}")
    return tuple(
        write_score_partition(
            output_dir,
            model=model,
            fold=fold,
            pairs=pairs,
            scores=predictions[model],
        )
        for model in ordered_models
    )


def audit_score_partition_directory(
    directory: Path,
    *,
    expected_drug_ids: Iterable[int],
    expected_rna_ids: Iterable[int],
    expected_models: Iterable[str],
    expected_fold_count: int,
) -> ScoreInputAudit:
    """Stream-audit immutable model/fold Parquet partitions."""

    expected_drugs = frozenset(int(value) for value in expected_drug_ids)
    expected_rnas = frozenset(int(value) for value in expected_rna_ids)
    required_models = frozenset(str(value) for value in expected_models)
    if not expected_drugs or not expected_rnas or not required_models:
        raise ValueError("expected drug, RNA, and model sets must be nonempty")
    if expected_fold_count <= 0:
        raise ValueError("expected_fold_count must be positive")
    issues: list[str] = []
    files = sorted(directory.glob("model=*/fold=*.parquet"))
    if not files:
        issues.append("score partition directory contains no model/fold Parquet files")
    coverage: dict[str, dict[int, int]] = {}
    observed_models: set[str] = set()
    row_count = 0
    for path in files:
        frame = pd.read_parquet(path)
        row_count += len(frame)
        if tuple(frame.columns) != PARTITION_COLUMNS:
            issues.append(
                f"partition {path} has columns {tuple(frame.columns)!r}, expected {PARTITION_COLUMNS!r}"
            )
            continue
        if frame.empty:
            issues.append(f"partition {path} is empty")
            continue
        if frame["model"].isna().any() or (
            frame["model"].astype(str).str.len() == 0
        ).any():
            issues.append(f"partition {path} contains missing or empty model values")
            continue
        models = frame["model"].astype(str).unique()
        folds, fold_issue = _integer_series(frame, "fold")
        if len(models) != 1:
            issues.append(f"partition {path} must contain exactly one model")
            continue
        if fold_issue is not None or folds.nunique() != 1:
            issues.append(f"partition {path} must contain exactly one integer fold")
            continue
        model = str(models[0])
        fold = int(folds.iloc[0])
        path_model = path.parent.name.removeprefix("model=")
        try:
            path_fold = int(path.stem.removeprefix("fold="))
        except ValueError:
            issues.append(f"partition {path} has an invalid fold path")
            continue
        if model != path_model:
            issues.append(
                f"partition {path} model {model!r} does not match path model {path_model!r}"
            )
        if fold != path_fold:
            issues.append(
                f"partition {path} fold {fold} does not match path fold {path_fold}"
            )
        if fold not in range(expected_fold_count):
            issues.append(
                f"partition {path} uses fold {fold}, expected 0..{expected_fold_count - 1}"
            )
        observed_models.add(model)
        if model not in required_models:
            issues.append(f"partition {path} contains unexpected model {model!r}")
        drug_ids, drug_issue = _integer_series(frame, "drug_id")
        rna_ids, rna_issue = _integer_series(frame, "rna_id")
        if drug_issue is not None or rna_issue is not None:
            issues.extend(issue for issue in (drug_issue, rna_issue) if issue is not None)
            continue
        frame = frame.assign(drug_id=drug_ids, rna_id=rna_ids)
        scores = pd.to_numeric(frame["score"], errors="coerce").to_numpy()
        if not np.isfinite(scores).all():
            issues.append(f"partition {path} contains non-finite scores")
        if frame.duplicated(["drug_id", "rna_id"]).any():
            issues.append(f"partition {path} contains duplicate drug/RNA coordinates")
            continue
        for drug_id, group in frame.groupby("drug_id", sort=False):
            drug = int(drug_id)
            observed_rnas = frozenset(group["rna_id"].tolist())
            if drug not in expected_drugs:
                issues.append(f"partition {path} contains unexpected drug {drug}")
            if observed_rnas != expected_rnas:
                issues.append(
                    f"partition {path} has incomplete RNA coverage for drug {drug}: "
                    f"{len(observed_rnas)}/{len(expected_rnas)}"
                )
            prior_fold = coverage.setdefault(model, {}).get(drug)
            if prior_fold is not None:
                issues.append(
                    f"model {model!r} drug {drug} appears in folds {prior_fold} and {fold}"
                )
            coverage[model][drug] = fold

    missing_models = sorted(required_models - observed_models)
    if missing_models:
        issues.append(f"missing frozen models: {', '.join(missing_models)}")
    for model in sorted(required_models & observed_models):
        observed_drugs = frozenset(coverage.get(model, {}))
        if observed_drugs != expected_drugs:
            issues.append(
                f"model {model!r} has drug coverage {len(observed_drugs)}/{len(expected_drugs)}"
            )
        observed_folds = frozenset(coverage.get(model, {}).values())
        expected_folds = frozenset(range(expected_fold_count))
        if observed_folds != expected_folds:
            issues.append(
                f"model {model!r} uses folds {sorted(observed_folds)}, "
                f"expected {sorted(expected_folds)}"
            )
    for drug in sorted(expected_drugs):
        folds_for_drug = {
            model: coverage.get(model, {}).get(drug) for model in sorted(required_models)
        }
        observed = {fold for fold in folds_for_drug.values() if fold is not None}
        if len(observed) > 1:
            issues.append(
                f"drug {drug} uses different outer folds across models: {folds_for_drug}"
            )
    expected_rows = len(expected_drugs) * len(expected_rnas)
    return ScoreInputAudit(
        status="PASS" if not issues else "INPUT_BLOCKED",
        issues=tuple(issues),
        model_count=len(observed_models),
        row_count=row_count,
        expected_drug_count=len(expected_drugs),
        expected_rna_count=len(expected_rnas),
        expected_rows_per_model=expected_rows,
    )


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_candidate_manifest(
    manifest: dict[str, object], *, root: Path, pair_matrix_path: Path
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Validate a frozen candidate manifest and return its ordered model IDs."""

    issues: list[str] = []
    if manifest.get("schema_version") != 1:
        issues.append("candidate manifest schema_version must equal 1")
    if manifest.get("status") != "FROZEN_PRE_SCORE_EXPORT":
        issues.append("candidate manifest is not in FROZEN_PRE_SCORE_EXPORT state")

    raw_candidates = manifest.get("candidates")
    if not isinstance(raw_candidates, list):
        return (), ("candidate manifest candidates must be a list",)
    model_ids: list[str] = []
    for index, candidate in enumerate(raw_candidates):
        if not isinstance(candidate, dict) or not isinstance(candidate.get("id"), str):
            issues.append(f"candidate {index} has no string id")
            continue
        model_ids.append(candidate["id"])
    if len(model_ids) != len(set(model_ids)):
        issues.append("candidate manifest contains duplicate model ids")

    dataset = manifest.get("dataset")
    required_score = manifest.get("required_score_artifact")
    if not isinstance(dataset, dict) or not isinstance(required_score, dict):
        issues.append("candidate manifest lacks dataset or required_score_artifact")
        return tuple(model_ids), tuple(issues)
    fold_count = dataset.get("fold_count")
    if not isinstance(fold_count, int) or isinstance(fold_count, bool) or fold_count <= 0:
        issues.append("manifest dataset.fold_count must be a positive integer")
    expected_candidate_count = required_score.get("expected_candidate_count")
    if expected_candidate_count != len(model_ids):
        issues.append(
            "candidate count does not match required_score_artifact.expected_candidate_count"
        )

    pair_matrix = np.load(pair_matrix_path, allow_pickle=False)
    if pair_matrix.ndim != 2:
        issues.append("pair matrix must be two-dimensional")
    else:
        active_drugs = int((pair_matrix > 0).any(axis=0).sum())
        expected_rows = active_drugs * int(pair_matrix.shape[0])
        for key, observed in (
            ("active_drug_count", active_drugs),
            ("rna_candidate_count", int(pair_matrix.shape[0])),
            ("expected_rows_per_candidate", expected_rows),
        ):
            if dataset.get(key) != observed:
                issues.append(f"manifest dataset.{key} does not match the pair matrix")
        if required_score.get("expected_rows_per_candidate") != expected_rows:
            issues.append(
                "required_score_artifact.expected_rows_per_candidate does not match the pair matrix"
            )
    if dataset.get("pair_matrix_sha256") != file_sha256(pair_matrix_path):
        issues.append("pair matrix SHA-256 does not match candidate manifest")

    def validate_artifacts(
        raw_artifacts: object, *, owner: str, artifact_kind: str
    ) -> None:
        if raw_artifacts is None:
            return
        if not isinstance(raw_artifacts, list):
            issues.append(f"{owner} {artifact_kind}_artifacts must be a list")
            return
        for artifact_index, artifact in enumerate(raw_artifacts):
            if not isinstance(artifact, dict):
                issues.append(
                    f"{owner} {artifact_kind} artifact {artifact_index} is not an object"
                )
                continue
            raw_path = artifact.get("path")
            expected_hash = artifact.get("sha256")
            path = root / str(raw_path or "")
            if not raw_path or not path.is_file():
                issues.append(
                    f"{owner} {artifact_kind} artifact {raw_path!r} is missing"
                )
            elif expected_hash != file_sha256(path):
                issues.append(
                    f"{owner} {artifact_kind} artifact hash mismatch for {raw_path}"
                )

    validate_artifacts(
        manifest.get("input_artifacts"), owner="manifest", artifact_kind="input"
    )

    export_utility = manifest.get("export_utility")
    if not isinstance(export_utility, dict):
        issues.append("candidate manifest lacks export_utility")
    else:
        utility_path = root / str(export_utility.get("path", ""))
        if not utility_path.is_file():
            issues.append("candidate manifest export_utility is missing")
        elif export_utility.get("sha256") != file_sha256(utility_path):
            issues.append("candidate manifest export_utility hash mismatch")

    source_groups = manifest.get("source_groups")
    if not isinstance(source_groups, list):
        issues.append("candidate manifest source_groups must be a list")
        return tuple(model_ids), tuple(issues)
    group_names: list[str] = []
    for group_index, group in enumerate(source_groups):
        if not isinstance(group, dict):
            issues.append(f"source group {group_index} is not an object")
            continue
        family = group.get("family")
        if not isinstance(family, str) or not family:
            issues.append(f"source group {group_index} has no family")
        else:
            group_names.append(family)
        validate_artifacts(
            group.get("input_artifacts"),
            owner=f"source group {group_index}",
            artifact_kind="input",
        )
        validate_artifacts(
            group.get("code_artifacts"),
            owner=f"source group {group_index}",
            artifact_kind="code",
        )
        for artifact_key in (
            "eligibility_artifact",
            "config_artifact",
            "selection_artifact",
            "generator_script",
        ):
            raw_path = group.get(artifact_key)
            if raw_path is None:
                continue
            hash_key = (
                "current_generator_sha256"
                if artifact_key == "generator_script"
                else artifact_key.replace("_artifact", "_sha256")
            )
            path = root / str(raw_path)
            if not path.is_file():
                issues.append(f"source group {group_index} is missing {raw_path}")
            elif group.get(hash_key) != file_sha256(path):
                issues.append(f"source group {group_index} hash mismatch for {raw_path}")
    if len(group_names) != len(set(group_names)):
        issues.append("candidate manifest contains duplicate source group families")
    known_groups = frozenset(group_names)
    for candidate in raw_candidates:
        if not isinstance(candidate, dict) or not isinstance(candidate.get("id"), str):
            continue
        source_group = candidate.get("source_group")
        if source_group not in known_groups:
            issues.append(
                f"candidate {candidate['id']!r} refers to unknown source group {source_group!r}"
            )
    return tuple(model_ids), tuple(issues)


def validated_candidate_ids_for_export(
    manifest_path: Path,
    *,
    root: Path,
    pair_matrix_path: Path,
    source_group: str,
    runtime_parameters: dict[str, object],
) -> tuple[str, ...]:
    """Fail before training unless manifest, inputs, and runtime semantics match."""

    manifest = json.loads(manifest_path.read_text())
    _, issues = validate_candidate_manifest(
        manifest, root=root, pair_matrix_path=pair_matrix_path
    )
    if issues:
        raise ValueError("candidate manifest invalid: " + "; ".join(issues))
    raw_groups = manifest.get("source_groups")
    groups = [
        group
        for group in raw_groups
        if isinstance(group, dict) and group.get("family") == source_group
    ]
    if len(groups) != 1:
        raise ValueError(
            f"candidate manifest must contain exactly one source group {source_group!r}"
        )
    contract = groups[0].get("export_runtime_contract")
    if not isinstance(contract, dict) or not contract:
        raise ValueError(f"source group {source_group!r} lacks export_runtime_contract")
    path_arguments = {
        str(artifact["argument"])
        for artifact in groups[0].get("input_artifacts", [])
        if isinstance(artifact, dict) and artifact.get("argument") is not None
    }
    for key, expected in contract.items():
        actual = runtime_parameters.get(key)
        if (
            (key.endswith("_dir") or key in path_arguments)
            and isinstance(expected, str)
            and actual is not None
        ):
            matches = Path(actual).resolve() == (root / expected).resolve()
        else:
            matches = actual == expected
        if not matches:
            raise ValueError(
                f"runtime contract mismatch for {key}: observed {actual!r}, expected {expected!r}"
            )
    for artifact in groups[0].get("input_artifacts", []):
        argument = artifact.get("argument")
        if argument is None:
            continue
        actual_path = runtime_parameters.get(str(argument))
        expected_path = (root / str(artifact["path"])).resolve()
        if actual_path is None or Path(actual_path).resolve() != expected_path:
            raise ValueError(
                f"runtime contract mismatch for {argument}: observed {actual_path!r}, "
                f"expected {str(expected_path)!r}"
            )
    return candidate_ids_for_source_group(manifest_path, source_group)


def validated_seed_candidate_ids_for_export(
    manifest_path: Path,
    *,
    root: Path,
    pair_matrix_path: Path,
    source_group: str,
    runtime_parameters: dict[str, object],
) -> dict[int, str]:
    """Bind every seed-qualified candidate to its manifest training seed."""

    model_ids = validated_candidate_ids_for_export(
        manifest_path,
        root=root,
        pair_matrix_path=pair_matrix_path,
        source_group=source_group,
        runtime_parameters=runtime_parameters,
    )
    manifest = json.loads(manifest_path.read_text())
    selected = {
        str(candidate["id"]): candidate
        for candidate in manifest["candidates"]
        if isinstance(candidate, dict) and candidate.get("source_group") == source_group
    }
    mapping: dict[int, str] = {}
    for model_id in model_ids:
        seed = selected[model_id].get("training_seed")
        if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
            raise ValueError(
                f"candidate {model_id!r} must have a nonnegative integer training_seed"
            )
        if seed in mapping:
            raise ValueError(
                f"duplicate training_seed {seed} for {mapping[seed]!r} and {model_id!r}"
            )
        mapping[seed] = model_id
    return mapping


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--score-dir", type=Path, required=True)
    parser.add_argument("--pair-matrix", type=Path, default=Path("Dataset/MiPair.npy"))
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()

    pair_matrix = np.load(args.pair_matrix, allow_pickle=False)
    if pair_matrix.ndim != 2:
        raise ValueError("pair matrix must be two-dimensional")
    expected_drugs = np.flatnonzero((pair_matrix > 0).any(axis=0))
    expected_rnas = np.arange(pair_matrix.shape[0], dtype=np.int64)
    manifest = json.loads(args.manifest.read_text())
    expected_models, manifest_issues = validate_candidate_manifest(
        manifest, root=Path.cwd(), pair_matrix_path=args.pair_matrix
    )
    dataset = manifest.get("dataset")
    expected_fold_count = dataset.get("fold_count") if isinstance(dataset, dict) else 0
    if not isinstance(expected_fold_count, int) or isinstance(expected_fold_count, bool):
        expected_fold_count = 0
    manifest_sha256 = file_sha256(args.manifest)
    if manifest_issues:
        audit = ScoreInputAudit(
            status="INPUT_BLOCKED",
            issues=manifest_issues,
            model_count=0,
            row_count=0,
            expected_drug_count=len(expected_drugs),
            expected_rna_count=len(expected_rnas),
            expected_rows_per_model=len(expected_drugs) * len(expected_rnas),
        )
    else:
        audit = audit_score_partition_directory(
            args.score_dir,
            expected_drug_ids=expected_drugs,
            expected_rna_ids=expected_rnas,
            expected_models=expected_models,
            expected_fold_count=expected_fold_count,
        )
    payload = audit.to_dict()
    payload.update(
        {
            "score_dir": str(args.score_dir),
            "pair_matrix_path": str(args.pair_matrix),
            "pair_matrix_sha256": file_sha256(args.pair_matrix),
            "score_column": "score",
            "manifest_path": str(args.manifest),
            "manifest_sha256": manifest_sha256,
            "model_training_run": False,
        }
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if audit.status == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
