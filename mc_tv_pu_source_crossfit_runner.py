"""Run selector-anchor source cross-fit over frozen target-drug-OOF scores."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
from collections import defaultdict
from collections.abc import Iterable, Iterator, Mapping, Sequence
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pandas as pd

from generalization_audit import load_drug_scaffolds, scaffold_fold_assignment
from mc_tv_pu_q_tau_semantics_audit import (
    restrict_pair_sources_to_positive_matrix,
    source_sets_by_drug,
)
from mc_tv_pu_score_input_gate import (
    ScoreInputAudit,
    audit_score_partition_directory,
    score_partition_path,
    validate_candidate_manifest,
)
from mc_tv_pu_selector_evaluation import (
    CENTER_MODES,
    CHECKPOINT_COLUMNS,
    TAU_GRID,
    _empirical_center_values,
    _validate_checkpoint_empirical_values,
    aggregate_selector_intervals,
    checkpoint_path,
    evaluate_drug_center,
    file_sha256,
    iter_fold_score_matrices,
    run_configuration,
    source_center_profile,
    validate_checkpoint,
    write_checkpoint,
    write_or_validate_run_configuration,
    write_or_validate_text,
)
from mc_tv_pu_source_isolated_validation import (
    BOOTSTRAP_ITERATIONS,
    BOOTSTRAP_SEED,
    COMPATIBLE_SELECTION_RULE,
    CROSS_FIT_FOLD_COUNT,
    CROSS_FIT_SEED,
    METRICS,
    PRIMARY_BASELINES,
    SELECTION_RULE_ABLATIONS,
    SourceCrossFitEpisode,
    bootstrap_crossfit_rule_ablation_performance,
    bootstrap_crossfit_set_valued_performance,
    build_source_crossfit_episodes,
    evaluate_crossfit_episode_scores,
    evaluate_rule_ablation_claim_routing,
    evaluate_success_gate,
    source_crossfit_coverage_summary,
)
from source_balanced_pairwise_pilot import build_pair_sources

CROSSFIT_METRIC_COLUMNS = (
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
)

IMPLEMENTATION_SOURCE_FILES = (
    "generalization_audit.py",
    "mc_tv_pu_oracle.py",
    "mc_tv_pu_q_tau_semantics_audit.py",
    "mc_tv_pu_score_input_gate.py",
    "mc_tv_pu_selector_evaluation.py",
    "mc_tv_pu_source_crossfit_runner.py",
    "mc_tv_pu_source_isolated_validation.py",
    "source_balanced_pairwise_pilot.py",
)

CROSSFIT_FINAL_ARTIFACTS = (
    "run_config.json",
    "source_fold_assignments.json",
    "crossfit_candidate_intervals.csv",
    "crossfit_episode_metrics.csv",
    "crossfit_inference.csv",
    "crossfit_rule_ablation_inference.csv",
)


@contextmanager
def exclusive_output_lock(output_dir: Path) -> Iterator[Path]:
    """Hold a nonblocking process lock for one formal output directory."""

    lock_path = output_dir.parent / f".{output_dir.name}.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    handle = lock_path.open("a+")
    try:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(
                f"cross-fit output directory is already locked: {output_dir}"
            ) from exc
        handle.seek(0)
        handle.truncate()
        handle.write(f"pid={os.getpid()}\noutput_dir={output_dir.resolve()}\n")
        handle.flush()
        yield lock_path
    finally:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        finally:
            handle.close()


def validate_dataset_dir_binding(root: Path, dataset_dir: Path) -> dict[str, str]:
    """Require source mappings to come from the manifest-hashed Dataset directory."""

    expected = (root / "Dataset").resolve()
    observed = dataset_dir.resolve()
    if observed != expected:
        raise ValueError(
            "dataset-dir must equal the manifest-hashed root/Dataset directory: "
            f"expected={expected} observed={observed}"
        )
    return {"status": "PASS", "dataset_dir": str(observed)}


def crossfit_implementation_hashes(
    source_dir: Path | None = None,
) -> dict[str, str]:
    """Bind formal cross-fit outputs to every directly used local implementation."""

    base_dir = Path(__file__).resolve().parent if source_dir is None else source_dir
    return {
        name: file_sha256(base_dir / name) for name in IMPLEMENTATION_SOURCE_FILES
    }

SELECTOR_PATH_ONLY_CONFIG_KEYS = frozenset(
    {"root", "pair_matrix", "candidate_manifest", "source_table", "score_dir"}
)


def validate_target_drug_oof_assignment(
    score_dir: Path,
    *,
    reference_model: str,
    pair_matrix: np.ndarray,
    drug_scaffolds: np.ndarray,
    fold_count: int,
    split_seed: int,
    score_input_audit: ScoreInputAudit,
) -> dict[str, object]:
    """Bind exported drug folds to the deterministic scaffold test-fold map."""

    if score_input_audit.status != "PASS":
        raise ValueError("target-drug OOF audit requires a passing score input audit")
    folds = scaffold_fold_assignment(
        pair_matrix,
        drug_scaffolds,
        n_folds=fold_count,
        seed=split_seed,
    )
    expected = {
        int(drug_id): fold
        for fold, drug_ids in enumerate(folds)
        for drug_id in drug_ids
    }
    observed: dict[int, int] = {}
    for fold in range(fold_count):
        path = score_partition_path(score_dir, model=reference_model, fold=fold)
        if not path.is_file():
            raise ValueError(f"target-drug OOF reference partition is missing: {path}")
        frame = pd.read_parquet(path, columns=["model", "fold", "drug_id"])
        if set(frame["model"].astype(str)) != {reference_model}:
            raise ValueError(f"target-drug OOF reference model mismatch: {path}")
        fold_values = pd.to_numeric(frame["fold"], errors="coerce")
        if (
            fold_values.isna().any()
            or not np.equal(
                fold_values.to_numpy(), np.floor(fold_values.to_numpy())
            ).all()
            or set(fold_values.astype(int)) != {fold}
        ):
            raise ValueError(f"target-drug OOF reference fold mismatch: {path}")
        drug_values = pd.to_numeric(frame["drug_id"], errors="coerce")
        if drug_values.isna().any() or not np.equal(
            drug_values.to_numpy(), np.floor(drug_values.to_numpy())
        ).all():
            raise ValueError(f"target-drug OOF reference drug ids are invalid: {path}")
        for drug_id in np.unique(drug_values.astype(np.int64)):
            drug = int(drug_id)
            prior = observed.get(drug)
            if prior is not None:
                raise ValueError(
                    f"target-drug OOF reference drug {drug} appears in folds {prior} and {fold}"
                )
            observed[drug] = fold
    if observed != expected:
        mismatched = sorted(
            drug
            for drug in set(expected) | set(observed)
            if expected.get(drug) != observed.get(drug)
        )
        raise ValueError(
            "score partitions do not match the deterministic scaffold test-fold map: "
            f"mismatched_drugs={mismatched[:10]} total={len(mismatched)}"
        )
    assignment_payload = json.dumps(expected, sort_keys=True, separators=(",", ":"))
    scaffold_payload = json.dumps(
        [str(value) for value in drug_scaffolds], separators=(",", ":")
    )
    return {
        "status": "PASS",
        "basis": "recomputed_bemis_murcko_scaffold_test_fold",
        "reference_model": reference_model,
        "drug_count": len(expected),
        "fold_count": fold_count,
        "split_seed": split_seed,
        "assignment_sha256": hashlib.sha256(assignment_payload.encode()).hexdigest(),
        "scaffold_sha256": hashlib.sha256(scaffold_payload.encode()).hexdigest(),
        "cross_model_fold_mapping_verified_by_score_input_gate": True,
    }


def source_isolation_scope_metadata(
    target_drug_oof_audit: Mapping[str, object],
) -> dict[str, object]:
    """Declare the exact isolation scope without implying source-disjoint training."""

    if target_drug_oof_audit.get("status") != "PASS":
        raise ValueError("target-drug OOF evidence must pass before declaring scope")
    return {
        "target_drug_oof": True,
        "target_drug_oof_audit": dict(target_drug_oof_audit),
        "selector_anchor_source_disjoint": True,
        "source_isolation_scope": "selector_anchors_only",
        "base_model_source_disjoint": False,
        "episode_membership_uses_heldout_source_presence": True,
    }


def validate_main_selector_completion(
    selector_dir: Path,
    *,
    exit_log: Path,
    expected_config: Mapping[str, object],
    candidate_ids: Sequence[str],
    active_drugs: Sequence[int],
) -> dict[str, object]:
    """Require a terminal, complete, and reproducible main-selector run."""

    if not selector_dir.is_dir():
        raise ValueError(f"main selector directory does not exist: {selector_dir}")
    if not exit_log.is_file():
        raise ValueError(f"main selector exit log does not exist: {exit_log}")
    log_lines = [line.strip() for line in exit_log.read_text().splitlines() if line.strip()]
    if not log_lines or log_lines[-1] != "EXIT_STATUS=0":
        raise ValueError("main selector log does not end with EXIT_STATUS=0")

    unfinished = sorted(selector_dir.rglob("*.tmp.parquet")) + sorted(
        selector_dir.rglob("*.partial")
    )
    if unfinished:
        raise ValueError(
            "main selector contains unfinished files: "
            + ", ".join(str(path) for path in unfinished[:5])
        )

    config_path = selector_dir / "run_config.json"
    if not config_path.is_file():
        raise ValueError("main selector run_config.json is missing")
    actual_config = json.loads(config_path.read_text())
    expected_semantics = {
        key: value
        for key, value in expected_config.items()
        if key not in SELECTOR_PATH_ONLY_CONFIG_KEYS
    }
    mismatched_keys = [
        key
        for key, value in expected_semantics.items()
        if actual_config.get(key) != value
    ]
    if mismatched_keys:
        raise ValueError(
            "main selector frozen configuration mismatch: "
            + ", ".join(sorted(mismatched_keys))
        )

    expected_paths = {
        checkpoint_path(selector_dir, center=center, drug_id=int(drug_id))
        for center in CENTER_MODES
        for drug_id in active_drugs
    }
    actual_paths = set(selector_dir.glob("oracle/center=*/drug=*.parquet"))
    missing = sorted(expected_paths - actual_paths)
    extra = sorted(actual_paths - expected_paths)
    if missing or extra:
        raise ValueError(
            "main selector checkpoint coverage mismatch: "
            f"expected={len(expected_paths)} actual={len(actual_paths)} "
            f"missing={len(missing)} extra={len(extra)}"
        )

    ordered_paths = [
        checkpoint_path(selector_dir, center=center, drug_id=int(drug_id))
        for center in CENTER_MODES
        for drug_id in active_drugs
    ]
    for center in CENTER_MODES:
        for drug_id in active_drugs:
            validate_checkpoint(
                checkpoint_path(selector_dir, center=center, drug_id=int(drug_id)),
                center=center,
                drug_id=int(drug_id),
                candidate_ids=candidate_ids,
            )

    intervals, summary = aggregate_selector_intervals(
        (pd.read_parquet(path) for path in ordered_paths),
        candidate_ids=candidate_ids,
    )
    intervals_path = selector_dir / "candidate_intervals.csv"
    summary_path = selector_dir / "selector_summary.json"
    if not intervals_path.is_file() or not summary_path.is_file():
        raise ValueError("main selector final aggregate artifacts are missing")
    expected_intervals = intervals.to_csv(index=False)
    expected_summary = json.dumps(summary, ensure_ascii=False, indent=2) + "\n"
    if intervals_path.read_text() != expected_intervals:
        raise ValueError("main selector candidate_intervals.csv failed recomputation")
    if summary_path.read_text() != expected_summary:
        raise ValueError("main selector selector_summary.json failed recomputation")

    practical_selection_gate = summary.get("practical_selection_gate")
    if practical_selection_gate not in {
        "NO_SELECTION_VALUE",
        "AT_LEAST_ONE_CELL_NARROWS_LIBRARY",
    }:
        raise ValueError("main selector practical-selection gate is invalid")

    return {
        "selector_dir": str(selector_dir.resolve()),
        "exit_log": str(exit_log.resolve()),
        "checkpoint_count": len(ordered_paths),
        "run_config_sha256": file_sha256(config_path),
        "candidate_intervals_sha256": file_sha256(intervals_path),
        "selector_summary_sha256": file_sha256(summary_path),
        "exit_log_sha256": file_sha256(exit_log),
        "practical_selection_gate": practical_selection_gate,
    }


def crossfit_oracle_checkpoint_path(
    output_dir: Path, *, fold: int, center: str, drug_id: int
) -> Path:
    return checkpoint_path(output_dir / f"source_fold={fold}", center=center, drug_id=drug_id)


def crossfit_metric_checkpoint_path(
    output_dir: Path, *, fold: int, drug_id: int
) -> Path:
    return output_dir / f"source_fold={fold}" / "metrics" / f"drug={drug_id}.parquet"


def write_or_validate_crossfit_run_configuration(
    output_dir: Path, config: dict[str, object]
) -> None:
    """Bind nested cross-fit checkpoints to a frozen run configuration."""

    config_path = output_dir / "run_config.json"
    if not config_path.exists():
        orphan = next(
            iter(
                sorted(output_dir.glob("source_fold=*/oracle/center=*/drug=*.parquet"))
                + sorted(output_dir.glob("source_fold=*/metrics/drug=*.parquet"))
                + sorted(output_dir.glob("source_fold=*/**/drug=*.partial"))
            ),
            None,
        )
        if orphan is not None:
            raise ValueError(
                "cross-fit checkpoints exist without run_config.json; "
                "explicit migration required"
            )
    write_or_validate_run_configuration(output_dir, config)


def _write_metric_checkpoint(frame: pd.DataFrame, path: Path) -> None:
    if tuple(frame.columns) != CROSSFIT_METRIC_COLUMNS:
        raise ValueError("cross-fit metric frame does not satisfy the frozen schema")
    if path.exists():
        raise FileExistsError(f"checkpoint already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp.parquet")
    if temporary.exists():
        raise FileExistsError(f"unfinished checkpoint requires manual review: {temporary}")
    frame.to_parquet(temporary, index=False, compression="zstd")
    temporary.replace(path)


def validate_crossfit_metric_checkpoint(
    path: Path,
    *,
    episode: SourceCrossFitEpisode,
    candidate_ids: Sequence[str],
    n_rna: int,
    scores: np.ndarray,
) -> None:
    frame = pd.read_parquet(path)
    if tuple(frame.columns) != CROSSFIT_METRIC_COLUMNS:
        raise ValueError(f"cross-fit metric checkpoint schema mismatch: {path}")
    expected_rows = len(candidate_ids) if episode.evaluable else 0
    if len(frame) != expected_rows:
        raise ValueError(f"cross-fit metric checkpoint row count mismatch: {path}")
    if frame.empty:
        return
    if set(frame["fold"]) != {episode.fold} or set(frame["drug_id"]) != {
        episode.drug_id
    }:
        raise ValueError(f"cross-fit metric checkpoint identity mismatch: {path}")
    if set(frame["candidate"]) != set(candidate_ids):
        raise ValueError(f"cross-fit metric candidate coverage mismatch: {path}")
    if frame.duplicated(["fold", "drug_id", "candidate"]).any():
        raise ValueError(f"cross-fit metric checkpoint contains duplicates: {path}")
    expected_candidate_count = n_rna - len(episode.anchor_rnas)
    expected_positive_count = len(episode.heldout_novel_rnas)
    if expected_candidate_count <= 0 or expected_positive_count <= 0:
        raise ValueError(f"cross-fit metric episode is not evaluable: {path}")
    expected_counts = {
        "candidate_count": expected_candidate_count,
        "positive_count": expected_positive_count,
        "actual_k_10": min(10, expected_candidate_count),
        "actual_k_50": min(50, expected_candidate_count),
    }
    for column, expected in expected_counts.items():
        if set(frame[column]) != {expected}:
            raise ValueError(
                f"cross-fit metric checkpoint {column} mismatch: {path}"
            )
    numeric = frame[
        [
            "average_precision",
            "pairwise_auc",
            "recall_at_10",
            "recall_at_50",
            "actual_k_10",
            "actual_k_50",
        ]
    ]
    if not np.isfinite(numeric.to_numpy(dtype=np.float64)).all():
        raise ValueError(f"cross-fit metric checkpoint contains non-finite values: {path}")
    bounded = frame[
        ["average_precision", "pairwise_auc", "recall_at_10", "recall_at_50"]
    ].to_numpy(dtype=np.float64)
    if ((bounded < 0.0) | (bounded > 1.0)).any():
        raise ValueError(f"cross-fit metric checkpoint metric out of range: {path}")

    expected = evaluate_crossfit_episode_scores(
        episode,
        scores=scores,
        candidate_ids=candidate_ids,
    ).reindex(columns=CROSSFIT_METRIC_COLUMNS)
    observed = frame.sort_values("candidate", kind="mergesort").reset_index(drop=True)
    expected = expected.sort_values("candidate", kind="mergesort").reset_index(drop=True)
    try:
        pd.testing.assert_frame_equal(
            observed,
            expected,
            check_exact=False,
            rtol=0.0,
            atol=1e-12,
        )
    except AssertionError as error:
        raise ValueError(
            f"cross-fit metric checkpoint failed score/episode recomputation: {path}"
        ) from error


def evaluate_crossfit_drug_outputs(
    *,
    drug_id: int,
    score_matrix: np.ndarray,
    candidate_ids: tuple[str, ...],
    source_sets: dict[str, set[int]],
    episodes: tuple[SourceCrossFitEpisode, ...],
    n_rna: int,
    output_dir: Path,
) -> tuple[int, int]:
    """Create immutable oracle and metric checkpoints for all folds of one drug."""

    created = 0
    core_cache: dict[tuple[object, ...], pd.DataFrame] = {}
    tau_one_frame: pd.DataFrame | None = None
    for episode in episodes:
        metric_path = crossfit_metric_checkpoint_path(
            output_dir, fold=episode.fold, drug_id=drug_id
        )
        if metric_path.exists():
            validate_crossfit_metric_checkpoint(
                metric_path,
                episode=episode,
                candidate_ids=candidate_ids,
                n_rna=n_rna,
                scores=score_matrix,
            )
        else:
            metrics = evaluate_crossfit_episode_scores(
                episode, scores=score_matrix, candidate_ids=candidate_ids
            )
            metrics = metrics.reindex(columns=CROSSFIT_METRIC_COLUMNS)
            _write_metric_checkpoint(metrics, metric_path)
            created += 1

        anchors = {source: source_sets[source] for source in episode.anchor_sources}
        for center in CENTER_MODES:
            p0 = source_center_profile(
                anchors, n_rna_candidates=n_rna, mode=center
            )
            path = crossfit_oracle_checkpoint_path(
                output_dir,
                fold=episode.fold,
                center=center,
                drug_id=drug_id,
            )
            if path.exists():
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
                    p0=p0,
                )
                core_cache[tuple(p0)] = frame
                if tau_one_frame is None:
                    tau_one_frame = frame.loc[frame["tau"] == "1/1"].copy()
                continue
            cache_key = tuple(p0)
            if cache_key in core_cache:
                frame = core_cache[cache_key].copy()
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
                core_cache[cache_key] = frame
            frame["center"] = center
            frame = frame.reindex(columns=CHECKPOINT_COLUMNS)
            write_checkpoint(frame, path)
            created += 1
    return drug_id, created


def evaluate_crossfit_drugs(
    score_items: Iterable[tuple[int, np.ndarray]],
    *,
    candidate_ids: tuple[str, ...],
    source_sets: Mapping[int, dict[str, set[int]]],
    episodes_by_drug: Mapping[int, tuple[SourceCrossFitEpisode, ...]],
    n_rna: int,
    output_dir: Path,
    workers: int,
) -> set[int]:
    """Evaluate eligible drugs once each, optionally in parallel."""

    if workers <= 0:
        raise ValueError("workers must be a positive integer")

    def task(drug_id: int, score_matrix: np.ndarray) -> dict[str, object]:
        return {
            "drug_id": drug_id,
            "score_matrix": score_matrix,
            "candidate_ids": candidate_ids,
            "source_sets": source_sets[drug_id],
            "episodes": episodes_by_drug[drug_id],
            "n_rna": n_rna,
            "output_dir": output_dir,
        }

    eligible = (
        (drug_id, matrix)
        for drug_id, matrix in score_items
        if drug_id in episodes_by_drug
    )
    observed: set[int] = set()
    if workers == 1:
        for drug_id, matrix in eligible:
            observed.add(drug_id)
            completed, created = evaluate_crossfit_drug_outputs(**task(drug_id, matrix))
            print(
                f"[mc-tv-pu-crossfit] completed drug={completed} checkpoints={created}",
                flush=True,
            )
        return observed

    with ProcessPoolExecutor(max_workers=workers) as executor:
        pending = set()
        for drug_id, matrix in eligible:
            observed.add(drug_id)
            pending.add(
                executor.submit(evaluate_crossfit_drug_outputs, **task(drug_id, matrix))
            )
            if len(pending) >= 2 * workers:
                done, pending = wait(pending, return_when=FIRST_COMPLETED)
                for future in done:
                    completed, created = future.result()
                    print(
                        f"[mc-tv-pu-crossfit] completed drug={completed} "
                        f"checkpoints={created}",
                        flush=True,
                    )
        while pending:
            done, pending = wait(pending, return_when=FIRST_COMPLETED)
            for future in done:
                completed, created = future.result()
                print(
                    f"[mc-tv-pu-crossfit] completed drug={completed} checkpoints={created}",
                    flush=True,
                )
    return observed


def _assignment_digest(assignments: Mapping[str, int]) -> str:
    payload = json.dumps(dict(sorted(assignments.items())), separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


def crossfit_final_artifact_hashes(
    output_dir: Path,
    *,
    candidate_intervals_text: str,
    episode_metrics_text: str,
    inference_text: str,
    rule_ablation_inference_text: str,
) -> dict[str, str]:
    """Bind the terminal summary to its frozen config and final result tables."""

    run_config = output_dir / "run_config.json"
    assignments = output_dir / "source_fold_assignments.json"
    for path in (run_config, assignments):
        if not path.is_file():
            raise ValueError(f"required cross-fit provenance artifact is missing: {path}")
    table_texts = {
        "crossfit_candidate_intervals.csv": candidate_intervals_text,
        "crossfit_episode_metrics.csv": episode_metrics_text,
        "crossfit_inference.csv": inference_text,
        "crossfit_rule_ablation_inference.csv": rule_ablation_inference_text,
    }
    return {
        "run_config.json": file_sha256(run_config),
        "source_fold_assignments.json": file_sha256(assignments),
        **{
            name: hashlib.sha256(content.encode()).hexdigest()
            for name, content in table_texts.items()
        },
    }


def validate_crossfit_artifact_package(output_dir: Path) -> dict[str, object]:
    """Verify every terminal artifact against the hashes frozen in the summary."""

    summary_path = output_dir / "crossfit_validation_summary.json"
    if not summary_path.is_file():
        raise ValueError("cross-fit terminal summary is missing")
    unfinished = sorted(output_dir.rglob("*.tmp.parquet")) + sorted(
        output_dir.rglob("*.partial")
    )
    if unfinished:
        raise ValueError("cross-fit output contains unfinished files")
    summary = json.loads(summary_path.read_text())
    recorded = summary.get("artifact_sha256")
    if not isinstance(recorded, dict) or set(recorded) != set(
        CROSSFIT_FINAL_ARTIFACTS
    ):
        raise ValueError("cross-fit summary artifact hash coverage mismatch")
    for name in CROSSFIT_FINAL_ARTIFACTS:
        digest = recorded[name]
        if not isinstance(digest, str) or len(digest) != 64:
            raise ValueError(f"cross-fit summary artifact hash is invalid: {name}")
        path = output_dir / name
        if not path.is_file():
            raise ValueError(f"cross-fit terminal artifact is missing: {name}")
        if file_sha256(path) != digest:
            raise ValueError(f"cross-fit terminal artifact hash mismatch: {name}")
    return {
        "status": "PASS",
        "artifact_count": len(CROSSFIT_FINAL_ARTIFACTS),
        "summary_sha256": file_sha256(summary_path),
    }


def _run(args: argparse.Namespace) -> int:
    dataset_dir_binding = validate_dataset_dir_binding(args.root, args.dataset_dir)
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

    dataset = manifest["dataset"]
    split_seed = dataset.get("split_seed")
    dataset_name = dataset.get("name")
    if isinstance(split_seed, bool) or not isinstance(split_seed, int):
        raise ValueError("manifest dataset.split_seed must be an integer")
    if not isinstance(dataset_name, str) or not dataset_name:
        raise ValueError("manifest dataset.name must be a nonempty string")
    target_drug_oof_audit = validate_target_drug_oof_assignment(
        args.score_dir,
        reference_model=candidate_ids[0],
        pair_matrix=pair_matrix,
        drug_scaffolds=load_drug_scaffolds(args.dataset_dir, dataset_name),
        fold_count=fold_count,
        split_seed=split_seed,
        score_input_audit=audit,
    )
    isolation_scope = source_isolation_scope_metadata(target_drug_oof_audit)

    selector_config = run_configuration(
        root=args.root,
        score_dir=args.score_dir,
        pair_matrix_path=args.pair_matrix,
        manifest_path=args.manifest,
        source_table=args.source_table,
    )
    selector_config["candidate_ids"] = list(candidate_ids)
    selector_config["score_input_audit"] = audit.to_dict()
    main_selector_gate = validate_main_selector_completion(
        args.main_selector_dir,
        exit_log=args.main_selector_log,
        expected_config=selector_config,
        candidate_ids=candidate_ids,
        active_drugs=active_drugs,
    )

    pair_sources = restrict_pair_sources_to_positive_matrix(
        build_pair_sources(args.source_table, args.dataset_dir), pair_matrix
    )
    source_sets = source_sets_by_drug(pair_sources)
    assignments, episodes_by_fold = build_source_crossfit_episodes(
        pair_sources,
        fold_count=CROSS_FIT_FOLD_COUNT,
        seed=CROSS_FIT_SEED,
    )
    coverage = source_crossfit_coverage_summary(episodes_by_fold)
    episodes_by_drug_lists: dict[int, list[SourceCrossFitEpisode]] = defaultdict(list)
    for episodes in episodes_by_fold.values():
        for episode in episodes:
            episodes_by_drug_lists[episode.drug_id].append(episode)
    episodes_by_drug = {
        drug: tuple(sorted(episodes, key=lambda episode: episode.fold))
        for drug, episodes in episodes_by_drug_lists.items()
    }
    expected_ablation_family_size = (
        len(CENTER_MODES) * len(TAU_GRID) * len(SELECTION_RULE_ABLATIONS)
    )

    config = dict(selector_config)
    config.update(
        {
            "analysis": "selector_anchor_source_crossfit_validation",
            "implementation_source_sha256": crossfit_implementation_hashes(),
            "dataset_dir_binding": dataset_dir_binding,
            "isolation_scope": isolation_scope,
            "main_selector_gate": main_selector_gate,
            "source_fold_count": CROSS_FIT_FOLD_COUNT,
            "source_fold_seed": CROSS_FIT_SEED,
            "source_assignment_sha256": _assignment_digest(assignments),
            "candidate_ids": list(candidate_ids),
            "score_input_audit": audit.to_dict(),
            "coverage": coverage,
            "bootstrap_iterations": BOOTSTRAP_ITERATIONS,
            "bootstrap_seed": BOOTSTRAP_SEED,
            "evaluation_metrics": list(METRICS),
            "primary_success_metric": "average_precision",
            "objective_aligned_secondary_metric": "pairwise_auc",
            "primary_baselines": list(PRIMARY_BASELINES),
            "rule_ablation": {
                "reference_rule": COMPATIBLE_SELECTION_RULE,
                "comparison_rules": list(SELECTION_RULE_ABLATIONS),
                "selection_set_performance": "worst_member_for_each_rule",
                "ap_simultaneous_family_size": expected_ablation_family_size,
                "affects_primary_success_gate": False,
            },
        }
    )
    write_or_validate_crossfit_run_configuration(args.output_dir, config)
    write_or_validate_text(
        args.output_dir / "source_fold_assignments.json",
        json.dumps(dict(sorted(assignments.items())), indent=2) + "\n",
    )

    observed = evaluate_crossfit_drugs(
        iter_fold_score_matrices(
            args.score_dir,
            candidate_ids=candidate_ids,
            fold_count=fold_count,
            n_rna_candidates=n_rna,
        ),
        candidate_ids=tuple(candidate_ids),
        source_sets=source_sets,
        episodes_by_drug=episodes_by_drug,
        n_rna=n_rna,
        output_dir=args.output_dir,
        workers=args.workers,
    )
    if observed != set(episodes_by_drug):
        raise ValueError("score partitions did not yield every cross-fit eligible drug")

    selector_by_fold: dict[int, dict[str, object]] = {}
    selector_cells_by_fold: dict[int, Sequence[Mapping[str, object]]] = {}
    interval_frames: list[pd.DataFrame] = []
    metric_frames: list[pd.DataFrame] = []
    for source_fold, episodes in sorted(episodes_by_fold.items()):
        oracle_paths = [
            crossfit_oracle_checkpoint_path(
                args.output_dir,
                fold=source_fold,
                center=center,
                drug_id=episode.drug_id,
            )
            for episode in episodes
            for center in CENTER_MODES
        ]
        intervals, selector = aggregate_selector_intervals(
            (pd.read_parquet(path) for path in oracle_paths),
            candidate_ids=candidate_ids,
        )
        intervals.insert(0, "source_fold", source_fold)
        interval_frames.append(intervals)
        selector_by_fold[source_fold] = selector
        selector_cells_by_fold[source_fold] = selector["cells"]
        metric_frames.extend(
            pd.read_parquet(
                crossfit_metric_checkpoint_path(
                    args.output_dir, fold=source_fold, drug_id=episode.drug_id
                )
            )
            for episode in episodes
        )

    episode_metrics = pd.concat(metric_frames, ignore_index=True)
    inference, bootstrap_config = bootstrap_crossfit_set_valued_performance(
        episode_metrics,
        selector_cells_by_fold=selector_cells_by_fold,
        candidate_ids=candidate_ids,
    )
    rule_ablation_inference, rule_ablation_bootstrap = (
        bootstrap_crossfit_rule_ablation_performance(
            episode_metrics,
            selector_cells_by_fold=selector_cells_by_fold,
            candidate_ids=candidate_ids,
        )
    )
    if (
        rule_ablation_bootstrap["ap_simultaneous_family_size"]
        != expected_ablation_family_size
    ):
        raise ValueError("rule-ablation inference does not cover the frozen 48 cells")
    flattened_cells = [
        {"source_fold": fold, **cell}
        for fold, cells in selector_cells_by_fold.items()
        for cell in cells
    ]
    success = evaluate_success_gate(
        inference,
        selector_cells=flattened_cells,
        coverage=coverage,
        candidate_count=len(candidate_ids),
        main_selector_narrowed=(
            main_selector_gate["practical_selection_gate"]
            == "AT_LEAST_ONE_CELL_NARROWS_LIBRARY"
        ),
    )
    rule_ablation_claim_routing = evaluate_rule_ablation_claim_routing(
        rule_ablation_inference,
        primary_success=success,
    )
    candidate_intervals_text = pd.concat(interval_frames, ignore_index=True).to_csv(
        index=False
    )
    episode_metrics_text = episode_metrics.to_csv(index=False)
    inference_text = inference.to_csv(index=False)
    rule_ablation_inference_text = rule_ablation_inference.to_csv(index=False)
    artifact_sha256 = crossfit_final_artifact_hashes(
        args.output_dir,
        candidate_intervals_text=candidate_intervals_text,
        episode_metrics_text=episode_metrics_text,
        inference_text=inference_text,
        rule_ablation_inference_text=rule_ablation_inference_text,
    )
    summary = {
        "isolation_scope": isolation_scope,
        "evaluation_metrics": list(METRICS),
        "primary_success_metric": "average_precision",
        "objective_aligned_secondary_metric": "pairwise_auc",
        "coverage": coverage,
        "selector_by_fold": {str(fold): value for fold, value in selector_by_fold.items()},
        "bootstrap": bootstrap_config,
        "rule_ablation_bootstrap": rule_ablation_bootstrap,
        "rule_ablation_claim_routing": rule_ablation_claim_routing,
        "success_gate": success,
        "artifact_sha256": artifact_sha256,
    }
    write_or_validate_text(
        args.output_dir / "crossfit_candidate_intervals.csv",
        candidate_intervals_text,
    )
    write_or_validate_text(
        args.output_dir / "crossfit_episode_metrics.csv",
        episode_metrics_text,
    )
    write_or_validate_text(
        args.output_dir / "crossfit_inference.csv", inference_text
    )
    write_or_validate_text(
        args.output_dir / "crossfit_rule_ablation_inference.csv",
        rule_ablation_inference_text,
    )
    write_or_validate_text(
        args.output_dir / "crossfit_validation_summary.json",
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
    )
    artifact_validation = validate_crossfit_artifact_package(args.output_dir)
    print(
        "[mc-tv-pu-crossfit] artifact_package="
        + json.dumps(artifact_validation, ensure_ascii=False, sort_keys=True),
        flush=True,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--score-dir", type=Path, required=True)
    parser.add_argument("--pair-matrix", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--source-table", type=Path, required=True)
    parser.add_argument("--main-selector-dir", type=Path, required=True)
    parser.add_argument("--main-selector-log", type=Path, required=True)
    parser.add_argument("--dataset-dir", type=Path, default=Path("Dataset"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()

    with exclusive_output_lock(args.output_dir):
        return _run(args)


if __name__ == "__main__":
    raise SystemExit(main())
