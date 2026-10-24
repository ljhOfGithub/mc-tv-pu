"""Read-only validation of the frozen MC-TV-PU publication package."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from mc_tv_pu_source_crossfit_runner import (
    validate_crossfit_artifact_package,
    validate_main_selector_completion,
)


class ReleaseValidationError(ValueError):
    """Raised when the frozen release package is incomplete or inconsistent."""


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, object]:
    if not path.is_file():
        raise ReleaseValidationError(f"required JSON file is missing: {path}")
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ReleaseValidationError(f"expected a JSON object: {path}")
    return value


def _last_nonempty_line(path: Path) -> str:
    if not path.is_file():
        raise ReleaseValidationError(f"required log is missing: {path}")
    lines = [line.strip() for line in path.read_text().splitlines() if line.strip()]
    if not lines:
        raise ReleaseValidationError(f"required log is empty: {path}")
    return lines[-1]


def _require_equal(actual: object, expected: object, label: str) -> None:
    if actual != expected:
        raise ReleaseValidationError(
            f"{label} mismatch: expected={expected!r} actual={actual!r}"
        )


def validate_implementation_hashes(
    repo_root: Path, expected_hashes: Mapping[str, object]
) -> dict[str, str]:
    """Validate the complete release-level Python source closure."""

    observed: dict[str, str] = {}
    for relative_path, expected_digest in sorted(expected_hashes.items()):
        if not isinstance(relative_path, str) or not isinstance(expected_digest, str):
            raise ReleaseValidationError("implementation hash map must be string-to-string")
        path = repo_root / relative_path
        if not path.is_file():
            raise ReleaseValidationError(
                f"release implementation source is missing: {relative_path}"
            )
        observed_digest = file_sha256(path)
        _require_equal(
            observed_digest,
            expected_digest,
            f"implementation source sha256 for {relative_path}",
        )
        observed[relative_path] = observed_digest
    return observed


def validate_release_manifest(
    repo_root: Path, manifest_path: Path, checksum_path: Path
) -> dict[str, object]:
    """Validate every release file listed in the manifest."""

    if not manifest_path.is_file() or not checksum_path.is_file():
        raise ReleaseValidationError("release manifest or checksum file is missing")
    manifest_entries = [
        line.strip()
        for line in manifest_path.read_text().splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    if len(manifest_entries) != len(set(manifest_entries)):
        raise ReleaseValidationError("release manifest contains duplicate paths")
    if any(Path(entry).is_absolute() or ".." in Path(entry).parts for entry in manifest_entries):
        raise ReleaseValidationError("release manifest paths must stay inside the repository")

    recorded: dict[str, str] = {}
    for line in checksum_path.read_text().splitlines():
        if not line.strip():
            continue
        digest, separator, relative_path = line.partition("  ")
        if not separator or len(digest) != 64 or relative_path in recorded:
            raise ReleaseValidationError("release checksum file has an invalid line")
        recorded[relative_path] = digest
    _require_equal(set(recorded), set(manifest_entries), "release checksum coverage")

    for relative_path in manifest_entries:
        path = repo_root / relative_path
        if not path.is_file():
            raise ReleaseValidationError(f"release file is missing: {relative_path}")
        _require_equal(
            file_sha256(path),
            recorded[relative_path],
            f"release file sha256 for {relative_path}",
        )
    return {
        "status": "PASS",
        "file_count": len(manifest_entries),
        "manifest_sha256": file_sha256(manifest_path),
    }


def validate_claim_boundary(
    summary: Mapping[str, object], expected: Mapping[str, object]
) -> dict[str, object]:
    """Require the exact frozen decision and its permitted claim boundary."""

    success_gate = summary.get("success_gate")
    isolation_scope = summary.get("isolation_scope")
    coverage = summary.get("coverage")
    if not isinstance(success_gate, dict):
        raise ReleaseValidationError("cross-fit success_gate is missing")
    if not isinstance(isolation_scope, dict):
        raise ReleaseValidationError("cross-fit isolation_scope is missing")
    if not isinstance(coverage, dict):
        raise ReleaseValidationError("cross-fit coverage is missing")

    checks = {
        "success_decision": success_gate.get("decision"),
        "selector_narrowed_at_least_one_cell": success_gate.get(
            "selector_narrowed_at_least_one_cell"
        ),
        "publication_coverage_passed": coverage.get("publication_coverage_passed"),
        "source_isolation_scope": isolation_scope.get("source_isolation_scope"),
        "target_drug_oof": isolation_scope.get("target_drug_oof"),
        "selector_anchor_source_disjoint": isolation_scope.get(
            "selector_anchor_source_disjoint"
        ),
        "base_model_source_disjoint": isolation_scope.get("base_model_source_disjoint"),
    }
    for label, actual in checks.items():
        _require_equal(actual, expected.get(label), f"claim boundary {label}")
    return checks


def validate_frozen_release(
    *,
    repo_root: Path,
    result_root: Path,
    pair_matrix_path: Path,
    spec_path: Path,
) -> dict[str, object]:
    """Run all read-only checks for the versioned frozen result package."""

    spec = _load_json(spec_path)
    selector_spec = spec.get("selector")
    crossfit_spec = spec.get("crossfit")
    implementation_hashes = spec.get("implementation_source_sha256")
    if not isinstance(selector_spec, dict) or not isinstance(crossfit_spec, dict):
        raise ReleaseValidationError("frozen result spec lacks selector/crossfit objects")
    if not isinstance(implementation_hashes, dict):
        raise ReleaseValidationError("frozen result spec lacks implementation hashes")

    manifest_path = repo_root / "release/mc-tv-pu/MANIFEST.txt"
    checksum_path = repo_root / "release/mc-tv-pu/MANIFEST.sha256"
    manifest_gate = validate_release_manifest(repo_root, manifest_path, checksum_path)
    source_gate = validate_implementation_hashes(repo_root, implementation_hashes)

    selector_dir = result_root / str(selector_spec["directory"])
    selector_log = result_root / str(selector_spec["log"])
    crossfit_dir = result_root / str(crossfit_spec["directory"])
    crossfit_log = result_root / str(crossfit_spec["log"])
    selector_config = _load_json(selector_dir / "run_config.json")
    crossfit_config = _load_json(crossfit_dir / "run_config.json")
    summary_path = crossfit_dir / "crossfit_validation_summary.json"
    summary = _load_json(summary_path)

    if not pair_matrix_path.is_file():
        raise ReleaseValidationError(f"pair matrix is missing: {pair_matrix_path}")
    pair_matrix_digest = file_sha256(pair_matrix_path)
    _require_equal(pair_matrix_digest, spec.get("pair_matrix_sha256"), "pair matrix sha256")
    _require_equal(
        selector_config.get("pair_matrix_sha256"),
        pair_matrix_digest,
        "selector pair matrix binding",
    )
    _require_equal(
        crossfit_config.get("pair_matrix_sha256"),
        pair_matrix_digest,
        "cross-fit pair matrix binding",
    )

    pair_matrix = np.load(pair_matrix_path, allow_pickle=False)
    if pair_matrix.ndim != 2:
        raise ReleaseValidationError("pair matrix must be two-dimensional")
    active_drugs: Sequence[int] = tuple(
        np.flatnonzero((pair_matrix > 0).any(axis=0)).tolist()
    )
    candidate_ids = selector_config.get("candidate_ids")
    if not isinstance(candidate_ids, list) or not all(
        isinstance(value, str) for value in candidate_ids
    ):
        raise ReleaseValidationError("selector candidate_ids are invalid")

    main_gate = validate_main_selector_completion(
        selector_dir,
        exit_log=selector_log,
        expected_config=selector_config,
        candidate_ids=candidate_ids,
        active_drugs=active_drugs,
    )
    for key in (
        "checkpoint_count",
        "run_config_sha256",
        "candidate_intervals_sha256",
        "selector_summary_sha256",
        "exit_log_sha256",
        "practical_selection_gate",
    ):
        _require_equal(main_gate.get(key), selector_spec.get(key), f"selector {key}")
        embedded_gate = crossfit_config.get("main_selector_gate")
        if not isinstance(embedded_gate, dict):
            raise ReleaseValidationError("cross-fit config lacks main_selector_gate")
        _require_equal(
            embedded_gate.get(key), main_gate.get(key), f"cross-fit selector binding {key}"
        )

    _require_equal(_last_nonempty_line(selector_log), "EXIT_STATUS=0", "selector exit")
    _require_equal(_last_nonempty_line(crossfit_log), "EXIT_STATUS=0", "cross-fit exit")
    crossfit_gate = validate_crossfit_artifact_package(crossfit_dir)
    _require_equal(
        crossfit_gate.get("summary_sha256"),
        crossfit_spec.get("summary_sha256"),
        "cross-fit summary sha256",
    )
    _require_equal(
        crossfit_gate.get("artifact_count"),
        crossfit_spec.get("artifact_count"),
        "cross-fit terminal artifact count",
    )

    oracle_count = len(
        list(crossfit_dir.glob("source_fold=*/oracle/center=*/drug=*.parquet"))
    )
    metric_count = len(list(crossfit_dir.glob("source_fold=*/metrics/drug=*.parquet")))
    _require_equal(
        oracle_count, crossfit_spec.get("oracle_checkpoint_count"), "cross-fit oracle count"
    )
    _require_equal(
        metric_count, crossfit_spec.get("metric_checkpoint_count"), "cross-fit metric count"
    )
    unfinished = sorted(result_root.rglob("*.tmp.parquet")) + sorted(
        result_root.rglob("*.partial")
    )
    if unfinished:
        raise ReleaseValidationError(
            "result package contains unfinished files: "
            + ", ".join(str(path) for path in unfinished[:5])
        )

    claim_boundary = validate_claim_boundary(summary, crossfit_spec)
    selector_report = {
        key: main_gate[key]
        for key in (
            "checkpoint_count",
            "run_config_sha256",
            "candidate_intervals_sha256",
            "selector_summary_sha256",
            "exit_log_sha256",
            "practical_selection_gate",
        )
    }
    return {
        "status": "PASS",
        "validation_scope": "frozen_result_integrity_not_model_retraining",
        "release_manifest": manifest_gate,
        "implementation_source_count": len(source_gate),
        "pair_matrix_sha256": pair_matrix_digest,
        "selector_gate": selector_report,
        "crossfit_gate": {
            **crossfit_gate,
            "oracle_checkpoint_count": oracle_count,
            "metric_checkpoint_count": metric_count,
            "exit_log_last_nonempty_line": "EXIT_STATUS=0",
        },
        "claim_boundary": claim_boundary,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate the frozen MC-TV-PU result package without recomputation."
    )
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument("--pair-matrix", type=Path, required=True)
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parent,
    )
    parser.add_argument(
        "--spec",
        type=Path,
        default=None,
        help="Defaults to release/mc-tv-pu/FROZEN_RESULT.json under repo root.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    repo_root = args.repo_root.resolve()
    spec_path = (
        args.spec.resolve()
        if args.spec is not None
        else repo_root / "release/mc-tv-pu/FROZEN_RESULT.json"
    )
    try:
        report = validate_frozen_release(
            repo_root=repo_root,
            result_root=args.result_root.resolve(),
            pair_matrix_path=args.pair_matrix.resolve(),
            spec_path=spec_path,
        )
    except (OSError, ValueError, json.JSONDecodeError) as error:
        print(
            json.dumps(
                {"status": "FAIL", "error": str(error)},
                ensure_ascii=False,
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        return 1
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
