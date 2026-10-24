import hashlib
from pathlib import Path

import pytest

from mc_tv_pu_release_validate import (
    ReleaseValidationError,
    validate_claim_boundary,
    validate_implementation_hashes,
    validate_release_manifest,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_validate_implementation_hashes_checks_complete_map(tmp_path: Path) -> None:
    source = tmp_path / "method.py"
    source.write_text("VALUE = 1\n")

    observed = validate_implementation_hashes(
        tmp_path, {"method.py": _sha256(source)}
    )

    assert observed == {"method.py": _sha256(source)}


def test_validate_release_manifest_rejects_drift(tmp_path: Path) -> None:
    artifact = tmp_path / "artifact.txt"
    artifact.write_text("frozen\n")
    manifest = tmp_path / "MANIFEST.txt"
    manifest.write_text("artifact.txt\n")
    checksums = tmp_path / "MANIFEST.sha256"
    checksums.write_text(f"{_sha256(artifact)}  artifact.txt\n")
    assert validate_release_manifest(tmp_path, manifest, checksums)["status"] == "PASS"

    artifact.write_text("drifted\n")
    with pytest.raises(ReleaseValidationError, match="release file sha256"):
        validate_release_manifest(tmp_path, manifest, checksums)


def test_validate_claim_boundary_preserves_negative_result() -> None:
    summary = {
        "success_gate": {
            "decision": "FAIL_RETAIN_AS_BOUNDARY_RESULT",
            "selector_narrowed_at_least_one_cell": True,
        },
        "coverage": {"publication_coverage_passed": True},
        "isolation_scope": {
            "source_isolation_scope": "selector_anchors_only",
            "target_drug_oof": True,
            "selector_anchor_source_disjoint": True,
            "base_model_source_disjoint": False,
        },
    }
    expected = {
        "success_decision": "FAIL_RETAIN_AS_BOUNDARY_RESULT",
        "selector_narrowed_at_least_one_cell": True,
        "publication_coverage_passed": True,
        "source_isolation_scope": "selector_anchors_only",
        "target_drug_oof": True,
        "selector_anchor_source_disjoint": True,
        "base_model_source_disjoint": False,
    }

    assert validate_claim_boundary(summary, expected) == expected

    summary["isolation_scope"]["base_model_source_disjoint"] = True
    with pytest.raises(ReleaseValidationError, match="base_model_source_disjoint"):
        validate_claim_boundary(summary, expected)
