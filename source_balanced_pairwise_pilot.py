from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from generalization_audit import (
    DatasetBundle,
    PriorPreservingPairwiseResidualEstimator,
    load_bundle,
    load_drug_fingerprints,
    load_drug_scaffolds,
    scaffold_cv_split,
)
from scaffold_cv_confirm import paired_drug_metric_bootstrap


def build_pair_sources(
    table_path: Path, dataset_dir: Path
) -> dict[tuple[int, int], set[str]]:
    table = pd.read_csv(table_path, sep="\t", low_memory=False)
    rna = pd.read_excel(dataset_dir / "miRNA_with_sequence.xlsx")
    drug = pd.read_csv(dataset_dir / "mi_drug_with_SMILES.csv")
    rna_name = {str(v).strip().upper(): i for i, v in enumerate(rna["ncRNA_Name"])}
    rna_mirbase = {
        str(v).strip().upper(): i
        for i, v in enumerate(rna["miRBase_ID"])
        if pd.notna(v)
    }
    drug_name = {str(v).strip().upper(): i for i, v in enumerate(drug["Drug_Name"])}
    drug_cid = {int(v): i for i, v in enumerate(drug["CID"]) if pd.notna(v)}
    sources: dict[tuple[int, int], set[str]] = defaultdict(set)
    for row in table.itertuples(index=False):
        rna_id = None
        if pd.notna(row.miRBase_ID):
            rna_id = rna_mirbase.get(str(row.miRBase_ID).strip().upper())
        if rna_id is None:
            rna_id = rna_name.get(str(row.ncRNA_Name).strip().upper())
        drug_id = None
        if pd.notna(row.CID):
            try:
                drug_id = drug_cid.get(int(float(row.CID)))
            except (TypeError, ValueError):
                pass
        if drug_id is None:
            drug_id = drug_name.get(str(row.Drug_Name).strip().upper())
        if rna_id is None or drug_id is None or pd.isna(row.PMID):
            continue
        sources[(rna_id, drug_id)].add(str(row.PMID).strip())
    return dict(sources)


def fold_local_source_weights(
    positive_pairs: np.ndarray,
    pair_sources: dict[tuple[int, int], set[str]],
    cap_multiplier: float = 10.0,
) -> np.ndarray:
    source_counts: dict[str, int] = defaultdict(int)
    for pair in map(tuple, positive_pairs.tolist()):
        for source in pair_sources.get(pair, set()):
            source_counts[source] += 1
    raw = np.full(len(positive_pairs), np.nan, dtype=np.float64)
    for index, pair in enumerate(map(tuple, positive_pairs.tolist())):
        sources = pair_sources.get(pair, set())
        if sources:
            raw[index] = sum(1.0 / source_counts[source] for source in sources)
    mapped = np.isfinite(raw)
    if not mapped.any():
        return np.ones(len(positive_pairs), dtype=np.float32)
    raw[~mapped] = np.median(raw[mapped])
    raw = np.minimum(raw, raw.mean() * cap_multiplier)
    raw /= raw.mean()
    return raw.astype(np.float32)


class SourceBalancedPairwiseEstimator(PriorPreservingPairwiseResidualEstimator):
    def __init__(self, *args, pair_sources: dict[tuple[int, int], set[str]], **kwargs):
        super().__init__(*args, **kwargs)
        self.pair_sources = pair_sources

    def _training_pair_weights(
        self, train_positive_matrix: np.ndarray, positive_pairs: np.ndarray
    ) -> np.ndarray:
        return fold_local_source_weights(positive_pairs, self.pair_sources)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", type=Path, default=Path("Dataset"))
    parser.add_argument("--source-table", type=Path, default=Path("/Users/jackie/Downloads/DT_Curated.txt"))
    parser.add_argument("--split-seed", type=int, default=20261019)
    parser.add_argument("--output-dir", type=Path, default=Path("results/source-balanced-pairwise-pilot-20261019"))
    args = parser.parse_args()
    source = load_bundle(args.dataset_dir, "Mi")
    bundle = DatasetBundle(
        rna_features=source.rna_features,
        drug_features=load_drug_fingerprints(args.dataset_dir, "Mi"),
        pair=source.pair,
        pair_new=source.pair_new,
    )
    scaffolds = load_drug_scaffolds(args.dataset_dir, "Mi")
    pair_sources = build_pair_sources(args.source_table, args.dataset_dir)
    rows = []
    for fold in range(5):
        print(f"[source-balanced] fold={fold + 1}/5", flush=True)
        split = scaffold_cv_split(source, scaffolds, fold=fold, n_folds=5, seed=args.split_seed)
        estimators = {
            "ecfp_pairwise": PriorPreservingPairwiseResidualEstimator(
                seed=args.split_seed + 1000 + fold, max_epochs=120
            ),
            "source_balanced_pairwise": SourceBalancedPairwiseEstimator(
                seed=args.split_seed + 1000 + fold,
                max_epochs=120,
                pair_sources=pair_sources,
            ),
        }
        test_drugs = np.unique(split.test.pairs[:, 1])
        full_pairs = np.column_stack(
            [
                np.tile(np.arange(bundle.pair.shape[0], dtype=np.int64), len(test_drugs)),
                np.repeat(test_drugs, bundle.pair.shape[0]),
            ]
        )
        labels = bundle.pair[full_pairs[:, 0], full_pairs[:, 1]].astype(np.float32)
        predictions = {}
        for name, estimator in estimators.items():
            estimator.fit(bundle, split)
            predictions[name] = estimator.predict_proba(bundle, full_pairs)
            if name == "ecfp_pairwise":
                predictions["rna_prior_only"] = estimator.predict_prior_proba(full_pairs)
        for model, probability in predictions.items():
            for drug_id in test_drugs:
                include = full_pairs[:, 1] == drug_id
                rows.append(
                    {
                        "fold": fold,
                        "model": model,
                        "drug_id": int(drug_id),
                        "positive_count": int(labels[include].sum()),
                        "average_precision": float(
                            average_precision_score(labels[include], probability[include])
                        ),
                    }
                )
    metrics = pd.DataFrame(rows)
    summary = metrics.groupby("model")["average_precision"].agg(["mean", "std"])
    candidate = "source_balanced_pairwise"
    intervals = {}
    deltas = {}
    for baseline in ["rna_prior_only", "ecfp_pairwise"]:
        intervals[baseline] = paired_drug_metric_bootstrap(
            metrics, candidate, baseline, "average_precision", 5000, args.split_seed
        )
        means = summary["mean"]
        deltas[baseline] = float(means[candidate] - means[baseline])
    fold_mean = metrics.groupby(["fold", "model"])["average_precision"].mean().unstack()
    fold_non_decrease = int((fold_mean[candidate] >= fold_mean["ecfp_pairwise"]).sum())
    config = {
        "pair_source_count": len(pair_sources),
        "source_weight_scope": "training positive pairs only",
        "weight_cap_multiplier": 10.0,
        "candidate_vs_prior_delta": deltas["rna_prior_only"],
        "candidate_vs_prior_bootstrap_95": dict(zip(["low", "median", "high"], intervals["rna_prior_only"])),
        "candidate_vs_ecfp_pairwise_delta": deltas["ecfp_pairwise"],
        "candidate_vs_ecfp_pairwise_bootstrap_95": dict(zip(["low", "median", "high"], intervals["ecfp_pairwise"])),
        "candidate_vs_ecfp_fold_non_decrease_count": fold_non_decrease,
        "passed_preregistered_gate": bool(
            deltas["rna_prior_only"] >= 0.01
            and intervals["rna_prior_only"][0] > 0
            and intervals["ecfp_pairwise"][0] > 0
            and fold_non_decrease >= 4
        ),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(args.output_dir / "drug_metrics.csv", index=False)
    summary.reset_index().to_csv(args.output_dir / "summary.csv", index=False)
    (args.output_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    print(summary.to_string())
    print(json.dumps(config, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
