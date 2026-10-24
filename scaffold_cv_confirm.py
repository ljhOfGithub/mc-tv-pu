import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from generalization_audit import (
    DatasetBundle,
    PropensityResidualEstimator,
    PriorOffsetStructuralResidualEstimator,
    SemanticEstimator,
    assert_split_integrity,
    load_bundle,
    load_drug_fingerprints,
    load_drug_scaffolds,
    metrics,
    scaffold_cv_split,
)
from mc_tv_pu_score_input_gate import (
    export_frozen_score_partitions,
    validated_candidate_ids_for_export,
)


def drug_cluster_bootstrap_delta(
    labels: np.ndarray,
    candidate: np.ndarray,
    baseline: np.ndarray,
    drug_ids: np.ndarray,
    iterations: int,
    seed: int,
) -> tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    unique_drugs = np.unique(drug_ids)
    index_by_drug = {drug: np.flatnonzero(drug_ids == drug) for drug in unique_drugs}
    deltas = []
    for _ in range(iterations):
        sampled_drugs = rng.choice(unique_drugs, size=len(unique_drugs), replace=True)
        indices = np.concatenate([index_by_drug[drug] for drug in sampled_drugs])
        deltas.append(
            average_precision_score(labels[indices], candidate[indices])
            - average_precision_score(labels[indices], baseline[indices])
        )
    low, median, high = np.quantile(deltas, [0.025, 0.5, 0.975])
    return float(low), float(median), float(high)


def run_fold(
    bundle: DatasetBundle,
    fingerprints: np.ndarray,
    scaffolds: np.ndarray,
    fold: int,
    n_folds: int,
    split_seed: int,
    max_epochs: int,
    score_export_dir: Path | None = None,
    score_export_models: tuple[str, ...] = (),
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    split = scaffold_cv_split(
        bundle, scaffolds, fold=fold, n_folds=n_folds, seed=split_seed
    )
    assert_split_integrity(split, cold_axis=1)
    model_seed = split_seed + 1000 + fold
    semantic = SemanticEstimator(seed=model_seed, max_epochs=max_epochs).fit(bundle, split)
    semantic_test = semantic.predict_proba(bundle, split.test.pairs)

    ecfp_bundle = DatasetBundle(
        rna_features=bundle.rna_features,
        drug_features=fingerprints,
        pair=bundle.pair,
        pair_new=bundle.pair_new,
    )
    ecfp = SemanticEstimator(seed=model_seed, max_epochs=max_epochs).fit(ecfp_bundle, split)
    ecfp_test = ecfp.predict_proba(ecfp_bundle, split.test.pairs)
    propensity = PropensityResidualEstimator(seed=model_seed, max_epochs=max_epochs).fit(
        bundle, split
    )
    propensity_test = propensity.predict_proba(bundle, split.test.pairs)
    ecfp_propensity = PropensityResidualEstimator(
        seed=model_seed, max_epochs=max_epochs
    ).fit(ecfp_bundle, split)
    ecfp_propensity_test = ecfp_propensity.predict_proba(ecfp_bundle, split.test.pairs)
    prior_offset = PriorOffsetStructuralResidualEstimator(
        seed=model_seed, max_epochs=max_epochs
    ).fit(ecfp_bundle, split)
    prior_offset_test = prior_offset.predict_proba(ecfp_bundle, split.test.pairs)
    prior_only_test = prior_offset.predict_prior_proba(split.test.pairs)
    prior_offset_semantic = PriorOffsetStructuralResidualEstimator(
        seed=model_seed, max_epochs=max_epochs
    ).fit(bundle, split)
    prior_offset_semantic_test = prior_offset_semantic.predict_proba(
        bundle, split.test.pairs
    )
    prior_offset_no_loo = PriorOffsetStructuralResidualEstimator(
        seed=model_seed, max_epochs=max_epochs, leave_one_out=False
    ).fit(ecfp_bundle, split)
    prior_offset_no_loo_test = prior_offset_no_loo.predict_proba(
        ecfp_bundle, split.test.pairs
    )

    train_degree = split.train_positive_matrix.sum(axis=0)
    active_degree = train_degree[train_degree > 0]
    top_k = min(10, len(active_degree))
    concentration = float(np.sort(active_degree)[-top_k:].sum() / active_degree.sum())
    predictions = {
        "semantic": semantic_test,
        "semantic_ecfp": ecfp_test,
        "propensity_residual": propensity_test,
        "ecfp_propensity_joint": ecfp_propensity_test,
        "rna_prior_only": prior_only_test,
        "prior_offset_structural_residual": prior_offset_test,
        "prior_offset_semantic_residual": prior_offset_semantic_test,
        "prior_offset_no_loo": prior_offset_no_loo_test,
        "concentration_adaptive_router": (
            concentration * ecfp_test + (1.0 - concentration) * propensity_test
        ),
    }

    test_drugs = np.unique(split.test.pairs[:, 1])
    full_pairs = np.column_stack(
        [
            np.tile(np.arange(bundle.pair.shape[0], dtype=np.int64), len(test_drugs)),
            np.repeat(test_drugs, bundle.pair.shape[0]),
        ]
    )
    full_labels = bundle.pair[full_pairs[:, 0], full_pairs[:, 1]].astype(np.float32)
    full_semantic = semantic.predict_proba(bundle, full_pairs)
    full_ecfp = ecfp.predict_proba(ecfp_bundle, full_pairs)
    full_propensity = propensity.predict_proba(bundle, full_pairs)
    full_ecfp_propensity = ecfp_propensity.predict_proba(ecfp_bundle, full_pairs)
    full_prior_offset = prior_offset.predict_proba(ecfp_bundle, full_pairs)
    full_prior_only = prior_offset.predict_prior_proba(full_pairs)
    full_prior_offset_semantic = prior_offset_semantic.predict_proba(bundle, full_pairs)
    full_prior_offset_no_loo = prior_offset_no_loo.predict_proba(ecfp_bundle, full_pairs)
    full_predictions = {
        "semantic": full_semantic,
        "semantic_ecfp": full_ecfp,
        "propensity_residual": full_propensity,
        "ecfp_propensity_joint": full_ecfp_propensity,
        "rna_prior_only": full_prior_only,
        "prior_offset_structural_residual": full_prior_offset,
        "prior_offset_semantic_residual": full_prior_offset_semantic,
        "prior_offset_no_loo": full_prior_offset_no_loo,
        "concentration_adaptive_router": (
            concentration * full_ecfp + (1.0 - concentration) * full_propensity
        ),
    }
    if score_export_dir is not None:
        export_frozen_score_partitions(
            score_export_dir,
            fold=fold,
            pairs=full_pairs,
            predictions=full_predictions,
            expected_models=score_export_models,
        )

    fold_rows = []
    prediction_rows = []
    ranking_rows = []
    for model, probability in predictions.items():
        fold_row = {
            "fold": fold,
            "model": model,
            "n_train": len(split.train.labels),
            "n_val": len(split.val.labels),
            "n_test": len(split.test.labels),
            "n_test_drugs": len(np.unique(split.test.pairs[:, 1])),
            "n_test_scaffolds": len(np.unique(scaffolds[split.test.pairs[:, 1]])),
            "drug_top10_concentration": concentration,
        }
        fold_row.update(metrics(split.test.labels, probability))
        fold_rows.append(fold_row)
        prediction_rows.extend(
            {
                "fold": fold,
                "model": model,
                "rna_id": int(pair[0]),
                "drug_id": int(pair[1]),
                "label": float(label),
                "probability": float(score),
            }
            for pair, label, score in zip(
                split.test.pairs, split.test.labels, probability, strict=True
            )
        )
    for model, probability in full_predictions.items():
        for drug_id in test_drugs:
            include = full_pairs[:, 1] == drug_id
            labels = full_labels[include]
            scores = probability[include]
            positive_count = int(labels.sum())
            order = np.argsort(-scores)
            top10 = order[:10]
            discounts = 1.0 / np.log2(np.arange(2, 12))
            dcg10 = float((labels[top10] * discounts).sum())
            ideal10 = float(discounts[: min(positive_count, 10)].sum())
            positive_ranks = np.flatnonzero(labels[order] == 1)
            ranking_rows.append(
                {
                    "fold": fold,
                    "model": model,
                    "drug_id": int(drug_id),
                    "positive_count": positive_count,
                    "candidate_count": len(labels),
                    "average_precision": float(average_precision_score(labels, scores)),
                    "recall_at_10": float(labels[top10].sum() / positive_count),
                    "ndcg_at_10": dcg10 / ideal10 if ideal10 > 0 else 0.0,
                    "mrr": float(1.0 / (positive_ranks[0] + 1)),
                }
            )
    return (
        pd.DataFrame(fold_rows),
        pd.DataFrame(prediction_rows),
        pd.DataFrame(ranking_rows),
    )


def paired_drug_metric_bootstrap(
    frame: pd.DataFrame,
    candidate: str,
    baseline: str,
    metric: str,
    iterations: int,
    seed: int,
) -> tuple[float, float, float]:
    wide = frame.pivot(index="drug_id", columns="model", values=metric)
    differences = (wide[candidate] - wide[baseline]).to_numpy()
    rng = np.random.default_rng(seed)
    draws = rng.choice(differences, size=(iterations, len(differences)), replace=True).mean(axis=1)
    low, median, high = np.quantile(draws, [0.025, 0.5, 0.975])
    return float(low), float(median), float(high)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=("Mi",), default="Mi")
    parser.add_argument("--dataset-dir", type=Path, default=Path("Dataset"))
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--split-seed", type=int, default=20261019)
    parser.add_argument("--max-epochs", type=int, default=120)
    parser.add_argument("--bootstrap-iterations", type=int, default=2000)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("results/scaffold-five-fold-confirmation")
    )
    parser.add_argument("--score-export-dir", type=Path)
    parser.add_argument(
        "--candidate-manifest",
        type=Path,
        default=Path("implement-stage/mc-tv-pu-oracle/CANDIDATE_MANIFEST.json"),
    )
    args = parser.parse_args()

    score_export_models = (
        validated_candidate_ids_for_export(
            args.candidate_manifest,
            root=Path.cwd(),
            pair_matrix_path=args.dataset_dir / "MiPair.npy",
            source_group="native_confirmation",
            runtime_parameters={
                "dataset": args.dataset,
                "dataset_dir": args.dataset_dir,
                "folds": args.folds,
                "split_seed": args.split_seed,
                "max_epochs": args.max_epochs,
            },
        )
        if args.score_export_dir is not None
        else ()
    )
    bundle = load_bundle(args.dataset_dir, args.dataset)
    fingerprints = load_drug_fingerprints(args.dataset_dir, args.dataset)
    scaffolds = load_drug_scaffolds(args.dataset_dir, args.dataset)
    fold_frames = []
    prediction_frames = []
    ranking_frames = []
    for fold in range(args.folds):
        print(f"[confirm] fold={fold + 1}/{args.folds}", flush=True)
        fold_metrics, predictions, ranking = run_fold(
            bundle,
            fingerprints,
            scaffolds,
            fold,
            args.folds,
            args.split_seed,
            args.max_epochs,
            args.score_export_dir,
            score_export_models,
        )
        fold_frames.append(fold_metrics)
        prediction_frames.append(predictions)
        ranking_frames.append(ranking)

    fold_metrics = pd.concat(fold_frames, ignore_index=True)
    predictions = pd.concat(prediction_frames, ignore_index=True)
    rankings = pd.concat(ranking_frames, ignore_index=True)
    summary_rows = []
    for model, frame in predictions.groupby("model", sort=True):
        pooled = metrics(frame["label"].to_numpy(), frame["probability"].to_numpy())
        model_folds = fold_metrics[fold_metrics["model"] == model]
        summary_rows.append(
            {
                "model": model,
                **{f"pooled_{key}": value for key, value in pooled.items()},
                "fold_aupr_mean": float(model_folds["aupr"].mean()),
                "fold_aupr_std": float(model_folds["aupr"].std(ddof=1)),
            }
        )
    summary = pd.DataFrame(summary_rows)

    wide = predictions.pivot(
        index=["fold", "rna_id", "drug_id", "label"],
        columns="model",
        values="probability",
    ).reset_index()
    intervals = {}
    for baseline in ("semantic", "semantic_ecfp"):
        low, median, high = drug_cluster_bootstrap_delta(
            wide["label"].to_numpy(),
            wide["concentration_adaptive_router"].to_numpy(),
            wide[baseline].to_numpy(),
            wide["drug_id"].to_numpy(),
            args.bootstrap_iterations,
            args.split_seed,
        )
        intervals[f"router_minus_{baseline}"] = {
            "low_95": low,
            "median": median,
            "high_95": high,
        }
    for baseline in (
        "concentration_adaptive_router",
        "rna_prior_only",
        "prior_offset_semantic_residual",
        "prior_offset_no_loo",
    ):
        low, median, high = drug_cluster_bootstrap_delta(
            wide["label"].to_numpy(),
            wide["prior_offset_structural_residual"].to_numpy(),
            wide[baseline].to_numpy(),
            wide["drug_id"].to_numpy(),
            args.bootstrap_iterations,
            args.split_seed + 1,
        )
        intervals[f"prior_offset_structural_residual_minus_{baseline}"] = {
            "low_95": low,
            "median": median,
            "high_95": high,
        }

    ranking_summary = (
        rankings.groupby("model")[["average_precision", "recall_at_10", "ndcg_at_10", "mrr"]]
        .agg(["mean", "std"])
        .reset_index()
    )
    ranking_summary.columns = [
        "model" if column[0] == "model" else f"{column[0]}_{column[1]}"
        for column in ranking_summary.columns
    ]
    ranking_intervals = {}
    for baseline in ("semantic", "semantic_ecfp"):
        low, median, high = paired_drug_metric_bootstrap(
            rankings,
            "concentration_adaptive_router",
            baseline,
            "average_precision",
            5000,
            args.split_seed,
        )
        ranking_intervals[f"router_minus_{baseline}_macro_ap"] = {
            "low_95": low,
            "median": median,
            "high_95": high,
        }
    for baseline in (
        "concentration_adaptive_router",
        "rna_prior_only",
        "prior_offset_semantic_residual",
        "prior_offset_no_loo",
    ):
        low, median, high = paired_drug_metric_bootstrap(
            rankings,
            "prior_offset_structural_residual",
            baseline,
            "average_precision",
            5000,
            args.split_seed + 1,
        )
        ranking_intervals[
            f"prior_offset_structural_residual_minus_{baseline}_macro_ap"
        ] = {
            "low_95": low,
            "median": median,
            "high_95": high,
        }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    fold_metrics.to_csv(args.output_dir / "mi_fold_metrics.csv", index=False)
    predictions.to_csv(args.output_dir / "mi_oof_predictions.csv", index=False)
    summary.to_csv(args.output_dir / "mi_pooled_summary.csv", index=False)
    rankings.to_csv(args.output_dir / "mi_full_candidate_drug_metrics.csv", index=False)
    ranking_summary.to_csv(args.output_dir / "mi_full_candidate_summary.csv", index=False)
    config = {
        "dataset": args.dataset,
        "folds": args.folds,
        "split_seed": args.split_seed,
        "max_epochs": args.max_epochs,
        "bootstrap_iterations": args.bootstrap_iterations,
        "bootstrap_intervals": intervals,
        "full_candidate_bootstrap_intervals": ranking_intervals,
        "score_export_dir": (
            str(args.score_export_dir) if args.score_export_dir is not None else None
        ),
        "score_export_models": list(score_export_models),
    }
    (args.output_dir / "mi_config.json").write_text(
        json.dumps(config, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(summary.to_string(index=False), flush=True)
    print(json.dumps(intervals, indent=2, ensure_ascii=False), flush=True)
    print(ranking_summary.to_string(index=False), flush=True)
    print(json.dumps(ranking_intervals, indent=2, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
