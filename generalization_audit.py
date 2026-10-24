"""可复现的 ncRD-LG 泛化审计与 RAST 发现性实验。"""

from __future__ import annotations

import argparse
import copy
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch
from scipy import sparse
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


@dataclass(frozen=True)
class PairPartition:
    pairs: np.ndarray
    labels: np.ndarray


@dataclass(frozen=True)
class ProtocolSplit:
    train: PairPartition
    val: PairPartition
    test: PairPartition
    train_positive_matrix: np.ndarray


@dataclass(frozen=True)
class DatasetBundle:
    rna_features: np.ndarray
    drug_features: np.ndarray
    pair: np.ndarray
    pair_new: np.ndarray


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_bundle(dataset_dir: Path, prefix: str) -> DatasetBundle:
    if prefix not in {"Lnc", "Mi"}:
        raise ValueError("prefix 必须是 Lnc 或 Mi")
    return DatasetBundle(
        rna_features=torch.load(
            dataset_dir / f"{prefix}RNA.pt", map_location="cpu", weights_only=True
        ).float().numpy(),
        drug_features=torch.load(
            dataset_dir / f"{prefix}Drug.pt", map_location="cpu", weights_only=True
        ).float().numpy(),
        pair=np.load(dataset_dir / f"{prefix}Pair.npy", allow_pickle=False),
        pair_new=np.load(dataset_dir / f"{prefix}Pair_new.npy", allow_pickle=False),
    )


def load_drug_fingerprints(dataset_dir: Path, prefix: str, radius: int = 2, n_bits: int = 2048) -> np.ndarray:
    """按公开药物表行序生成 Morgan 指纹，不读取关联标签。"""
    from rdkit import Chem, DataStructs
    from rdkit.Chem import rdFingerprintGenerator

    if prefix == "Lnc":
        frame = pd.read_excel(dataset_dir / "lnc_drug_with_SMILES.xlsx")
    elif prefix == "Mi":
        frame = pd.read_csv(dataset_dir / "mi_drug_with_SMILES.csv")
    else:
        raise ValueError("prefix 必须是 Lnc 或 Mi")
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=n_bits)
    fingerprints = np.zeros((len(frame), n_bits), dtype=np.float32)
    for index, smiles in enumerate(frame["SMILES"].astype(str)):
        molecule = Chem.MolFromSmiles(smiles)
        if molecule is None:
            raise ValueError(f"无法解析第 {index} 行 SMILES")
        fingerprint = generator.GetFingerprint(molecule)
        DataStructs.ConvertToNumpyArray(fingerprint, fingerprints[index])
    return fingerprints


def load_precomputed_drug_features(path: Path, expected_rows: int) -> np.ndarray:
    """读取不含标签的药物表征，并严格核对行序长度与数值。"""
    features = np.load(path, allow_pickle=False)
    if features.ndim != 2:
        raise ValueError("药物表征必须是二维数组")
    if features.shape[0] != expected_rows:
        raise ValueError(
            f"药物表征行数 {features.shape[0]} 与数据集药物数 {expected_rows} 不一致"
        )
    if not np.issubdtype(features.dtype, np.number) or not np.isfinite(features).all():
        raise ValueError("药物表征必须全部为有限数值")
    return features.astype(np.float32, copy=False)


def load_drug_scaffolds(dataset_dir: Path, prefix: str) -> np.ndarray:
    """提取 Bemis–Murcko scaffold；无环分子按 canonical SMILES 单独分组。"""
    from rdkit import Chem
    from rdkit.Chem.Scaffolds import MurckoScaffold

    if prefix == "Lnc":
        frame = pd.read_excel(dataset_dir / "lnc_drug_with_SMILES.xlsx")
    elif prefix == "Mi":
        frame = pd.read_csv(dataset_dir / "mi_drug_with_SMILES.csv")
    else:
        raise ValueError("prefix 必须是 Lnc 或 Mi")
    scaffolds: list[str] = []
    for index, smiles in enumerate(frame["SMILES"].astype(str)):
        molecule = Chem.MolFromSmiles(smiles)
        if molecule is None:
            raise ValueError(f"无法解析第 {index} 行 SMILES")
        scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=molecule, includeChirality=False)
        if not scaffold:
            scaffold = f"ACYCLIC::{Chem.MolToSmiles(molecule, canonical=True)}"
        scaffolds.append(scaffold)
    return np.asarray(scaffolds, dtype=str)


def tanimoto_similarity(query: np.ndarray, reference: np.ndarray) -> np.ndarray:
    intersection = query @ reference.T
    union = query.sum(axis=1, keepdims=True) + reference.sum(axis=1)[None, :] - intersection
    return intersection / np.maximum(union, 1e-8)


def binary_jaccard_similarity_matrix(features) -> np.ndarray:
    """计算稀疏或稠密非零特征集合之间的 Jaccard 相似度。"""
    binary = sparse.csr_matrix(features != 0, dtype=np.float64)
    intersection = (binary @ binary.T).toarray()
    size = np.asarray(binary.sum(axis=1)).ravel()
    union = size[:, None] + size[None, :] - intersection
    return np.divide(
        intersection,
        union,
        out=np.zeros_like(intersection),
        where=union > 0,
    )


def chemical_coverage_weight(
    train_fingerprints: np.ndarray, query_fingerprints: np.ndarray
) -> tuple[np.ndarray, float, float]:
    """以训练集 leave-one-out 覆盖分位标定未见药物的结构可靠性。"""
    if len(train_fingerprints) < 2:
        return np.zeros(len(query_fingerprints), dtype=np.float64), 0.0, 1.0
    train_similarity = tanimoto_similarity(train_fingerprints, train_fingerprints)
    np.fill_diagonal(train_similarity, -np.inf)
    nearest_train = train_similarity.max(axis=1)
    low, high = np.quantile(nearest_train, [0.10, 0.90])
    query_coverage = tanimoto_similarity(query_fingerprints, train_fingerprints).max(axis=1)
    if high <= low + 1e-12:
        weight = (query_coverage >= high).astype(np.float64)
    else:
        weight = np.clip((query_coverage - low) / (high - low), 0.0, 1.0)
    return weight, float(low), float(high)


def rna_association_prior_scores(
    train_positive_matrix: np.ndarray, pairs: np.ndarray
) -> np.ndarray:
    active_drug_count = int(np.count_nonzero(train_positive_matrix.sum(axis=0) > 0))
    if active_drug_count == 0:
        raise ValueError("训练正例图没有活跃药物")
    row_degree = train_positive_matrix.sum(axis=1).astype(np.float64)
    probability = (row_degree + 1.0) / (active_drug_count + 2.0)
    return probability[pairs[:, 0]]


def gene_bridge_score_matrix(rna_gene_matrix, drug_gene_matrix) -> np.ndarray:
    """计算 RNA 与药物靶基因证据向量的余弦桥接分数。"""
    if rna_gene_matrix.shape[1] != drug_gene_matrix.shape[1]:
        raise ValueError("RNA 与药物基因矩阵的列数必须一致")
    numerator = rna_gene_matrix @ drug_gene_matrix.T
    if hasattr(numerator, "toarray"):
        numerator = numerator.toarray()
    numerator = np.asarray(numerator, dtype=np.float64)
    if hasattr(rna_gene_matrix, "multiply"):
        rna_norm = np.sqrt(np.asarray(rna_gene_matrix.multiply(rna_gene_matrix).sum(axis=1)).ravel())
        drug_norm = np.sqrt(np.asarray(drug_gene_matrix.multiply(drug_gene_matrix).sum(axis=1)).ravel())
    else:
        rna_norm = np.linalg.norm(np.asarray(rna_gene_matrix), axis=1)
        drug_norm = np.linalg.norm(np.asarray(drug_gene_matrix), axis=1)
    denominator = rna_norm[:, None] * drug_norm[None, :]
    return np.divide(
        numerator,
        denominator,
        out=np.zeros_like(numerator),
        where=denominator > 0,
    )


def idf_context_bridge_score_matrix(rna_context, drug_context) -> np.ndarray:
    """对两侧共享上下文做实体频率逆权重后的余弦桥接。"""
    if rna_context.shape[1] != drug_context.shape[1]:
        raise ValueError("RNA 与药物上下文维度不一致")
    rna = sparse.csr_matrix(rna_context, dtype=np.float64)
    drug = sparse.csr_matrix(drug_context, dtype=np.float64)
    document_frequency = np.asarray((rna > 0).sum(axis=0)).ravel()
    document_frequency += np.asarray((drug > 0).sum(axis=0)).ravel()
    entity_count = rna.shape[0] + drug.shape[0]
    idf = np.log((entity_count + 1.0) / (document_frequency + 1.0)) + 1.0
    return gene_bridge_score_matrix(rna.multiply(idf), drug.multiply(idf))


def pathway_projection_bridge_score_matrix(
    rna_gene_matrix,
    drug_gene_matrix,
    gene_pathway_incidence,
) -> np.ndarray:
    """以通路大小平方根归一化后投影基因证据，并计算实体间通路余弦。"""
    if rna_gene_matrix.shape[1] != drug_gene_matrix.shape[1]:
        raise ValueError("RNA 与药物基因矩阵的列数必须一致")
    if gene_pathway_incidence.shape[0] != rna_gene_matrix.shape[1]:
        raise ValueError("gene_pathway_incidence 与基因维度不一致")
    incidence = sparse.csr_matrix(gene_pathway_incidence, dtype=np.float64)
    pathway_size = np.asarray(incidence.sum(axis=0)).ravel()
    if np.any(pathway_size <= 0):
        raise ValueError("通路 incidence 不得包含空列")
    scaled = incidence @ sparse.diags(1.0 / np.sqrt(pathway_size))
    rna_pathway = sparse.csr_matrix(rna_gene_matrix, dtype=np.float64) @ scaled
    drug_pathway = sparse.csr_matrix(drug_gene_matrix, dtype=np.float64) @ scaled
    return gene_bridge_score_matrix(rna_pathway, drug_pathway)


def ppi_propagated_gene_bridge_score_matrix(
    rna_gene_matrix,
    drug_gene_matrix,
    ppi_adjacency,
    beta: float,
) -> np.ndarray:
    """以对称归一化的一跳 PPI 扩展 RNA-drug 靶基因桥。"""
    if beta < 0:
        raise ValueError("beta 必须非负")
    if rna_gene_matrix.shape[1] != drug_gene_matrix.shape[1]:
        raise ValueError("RNA 与药物基因矩阵的列数必须一致")
    gene_count = rna_gene_matrix.shape[1]
    if ppi_adjacency.shape != (gene_count, gene_count):
        raise ValueError("PPI 邻接矩阵与基因维度不一致")

    rna = sparse.csr_matrix(rna_gene_matrix, dtype=np.float64)
    drug = sparse.csr_matrix(drug_gene_matrix, dtype=np.float64)
    adjacency = sparse.csr_matrix(ppi_adjacency, dtype=np.float64)
    adjacency = adjacency.maximum(adjacency.T)
    adjacency.setdiag(0.0)
    adjacency.eliminate_zeros()

    degree = np.asarray(adjacency.sum(axis=1)).ravel()
    inverse_sqrt_degree = np.divide(
        1.0,
        np.sqrt(degree),
        out=np.zeros_like(degree),
        where=degree > 0,
    )
    normalized_adjacency = (
        sparse.diags(inverse_sqrt_degree)
        @ adjacency
        @ sparse.diags(inverse_sqrt_degree)
    )

    def normalize_rows(matrix):
        norm = np.sqrt(np.asarray(matrix.multiply(matrix).sum(axis=1)).ravel())
        inverse_norm = np.divide(
            1.0,
            norm,
            out=np.zeros_like(norm),
            where=norm > 0,
        )
        return sparse.diags(inverse_norm) @ matrix

    rna_normalized = normalize_rows(rna)
    drug_normalized = normalize_rows(drug)
    operator = sparse.eye(gene_count, format="csr") + beta * normalized_adjacency
    score = rna_normalized @ operator @ drug_normalized.T
    return np.asarray(score.toarray(), dtype=np.float64)


def ppi_restart_diffusion_gene_bridge_score_matrix(
    rna_gene_matrix,
    drug_gene_matrix,
    ppi_adjacency,
    continuation: float,
    steps: int = 10,
) -> np.ndarray:
    """在 PPI 随机游走上对两侧靶基因做带 restart 的固定步数扩散。"""
    if not 0.0 <= continuation < 1.0:
        raise ValueError("continuation 必须位于 [0,1)")
    if steps <= 0:
        raise ValueError("steps 必须为正")
    if rna_gene_matrix.shape[1] != drug_gene_matrix.shape[1]:
        raise ValueError("RNA 与药物基因矩阵的列数必须一致")
    gene_count = rna_gene_matrix.shape[1]
    if ppi_adjacency.shape != (gene_count, gene_count):
        raise ValueError("PPI 邻接矩阵与基因维度不一致")

    adjacency = sparse.csr_matrix(ppi_adjacency, dtype=np.float32)
    adjacency = adjacency.maximum(adjacency.T)
    adjacency.setdiag(0.0)
    adjacency.eliminate_zeros()
    degree = np.asarray(adjacency.sum(axis=1)).ravel()
    inverse_degree = np.divide(
        1.0, degree, out=np.zeros_like(degree), where=degree > 0
    )
    transition = sparse.diags(inverse_degree) @ adjacency

    def diffuse(matrix) -> np.ndarray:
        seed = sparse.csr_matrix(matrix, dtype=np.float32).toarray()
        row_sum = seed.sum(axis=1, keepdims=True)
        seed = np.divide(
            seed,
            row_sum,
            out=np.zeros_like(seed),
            where=row_sum > 0,
        )
        state = seed.copy()
        for _ in range(steps):
            state = (1.0 - continuation) * seed + continuation * (
                state @ transition
            )
        norm = np.linalg.norm(state, axis=1, keepdims=True)
        return np.divide(
            state,
            norm,
            out=np.zeros_like(state),
            where=norm > 0,
        )

    return np.asarray(diffuse(rna_gene_matrix) @ diffuse(drug_gene_matrix).T)


def select_ppi_gene_bridge_prior(
    train_positive_matrix: np.ndarray,
    bridge_by_beta: dict[float, np.ndarray],
    validation_pairs: np.ndarray,
    validation_labels: np.ndarray,
    test_pairs: np.ndarray,
    alpha_values: tuple[float, ...] = (0.0, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5),
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    """仅用验证药物选择 PPI 传播和先验融合强度。"""
    if not bridge_by_beta:
        raise ValueError("bridge_by_beta 不能为空")
    if any(matrix.shape != train_positive_matrix.shape for matrix in bridge_by_beta.values()):
        raise ValueError("PPI bridge 矩阵与关联矩阵形状必须一致")
    prior_validation = rna_association_prior_scores(train_positive_matrix, validation_pairs)
    prior_test = rna_association_prior_scores(train_positive_matrix, test_pairs)
    best_ap = -math.inf
    best_beta = min(bridge_by_beta)
    best_alpha = 0.0
    best_validation = prior_validation
    best_test = prior_test
    for beta in sorted(bridge_by_beta):
        bridge_matrix = bridge_by_beta[beta]
        validation_bridge = bridge_matrix[
            validation_pairs[:, 0], validation_pairs[:, 1]
        ]
        test_bridge = bridge_matrix[test_pairs[:, 0], test_pairs[:, 1]]
        for alpha in sorted(alpha_values):
            validation = prior_validation + alpha * validation_bridge
            macro_ap = drug_macro_average_precision(
                validation_labels, validation, validation_pairs[:, 1]
            )
            if macro_ap > best_ap + 1e-12:
                best_ap = macro_ap
                best_beta = beta
                best_alpha = alpha
                best_validation = validation
                best_test = prior_test + alpha * test_bridge
    return best_validation, best_test, {
        "beta": float(best_beta),
        "alpha": float(best_alpha),
        "validation_macro_ap": float(best_ap),
    }


def select_gene_bridge_prior(
    train_positive_matrix: np.ndarray,
    bridge_matrix: np.ndarray,
    validation_pairs: np.ndarray,
    validation_labels: np.ndarray,
    test_pairs: np.ndarray,
    alpha_values: tuple[float, ...] = (0.0, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5),
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    if bridge_matrix.shape != train_positive_matrix.shape:
        raise ValueError("bridge_matrix 与关联矩阵形状必须一致")
    prior_validation = rna_association_prior_scores(train_positive_matrix, validation_pairs)
    prior_test = rna_association_prior_scores(train_positive_matrix, test_pairs)
    validation_bridge = bridge_matrix[validation_pairs[:, 0], validation_pairs[:, 1]]
    test_bridge = bridge_matrix[test_pairs[:, 0], test_pairs[:, 1]]
    best_ap = -math.inf
    best_alpha = 0.0
    best_validation = prior_validation
    best_test = prior_test
    for alpha in sorted(alpha_values):
        validation = prior_validation + alpha * validation_bridge
        macro_ap = drug_macro_average_precision(
            validation_labels, validation, validation_pairs[:, 1]
        )
        if macro_ap > best_ap + 1e-12:
            best_ap = macro_ap
            best_alpha = alpha
            best_validation = validation
            best_test = prior_test + alpha * test_bridge
    return best_validation, best_test, {
        "alpha": float(best_alpha),
        "validation_macro_ap": float(best_ap),
    }


def mirna_seed_family_ids(sequences: np.ndarray) -> np.ndarray:
    normalized = np.asarray(
        [str(sequence).upper().replace("T", "U") for sequence in sequences], dtype=str
    )
    if np.any(np.char.str_len(normalized) < 8):
        raise ValueError("成熟 miRNA 序列长度必须至少为 8")
    seeds = np.asarray([sequence[1:8] for sequence in normalized], dtype=str)
    _, family_ids = np.unique(seeds, return_inverse=True)
    return family_ids.astype(np.int64)


def seed_family_hierarchical_prior_scores(
    train_positive_matrix: np.ndarray,
    family_ids: np.ndarray,
    pairs: np.ndarray,
    alpha: float,
) -> np.ndarray:
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha 必须位于 [0,1]")
    if len(family_ids) != train_positive_matrix.shape[0]:
        raise ValueError("family_ids 与 RNA 行数不一致")
    active_drug_count = int(np.count_nonzero(train_positive_matrix.sum(axis=0) > 0))
    if active_drug_count == 0:
        raise ValueError("训练正例图没有活跃药物")
    degree = train_positive_matrix.sum(axis=1).astype(np.float64)
    individual = (degree + 1.0) / (active_drug_count + 2.0)
    family_probability = np.empty_like(individual)
    for family_id in np.unique(family_ids):
        members = np.flatnonzero(family_ids == family_id)
        probability = (degree[members].sum() + 1.0) / (
            active_drug_count * len(members) + 2.0
        )
        family_probability[members] = probability
    rna_ids = pairs[:, 0]
    return (1.0 - alpha) * individual[rna_ids] + alpha * family_probability[rna_ids]


def select_seed_family_hierarchical_prior(
    train_positive_matrix: np.ndarray,
    family_ids: np.ndarray,
    validation_pairs: np.ndarray,
    validation_labels: np.ndarray,
    test_pairs: np.ndarray,
    alpha_values: tuple[float, ...] = (0.0, 0.1, 0.25, 0.5, 0.75, 1.0),
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    best_ap = -math.inf
    best_alpha = 0.0
    best_validation = None
    for alpha in sorted(alpha_values):
        validation = seed_family_hierarchical_prior_scores(
            train_positive_matrix, family_ids, validation_pairs, alpha
        )
        macro_ap = drug_macro_average_precision(
            validation_labels, validation, validation_pairs[:, 1]
        )
        if macro_ap > best_ap + 1e-12:
            best_ap = macro_ap
            best_alpha = alpha
            best_validation = validation
    if best_validation is None:
        raise RuntimeError("seed-family 验证选择失败")
    test = seed_family_hierarchical_prior_scores(
        train_positive_matrix, family_ids, test_pairs, best_alpha
    )
    return best_validation, test, {
        "alpha": float(best_alpha),
        "validation_macro_ap": float(best_ap),
    }


def structural_empirical_bayes_scores(
    fingerprints: np.ndarray,
    train_positive_matrix: np.ndarray,
    pairs: np.ndarray,
    k: int,
    kappa: float,
) -> np.ndarray:
    """用 Tanimoto 加权训练关联谱，并向 RNA 全局先验收缩。"""
    if k <= 0 or kappa <= 0:
        raise ValueError("k 与 kappa 必须为正")
    active_drugs = np.flatnonzero(train_positive_matrix.sum(axis=0) > 0)
    if len(active_drugs) == 0:
        raise ValueError("训练正例图没有活跃药物")
    row_degree = train_positive_matrix.sum(axis=1).astype(np.float64)
    global_prior = (row_degree + 1.0) / (len(active_drugs) + 2.0)
    query_drugs, inverse = np.unique(pairs[:, 1], return_inverse=True)
    similarity = tanimoto_similarity(
        fingerprints[query_drugs], fingerprints[active_drugs]
    )
    active_position = {int(drug_id): index for index, drug_id in enumerate(active_drugs)}
    for query_index, drug_id in enumerate(query_drugs):
        if int(drug_id) in active_position:
            similarity[query_index, active_position[int(drug_id)]] = -np.inf
    effective_k = min(k, len(active_drugs))
    neighbor_positions = np.argpartition(
        similarity, -effective_k, axis=1
    )[:, -effective_k:]
    neighbor_similarity = np.take_along_axis(similarity, neighbor_positions, axis=1)
    neighbor_similarity = np.where(
        np.isfinite(neighbor_similarity), np.maximum(neighbor_similarity, 0.0), 0.0
    )
    neighbor_drugs = active_drugs[neighbor_positions]
    profiles = np.empty((len(query_drugs), train_positive_matrix.shape[0]), dtype=np.float64)
    for query_index in range(len(query_drugs)):
        weights = neighbor_similarity[query_index]
        evidence = train_positive_matrix[:, neighbor_drugs[query_index]] @ weights
        profiles[query_index] = (
            kappa * global_prior + evidence
        ) / (kappa + weights.sum())
    return profiles[inverse, pairs[:, 0]]


def select_structural_empirical_bayes(
    fingerprints: np.ndarray,
    train_positive_matrix: np.ndarray,
    validation_pairs: np.ndarray,
    validation_labels: np.ndarray,
    test_pairs: np.ndarray,
    k_values: tuple[int, ...] = (1, 3, 5, 10, 20),
    kappa_values: tuple[float, ...] = (0.5, 1.0, 2.0, 5.0, 10.0, 20.0),
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    prior_validation = rna_association_prior_scores(
        train_positive_matrix, validation_pairs
    )
    prior_test = rna_association_prior_scores(train_positive_matrix, test_pairs)
    best_ap = drug_macro_average_precision(
        validation_labels, prior_validation, validation_pairs[:, 1]
    )
    best_validation = prior_validation
    best_test = prior_test
    best_k = 0
    best_kappa = math.inf
    for k in k_values:
        for kappa in sorted(kappa_values, reverse=True):
            validation = structural_empirical_bayes_scores(
                fingerprints, train_positive_matrix, validation_pairs, k, kappa
            )
            macro_ap = drug_macro_average_precision(
                validation_labels, validation, validation_pairs[:, 1]
            )
            if macro_ap > best_ap + 1e-12:
                best_ap = macro_ap
                best_validation = validation
                best_test = structural_empirical_bayes_scores(
                    fingerprints, train_positive_matrix, test_pairs, k, kappa
                )
                best_k = k
                best_kappa = kappa
    return best_validation, best_test, {
        "k": float(best_k),
        "kappa": float(best_kappa),
        "validation_macro_ap": float(best_ap),
    }


def degree_debiased_neighborhood_enrichment_scores(
    fingerprints: np.ndarray,
    train_positive_matrix: np.ndarray,
    pairs: np.ndarray,
    k: int,
    kappa: float,
    tau: float,
    beta: float,
) -> np.ndarray:
    """按结构邻域相对全局 RNA 度数的标准化富集重排。"""
    active_drugs = np.flatnonzero(train_positive_matrix.sum(axis=0) > 0)
    row_degree = train_positive_matrix.sum(axis=1).astype(np.float64)
    prior = (row_degree + 1.0) / (len(active_drugs) + 2.0)
    if len(active_drugs) == 0:
        return prior[pairs[:, 0]]
    query_drugs, inverse = np.unique(pairs[:, 1], return_inverse=True)
    similarity = tanimoto_similarity(
        fingerprints[query_drugs], fingerprints[active_drugs]
    )
    effective_k = min(k, len(active_drugs))
    positions = np.argpartition(similarity, -effective_k, axis=1)[:, -effective_k:]
    weights = np.take_along_axis(similarity, positions, axis=1)
    weights = np.maximum(weights, 0.0)
    neighbor_drugs = active_drugs[positions]
    profiles = np.empty((len(query_drugs), train_positive_matrix.shape[0]), dtype=np.float64)
    prior_logit = np.log(prior) - np.log1p(-prior)
    for query_index in range(len(query_drugs)):
        weight = weights[query_index]
        weight_sum = float(weight.sum())
        if weight_sum <= 0:
            profiles[query_index] = prior
            continue
        evidence = train_positive_matrix[:, neighbor_drugs[query_index]] @ weight
        variance = prior * (1.0 - prior) * float(np.square(weight).sum())
        enrichment = (evidence - prior * weight_sum) / np.sqrt(variance + kappa)
        reliability = weight_sum / (weight_sum + tau)
        logit = prior_logit + beta * reliability * enrichment
        profiles[query_index] = 1.0 / (1.0 + np.exp(-np.clip(logit, -30.0, 30.0)))
    return profiles[inverse, pairs[:, 0]]


def select_degree_debiased_neighborhood_enrichment(
    fingerprints: np.ndarray,
    train_positive_matrix: np.ndarray,
    validation_pairs: np.ndarray,
    validation_labels: np.ndarray,
    test_pairs: np.ndarray,
    k_values: tuple[int, ...] = (5, 10, 20, 50),
    kappa_values: tuple[float, ...] = (0.25, 1.0, 4.0),
    tau_values: tuple[float, ...] = (0.5, 2.0, 8.0),
    beta_values: tuple[float, ...] = (0.1, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0),
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    prior_validation = rna_association_prior_scores(
        train_positive_matrix, validation_pairs
    )
    prior_test = rna_association_prior_scores(train_positive_matrix, test_pairs)
    best_ap = drug_macro_average_precision(
        validation_labels, prior_validation, validation_pairs[:, 1]
    )
    best_validation = prior_validation
    best_test = prior_test
    best_parameters = (0, math.inf, math.inf, 0.0)
    for k in sorted(k_values):
        for kappa in sorted(kappa_values, reverse=True):
            for tau in sorted(tau_values, reverse=True):
                for beta in sorted(beta_values):
                    validation = degree_debiased_neighborhood_enrichment_scores(
                        fingerprints,
                        train_positive_matrix,
                        validation_pairs,
                        k,
                        kappa,
                        tau,
                        beta,
                    )
                    macro_ap = drug_macro_average_precision(
                        validation_labels, validation, validation_pairs[:, 1]
                    )
                    if macro_ap > best_ap + 1e-12:
                        best_ap = macro_ap
                        best_validation = validation
                        best_test = degree_debiased_neighborhood_enrichment_scores(
                            fingerprints,
                            train_positive_matrix,
                            test_pairs,
                            k,
                            kappa,
                            tau,
                            beta,
                        )
                        best_parameters = (k, kappa, tau, beta)
    k, kappa, tau, beta = best_parameters
    return best_validation, best_test, {
        "k": float(k),
        "kappa": float(kappa),
        "tau": float(tau),
        "beta": float(beta),
        "validation_macro_ap": float(best_ap),
    }


def select_prior_centered_drug_kernel_ridge(
    drug_kernel: np.ndarray,
    train_positive_matrix: np.ndarray,
    validation_pairs: np.ndarray,
    validation_labels: np.ndarray,
    test_pairs: np.ndarray,
    lambda_values: tuple[float, ...] = (0.01, 0.1, 1.0, 10.0, 100.0),
    alpha_values: tuple[float, ...] = (0.0, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0),
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    if drug_kernel.shape != (
        train_positive_matrix.shape[1],
        train_positive_matrix.shape[1],
    ):
        raise ValueError("drug_kernel 与药物维度不一致")
    active_drugs = np.flatnonzero(train_positive_matrix.sum(axis=0) > 0)
    if len(active_drugs) == 0:
        raise ValueError("训练正例图没有活跃药物")
    row_degree = train_positive_matrix.sum(axis=1).astype(np.float64)
    global_prior = (row_degree + 1.0) / (len(active_drugs) + 2.0)
    target_residual = train_positive_matrix[:, active_drugs].T - global_prior[None, :]
    train_kernel = drug_kernel[np.ix_(active_drugs, active_drugs)]
    validation_drugs, validation_inverse = np.unique(
        validation_pairs[:, 1], return_inverse=True
    )
    test_drugs, test_inverse = np.unique(test_pairs[:, 1], return_inverse=True)
    validation_kernel = drug_kernel[np.ix_(validation_drugs, active_drugs)]
    test_kernel = drug_kernel[np.ix_(test_drugs, active_drugs)]
    prior_validation = global_prior[validation_pairs[:, 0]]
    prior_test = global_prior[test_pairs[:, 0]]
    best_ap = drug_macro_average_precision(
        validation_labels, prior_validation, validation_pairs[:, 1]
    )
    best_validation = prior_validation
    best_test = prior_test
    best_lambda = math.inf
    best_alpha = 0.0
    identity = np.eye(len(active_drugs), dtype=np.float64)
    for ridge_lambda in sorted(lambda_values, reverse=True):
        coefficients = np.linalg.solve(
            train_kernel + ridge_lambda * identity, target_residual
        )
        validation_profile = validation_kernel @ coefficients
        test_profile = test_kernel @ coefficients
        validation_residual = validation_profile[
            validation_inverse, validation_pairs[:, 0]
        ]
        test_residual = test_profile[test_inverse, test_pairs[:, 0]]
        for alpha in sorted(alpha_values):
            if alpha == 0:
                continue
            validation = np.clip(
                prior_validation + alpha * validation_residual, 0.0, 1.0
            )
            macro_ap = drug_macro_average_precision(
                validation_labels, validation, validation_pairs[:, 1]
            )
            if macro_ap > best_ap + 1e-12:
                best_ap = macro_ap
                best_validation = validation
                best_test = np.clip(prior_test + alpha * test_residual, 0.0, 1.0)
                best_lambda = ridge_lambda
                best_alpha = alpha
    return best_validation, best_test, {
        "lambda": float(best_lambda),
        "alpha": float(best_alpha),
        "validation_macro_ap": float(best_ap),
    }


def select_prior_centered_tanimoto_krr(
    fingerprints: np.ndarray,
    train_positive_matrix: np.ndarray,
    validation_pairs: np.ndarray,
    validation_labels: np.ndarray,
    test_pairs: np.ndarray,
    lambda_values: tuple[float, ...] = (0.01, 0.1, 1.0, 10.0, 100.0),
    alpha_values: tuple[float, ...] = (0.0, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0),
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    kernel = tanimoto_similarity(fingerprints, fingerprints)
    return select_prior_centered_drug_kernel_ridge(
        kernel,
        train_positive_matrix,
        validation_pairs,
        validation_labels,
        test_pairs,
        lambda_values,
        alpha_values,
    )


def select_low_rank_prior_centered_drug_kernel_ridge(
    drug_kernel: np.ndarray,
    train_positive_matrix: np.ndarray,
    validation_pairs: np.ndarray,
    validation_labels: np.ndarray,
    test_pairs: np.ndarray,
    rank_values: tuple[int, ...] = (8, 16, 32, 64),
    lambda_values: tuple[float, ...] = (0.01, 0.1, 1.0, 10.0, 100.0),
    alpha_values: tuple[float, ...] = (0.0, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0),
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    """把先验中心化关联谱压缩后，在药物核上归纳迁移。"""
    drug_count = train_positive_matrix.shape[1]
    if drug_kernel.shape != (drug_count, drug_count):
        raise ValueError("drug_kernel 与药物维度不一致")
    active_drugs = np.flatnonzero(train_positive_matrix.sum(axis=0) > 0)
    if len(active_drugs) == 0:
        raise ValueError("训练正例图没有活跃药物")
    row_degree = train_positive_matrix.sum(axis=1).astype(np.float64)
    global_prior = (row_degree + 1.0) / (len(active_drugs) + 2.0)
    target_residual = (
        train_positive_matrix[:, active_drugs].T.astype(np.float64)
        - global_prior[None, :]
    )
    left, singular_values, right = np.linalg.svd(
        target_residual, full_matrices=False
    )
    maximum_rank = min(target_residual.shape)
    valid_ranks = sorted({rank for rank in rank_values if 0 < rank <= maximum_rank})
    if not valid_ranks:
        raise ValueError("没有可用的低秩维度")

    train_kernel = drug_kernel[np.ix_(active_drugs, active_drugs)]
    validation_drugs, validation_inverse = np.unique(
        validation_pairs[:, 1], return_inverse=True
    )
    test_drugs, test_inverse = np.unique(test_pairs[:, 1], return_inverse=True)
    validation_kernel = drug_kernel[np.ix_(validation_drugs, active_drugs)]
    test_kernel = drug_kernel[np.ix_(test_drugs, active_drugs)]
    prior_validation = global_prior[validation_pairs[:, 0]]
    prior_test = global_prior[test_pairs[:, 0]]
    best_ap = drug_macro_average_precision(
        validation_labels, prior_validation, validation_pairs[:, 1]
    )
    best_validation = prior_validation
    best_test = prior_test
    best_rank = 0
    best_lambda = math.inf
    best_alpha = 0.0
    identity = np.eye(len(active_drugs), dtype=np.float64)
    for rank in valid_ranks:
        train_factors = left[:, :rank] * singular_values[:rank]
        reconstruction_basis = right[:rank]
        for ridge_lambda in sorted(lambda_values, reverse=True):
            coefficients = np.linalg.solve(
                train_kernel + ridge_lambda * identity, train_factors
            )
            validation_profile = (
                validation_kernel @ coefficients @ reconstruction_basis
            )
            test_profile = test_kernel @ coefficients @ reconstruction_basis
            validation_residual = validation_profile[
                validation_inverse, validation_pairs[:, 0]
            ]
            test_residual = test_profile[test_inverse, test_pairs[:, 0]]
            for alpha in sorted(alpha_values):
                if alpha == 0:
                    continue
                validation = np.clip(
                    prior_validation + alpha * validation_residual, 0.0, 1.0
                )
                macro_ap = drug_macro_average_precision(
                    validation_labels, validation, validation_pairs[:, 1]
                )
                if macro_ap > best_ap + 1e-12:
                    best_ap = macro_ap
                    best_validation = validation
                    best_test = np.clip(
                        prior_test + alpha * test_residual, 0.0, 1.0
                    )
                    best_rank = rank
                    best_lambda = ridge_lambda
                    best_alpha = alpha
    return best_validation, best_test, {
        "rank": float(best_rank),
        "lambda": float(best_lambda),
        "alpha": float(best_alpha),
        "validation_macro_ap": float(best_ap),
    }


def _extra_trees_latent_residual_profiles(
    fingerprints: np.ndarray,
    train_positive_matrix: np.ndarray,
    query_drugs: np.ndarray,
    rank: int,
    min_samples_leaf: int,
    max_features: float,
    seed: int,
    n_estimators: int = 256,
) -> tuple[np.ndarray, np.ndarray]:
    from sklearn.ensemble import ExtraTreesRegressor

    active_drugs = np.flatnonzero(train_positive_matrix.sum(axis=0) > 0)
    if len(active_drugs) == 0:
        raise ValueError("训练正例图没有活跃药物")
    row_degree = train_positive_matrix.sum(axis=1).astype(np.float64)
    prior = (row_degree + 1.0) / (len(active_drugs) + 2.0)
    target = (
        train_positive_matrix[:, active_drugs].T.astype(np.float64)
        - prior[None, :]
    )
    left, singular_values, right = np.linalg.svd(target, full_matrices=False)
    effective_rank = min(rank, len(singular_values))
    factors = left[:, :effective_rank] * singular_values[:effective_rank]
    model = ExtraTreesRegressor(
        n_estimators=n_estimators,
        min_samples_leaf=min_samples_leaf,
        max_features=max_features,
        bootstrap=False,
        random_state=seed,
        n_jobs=-1,
    )
    model.fit(fingerprints[active_drugs], factors)
    predicted_factors = model.predict(fingerprints[query_drugs])
    residual = predicted_factors @ right[:effective_rank]
    return prior, residual


def extra_trees_latent_profile_scores(
    fingerprints: np.ndarray,
    train_positive_matrix: np.ndarray,
    pairs: np.ndarray,
    rank: int,
    min_samples_leaf: int,
    max_features: float,
    alpha: float,
    seed: int,
    n_estimators: int = 256,
) -> np.ndarray:
    """以非线性树模型把药物结构映射到低秩关联残差谱。"""
    active_drugs = np.flatnonzero(train_positive_matrix.sum(axis=0) > 0)
    row_degree = train_positive_matrix.sum(axis=1).astype(np.float64)
    prior = (row_degree + 1.0) / (len(active_drugs) + 2.0)
    if alpha == 0:
        return prior[pairs[:, 0]]
    query_drugs, inverse = np.unique(pairs[:, 1], return_inverse=True)
    prior, residual = _extra_trees_latent_residual_profiles(
        fingerprints,
        train_positive_matrix,
        query_drugs,
        rank,
        min_samples_leaf,
        max_features,
        seed,
        n_estimators,
    )
    profiles = np.clip(prior[None, :] + alpha * residual, 0.0, 1.0)
    return profiles[inverse, pairs[:, 0]]


def select_extra_trees_latent_profile(
    fingerprints: np.ndarray,
    train_positive_matrix: np.ndarray,
    validation_pairs: np.ndarray,
    validation_labels: np.ndarray,
    test_pairs: np.ndarray,
    seed: int,
    rank_values: tuple[int, ...] = (16, 32, 64),
    leaf_values: tuple[int, ...] = (5, 2),
    feature_values: tuple[float, ...] = (0.25, 0.5),
    alpha_values: tuple[float, ...] = (0.05, 0.1, 0.25, 0.5, 1.0, 2.0),
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    prior_validation = rna_association_prior_scores(
        train_positive_matrix, validation_pairs
    )
    prior_test = rna_association_prior_scores(train_positive_matrix, test_pairs)
    best_ap = drug_macro_average_precision(
        validation_labels, prior_validation, validation_pairs[:, 1]
    )
    best_validation = prior_validation
    best_test = prior_test
    best_parameters = (0, 0, 0.0, 0.0)
    for rank in sorted(rank_values):
        for min_samples_leaf in sorted(leaf_values, reverse=True):
            for max_features in sorted(feature_values):
                validation_drugs, validation_inverse = np.unique(
                    validation_pairs[:, 1], return_inverse=True
                )
                test_drugs, test_inverse = np.unique(
                    test_pairs[:, 1], return_inverse=True
                )
                all_drugs = np.concatenate([validation_drugs, test_drugs])
                prior, all_residual = _extra_trees_latent_residual_profiles(
                    fingerprints,
                    train_positive_matrix,
                    all_drugs,
                    rank,
                    min_samples_leaf,
                    max_features,
                    seed,
                )
                validation_residual = all_residual[: len(validation_drugs)]
                test_residual = all_residual[len(validation_drugs) :]
                for alpha in sorted(alpha_values):
                    validation_profiles = np.clip(
                        prior[None, :] + alpha * validation_residual, 0.0, 1.0
                    )
                    validation = validation_profiles[
                        validation_inverse, validation_pairs[:, 0]
                    ]
                    macro_ap = drug_macro_average_precision(
                        validation_labels, validation, validation_pairs[:, 1]
                    )
                    if macro_ap > best_ap + 1e-12:
                        best_ap = macro_ap
                        best_validation = validation
                        test_profiles = np.clip(
                            prior[None, :] + alpha * test_residual, 0.0, 1.0
                        )
                        best_test = test_profiles[test_inverse, test_pairs[:, 0]]
                        best_parameters = (
                            rank,
                            min_samples_leaf,
                            max_features,
                            alpha,
                        )
    rank, min_samples_leaf, max_features, alpha = best_parameters
    return best_validation, best_test, {
        "rank": float(rank),
        "min_samples_leaf": float(min_samples_leaf),
        "max_features": float(max_features),
        "alpha": float(alpha),
        "validation_macro_ap": float(best_ap),
    }


def _substructure_enrichment_log_likelihood_ratio(
    fingerprints: np.ndarray,
    train_positive_matrix: np.ndarray,
    query_drugs: np.ndarray,
    prior_strength: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    active_drugs = np.flatnonzero(train_positive_matrix.sum(axis=0) > 0)
    if len(active_drugs) == 0:
        raise ValueError("训练正例图没有活跃药物")
    train_x = fingerprints[active_drugs].astype(np.float64)
    row_degree = train_positive_matrix[:, active_drugs].sum(axis=1).astype(np.float64)
    prior = (row_degree + 1.0) / (len(active_drugs) + 2.0)
    global_bit = (train_x.sum(axis=0) + 1.0) / (len(active_drugs) + 2.0)
    positive_bit_count = train_positive_matrix[:, active_drugs] @ train_x
    theta = (
        positive_bit_count + prior_strength * global_bit[None, :]
    ) / (row_degree[:, None] + prior_strength)
    epsilon = 1e-6
    theta = np.clip(theta, epsilon, 1.0 - epsilon)
    global_bit = np.clip(global_bit, epsilon, 1.0 - epsilon)
    present = np.log(theta) - np.log(global_bit[None, :])
    absent = np.log1p(-theta) - np.log1p(-global_bit[None, :])
    query_x = fingerprints[query_drugs].astype(np.float64)
    log_likelihood_ratio = (
        query_x @ (present - absent).T + absent.sum(axis=1)[None, :]
    ) / math.sqrt(fingerprints.shape[1])
    return prior, row_degree, log_likelihood_ratio


def substructure_enrichment_scores(
    fingerprints: np.ndarray,
    train_positive_matrix: np.ndarray,
    pairs: np.ndarray,
    prior_strength: float,
    degree_tau: float,
    beta: float,
) -> np.ndarray:
    """以 RNA 特异 ECFP Bernoulli 富集似然比校正全局先验。"""
    active_drugs = np.flatnonzero(train_positive_matrix.sum(axis=0) > 0)
    row_degree = train_positive_matrix[:, active_drugs].sum(axis=1).astype(np.float64)
    prior = (row_degree + 1.0) / (len(active_drugs) + 2.0)
    if beta == 0:
        return prior[pairs[:, 0]]
    query_drugs, inverse = np.unique(pairs[:, 1], return_inverse=True)
    prior, row_degree, log_likelihood_ratio = (
        _substructure_enrichment_log_likelihood_ratio(
            fingerprints, train_positive_matrix, query_drugs, prior_strength
        )
    )
    reliability = row_degree / (row_degree + degree_tau)
    prior_logit = np.log(prior) - np.log1p(-prior)
    logits = prior_logit[None, :] + beta * reliability[None, :] * log_likelihood_ratio
    profiles = 1.0 / (1.0 + np.exp(-np.clip(logits, -30.0, 30.0)))
    return profiles[inverse, pairs[:, 0]]


def select_substructure_enrichment(
    fingerprints: np.ndarray,
    train_positive_matrix: np.ndarray,
    validation_pairs: np.ndarray,
    validation_labels: np.ndarray,
    test_pairs: np.ndarray,
    prior_strength_values: tuple[float, ...] = (20.0, 10.0, 5.0, 2.0),
    degree_tau_values: tuple[float, ...] = (20.0, 5.0, 1.0),
    beta_values: tuple[float, ...] = (0.1, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0),
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    prior_validation = rna_association_prior_scores(train_positive_matrix, validation_pairs)
    prior_test = rna_association_prior_scores(train_positive_matrix, test_pairs)
    best_ap = drug_macro_average_precision(
        validation_labels, prior_validation, validation_pairs[:, 1]
    )
    best_validation, best_test = prior_validation, prior_test
    best_parameters = (math.inf, math.inf, 0.0)
    for prior_strength in prior_strength_values:
        validation_drugs, validation_inverse = np.unique(
            validation_pairs[:, 1], return_inverse=True
        )
        test_drugs, test_inverse = np.unique(test_pairs[:, 1], return_inverse=True)
        all_drugs = np.concatenate([validation_drugs, test_drugs])
        prior, row_degree, all_llr = _substructure_enrichment_log_likelihood_ratio(
            fingerprints, train_positive_matrix, all_drugs, prior_strength
        )
        validation_llr = all_llr[: len(validation_drugs)]
        test_llr = all_llr[len(validation_drugs) :]
        prior_logit = np.log(prior) - np.log1p(-prior)
        for degree_tau in degree_tau_values:
            reliability = row_degree / (row_degree + degree_tau)
            for beta in sorted(beta_values):
                validation_profiles = 1.0 / (
                    1.0
                    + np.exp(
                        -np.clip(
                            prior_logit[None, :]
                            + beta * reliability[None, :] * validation_llr,
                            -30.0,
                            30.0,
                        )
                    )
                )
                validation = validation_profiles[
                    validation_inverse, validation_pairs[:, 0]
                ]
                macro_ap = drug_macro_average_precision(
                    validation_labels, validation, validation_pairs[:, 1]
                )
                if macro_ap > best_ap + 1e-12:
                    best_ap = macro_ap
                    best_validation = validation
                    test_profiles = 1.0 / (
                        1.0
                        + np.exp(
                            -np.clip(
                                prior_logit[None, :]
                                + beta * reliability[None, :] * test_llr,
                                -30.0,
                                30.0,
                            )
                        )
                    )
                    best_test = test_profiles[test_inverse, test_pairs[:, 0]]
                    best_parameters = (prior_strength, degree_tau, beta)
    prior_strength, degree_tau, beta = best_parameters
    return best_validation, best_test, {
        "prior_strength": float(prior_strength),
        "degree_tau": float(degree_tau),
        "beta": float(beta),
        "validation_macro_ap": float(best_ap),
    }


def agreement_gated_prior_consensus(
    prior_probability: np.ndarray,
    first_probability: np.ndarray,
    second_probability: np.ndarray,
) -> np.ndarray:
    """仅在两个专家相对 prior 的 logit 修正同号时取平均。"""
    epsilon = 1e-6
    prior = np.clip(np.asarray(prior_probability, dtype=np.float64), epsilon, 1 - epsilon)
    first = np.clip(np.asarray(first_probability, dtype=np.float64), epsilon, 1 - epsilon)
    second = np.clip(np.asarray(second_probability, dtype=np.float64), epsilon, 1 - epsilon)
    if prior.shape != first.shape or prior.shape != second.shape:
        raise ValueError("prior 与专家概率形状必须一致")
    logit = lambda value: np.log(value) - np.log1p(-value)
    prior_logit = logit(prior)
    first_residual = logit(first) - prior_logit
    second_residual = logit(second) - prior_logit
    consensus = np.where(
        first_residual * second_residual > 0,
        0.5 * (first_residual + second_residual),
        0.0,
    )
    combined = prior_logit + consensus
    return 1.0 / (1.0 + np.exp(-np.clip(combined, -30.0, 30.0)))


def prototype_residual_profiles(
    predicted_similarity: np.ndarray,
    train_residual_profiles: np.ndarray,
    k: int,
    kappa: float,
) -> np.ndarray:
    """以正预测关联度聚合 top-k 训练原型，并向零残差收缩。"""
    if k <= 0 or kappa <= 0:
        raise ValueError("k 与 kappa 必须为正")
    if predicted_similarity.shape[1] != train_residual_profiles.shape[0]:
        raise ValueError("预测相似度与训练原型数量不一致")
    effective_k = min(k, predicted_similarity.shape[1])
    order = np.argsort(-predicted_similarity, axis=1, kind="stable")[
        :, :effective_k
    ]
    output = np.zeros(
        (predicted_similarity.shape[0], train_residual_profiles.shape[1]),
        dtype=np.float64,
    )
    for query_index, positions in enumerate(order):
        weights = np.maximum(predicted_similarity[query_index, positions], 0.0)
        if weights.sum() <= 0:
            continue
        output[query_index] = (
            weights @ train_residual_profiles[positions]
        ) / (kappa + weights.sum())
    return output


def select_association_aware_prototype_transfer(
    side_kernels: tuple[np.ndarray, ...],
    target_coverage: np.ndarray,
    train_positive_matrix: np.ndarray,
    validation_pairs: np.ndarray,
    validation_labels: np.ndarray,
    test_pairs: np.ndarray,
    seed: int,
    k_values: tuple[int, ...] = (5, 10, 20, 50),
    kappa_values: tuple[float, ...] = (1.0, 5.0, 10.0, 20.0),
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    """学习 side-view 到训练关联原型相似度的映射，再迁移残差谱。"""
    drug_count = train_positive_matrix.shape[1]
    if not side_kernels or any(
        kernel.shape != (drug_count, drug_count) for kernel in side_kernels
    ):
        raise ValueError("所有 side kernel 都必须与药物维度一致")
    coverage = np.asarray(target_coverage, dtype=bool)
    if coverage.shape != (drug_count,):
        raise ValueError("target_coverage 与药物维度不一致")
    active_drugs = np.flatnonzero(train_positive_matrix.sum(axis=0) > 0)
    if len(active_drugs) < 2:
        raise ValueError("关联度回归至少需要两个活跃训练药物")
    row_degree = train_positive_matrix.sum(axis=1).astype(np.float64)
    global_prior = (row_degree + 1.0) / (len(active_drugs) + 2.0)
    train_residual = (
        train_positive_matrix[:, active_drugs].T.astype(np.float64)
        - global_prior[None, :]
    )
    residual_norm = np.linalg.norm(train_residual, axis=1, keepdims=True)
    normalized_residual = train_residual / np.maximum(residual_norm, 1e-12)
    target_similarity = normalized_residual @ normalized_residual.T
    left, right = np.triu_indices(len(active_drugs), k=1)
    pair_features = [
        kernel[np.ix_(active_drugs, active_drugs)][left, right]
        for kernel in side_kernels
    ]
    both_target = (
        coverage[active_drugs[left]] & coverage[active_drugs[right]]
    ).astype(np.float64)
    train_features = np.column_stack([*pair_features, both_target])
    regressor = HistGradientBoostingRegressor(
        max_iter=100,
        max_depth=3,
        learning_rate=0.05,
        l2_regularization=1.0,
        random_state=seed,
    ).fit(train_features, target_similarity[left, right])

    def predicted_similarity(query_drugs: np.ndarray) -> np.ndarray:
        query_features = [
            kernel[np.ix_(query_drugs, active_drugs)].reshape(-1)
            for kernel in side_kernels
        ]
        query_both_target = (
            coverage[query_drugs, None] & coverage[active_drugs][None, :]
        ).astype(np.float64).reshape(-1)
        features = np.column_stack([*query_features, query_both_target])
        return regressor.predict(features).reshape(len(query_drugs), len(active_drugs))

    validation_drugs, validation_inverse = np.unique(
        validation_pairs[:, 1], return_inverse=True
    )
    test_drugs, test_inverse = np.unique(test_pairs[:, 1], return_inverse=True)
    validation_similarity = predicted_similarity(validation_drugs)
    test_similarity = predicted_similarity(test_drugs)
    prior_validation = global_prior[validation_pairs[:, 0]]
    prior_test = global_prior[test_pairs[:, 0]]
    best_ap = drug_macro_average_precision(
        validation_labels, prior_validation, validation_pairs[:, 1]
    )
    best_validation = prior_validation
    best_test = prior_test
    best_k = 0
    best_kappa = math.inf
    for k in sorted(k_values):
        for kappa in sorted(kappa_values, reverse=True):
            validation_profiles = prototype_residual_profiles(
                validation_similarity, train_residual, k, kappa
            )
            validation = np.clip(
                prior_validation
                + validation_profiles[validation_inverse, validation_pairs[:, 0]],
                0.0,
                1.0,
            )
            macro_ap = drug_macro_average_precision(
                validation_labels, validation, validation_pairs[:, 1]
            )
            if macro_ap > best_ap + 1e-12:
                best_ap = macro_ap
                best_validation = validation
                test_profiles = prototype_residual_profiles(
                    test_similarity, train_residual, k, kappa
                )
                best_test = np.clip(
                    prior_test + test_profiles[test_inverse, test_pairs[:, 0]],
                    0.0,
                    1.0,
                )
                best_k = k
                best_kappa = kappa
    return best_validation, best_test, {
        "k": float(best_k),
        "kappa": float(best_kappa),
        "validation_macro_ap": float(best_ap),
        "metric_train_pair_count": float(len(left)),
    }


def _partition_from_codes(pair_new: np.ndarray, positive_code: int, negative_code: int) -> PairPartition:
    positive = np.argwhere(pair_new == positive_code)
    negative = np.argwhere(pair_new == negative_code)
    pairs = np.concatenate([positive, negative], axis=0).astype(np.int64)
    labels = np.concatenate(
        [np.ones(len(positive), dtype=np.float32), np.zeros(len(negative), dtype=np.float32)]
    )
    return PairPartition(pairs, labels)


def published_pair_split(bundle: DatasetBundle) -> ProtocolSplit:
    train = _partition_from_codes(bundle.pair_new, 1, -20)
    val = _partition_from_codes(bundle.pair_new, -1, -10)
    test = _partition_from_codes(bundle.pair_new, 2, -30)
    graph = np.zeros_like(bundle.pair, dtype=np.float32)
    train_positive = train.pairs[train.labels == 1]
    graph[train_positive[:, 0], train_positive[:, 1]] = 1.0
    return ProtocolSplit(train, val, test, graph)


def _degree_strata(degrees: np.ndarray, max_strata: int = 5) -> np.ndarray:
    """按度数秩分层，避免重复度数导致 qcut 退化。"""
    n = len(degrees)
    n_strata = max(1, min(max_strata, n // 5))
    order = np.argsort(degrees, kind="stable")
    strata = np.empty(n, dtype=np.int64)
    strata[order] = np.minimum(np.arange(n) * n_strata // n, n_strata - 1)
    return strata


def _entity_assignment(pair: np.ndarray, axis: int, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    degrees = pair.sum(axis=1 if axis == 0 else 0)
    active = np.flatnonzero(degrees > 0)
    strata = _degree_strata(degrees[active])
    train_val, test, strata_train_val, _ = train_test_split(
        active,
        strata,
        test_size=0.20,
        random_state=seed,
        stratify=strata,
    )
    train, val = train_test_split(
        train_val,
        test_size=0.25,
        random_state=seed + 10_000,
        stratify=strata_train_val,
    )
    return np.sort(train), np.sort(val), np.sort(test)


def _entity_partition(pair: np.ndarray, entity_ids: np.ndarray, axis: int, rng: np.random.Generator) -> PairPartition:
    mask = np.zeros_like(pair, dtype=bool)
    if axis == 0:
        mask[entity_ids, :] = True
    else:
        mask[:, entity_ids] = True
    positive = np.argwhere(mask & (pair == 1))
    unknown = np.argwhere(mask & (pair == 0))
    if len(unknown) < len(positive):
        raise ValueError("未知 pair 数少于正例数，无法构造平衡 sampled-unlabeled split")
    sampled_unknown = unknown[rng.choice(len(unknown), size=len(positive), replace=False)]
    pairs = np.concatenate([positive, sampled_unknown], axis=0).astype(np.int64)
    labels = np.concatenate(
        [np.ones(len(positive), dtype=np.float32), np.zeros(len(sampled_unknown), dtype=np.float32)]
    )
    permutation = rng.permutation(len(labels))
    return PairPartition(pairs[permutation], labels[permutation])


def cold_entity_split(bundle: DatasetBundle, axis: int, seed: int) -> ProtocolSplit:
    if axis not in {0, 1}:
        raise ValueError("axis=0 表示 RNA cold，axis=1 表示 drug cold")
    train_ids, val_ids, test_ids = _entity_assignment(bundle.pair, axis, seed)
    rng = np.random.default_rng(seed)
    train = _entity_partition(bundle.pair, train_ids, axis, rng)
    val = _entity_partition(bundle.pair, val_ids, axis, rng)
    test = _entity_partition(bundle.pair, test_ids, axis, rng)
    graph = np.zeros_like(bundle.pair, dtype=np.float32)
    train_positive = train.pairs[train.labels == 1]
    graph[train_positive[:, 0], train_positive[:, 1]] = 1.0
    return ProtocolSplit(train, val, test, graph)


def _scaffold_assignment(
    pair: np.ndarray, drug_scaffolds: np.ndarray, seed: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if len(drug_scaffolds) != pair.shape[1]:
        raise ValueError("scaffold 数量与药物轴不一致")
    drug_degree = pair.sum(axis=0)
    active_drugs = np.flatnonzero(drug_degree > 0)
    groups: dict[str, list[int]] = {}
    for drug_id in active_drugs:
        groups.setdefault(str(drug_scaffolds[drug_id]), []).append(int(drug_id))
    rng = np.random.default_rng(seed)
    group_items = []
    for scaffold, ids in groups.items():
        mass = float(drug_degree[ids].sum())
        group_items.append((scaffold, np.asarray(ids, dtype=np.int64), mass, float(rng.random())))
    group_items.sort(key=lambda item: (-item[2], item[3]))

    fractions = np.asarray([0.60, 0.20, 0.20])
    target_mass = max(float(drug_degree[active_drugs].sum()), 1.0) * fractions
    target_count = max(float(len(active_drugs)), 1.0) * fractions
    current_mass = np.zeros(3, dtype=np.float64)
    current_count = np.zeros(3, dtype=np.float64)
    assignments: list[list[np.ndarray]] = [[], [], []]
    for _, ids, mass, _ in group_items:
        costs = []
        for partition in range(3):
            candidate_mass = current_mass.copy()
            candidate_count = current_count.copy()
            candidate_mass[partition] += mass
            candidate_count[partition] += len(ids)
            mass_cost = np.square((candidate_mass - target_mass) / target_mass).sum()
            count_cost = np.square((candidate_count - target_count) / target_count).sum()
            costs.append(mass_cost + 0.15 * count_cost)
        chosen = int(np.argmin(costs))
        assignments[chosen].append(ids)
        current_mass[chosen] += mass
        current_count[chosen] += len(ids)
    if any(not assignment for assignment in assignments):
        raise ValueError("scaffold group 太少，无法形成 train/validation/test")
    return tuple(np.sort(np.concatenate(assignment)) for assignment in assignments)  # type: ignore[return-value]


def scaffold_cold_split(
    bundle: DatasetBundle, drug_scaffolds: np.ndarray, seed: int
) -> ProtocolSplit:
    train_ids, val_ids, test_ids = _scaffold_assignment(bundle.pair, drug_scaffolds, seed)
    rng = np.random.default_rng(seed)
    train = _entity_partition(bundle.pair, train_ids, axis=1, rng=rng)
    val = _entity_partition(bundle.pair, val_ids, axis=1, rng=rng)
    test = _entity_partition(bundle.pair, test_ids, axis=1, rng=rng)
    graph = np.zeros_like(bundle.pair, dtype=np.float32)
    train_positive = train.pairs[train.labels == 1]
    graph[train_positive[:, 0], train_positive[:, 1]] = 1.0
    return ProtocolSplit(train, val, test, graph)


def scaffold_fold_assignment(
    pair: np.ndarray,
    drug_scaffolds: np.ndarray,
    n_folds: int = 5,
    seed: int = 20261019,
) -> tuple[np.ndarray, ...]:
    """把完整 scaffold 组分配到互斥折，并同时平衡正例质量和药物数。"""
    if n_folds < 3:
        raise ValueError("scaffold CV 至少需要 3 折，才能分离 train/validation/test")
    if len(drug_scaffolds) != pair.shape[1]:
        raise ValueError("scaffold 数量与药物轴不一致")
    drug_degree = pair.sum(axis=0)
    active_drugs = np.flatnonzero(drug_degree > 0)
    groups: dict[str, list[int]] = {}
    for drug_id in active_drugs:
        groups.setdefault(str(drug_scaffolds[drug_id]), []).append(int(drug_id))
    if len(groups) < n_folds:
        raise ValueError("scaffold group 少于折数")

    rng = np.random.default_rng(seed)
    group_items = []
    for scaffold, ids in groups.items():
        ids_array = np.asarray(ids, dtype=np.int64)
        group_items.append(
            (
                scaffold,
                ids_array,
                float(drug_degree[ids_array].sum()),
                float(rng.random()),
            )
        )
    group_items.sort(key=lambda item: (-item[2], item[3]))

    target_mass = max(float(drug_degree[active_drugs].sum()), 1.0) / n_folds
    target_count = max(float(len(active_drugs)), 1.0) / n_folds
    current_mass = np.zeros(n_folds, dtype=np.float64)
    current_count = np.zeros(n_folds, dtype=np.float64)
    assignments: list[list[np.ndarray]] = [[] for _ in range(n_folds)]
    for _, ids, mass, _ in group_items:
        costs = []
        for fold in range(n_folds):
            candidate_mass = current_mass.copy()
            candidate_count = current_count.copy()
            candidate_mass[fold] += mass
            candidate_count[fold] += len(ids)
            mass_cost = np.square((candidate_mass - target_mass) / target_mass).sum()
            count_cost = np.square((candidate_count - target_count) / target_count).sum()
            costs.append(mass_cost + 0.15 * count_cost)
        chosen = int(np.argmin(costs))
        assignments[chosen].append(ids)
        current_mass[chosen] += mass
        current_count[chosen] += len(ids)
    if any(not assignment for assignment in assignments):
        raise ValueError("至少一个 scaffold fold 为空")
    return tuple(np.sort(np.concatenate(assignment)) for assignment in assignments)


def scaffold_cv_split(
    bundle: DatasetBundle,
    drug_scaffolds: np.ndarray,
    fold: int,
    n_folds: int = 5,
    seed: int = 20261019,
) -> ProtocolSplit:
    folds = scaffold_fold_assignment(bundle.pair, drug_scaffolds, n_folds=n_folds, seed=seed)
    if fold < 0 or fold >= n_folds:
        raise ValueError(f"fold 必须位于 [0, {n_folds})")
    test_ids = folds[fold]
    val_ids = folds[(fold + 1) % n_folds]
    train_ids = np.sort(
        np.concatenate([folds[index] for index in range(n_folds) if index not in {fold, (fold + 1) % n_folds}])
    )
    rng = np.random.default_rng(seed + fold)
    train = _entity_partition(bundle.pair, train_ids, axis=1, rng=rng)
    val = _entity_partition(bundle.pair, val_ids, axis=1, rng=rng)
    test = _entity_partition(bundle.pair, test_ids, axis=1, rng=rng)
    graph = np.zeros_like(bundle.pair, dtype=np.float32)
    train_positive = train.pairs[train.labels == 1]
    graph[train_positive[:, 0], train_positive[:, 1]] = 1.0
    return ProtocolSplit(train, val, test, graph)


def assert_split_integrity(split: ProtocolSplit, cold_axis: int | None = None) -> None:
    pair_sets = [set(map(tuple, part.pairs.tolist())) for part in (split.train, split.val, split.test)]
    if pair_sets[0] & pair_sets[1] or pair_sets[0] & pair_sets[2] or pair_sets[1] & pair_sets[2]:
        raise AssertionError("train/val/test pair 有重叠")
    for part in (split.train, split.val, split.test):
        if set(np.unique(part.labels)) != {0.0, 1.0}:
            raise AssertionError("每个 split 必须同时包含正例和 sampled-unlabeled pair")
    if cold_axis is not None:
        entities = [set(part.pairs[:, cold_axis].tolist()) for part in (split.train, split.val, split.test)]
        if entities[0] & entities[1] or entities[0] & entities[2] or entities[1] & entities[2]:
            raise AssertionError("cold split 发生实体泄漏")


class TwoTowerSemantic(nn.Module):
    def __init__(self, rna_dim: int, drug_dim: int, latent_dim: int = 48, dropout: float = 0.15):
        super().__init__()
        self.rna_tower = nn.Sequential(nn.Linear(rna_dim, latent_dim), nn.ReLU(), nn.Dropout(dropout))
        self.drug_tower = nn.Sequential(nn.Linear(drug_dim, latent_dim), nn.ReLU(), nn.Dropout(dropout))
        self.head = nn.Sequential(
            nn.Linear(4 * latent_dim, 64), nn.ReLU(), nn.Dropout(dropout), nn.Linear(64, 1)
        )

    def forward(self, rna: torch.Tensor, drug: torch.Tensor) -> torch.Tensor:
        zr = self.rna_tower(rna)
        zd = self.drug_tower(drug)
        joint = torch.cat([zr, zd, torch.abs(zr - zd), zr * zd], dim=1)
        return self.head(joint).squeeze(1)


class ContrastiveTwoTower(nn.Module):
    """用共享余弦空间约束未见实体打分，避免任意 pair head。"""

    def __init__(self, rna_dim: int, drug_dim: int, latent_dim: int = 48, dropout: float = 0.15):
        super().__init__()
        self.rna_tower = nn.Sequential(nn.Linear(rna_dim, latent_dim), nn.ReLU(), nn.Dropout(dropout))
        self.drug_tower = nn.Sequential(nn.Linear(drug_dim, latent_dim), nn.ReLU(), nn.Dropout(dropout))
        self.logit_scale = nn.Parameter(torch.tensor(math.log(1.0 / 0.07)))

    def encode_rna(self, rna: torch.Tensor) -> torch.Tensor:
        return nn.functional.normalize(self.rna_tower(rna), dim=1)

    def encode_drug(self, drug: torch.Tensor) -> torch.Tensor:
        return nn.functional.normalize(self.drug_tower(drug), dim=1)

    def similarity(self, rna: torch.Tensor, drug: torch.Tensor) -> torch.Tensor:
        scale = self.logit_scale.exp().clamp(max=100.0)
        return scale * (self.encode_rna(rna) @ self.encode_drug(drug).T)

    def forward(self, rna: torch.Tensor, drug: torch.Tensor) -> torch.Tensor:
        scale = self.logit_scale.exp().clamp(max=100.0)
        return scale * (self.encode_rna(rna) * self.encode_drug(drug)).sum(dim=1)


class RNAIdentityDrugInteraction(nn.Module):
    """warm RNA 自由潜因子与 cold-drug 特征映射的缩放点积。"""

    def __init__(
        self, rna_count: int, drug_dim: int, latent_dim: int = 64, dropout: float = 0.15
    ):
        super().__init__()
        self.rna_embedding = nn.Embedding(rna_count, latent_dim)
        nn.init.normal_(self.rna_embedding.weight, std=0.02)
        self.drug_tower = nn.Sequential(
            nn.Linear(drug_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, latent_dim),
        )
        self.logit_scale = nn.Parameter(torch.tensor(0.0))

    def forward(self, rna_ids: torch.Tensor, drug: torch.Tensor) -> torch.Tensor:
        rna = self.rna_embedding(rna_ids)
        drug_latent = self.drug_tower(drug)
        scale = self.logit_scale.exp().clamp(max=100.0) / math.sqrt(rna.shape[1])
        return scale * (rna * drug_latent).sum(dim=1)

    def similarity(self, drug: torch.Tensor) -> torch.Tensor:
        drug_latent = self.drug_tower(drug)
        scale = self.logit_scale.exp().clamp(max=100.0) / math.sqrt(
            self.rna_embedding.embedding_dim
        )
        return scale * (self.rna_embedding.weight @ drug_latent.T)


class TwoTowerWithPropensity(nn.Module):
    def __init__(self, rna_dim: int, drug_dim: int, latent_dim: int = 48, dropout: float = 0.15):
        super().__init__()
        self.rna_tower = nn.Sequential(nn.Linear(rna_dim, latent_dim), nn.ReLU(), nn.Dropout(dropout))
        self.drug_tower = nn.Sequential(nn.Linear(drug_dim, latent_dim), nn.ReLU(), nn.Dropout(dropout))
        self.head = nn.Sequential(
            nn.Linear(4 * latent_dim + 2, 64), nn.ReLU(), nn.Dropout(dropout), nn.Linear(64, 1)
        )

    def forward(
        self, rna: torch.Tensor, drug: torch.Tensor, propensity: torch.Tensor
    ) -> torch.Tensor:
        zr = self.rna_tower(rna)
        zd = self.drug_tower(drug)
        joint = torch.cat([zr, zd, torch.abs(zr - zd), zr * zd, propensity], dim=1)
        return self.head(joint).squeeze(1)


class BilinearTwoTower(nn.Module):
    def __init__(self, rna_dim: int, drug_dim: int, latent_dim: int = 48, dropout: float = 0.15):
        super().__init__()
        self.rna_tower = nn.Sequential(nn.Linear(rna_dim, latent_dim), nn.ReLU(), nn.Dropout(dropout))
        self.drug_tower = nn.Sequential(nn.Linear(drug_dim, latent_dim), nn.ReLU(), nn.Dropout(dropout))
        self.logit_scale = nn.Parameter(torch.tensor(0.0))
        self.bias = nn.Parameter(torch.tensor(0.0))

    def forward(self, rna: torch.Tensor, drug: torch.Tensor) -> torch.Tensor:
        zr = self.rna_tower(rna)
        zd = self.drug_tower(drug)
        scale = self.logit_scale.exp().clamp(max=100.0) / math.sqrt(zr.shape[1])
        return scale * (zr * zd).sum(dim=1) + self.bias


def active_positive_subgraph(train_positive_matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """只返回训练正例图中活跃的实体及其二部邻接掩码。"""
    rna_ids = np.flatnonzero(train_positive_matrix.sum(axis=1) > 0)
    drug_ids = np.flatnonzero(train_positive_matrix.sum(axis=0) > 0)
    if len(rna_ids) == 0 or len(drug_ids) == 0:
        raise ValueError("训练正例图没有可用于对比学习的活跃实体")
    positive_mask = train_positive_matrix[np.ix_(rna_ids, drug_ids)] > 0
    return rna_ids, drug_ids, positive_mask


def entity_balanced_multi_positive_loss(
    logits: torch.Tensor, positive_mask: torch.Tensor
) -> torch.Tensor:
    """在 RNA 行和药物列两个方向分别按实体等权平均。"""
    if logits.shape != positive_mask.shape:
        raise ValueError("logits 与 positive_mask 形状必须一致")
    mask = positive_mask.to(dtype=logits.dtype)
    row_count = mask.sum(dim=1)
    col_count = mask.sum(dim=0)
    if torch.any(row_count == 0) or torch.any(col_count == 0):
        raise ValueError("对比子图的每个实体都必须至少有一个正例")
    row_log_probability = nn.functional.log_softmax(logits, dim=1)
    col_log_probability = nn.functional.log_softmax(logits, dim=0)
    row_loss = -((row_log_probability * mask).sum(dim=1) / row_count).mean()
    col_loss = -((col_log_probability * mask).sum(dim=0) / col_count).mean()
    return 0.5 * (row_loss + col_loss)


def drug_listwise_multi_positive_loss(
    logits: torch.Tensor, positive_mask: torch.Tensor
) -> torch.Tensor:
    """对每个药物的完整 RNA 候选列表计算多正例 softmax 损失。"""
    if logits.shape != positive_mask.shape:
        raise ValueError("logits 与 positive_mask 形状必须一致")
    mask = positive_mask.to(dtype=logits.dtype)
    positive_count = mask.sum(dim=0)
    if torch.any(positive_count == 0):
        raise ValueError("每个训练药物都必须至少有一个正例")
    log_probability = nn.functional.log_softmax(logits, dim=0)
    return -((log_probability * mask).sum(dim=0) / positive_count).mean()


def sample_within_drug_pairs(
    train_positive_matrix: np.ndarray, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    """为每个训练正例采样同药物的一个未观察 RNA。"""
    rng = np.random.default_rng(seed)
    positive_pairs = []
    unknown_pairs = []
    for drug_id in np.flatnonzero(train_positive_matrix.sum(axis=0) > 0):
        positive_rnas = np.flatnonzero(train_positive_matrix[:, drug_id] > 0)
        unknown_rnas = np.flatnonzero(train_positive_matrix[:, drug_id] == 0)
        if len(unknown_rnas) == 0:
            continue
        sampled = rng.choice(
            unknown_rnas,
            size=len(positive_rnas),
            replace=len(unknown_rnas) < len(positive_rnas),
        )
        positive_pairs.extend((int(rna_id), int(drug_id)) for rna_id in positive_rnas)
        unknown_pairs.extend((int(rna_id), int(drug_id)) for rna_id in sampled)
    if not positive_pairs:
        raise ValueError("训练图没有可构造成对排序的正例")
    return np.asarray(positive_pairs, dtype=np.int64), np.asarray(unknown_pairs, dtype=np.int64)


def sample_propensity_matched_within_drug_pairs(
    train_positive_matrix: np.ndarray,
    seed: int,
    pool_size: int = 32,
    negatives_per_positive: int = 4,
) -> tuple[np.ndarray, np.ndarray]:
    """从 RNA 度数最接近的未知候选池采样药物内硬负例。"""
    if pool_size <= 0 or negatives_per_positive <= 0:
        raise ValueError("pool_size 与 negatives_per_positive 必须为正")
    rng = np.random.default_rng(seed)
    rna_propensity = np.log1p(train_positive_matrix.sum(axis=1).astype(np.float64))
    positive_pairs = []
    unknown_pairs = []
    for drug_id in np.flatnonzero(train_positive_matrix.sum(axis=0) > 0):
        positive_rnas = np.flatnonzero(train_positive_matrix[:, drug_id] > 0)
        unknown_rnas = np.flatnonzero(train_positive_matrix[:, drug_id] == 0)
        if len(unknown_rnas) == 0:
            continue
        unknown_propensity = rna_propensity[unknown_rnas]
        for positive_rna in positive_rnas:
            distance = np.abs(unknown_propensity - rna_propensity[positive_rna])
            order = np.argsort(distance, kind="stable")
            pool = unknown_rnas[order[: min(pool_size, len(order))]]
            sampled = rng.choice(
                pool,
                size=negatives_per_positive,
                replace=len(pool) < negatives_per_positive,
            )
            positive_pairs.extend(
                (int(positive_rna), int(drug_id))
                for _ in range(negatives_per_positive)
            )
            unknown_pairs.extend(
                (int(unknown_rna), int(drug_id)) for unknown_rna in sampled
            )
    if not positive_pairs:
        raise ValueError("训练图没有可构造成对排序的正例")
    return np.asarray(positive_pairs, dtype=np.int64), np.asarray(unknown_pairs, dtype=np.int64)


def sample_prior_violating_within_drug_pairs(
    train_positive_matrix: np.ndarray,
    seed: int,
    pool_size: int = 64,
    negatives_per_positive: int = 8,
) -> tuple[np.ndarray, np.ndarray]:
    """优先采样在 leave-one-out RNA prior 下不低于正例的未知项。"""
    if pool_size <= 0 or negatives_per_positive <= 0:
        raise ValueError("pool_size 与 negatives_per_positive 必须为正")
    rng = np.random.default_rng(seed)
    row_degree = train_positive_matrix.sum(axis=1).astype(np.int64)
    positive_pairs = []
    unknown_pairs = []
    for drug_id in np.flatnonzero(train_positive_matrix.sum(axis=0) > 0):
        positive_rnas = np.flatnonzero(train_positive_matrix[:, drug_id] > 0)
        unknown_rnas = np.flatnonzero(train_positive_matrix[:, drug_id] == 0)
        if len(unknown_rnas) == 0:
            continue
        order = np.argsort(-row_degree[unknown_rnas], kind="stable")
        ranked_unknown = unknown_rnas[order]
        for positive_rna in positive_rnas:
            positive_loo_degree = row_degree[positive_rna] - 1
            violating = ranked_unknown[
                row_degree[ranked_unknown] >= positive_loo_degree
            ]
            candidate = violating if len(violating) else ranked_unknown
            pool = candidate[: min(pool_size, len(candidate))]
            sampled = rng.choice(
                pool,
                size=negatives_per_positive,
                replace=len(pool) < negatives_per_positive,
            )
            positive_pairs.extend(
                (int(positive_rna), int(drug_id))
                for _ in range(negatives_per_positive)
            )
            unknown_pairs.extend(
                (int(unknown_rna), int(drug_id)) for unknown_rna in sampled
            )
    if not positive_pairs:
        raise ValueError("训练图没有可构造成对排序的正例")
    return np.asarray(positive_pairs, dtype=np.int64), np.asarray(unknown_pairs, dtype=np.int64)


def entity_balanced_full_candidate_weights(labels_by_drug: np.ndarray) -> np.ndarray:
    """令每个药物等权，并在药物内部让正例与零项各占一半权重。"""
    if labels_by_drug.ndim != 2:
        raise ValueError("labels_by_drug 必须是二维药物×RNA矩阵")
    positive_count = labels_by_drug.sum(axis=1, keepdims=True)
    negative_count = labels_by_drug.shape[1] - positive_count
    if np.any(positive_count <= 0) or np.any(negative_count <= 0):
        raise ValueError("每个训练药物必须同时包含正例和零项")
    return np.where(
        labels_by_drug > 0,
        0.5 / positive_count,
        0.5 / negative_count,
    ).astype(np.float32)


def drug_macro_average_precision(
    labels: np.ndarray, scores: np.ndarray, drug_ids: np.ndarray
) -> float:
    values = []
    for drug_id in np.unique(drug_ids):
        include = drug_ids == drug_id
        if labels[include].sum() == 0:
            continue
        values.append(average_precision_score(labels[include], scores[include]))
    if not values:
        raise ValueError("没有可计算 average precision 的药物")
    return float(np.mean(values))


def select_residual_scale(
    labels: np.ndarray,
    prior_logits: np.ndarray,
    residual_logits: np.ndarray,
    drug_ids: np.ndarray,
    scales: tuple[float, ...] = (0.0, 0.05, 0.1, 0.25, 0.5, 1.0),
) -> tuple[float, float]:
    """只用验证药物的 full-candidate macro AP 选择残差缩放。"""
    best_scale = scales[0]
    best_ap = -math.inf
    for scale in sorted(scales):
        score = prior_logits + scale * residual_logits
        macro_ap = drug_macro_average_precision(labels, score, drug_ids)
        if macro_ap > best_ap + 1e-12:
            best_scale = scale
            best_ap = macro_ap
    return float(best_scale), float(best_ap)


class SemanticEstimator:
    def __init__(self, seed: int, max_epochs: int = 120, patience: int = 15):
        self.seed = seed
        self.max_epochs = max_epochs
        self.patience = patience
        self.model: TwoTowerSemantic | None = None
        self.rna_mean: np.ndarray | None = None
        self.rna_std: np.ndarray | None = None
        self.drug_mean: np.ndarray | None = None
        self.drug_std: np.ndarray | None = None
        self.best_epoch = 0

    @staticmethod
    def _stats(features: np.ndarray, ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        selected = features[np.unique(ids)]
        mean = selected.mean(axis=0)
        std = selected.std(axis=0)
        return mean, np.where(std < 1e-6, 1.0, std)

    def _pair_tensors(self, features_rna: np.ndarray, features_drug: np.ndarray, pairs: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
        rna = (features_rna[pairs[:, 0]] - self.rna_mean) / self.rna_std
        drug = (features_drug[pairs[:, 1]] - self.drug_mean) / self.drug_std
        return torch.from_numpy(rna.astype(np.float32)), torch.from_numpy(drug.astype(np.float32))

    def fit(self, bundle: DatasetBundle, split: ProtocolSplit) -> "SemanticEstimator":
        seed_everything(self.seed)
        self.rna_mean, self.rna_std = self._stats(bundle.rna_features, split.train.pairs[:, 0])
        self.drug_mean, self.drug_std = self._stats(bundle.drug_features, split.train.pairs[:, 1])
        self.model = TwoTowerSemantic(bundle.rna_features.shape[1], bundle.drug_features.shape[1])
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-3, weight_decay=1e-4)
        criterion = nn.BCEWithLogitsLoss()
        train_rna, train_drug = self._pair_tensors(bundle.rna_features, bundle.drug_features, split.train.pairs)
        train_labels = torch.from_numpy(split.train.labels.astype(np.float32))
        generator = torch.Generator().manual_seed(self.seed)
        loader = DataLoader(
            TensorDataset(train_rna, train_drug, train_labels),
            batch_size=min(256, len(train_labels)),
            shuffle=True,
            generator=generator,
        )
        best_ap = -math.inf
        best_state = None
        stale = 0
        for epoch in range(1, self.max_epochs + 1):
            self.model.train()
            for rna_batch, drug_batch, label_batch in loader:
                optimizer.zero_grad()
                loss = criterion(self.model(rna_batch, drug_batch), label_batch)
                loss.backward()
                optimizer.step()
            val_prob = self.predict_proba(bundle, split.val.pairs)
            val_ap = average_precision_score(split.val.labels, val_prob)
            if val_ap > best_ap + 1e-5:
                best_ap = val_ap
                best_state = copy.deepcopy(self.model.state_dict())
                self.best_epoch = epoch
                stale = 0
            else:
                stale += 1
                if stale >= self.patience:
                    break
        if best_state is None:
            raise RuntimeError("语义模型未产生有效 checkpoint")
        self.model.load_state_dict(best_state)
        return self

    def predict_logit(self, bundle: DatasetBundle, pairs: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("必须先 fit")
        rna, drug = self._pair_tensors(bundle.rna_features, bundle.drug_features, pairs)
        self.model.eval()
        with torch.no_grad():
            return self.model(rna, drug).numpy()

    def predict_proba(self, bundle: DatasetBundle, pairs: np.ndarray) -> np.ndarray:
        logits = self.predict_logit(bundle, pairs)
        return 1.0 / (1.0 + np.exp(-np.clip(logits, -30, 30)))


class PropensityResidualEstimator(SemanticEstimator):
    def __init__(self, seed: int, max_epochs: int = 120, patience: int = 15):
        super().__init__(seed=seed, max_epochs=max_epochs, patience=patience)
        self.model: TwoTowerWithPropensity | None = None
        self.train_positive_matrix: np.ndarray | None = None

    def _propensity_tensor(self, pairs: np.ndarray) -> torch.Tensor:
        if self.train_positive_matrix is None:
            raise RuntimeError("必须先 fit")
        row_degree = self.train_positive_matrix.sum(axis=1)
        col_degree = self.train_positive_matrix.sum(axis=0)
        row_scale = max(float(np.log1p(row_degree.max())), 1.0)
        col_scale = max(float(np.log1p(col_degree.max())), 1.0)
        values = np.column_stack(
            [
                np.log1p(row_degree[pairs[:, 0]]) / row_scale,
                np.log1p(col_degree[pairs[:, 1]]) / col_scale,
            ]
        )
        return torch.from_numpy(values.astype(np.float32))

    def fit(self, bundle: DatasetBundle, split: ProtocolSplit) -> "PropensityResidualEstimator":
        seed_everything(self.seed)
        self.train_positive_matrix = split.train_positive_matrix
        self.rna_mean, self.rna_std = self._stats(bundle.rna_features, split.train.pairs[:, 0])
        self.drug_mean, self.drug_std = self._stats(bundle.drug_features, split.train.pairs[:, 1])
        self.model = TwoTowerWithPropensity(
            bundle.rna_features.shape[1], bundle.drug_features.shape[1]
        )
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-3, weight_decay=1e-4)
        train_rna, train_drug = self._pair_tensors(
            bundle.rna_features, bundle.drug_features, split.train.pairs
        )
        train_propensity = self._propensity_tensor(split.train.pairs)
        train_labels = torch.from_numpy(split.train.labels.astype(np.float32))
        generator = torch.Generator().manual_seed(self.seed)
        loader = DataLoader(
            TensorDataset(train_rna, train_drug, train_propensity, train_labels),
            batch_size=min(256, len(train_labels)),
            shuffle=True,
            generator=generator,
        )
        criterion = nn.BCEWithLogitsLoss()
        best_ap = -math.inf
        best_state = None
        stale = 0
        for epoch in range(1, self.max_epochs + 1):
            self.model.train()
            for rna_batch, drug_batch, propensity_batch, label_batch in loader:
                optimizer.zero_grad()
                loss = criterion(
                    self.model(rna_batch, drug_batch, propensity_batch), label_batch
                )
                loss.backward()
                optimizer.step()
            val_probability = self.predict_proba(bundle, split.val.pairs)
            val_ap = average_precision_score(split.val.labels, val_probability)
            if val_ap > best_ap + 1e-5:
                best_ap = val_ap
                best_state = copy.deepcopy(self.model.state_dict())
                self.best_epoch = epoch
                stale = 0
            else:
                stale += 1
                if stale >= self.patience:
                    break
        if best_state is None:
            raise RuntimeError("倾向残差模型未产生有效 checkpoint")
        self.model.load_state_dict(best_state)
        return self

    def predict_logit(self, bundle: DatasetBundle, pairs: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("必须先 fit")
        rna, drug = self._pair_tensors(bundle.rna_features, bundle.drug_features, pairs)
        propensity = self._propensity_tensor(pairs)
        self.model.eval()
        with torch.no_grad():
            return self.model(rna, drug, propensity).numpy()


class PriorOffsetStructuralResidualEstimator(SemanticEstimator):
    """以训练图 RNA 关联倾向 log-odds 为固定 offset，学习结构残差。"""

    def __init__(
        self,
        seed: int,
        max_epochs: int = 120,
        patience: int = 15,
        leave_one_out: bool = True,
    ):
        super().__init__(seed=seed, max_epochs=max_epochs, patience=patience)
        self.leave_one_out = leave_one_out
        self.train_positive_matrix: np.ndarray | None = None
        self.active_train_drug_count = 0

    def _prior_logit(
        self, pairs: np.ndarray, train_labels: np.ndarray | None = None
    ) -> torch.Tensor:
        if self.train_positive_matrix is None or self.active_train_drug_count == 0:
            raise RuntimeError("必须先 fit")
        row_degree = self.train_positive_matrix.sum(axis=1).astype(np.float64)
        numerator = row_degree[pairs[:, 0]].copy()
        if train_labels is not None:
            numerator -= train_labels
            trials = self.active_train_drug_count - 1
        else:
            trials = self.active_train_drug_count
        probability = (numerator + 1.0) / (trials + 2.0)
        probability = np.clip(probability, 1e-6, 1.0 - 1e-6)
        return torch.from_numpy(np.log(probability / (1.0 - probability)).astype(np.float32))

    def fit(
        self, bundle: DatasetBundle, split: ProtocolSplit
    ) -> "PriorOffsetStructuralResidualEstimator":
        seed_everything(self.seed)
        self.train_positive_matrix = split.train_positive_matrix
        self.active_train_drug_count = int(
            np.count_nonzero(split.train_positive_matrix.sum(axis=0) > 0)
        )
        if self.active_train_drug_count < 2:
            raise ValueError("留一倾向先验至少需要两个训练药物")
        self.rna_mean, self.rna_std = self._stats(
            bundle.rna_features, split.train.pairs[:, 0]
        )
        self.drug_mean, self.drug_std = self._stats(
            bundle.drug_features, split.train.pairs[:, 1]
        )
        self.model = TwoTowerSemantic(
            bundle.rna_features.shape[1], bundle.drug_features.shape[1]
        )
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-3, weight_decay=1e-4)
        criterion = nn.BCEWithLogitsLoss()
        train_rna, train_drug = self._pair_tensors(
            bundle.rna_features, bundle.drug_features, split.train.pairs
        )
        train_labels = torch.from_numpy(split.train.labels.astype(np.float32))
        train_prior = self._prior_logit(
            split.train.pairs,
            split.train.labels if self.leave_one_out else None,
        )
        generator = torch.Generator().manual_seed(self.seed)
        loader = DataLoader(
            TensorDataset(train_rna, train_drug, train_prior, train_labels),
            batch_size=min(256, len(train_labels)),
            shuffle=True,
            generator=generator,
        )
        best_ap = -math.inf
        best_state = None
        stale = 0
        for epoch in range(1, self.max_epochs + 1):
            self.model.train()
            for rna_batch, drug_batch, prior_batch, label_batch in loader:
                optimizer.zero_grad()
                logits = self.model(rna_batch, drug_batch) + prior_batch
                loss = criterion(logits, label_batch)
                loss.backward()
                optimizer.step()
            val_probability = self.predict_proba(bundle, split.val.pairs)
            val_ap = average_precision_score(split.val.labels, val_probability)
            if val_ap > best_ap + 1e-5:
                best_ap = val_ap
                best_state = copy.deepcopy(self.model.state_dict())
                self.best_epoch = epoch
                stale = 0
            else:
                stale += 1
                if stale >= self.patience:
                    break
        if best_state is None:
            raise RuntimeError("先验 offset 残差模型未产生有效 checkpoint")
        self.model.load_state_dict(best_state)
        return self

    def predict_logit(self, bundle: DatasetBundle, pairs: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("必须先 fit")
        rna, drug = self._pair_tensors(bundle.rna_features, bundle.drug_features, pairs)
        prior = self._prior_logit(pairs)
        self.model.eval()
        with torch.no_grad():
            return (self.model(rna, drug) + prior).numpy()

    def predict_prior_proba(self, pairs: np.ndarray) -> np.ndarray:
        return torch.sigmoid(self._prior_logit(pairs)).numpy()


class PriorPreservingPairwiseResidualEstimator(PriorOffsetStructuralResidualEstimator):
    """用药物内成对排序训练残差，并允许验证集选择回退到纯先验。"""

    def __init__(
        self,
        seed: int,
        max_epochs: int = 120,
        patience: int = 15,
        residual_penalty: float = 1e-3,
        residual_scales: tuple[float, ...] = (0.0, 0.05, 0.1, 0.25, 0.5, 1.0),
    ):
        super().__init__(seed=seed, max_epochs=max_epochs, patience=patience)
        self.residual_penalty = residual_penalty
        self.residual_scales = residual_scales
        self.residual_scale = 0.0
        self.best_validation_macro_ap = -math.inf

    def _training_pairs(
        self, train_positive_matrix: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        return sample_within_drug_pairs(train_positive_matrix, self.seed)

    def _training_pair_weights(
        self,
        train_positive_matrix: np.ndarray,
        positive_pairs: np.ndarray,
    ) -> np.ndarray:
        return np.ones(len(positive_pairs), dtype=np.float32)

    def _predict_residual_logit(
        self, bundle: DatasetBundle, pairs: np.ndarray
    ) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("必须先 fit")
        rna, drug = self._pair_tensors(bundle.rna_features, bundle.drug_features, pairs)
        self.model.eval()
        with torch.no_grad():
            return self.model(rna, drug).numpy()

    def fit(
        self, bundle: DatasetBundle, split: ProtocolSplit
    ) -> "PriorPreservingPairwiseResidualEstimator":
        seed_everything(self.seed)
        self.train_positive_matrix = split.train_positive_matrix
        self.active_train_drug_count = int(
            np.count_nonzero(split.train_positive_matrix.sum(axis=0) > 0)
        )
        if self.active_train_drug_count < 2:
            raise ValueError("留一倾向先验至少需要两个训练药物")
        self.rna_mean, self.rna_std = self._stats(
            bundle.rna_features, split.train.pairs[:, 0]
        )
        self.drug_mean, self.drug_std = self._stats(
            bundle.drug_features, split.train.pairs[:, 1]
        )
        self.model = TwoTowerSemantic(
            bundle.rna_features.shape[1], bundle.drug_features.shape[1]
        )
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-3, weight_decay=1e-4)

        positive_pairs, unknown_pairs = self._training_pairs(
            split.train_positive_matrix
        )
        pair_weights = torch.from_numpy(
            self._training_pair_weights(
                split.train_positive_matrix, positive_pairs
            )
        )
        positive_rna, positive_drug = self._pair_tensors(
            bundle.rna_features, bundle.drug_features, positive_pairs
        )
        unknown_rna, unknown_drug = self._pair_tensors(
            bundle.rna_features, bundle.drug_features, unknown_pairs
        )
        prior_margin = self._prior_logit(
            positive_pairs, np.ones(len(positive_pairs), dtype=np.float32)
        ) - self._prior_logit(
            unknown_pairs, np.zeros(len(unknown_pairs), dtype=np.float32)
        )
        generator = torch.Generator().manual_seed(self.seed)
        loader = DataLoader(
            TensorDataset(
                positive_rna,
                positive_drug,
                unknown_rna,
                unknown_drug,
                prior_margin,
                pair_weights,
            ),
            batch_size=min(256, len(positive_pairs)),
            shuffle=True,
            generator=generator,
        )

        validation_drugs = np.unique(split.val.pairs[:, 1])
        validation_pairs = np.column_stack(
            [
                np.tile(np.arange(bundle.pair.shape[0], dtype=np.int64), len(validation_drugs)),
                np.repeat(validation_drugs, bundle.pair.shape[0]),
            ]
        )
        validation_labels = bundle.pair[
            validation_pairs[:, 0], validation_pairs[:, 1]
        ].astype(np.float32)
        validation_prior = self._prior_logit(validation_pairs).numpy()
        baseline_ap = drug_macro_average_precision(
            validation_labels, validation_prior, validation_pairs[:, 1]
        )
        best_ap = baseline_ap
        best_scale = 0.0
        best_state = copy.deepcopy(self.model.state_dict())
        stale = 0
        for epoch in range(1, self.max_epochs + 1):
            self.model.train()
            for (
                pos_rna,
                pos_drug,
                neg_rna,
                neg_drug,
                batch_prior_margin,
                batch_weight,
            ) in loader:
                optimizer.zero_grad()
                positive_residual = self.model(pos_rna, pos_drug)
                unknown_residual = self.model(neg_rna, neg_drug)
                margin = batch_prior_margin + positive_residual - unknown_residual
                penalty = 0.5 * (
                    positive_residual.square() + unknown_residual.square()
                )
                ranking_loss = nn.functional.softplus(-margin)
                loss = (
                    (
                        (ranking_loss + self.residual_penalty * penalty)
                        * batch_weight
                    ).sum()
                    / batch_weight.sum()
                )
                loss.backward()
                optimizer.step()

            validation_residual = self._predict_residual_logit(bundle, validation_pairs)
            scale, validation_ap = select_residual_scale(
                validation_labels,
                validation_prior,
                validation_residual,
                validation_pairs[:, 1],
                self.residual_scales,
            )
            if validation_ap > best_ap + 1e-5:
                best_ap = validation_ap
                best_scale = scale
                best_state = copy.deepcopy(self.model.state_dict())
                self.best_epoch = epoch
                stale = 0
            else:
                stale += 1
                if stale >= self.patience:
                    break
        self.model.load_state_dict(best_state)
        self.residual_scale = best_scale
        self.best_validation_macro_ap = best_ap
        return self

    def predict_logit(self, bundle: DatasetBundle, pairs: np.ndarray) -> np.ndarray:
        prior = self._prior_logit(pairs).numpy()
        residual = self._predict_residual_logit(bundle, pairs)
        return prior + self.residual_scale * residual


class EntityBalancedPairwiseResidualEstimator(
    PriorPreservingPairwiseResidualEstimator
):
    """让每个训练药物对 pairwise 排序损失的总权重相同。"""

    def _training_pair_weights(
        self,
        train_positive_matrix: np.ndarray,
        positive_pairs: np.ndarray,
    ) -> np.ndarray:
        positive_count = train_positive_matrix.sum(axis=0).astype(np.float64)
        weights = 1.0 / positive_count[positive_pairs[:, 1]]
        weights /= weights.mean()
        return weights.astype(np.float32)


class PriorOffsetBoostedResidualEstimator:
    """在训练药物完整候选矩阵上拟合实体均衡的树提升残差。"""

    def __init__(
        self,
        seed: int,
        max_epochs: int = 100,
        rank: int = 32,
        residual_scales: tuple[float, ...] = (0.0, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0),
    ):
        self.seed = seed
        self.max_epochs = max_epochs
        self.rank = rank
        self.residual_scales = residual_scales
        self.residual_scale = 0.0
        self.best_epoch = max_epochs
        self.best_validation_macro_ap = -math.inf

    def _prior_probability(
        self, pairs: np.ndarray, labels: np.ndarray | None = None
    ) -> np.ndarray:
        numerator = self.row_degree[pairs[:, 0]].astype(np.float64).copy()
        if labels is not None:
            numerator -= labels
            denominator = self.active_train_drug_count + 1.0
        else:
            denominator = self.active_train_drug_count + 2.0
        return np.clip((numerator + 1.0) / denominator, 1e-6, 1.0 - 1e-6)

    def _pair_features(
        self, pairs: np.ndarray, labels: np.ndarray | None = None
    ) -> np.ndarray:
        prior = self._prior_probability(pairs, labels)[:, None]
        return np.concatenate(
            [self.rna_latent[pairs[:, 0]], self.drug_latent[pairs[:, 1]], prior],
            axis=1,
        ).astype(np.float32, copy=False)

    def fit(
        self, bundle: DatasetBundle, split: ProtocolSplit
    ) -> "PriorOffsetBoostedResidualEstimator":
        seed_everything(self.seed)
        matrix = split.train_positive_matrix
        active_drugs = np.flatnonzero(matrix.sum(axis=0) > 0)
        self.active_train_drug_count = len(active_drugs)
        if self.active_train_drug_count < 2:
            raise ValueError("leave-one-out prior 至少需要两个训练药物")
        self.row_degree = matrix.sum(axis=1).astype(np.float64)
        rna_rank = min(self.rank, min(bundle.rna_features.shape) - 1)
        drug_rank = min(
            self.rank,
            len(active_drugs) - 1,
            bundle.drug_features.shape[1] - 1,
        )
        self.rna_svd = TruncatedSVD(
            n_components=rna_rank, algorithm="randomized", random_state=self.seed
        )
        self.drug_svd = TruncatedSVD(
            n_components=drug_rank, algorithm="randomized", random_state=self.seed
        )
        self.rna_latent = self.rna_svd.fit_transform(bundle.rna_features).astype(
            np.float32
        )
        self.drug_svd.fit(bundle.drug_features[active_drugs])
        self.drug_latent = self.drug_svd.transform(bundle.drug_features).astype(
            np.float32
        )

        rna_count = bundle.pair.shape[0]
        train_pairs = np.column_stack(
            [
                np.tile(np.arange(rna_count, dtype=np.int64), len(active_drugs)),
                np.repeat(active_drugs, rna_count),
            ]
        )
        labels_by_drug = matrix[:, active_drugs].T.astype(np.float32)
        train_labels = labels_by_drug.reshape(-1)
        train_prior = self._prior_probability(train_pairs, train_labels)
        train_target = train_labels - train_prior
        train_weights = entity_balanced_full_candidate_weights(labels_by_drug).reshape(-1)
        self.model = HistGradientBoostingRegressor(
            max_iter=self.max_epochs,
            max_depth=6,
            learning_rate=0.05,
            l2_regularization=1.0,
            early_stopping=False,
            random_state=self.seed,
        )
        self.model.fit(
            self._pair_features(train_pairs, train_labels),
            train_target,
            sample_weight=train_weights,
        )

        validation_drugs = np.unique(split.val.pairs[:, 1])
        validation_pairs = np.column_stack(
            [
                np.tile(np.arange(rna_count, dtype=np.int64), len(validation_drugs)),
                np.repeat(validation_drugs, rna_count),
            ]
        )
        validation_labels = bundle.pair[
            validation_pairs[:, 0], validation_pairs[:, 1]
        ].astype(np.float32)
        validation_prior = self._prior_probability(validation_pairs)
        validation_residual = self.model.predict(self._pair_features(validation_pairs))
        best_scale = 0.0
        best_ap = drug_macro_average_precision(
            validation_labels, validation_prior, validation_pairs[:, 1]
        )
        for scale in self.residual_scales:
            if scale == 0:
                continue
            prediction = np.clip(
                validation_prior + scale * validation_residual, 0.0, 1.0
            )
            macro_ap = drug_macro_average_precision(
                validation_labels, prediction, validation_pairs[:, 1]
            )
            if macro_ap > best_ap + 1e-12:
                best_scale = scale
                best_ap = macro_ap
        self.residual_scale = best_scale
        self.best_validation_macro_ap = best_ap
        return self

    def predict_prior_proba(self, pairs: np.ndarray) -> np.ndarray:
        return self._prior_probability(pairs)

    def predict_proba(self, bundle: DatasetBundle, pairs: np.ndarray) -> np.ndarray:
        prior = self._prior_probability(pairs)
        residual = self.model.predict(self._pair_features(pairs))
        return np.clip(prior + self.residual_scale * residual, 0.0, 1.0)


class AsymmetricInductiveMatrixFactorizationEstimator(
    PriorPreservingPairwiseResidualEstimator
):
    """warm RNA identity factor + cold drug feature tower 的 pairwise 残差。"""

    def _drug_tensor(
        self, drug_features: np.ndarray, pairs: np.ndarray
    ) -> torch.Tensor:
        return torch.from_numpy(
            (
                (drug_features[pairs[:, 1]] - self.drug_mean)
                / self.drug_std
            ).astype(np.float32)
        )

    def _predict_residual_logit(
        self, bundle: DatasetBundle, pairs: np.ndarray
    ) -> np.ndarray:
        self.model.eval()
        rna_ids = torch.from_numpy(pairs[:, 0].astype(np.int64))
        drug = self._drug_tensor(bundle.drug_features, pairs)
        with torch.no_grad():
            return self.model(rna_ids, drug).numpy()

    def fit(
        self, bundle: DatasetBundle, split: ProtocolSplit
    ) -> "AsymmetricInductiveMatrixFactorizationEstimator":
        seed_everything(self.seed)
        self.train_positive_matrix = split.train_positive_matrix
        self.active_train_drug_count = int(
            np.count_nonzero(split.train_positive_matrix.sum(axis=0) > 0)
        )
        if self.active_train_drug_count < 2:
            raise ValueError("留一倾向先验至少需要两个训练药物")
        self.drug_mean, self.drug_std = self._stats(
            bundle.drug_features, split.train.pairs[:, 1]
        )
        self.model = RNAIdentityDrugInteraction(
            bundle.pair.shape[0], bundle.drug_features.shape[1]
        )
        optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=1e-3, weight_decay=1e-4
        )
        positive_pairs, unknown_pairs = self._training_pairs(
            split.train_positive_matrix
        )
        positive_rna = torch.from_numpy(positive_pairs[:, 0].astype(np.int64))
        unknown_rna = torch.from_numpy(unknown_pairs[:, 0].astype(np.int64))
        positive_drug = self._drug_tensor(bundle.drug_features, positive_pairs)
        unknown_drug = self._drug_tensor(bundle.drug_features, unknown_pairs)
        prior_margin = self._prior_logit(
            positive_pairs, np.ones(len(positive_pairs), dtype=np.float32)
        ) - self._prior_logit(
            unknown_pairs, np.zeros(len(unknown_pairs), dtype=np.float32)
        )
        generator = torch.Generator().manual_seed(self.seed)
        loader = DataLoader(
            TensorDataset(
                positive_rna,
                positive_drug,
                unknown_rna,
                unknown_drug,
                prior_margin,
            ),
            batch_size=min(256, len(positive_pairs)),
            shuffle=True,
            generator=generator,
        )

        validation_drugs = np.unique(split.val.pairs[:, 1])
        validation_pairs = np.column_stack(
            [
                np.tile(np.arange(bundle.pair.shape[0], dtype=np.int64), len(validation_drugs)),
                np.repeat(validation_drugs, bundle.pair.shape[0]),
            ]
        )
        validation_labels = bundle.pair[
            validation_pairs[:, 0], validation_pairs[:, 1]
        ].astype(np.float32)
        validation_prior = self._prior_logit(validation_pairs).numpy()
        best_ap = drug_macro_average_precision(
            validation_labels, validation_prior, validation_pairs[:, 1]
        )
        best_scale = 0.0
        best_state = copy.deepcopy(self.model.state_dict())
        stale = 0
        for epoch in range(1, self.max_epochs + 1):
            self.model.train()
            for pos_rna, pos_drug, neg_rna, neg_drug, batch_prior_margin in loader:
                optimizer.zero_grad()
                positive_residual = self.model(pos_rna, pos_drug)
                unknown_residual = self.model(neg_rna, neg_drug)
                margin = batch_prior_margin + positive_residual - unknown_residual
                penalty = 0.5 * (
                    positive_residual.square().mean()
                    + unknown_residual.square().mean()
                )
                loss = (
                    nn.functional.softplus(-margin).mean()
                    + self.residual_penalty * penalty
                )
                loss.backward()
                optimizer.step()

            validation_residual = self._predict_residual_logit(
                bundle, validation_pairs
            )
            scale, validation_ap = select_residual_scale(
                validation_labels,
                validation_prior,
                validation_residual,
                validation_pairs[:, 1],
                self.residual_scales,
            )
            if validation_ap > best_ap + 1e-5:
                best_ap = validation_ap
                best_scale = scale
                best_state = copy.deepcopy(self.model.state_dict())
                self.best_epoch = epoch
                stale = 0
            else:
                stale += 1
                if stale >= self.patience:
                    break
        self.model.load_state_dict(best_state)
        self.residual_scale = best_scale
        self.best_validation_macro_ap = best_ap
        return self


class AsymmetricListwiseMatrixFactorizationEstimator(
    PriorPreservingPairwiseResidualEstimator
):
    """warm RNA identity factor + cold drug tower 的完整候选 listwise 残差。"""

    def _drug_tensor(
        self, drug_features: np.ndarray, drug_ids: np.ndarray
    ) -> torch.Tensor:
        return torch.from_numpy(
            (
                (drug_features[drug_ids] - self.drug_mean)
                / self.drug_std
            ).astype(np.float32)
        )

    def _predict_residual_logit(
        self, bundle: DatasetBundle, pairs: np.ndarray
    ) -> np.ndarray:
        self.model.eval()
        rna_ids = torch.from_numpy(pairs[:, 0].astype(np.int64))
        drug = self._drug_tensor(bundle.drug_features, pairs[:, 1])
        with torch.no_grad():
            return self.model(rna_ids, drug).numpy()

    def fit(
        self, bundle: DatasetBundle, split: ProtocolSplit
    ) -> "AsymmetricListwiseMatrixFactorizationEstimator":
        seed_everything(self.seed)
        self.train_positive_matrix = split.train_positive_matrix
        active_drugs = np.flatnonzero(split.train_positive_matrix.sum(axis=0) > 0)
        self.active_train_drug_count = len(active_drugs)
        if self.active_train_drug_count < 2:
            raise ValueError("留一倾向先验至少需要两个训练药物")
        self.drug_mean, self.drug_std = self._stats(
            bundle.drug_features, active_drugs
        )
        self.model = RNAIdentityDrugInteraction(
            bundle.pair.shape[0], bundle.drug_features.shape[1]
        )
        optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=1e-3, weight_decay=1e-4
        )
        train_drug = self._drug_tensor(bundle.drug_features, active_drugs)
        positive_mask = split.train_positive_matrix[:, active_drugs] > 0
        mask_tensor = torch.from_numpy(positive_mask)
        row_degree = split.train_positive_matrix.sum(axis=1).astype(np.float64)
        leave_one_out_count = row_degree[:, None] - positive_mask.astype(np.float64)
        prior_probability = (leave_one_out_count + 1.0) / (
            self.active_train_drug_count + 1.0
        )
        prior_probability = np.clip(prior_probability, 1e-6, 1.0 - 1e-6)
        train_prior = torch.from_numpy(
            np.log(prior_probability / (1.0 - prior_probability)).astype(np.float32)
        )

        validation_drugs = np.unique(split.val.pairs[:, 1])
        validation_pairs = np.column_stack(
            [
                np.tile(np.arange(bundle.pair.shape[0], dtype=np.int64), len(validation_drugs)),
                np.repeat(validation_drugs, bundle.pair.shape[0]),
            ]
        )
        validation_labels = bundle.pair[
            validation_pairs[:, 0], validation_pairs[:, 1]
        ].astype(np.float32)
        validation_prior = self._prior_logit(validation_pairs).numpy()
        best_ap = drug_macro_average_precision(
            validation_labels, validation_prior, validation_pairs[:, 1]
        )
        best_scale = 0.0
        best_state = copy.deepcopy(self.model.state_dict())
        stale = 0
        for epoch in range(1, self.max_epochs + 1):
            self.model.train()
            optimizer.zero_grad()
            residual = self.model.similarity(train_drug)
            loss = drug_listwise_multi_positive_loss(
                train_prior + residual, mask_tensor
            ) + self.residual_penalty * residual.square().mean()
            if not torch.isfinite(loss):
                raise RuntimeError("ALMF listwise 残差损失出现非有限值")
            loss.backward()
            optimizer.step()

            validation_residual = self._predict_residual_logit(
                bundle, validation_pairs
            )
            scale, validation_ap = select_residual_scale(
                validation_labels,
                validation_prior,
                validation_residual,
                validation_pairs[:, 1],
                self.residual_scales,
            )
            if validation_ap > best_ap + 1e-5:
                best_ap = validation_ap
                best_scale = scale
                best_state = copy.deepcopy(self.model.state_dict())
                self.best_epoch = epoch
                stale = 0
            else:
                stale += 1
                if stale >= self.patience:
                    break
        self.model.load_state_dict(best_state)
        self.residual_scale = best_scale
        self.best_validation_macro_ap = best_ap
        return self


class PropensityMatchedPairwiseResidualEstimator(
    PriorPreservingPairwiseResidualEstimator
):
    """使用 RNA 倾向匹配硬负例的先验保持 pairwise 残差模型。"""

    def __init__(
        self,
        seed: int,
        max_epochs: int = 120,
        patience: int = 15,
        residual_penalty: float = 1e-3,
        residual_scales: tuple[float, ...] = (0.0, 0.05, 0.1, 0.25, 0.5, 1.0),
        pool_size: int = 32,
        negatives_per_positive: int = 4,
    ):
        super().__init__(
            seed=seed,
            max_epochs=max_epochs,
            patience=patience,
            residual_penalty=residual_penalty,
            residual_scales=residual_scales,
        )
        self.pool_size = pool_size
        self.negatives_per_positive = negatives_per_positive

    def _training_pairs(
        self, train_positive_matrix: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        return sample_propensity_matched_within_drug_pairs(
            train_positive_matrix,
            self.seed,
            pool_size=self.pool_size,
            negatives_per_positive=self.negatives_per_positive,
        )


class PriorViolatingPairwiseResidualEstimator(
    PriorPreservingPairwiseResidualEstimator
):
    """使用 prior-violating 未知项的先验保持 pairwise 残差模型。"""

    def __init__(
        self,
        seed: int,
        max_epochs: int = 120,
        patience: int = 15,
        residual_penalty: float = 1e-3,
        residual_scales: tuple[float, ...] = (0.0, 0.05, 0.1, 0.25, 0.5, 1.0),
        pool_size: int = 64,
        negatives_per_positive: int = 8,
    ):
        super().__init__(
            seed=seed,
            max_epochs=max_epochs,
            patience=patience,
            residual_penalty=residual_penalty,
            residual_scales=residual_scales,
        )
        self.pool_size = pool_size
        self.negatives_per_positive = negatives_per_positive

    def _training_pairs(
        self, train_positive_matrix: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        return sample_prior_violating_within_drug_pairs(
            train_positive_matrix,
            self.seed,
            pool_size=self.pool_size,
            negatives_per_positive=self.negatives_per_positive,
        )


class PriorOffsetListwiseResidualEstimator(PriorOffsetStructuralResidualEstimator):
    """在 RNA 先验 offset 上对每个药物的完整 RNA 列表学习残差。"""

    def __init__(
        self,
        seed: int,
        max_epochs: int = 120,
        patience: int = 15,
        residual_penalty: float = 1e-3,
        residual_scales: tuple[float, ...] = (0.0, 0.05, 0.1, 0.25, 0.5, 1.0),
    ):
        super().__init__(seed=seed, max_epochs=max_epochs, patience=patience)
        self.residual_penalty = residual_penalty
        self.residual_scales = residual_scales
        self.residual_scale = 0.0
        self.best_validation_macro_ap = -math.inf
        self.model: ContrastiveTwoTower | None = None

    def _predict_residual_logit(
        self, bundle: DatasetBundle, pairs: np.ndarray
    ) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("必须先 fit")
        rna, drug = self._pair_tensors(bundle.rna_features, bundle.drug_features, pairs)
        self.model.eval()
        with torch.no_grad():
            return self.model(rna, drug).numpy()

    def fit(
        self, bundle: DatasetBundle, split: ProtocolSplit
    ) -> "PriorOffsetListwiseResidualEstimator":
        seed_everything(self.seed)
        self.train_positive_matrix = split.train_positive_matrix
        active_drugs = np.flatnonzero(split.train_positive_matrix.sum(axis=0) > 0)
        self.active_train_drug_count = len(active_drugs)
        if self.active_train_drug_count < 2:
            raise ValueError("留一倾向先验至少需要两个训练药物")
        all_rnas = np.arange(bundle.pair.shape[0], dtype=np.int64)
        self.rna_mean, self.rna_std = self._stats(bundle.rna_features, all_rnas)
        self.drug_mean, self.drug_std = self._stats(bundle.drug_features, active_drugs)
        self.model = ContrastiveTwoTower(
            bundle.rna_features.shape[1], bundle.drug_features.shape[1]
        )
        optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=1e-3, weight_decay=1e-4
        )
        train_rna = torch.from_numpy(
            ((bundle.rna_features - self.rna_mean) / self.rna_std).astype(np.float32)
        )
        train_drug = torch.from_numpy(
            (
                (bundle.drug_features[active_drugs] - self.drug_mean)
                / self.drug_std
            ).astype(np.float32)
        )
        positive_mask = split.train_positive_matrix[:, active_drugs] > 0
        mask_tensor = torch.from_numpy(positive_mask)
        row_degree = split.train_positive_matrix.sum(axis=1).astype(np.float64)
        leave_one_out_count = row_degree[:, None] - positive_mask.astype(np.float64)
        prior_probability = (leave_one_out_count + 1.0) / (
            self.active_train_drug_count + 1.0
        )
        prior_probability = np.clip(prior_probability, 1e-6, 1.0 - 1e-6)
        train_prior = torch.from_numpy(
            np.log(prior_probability / (1.0 - prior_probability)).astype(np.float32)
        )

        validation_drugs = np.unique(split.val.pairs[:, 1])
        validation_pairs = np.column_stack(
            [
                np.tile(np.arange(bundle.pair.shape[0], dtype=np.int64), len(validation_drugs)),
                np.repeat(validation_drugs, bundle.pair.shape[0]),
            ]
        )
        validation_labels = bundle.pair[
            validation_pairs[:, 0], validation_pairs[:, 1]
        ].astype(np.float32)
        validation_prior = self._prior_logit(validation_pairs).numpy()
        best_ap = drug_macro_average_precision(
            validation_labels, validation_prior, validation_pairs[:, 1]
        )
        best_scale = 0.0
        best_state = copy.deepcopy(self.model.state_dict())
        stale = 0
        for epoch in range(1, self.max_epochs + 1):
            self.model.train()
            optimizer.zero_grad()
            residual = self.model.similarity(train_rna, train_drug)
            loss = drug_listwise_multi_positive_loss(
                train_prior + residual, mask_tensor
            ) + self.residual_penalty * residual.square().mean()
            if not torch.isfinite(loss):
                raise RuntimeError("listwise 残差损失出现非有限值")
            loss.backward()
            optimizer.step()

            validation_residual = self._predict_residual_logit(
                bundle, validation_pairs
            )
            scale, validation_ap = select_residual_scale(
                validation_labels,
                validation_prior,
                validation_residual,
                validation_pairs[:, 1],
                self.residual_scales,
            )
            if validation_ap > best_ap + 1e-5:
                best_ap = validation_ap
                best_scale = scale
                best_state = copy.deepcopy(self.model.state_dict())
                self.best_epoch = epoch
                stale = 0
            else:
                stale += 1
                if stale >= self.patience:
                    break
        self.model.load_state_dict(best_state)
        self.residual_scale = best_scale
        self.best_validation_macro_ap = best_ap
        return self

    def predict_logit(self, bundle: DatasetBundle, pairs: np.ndarray) -> np.ndarray:
        prior = self._prior_logit(pairs).numpy()
        residual = self._predict_residual_logit(bundle, pairs)
        return prior + self.residual_scale * residual


class BilinearBCEEstimator(SemanticEstimator):
    def __init__(self, seed: int, max_epochs: int = 120, patience: int = 15):
        super().__init__(seed=seed, max_epochs=max_epochs, patience=patience)
        self.model: BilinearTwoTower | None = None

    def fit(self, bundle: DatasetBundle, split: ProtocolSplit) -> "BilinearBCEEstimator":
        seed_everything(self.seed)
        self.rna_mean, self.rna_std = self._stats(bundle.rna_features, split.train.pairs[:, 0])
        self.drug_mean, self.drug_std = self._stats(bundle.drug_features, split.train.pairs[:, 1])
        self.model = BilinearTwoTower(bundle.rna_features.shape[1], bundle.drug_features.shape[1])
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-3, weight_decay=1e-4)
        criterion = nn.BCEWithLogitsLoss()
        train_rna, train_drug = self._pair_tensors(
            bundle.rna_features, bundle.drug_features, split.train.pairs
        )
        train_labels = torch.from_numpy(split.train.labels.astype(np.float32))
        generator = torch.Generator().manual_seed(self.seed)
        loader = DataLoader(
            TensorDataset(train_rna, train_drug, train_labels),
            batch_size=min(256, len(train_labels)),
            shuffle=True,
            generator=generator,
        )
        best_ap = -math.inf
        best_state = None
        stale = 0
        for epoch in range(1, self.max_epochs + 1):
            self.model.train()
            for rna_batch, drug_batch, label_batch in loader:
                optimizer.zero_grad()
                loss = criterion(self.model(rna_batch, drug_batch), label_batch)
                loss.backward()
                optimizer.step()
            val_probability = self.predict_proba(bundle, split.val.pairs)
            val_ap = average_precision_score(split.val.labels, val_probability)
            if val_ap > best_ap + 1e-5:
                best_ap = val_ap
                best_state = copy.deepcopy(self.model.state_dict())
                self.best_epoch = epoch
                stale = 0
            else:
                stale += 1
                if stale >= self.patience:
                    break
        if best_state is None:
            raise RuntimeError("双线性 BCE 模型未产生有效 checkpoint")
        self.model.load_state_dict(best_state)
        return self


def nnpu_risk(
    positive_logits: torch.Tensor, unlabeled_logits: torch.Tensor, class_prior: float
) -> torch.Tensor:
    if not 0.0 < class_prior < 1.0:
        raise ValueError("class_prior 必须在 (0,1) 内")
    positive_as_positive = nn.functional.softplus(-positive_logits).mean()
    positive_as_negative = nn.functional.softplus(positive_logits).mean()
    unlabeled_as_negative = nn.functional.softplus(unlabeled_logits).mean()
    negative_risk = unlabeled_as_negative - class_prior * positive_as_negative
    if negative_risk.detach().item() < 0.0:
        return -negative_risk
    return class_prior * positive_as_positive + negative_risk


class NNPUEstimator(SemanticEstimator):
    def __init__(
        self, class_prior: float, seed: int, max_epochs: int = 120, patience: int = 15
    ):
        super().__init__(seed=seed, max_epochs=max_epochs, patience=patience)
        self.class_prior = class_prior
        self.best_val_aupr = -math.inf

    def fit(self, bundle: DatasetBundle, split: ProtocolSplit) -> "NNPUEstimator":
        seed_everything(self.seed)
        self.rna_mean, self.rna_std = self._stats(bundle.rna_features, split.train.pairs[:, 0])
        self.drug_mean, self.drug_std = self._stats(bundle.drug_features, split.train.pairs[:, 1])
        self.model = TwoTowerSemantic(bundle.rna_features.shape[1], bundle.drug_features.shape[1])
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-3, weight_decay=1e-4)
        positive_pairs = split.train.pairs[split.train.labels == 1]
        unlabeled_pairs = split.train.pairs[split.train.labels == 0]
        positive_rna, positive_drug = self._pair_tensors(
            bundle.rna_features, bundle.drug_features, positive_pairs
        )
        unlabeled_rna, unlabeled_drug = self._pair_tensors(
            bundle.rna_features, bundle.drug_features, unlabeled_pairs
        )
        positive_loader = DataLoader(
            TensorDataset(positive_rna, positive_drug),
            batch_size=min(256, len(positive_pairs)),
            shuffle=True,
            generator=torch.Generator().manual_seed(self.seed),
        )
        unlabeled_loader = DataLoader(
            TensorDataset(unlabeled_rna, unlabeled_drug),
            batch_size=min(256, len(unlabeled_pairs)),
            shuffle=True,
            generator=torch.Generator().manual_seed(self.seed + 1),
        )
        best_state = None
        stale = 0
        for epoch in range(1, self.max_epochs + 1):
            self.model.train()
            for (positive_rna_batch, positive_drug_batch), (
                unlabeled_rna_batch,
                unlabeled_drug_batch,
            ) in zip(positive_loader, unlabeled_loader):
                optimizer.zero_grad()
                loss = nnpu_risk(
                    self.model(positive_rna_batch, positive_drug_batch),
                    self.model(unlabeled_rna_batch, unlabeled_drug_batch),
                    self.class_prior,
                )
                if not torch.isfinite(loss):
                    raise RuntimeError("nnPU 风险出现非有限值")
                loss.backward()
                optimizer.step()
            val_probability = self.predict_proba(bundle, split.val.pairs)
            val_ap = average_precision_score(split.val.labels, val_probability)
            if val_ap > self.best_val_aupr + 1e-5:
                self.best_val_aupr = float(val_ap)
                best_state = copy.deepcopy(self.model.state_dict())
                self.best_epoch = epoch
                stale = 0
            else:
                stale += 1
                if stale >= self.patience:
                    break
        if best_state is None:
            raise RuntimeError("nnPU 模型未产生有效 checkpoint")
        self.model.load_state_dict(best_state)
        return self


def select_nnpu_estimator(
    bundle: DatasetBundle,
    split: ProtocolSplit,
    seed: int,
    max_epochs: int,
) -> tuple[NNPUEstimator, float]:
    active_drugs = int((split.train_positive_matrix.sum(axis=0) > 0).sum())
    observed_density = float(
        split.train_positive_matrix.sum()
        / max(1, split.train_positive_matrix.shape[0] * active_drugs)
    )
    best_estimator = None
    best_multiplier = math.nan
    for multiplier in (1.0, 2.0, 4.0):
        class_prior = min(0.49, max(1e-4, observed_density * multiplier))
        estimator = NNPUEstimator(
            class_prior=class_prior,
            seed=seed,
            max_epochs=max_epochs,
        ).fit(bundle, split)
        if best_estimator is None or estimator.best_val_aupr > best_estimator.best_val_aupr + 1e-12:
            best_estimator = estimator
            best_multiplier = multiplier
    if best_estimator is None:
        raise RuntimeError("nnPU prior 选择失败")
    return best_estimator, best_multiplier


class EntityBalancedContrastiveEstimator:
    def __init__(self, seed: int, max_epochs: int = 120, patience: int = 15):
        self.seed = seed
        self.max_epochs = max_epochs
        self.patience = patience
        self.model: ContrastiveTwoTower | None = None
        self.rna_mean: np.ndarray | None = None
        self.rna_std: np.ndarray | None = None
        self.drug_mean: np.ndarray | None = None
        self.drug_std: np.ndarray | None = None
        self.best_epoch = 0

    @staticmethod
    def _stats(features: np.ndarray, ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        selected = features[ids]
        mean = selected.mean(axis=0)
        std = selected.std(axis=0)
        return mean, np.where(std < 1e-6, 1.0, std)

    def _entity_tensors(
        self, bundle: DatasetBundle, rna_ids: np.ndarray, drug_ids: np.ndarray
    ) -> tuple[torch.Tensor, torch.Tensor]:
        rna = (bundle.rna_features[rna_ids] - self.rna_mean) / self.rna_std
        drug = (bundle.drug_features[drug_ids] - self.drug_mean) / self.drug_std
        return torch.from_numpy(rna.astype(np.float32)), torch.from_numpy(drug.astype(np.float32))

    def _pair_tensors(self, bundle: DatasetBundle, pairs: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
        rna = (bundle.rna_features[pairs[:, 0]] - self.rna_mean) / self.rna_std
        drug = (bundle.drug_features[pairs[:, 1]] - self.drug_mean) / self.drug_std
        return torch.from_numpy(rna.astype(np.float32)), torch.from_numpy(drug.astype(np.float32))

    def fit(self, bundle: DatasetBundle, split: ProtocolSplit) -> "EntityBalancedContrastiveEstimator":
        seed_everything(self.seed)
        rna_ids, drug_ids, positive_mask = active_positive_subgraph(split.train_positive_matrix)
        self.rna_mean, self.rna_std = self._stats(bundle.rna_features, rna_ids)
        self.drug_mean, self.drug_std = self._stats(bundle.drug_features, drug_ids)
        self.model = ContrastiveTwoTower(bundle.rna_features.shape[1], bundle.drug_features.shape[1])
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-3, weight_decay=1e-4)
        train_rna, train_drug = self._entity_tensors(bundle, rna_ids, drug_ids)
        mask_tensor = torch.from_numpy(positive_mask)
        best_ap = -math.inf
        best_state = None
        stale = 0
        for epoch in range(1, self.max_epochs + 1):
            self.model.train()
            optimizer.zero_grad()
            logits = self.model.similarity(train_rna, train_drug)
            loss = entity_balanced_multi_positive_loss(logits, mask_tensor)
            if not torch.isfinite(loss):
                raise RuntimeError("实体均衡对比损失出现非有限值")
            loss.backward()
            optimizer.step()
            val_score = self.predict_score(bundle, split.val.pairs)
            val_ap = average_precision_score(split.val.labels, val_score)
            if val_ap > best_ap + 1e-5:
                best_ap = val_ap
                best_state = copy.deepcopy(self.model.state_dict())
                self.best_epoch = epoch
                stale = 0
            else:
                stale += 1
                if stale >= self.patience:
                    break
        if best_state is None:
            raise RuntimeError("实体均衡对比模型未产生有效 checkpoint")
        self.model.load_state_dict(best_state)
        return self

    def predict_score(self, bundle: DatasetBundle, pairs: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("必须先 fit")
        rna, drug = self._pair_tensors(bundle, pairs)
        self.model.eval()
        with torch.no_grad():
            return self.model(rna, drug).numpy()


class SemanticProfileRidgeEstimator:
    """用语义核从已见实体回归完整关联谱。"""

    def __init__(self, axis: int, lambdas: tuple[float, ...] = (0.01, 0.1, 1.0, 10.0, 100.0)):
        if axis not in {0, 1}:
            raise ValueError("axis=0 表示 RNA profile，axis=1 表示 drug profile")
        self.axis = axis
        self.lambdas = lambdas
        self.feature_mean: np.ndarray | None = None
        self.feature_std: np.ndarray | None = None
        self.reference_features: np.ndarray | None = None
        self.coefficients: np.ndarray | None = None
        self.selected_lambda = math.nan
        self.best_val_aupr = -math.inf

    @staticmethod
    def _normalize(features: np.ndarray) -> np.ndarray:
        norm = np.linalg.norm(features, axis=1, keepdims=True)
        return features / np.maximum(norm, 1e-8)

    def _features(self, bundle: DatasetBundle) -> np.ndarray:
        return bundle.rna_features if self.axis == 0 else bundle.drug_features

    def _query_kernel(self, bundle: DatasetBundle, ids: np.ndarray) -> np.ndarray:
        if self.reference_features is None or self.feature_mean is None or self.feature_std is None:
            raise RuntimeError("必须先 fit")
        features = (self._features(bundle)[ids] - self.feature_mean) / self.feature_std
        return self._normalize(features) @ self.reference_features.T

    def fit(self, bundle: DatasetBundle, split: ProtocolSplit) -> "SemanticProfileRidgeEstimator":
        rna_ids, drug_ids, _ = active_positive_subgraph(split.train_positive_matrix)
        reference_ids = rna_ids if self.axis == 0 else drug_ids
        raw_features = self._features(bundle)[reference_ids]
        self.feature_mean = raw_features.mean(axis=0)
        feature_std = raw_features.std(axis=0)
        self.feature_std = np.where(feature_std < 1e-6, 1.0, feature_std)
        self.reference_features = self._normalize(
            (raw_features - self.feature_mean) / self.feature_std
        )
        kernel = self.reference_features @ self.reference_features.T
        eigenvalues, eigenvectors = np.linalg.eigh(kernel.astype(np.float64))
        eigenvalues = np.maximum(eigenvalues, 0.0)
        target = (
            split.train_positive_matrix[reference_ids, :]
            if self.axis == 0
            else split.train_positive_matrix[:, reference_ids].T
        ).astype(np.float64)
        projected_target = eigenvectors.T @ target
        best_coefficients = None
        for ridge_lambda in self.lambdas:
            coefficients = eigenvectors @ (
                projected_target / (eigenvalues[:, None] + ridge_lambda)
            )
            self.coefficients = coefficients
            val_score = self.predict_score(bundle, split.val.pairs)
            val_ap = average_precision_score(split.val.labels, val_score)
            if val_ap > self.best_val_aupr + 1e-12:
                self.best_val_aupr = float(val_ap)
                self.selected_lambda = ridge_lambda
                best_coefficients = coefficients.copy()
        if best_coefficients is None:
            raise RuntimeError("语义关联谱核回归未选择出有效参数")
        self.coefficients = best_coefficients
        return self

    def predict_score(self, bundle: DatasetBundle, pairs: np.ndarray) -> np.ndarray:
        if self.coefficients is None:
            raise RuntimeError("必须先 fit")
        query_ids, inverse = np.unique(pairs[:, self.axis], return_inverse=True)
        profiles = self._query_kernel(bundle, query_ids) @ self.coefficients
        counterpart = pairs[:, 1] if self.axis == 0 else pairs[:, 0]
        return profiles[inverse, counterpart]


def topology_scores(train_positive_matrix: np.ndarray, pairs: np.ndarray, rank: int, seed: int) -> np.ndarray:
    row_degree = train_positive_matrix.sum(axis=1)
    col_degree = train_positive_matrix.sum(axis=0)
    scale = np.sqrt(np.outer(np.maximum(row_degree, 1.0), np.maximum(col_degree, 1.0)))
    normalized = train_positive_matrix / scale
    effective_rank = max(1, min(rank, min(normalized.shape) - 1))
    svd = TruncatedSVD(n_components=effective_rank, algorithm="randomized", random_state=seed)
    reconstructed = svd.fit_transform(normalized) @ svd.components_
    return reconstructed[pairs[:, 0], pairs[:, 1]]


def _topk_neighbor_table(features: np.ndarray, seen_ids: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """返回每个实体在已见实体中的近邻 ID 与 softmax 权重。"""
    if len(seen_ids) == 0:
        raise ValueError("训练图中没有已见实体")
    centered = features - features[seen_ids].mean(axis=0, keepdims=True)
    normalized = centered / np.maximum(np.linalg.norm(centered, axis=1, keepdims=True), 1e-8)
    similarity = normalized @ normalized[seen_ids].T
    seen_position = {int(entity_id): position for position, entity_id in enumerate(seen_ids)}
    for entity_id, position in seen_position.items():
        similarity[entity_id, position] = -np.inf
    effective_k = max(1, min(k, len(seen_ids) - 1 if len(seen_ids) > 1 else 1))
    neighbor_position = np.argpartition(similarity, -effective_k, axis=1)[:, -effective_k:]
    neighbor_similarity = np.take_along_axis(similarity, neighbor_position, axis=1)
    order = np.argsort(-neighbor_similarity, axis=1)
    neighbor_position = np.take_along_axis(neighbor_position, order, axis=1)
    neighbor_similarity = np.take_along_axis(neighbor_similarity, order, axis=1)
    finite_similarity = np.where(np.isfinite(neighbor_similarity), neighbor_similarity, -1e6)
    stabilized = (finite_similarity - finite_similarity.max(axis=1, keepdims=True)) / 0.10
    weight = np.exp(np.clip(stabilized, -50, 0))
    weight /= np.maximum(weight.sum(axis=1, keepdims=True), 1e-12)
    return seen_ids[neighbor_position], weight


def content_knn_scores(
    rna_features: np.ndarray,
    drug_features: np.ndarray,
    train_positive_matrix: np.ndarray,
    pairs: np.ndarray,
    k: int,
) -> np.ndarray:
    """用语言表征近邻传播训练关联谱，不读取完整标签矩阵。"""
    row_degree = train_positive_matrix.sum(axis=1)
    col_degree = train_positive_matrix.sum(axis=0)
    seen_rna = np.flatnonzero(row_degree > 0)
    seen_drug = np.flatnonzero(col_degree > 0)
    if len(seen_rna) == 0 or len(seen_drug) == 0:
        return np.zeros(len(pairs), dtype=np.float64)
    rna_neighbors, rna_weight = _topk_neighbor_table(rna_features, seen_rna, k)
    drug_neighbors, drug_weight = _topk_neighbor_table(drug_features, seen_drug, k)

    rna_id = pairs[:, 0]
    drug_id = pairs[:, 1]
    rna_profile = train_positive_matrix[rna_neighbors[rna_id], drug_id[:, None]]
    drug_profile = train_positive_matrix[rna_id[:, None], drug_neighbors[drug_id]]
    rna_score = (rna_profile * rna_weight[rna_id]).sum(axis=1)
    drug_score = (drug_profile * drug_weight[drug_id]).sum(axis=1)

    row_cold = row_degree[rna_id] == 0
    col_cold = col_degree[drug_id] == 0
    score = 0.5 * (rna_score + drug_score)
    score = np.where(row_cold & ~col_cold, rna_score, score)
    score = np.where(col_cold & ~row_cold, drug_score, score)
    return np.where(row_cold & col_cold, 0.5 * (rna_score + drug_score), score)


def platt_calibrate(val_score: np.ndarray, val_label: np.ndarray, test_score: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    model = LogisticRegression(C=1.0, solver="lbfgs", random_state=0)
    model.fit(val_score.reshape(-1, 1), val_label)
    return (
        model.predict_proba(val_score.reshape(-1, 1))[:, 1],
        model.predict_proba(test_score.reshape(-1, 1))[:, 1],
    )


def reliability(train_positive_matrix: np.ndarray, pairs: np.ndarray, tau_scale: float = 1.0) -> np.ndarray:
    row_degree = train_positive_matrix.sum(axis=1)
    col_degree = train_positive_matrix.sum(axis=0)
    row_nonzero = row_degree[row_degree > 0]
    col_nonzero = col_degree[col_degree > 0]
    tau_row = max(1.0, float(np.median(row_nonzero)) if len(row_nonzero) else 1.0) * tau_scale
    tau_col = max(1.0, float(np.median(col_nonzero)) if len(col_nonzero) else 1.0) * tau_scale
    r = row_degree[pairs[:, 0]] / (row_degree[pairs[:, 0]] + tau_row)
    d = col_degree[pairs[:, 1]] / (col_degree[pairs[:, 1]] + tau_col)
    return np.sqrt(r * d)


def select_rast(
    semantic_val: np.ndarray,
    topology_val: np.ndarray,
    labels_val: np.ndarray,
    semantic_test: np.ndarray,
    topology_test: np.ndarray,
    split: ProtocolSplit,
) -> tuple[np.ndarray, dict[str, float]]:
    best = (-math.inf, 0.0, 1.0)
    for tau_scale in (0.5, 1.0, 2.0, 4.0):
        rel_val = reliability(split.train_positive_matrix, split.val.pairs, tau_scale)
        for alpha in np.linspace(0.0, 1.0, 11):
            weight = alpha * rel_val
            pred = (1.0 - weight) * semantic_val + weight * topology_val
            ap = average_precision_score(labels_val, pred)
            if ap > best[0] + 1e-12:
                best = (ap, float(alpha), float(tau_scale))
    _, alpha, tau_scale = best
    rel_test = reliability(split.train_positive_matrix, split.test.pairs, tau_scale)
    weight_test = alpha * rel_test
    prediction = (1.0 - weight_test) * semantic_test + weight_test * topology_test
    return prediction, {
        "alpha": alpha,
        "tau_scale": tau_scale,
        "mean_test_topology_weight": float(weight_test.mean()),
        "max_test_topology_weight": float(weight_test.max()),
    }


def select_knn_transfer(
    bundle: DatasetBundle, split: ProtocolSplit
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    best_ap = -math.inf
    best_k = 3
    best_val_raw: np.ndarray | None = None
    for k in (3, 5, 10, 20):
        val_raw = content_knn_scores(
            bundle.rna_features,
            bundle.drug_features,
            split.train_positive_matrix,
            split.val.pairs,
            k,
        )
        ap = average_precision_score(split.val.labels, val_raw)
        if ap > best_ap + 1e-12:
            best_ap = ap
            best_k = k
            best_val_raw = val_raw
    if best_val_raw is None:
        raise RuntimeError("kNN validation 选择失败")
    test_raw = content_knn_scores(
        bundle.rna_features,
        bundle.drug_features,
        split.train_positive_matrix,
        split.test.pairs,
        best_k,
    )
    val_probability, test_probability = platt_calibrate(best_val_raw, split.val.labels, test_raw)
    return val_probability, test_probability, {"knn_k": float(best_k), "knn_val_aupr": float(best_ap)}


def select_inductive_fusion(
    semantic_val: np.ndarray,
    knn_val: np.ndarray,
    labels_val: np.ndarray,
    semantic_test: np.ndarray,
    knn_test: np.ndarray,
) -> tuple[np.ndarray, float]:
    best_ap = -math.inf
    best_alpha = 0.0
    for alpha in np.linspace(0.0, 1.0, 11):
        prediction = (1.0 - alpha) * semantic_val + alpha * knn_val
        ap = average_precision_score(labels_val, prediction)
        if ap > best_ap + 1e-12:
            best_ap = ap
            best_alpha = float(alpha)
    return (1.0 - best_alpha) * semantic_test + best_alpha * knn_test, best_alpha


def select_three_expert_mixture(
    validation_predictions: tuple[np.ndarray, np.ndarray, np.ndarray],
    validation_labels: np.ndarray,
    test_predictions: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> tuple[np.ndarray, tuple[float, float, float], float]:
    best_ap = -math.inf
    best_weights = (1.0, 0.0, 0.0)
    for sem_step in range(10, -1, -1):
        for ecfp_step in range(10 - sem_step, -1, -1):
            propensity_step = 10 - sem_step - ecfp_step
            weights = (sem_step / 10.0, ecfp_step / 10.0, propensity_step / 10.0)
            validation_probability = sum(
                weight * prediction
                for weight, prediction in zip(weights, validation_predictions)
            )
            ap = average_precision_score(validation_labels, validation_probability)
            if ap > best_ap + 1e-12:
                best_ap = float(ap)
                best_weights = weights
    test_probability = sum(
        weight * prediction for weight, prediction in zip(best_weights, test_predictions)
    )
    return test_probability, best_weights, best_ap


def expected_calibration_error(labels: np.ndarray, probabilities: np.ndarray, bins: int = 10) -> float:
    boundaries = np.linspace(0.0, 1.0, bins + 1)
    total = len(labels)
    ece = 0.0
    for lower, upper in zip(boundaries[:-1], boundaries[1:]):
        include = (probabilities >= lower) & (probabilities < upper if upper < 1 else probabilities <= upper)
        if include.any():
            ece += include.mean() * abs(float(labels[include].mean() - probabilities[include].mean()))
    return float(ece)


def metrics(labels: np.ndarray, probabilities: np.ndarray) -> dict[str, float]:
    positive_count = int(labels.sum())
    order = np.argsort(-probabilities)[:positive_count]
    return {
        "aupr": float(average_precision_score(labels, probabilities)),
        "roc_auc": float(roc_auc_score(labels, probabilities)),
        "brier": float(brier_score_loss(labels, probabilities)),
        "ece": expected_calibration_error(labels, probabilities),
        "recall_at_p": float(labels[order].sum() / positive_count),
    }


def run_one(
    bundle: DatasetBundle,
    protocol: str,
    seed: int,
    max_epochs: int,
    rank: int,
    drug_fingerprints: np.ndarray | None = None,
    drug_scaffolds: np.ndarray | None = None,
    include_nnpu: bool = False,
) -> list[dict[str, float | int | str]]:
    if protocol == "published_pair":
        split = published_pair_split(bundle)
        cold_axis = None
    elif protocol == "rna_cold":
        split = cold_entity_split(bundle, axis=0, seed=seed)
        cold_axis = 0
    elif protocol == "drug_cold":
        split = cold_entity_split(bundle, axis=1, seed=seed)
        cold_axis = 1
    elif protocol == "drug_scaffold_cold":
        if drug_scaffolds is None:
            raise ValueError("drug_scaffold_cold 协议需要 drug_scaffolds")
        split = scaffold_cold_split(bundle, drug_scaffolds, seed)
        cold_axis = 1
    else:
        raise ValueError(f"未知协议: {protocol}")
    assert_split_integrity(split, cold_axis)

    semantic = SemanticEstimator(seed=seed, max_epochs=max_epochs).fit(bundle, split)
    sem_val = semantic.predict_proba(bundle, split.val.pairs)
    sem_test = semantic.predict_proba(bundle, split.test.pairs)
    contrastive = EntityBalancedContrastiveEstimator(seed=seed, max_epochs=max_epochs).fit(bundle, split)
    contrastive_val_raw = contrastive.predict_score(bundle, split.val.pairs)
    contrastive_test_raw = contrastive.predict_score(bundle, split.test.pairs)
    contrastive_val, contrastive_test = platt_calibrate(
        contrastive_val_raw, split.val.labels, contrastive_test_raw
    )
    propensity_residual = PropensityResidualEstimator(seed=seed, max_epochs=max_epochs).fit(
        bundle, split
    )
    propensity_val = propensity_residual.predict_proba(bundle, split.val.pairs)
    propensity_test = propensity_residual.predict_proba(bundle, split.test.pairs)
    bilinear_chemberta = BilinearBCEEstimator(seed=seed, max_epochs=max_epochs).fit(bundle, split)
    bilinear_chemberta_test = bilinear_chemberta.predict_proba(bundle, split.test.pairs)
    topo_val_raw = topology_scores(split.train_positive_matrix, split.val.pairs, rank, seed)
    topo_test_raw = topology_scores(split.train_positive_matrix, split.test.pairs, rank, seed)
    topo_val, topo_test = platt_calibrate(topo_val_raw, split.val.labels, topo_test_raw)
    knn_val, knn_test, knn_config = select_knn_transfer(bundle, split)
    rast_test, rast_config = select_rast(
        sem_val, topo_val, split.val.labels, sem_test, topo_test, split
    )
    inductive_test, inductive_alpha = select_inductive_fusion(
        sem_val, knn_val, split.val.labels, sem_test, knn_test
    )
    predictions = {
        "semantic": sem_test,
        "entity_balanced_contrastive": contrastive_test,
        "propensity_residual": propensity_test,
        "bilinear_chemberta": bilinear_chemberta_test,
        "topology": topo_test,
        "equal_blend": 0.5 * sem_test + 0.5 * topo_test,
        "rast": rast_test,
        "knn_transfer": knn_test,
        "inductive_fusion": inductive_test,
    }
    view_selection = "not_run"
    coverage_low = math.nan
    coverage_high = math.nan
    mean_test_structure_weight = math.nan
    expert_weights = (math.nan, math.nan, math.nan)
    expert_mixture_val_aupr = math.nan
    drug_top10_concentration = math.nan
    nnpu_estimator: NNPUEstimator | None = None
    nnpu_prior_multiplier = math.nan
    view_estimators: dict[str, SemanticEstimator] = {}
    if drug_fingerprints is not None:
        if len(drug_fingerprints) != len(bundle.drug_features):
            raise ValueError("ECFP 行数与药物特征行数不一致")
        feature_views = {
            "chemberta": bundle.drug_features,
            "ecfp": drug_fingerprints,
            "hybrid": np.concatenate([bundle.drug_features, drug_fingerprints], axis=1),
        }
        view_validation_aupr: dict[str, float] = {"chemberta": average_precision_score(split.val.labels, sem_val)}
        view_test_probability: dict[str, np.ndarray] = {"chemberta": sem_test}
        for view_name in ("ecfp", "hybrid"):
            view_bundle = DatasetBundle(
                rna_features=bundle.rna_features,
                drug_features=feature_views[view_name],
                pair=bundle.pair,
                pair_new=bundle.pair_new,
            )
            estimator = SemanticEstimator(seed=seed, max_epochs=max_epochs).fit(view_bundle, split)
            view_estimators[view_name] = estimator
            val_probability = estimator.predict_proba(view_bundle, split.val.pairs)
            test_probability = estimator.predict_proba(view_bundle, split.test.pairs)
            view_validation_aupr[view_name] = average_precision_score(split.val.labels, val_probability)
            view_test_probability[view_name] = test_probability
            predictions[f"semantic_{view_name}"] = test_probability
        ecfp_bundle = DatasetBundle(
            rna_features=bundle.rna_features,
            drug_features=feature_views["ecfp"],
            pair=bundle.pair,
            pair_new=bundle.pair_new,
        )
        bilinear_ecfp = BilinearBCEEstimator(seed=seed, max_epochs=max_epochs).fit(
            ecfp_bundle, split
        )
        predictions["bilinear_ecfp"] = bilinear_ecfp.predict_proba(
            ecfp_bundle, split.test.pairs
        )
        if include_nnpu:
            nnpu_estimator, nnpu_prior_multiplier = select_nnpu_estimator(
                ecfp_bundle, split, seed=seed, max_epochs=max_epochs
            )
            predictions["nnpu_ecfp"] = nnpu_estimator.predict_proba(
                ecfp_bundle, split.test.pairs
            )
        view_selection = max(view_validation_aupr, key=view_validation_aupr.get)
        predictions["drug_view_selected"] = view_test_probability[view_selection]
        expert_test, expert_weights, expert_mixture_val_aupr = select_three_expert_mixture(
            (
                sem_val,
                view_estimators["ecfp"].predict_proba(
                    DatasetBundle(
                        rna_features=bundle.rna_features,
                        drug_features=feature_views["ecfp"],
                        pair=bundle.pair,
                        pair_new=bundle.pair_new,
                    ),
                    split.val.pairs,
                ),
                propensity_val,
            ),
            split.val.labels,
            (sem_test, view_test_probability["ecfp"], propensity_test),
        )
        predictions["adaptive_three_expert"] = expert_test
        train_drug_degree = split.train_positive_matrix.sum(axis=0)
        active_train_drug_degree = train_drug_degree[train_drug_degree > 0]
        top_k = min(10, len(active_train_drug_degree))
        drug_top10_concentration = float(
            np.sort(active_train_drug_degree)[-top_k:].sum()
            / active_train_drug_degree.sum()
        )
        predictions["concentration_adaptive_router"] = (
            drug_top10_concentration * view_test_probability["ecfp"]
            + (1.0 - drug_top10_concentration) * propensity_test
        )
        if cold_axis == 1:
            train_drug_ids = np.flatnonzero(split.train_positive_matrix.sum(axis=0) > 0)
            test_drug_ids, inverse = np.unique(split.test.pairs[:, 1], return_inverse=True)
            entity_weight, coverage_low, coverage_high = chemical_coverage_weight(
                drug_fingerprints[train_drug_ids], drug_fingerprints[test_drug_ids]
            )
            pair_weight = entity_weight[inverse]
            mean_test_structure_weight = float(pair_weight.mean())
            predictions["chemical_coverage_routing"] = (
                pair_weight * view_test_probability["ecfp"]
                + (1.0 - pair_weight) * view_test_probability["chemberta"]
            )
    profile_ridge: SemanticProfileRidgeEstimator | None = None
    if cold_axis is not None:
        profile_ridge = SemanticProfileRidgeEstimator(axis=cold_axis).fit(bundle, split)
        profile_val_raw = profile_ridge.predict_score(bundle, split.val.pairs)
        profile_test_raw = profile_ridge.predict_score(bundle, split.test.pairs)
        _, profile_test = platt_calibrate(profile_val_raw, split.val.labels, profile_test_raw)
        predictions["semantic_profile_ridge"] = profile_test
    rows: list[dict[str, float | int | str]] = []
    for model_name, probability in predictions.items():
        row: dict[str, float | int | str] = {
            "protocol": protocol,
            "seed": seed,
            "model": model_name,
            "n_train": len(split.train.labels),
            "n_val": len(split.val.labels),
            "n_test": len(split.test.labels),
            "semantic_best_epoch": semantic.best_epoch,
            "contrastive_best_epoch": contrastive.best_epoch,
            "propensity_best_epoch": propensity_residual.best_epoch,
            "bilinear_chemberta_best_epoch": bilinear_chemberta.best_epoch,
            "rast_alpha": rast_config["alpha"],
            "rast_tau_scale": rast_config["tau_scale"],
            "mean_test_topology_weight": rast_config["mean_test_topology_weight"],
            "max_test_topology_weight": rast_config["max_test_topology_weight"],
            "knn_k": knn_config["knn_k"],
            "knn_val_aupr": knn_config["knn_val_aupr"],
            "inductive_alpha": inductive_alpha,
            "profile_ridge_lambda": (
                profile_ridge.selected_lambda if profile_ridge is not None else math.nan
            ),
            "profile_ridge_val_aupr": (
                profile_ridge.best_val_aupr if profile_ridge is not None else math.nan
            ),
            "drug_view_selected": view_selection,
            "ecfp_best_epoch": (
                view_estimators["ecfp"].best_epoch if "ecfp" in view_estimators else math.nan
            ),
            "hybrid_best_epoch": (
                view_estimators["hybrid"].best_epoch if "hybrid" in view_estimators else math.nan
            ),
            "coverage_low": coverage_low,
            "coverage_high": coverage_high,
            "mean_test_structure_weight": mean_test_structure_weight,
            "expert_weight_semantic": expert_weights[0],
            "expert_weight_ecfp": expert_weights[1],
            "expert_weight_propensity": expert_weights[2],
            "expert_mixture_val_aupr": expert_mixture_val_aupr,
            "drug_top10_concentration": drug_top10_concentration,
            "bilinear_ecfp_best_epoch": (
                bilinear_ecfp.best_epoch if drug_fingerprints is not None else math.nan
            ),
            "nnpu_class_prior": (
                nnpu_estimator.class_prior if nnpu_estimator is not None else math.nan
            ),
            "nnpu_prior_multiplier": nnpu_prior_multiplier,
            "nnpu_best_epoch": (
                nnpu_estimator.best_epoch if nnpu_estimator is not None else math.nan
            ),
        }
        row.update(metrics(split.test.labels, probability))
        rows.append(row)
    return rows


def summarize(raw: pd.DataFrame) -> pd.DataFrame:
    metric_columns = ["aupr", "roc_auc", "brier", "ece", "recall_at_p"]
    grouped = raw.groupby(["protocol", "model"])[metric_columns].agg(["mean", "std"])
    grouped.columns = [f"{metric}_{stat}" for metric, stat in grouped.columns]
    return grouped.reset_index()


def write_markdown_summary(summary: pd.DataFrame, path: Path) -> None:
    columns = ["protocol", "model", "aupr_mean", "aupr_std", "roc_auc_mean", "brier_mean", "ece_mean"]
    table = summary[columns].copy()
    for column in columns[2:]:
        table[column] = table[column].map(lambda value: f"{value:.4f}")
    lines = ["# lncRNA 发现性实验结果", "", table.to_markdown(index=False), ""]
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_seeds(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=("Lnc", "Mi"), default="Lnc")
    parser.add_argument("--dataset-dir", type=Path, default=Path("Dataset"))
    parser.add_argument("--protocols", default="published_pair,rna_cold,drug_cold")
    parser.add_argument("--seeds", default="1,2,3")
    parser.add_argument("--max-epochs", type=int, default=120)
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--include-ecfp", action="store_true")
    parser.add_argument("--include-nnpu", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=Path("results/generalization-audit"))
    args = parser.parse_args(list(argv) if argv is not None else None)
    if args.include_nnpu and not args.include_ecfp:
        parser.error("--include-nnpu 需要同时指定 --include-ecfp")

    bundle = load_bundle(args.dataset_dir, args.dataset)
    drug_fingerprints = (
        load_drug_fingerprints(args.dataset_dir, args.dataset) if args.include_ecfp else None
    )
    protocols = [item.strip() for item in args.protocols.split(",") if item.strip()]
    drug_scaffolds = (
        load_drug_scaffolds(args.dataset_dir, args.dataset)
        if "drug_scaffold_cold" in protocols
        else None
    )
    seeds = parse_seeds(args.seeds)
    rows: list[dict[str, float | int | str]] = []
    for protocol in protocols:
        for seed in seeds:
            print(f"[run] protocol={protocol} seed={seed}", flush=True)
            rows.extend(
                run_one(
                    bundle,
                    protocol,
                    seed,
                    args.max_epochs,
                    args.rank,
                    drug_fingerprints=drug_fingerprints,
                    drug_scaffolds=drug_scaffolds,
                    include_nnpu=args.include_nnpu,
                )
            )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    raw = pd.DataFrame(rows)
    summary = summarize(raw)
    raw.to_csv(args.output_dir / f"{args.dataset.lower()}_raw.csv", index=False)
    summary.to_csv(args.output_dir / f"{args.dataset.lower()}_summary.csv", index=False)
    write_markdown_summary(summary, args.output_dir / f"{args.dataset.lower()}_summary.md")
    config = {
        "dataset": args.dataset,
        "protocols": protocols,
        "seeds": seeds,
        "max_epochs": args.max_epochs,
        "rank": args.rank,
        "include_ecfp": args.include_ecfp,
        "scaffold_protocol": "drug_scaffold_cold" in protocols,
        "include_nnpu": args.include_nnpu,
        "torch_version": torch.__version__,
        "numpy_version": np.__version__,
    }
    (args.output_dir / f"{args.dataset.lower()}_config.json").write_text(
        json.dumps(config, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(summary.to_string(index=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
