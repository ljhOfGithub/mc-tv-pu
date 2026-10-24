# MC-TV-PU 冻结分数输入审计

## 结论

截至 2026-10-23，干净复现已经生成 15 个冻结候选的全部 75 个 `model/fold`
分区。分数输入门在 606 个有正例药物、1,805 个 RNA、每候选 1,093,830 行、总计
16,407,450 行上返回 `PASS`。

在此基础上，source-crossfit 启动前的第二道门使用原始 Mi 正矩阵、固定 SMILES、
Bemis--Murcko scaffold、`split_seed=20261019` 与 5 折规则重新计算每个药物的测试折，
再与实际 score 分区逐药物比较。606/606 个药物完全一致，因此
`target_drug_oof=PASS`。这不等于基础模型训练按 PMID/source 隔离。

## 必需的冻结产物

候选库已冻结在
`implement-stage/mc-tv-pu-oracle/CANDIDATE_MANIFEST.json`，共 15 个 ranker：早期六模型
confirmation 全集、RNA prior、prior-preserving residual、LocalGL，以及 PLMF-MDA 和
sChemNET 各三个预定训练种子。没有按历史 AP 阈值删候选，也没有挑最好 seed。

一个合法分片至少包含：

- `model`：读取 anchor 和 selector 输出前已冻结的候选名；
- `drug_id`：Mi 矩阵中 606 个至少一个已记录正例的药物；
- `rna_id`：全部 1,805 个 RNA 候选；
- `score`：有限实数，且每个 `(model, drug_id, rna_id)` 恰好一行；
- `fold`：必需，同一候选与药物不得跨外层测试折重复。

因此每个模型必须有 `606 × 1,805 = 1,093,830` 个唯一点分数。实际产物按
`model/fold` 分片为 Zstandard Parquet，严格不带 label；manifest 锁定候选、证据表、
配置、当前生成脚本、核心依赖代码与导出工具哈希。正式输入 CLI 不接受 CSV 或手写模型列表；
所有模型必须使用完整 `0..4` 折集合，且同一药物在所有候选中必须属于相同外折。

## 历史阻断与当前完成状态

首次审计时，对旧 `results/` 做只读扫描只发现抽样 OOF 和聚合指标，完整候选点分数为 0；
这解释了当时的 `INPUT_BLOCKED`，不是当前状态。随后按冻结 manifest 在 clean checkout
重新导出，当前正式输入为：

- 15 个 manifest 候选；
- 每个候选 5 个不可变 Parquet 分区，共 75 个分区；
- 606 个药物与 1,805 个 RNA 的完整笛卡尔积；
- 总计 16,407,450 行有限点分数；
- schema 严格为 `model,fold,drug_id,rna_id,score`，不含 label；
- 每个分区内部 `model`、`fold` 唯一且非空，并与其
  `model=<id>/fold=<n>.parquet` 路径身份逐文件一致；75/75 个现有分区均已通过该绑定检查；
- 所有候选对同一药物使用完全相同的外折，且折集合严格为 `0..4`；
- score-input status=`PASS`，issues=`[]`。

第二道 target-drug OOF 审计的确定性证据为：

- reference model：`semantic`；
- drug count：606；fold count：5；split seed：20261019；
- scaffold digest：`4be9bc53f2f39c74e3a432c5fc9c09b62fc804d92c13e4acb01ec17bf83203ce`；
- drug-to-test-fold digest：`c7209c13d0c7a41bf7f161d4a85aeb5d1ae84dc399439b6bb1e6146675a1d9c2`；
- 参考候选逐药物映射与重算 scaffold fold 完全一致；其余 14 个候选由已通过的 score gate
  证明使用相同 drug-to-fold 映射。

## 为什么两种代替都不合法

1. **抽样 OOF 不合法**：A-029 把 `Q_d` 固定为 1,805 个 RNA 上的均匀查询设计。只在
   抽样未标注对上运行 oracle 会换掉 `Q_d`，不是同一个问题的便宜近似。
2. **AP 不合法**：MC-TV-PU 的输入是每个 RNA 坐标上的 pairwise bounded-loss 差向量
   `s` 。不同点分数向量可以得到同一 AP，所以无法从聚合值逆推 `s`。

## 已执行的合法路径与剩余边界

1. 候选身份 manifest 在 selector 结果前冻结；后续运行按“新冻结实现的 clean confirmation”
   管理，不冒充缺少完整状态的历史运行精确 replay；
2. committed clean checkout 只增加全候选分数导出和数值等价的 LocalGL 内存修复，没有改变
   候选库、分割、训练目标、tau 或成功门；
3. 完整分数门与 target-drug OOF 重算门均通过；正式 selector 仍须达到终态、checkpoint
   全覆盖和汇总可重算，才能启动 source-crossfit；
4. target-drug OOF 只排除目标药物本身进入该 fold 的训练/验证。它不证明同一 held-out PMID
   未通过其他训练药物进入底层模型，也不授权端到端来源隔离主张。

这条路径属于干净复现与必需产物补录，不是新增探索实验。
当前 native confirmation、prior/prior-residual、LocalGL、PLMF-MDA 和 sChemNET 五个
source group 均已接入训练前 manifest/runtime-contract 预检与分片导出。PLMF-MDA 与
sChemNET 按 manifest 的 `training_seed` 显式绑定三个 seed-qualified 候选；运行设备锁为
LocalGL=`mps`、PLMF-MDA/sChemNET=`cpu`，不接受随机器变化的 `auto`。PLMF 在导出模式禁止
恢复旧 completed-unit，即使同路径已有 Parquet
也必须使用干净输出目录重新生成。LocalGL 已补锁其实际使用的 `MiDrug.pt`，所有 source
group 也冻结了核心依赖代码哈希。相关测试、入口 `--help` 和错误
split seed 负探针均通过；正式 75 个分区已经生成并通过两道输入门。final pairwise 候选已
纠正为 ChemBERTa 运行，不再错误指向同名 ECFP 产物。selector 性能结果仍未终态，因此
输入 `PASS` 不能写成算法成功。
