# MC-TV-PU 选择器预分析设计

状态：`FROZEN_BEFORE_SELECTOR_RESULTS`

冻结日期：2026-10-21

验证层级：Publication。本文档冻结后，才允许程序读取完整的 15 候选分数并计算真实选择结果。底层模型的历史指标在冻结候选库前已经看过，因此这里只能称为“选择器结果前冻结”，不能称为底层模型开发前预注册。

## 1. 研究问题与使用场景

目标不是再训练一个表示模型，而是在已经冻结的候选排序器集合中，选择对来源偏移和正例未完全标注更稳健的排序器。

部署对象是**已有至少一个来源支持关联的新药物**。这些关联作为 anchor 构造来源中心；该任务不是零关联 strict cold-start。方法输出只针对固定 RNA 候选宇宙上的查询设计风险，不能外推为自然生物总体风险。

主要 estimand 为式 (48) 的 compatible shared-class minimax regret：

\[
V_{\mathrm{comp}}(h;\tau,p^0)=
\max_{g\in\mathcal H}\frac{1}{m}\sum_{d=1}^{m}C_d(h,g;\tau,p_d^0).
\]

其中药物等权，联合不确定性集合固定为逐药笛卡尔积；不加入共享先验或跨药物总漂移预算。

## 2. 冻结输入

- 数据集：Mi。
- 活跃药物：`MiPair.npy` 中至少有一个记录正例的 606 个药物。
- RNA 候选宇宙：1,805 个 RNA。
- 候选排序器：严格使用 `implement-stage/mc-tv-pu-oracle/CANDIDATE_MANIFEST.json` 中按顺序列出的 15 个 candidate ID，不增删、不按历史 AP 过滤、不跨 seed 平均。
- 分数输入：每个候选必须覆盖 `606 × 1,805 = 1,093,830` 个唯一有限分数，并保持五折药物映射一致。
- 正式运行前，`mc_tv_pu_score_input_gate.audit_score_partition_directory` 必须返回 `PASS`；否则整个分析阻断。
- 来源证据：由命令行显式传入来源表，限制到冻结 `MiPair.npy` 的正 pair。不得在代码中默认个人绝对路径。

## 3. 查询分布、来源中心与先验

### 3.1 查询分布

对每个药物固定

\[
Q_d(i)=1/1805.
\]

它表示从固定 RNA 候选表均匀抽取一个查询，不表示自然界 RNA 或真实关联的总体分布。

### 3.2 来源中心

主中心 `pmid_equal`：每个 PMID 等质量；每篇 PMID 内出现的唯一 RNA 等质量。同一 PMID 内重复记录先去重。

预注册敏感性中心 `positive_pair_equal`：该药物在冻结矩阵中有来源映射的每个唯一正 RNA 等质量，不因 PMID 数量重复加权。

两个中心都必须完整报告。不能在看到选择结果后改用更有利的中心。

### 3.3 先验与 TV 半径

- 正类先验区间固定为 `[0,1]`；不使用经验正例率、sampled-unlabeled 比例或调参常数。
- 兼容性上界只能由 `Q_d`、`p_d^0` 与 `tau` 自动推出。
- `tau` 只作为敏感性轴，不宣称覆盖真实未来正例分布。
- 冻结网格：`0, 1/20, 1/10, 1/5, 3/10, 2/5, 1/2, 3/5, 7/10, 4/5, 9/10, 1`。
- 不指定“主 tau”，不允许根据 AP 或 selector 结果挑选 tau。

## 4. 成对损失与精确输入

采用只依赖排序的有界对称损失：

\[
\ell(z)=
\begin{cases}
0,&z>0,\\
1/2,&z=0,\\
1,&z<0.
\end{cases}
\]

它满足 `0 ≤ ell ≤ 1` 和 `ell(z)+ell(-z)=1`，对应 AUC 的成对排序损失。对每个模型和药物，先按原始 Float64 分数的严格大小及精确相等关系计算 midrank loss profile：

\[
L_h(i)=\frac{\#\{j:f_h(j)>f_h(i)\}+\frac12\#\{j:f_h(j)=f_h(i)\}}{1805}.
\]

随后令 `s_i = L_h(i)-L_g(i)`。因此 `s_i` 是分母 `2×1805` 的有理数，且在均匀 `Q` 下精确满足 `Q^T s=0`。

平局只按读取后的数值精确相等定义，不使用绝对或相对容差。该损失对不改变平局的严格单调变换不变，避免不同模型分数尺度影响结果。论文主要经验终点仍是 AP；本理论选择器只针对 pairwise/AUC surrogate，两者必须分层报告。

为使经验评价直接对应理论目标，source-crossfit 另外固定报告 tie-aware pairwise AUC。对一个
可评价 drug--fold，设移除 anchors 后有 `P` 个 held-out 正例、`N` 个未标注候选，
`n=P+N`。使用与 selector 完全相同的精确平局规则，有恒等式

\[
\frac{1}{P}\sum_{i:y_i=1}L_h(i)
=\frac{N(1-\operatorname{AUC}_h)+P/2}{P+N}.
\]

因此在固定 drug--fold 内，最大化该 AUC 与最小化 held-out 正例上的平均 selector loss
严格等价；这里的 loss 必须在同一个去除 anchors 后的候选集上重算。正例内部的损失和为
`P^2/(2n)`，正例--未标注对的损失和为 `PN(1-AUC)/n`，两项相加即得上式。

AUC 只作为损失函数一致性次要指标，不能取代 AP 主成功门，也不能把 AP 改写为已有理论
保证；AUC 中的比较类是未标注候选，不解释为经过实验确认的生物真负类。selector oracle
使用完整 1,805 RNA 宇宙、来源中心和 PU 模糊集，因此 episode AUC 不是 selector 的同一个
数值目标，也不承诺 MC-TV-PU 必然提高经验 AUC。正式命题与证明见
`implement-stage/mc-tv-pu-oracle/SPEC.md`。

## 5. Oracle 与数值证书

- 正式结果只调用 `solve_mc_tv_pu_oracle_exact`，全部 `Q`、`p0`、`s`、`tau` 以 `fractions.Fraction` 输入。
- 每个方向 `(drug, h, g, center, tau)` 的绝对 enclosure gap 固定为 `1/1000`，最大 exact evaluations 固定为 4,096。
- `h == g` 直接记录精确区间 `[0,0]`，不调用 oracle。
- 任一非对角结果若 `enclosure_valid != true`、`certified != true`、状态不是 `gap_met`，则该中心与 tau 的选择结果标记为 `NUMERIC_BLOCKED`，不得用点估计补齐。
- 普通浮点 oracle 只允许开发期交叉检查，不能进入正式选择或论文主表。

## 6. 区间聚合与选择规则

对固定候选 `h`、比较器 `g`，先分别对药物方向区间的下端和上端做等权平均。随后对比较器取最大：

\[
\underline V_h=\max_g\frac1m\sum_d\underline C_d(h,g),\qquad
\overline V_h=\max_g\frac1m\sum_d\overline C_d(h,g).
\]

不得把每个药物各自的最坏比较器先最大化再平均；式 (48) 使用一个共享比较器 `g`。

定义：

- `possibly_optimal`：`lower_V_h <= min_k upper_V_k`；
- `certifiably_unique`：`upper_V_h < min_{k != h} lower_V_k`；
- 边界相等时保留全部候选，不用 candidate 顺序或浮点容差强行破平局；
- 同时记录实现 `lower_V_h` 和 `upper_V_h` 的最坏比较器集合。

主输出是每个中心、每个 tau 的区间和 `possibly_optimal` 集合，不是单个事后挑选的模型。

### 6.1 冻结匹配消融

同一个 per-drug checkpoint 同时保存三种规则，保证候选、anchors、`Q`、tau 与 drug 权重完全匹配：

1. 完整 PU 相容集的认证区间；
2. 先把先验上界截到同一个 `pi_comp`、再删除逐坐标 `p_i <= t Q_i` 的 TV 矩形精确值；
3. 经验中心 `p0` 上的精确风险差。

三者都先对 drug 等权平均有向风险差，再对同一个共享 comparator 集合取最大，最后最小化候选值。若匹配矩形在 `pi_comp=1` 且 TV 方向最大值为正，则该方向严格记为 `unbounded`，不得用有限截断常数替代。正式 summary 必须并列输出 compatible `possibly_optimal`、`matched_rectangular.optimal` 和 `empirical_center.optimal`。

## 7. Selector-anchor 来源隔离的下游验证

完整来源中心上的选择路径只验证选择器的计算与稳定性，不能证明新关联预测有效。用于论文性能主张时，必须另外执行 selector-anchor 来源交叉验证：

1. PMID/source ID 在全数据集上确定性分为 5 个外层 folds；同一 PMID 跨药物时仍只能属于同一 fold；
2. 对每个 fold，属于该 fold 的 PMID 不得进入任何药物的 anchor；固定合法 drug--fold 集合后，held-out RNA 正例身份与评价指标不得直接进入该 fold 的 oracle 聚合或候选选择；
3. 每个 fold 独立生成选择集合 `S_f`，禁止把不同 held-out folds 的 oracle 汇总为一个集合后回评这些 folds；
4. held-out 正例只保留未在该 drug--fold anchors 出现的 RNA；没有新 held-out RNA 的 drug--fold 仍进入 selector，仅从性能指标中排除并报告；
5. 评估时从候选列表中移除 anchor RNA，在剩余完整候选空间计算 AP、tie-aware pairwise AUC、Recall@10、Recall@50；
6. 对每个 drug--fold 使用 `S_f` 的最差成员差异，先在 drug 内平均可评价 folds，再对 drug 等权；
7. 与每个冻结候选、历史 AP 最优候选和 `rna_prior_only` 比较；不新增模型训练；
8. pairwise AUC 对两个预设基线的全部 `center × tau` 格点按结果前固定的四类规则公开，
   只作损失函数机制诊断，不进入 AP 主成功门；机制表述必须覆盖所有 AP 强连续窗口的格点
   并集，不得从多个通过窗口中再挑选 AUC 有利窗口；
9. compatible、匹配矩形与经验中心三种选择规则都在同一 fold 上以各自 set-valued 最差成员
   评价；compatible 与两个消融规则的 AP 差异构成独立 48 项双侧 simultaneous 推断 family，
   不进入主门。只有 AP 主门通过时，才在全部 AP 强连续窗口格点并集上路由消融主张：全区域
   simultaneous 下界为正才称 compatible 占优，任一 simultaneous 上界为负优先记反向证据，
   不得挑选单个有利格点或窗口。

原 leave-one-PMID runner 因跨 episode 来源回流已在结果前降级为 diagnostic-only。Publication cross-fit runner 的输出格式和不确定性区间必须在读取结果前另行冻结；本轮 selector 程序不得把完整来源中心上的 anchor pair 当作 held-out 性能标签。

## 8. 预注册失败门

以下任一情况都会阻断“实用鲁棒模型选择”主张，但不修改规则补救：

1. 分数输入门不是 15/15 `PASS`；
2. 任一正式 oracle 区间未认证；
3. 两个中心下所有 tau 的 `possibly_optimal` 集合均包含全部 15 个候选，说明证据没有选择价值；
4. selector-anchor 来源交叉验证未覆盖足够的非重叠 held-out 正例，或所选集合相对历史 AP 最优候选没有可重复优势；
5. 选择只在某一个事后有利 tau 或某一个中心成立；
6. 直接近邻检索表明 PU 混合兼容 oracle、自动先验截断和 shared-comparator macro selector 的完整组合已有等价方法。

失败时保留为边界/否定结果，不继续搜索新的 tau、中心、候选子集或分数变换。

## 9. 允许的论文表述

若全部门通过，可表述为：在冻结的 target-drug-OOF 候选库、均匀有限查询设计、来源 anchor 中心和预设 TV 敏感性集合下，MC-TV-PU 给出经过有理数 enclosure 认证的 compatible shared-class minimax-regret 选择路径，并在 selector-anchor source cross-fit 中检验其条件性预测价值。该协议不认证底层模型训练、先验估计、特征构造或历史开发的 PMID 隔离。

不得表述为：端到端 PMID/source 隔离、真实生物总体上的最优模型、严格零样本 cold-start、真实未知关联风险的无条件保证、AP 最优性的理论证明，或底层模型开发前预注册。
