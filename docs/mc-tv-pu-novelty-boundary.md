# MC-TV-PU 新颖性边界与纸面门

状态：`N1_PASS_N2_COMPLETE_NO_CLAIM_V1_PASS_BOUNDARY_RESULT`

首次冻结日期：2026-10-21；当前状态更新：2026-10-24

## 1. 当前判定

独立复核判定为 `PROCEED WITH CAUTION`，方法新颖性暂定 `5/10`。未发现一篇已核实论文完整覆盖“PU 混合相容集合 + 正类 TV 不确定性 + 冻结有限排序器库的 shared-comparator minimax regret + 严格有理数 enclosure oracle”；但每个通用部件都有强前驱，不能拆开计为多项原创。

唯一可保留的方法 delta 是：

> 对固定有限查询分布与正类 TV 不确定集，保留未知类先验和正类分布在 PU 混合式中的逐坐标兼容耦合，精确求解冻结排序器之间的最坏排序风险差，并证明该耦合能够改变 minimax-regret 决策。

## 2. 不得声称为原创的部分

- 通用 minimax-regret optimization、shared comparator 与对 loss difference 做 DRO：Agarwal and Zhang, *Minimax Regret Optimization for Robust Machine Learning under Distribution Shift*, COLT 2022。
- PU distribution shift 本身：Hammoudeh and Lowd, *Learning from Positive and Unlabeled Data with Arbitrary Positive Shift*, NeurIPS 2020；Kumagai et al., *Importance-weighted Positive-unlabeled Learning for Distribution Shift Adaptation*, AISTATS 2025。
- 未知类先验或 PU 标注机制诊断本身：He, Liang, and Liu, UAI 2026 的 labeling-mechanism 工作。
- `possibly_optimal` 只是数值区间下不能排除的 minimax-regret 候选，不是“在某个合法 PU 世界中 Bayes 最优”的统计识别集合。
- Fraction 算术、TV 质量搬运、参数 LP、leave-one-source-out 与 drug bootstrap 都是实现或研究设计工具，不是独立方法贡献。

## 3. 已核近邻的非等价边界

- IWPU（AISTATS 2025）从源域和目标域的正例与未标注样本联合估计重要性权重并重新训练分类器；主设定把两个类先验作为已知输入，未知时只外接点估计与敏感性分析。它没有 TV 模糊集、联合先验区间或冻结候选 minimax regret。
- S-PUNA（ECCV 2026，arXiv:2605.31187v2）把 TV 用于平衡二分类 Bayes 难度解释，方法是 kNN 双向伪标签、谱熵停止和分类器蒸馏；没有 TV 模糊集、相容先验区间、有限模型库 minimax regret 或认证 oracle。其 Lemma 1 所称“一般正定协方差下，小污染必严格降低谱熵”已被显式二维反例推翻，正文只引用问题与算法边界，不依赖该定理。
- *Online Adaptation to Class-Prior Shift from Positive and Unlabeled Data*（OpenReview `7Wp1c6Xbdh`）只核实到作者列出的 WiML @ NeurIPS 2026 非归档 workshop 元数据；OpenReview 正文受浏览器验证阻断，未取得全文。它不进入本稿技术参考文献和直接近邻表，也不据标题推断方法细节。

## 4. 必须通过的纸面门

### Gate N1：主设定下的非冗余性 — `PASS`

已在均匀 `Q`、先验 `[0,1]` 和冻结 `tau=3/5` 下证明：完整相容集与“先把先验上界截到 `pi_comp`、再移除逐坐标兼容约束”的匹配矩形基线产生严格不同的 minimax-regret 决策。

新例取 `Q=(1/4,1/4,1/4,1/4)`、`p0=(1,0,0,0)` 和两条可实现严格排序。它给出 `t_min=8/5`、`pi_comp=5/8`。匹配矩形风险精确为 `(14/15, 2/3)`，选择 B；对完整相容集可解析求得 `G_s(t)=(t-1)/4`（`8/5<=t<=12/5`）并在之后饱和为 `7/20`，反方向固定为 `1/4`，所以完整相容风险精确为 `(3/5, 2/3)`，选择 A。永久回归见 `test_uniform_q_compatible_selector_separates_from_prior_clipped_rectangle` 与 `test_uniform_q_separation_inner_values_match_closed_form`。

旧四点例使用非均匀 `Q` 与 `[1/5,3/4]`；将其复制成均匀查询后，在 `tau=1/5` 有 `t_min=1`，匹配矩形仍发散，因此没有被用来关闭 N1。

### Gate N2：匹配消融 — `COMPLETE_NO_EMPIRICAL_CLAIM`

checkpoint schema v2 已在读取正式 selector 结果前实现同候选、同 anchors、同 `Q/tau` 的三路比较：

1. 完整 PU 相容集；
2. 先截断到 `pi_comp`、再移除逐坐标兼容约束的矩形集；
3. 经验中心选择。

方向值和 shared-comparator 聚合均有回归测试。若 `pi_comp=1` 时矩形方向发散，输出显式 `unbounded`；论文不能把这种格点的失败当成完整相容集的经验优势。终态运行与 source cross-fit 均已完成。虽然 selector 至少在一个格点收窄候选集，cross-source AP 主成功门失败，AP 强区域为空；冻结路由对 `matched_rectangular` 和 `empirical_center` 均为 `PRIMARY_GATE_FAILED_NO_RULE_ABLATION_CLAIM`。因此 N2 关闭为“实现与评价完整、无经验消融优势主张”，不能改写成匹配消融通过。

结果前又补齐同协议性能消融：source-crossfit 将 compatible、匹配矩形和经验中心三种
fold-specific 选择规则分别按 set-valued 最差成员评价，并输出 compatible 减两种消融的
paired drug-bootstrap 差异。AP 使用独立 48 项 simultaneous family，不改变主成功门。
因此 N2 不再只比较“选了谁”，还会检验完整兼容耦合的选择是否在 held-out 记录上更有价值；
该 family 在结果前冻结双侧同时带，并只在 AP 主门通过后覆盖所有 AP 强窗口格点的并集形成
主张：全区域同时下界为正才允许称优于某个消融，任一同时上界为负优先记为反向证据。
终态结果没有授权“优于消融”主张；该句从结果前约束转为永久主张边界。

### Gate V1：selector-anchor 全局来源隔离 — `PASS_PUBLICATION_COVERAGE`

Publication 性能必须使用 selector-anchor source cross-fit。`mc_tv_pu_source_crossfit_runner.py` 已在正式结果前实现 fold-specific selector、不可覆盖 checkpoint、fold-specific 最差成员差异和 drug bootstrap，并以必填主 selector 目录和终态日志硬门阻止提前启动；原 leave-one-PMID 聚合存在来源回流，只能作为内部来源轮换敏感性。V1 只保证同 fold 的 held-out PMID 不进入任何药物的 selector anchor；冻结基础模型没有按 PMID 重训，不能支持端到端来源隔离。

终态主 selector 为 1212/1212 checkpoints，日志最后一个非空行为 `EXIT_STATUS=0`，无 `.tmp.parquet` 或 `.partial`。source cross-fit 覆盖 5 个 folds、208 个可评价药物、612 个 eligible drug-folds，其中 563 个可评价，含 6408 条 held-out records；`publication_coverage_passed=true`。结果包日志为 `EXIT_STATUS=0`，六项 artifact SHA-256 逐项匹配且无残片。因此 V1 的 publication coverage 门通过，但这只证明 selector-anchor 层来源隔离和结果包完整，不升级为端到端训练来源隔离。

2026-10-23 的结果前预检进一步把 Recall 三点窗口语义、主 selector `NO_SELECTION_VALUE` 传递、正式输出目录单进程锁、`root/Dataset` 绑定、分区路径/内容身份和 checkpoint exact-gap 复核固化为代码与测试。该修复没有改变候选、tau、中心、阈值、bootstrap 或 claim routing。全套测试为 `290 passed, 7 warnings`；75/75 个实际分区的路径/内容身份复核问题数为 0。

## 5. 决策规则

- `N1` 理论门通过、`V1` publication coverage 通过；`N2` 已完整执行但因 AP 主门失败而没有经验消融主张。当前按“理论非冗余 + 来源隔离的负向边界结果”保留窄 research 论文定位。
- `N1` 通过、实证门失败：保留计算/识别边界论文，不追新候选或新 tau。当前已经进入这一路由，终态决定为 `FAIL_RETAIN_AS_BOUNDARY_RESULT`。
- `N1` 失败：停止方法原创主张，B 轨只作为 A 轨的条件性稳健性附录。
- 任一直接近邻被全文核实为同一信息结构、同一相容集合及等价 oracle：停止 MC-TV-PU 原创主张。

## 6. 可投稿的一句话定位

我们研究来源偏倚正例条件下的冻结排序器选择，刻画 PU 混合兼容约束造成的决策差异，给出具有严格数值误差保证的条件性 minimax-regret 求解方法，并以 selector-anchor source cross-fit 检验其跨文献来源记录关联发现价值；该验证不声称底层模型训练已按 PMID 隔离。

## 7. 论文初稿入口

B 轨独立方法论文的[终态中文事实稿](mc-tv-pu-research-paper-draft.md)已经组织标题、摘要、引言、相关工作边界、方法公式、非冗余命题、认证 oracle、实验协议、冻结结果、局限性和图表计划；所有实证数字来自终态结果包，不再保留运行中 checkpoint 占位符。
