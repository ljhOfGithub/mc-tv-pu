# 规格：MC-TV-PU 条件最坏风险差求解器

## 目标

实现一个纯数值、可复核的有限支持求解器。给定未标注边际分布 `Q`、正类中心
`p0`、TV 半径 `tau`、候选相对基线的逐点差值 `s` 与先验区间，返回

\[
\sup_{t\in\mathcal T^\circ}\frac{t}{t-1}
\max_{p\in\Delta:\,\mathrm{TV}(p,p^0)\le\tau,\;p\le tQ}s^\top p
\]

的数值上下界、当前最佳见证点和求解诊断。它实现
本规格下文定义的条件性 oracle，不训练模型，也不把输入中的 `Q` 解释成未经声明的生物总体。
当前可复核理论入口是本文件的“匹配矩形基线与包含命题”和
`NONTRIVIAL_SEPARATION.md`；不存在可引用的独立 `PROOF_PACKAGE.md`。

## 输入

- 一维有限数组 `q`、`p0`、`s`，长度相同；`q` 与 `p0` 是概率向量；
- `tau in [0,1]`；
- `prior_lower`、`prior_upper`，满足 `0 <= lower <= upper <= 1`；
- 绝对认证间隙 `absolute_gap`、LP 数值容差与最大 LP 调用数。

在真实 selector 调用 oracle 前，候选分数长表必须先通过
`mc_tv_pu_score_input_gate.py`：每个事前冻结候选须对 606 个有正例药物与 1,805 个 RNA
的完整笛卡尔积提供唯一、有限点分数。sampled-unlabeled OOF 与 per-drug 聚合指标
不符合这个契约，不得作为近似输入。
真实输入使用 `CANDIDATE_MANIFEST.json` 冻结的 15 个候选，并按 `model/fold` 写为
Zstandard Parquet 分片。每片严格只包含 `model,fold,drug_id,rna_id,score`，不包含
label；正式 CLI 不接受 CSV 或手写模型清单。审计器流式检查每个药物在每个候选内恰好
属于一个外折、所有候选对同一药物使用完全相同的外折、折集合严格为 `0..4`，并覆盖
全部 RNA。启用分数导出时，入口必须在加载数据或训练前验证 manifest、输入哈希、入口生成器
与核心依赖代码哈希及 source-group runtime contract；改变 split、训练预算或模型语义时不得
沿用冻结 ID。PLMF-MDA 与 sChemNET 的三个训练 seed 必须逐一读取 manifest 中显式的
`training_seed` 绑定，不得按列表顺序、动态拼名或跨 seed 平均生成新候选。
这里的外折是候选分数的药物折，不是后续的 PMID/source fold；输入门验证完整坐标、共同
药物折和冻结工件身份。source-crossfit 启动前还必须从固定 pair matrix、SMILES、
Bemis--Murcko scaffold、split seed 和 fold count 重算 drug-to-test-fold 映射，并与 score
分区逐药物一致；没有该 `PASS` 证据不得写 `target_drug_oof=true`。两道门都不认证底层模型
训练的 PMID/source 隔离。
Publication source-crossfit 的 `run_config.json` 必须另外锁定 runner、统计实现、selector、
oracle、输入门和来源解析所直接使用的本地源码 SHA-256；任一源码变化都属于不同冻结运行，
不得继续写入或复用原输出目录。
终态 `crossfit_validation_summary.json` 必须记录 `run_config.json`、来源 fold assignment 和
四张最终 CSV 的 SHA-256；因此 summary 脱离或错配任一配套工件时都不能作为正式结果包。
runner 只有在 summary 写出后重新读取六个工件、逐项复算哈希并确认不存在 `.tmp.parquet` 或
`.partial` 残片时才允许成功退出。
source-crossfit 的经验主终点保持 AP；另外必须报告使用同一精确平局规则的 pairwise AUC。
对每个固定 drug--fold，在同一个去除 anchors 后的候选集上重算 loss 时，该 AUC 与
held-out 正例平均 midrank loss 严格单调等价。它只提供损失函数一致性诊断，不进入 AP
成功门，也不把 AP 升格为理论保证。

### 命题：episode 内 midrank loss 与 tie-aware AUC 的仿射等价

固定一个移除 anchors 后的评价候选集，含 `P>0` 个 held-out 记录正例和 `N>0` 个
未标注候选，令 `n=P+N`。对任一冻结排序器 `h`，在这个相同候选集上重新计算

\[
L_h(i)=\frac{\#\{j:f_h(j)>f_h(i)\}+\frac12\#\{j:f_h(j)=f_h(i)\}}{n}.
\]

若 `AUC_h` 对每个正例--未标注对按正确、平局、反序分别计 `1,1/2,0`，则

\[
\frac1P\sum_{i:y_i=1}L_h(i)
=\frac{N(1-\operatorname{AUC}_h)+P/2}{P+N}.
\]

证明：所有正例--正例无序对无论严格排序还是平局，对正例损失和的总贡献均为 1；
`P` 个自平局各贡献 `1/2`，故正例内部贡献为
`P(P-1)/2+P/2=P^2/2`。每个正例--未标注对对损失和的贡献恰为 AUC 得分的补数，
所以跨类贡献为 `PN(1-AUC_h)`。两项相加、除以 `n`，再除以 `P` 即得。

该命题只说明评价统计量使用了与 selector 相同的 midrank 损失函数。selector 的 oracle
在完整 1,805 RNA 宇宙、来源中心和 PU 模糊集上比较最坏风险差；episode AUC 则在去除
anchors 后的候选集上重算。两者不是同一个数值目标，命题也不推出 MC-TV-PU 必然提高
经验 AUC，更不推出 AP 提升。

正式 summary 必须覆盖全部 `center × tau` 格点并把 AUC 证据分为：两个预设基线的普通
95% 下界均大于 0、两个点估计均大于 0但区间未全通过、任一基线普通 95% 上界小于 0、
以及其余混合/不确定四类。该分类不参与 `practical_selection_passed` 或 `decision`，防止把
次要、未同时校正的 AUC 区间事后升级为主成功门。
若 AP 主门通过，机制 claim routing 使用所有 AP 强连续窗口所覆盖格点的并集：并集内全部
ordinary-95 支持才允许写普通区间一致，任一反向格点即标记冲突，全部至少为正点估计才写
点一致但区间不确定，其余保持混合/不确定。不得选择其中单个有利窗口。主门失败时不作机制
支持主张。

N2 性能消融必须在相同 source fold、候选分数和 held-out episode 上比较 compatible、匹配
矩形与经验中心三种 selector。每个规则均取其选择集合内的最差成员，先在 drug 内平均 folds，
再对 drug 等权；compatible 减两个消融规则的 AP 差异构成独立的 48 项 simultaneous family。
该 family 不与主 AP family 合并，也不改变主 success gate。正式 runner 必须恰好输出
`2 centers × 12 tau × 2 ablations=48` 行，否则拒绝生成最终 summary。
该独立 family 以中心化 bootstrap 偏差最小值的 2.5% 分位数和最大值的 97.5% 分位数形成
联合双侧 familywise 95% 带。若主 AP 门通过，N2 claim routing
使用所有 AP 强连续窗口覆盖格点的并集：全区域下界均大于 0 为
`SIMULTANEOUS_95_DOMINATES_ACROSS_AP_STRONG_REGION`；任一上界小于 0 优先为
`SIMULTANEOUS_95_WORSE_WITHIN_AP_STRONG_REGION`；全部点估计非负但未达到前两类为
`POINT_NONNEGATIVE_SIMULTANEOUS_95_INCONCLUSIVE`；其余为
`MIXED_OR_INCONCLUSIVE_WITHIN_AP_STRONG_REGION`。主门失败时为
`PRIMARY_GATE_FAILED_NO_RULE_ABLATION_CLAIM`。这些状态不回写主 `decision`。

理论要求 `q @ s = 0`。实现先强制检查残差不超过 `centering_tolerance`，随后显式执行
`s <- s - q@s`；因此输出对应重中心化后的 `s`。原始残差和 `recentered=true` 会写入
结果 diagnostics，防止调用者把投影后的问题误认为未经调整的输入问题。

## 输出

不可变结果对象，至少包含：

- 目标值的 `lower_bound` 与 `upper_bound`；
- 最佳有限见证 `best_t`、`best_value`；
- 相容阈值 `t_min` 与先验上确界 `prior_supremum`；
- LP 调用数、剩余区间数、是否达到请求的结构间隙；
- `t -> 1+` 和 `t -> infinity` 的适用边界值；
- 可序列化的诊断信息。

若先验区间内不存在合法且相容的 `pi in (0,1)`，抛出专用异常，不返回伪结果。

## 数值证书边界

实现利用 `G(t)` 的单调性、凹性及 `t/(t-1)` 的单调性形成条件性分支定界区间。
但当前 HiGHS 浮点结果没有经过向外舍入，也没有从 primal/dual residual 推导严格的
目标误差外包络。因此普通 LP 路径即使达到 `absolute_gap`，也只能设置
`structural_gap_met=true`，必须保持 `certified=false`。只有无需 LP 的单支持退化
情形可设置 `certified=true`。这不替代论文中的数学证明；若论文使用普通浮点路径，严格
数值证书仍是阻断项。下方 exact 扩展关闭有理 plug-in 上的有限区间与两个开放端点；它不
关闭经验分布到真实总体的覆盖问题。

## 成功命令

```bash
/Users/jackie/Documents/Git/.venv/bin/python -m pytest -q tests/test_mc_tv_pu_oracle.py
/Users/jackie/Documents/Git/.venv/bin/python -m pytest -q tests/test_mc_tv_pu_score_input_gate.py
```

## 基线提交

`f1358d53299bb33ccb3b23bb5d7f9e61195f2caa`

LocalGL 干净复现内存修复的增量基点为
`fa15cb5`。该修复只允许改变 Python/MPS 对象生命周期与缓存释放，不得改变图采样种子、
批次内容和顺序、模型、损失、优化器、训练预算或分数导出契约。

正式 selector exact 计算性能优化的增量基点为 `3dcf1a4`。该优化只允许消除重复求值或
减少等价数据结构开销；不得改变 active drug、候选、center、tau、exact gap、最大求值预算、
checkpoint schema 或任一 checkpoint 单元格的值。已有 checkpoint 只有在旧新实现逐值一致后才能复用。

跨 `tau`/center 的 exact score-order 缓存增量基点为 `18b6fe8`。缓存键必须是完整且精确相等的
`Fraction` 方向元组，缓存值只能是既有稳定排序的不可变索引元组；容量上限为 256，淘汰不得改变
任何求值或落盘结果。正式运行迁移前须验证命中/未命中逐值一致、缓存有界且重复调用确有命中。

prepared 外层 exact value-only 路径的增量基点为 `953fd20`。公开 fixed-`t` API 及其完整 `p`
witness 保持不变；内部路径只允许按既有运输顺序累计同一 exact objective，不缓存未被外层使用的
witness。旧新 `ExactOracleResult.to_dict()`、evaluation count 与正式 checkpoint 必须逐值一致。

E8i schema-v2 到 schema-v3 不可变迁移的增量基点为 `e33040a`。该基点已经包含
center-specific empirical 修复与 schema-v3 接纳门；本梯级只实现旧 checkpoint 的显式、
可拒绝、可重复导入，不改变 oracle 数值、候选库或 selector 规则。

两个 source center 在 `tau=1` 时只共享 compatible oracle 与匹配矩形结果；经验中心消融仍依赖
各自的 `p0`，必须在共享行写入前按当前 center 重算。旧 schema-v2 checkpoint 若含跨 center
复用的 `empirical_value`，不得原地覆盖或直接聚合，只能经逐文件验证迁移到新的不可变目录。

修复后的 checkpoint run configuration 使用 schema v3，并声明
`checkpoint_semantics=center_specific_empirical_v1`。恢复计算前须用当前冻结分数和当前 center
逐值复核所有已有 `empirical_value`；任一不一致时，在写入任何缺失 checkpoint 前整体拒绝该 drug。
输出目录若已有 checkpoint 却没有 `run_config.json`，不得由新运行自动认领。

schema-v2 导入只能通过显式 `--checkpoint-source-dir` 进入不同的 schema-v3 输出目录。源配置除
`schema_version` 和新增的 `checkpoint_semantics` 外必须与当前冻结配置逐项相同；迁移对每个
checkpoint 保留全部非 empirical 列，只把 empirical 列替换为当前 center 的精确重算值，并拒绝
已知 `positive_pair_equal × tau=1 × 非对角` 范围之外的任何变化。中断后重跑时，既有目标文件
必须与同一内存 proposal 全表逐值相等；稳定迁移报告记录源/目标数量、修正单元格数和文件清单哈希。
源、目标树不得相互嵌套；整个源 `oracle/` 子树不得含规范 checkpoint 集以外的文件。迁移输入的
drug ID 必须唯一且恰好覆盖冻结 active drugs，并按 drug ID 与规范相对路径排序后形成报告。对每个
drug，先验证全部既有目标，再写任何缺失目标。`--migration-only` 只与显式源目录同时使用；它只写
schema-v3 配置与迁移报告并退出，不启动新 oracle 求值或 selector 聚合。

## 范围边界

- E8 导出钩子实现阶段不运行 ncRD-LG、LocalGL 或任何新模型训练；E8e 只允许对已经冻结的
  LocalGL 候选做数值等价的内存修复验收与干净五折重放，不新增候选、不改训练语义；
- 不从当前 Mi 数据自动构造或宣称生物总体 `Q`；
- 不把数值实现通过写成论文新颖性或经验有效性已经成立；
- 不宣称一般 min-cost-flow 等价；底层固定 `t` 问题使用线性规划。

## Exact reference 扩展

新增独立 exact API：输入只允许 `Fraction`、整数或有理数字符串，固定 `t` 内层使用
有理质量搬运算法，外层用有理区间分支定界。它返回规范分数字符串和严格 enclosure；只有
间隙达标才 `certified=true`。若合法域从开端点 `t=1` 开始，以 TV 切锥的一维精确对偶计算
右极限，并由凹性上界覆盖相邻开区间；`t=1` 仍不作为合法先验见证。`t->infinity` 的饱和
尾部同样纳入证书。该后端是审计参考，不把经验分布的 plug-in 精确性升级为总体覆盖保证。

## 匹配矩形基线与包含命题

本节固定全支持有限查询分布 `Q`，并令

\[
\mathcal P_\tau=\{p\in\Delta:\operatorname{TV}(p,p^0)\le\tau\}.
\]

对请求先验区间 `I=[pi_lower,pi_upper]`，完整相容集合为

\[
\mathcal A_{\mathrm{comp}}=
\{(\pi,p):\pi\in I\cap(0,1),\ p\in\mathcal P_\tau,\
p_i\le Q_i/\pi\ \forall i\}.
\]

令 `t_min` 为存在 `p in P_tau` 且 `p <= tQ` 的最小 `t`，并定义

\[
\bar\pi_{\mathrm{comp}}=
\min\{\pi_{\mathrm{upper}},1/t_{\min}\}.
\]

匹配矩形只删除逐坐标耦合，不放宽标量先验上界：

\[
\mathcal A_{\mathrm{rect}}=
\bigl([\pi_{\mathrm{lower}},\bar\pi_{\mathrm{comp}}]\cap(0,1)\bigr)
\times\mathcal P_\tau.
\]

因此有直接包含关系

\[
\mathcal A_{\mathrm{comp}}\subseteq\mathcal A_{\mathrm{rect}}.
\]

对任意满足 `Q^T s=0` 的有向 loss difference，风险差写成

\[
D(\pi,p;s)=\frac{p^\top s}{1-\pi}.
\]

令 `M_tau(s)=max_{p in P_tau} p^T s`。矩形集合中 `p` 与 `pi` 独立，所以其 supremum 有闭式：

\[
C_{\mathrm{rect}}(s)=
\begin{cases}
+\infty,&M_\tau(s)>0,\ \bar\pi_{\mathrm{comp}}=1,\\
M_\tau(s)/(1-\bar\pi_{\mathrm{comp}}),
&M_\tau(s)>0,\ \bar\pi_{\mathrm{comp}}<1,\\
0,&M_\tau(s)=0,\\
M_\tau(s)/(1-\pi_{\mathrm{lower}}),&M_\tau(s)<0.
\end{cases}
\]

最后一行在 `pi_lower=0` 时按右极限解释。集合包含立即给出

\[
C_{\mathrm{comp}}(s)\le C_{\mathrm{rect}}(s).
\]

这个不等式只比较同一个有向候选差；它不意味着两个 minimax-regret selector 必须选择同一候选。`NONTRIVIAL_SEPARATION.md` 的均匀四点例使用相同 `pi_comp=5/8`，解析证明匹配矩形风险为 `(14/15,2/3)`、选择 B，而完整相容风险为 `(3/5,2/3)`、选择 A，证明逐坐标耦合可以严格收紧不同方向并改变最终动作。该例与闭式共同构成匹配消融的理论依据。
