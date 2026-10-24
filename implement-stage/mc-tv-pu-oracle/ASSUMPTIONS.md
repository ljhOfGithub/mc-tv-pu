# 假设账本：MC-TV-PU 条件最坏风险差求解器
<!-- ASK mode: never -->

| ID | 请求未确定的事项 | 选择 | 类别 | 来源 |
|---|---|---|---|---|
| A-001 | “精确 oracle”在代码里如何表述 | 返回条件性浮点区间；普通 LP 路径即使结构间隙达标也保持 `certified=false`，只有解析退化情形可认证 | semantic | default |
| A-002 | `Q` 的语义由谁确定 | `Q` 必须由调用者按明确 estimand 提供；求解器不从 Mi 的未标注 pair 自动推断生物总体 | semantic | default |
| A-003 | `s` 是否可为任意向量 | 必须有限且 `abs(q @ s)` 不超过容差；通过后显式执行 `s <- s - q@s`，使实际求解对象严格落在理论中心子空间 | semantic | default |
| A-004 | 损失上下界如何进入实现 | 求解器不重建损失；可选 `loss_bound` 只检查 `s` 与最终界是否落在理论允许范围，不做裁剪 | semantic | default |
| A-005 | TV 的定义 | 使用 `TV(p,p0)=0.5*sum(abs(p-p0))` | semantic | default |
| A-006 | `Q_i=0` 如何处理 | 相容约束强制 `p_i=0`；若这些坐标上的 `p0` 质量超过 `tau`，歧义集永不相容 | semantic | default |
| A-007 | 先验端点是否合法 | 只优化 `pi in [lower,upper] intersect (0,1)`；`pi=1` 与 `pi=0` 仅按单侧极限处理 | semantic | default |
| A-008 | `t=1` 的奇点如何处理 | 仅当 `Q` 在 TV 集内时计算方向 LP，得到 `t -> 1+` 极限；不把 `pi=1` 当合法见证 | semantic | default |
| A-009 | `t -> infinity` 如何处理 | 达到所有正 `Q_i` 上的 cap 饱和点后 `G` 恒定；依据其符号选择有限端点或无穷极限 | semantic | default |
| A-010 | 全局优化如何停止 | 用单调/凹结构的区间上界做确定性分支定界，达到 `absolute_gap` 或 `max_lp_evaluations` 停止 | semantic | default |
| A-011 | LP 后端 | 使用 SciPy HiGHS `linprog` 与 `float64`；任何非成功状态均显式报错 | interface | default |
| A-012 | 容差如何暴露 | `probability_tolerance` 只用于输入合法性/归一和报告，不改变 TV 内外、零支持、先验交集或候选域的集合语义；`q @ s` 与结构间隙另有参数 | interface | default |
| A-013 | 无相容合法先验如何处理 | 抛出 `IncompatibleAmbiguitySet`，消息含相容阈值和请求区间 | interface | default |
| A-014 | API 范围 | 纯函数式数组 API，不读取数据文件、不写结果文件、不启动训练 | interface | default |
| A-015 | 认证对 SciPy 数值误差的依赖 | 当前 `lp_value_tolerance` 未由 HiGHS 残差推导，不能称严格外包络；仅报告条件性结构间隙并保持 `certified=false` | semantic | default |
| A-016 | 初始求解器阶段的工作树如何提交 | 当时分支含大量既有未跟踪材料及三处先存 tracked 修改，因此初始梯级不制造混合提交；当前 E8i 迁移阶段的独立提交边界改由 A-051 记录 | interface | default |
| A-017 | exact API 接受什么数值 | 只接受 `Fraction`、整数或有理数字符串；拒绝二进制浮点，避免把近似输入冒充用户未声明的精确对象 | interface | default |
| A-018 | 小数如何进入 exact API | 调用者以字符串冻结十进制值，例如 `"0.125"`；证书只对该序列化后的有理 plug-in 问题成立 | semantic | default |
| A-019 | exact API 如何处理 `q@s` 残差 | 必须严格等于零，不执行重中心化；否则报错，防止证书对应另一个目标 | semantic | default |
| A-020 | 固定 `t` 如何精确求解 | 将强制 cap 移出质量、最高收益补回和正收益的最低到最高分数搬运分开，用有理贪心求解；不调用浮点 LP | semantic | default |
| A-021 | exact 外层是否覆盖 `t->1+` | 已被 A-026--A-028 取代：合法域开端点为 1 时使用精确切锥对偶和开区间 enclosure，不再拒绝该域 | semantic | sweep |
| A-022 | exact 外层是否覆盖 `t->infinity` | 支持；cap 饱和后 `G` 恒定，按符号在饱和起点或无穷极限取上确界 | semantic | default |
| A-023 | exact 外层的证书含义 | 固定点值与区间上下界均用 `Fraction`；达到请求的有理绝对间隙才 `certified=true`，预算耗尽仍返回有效但较宽的 enclosure | semantic | default |
| A-024 | exact 后端的角色 | 作为审计/参考后端和小支持严格证书，不宣称已替代大规模浮点路径；规模与运行时单独报告 | semantic | default |
| A-025 | exact 结果如何序列化 | Python 对象保留 `Fraction`，`to_dict()` 用规范 `numerator/denominator` 字符串，避免 JSON 浮点化 | interface | default |
| A-026 | exact API 如何处理 `t\downarrow1` | 当 `Q` 位于 TV 球内时，用 TV 切锥 LP 的一维精确对偶闭式计算右极限；`t=1` 仍不是合法先验见证，若极限最优则 `best_t=None` | semantic | default |
| A-027 | exact 切锥边界如何判定 | 使用 `Fraction` 对 `TV(Q,p0)` 与 `tau` 做严格小于、等于或大于判断，不把浮点容差引入 exact 世界 | semantic | default |
| A-028 | exact 外层如何覆盖以 1 为开端点的区间 | 由凹性使用 `G(t) <= (t-1)D_+G(1)`，故 `(1,r]` 上 `F(t)<=rD_+G(1)`；按有理中点细分直至间隙达标 | semantic | default |
| A-029 | 当前任务中的 `Q_d` 表示什么 | 固定为从 1,805 个 RNA 候选身份中均匀抽取的 full-candidate 设计分布；只解释有限候选池排序，不称自然生物总体 | semantic | default |
| A-030 | 来源中心 `p0_d` 如何定义 | 在来源表与冻结 Mi 正矩阵的交集上，每个 PMID 获得相同总质量，PMID 内每个唯一 RNA 再等分；同一 PMID 的重复条件不增加质量 | semantic | default |
| A-031 | 是否从当前快照估计 `tau_d` | 不估计；来源间 TV 只作异质性诊断，`tau` 保留为完整敏感性轴，不提供置信覆盖 | semantic | default |
| A-032 | 来源异质性审计的单位 | 只纳入至少两个非空 PMID 的药物；按 PMID 去重 RNA，报告 source-to-center、source-pair 与 leave-one-source-out TV | semantic | default |
| A-033 | E7 输出能否用于选择论文实验参数 | 不能；E7 只判断小 `tau` 是否有来源证据，不读取模型分数、不选择候选、不作为冻结 test 上调参依据 | semantic | default |
| A-034 | 来源表中映射成功但不在冻结 Mi 正矩阵的 pair 如何处理 | 100 个 source-only pair（含 1 个额外药物）只报告并排除，不进入 `p0`；防止较新来源快照泄入冻结任务 | semantic | sweep |
| A-035 | E8 的候选分数必须覆盖哪些 pair | 每个冻结候选都必须对 606 个有正例药物与 1,805 个 RNA 的完整笛卡尔积提供唯一、有限点分数，即每模型 1,093,830 行 | semantic | default |
| A-036 | 现有 sampled-unlabeled OOF 分数能否代替 full-candidate 分数 | 不能；它每模型只有 18,790 行，改变 A-029 的均匀候选查询分布，且缺少绝大多数 RNA 坐标上的损失差 | semantic | sweep |
| A-037 | per-drug AP/召回率等聚合指标能否重建 oracle 输入 | 不能；MC-TV-PU 需要每个 RNA 坐标上的 pairwise 损失差向量 `s`，标量排序指标是多对一压缩 | semantic | sweep |
| A-038 | 纸面门中“例如”的模型类别是否已冻结为具体产物清单 | 不是；在观察 anchor/选择输出前必须单独冻结确切的模型名、运行目录与文件哈希，不得从 43 份既有 Mi pilot 中事后挑选 | semantic | default |
| A-039 | 从哪些已有运行冻结候选库 | 取“早期六模型五折 confirmation 全集”与“最终 Mi 匹配基准包中的 prior、prior-residual、LocalGL、PLMF-MDA、sChemNET”并集；只要同为 Mi、5 折、`split_seed=20261019`、覆盖 606 个活跃药物就纳入，不按 AP 阈值剔除 | semantic | default |
| A-040 | 哪些已有方法不进入同一 Mi selector | RAHGL 只覆盖 197 个药物且 split 不同；HMVCC 属于其他数据集；`split_seed=20261020` 的重复 confirmation、smoke 和未进入最终基准包的 exploratory pilot 均不混入 | semantic | default |
| A-041 | 本次冻结在什么时点发生 | 已经看过各底层模型的历史基准指标，但没有运行或观察过 MC-TV-PU anchor/selector 输出；因此只能声称 selector 前冻结，不得写成原模型开发前预注册 | semantic | default |
| A-042 | 候选的原运行源码能否从 Git 重放 | 不能默认；相关脚本均未跟踪，多数结果配置未记录源码哈希，PLMF 记录的源码哈希与现文件不同；manifest 必须分开“冻结候选身份”与“可原样重放” | semantic | sweep |
| A-043 | PLMF-MDA 和 sChemNET 的三个训练种子如何进入候选库 | 每个种子作为独立、带 seed 后缀的冻结 ranker；不事后挑最优 seed，不新造未评估的均分 ensemble，候选数与有限库复杂度按 15 计 | semantic | default |
| A-044 | 1,640 万行完整分数如何落盘 | 按 `model/fold` 分片为 Parquet（Zstandard），每片只含 `model,fold,drug_id,rna_id,score`；不写入 label，不要求将全库一次读入内存 | interface | default |
| A-045 | 导出钩子能否改变训练或排序 | 不能；钩子只在既有 `full_predictions` 已计算后写盘，分割、超参数、候选集和分数数值均不变；这里的“不执行训练”仅约束 E8 导出钩子实现阶段，不覆盖 A-055 所定义的 E8e 等价修复验收与干净五折重放 | semantic | default |
| A-046 | 不同候选可否给同一药物使用不同外折 | 不可以；所有候选必须对每个药物使用完全相同的外折编号，折编号必须是整数且全集严格为 `0..4`，否则整个 selector 输入阻断 | semantic | sweep |
| A-047 | selector 的正式输入是否允许 CSV、额外列或缺失 `fold` | 不允许；正式 CLI 只接受 manifest 绑定的分片 Parquet，且每片列顺序严格为 `model,fold,drug_id,rna_id,score`；不保留可 PASS 的 CSV/手写模型列表旁路 | semantic | sweep |
| A-048 | 导出训练何时验证 manifest，哪些运行参数可沿用同一候选 ID | 在加载数据或开始训练前验证整个 manifest、输入哈希、入口生成器哈希、核心依赖代码哈希和导出工具哈希；启用导出时，改变 split、fold 数、训练预算、device、模型/负采样/药物视图或 LocalGL 采样语义的参数必须与 source-group runtime contract 完全一致，否则拒绝启动 | semantic | sweep |
| A-049 | `prior_preserving_pairwise_residual` 的冻结药物视图和证据目录 | 最终 Mi 候选是 `results/chemberta-pairwise-residual-pilot-20261019` 的 ChemBERTa 运行，不是同名的早期 ECFP 运行；manifest 改指该结果、配置和固定 ChemBERTa 特征哈希 | semantic | sweep |
| A-050 | 哪些原始数据文件决定当前导出候选身份 | 共同锁定 `MiPair.npy`、`MiPair_new.npy`、`MiRNA.pt` 和 `mi_drug_with_SMILES.csv`；native confirmation 与 LocalGL 还锁 `MiDrug.pt`，ChemBERTa 候选锁预计算特征，PLMF-MDA 锁 RNA-FM/ChemBERTa 特征、特征清单和 miRNA 序列表，防止只保持矩阵形状却更换表征或 scaffold 分组 | semantic | default |
| A-051 | 当前项目能否按 Skill 形成本梯级的独立提交 | 初始求解器梯级基点仍为 `f1358d5`；当前 E8i 迁移梯级位于 `feat/mc-tv-pu-selector-evaluation`，增量基点为 `e33040a`。迁移实现、回归、账本和审计作为一个小提交，不混入正式 selector 结果或旧目录产物 | interface | default |
| A-052 | PLMF-MDA 与 sChemNET 的训练 seed 如何绑定冻结 candidate ID | 只读取 manifest 中每个候选显式的整数 `training_seed`，映射为唯一 seed-qualified ID；不得依赖列表顺序、动态拼写 ID 或跨 seed 平均后导出一个新候选 | semantic | default |
| A-053 | PLMF 的旧断点 metrics 能否在分数导出模式直接恢复 | 不允许。即使同路径存在不可变 Parquet，也没有独立证据证明它与当前 progress fingerprint 一致；导出模式发现 completed-unit 就立即阻断，必须用干净输出目录重新生成分数 | semantic | default |
| A-054 | LocalGL 的 `device=auto` 能否视为已冻结设备 | 不能；`auto` 会随机器解析为 MPS、CUDA 或 CPU，而历史五折配置记录为 MPS。正式导出默认与 runtime contract 均显式锁为 `mps`，其他后端不得沿用该候选 ID | semantic | sweep |
| A-055 | LocalGL 导出运行因 `EXIT_STATUS=137` 中断后，哪些内存修复仍可沿用冻结候选 ID | 只允许在每个重采样 epoch 结束后释放不再使用的图、批次、前向张量和设备缓存，并避免让首轮预批次永久驻留；图种子、批次内容和排列、模型、损失、优化器、epoch 数、设备与导出 schema 全部保持不变。修复后必须从新的干净输出目录重跑全部五折，旧 fold 0 不与新实现混用 | semantic | default |
| A-056 | 正式 selector 的 1,212 个 exact checkpoint 预计需约 40--50 小时时，哪些性能优化仍可复用已完成 checkpoint | 只允许消除重复 exact 求值或降低等价 Python 数据结构开销；active drug、15 候选、两个 center、12 个 tau、`absolute_gap=1/1000`、`max_evaluations=4096`、有理输入、checkpoint schema 与每个输出单元格全部保持不变。必须在未用于论文解释的冻结输入 drug 上验证旧新 checkpoint 逐值一致并报告稳定加速，之后才可让新实现验证并复用既有 checkpoint | semantic | default |
| A-057 | 同一 exact loss-difference 在多个 `tau`、center 或连续药物调用间能否复用排序顺序 | 只对逐个 `Fraction` 完全相等的不可变方向元组复用稳定 recipient/donor 顺序；缓存固定最多 256 个方向，淘汰只影响性能。排序键、索引 tie-break、oracle API、gap、预算、checkpoint schema 与输出单元格均不得改变；命中和未命中路径必须逐值相等 | semantic | default |
| A-058 | prepared 外层 exact oracle 是否必须为每个 branch-and-bound `t` 构造并缓存完整的 1,805 维 `p` witness | 不必须；公开 `solve_fixed_t_exact` 继续返回完整 witness，只有不暴露 fixed-`t` witness 的 prepared 外层路径改用相同 mandatory redistribution 与 optional transport 顺序的 exact value-only 累加。`t`、value、evaluation count、区间、certificate、diagnostics、gap、预算、checkpoint schema 与所有输出单元格必须逐值不变 | semantic | default |
| A-059 | 两个 source center 在 `tau=1` 时可以复用哪些结果 | 只复用已证明与 center 无关的 compatible oracle 与匹配矩形字段；`empirical_value` 仍按当前 center 的 `p0` 逐候选对精确重算。含旧跨 center empirical 复用的 checkpoint 不得进入聚合，修复后只能迁移到新的不可变输出目录并保留旧目录作失败证据 | semantic | default |
| A-060 | schema-v2 checkpoint 可在什么条件下迁移 | 源目录必须有与当前冻结配置除 schema/语义标识外逐项相同的 run config，且无 unfinished 文件；源、目标目录解析后不得相同或相互嵌套。必须枚举并核对整个 `oracle/` 子树，只接受期望 drug 与两个 center 的规范 checkpoint 路径；score items 的 drug ID 必须唯一且与冻结 active drugs 集合完全相等。迁移只允许 `positive_pair_equal × tau=1 × 非对角` 的 `empirical_value` 与当前 center 重算值不同，其他任何 empirical 漂移均拒绝；所有非 empirical 单元格原样保留。每个 drug 的所有既有目标必须先逐表 exact equality 预检，再写任何缺失目标；报告按规范相对路径排序，重复迁移及 score 输入重排不得改变报告 | semantic | sweep |
| A-061 | 如何把已验证迁移与昂贵 selector 计算分开验收 | 正式 CLI 提供 `--migration-only`，且只允许与 `--checkpoint-source-dir` 同时使用；它完成同一套输入门、schema-v2 配置接纳、schema-v3 run config、逐文件迁移和稳定报告后退出，不生成新 oracle checkpoint、不聚合 selector 结果。后续恢复使用同一输出目录和相同迁移参数重新验证迁移，再进入正式 selector | interface | default |
| A-062 | source-crossfit 的 pairwise AUC 是否等于 selector 的完整数值目标 | 不等于。只有在同一个去除 anchors 后的 episode 候选集上重算 midrank loss 时，held-out 正例平均损失才与 tie-aware AUC 仿射等价；selector 使用完整 1,805 RNA 宇宙、来源中心和 PU 模糊集。AUC 是损失函数一致性诊断，不是 AUC 或 AP 提升保证 | semantic | default |
| A-063 | N2 规则性能消融何时允许形成机制主张 | 只在 AP 主门通过后，读取所有 AP 强连续窗口所覆盖格点的并集；matched-rectangle 与 empirical-center 共同组成一个独立于主 AP family 的 48 项 family，使用 paired drug-bootstrap 联合双侧同时 95% 带，分别以中心化偏差最小值的 2.5% 分位数和最大值的 97.5% 分位数形成上下偏移。全区域下界均大于 0 才称 compatible 占优，任一上界小于 0 先判反向证据，全部点估计非负但未通过同时带只称不确定；其余保持混合。该路由不改变主 success gate | semantic | default |
| A-064 | Publication source-crossfit 如何绑定产生结果的实现 | 在首次写 run config 前计算 runner、统计、selector、oracle、输入门、来源语义与 scaffold/source 解析共 8 个直接本地依赖的 SHA-256，并写入不可变 `implementation_source_sha256`；任一文件变化时不得复用既有输出目录 | interface | default |
| A-065 | 如何防止终态 summary 与其他批次的配置、fold assignment 或结果表混用 | 在写 summary 前对现有 run config、source-fold assignment 及待写入的 candidate intervals、episode metrics、主推断、规则消融四张 CSV 计算 SHA-256，并把六项摘要写入 `artifact_sha256`；summary 只与这些确切工件共同构成正式结果包 | interface | default |
| A-066 | 记录哈希后是否还需要验证实际落盘文件 | 需要。runner 在成功退出前重新读取 summary，要求六个键完整、对应文件均存在、实际 SHA-256 逐项相等且递归范围内无 `.tmp.parquet`/`.partial`；任一不满足即非零退出 | interface | default |

## 关键解释

- A-002 是当前研究的硬边界：若以后选择“全候选均匀分布”作为 `Q`，论文必须把
  estimand 明写为有限候选宇宙，而不能把它升级为自然世界靶点分布。
- A-008 的方向 LP 只用于封闭开端点的上确界；它不会返回 `best_t=1` 作为合法先验。
- A-015 意味着普通浮点路径没有通过严格数值证书门；论文主张仍以
  `SPEC.md` 的条件性命题、`NONTRIVIAL_SEPARATION.md` 的决策分离和 exact 回归证据为准，
  不能把 `structural_gap_met` 写成已认证。当前不存在独立 `PROOF_PACKAGE.md`；exact API
  对其声明支持的有理 plug-in 域提供严格 enclosure，不改变浮点路径状态。
- A-018/A-024：exact 证书首先解决“算法与算术是否可审计”，不自动解决经验
  `Q_d`、`p_d^0`、`tau_d` 是否覆盖真实总体；后者仍是独立科学语义门。
- A-026/A-028 只闭合参数域的开端点；它不把 `pi=1` 加入合法世界，也不改变
  `prior_supremum=1` 只是上确界的解释。
- A-029 把 $Q_d$ 从“未知自然总体”收缩为可复核的查询设计分布；对应保证只适用于随机查询
  RNA 的设计风险。若把每个 pair 的生物真值解释为固定二元常数，当前随机标签 LP 仍是松弛。
- A-031/A-033 阻止把观察到的来源差异倒算成有利半径；E7 只能否定不可信的小半径，不能认证
  某个半径覆盖未来或未报道正例分布。
- A-034 的交集恰好覆盖全部 9,395 个冻结 Mi 正 pair；排除项不是负例，也不用于性能评价。
- A-035--A-037 把“有 full-candidate AP”与“有 full-candidate 点分数”分开：前者只能核验
  既有排序结果，不能生成 Step 13 的有限支持损失差。
- A-038 阻止用现有大量 pilot 事后扩大候选库；在准确产物清单冻结前，E8 只能停在
  输入完整性门，不计算 selector 结果。
- A-039/A-040 是候选资格规则，不是按历史 AP 排名选前几名。六模型 confirmation
  中的全部 scorer 都保留，最终公开方法匹配基准也全部保留。
- A-041 意味着 selector 评估仍受底层模型历史开发偏差影响；后续可以评估“给定这个库如何选”，
  不能把库本身写成从未看过标签的独立 benchmark。
- 分数文件不含 label、且每个目标药物只出现在自己的药物外折，只能证明 target-drug OOF
  坐标契约；它不证明冻结基础模型训练、先验估计或特征构造按 PMID/source 隔离。
  Publication source cross-fit 的隔离范围仅为 selector anchors。
- A-042 要求 clean reproduction 先恢复或重建可哈希锁定的源码快照；直接用当前未跟踪脚本重跑只能称
  新的冻结实现，不能称原结果复现。
- A-044/A-045 使完整分数成为可流式审计的独立产物，并防止在 selector 输入里顺手夹带测试
  label。Parquet 只是存储契约，不改变 A-029 的均匀候选设计。
