# MC-TV-PU selector-anchor 来源交叉验证覆盖审计

状态：`CROSSFIT_COVERAGE_PASS_NO_MODEL_RESULTS`

审计日期：2026-10-21

本审计在修正版 selector-anchor source cross-fit 预分析冻结后运行，只读取来源表与冻结 Mi 正矩阵；未读取候选模型分数，未计算 selector、AP 或 Recall。原 leave-one-PMID 覆盖计数因跨 episode 来源回流已降级为历史 diagnostic，不再证明 Publication selector-anchor 来源隔离覆盖。本审计不检查或证明底层模型训练按 PMID 隔离。

## 固定输入

- 来源表 SHA-256：`22f2657460577b584f748536eea0592a9ae6d63bcabfc402b7d34c61ed1c4710`
- `Dataset/MiPair.npy` SHA-256：`2d5314944daf71aea5ecd13180fd7cb18936cd192882477c24f5ed873c8c4a73`
- 来源表共有 32,969 行、3,121 个原始唯一 PMID；当前 loader 将 `PMID` 读取为
  `int64`。全部原始 PMID 和实体映射后的 1,438 个 source ID 均为非空 8 位数字，
  未发现首尾空格、`.0`、科学计数法、前导零或复合 PMID。限制到冻结正矩阵后剩余
  1,421 个来源。该格式检查只证明本次固定来源表不存在 ID 拆分歧义，不证明 PMID
  本身是随机或无偏来源。
- 原 diagnostic 实现提交：`1317afd3b053ba8b05a697f897a0ec5ec28a2225`。
- 修正版构造器：`build_source_crossfit_episodes`；Publication 入口 `mc_tv_pu_source_crossfit_runner.py` 与合成端到端测试已在性能结果前实现。
- 来源 pair 先限制到冻结 `MiPair.npy` 的记录正例，再把 1,421 个规范化 PMID/source ID 在全数据集上一次性分入 5 folds。
- 分配时依次拼接 `20261021` 的 UTF-8 字节、一个 NUL 字节 `0x00` 和 source ID 的 UTF-8 字节，取 SHA-256 前 8 字节按大端解释后对 5 取模；不读取模型分数或 held-out 指标。
- 规范化 source-to-fold 映射按紧凑、键排序 JSON 序列化后的 SHA-256 为
  `8081265014af42145efa96f96e1b5571f67a981aac4294a6edb1207acc8072af`；正式 runner
  使用同一个 `_assignment_digest` 记录该值。
- 对 fold `f`，属于 `f` 的来源不进入该 fold 任何药物的 anchor；held-out RNA 若同时出现在该 drug--fold 的任一 anchor 来源中，则不计作 held-out 新记录。

## 覆盖结果

| 项目 | 数量 |
|---|---:|
| 全局唯一来源 | 1,421 |
| 同时具有 anchor 与 evaluation 来源的 drug--fold | 612 |
| 至少一个可评价 drug--fold 的药物 | 208 |
| held-out 新记录非空的 drug--fold | 563 |
| held-out 新记录为空的 drug--fold | 49 |
| drug--fold held-out 新记录总数 | 6,408 |
| 至少一个可评价 drug--fold 的 folds | 5/5 |

各 fold 的覆盖为：

| Fold | 来源 | 合法 drug--fold | 可评价 | 空 held-out | held-out 记录 |
|---:|---:|---:|---:|---:|---:|
| 0 | 302 | 134 | 125 | 9 | 972 |
| 1 | 269 | 126 | 120 | 6 | 1,900 |
| 2 | 268 | 106 | 95 | 11 | 1,258 |
| 3 | 291 | 125 | 110 | 15 | 1,143 |
| 4 | 291 | 121 | 113 | 8 | 1,135 |

修正版预分析沿用不少于 100 个可评价药物、200 个非空 drug--fold 和 500 条 held-out 记录的覆盖门，并额外要求至少 4/5 folds 可评价。四个条件均通过，因此**数据覆盖允许**继续实现 Publication cross-fit；这不是性能门通过，也不授权运行旧 diagnostic runner 代替 cross-fit。

## 证据边界

覆盖通过只说明可评价样本量足够，不说明 MC-TV-PU 会选择出更好的模型，也不说明 PMID 是未来自然总体的随机样本。性能主张仍须等待 15/15 分数输入门、每 fold 全部 exact oracle 证书、drug-level bootstrap 和预注册成功门共同通过。

历史 leave-one-PMID 计数为 234 个多来源药物、1,192 个 episodes、229 个可评价药物、952 个非空 episodes 和 6,667 条 held-out 记录；这些数字只保留用于重现旧 diagnostic，不再称为来源隔离覆盖证据。
