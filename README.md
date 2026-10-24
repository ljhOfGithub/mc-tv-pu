# MC-TV-PU

MC-TV-PU 是一个面向**冻结 PU 排序器库**的相容约束 minimax-regret 选择方法。
本仓库公开论文冻结版本的原创算法、测试、协议、图表和审计元数据。

## 结论边界

冻结实验的最终路由为 `FAIL_RETAIN_AS_BOUNDARY_RESULT`：

- 相容耦合在决策上非冗余，并可由严格有理数 enclosure oracle 求解；
- 主 selector 在至少一个预设单元缩小了候选库；
- selector-anchor source cross-fit 未建立相对预设基线的条件性 AP 优势；
- 本结果不证明底层模型按来源隔离，也不支持新骨架外推、真实负例识别、自然总体
  最优性或经验机制优势。

机器可读边界见
[`release/mc-tv-pu/FROZEN_RESULT.json`](release/mc-tv-pu/FROZEN_RESULT.json)。

## 快速检查

建议使用 Python 3.11，并在独立环境中安装审计依赖：

```bash
python -m pip install -r requirements-audit.txt
python -m pytest -q \
  tests/test_mc_tv_pu_oracle.py \
  tests/test_mc_tv_pu_q_tau_semantics_audit.py \
  tests/test_mc_tv_pu_score_input_gate.py \
  tests/test_mc_tv_pu_selector_evaluation.py \
  tests/test_mc_tv_pu_source_crossfit_runner.py \
  tests/test_mc_tv_pu_source_isolated_runner.py \
  tests/test_mc_tv_pu_source_isolated_validation.py \
  tests/test_mc_tv_pu_release_validate.py
shasum -c release/mc-tv-pu/MANIFEST.sha256
```

冻结版本在 committed clean checkout 中为 `200 passed`。完整结果验证还需要用户从
原始 ncRD-LG 来源合法取得 `MiPair.npy` 和本项目的冻结终态结果目录：

```bash
python mc_tv_pu_release_validate.py \
  --result-root /path/to/frozen-result-root \
  --pair-matrix /path/to/Dataset/MiPair.npy
```

成功输出的顶层 `status` 为 `PASS`。该验证只覆盖冻结结果完整性，不会重新训练
基础模型，也不证明从原始数据能够重建上游分数。

## 发布范围

- `release/mc-tv-pu/MANIFEST.txt` 列出 43 个许可进入独立发布包的 payload；
- `release/mc-tv-pu/MANIFEST.sha256` 对 43 项逐一绑定；
- 原始数据、score 分片、上游 checkpoint、第三方特征、论文原文和本地审计轨迹
  不在仓库中；
- 源提交为 `3695bebed830d6f8969f4f2f8e17c3df458a6c8c`。

具体权利边界见
[`release/mc-tv-pu/THIRD_PARTY.md`](release/mc-tv-pu/THIRD_PARTY.md)。

## 许可

- `MANIFEST.txt` 中“独立原创代码”部分：MIT；
- 原创论文文字和原创图：CC BY 4.0；
- 第三方材料不受上述许可覆盖。

## 稿件状态

`paper/main.tex` 是投稿前冻结稿，尚未经过同行评审。其经验主张必须服从上述失败
路由和边界，不得把本仓库描述为普适性能改进证明。
