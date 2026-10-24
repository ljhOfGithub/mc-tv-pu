# MC-TV-PU 冻结发布包

本目录定义论文所用 MC-TV-PU 结果的公开审查边界。它不启动训练、不恢复
checkpoint，也不搜索新超参数；验证器只检查版本化源码、冻结输入绑定、selector
重算一致性、source-crossfit 终态工件及论文允许的主张边界。

## 冻结结论

- 主 selector 的 1,212 个 checkpoint 完整，可从 checkpoint 重算最终汇总；其
  `practical_selection_gate` 为 `AT_LEAST_ONE_CELL_NARROWS_LIBRARY`。
- source-crossfit 有 1,224 个 oracle checkpoint 和 612 个 episode metric
  checkpoint，六项终态工件的 SHA-256 与 summary 一致。
- 最终成功门是 `FAIL_RETAIN_AS_BOUNDARY_RESULT`。结果支持“预注册 selector 在部分
  单元缩小候选库”这一边界事实，不支持“来源隔离后优于基线”的性能主张。
- 来源隔离只覆盖 selector anchors；基础模型不是 source-disjoint。因此不得据此
  宣称端到端来源外泛化，也不得提出机制或规则消融胜出主张。

## 只读验证

在仓库根目录运行：

```zsh
python mc_tv_pu_release_validate.py \
  --result-root /absolute/path/to/mc-tv-pu-clean-repro-memory-fix-20261021 \
  --pair-matrix /absolute/path/to/Dataset/MiPair.npy
```

成功时输出 JSON，顶层 `status` 为 `PASS`。该结果只证明冻结结果包完整且与本发布
规范一致，不证明从原始数据重新训练基础模型可以得到同一分数。

## 发布层补强

正式 crossfit `run_config.json` 锁定了 8 个直接使用的源码哈希，但实际 Python 导入
闭包还经过 `scaffold_cv_confirm.py`。本目录的 `FROZEN_RESULT.json` 额外锁定完整的
10 文件发布闭包。这是发布完整性补强，不追溯改写原运行记录，也不表示原运行当时
已经记录该间接依赖。

## 文件与许可

- `MANIFEST.txt` 是允许进入独立发布包的文件白名单。
- `MANIFEST.sha256` 绑定白名单中每个文件；验证器会逐项检查。
- 独立原创算法与验证代码按 `LICENSE-CODE-MIT.txt` 授权。
- 原创论文文字与原创方法图按 `LICENSE-PAPER-CC-BY-4.0.md` 授权。
- 上游 ncRD-LG、数据、预训练模型、第三方分数与浏览器/审计轨迹不在这些授权内，
  详见 `THIRD_PARTY.md`。

公共仓库为 `https://github.com/ljhOfGithub/mc-tv-pu`。不可变 `v1.0.0` release
位于 `https://github.com/ljhOfGithub/mc-tv-pu/releases/tag/v1.0.0`，指向提交
`b57ab470de03bcb8ccb02bdbbe5ad5e45789020c`；附带源码归档 SHA-256 为
`cf36b4056b702ff9a3ce32947a2ba13d441880339e05ccfeb83d5c797234b2c1`。
该 release 没有 DOI，也不包含上游数据、score 分片、checkpoint 或第三方特征。
