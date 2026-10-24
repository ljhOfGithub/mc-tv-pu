# 第三方材料与排除范围

本发布规范只授权 `MANIFEST.txt` 明确列出的独立原创 MC-TV-PU 代码、测试、论文文字
和原创方法图，不对整个 ncRD-LG 仓库作统一授权。

以下内容不进入独立发布包，也不受本目录中的 MIT 或 CC BY 4.0 许可覆盖：

- 上游 ncRD-LG 训练代码、原始作者模型、checkpoint 及其衍生文件；
- `Dataset/` 中的数据、特征矩阵、表格和图结构文件；
- `FeatureExtraction/`、`RelevantMethodsComparison/`、`CaseStudy/` 中的第三方材料；
- ChemBERTa、BiRNA-BERT、RDKit、PyTorch、PyTorch Geometric、Transformers 等依赖；
- 冻结 score 分片、下载文件、浏览器记录、机构访问记录和内部审计轨迹；
- 引用论文的正文、表格、图片和补充材料。

截至 2026-10-24 的本地核对，上游仓库根目录未发现可将整个仓库统一再授权的
`LICENSE` 文件。因此公开发布时必须保持上述排除，不得把独立 MC-TV-PU 许可描述为
上游项目的整体许可。第三方依赖和材料分别遵循其权利人给出的条款。
