# 来源与修改记录

- 论文：https://arxiv.org/abs/2502.12213
- 官方仓库：https://github.com/roarer008/STDN
- 基准提交：`e24a60fa9485fb9805df9c171d67ab60840d757d`

`model.py`、`utils.py`、`metrics.py`、`train.py` 以及 PEMS04/07 配置来源于作者仓库。作者和论文的署名、版权归原权利人所有。本项目不声称是官方发布。

新增 `run.py`、`prepare_data.py`、`training_options.py`、依赖说明、数据排除规则和运行文档。PEMS03/08 最初由 PEMS04 配置补充，现参考第三方 Torch-MTS 的 STDN 配置及种子233，详见 [REFERENCE_TORCH_MTS.md](REFERENCE_TORCH_MTS.md)。模型中删除对输入设备已有张量的硬编码 `.cuda()`，由训练入口统一指定设备；训练脚本添加可选随机种子、小规模检查、数值检查、可配置学习率调度和测试输出保存，删除未使用的图工具导入。04/07 配置的设备字段做单 GPU 适配。未重新设计模型结构或损失函数。

基准提交没有 LICENSE 文件。本项目未选择替代许可证；这不等于上游代码已获任意再分发许可。公开发布前应取得必要授权，也可仅发布适配代码和应用于官方仓库的补丁。

不包含任何数据集、上游论文 PDF、预训练权重、个人凭据或完整训练结果。
