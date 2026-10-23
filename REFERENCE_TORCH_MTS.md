# PEMS03 / PEMS08 参数参考

按用户选择，03、08采用第三方 Torch-MTS 的公开参数作为参考，04、07仍使用原作者发布参数。不是完整移植第三方实现。

- 配置：[configs/STDN.yaml](https://github.com/XDZhelheim/Torch-MTS/blob/2db4de371584067160f9a37f1ae59495699b4a0a/configs/STDN.yaml)
- 优化器、调度器与默认种子：[scripts/train.py](https://github.com/XDZhelheim/Torch-MTS/blob/2db4de371584067160f9a37f1ae59495699b4a0a/scripts/train.py)
- [03运行日志](https://github.com/XDZhelheim/Torch-MTS/blob/2db4de371584067160f9a37f1ae59495699b4a0a/logs/STDN/STDN-PEMS03-2025-02-20-09-38-49.log)
- [08运行日志](https://github.com/XDZhelheim/Torch-MTS/blob/2db4de371584067160f9a37f1ae59495699b4a0a/logs/STDN/STDN-PEMS08-2025-02-19-20-45-40.log)
- 固定参考提交：`2db4de371584067160f9a37f1ae59495699b4a0a`。

| 参数 | 03 / 08 采用值 |
| --- | --- |
| L / K / d | 2 / 16 / 8 |
| order / reference | 3 / 3 |
| 输入 / 输出 | 12 / 12 |
| batch | 64 |
| 优化器 / 初始学习率 | Adam / 0.001 |
| 最大轮数 / 早停 | 200 / 20 |
| 调度器 | MultiStepLR，milestones=[30,50]，gamma=0.1 |
| 默认种子 | 233，可通过 --seed 覆盖 |

保留原作者训练器和本地统一数据流程：先按时间划分6:2:2，再在各段构造样本；从训练输入计算标准化参数；保留原作者 L1Loss 和 MAE/RMSE/MAPE 实现。没有整体替换为 Torch-MTS 的数据加载、模型适配或训练器。

已知样本数量差异：第三方日志中03训练/验证/测试为15711/5237/5237，本项目完整数据为15701/5218/5220；08第三方为10700/3567/3566，本项目为10690/3548/3549。因此第三方指标仅供参考，不视为本项目的实测结果，也不是同协议比较结果。公开配置和单次日志不能证明参数已最优。

历史的小规模检查结果保留其当时配置；更新后应重新运行检查。完整训练仍需单独执行。
