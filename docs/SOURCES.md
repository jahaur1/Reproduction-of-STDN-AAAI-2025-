# 源码来源与修改范围

本仓库是复现与鲁棒性比较工程，不是三个模型作者的官方发布。

| 目录 | 来源 | 说明 |
| --- | --- | --- |
| `models/STDN/` | [roarer008/STDN](https://github.com/roarer008/STDN)，提交`e24a60fa9485fb9805df9c171d67ab60840d757d` | 保留模型结构和损失；增加CPU/单GPU设备适配、预测导出、可配置调度和可选共享标准化参数 |
| `models/STWave/` | [LMissher/STWave](https://github.com/LMissher/STWave)，工作区基准提交`80c74889fc939fab0c2c3ee0dab4c9ec872a971a` | 保留模型和损失；已有运行兼容修复、完整训练检查点及预测导出；本次增加外部样本起始下标以避免依赖仓库内数据 |
| `models/HyperD/` | 工作区HyperD-master源码，对应公开项目[ll121202/HyperD](https://github.com/ll121202/HyperD) | 包含模型、四份配置及其随附BasicTS框架；保留模型结构、损失和优化器；扩展预测保存和兼容当前环境 |
| `vendor/fastdtw/` | [slaypni/fastdtw v0.3.4](https://github.com/slaypni/fastdtw/tree/v0.3.4)的Python实现 | 附原MIT许可证；避免Windows安装时强制编译C扩展，算法与此前使用的Python实现一致 |
| `robustness/corruption_reference.py` | 用户提供的corruption.py | 保留副本作为协议测试依据 |
| `robustness/`其余文件、`launcher.py` | 本地新增实验入口 | 共享扰动样本、统一标准化、STDN/HyperD适配、统一评价、结果表格和绘图 |

STDN03/08参考[Torch-MTS配置](https://github.com/XDZhelheim/Torch-MTS/blob/2db4de371584067160f9a37f1ae59495699b4a0a/configs/STDN.yaml)，来源和差异见[REFERENCE_TORCH_MTS.md](REFERENCE_TORCH_MTS.md)。这些不是STDN作者给出的03/08最佳参数。

论文链接：

- STDN：[Spatiotemporal-aware Trend-Seasonality Decomposition Network for Traffic Flow Forecasting](https://arxiv.org/abs/2502.12213)，AAAI2025。
- STWave：[When Spatio-Temporal Meet Wavelets: Disentangled Traffic Forecasting via Efficient Spectral Graph Attention Networks](https://ieeexplore.ieee.org/document/10184591)，ICDE2023。
- HyperD：HyperD: Hybrid Periodicity Decoupling Framework for Traffic Forecasting，参见其[官方项目说明](https://github.com/ll121202/HyperD)。

各模型代码的版权与署名归原权利人所有。所用模型源码副本没有独立LICENSE文件；本工程没有为这些代码擅自添加MIT等替代许可证。FastDTW的MIT许可仅适用于`vendor/fastdtw/`，不扩展至其他目录。

本仓库不包含任何数据集、训练权重、运行日志、个人凭据或小样本测试指标。依赖允许范围用于当前运行兼容，不代表完全复刻各论文的软件环境。

当前STWave源码仅保留作干净实验结果来源参考；鲁棒性命令不接受STWave。`plot_stwave_12steps.py`是新增的只读绘图入口。
