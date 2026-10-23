# 共享鲁棒性实现

从仓库根目录使用统一入口`python run.py --help`。

- [实验协议](../docs/PROTOCOL.md)
- [快速运行](../README.md)
- `data.py`：共享样本与扰动。
- `worker.py`：STDN/HyperD的独立进程适配。
- `analyze.py`：统一指标CSV与对比图。
- `test_corruptions.py`：与用户提供代码的一致性及数据完整性检查。
