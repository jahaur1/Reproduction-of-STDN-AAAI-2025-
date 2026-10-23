# STDN 四数据集复现入口

独立一键入口为仓库根目录的`python run_stdn.py`：默认PEMS03/04/07/08各一次。单数据集示例：`python run_stdn.py --datasets PEMS08`。

目录整理后，模型源码和四份配置位于`models/STDN/`。仓库根目录的`python run.py reproduce`调用这一套原有复现入口；旧的`python run.py --dataset ... --data-root ...`写法也保留兼容。本文下方旧命令仍可从仓库根目录运行。

论文：[Spatiotemporal-aware Trend-Seasonality Decomposition Network for Traffic Flow Forecasting](https://arxiv.org/abs/2502.12213)（AAAI 2025）。

作者官方代码：[roarer008/STDN](https://github.com/roarer008/STDN)。本项目基于提交 `e24a60fa9485fb9805df9c171d67ab60840d757d`，补充可移植的数据准备、运行入口和测试结果导出。**本项目不是作者官方仓库。**

## 一条命令运行

需要 Python 3.10 或更高版本，建议先创建独立 Python 环境。下载本项目、准备外部数据后，在项目目录执行：

```sh
python run.py --dataset all --data-root /path/to/all_data --raw-root /path/to/raw_data --install-deps
```

该命令依次准备并训练 PEMS03、04、07、08，各运行一次。依赖缺失时安装到当前 Python 环境；已安装的依赖不会自动升级。默认自动使用可用 CUDA，否则使用 CPU。GPU 用户应预先安装适合自己设备的 [PyTorch](https://pytorch.org/get-started/locally/)，自动安装不保证 CUDA 可用。

仅运行单个数据集：

```sh
python run.py --dataset PEMS04 --data-root /path/to/all_data --raw-root /path/to/raw_data
```

先检查数据与环境是否能运行（每个划分最多 4 个样本、1 轮、batch 2，不用于论文指标比较）：

```sh
python run.py --dataset all --data-root /path/to/all_data --raw-root /path/to/raw_data --smoke --device cpu
```

路径含空格时加双引号。Windows 路径可直接使用，例如 `--data-root "D:/datasets/all_data"`。完整训练耗时较长，应在独立终端或自己的任务调度器中运行；本入口本身不提供后台驻留或中断续训功能。

## 外部数据

本项目不包含数据集、论文 PDF、训练权重和运行日志。请自行获取有权使用的数据。

推荐使用 BasicTS 格式，文件结构为：

```text
all_data/
  PEMS03/data.dat
  PEMS03/desc.json
  PEMS04/data.dat
  PEMS04/desc.json
  PEMS07/data.dat
  PEMS07/desc.json
  PEMS08/data.dat
  PEMS08/desc.json
raw_data/
  PEMS03/PEMS03.csv
  PEMS03/PEMS03.txt
  PEMS04/PEMS04.csv
  PEMS07/PEMS07.csv
  PEMS08/PEMS08.csv
```

`data.dat` 必须是 float32，形状由 `desc.json` 的 `shape` 指定，特征顺序为 `traffic flow, time of day, day of week`；后两者分别按 288 和 7 归一化。节点数依次为 358、307、883、170。CSV 为带表头的三列 `from,to,distance`，保持原有边方向。PEMS03 的原始传感器编号使用 `PEMS03.txt` 转为连续节点编号。

也支持原始 `PEMSxx.npz`（数组名 `data`，第一通道是流量）。此时必须用 `--start-date` 指定该数据集第一个采样点的真实日期时间，以生成正确的星期和日内时段；不同数据集请分别运行。不要猜测日期。

```sh
python run.py --dataset PEMS04 --data-root /path/to/raw_data --start-date YYYY-MM-DD
```

还支持已经准备好的 `samples_12_12_5.npz`，须搭配节点编号已连续化的 `PEMSxx.csv`。样本键为 `train_x,train_target,trainTE,val_x,val_target,valTE,test_x,test_target,testTE`。

从连续序列准备样本时，按时间顺序划分 60%/20%/剩余部分，在每个划分内生成 12 步输入、12 步输出，不跨划分取窗口。缓存放在 `.cache/`，训练结果放在 `results/`，两者均被 Git 忽略。

## 配置与来源

| 数据集 | 配置来源 | 节点数 | K / d | 图阶数 | 最大轮数 | batch |
| --- | --- | ---: | --- | ---: | ---: | ---: |
| PEMS03 | 参考第三方 Torch-MTS 配置 | 358 | 16 / 8 | 3 | 200 | 64 |
| PEMS04 | 作者发布配置 | 307 | 16 / 8 | 3 | 200 | 64 |
| PEMS07 | 作者发布配置 | 883 | 12 / 8 | 4 | 100 | 64 |
| PEMS08 | 参考第三方 Torch-MTS 配置 | 170 | 16 / 8 | 3 | 200 | 64 |

学习率均为 0.001，其余参数见 `conf/`。03、08 参考 [Torch-MTS 的 STDN 配置](https://github.com/XDZhelheim/Torch-MTS/blob/2db4de371584067160f9a37f1ae59495699b4a0a/configs/STDN.yaml)：早停耐心值20，MultiStepLR 在完成第30、50轮时各乘0.1；默认种子233，参考其训练入口和公开日志。04、07保持作者配置：早停10，StepLR 每10轮乘0.9，默认不显式固定随机种子。`--seed` 可以覆盖当前数据集的种子，实际值会记录在结果中。`--batch-size` 可以覆盖配置，但会改变实验条件。

03、08 是第三方参数参考配置，不能称为作者官方配置或已证明的最佳参数。本项目仍采用上文的作者式数据划分和样本构造、原作者训练损失与指标实现；没有改用 Torch-MTS 的完整数据和训练流程，因此不能直接对齐其公开日志中的测试指标。来源与已知差异见 [REFERENCE_TORCH_MTS.md](REFERENCE_TORCH_MTS.md)。

模型计算逻辑沿用上游；修改了设备分配以支持 CPU/单 GPU，添加数据转换、数值检查、最佳模型保存及重载检查、预测导出和命令行参数。依赖版本范围用于安装方便，不代表与论文环境完全一致。不能据此保证指标与论文逐位一致。

## 输出

每个数据集输出到 `results/full/PEMSxx/`（小规模检查为 `results/smoke/PEMSxx/`）：

- `best_model.pt`：验证集最优模型权重。
- `test_results.npz`：每个测试窗口的 `inputs`、`prediction`、`target` 和 `temporal_features`，流量值已还原至原始单位。
- `metrics.json`：整体及未来 1～12 步的 MAE、RMSE、MAPE；MAPE 字段单位为百分数。
- `runtime.conf`、`environment.json`：实际配置、依赖版本、设备和随机种子。
- `console.log`、`status.json`：训练日志和运行状态；四数据集汇总为上一级的 `summary.json`。

同一输出目录再次运行会覆盖该数据集的结果；需要保留多次实验时请指定不同的 `--output-root`。模型权重不包含优化器、随机状态等完整续训状态。

本地已在四个真实数据集上通过 CPU 小规模检查：训练、测试、预测有限值及权重重载一致性；这不是完整训练验证，也不承诺所有设备的默认 batch 都有足够显存。

## 引用与授权

使用时请引用原论文，并保留官方仓库来源。上游该提交未附带 LICENSE，本项目没有擅自为作者代码添加新的开源许可证；公开再分发和使用时请确认原作者授权。来源与修改范围详见 [UPSTREAM.md](UPSTREAM.md)。
