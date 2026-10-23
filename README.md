# STDN复现、双模型鲁棒性与STWave逐步可视化

论文：[STDN — Spatiotemporal-aware Trend-Seasonality Decomposition Network for Traffic Flow Forecasting](https://arxiv.org/abs/2502.12213)，AAAI2025。原作者代码：[roarer008/STDN](https://github.com/roarer008/STDN)。

本仓库保留之前的STDN四数据集复现；鲁棒性实验只比较STDN与HyperD，STWave只读取干净测试结果进行12步可视化。支持PEMS03/04/07/08。**这是复现工程，不是作者官方仓库。**

## 文件夹

```text
run.py                     统一Python入口
run_stdn.py                STDN独立一键入口，默认四数据集各一次
plot_stwave_12steps.py      STWave干净测试的12步指标与预测可视化，不训练
launcher.py                参数解析、依赖检查和任务分发
models/
  STDN/                    原有STDN复现代码、四份配置
  STWave/                  STWave模型、训练器与四份配置
  HyperD/                  HyperD模型、四份配置及随附BasicTS框架
robustness/                STDN/HyperD共享扰动、统一评价与绘图
docs/                      原复现说明、扰动协议、参数与源码来源
vendor/fastdtw/            原Python实现及其MIT许可
data/                      自行准备的外部数据，不上传
outputs/                   运行结果、预测与分析图，不上传
```

数据集、模型权重、日志、历史运行结果均不包含在仓库中。三个模型所需源码已包含，无需另外下载模型仓库。

## 数据与环境准备

建议Python3.12，并在独立环境中安装适合自己GPU的[PyTorch](https://pytorch.org/get-started/locally/)。然后执行`python -m pip install -r requirements.txt`。也可在实验命令后加`--install-deps`，缺少依赖时会安装到当前Python环境；不会自动替换已安装PyTorch。

数据按下面结构放置：

```text
data/
  flow/
    PEMS03/{data.dat,desc.json,adj_mx.pkl}
    PEMS04/{data.dat,desc.json,adj_mx.pkl}
    PEMS07/{data.dat,desc.json,adj_mx.pkl}
    PEMS08/{data.dat,desc.json,adj_mx.pkl}
  graphs/
    PEMS03/{PEMS03.csv,PEMS03.txt}
    PEMS04/PEMS04.csv
    PEMS07/PEMS07.csv
    PEMS08/PEMS08.csv
```

花括号表示同目录下的多个文件，不是文件名。`data.dat`为BasicTS的float32数据，通道顺序为流量、日内时段、星期；`desc.json`提供形状和regular_settings；`adj_mx.pkl`是HyperD使用的道路图。原始CSV带表头且有from/to/distance三列；03使用PEMS03.txt映射传感器编号。可参考[BasicTS数据说明](https://github.com/GestaltCogTeam/BasicTS/blob/v0.5.8/tutorial/dataset_design.md)。

数据也可以留在其他位置，每条命令追加：

```sh
--data-root /path/to/all_data --raw-root /path/to/raw_data
```

## 分别一键运行

### STDN独立入口

准备好上述数据后，在仓库根目录执行：

```sh
python run_stdn.py
```

默认仅运行STDN，依次复现PEMS03、04、07、08，每个数据集一次。可用`--datasets PEMS08`只跑单个数据集；外部数据路径通过`--data-root`、`--raw-root`指定，缺少依赖时可加`--install-deps`。

STDN单模型鲁棒性也可以分别一键运行：

```sh
python run_stdn.py node-missing
python run_stdn.py temporal-gap
python run_stdn.py gaussian-noise
```

这些命令只运行STDN，默认四数据集×三个比例，共12个组合。可追加`--plan`只查看计划，或`--smoke --device cpu`检查小样本运行。

### STDN与HyperD鲁棒性比较

在仓库根目录运行。下面每个命令都自动准备对应扰动、依次训练STDN和HyperD、保留测试样本和预测，再汇总CSV及分析图：

```sh
# 固定节点缺失，10%/20%/30%
python run.py node-missing

# 每个节点独立的连续3步缺失，10%/20%/30%
python run.py temporal-gap

# 高斯噪声，10%/20%/30%，均值10、标准差500、截断至非负
python run.py gaussian-noise
```

每种命令默认覆盖四个数据集、三种比例、两个模型，共24个独立组合，每组合只训练一次。下面是只跑一个数据集和一个比例的例子：

```sh
python run.py gaussian-noise --datasets PEMS08 --rates 0.1
```

其他入口：

```sh
# 三种扰动全部运行，共72个组合
python run.py robustness

# 同一协议的干净对照，每模型/数据集各一次
python run.py clean

# 之前的STDN原始复现入口，各数据集一次
python run.py reproduce

# 只汇总、绘图，不训练；默认扫描outputs/robustness下完整实验结果
python run.py analyze
```

`reproduce`保留之前STDN的数据与指标处理；`clean`是STDN与HyperD在统一鲁棒性协议下的对照。两类指标应分别使用。STDN03/08仍采用此前确定的Torch-MTS参考配置，04/07保留作者参数，详见[原复现说明](docs/BASELINE.md)和[参数来源](docs/REFERENCE_TORCH_MTS.md)。

## 先检查再完整训练

```sh
# 只列出计划，不生成大数据文件、不训练
python run.py gaussian-noise --plan

# 协议与完整性单元检查
python run.py check

# 两个模型的小样本检查：一个数据集、一种比例、一轮，使用CPU
python run.py gaussian-noise --datasets PEMS08 --rates 0.1 --smoke --device cpu
```

`--smoke`指标与图均标记为检查用途，不能作为论文结果。完整训练默认CUDA；可显式用`--device cpu`。本次发布仅完成代码和小样本验证，没有启动完整鲁棒性训练。


## STWave：只做未来12步可视化

将原STWave干净实验的输出放入`results/STWave/PEMSxx/`，每个已完成数据集包含`test_metrics.json`和`test_results.npz`，然后运行：

```sh
python plot_stwave_12steps.py
```

也可直接指定现有实验目录：

```sh
python plot_stwave_12steps.py --results-root /path/to/STWave/reproduction/official_seed1
```

脚本只读取结果，不训练、不添加扰动。每个已完成数据集输出：

- 未来1～12步的MAE、RMSE、MAPE曲线与CSV（MAPE为百分数）。
- 每一步单独的预测/真实值对照PNG、PDF，共12步；另有12子图总览。
- 默认展示节点0、前288个测试窗口，可用`--node`、`--start`、`--count`更改。示例节点曲线不代表全体节点的整体误差。

输出位于`outputs/stwave_12steps/`。未完成完整测试的数据集标记为pending，不读取smoke结果或填造数值；其干净实验完成后再次运行即可补图。

## 常用选项与输出

- `--models STDN HyperD`：选择鲁棒性模型，默认两者；STWave不参与。
- `--datasets PEMS03 PEMS08`、`--rates 0.1 0.3`：限定数据集和比例。
- `--corruption-seed 2024`：扰动种子。训练/验证/测试缺失节点一致，STDN与HyperD使用同一扰动包。
- `--training-seed 1`：显式覆盖模型训练种子。默认STDN03/08为233，其余为1。
- `--tag my_run`、`--output-root /path/to/outputs`：自定义结果分组和位置。
- `--action prepare`：只准备数据；`--action summarize`：汇总已有结果并绘图。
- `--include-clean`：给所选扰动额外加入干净对照。

输出按实验类型分开：

```text
outputs/
  baseline/stdn/                         之前STDN复现
  robustness/node_missing/{full,smoke}/  节点缺失
  robustness/temporal_gap/{full,smoke}/  连续缺失
  robustness/gaussian_noise/{full,smoke}/ 高斯噪声
  analysis/<实验名>/{full,smoke}/        overall.csv、horizons.csv、reports.json、PNG图
```

每个模型保存测试输入、干净输入、预测、真实值、掩码与时间下标，并输出整体及未来1～12步MAE/RMSE/MAPE。MAPE在统一报告中使用百分数。**只扰动输入流量，未来真实目标不扰动。**详细规则、共享标准化和训练历史初始化方式见[实验协议](docs/PROTOCOL.md)。

重新分析某个分组：`python run.py analyze --tag gaussian_noise`；检查结果追加`--smoke`。完整结果已存在会跳过；部分训练结果不会自动覆盖，失败重跑请使用新tag。训练需要在自己的终端持续运行，暂未提供后台驻留或鲁棒性中断续训。

## 来源

源码版权、论文、改动边界与第三方许可见[来源说明](docs/SOURCES.md)。本仓库没有为原作者代码另行赋予MIT等许可证。[检查记录](docs/VALIDATION.md)只记录程序验证，不代表完整训练或论文指标已复现。
