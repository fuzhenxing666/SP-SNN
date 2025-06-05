# SP-SNN 示例

本仓库提供了一系列使用 **Spectral Neural Network (SNN)** 方法求解偏微分方程（PDE）的示例代码。各算例均采用 PyTorch 实现，通过在神经网络生成的子空间中引入显式基函数，结合最小二乘求解得到近似解，以处理带有边界层或对流主导的复杂问题。

## 环境依赖

- Python 3.8 及以上
- PyTorch 2.2
- NumPy 1.26
- SciPy 1.15
- Matplotlib 3.8

可通过以下命令安装依赖：

```bash
pip install -r requirements.txt
```

## 目录结构

- `SNN-SP/`：存放主要的 Python 脚本和示例
  - `SP-PDES.ipynb`：Jupyter Notebook，列出了 8 个典型 PDE 问题的推导与实验
  - `算例*.py`：对应各个算例的完整实现，直接运行即可复现实验结果
  - `test.py`、`测试.py`：部分功能测试脚本
  - `best_model.pth`：示例训练所得的模型权重
- `requirements.txt`：依赖列表

## 快速开始

1. 安装依赖：`pip install -r requirements.txt`
2. 进入 `SNN-SP` 目录，运行对应的算例脚本，例如：

```bash
python 算例1.py
```

所有脚本均可在 CPU 上运行，如有 GPU 环境会自动使用 `cuda` 加速。

## 说明

每个算例脚本给出了特定 PDE 的问题描述、训练流程以及结果可视化。更多背景介绍与详细推导可参见 `SP-PDES.ipynb`。若在使用过程中遇到问题，欢迎在 Issues 中讨论。
