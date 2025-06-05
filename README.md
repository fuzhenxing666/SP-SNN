# SP-PDES: 基于子空间投影的偏微分方程求解框架

## 项目概述

SP-PDES (Subspace Projection for Partial Differential Equations) 是一个结合神经网络和子空间投影方法求解偏微分方程(PDEs)的框架。该方法特别适合处理含有边界层和奇异摄动项的复杂偏微分方程问题。主要思想是利用神经网络生成一组自适应的基函数，并结合针对边界层特征的显式基函数，通过最小二乘法求解投影系统，从而获得高精度的数值解。

## 特性

*   高效处理各类含边界层问题的偏微分方程
*   结合神经网络的自适应性和显式基函数的先验知识
*   适用于多维问题 (1D, 2D) 和空间-时间依赖问题
*   两阶段方法：神经网络训练 + 线性系统求解
*   支持多种类型的PDEs (椭圆型、抛物型、双调和等)

## 算法框架

### 数学原理

该方法将PDE的解表示为：

$$u(x) \approx \sum_{j=1}^{n} w_j \phi_j(x) + \sum_{k=1}^{m} v_k \psi_k(x)$$

其中：

*   $\phi_j(x)$ 是神经网络生成的基函数
*   $\psi_k(x)$ 是基于物理先验知识设计的显式基函数
*   $w_j, v_k$ 是通过求解线性系统获得的系数

### 主要步骤

1.  **神经网络训练阶段：**
    *   构建神经网络模型
    *   定义PDE残差和边界条件
    *   训练网络最小化残差项
2.  **子空间构建：**
    *   提取神经网络隐藏层输出作为第一组基函数
    *   添加专门为捕获边界层设计的显式基函数
3.  **投影系统构建：**
    *   计算PDE算子作用在各基函数上的结果
    *   组装线性系统矩阵和右端项
4.  **线性系统求解：**
    *   使用最小二乘法求解系数
    *   计算最终的近似解
5.  **评估与可视化：**
    *   计算与精确解的误差
    *   绘制解的分布和误差图

## 代码结构

### 核心组件

1.  **神经网络模型**

2.  **神经网络训练**
    
3.  **边界层显式基函数**

4.  **PDE算子计算**
   
5.  **系统矩阵组装**

6.  **线性系统求解**

## 案例示例

该框架支持多种类型的偏微分方程问题，包括：

算例1-2: 一维奇异摄动问题
*   示例1: $-\epsilon u_{xx} + u_x = \epsilon \pi^2 \sin(\pi x) + \pi \cos(\pi x)$
*   示例2: $-\epsilon u_{xx} + (1+\epsilon) u_x + u = 0$

算例3-4: 二维稳态问题
*   示例3: 一维反应-扩散方程 $u_t - \epsilon u_{xx} + (1+xe^{-t})u = f(x,t)$ (这里应为稳态或修正为时间依赖问题)
*   示例4: 带奇异摄动项的椭圆型PDE $-\epsilon (u_{xx} + u_{yy}) + u_x = -\sin(\pi y)$

算例5: 具有降维特征的问题
*   $-\epsilon (u_{xx} + u_{yy}) + u_x = 0$

算例6: 极坐标下的环形板弯曲问题
*   $\varepsilon^2 \Delta \Delta W - \Delta W = 0$

算例7-8: 时间依赖问题
*   示例7: 二维反应-扩散方程 $u_t - \epsilon u_{xx} + (1-x^2)u_x = f(x,t)$
*   示例8: 抛物型对流占优方程 $u_t - \epsilon \Delta u + u_x + u_y = F(x,y)$

## 使用方法

### 基本流程

1.  **设置问题参数和神经网络配置**
    ```python
    epsilon = 1e-5  # 小参数
    num_explicit_bases = 1 # 显式基函数数量
    net = Net(hidden_size=100, subspace_dim=100).to(device)
    ```

2.  **训练神经网络**
    ```python
    xy_interior, xy_b0, xy_b1, xy_b2, xy_b3 = train_network(net, epsilon, f, device)
    ```

3.  **组装系统矩阵和向量**
    ```python
    A, f_vec = assemble_matrix(net, explicit_bases, xy_interior, xy_b0, xy_b1, xy_b2, xy_b3, f, epsilon, num_explicit_bases, device)
    ```

4.  **求解线性系统**
    ```python
    w = solve_linear_system(A, f_vec)
    ```

5.  **评估和可视化结果**
    ```python
    results = evaluate_and_plot(net, w, epsilon, num_explicit_bases, device)
    ```

## 依赖库

*   PyTorch (神经网络和自动微分)
*   NumPy (数值计算)
*   SciPy (线性代数求解)
*   Matplotlib (结果可视化)

## 优势

1.  **高精度**：即使在 $\epsilon$ 极小时也能获得高精度解
2.  **高效率**：相比传统数值方法，计算效率更高
3.  **自适应性**：神经网络自动学习最适合的基函数
4.  **灵活性**：适用于各种类型的PDEs

## 局限性

*   需要为特定问题设计显式基函数
*   神经网络训练需要适当的超参数调整
*   对高维问题计算成本较高

## 结果示例

对于典型的边界层问题(如算例4)，当 $\epsilon=1\text{e-}5$ 时，该方法可实现：
*   $L_2$ 范数误差 $1\text{e-}5$ 量级
*   可以准确捕捉边界层行为而不需要边界层附近的网络加密
