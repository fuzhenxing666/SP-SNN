import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 设置双精度浮点数
torch.set_default_dtype(torch.float64)

# 定义参数
epsilon = 1e-3
x = torch.linspace(0, 1, 50)
y = torch.linspace(0, 1, 50)
x_grid, y_grid = torch.meshgrid(x, y, indexing='xy')


# 定义解析解函数
def analytical_solution(x, y):
    
    v = (torch.exp(-torch.tensor(1.0) / epsilon) + x * (1 - torch.exp(-torch.tensor(1.0) / epsilon)) - torch.exp(x - 1) / epsilon)
    return torch.exp(-y) * v
    
    return u

# 计算解
u = analytical_solution(x_grid, y_grid)

# 检查数值稳定性
print("是否存在无穷大值:", torch.isinf(u).any().item())
print("是否存在NaN值:", torch.isnan(u).any().item())

# 转换为numpy数组用于绘图
x_np = x_grid.numpy()
y_np = y_grid.numpy()
u_np = u.numpy()

# # 绘制三维曲面
# fig = plt.figure(figsize=(12, 8))
# ax = fig.add_subplot(111, projection='3d')
# surf = ax.plot_surface(x_np, y_np, u_np, cmap='viridis', rstride=1, cstride=1)
# fig.colorbar(surf, shrink=0.5, aspect=5)
# ax.set_xlabel('x')
# ax.set_ylabel('y')
# ax.set_zlabel('u(x,y)')
# plt.title(f'Analytical Solution (ε={epsilon})')
# plt.show()

# 绘制二维热图
plt.figure(figsize=(12, 8))
plt.imshow(u_np, extent=(0, 1, 0, 1), origin='lower', cmap='viridis', aspect='auto')
plt.colorbar(label='u(x,y)')
plt.xlabel('x')
plt.ylabel('y')
plt.title(f'Analytical Solution Heatmap (ε={epsilon})')
plt.show()