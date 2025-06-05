import torch
import torch.nn as nn
import numpy as np
from scipy.linalg import lstsq
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time

# Define device (use GPU if available, otherwise CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

torch.manual_seed(42)
torch.set_default_dtype(torch.float64)

# Define the neural network for 2D
class Net(nn.Module):
    def __init__(self, hidden_size=100, subspace_dim=100):
        super(Net, self).__init__()
        self.hidden = nn.Sequential(
            nn.Linear(3, hidden_size),  # Input: (x, y, t)
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, subspace_dim),
            nn.Tanh()
        )
        self.output = nn.Linear(subspace_dim, 1, bias=False)
        with torch.no_grad():
            self.output.weight.fill_(1.0)
        self.output.weight.requires_grad = True
        
        # Move model to device
        self.to(device)

    def forward(self, xyt):
        features = self.hidden(xyt)
        out = self.output(features)
        return out

    def get_hidden_layer_output(self, xyt):
        return self.hidden(xyt)

# Exact 2D solution
def exact_solution(xyt, epsilon):
    x = xyt[:, 0:1]
    y = xyt[:, 1:2]
    t = xyt[:, 2:3]
    exponent = -((1-x)*(1-y))/(2*epsilon)
    return torch.exp(1 - t) * torch.exp(exponent)

# PDE source term F(x, y) - now independent of t
def F_source(xy, epsilon):
    """
    计算 PDE 的源项 F(x,y) = u_t - ε(uxx + uyy) + ux + uy
    注意：解析解的时间导数部分在化简后会显式出现在表达式中
    """
    x = xy[:, 0:1]
    y = xy[:, 1:2]
    
    # 计算空间部分的指数因子
    spatial_factor = torch.exp(-((1-x)*(1-y))/(2*epsilon))
    
    # F(x,y)的表达式
    term = -1 + (2 - x**2 - y**2)/(4*epsilon)
    return spatial_factor * term

def train_network(net, epsilon, F_source, num_epochs=500, early_stop_threshold=1e-3, T=1.0):
    # 定义优化器
    optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
    
    # 定义采样点数量
    N_i, N_b, N_0 = 20, 20, 20  # 减少采样点以降低计算复杂度
    
    # 内部点 (均匀网格)
    x_i = torch.linspace(0, 1, N_i, dtype=torch.float64, device=device)[1:-1]
    y_i = torch.linspace(0, 1, N_i, dtype=torch.float64, device=device)[1:-1]
    t_i = torch.linspace(0, T, N_i, dtype=torch.float64, device=device)[1:-1]
    X, Y, T_grid = torch.meshgrid(x_i, y_i, t_i, indexing='ij')
    xyt_i_base = torch.stack([X.flatten(), Y.flatten(), T_grid.flatten()], dim=1)

    # 初始点 (t=0)
    x_0 = torch.linspace(0, 1, N_0, dtype=torch.float64, device=device).reshape(-1, 1)
    y_0 = torch.linspace(0, 1, N_0, dtype=torch.float64, device=device).reshape(-1, 1)
    t_0 = torch.zeros_like(x_0)
    xyt_0_base = torch.cat([x_0, y_0, t_0], dim=1)

    # 边界点 - 四个边界
    # 边界1: x=0
    y_b0 = torch.linspace(0, 1, N_i, dtype=torch.float64, device=device).reshape(-1, 1)
    x_b0 = torch.zeros_like(y_b0)
    t_b0 = torch.linspace(0, 1, N_b, dtype=torch.float64, device=device).reshape(-1, 1)
    xyt_b0_base = torch.cat([x_b0, y_b0, t_b0], dim=1)
    # y_b0 = torch.linspace(0, 1, N_i, dtype=torch.float64, device=device)  # [N_i]
    # t_b0 = torch.linspace(0, 1, N_b, dtype=torch.float64, device=device)  # [N_b]
    # Y_b0, T_b0 = torch.meshgrid(y_b0, t_b0, indexing='ij')  # 创建网格
    # x_b0 = torch.zeros_like(Y_b0)  # 全是0的网格
    # xyt_b0_base = torch.stack([x_b0.flatten(), Y_b0.flatten(), T_b0.flatten()], dim=1)  # [N_i*N_b, 3]
    
    # 边界2: x=1
    y_b1 = torch.linspace(0, 1, N_i, dtype=torch.float64, device=device).reshape(-1, 1)
    x_b1 = torch.ones_like(y_b1)
    t_b1 = torch.linspace(0, 1, N_b, dtype=torch.float64, device=device).reshape(-1, 1)
    xyt_b1_base = torch.cat([x_b1, y_b1, t_b1], dim=1)
    
    # 边界3: y=0
    x_b2 = torch.linspace(0, 1, N_i, dtype=torch.float64, device=device).reshape(-1, 1)
    y_b2 = torch.zeros_like(x_b2)
    t_b2 = torch.linspace(0, 1, N_b, dtype=torch.float64, device=device).reshape(-1, 1)
    xyt_b2_base = torch.cat([x_b2, y_b2, t_b2], dim=1)
    
    # 边界4: y=1
    x_b3 = torch.linspace(0, 1, N_i, dtype=torch.float64, device=device).reshape(-1, 1)
    y_b3 = torch.ones_like(x_b3)
    t_b3 = torch.linspace(0, 1, N_b, dtype=torch.float64, device=device).reshape(-1, 1)
    xyt_b3_base = torch.cat([x_b3, y_b3, t_b3], dim=1)
    
    # 合并所有边界点
    xyt_b_all = torch.cat([xyt_b0_base, xyt_b1_base, xyt_b2_base, xyt_b3_base], dim=0)
    
    # 训练循环
    start_time = time.time()
    losses = []
    
    for epoch in range(num_epochs):
        # 在每次迭代中创建新的张量并启用梯度计算，确保计算图独立
        xyt_i = xyt_i_base.clone().requires_grad_(True)  # 内部点
        xyt_0 = xyt_0_base.clone().requires_grad_(True)  # 初始条件点
        xyt_b = xyt_b_all.clone().requires_grad_(True)   # 边界点
        
        # 前向传播计算网络输出
        u_i = net(xyt_i)
        
        # 计算偏导数
        # 时间导数 u_t
        u_t = torch.autograd.grad(u_i, xyt_i, grad_outputs=torch.ones_like(u_i), 
                                  create_graph=True)[0][:, 2:3]  # (x,y,t) -> t索引为2
        
        # 空间偏导 u_x, u_y
        u_x = torch.autograd.grad(u_i, xyt_i, grad_outputs=torch.ones_like(u_i), 
                                  create_graph=True)[0][:, 0:1]  # x导数
        u_y = torch.autograd.grad(u_i, xyt_i, grad_outputs=torch.ones_like(u_i), 
                                  create_graph=True)[0][:, 1:2]  # y导数
        
        # 二阶偏导 u_xx, u_yy
        u_xx = torch.autograd.grad(u_x, xyt_i, grad_outputs=torch.ones_like(u_x), 
                                   create_graph=True)[0][:, 0:1]  # x的二阶导
        u_yy = torch.autograd.grad(u_y, xyt_i, grad_outputs=torch.ones_like(u_y), 
                                   create_graph=True)[0][:, 1:2]  # y的二阶导
        
        # 计算 PDE 残差 (只考虑空间部分)
        # 注意: F_source 只依赖x,y
        F_xy = F_source(xyt_i[:, :2], epsilon)  # 只取x,y部分
        pde_res = u_t - epsilon*(u_xx + u_yy) + u_x + u_y - F_xy
        
        # 计算初始条件和边界条件残差
        ic_res = net(xyt_0) - exact_solution(xyt_0, epsilon)  # 初始条件残差
        bc_res = net(xyt_b) - exact_solution(xyt_b, epsilon)  # 边界残差
        
        # 计算总损失
        loss_pde = torch.mean(pde_res**2)
        loss_ic = torch.mean(ic_res**2)
        loss_bc = torch.mean(bc_res**2)
        # loss = loss_pde + loss_ic + loss_bc
        loss = loss_pde
        
        if epoch == 0:
            initial_loss = loss.item()
        
        losses.append(loss.item())
        
        # 梯度清零，反向传播，更新参数
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if epoch % 100 == 0:
            elapsed_time = time.time() - start_time
            print(f"Epoch {epoch}, Loss: {loss.item():.6f}, PDE: {loss_pde.item():.6f}, IC: {loss_ic.item():.6f}, BC: {loss_bc.item():.6f}, Time: {elapsed_time:.1f}s")
        
        if loss.item() <= early_stop_threshold * initial_loss:
            elapsed_time = time.time() - start_time
            print(f"提前停止于第 {epoch} 轮, 损失: {loss.item():.6f}, 总时间: {elapsed_time:.1f}s")
            break
    
    # 返回基础网格点
    return xyt_i_base, xyt_0_base, xyt_b_all, losses

# Explicit basis functions for 2D
def explicit_bases(xyt, epsilon, num_explicit_bases):
    x = xyt[:, 0:1]
    y = xyt[:, 1:2]
    t = xyt[:, 2:3]
    if num_explicit_bases == 0:
        return torch.empty(xyt.shape[0], 0, dtype=torch.float64, device=device)
    
    basis1 = torch.exp(-((1-x)*(1-y))/(2*epsilon))
    basis2 = torch.exp(-t) * torch.exp(-((1-x)*(1-y))/(2*epsilon))
    
    if num_explicit_bases == 1:
        return basis1
    elif num_explicit_bases == 2:
        return torch.cat([basis1, basis2], dim=1)
    else:
        basis3 = (1-y) * torch.exp(-(1-y)/epsilon) 
        basis4 = (1-x) * (1-y) * basis1  
        return torch.cat([basis1, basis2, basis3, basis4], dim=1)[:, :num_explicit_bases]

# Compute PDE operator for 2D
def compute_pde_operator(net, xyt, epsilon, explicit_bases_func, num_explicit_bases):
    # 确保 xyt 可以求导
    xyt = xyt.requires_grad_(True)
    
    # 获取网络输出和显式基函数
    phi = net.get_hidden_layer_output(xyt)  # [num_points, subspace_dim]
    psi = explicit_bases_func(xyt, epsilon, num_explicit_bases)  # [num_points, num_explicit_bases]
    
    # 拼接基函数
    basis = torch.cat([phi, psi], dim=1) if num_explicit_bases > 0 else phi
    
    L_basis_list = []
    for j in range(basis.shape[1]):
        basis_j = basis[:, j].unsqueeze(1)  # [num_points, 1]
        
        # 计算一阶梯度
        grad_basis_j = torch.autograd.grad(basis_j, xyt, grad_outputs=torch.ones_like(basis_j), 
                                          create_graph=True)[0]  # [num_points, 3]
        
        # 提取导数
        basis_j_x = grad_basis_j[:, 0:1]  # 对 x 的偏导数
        basis_j_y = grad_basis_j[:, 1:2]  # 对 y 的偏导数
        basis_j_t = grad_basis_j[:, 2:3]  # 对 t 的偏导数
        
        # 计算二阶偏导
        basis_j_xx = torch.autograd.grad(basis_j_x, xyt, grad_outputs=torch.ones_like(basis_j_x), 
                                        create_graph=True)[0][:, 0:1]  # x的二阶导
        basis_j_yy = torch.autograd.grad(basis_j_y, xyt, grad_outputs=torch.ones_like(basis_j_y), 
                                        create_graph=True)[0][:, 1:2]  # y的二阶导
        
        # 计算 PDE 算子
        L_basis_j = basis_j_t - epsilon * (basis_j_xx + basis_j_yy) + basis_j_x + basis_j_y
        L_basis_list.append(L_basis_j)
    
    # 拼接所有基函数的 PDE 算子结果
    L_basis = torch.cat(L_basis_list, dim=1)  # [num_points, subspace_dim + num_explicit_bases]
    return L_basis

# Assemble the matrix for 2D
def assemble_matrix(net, explicit_bases_func, xyt_interior, xyt_initial, xyt_boundary, F_source, epsilon, num_explicit_bases):
    # PDE部分
    L_basis = compute_pde_operator(net, xyt_interior, epsilon, explicit_bases_func, num_explicit_bases)
    A_i = L_basis.cpu().detach().numpy()
    F_xy = F_source(xyt_interior[:, :2], epsilon).cpu().detach().numpy()  # 只取x,y部分
    f_i = F_xy.flatten()

    # 初始条件部分
    phi_0 = net.get_hidden_layer_output(xyt_initial)
    psi_0 = explicit_bases_func(xyt_initial, epsilon, num_explicit_bases)
    basis_0 = torch.cat([phi_0, psi_0], dim=1) if num_explicit_bases > 0 else phi_0
    A_0 = basis_0.cpu().detach().numpy()
    u0 = exact_solution(xyt_initial, epsilon).cpu().detach().numpy().flatten()

    # 边界条件部分
    phi_b = net.get_hidden_layer_output(xyt_boundary)
    psi_b = explicit_bases_func(xyt_boundary, epsilon, num_explicit_bases)
    basis_b = torch.cat([phi_b, psi_b], dim=1) if num_explicit_bases > 0 else phi_b
    A_b = basis_b.cpu().detach().numpy()
    u_b = exact_solution(xyt_boundary, epsilon).cpu().detach().numpy().flatten()
    
    # 组装整体矩阵
    A = np.vstack([A_i, A_0, A_b])
    f_vec = np.hstack([f_i, u0, u_b])
    
    return A, f_vec

# Solve the linear system
def solve_linear_system(A, f):
    w, _, _, _ = lstsq(A, f)
    return w

# Compute the approximate solution
def compute_solution(net, explicit_bases_func, w, xyt, epsilon, num_explicit_bases):
    xyt_tensor = torch.tensor(xyt, dtype=torch.float64, device=device)
    phi = net.get_hidden_layer_output(xyt_tensor).cpu().detach().numpy()
    psi = explicit_bases_func(xyt_tensor, epsilon, num_explicit_bases).cpu().detach().numpy()
    basis = np.hstack([phi, psi]) if num_explicit_bases > 0 else phi
    return basis @ w

# 可视化结果 (特定时间切片)
def visualize_at_time(net, w, epsilon, num_explicit_bases, device, time_slice=0.5, n_points=40):
    # 创建网格
    x = np.linspace(0, 1, n_points)
    y = np.linspace(0, 1, n_points)
    X, Y = np.meshgrid(x, y)
    t = np.full_like(X, time_slice)
    
    # 构建输入
    points = np.stack((X.flatten(), Y.flatten(), t.flatten()), axis=1)
    
    # 计算预测解和精确解
    u_pred = compute_solution(net, explicit_bases, w, points, epsilon, num_explicit_bases)
    u_exact = exact_solution(torch.tensor(points, dtype=torch.float64, device=device), epsilon).detach().cpu().numpy()
    
    # 重塑为网格
    U_pred = u_pred.reshape(n_points, n_points)
    U_exact = u_exact.reshape(n_points, n_points)
    error = np.abs(U_pred - U_exact)
    
    # 创建3D图
    fig = plt.figure(figsize=(18, 6))
    
    # 预测解
    ax1 = fig.add_subplot(131, projection='3d')
    surf1 = ax1.plot_surface(X, Y, U_pred, cmap='viridis', edgecolor='none')
    ax1.set_title(f'Predicted Solution at t={time_slice}')
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_zlabel('u')
    fig.colorbar(surf1, ax=ax1, shrink=0.5, aspect=5)
    
    # 精确解
    ax2 = fig.add_subplot(132, projection='3d')
    surf2 = ax2.plot_surface(X, Y, U_exact, cmap='viridis', edgecolor='none')
    ax2.set_title(f'Exact Solution at t={time_slice}')
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    ax2.set_zlabel('u')
    fig.colorbar(surf2, ax=ax2, shrink=0.5, aspect=5)
    
    # 误差
    ax3 = fig.add_subplot(133, projection='3d')
    surf3 = ax3.plot_surface(X, Y, error, cmap='hot', edgecolor='none')
    ax3.set_title(f'Error at t={time_slice}')
    ax3.set_xlabel('x')
    ax3.set_ylabel('y')
    ax3.set_zlabel('Error')
    fig.colorbar(surf3, ax=ax3, shrink=0.5, aspect=5)
    
    plt.tight_layout()
    plt.show()
    
    # 计算整体误差指标
    l2_error = np.sqrt(np.mean(error**2))
    max_error = np.max(error)
    print(f"时间切片 t={time_slice} 的误差:")
    print(f"  二范数误差: {l2_error:.4e}")
    print(f"  最大误差: {max_error:.4e}")

# Main execution
if __name__ == "__main__":
    # 问题参数
    epsilon = 1e-3  
    T = 1.0
    num_explicit_bases = 2
    
    # 创建网络
    print("创建神经网络...")
    net = Net(hidden_size=100, subspace_dim=100)  
    
    # 训练阶段
    print("开始训练网络...")
    xyt_interior, xyt_initial, xyt_boundary, losses = train_network(
        net, epsilon, F_source, num_epochs=5000, early_stop_threshold=1e-1, T=T
    )
    
    # 绘制训练损失
    plt.figure()
    plt.semilogy(losses)
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.show()
    
    # 显式基方法阶段
    print("开始组装矩阵...")
    A, f_vec = assemble_matrix(
        net, explicit_bases, xyt_interior, xyt_initial, xyt_boundary, F_source, epsilon, num_explicit_bases
    )
    
    print("求解线性系统...")
    w = solve_linear_system(A, f_vec)
    
    # 评估和可视化结果
    print("可视化结果...")
    visualize_at_time(net, w, epsilon, num_explicit_bases, device, time_slice=0.0)
    visualize_at_time(net, w, epsilon, num_explicit_bases, device, time_slice=0.5)
    visualize_at_time(net, w, epsilon, num_explicit_bases, device, time_slice=1.0)