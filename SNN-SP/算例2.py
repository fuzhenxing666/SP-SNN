import torch
import torch.nn as nn
import numpy as np
from scipy.linalg import lstsq
import matplotlib.pyplot as plt
import uuid

# PDE 源项 f(x)
def f(x):
    return torch.zeros_like(x)

# 设置随机种子
torch.manual_seed(42)

# 定义神经网络
class Net(nn.Module):
    def __init__(self, hidden_size=100, subspace_dim=200):
        super(Net, self).__init__()
        self.hidden = nn.Sequential(
            nn.Linear(1, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, subspace_dim),
            nn.Tanh()
        )
        self.output = nn.Linear(subspace_dim, 1, bias=False)  # 最后一层不训练
        # 初始化 output 层权重为全 1，并确保 requires_grad=True
        with torch.no_grad():
            self.output.weight.fill_(1.0)
        self.output.weight.requires_grad = True  # 确保权重参与梯度计算

    def forward(self, x):
        x = self.hidden(x)
        x = self.output(x)
        return x

    def get_hidden_layer_output(self, x):
        return self.hidden(x)

# 第一阶段：训练神经网络（除最后一层）
def train_network(net, epsilon, f, u0, u1, num_epochs=500, early_stop_threshold=1e-3):
    optimizer = torch.optim.Adam(net.hidden.parameters(), lr=0.001)
    x = torch.linspace(0, 1, 100).reshape(-1, 1).requires_grad_(True)
    x_b = torch.tensor([[0.0], [1.0]], requires_grad=True)

    # 计算初始损失
    u = net(x)
    if not u.requires_grad:
        raise RuntimeError("u does not require grad")
    u_x = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    u_xx = torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(u_x), create_graph=True)[0]
    pde_residual = epsilon * u_xx + (1 + epsilon) * u_x + u - f(x)
    # pde_residual = u_x + u - f(x)
    u_b = net(x_b)
    bc_residual = u_b - torch.tensor([[u0], [u1]], dtype=torch.float32)
    initial_loss = torch.mean(pde_residual**2) + torch.mean(bc_residual**2)
    initial_loss = initial_loss.item()
    print(f"Initial Loss: {initial_loss:.6f}")

    for epoch in range(num_epochs):
        optimizer.zero_grad()
        
        # PDE 残差
        u = net(x)
        u_x = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True)[0]
        u_xx = torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(u_x), create_graph=True)[0]
        pde_residual = epsilon * u_xx + (1 + epsilon) * u_x + u - f(x)
        # pde_residual = u_x + u - f(x)
        
        # 边界条件残差
        u_b = net(x_b)
        bc_residual = u_b - torch.tensor([[u0], [u1]], dtype=torch.float32)
        
        # 损失函数
        loss = torch.mean(pde_residual**2) + torch.mean(bc_residual**2)
        loss.backward()
        optimizer.step()
        
        # 检查提前停止条件
        current_loss = loss.item()
        if current_loss <= initial_loss * early_stop_threshold:
            print(f"Early stopping at Epoch {epoch}, Loss: {current_loss:.6f}")
            break
        
        if epoch % 100 == 0:
            print(f"Epoch {epoch}, Loss: {current_loss:.6f}")

# 定义多个显式基函数 
def explicit_bases(x, epsilon, num_explicit_bases, alpha=0.7):
    if num_explicit_bases == 0:
        return torch.empty((x.shape[0], 0), device=x.device)  # 返回空张量
    bases = []
    for k in range(1, num_explicit_bases + 1):
        psi_k = torch.exp(-x / epsilon).reshape(-1, 1)
        bases.append(psi_k)
    
    return torch.cat(bases, dim=1)  # [num_points, num_explicit_bases]

# 计算 PDE 算子 L[phi_j] = -epsilon * phi_j'' + (1 + epsilon) * phi_j' + phi_j
def compute_pde_operator(net, x, epsilon, explicit_bases_func, num_explicit_bases):
    x = x.requires_grad_(True)
    phi = net.get_hidden_layer_output(x)  # [num_points, subspace_dim]
    psi = explicit_bases_func(x, epsilon, num_explicit_bases)  # [num_points, num_explicit_bases] or [num_points, 0]
    
    if num_explicit_bases > 0:
        basis = torch.cat([phi, psi], dim=1)  # [num_points, subspace_dim + num_explicit_bases]
    else:
        basis = phi  # [num_points, subspace_dim]
    
    L_basis_list = []
    for j in range(basis.shape[1]):
        basis_j = basis[:, j].unsqueeze(1)  # [num_points, 1]
        
        # 计算一阶导数
        basis_j_x = torch.autograd.grad(basis_j, x, grad_outputs=torch.ones_like(basis_j), create_graph=True)[0]
        
        # 计算二阶导数
        basis_j_xx = torch.autograd.grad(basis_j_x, x, grad_outputs=torch.ones_like(basis_j_x), create_graph=True)[0]
        
        # 计算 PDE 算子
        L_basis_j = epsilon * basis_j_xx + (1 + epsilon) * basis_j_x + basis_j
        L_basis_list.append(L_basis_j)
    
    L_basis = torch.cat(L_basis_list, dim=1)  # [num_points, subspace_dim + num_explicit_bases]
    return L_basis

# 组装矩阵 A 和向量 f
def assemble_matrix(net, explicit_bases_func, x_interior, x_boundary, f, u0, u1, epsilon, num_explicit_bases):
    x_interior_tensor = torch.tensor(x_interior, dtype=torch.float32).reshape(-1, 1)
    x_boundary_tensor = torch.tensor(x_boundary, dtype=torch.float32).reshape(-1, 1)
    
    # 基函数值
    phi_boundary = net.get_hidden_layer_output(x_boundary_tensor)  # [2, subspace_dim]
    psi_boundary = explicit_bases_func(x_boundary_tensor, epsilon, num_explicit_bases)  # [2, num_explicit_bases] or [2, 0]
    
    if num_explicit_bases > 0:
        basis_boundary = torch.cat([phi_boundary, psi_boundary], dim=1)  # [2, subspace_dim + num_explicit_bases]
    else:
        basis_boundary = phi_boundary  # [2, subspace_dim]
    
    # 计算 PDE 算子
    L_basis = compute_pde_operator(net, x_interior_tensor, epsilon, explicit_bases_func, num_explicit_bases)
    
    # 组装 A 和 f
    A_interior = L_basis.detach().numpy()
    f_interior = f(x_interior_tensor).detach().numpy().flatten()
    
    A_boundary = basis_boundary.detach().numpy()
    f_boundary = np.array([u0, u1])
    
    A = np.vstack([A_interior, A_boundary])
    f = np.hstack([f_interior, f_boundary])
    
    return A, f

# 第二阶段：求解线性系统
def solve_linear_system(A, f):
    w, _, _, _ = lstsq(A, f)
    return w

# 计算近似解
def compute_solution(net, explicit_bases_func, w, x, epsilon, num_explicit_bases):
    x_tensor = torch.tensor(x, dtype=torch.float32).reshape(-1, 1)
    phi = net.get_hidden_layer_output(x_tensor).detach().numpy()
    if num_explicit_bases > 0:
        psi = explicit_bases_func(x_tensor, epsilon, num_explicit_bases).detach().numpy()
        basis = np.hstack([phi, psi])
    else:
        basis = phi
    u_approx = basis @ w
    return u_approx

# 主程序
if __name__ == "__main__":
    epsilon = 0.001 
    u0 = 0.0
    u1 = 1.0
    num_explicit_bases = 1  
    net = Net(hidden_size=100, subspace_dim=100)
    
    # 第一阶段：训练网络
    train_network(net, epsilon, f, u0, u1, early_stop_threshold=1e-3)
    
    # 采样点
    x_interior = np.linspace(0, 1, 50)[1:-1]  # 内部点
    x_boundary = np.array([0.0, 1.0])
    
    # 组装矩阵
    A, f_vec = assemble_matrix(net, explicit_bases, x_interior, x_boundary, f, u0, u1, epsilon, num_explicit_bases)
    
    # 求解线性系统
    w = solve_linear_system(A, f_vec)
    
    # 计算近似解
    x_eval = np.linspace(0, 1, 1000)
    u_approx = compute_solution(net, explicit_bases, w, x_eval, epsilon, num_explicit_bases)
    
    # 精确解
    # u_exact = (np.exp(-x_eval) - np.exp(-x_eval / epsilon)) / (np.exp(-1) - np.exp(-1 / epsilon))
    u_exact = (np.exp(-x_eval/epsilon) - np.exp(-x_eval)) / (np.exp(-1 / epsilon) - np.exp(-1))
    # 输出结果
    error = np.mean((u_approx - u_exact)**2)
    print(f"Mean Squared Error: {error:.2e}")

    # print(w[-1])
    
    # 可视化
    plt.plot(x_eval, u_approx, label="Approximate Solution")
    plt.plot(x_eval, u_exact, label="Exact Solution", linestyle="--")
    plt.legend()
    plt.show()