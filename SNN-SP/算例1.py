import torch
import torch.nn as nn
import numpy as np
from scipy.linalg import lstsq
import matplotlib.pyplot as plt

# PDE 源项 f(x)
def f(x):
    pi = torch.pi
    sin = torch.sin
    cos = torch.cos
    return epsilon * (pi**2) * sin(pi * x) + pi * cos(pi * x)

# 设置随机种子
torch.manual_seed(42)

# 定义神经网络
class Net(nn.Module):
    def __init__(self, hidden_size=100, subspace_dim=200):
        super(Net, self).__init__()
        self.hidden = nn.Sequential(
            nn.Linear(1, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, subspace_dim),
            nn.Tanh()
        )
        self.output = nn.Linear(subspace_dim, 1, bias=False)  # 最后一层不训练

    def forward(self, x):
        x = self.hidden(x)
        x = self.output(x)
        return x

    def get_hidden_layer_output(self, x):
        return self.hidden(x)

# 第一阶段：训练神经网络（除最后一层）
def train_network(net, epsilon, f, u0, u1, num_epochs=1000):
    optimizer = torch.optim.Adam(net.hidden.parameters(), lr=0.001)
    x = torch.linspace(0, 1, 100).reshape(-1, 1).requires_grad_(True)
    x_b = torch.tensor([[0.0], [1.0]], requires_grad=True)

    for epoch in range(num_epochs):
        optimizer.zero_grad()
        
        # PDE残差
        u = net(x)
        u_x = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True)[0]
        u_xx = torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(u_x), create_graph=True)[0]
        pde_residual = -epsilon * u_xx + u_x - f(x)
        
        # 边界条件残差
        u_b = net(x_b)
        bc_residual = u_b - torch.tensor([[u0], [u1]], dtype=torch.float32)
        
        # 损失函数
        loss = torch.mean(pde_residual**2) + torch.mean(bc_residual**2)
        # loss = torch.mean(pde_residual**2)
        loss.backward()
        optimizer.step()
        
        if epoch % 100 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item():.6f}")

# 定义显式基函数（例如边界层项）
def explicit_basis(x, epsilon):
    return torch.exp(-(1 - x) / epsilon).reshape(-1, 1)

# 计算 PDE 算子 L[phi_j] = -epsilon * phi_j'' + phi_j'
def compute_pde_operator(net, x, epsilon, explicit_basis_func):
    """
    计算 PDE 算子 L[phi_j] = -epsilon * phi_j'' + phi_j' 对所有基函数。
    
    参数：
    - net: 神经网络模型
    - x: 形状为 [num_points, 1] 的输入张量
    - epsilon: PDE 中的系数
    - explicit_basis_func: 显式基函数
    
    返回：
    - L_basis: 形状为 [num_points, subspace_dim + 1] 的张量
    """
    x = x.requires_grad_(True)
    phi = net.get_hidden_layer_output(x)  # [num_points, subspace_dim]
    psi = explicit_basis_func(x, epsilon)  # [num_points, 1]
    basis = torch.cat([phi, psi], dim=1)  # [num_points, subspace_dim + 1]
    
    L_basis_list = []
    for j in range(basis.shape[1]):
        basis_j = basis[:, j].unsqueeze(1)  # [num_points, 1]
        
        # 计算一阶导数
        basis_j_x = torch.autograd.grad(basis_j, x, grad_outputs=torch.ones_like(basis_j), create_graph=True)[0]
        
        # 计算二阶导数
        basis_j_xx = torch.autograd.grad(basis_j_x, x, grad_outputs=torch.ones_like(basis_j_x), create_graph=True)[0]
        
        # 计算 PDE 算子
        L_basis_j = -epsilon * basis_j_xx + basis_j_x
        L_basis_list.append(L_basis_j)
    
    L_basis = torch.cat(L_basis_list, dim=1)  # [num_points, subspace_dim + 1]
    return L_basis

# 组装矩阵 A 和向量 f
def assemble_matrix(net, explicit_basis_func, x_interior, x_boundary, f, u0, u1, epsilon):
    x_interior_tensor = torch.tensor(x_interior, dtype=torch.float32).reshape(-1, 1)
    x_boundary_tensor = torch.tensor(x_boundary, dtype=torch.float32).reshape(-1, 1)
    
    # 基函数值
    # phi_interior = net.get_hidden_layer_output(x_interior_tensor)  # [num_points, subspace_dim]
    # psi_interior = explicit_basis_func(x_interior_tensor, epsilon)  # [num_points, 1]
    # basis_interior = torch.cat([phi_interior, psi_interior], dim=1)  # [num_points, subspace_dim + 1]
    
    phi_boundary = net.get_hidden_layer_output(x_boundary_tensor)  # [2, subspace_dim]
    psi_boundary = explicit_basis_func(x_boundary_tensor, epsilon)  # [2, 1]
    basis_boundary = torch.cat([phi_boundary, psi_boundary], dim=1)  # [2, subspace_dim + 1]
    
    # 计算 PDE 算子
    L_basis = compute_pde_operator(net, x_interior_tensor, epsilon, explicit_basis_func)
    
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
def compute_solution(net, explicit_basis_func, w, x, epsilon):
    x_tensor = torch.tensor(x, dtype=torch.float32).reshape(-1, 1)
    phi = net.get_hidden_layer_output(x_tensor).detach().numpy()
    psi = explicit_basis_func(x_tensor, epsilon).detach().numpy()
    basis = np.hstack([phi, psi])
    u_approx = basis @ w
    return u_approx

# 主程序
if __name__ == "__main__":
    epsilon = 1e-5
    u0 = 0.0
    u1 = 1.0
    net = Net(hidden_size=100, subspace_dim=200)
    
    # 第一阶段：训练网络
    train_network(net, epsilon, f, u0, u1)
    
    # 采样点
    x_interior = np.linspace(0, 1, 50)[1:-1]  # 内部点
    x_boundary = np.array([0.0, 1.0])
    
    # 组装矩阵
    A, f_vec = assemble_matrix(net, explicit_basis, x_interior, x_boundary, f, u0, u1, epsilon)
    
    # 求解线性系统
    w = solve_linear_system(A, f_vec)
    
    # 计算近似解
    x_eval = np.linspace(0, 1, 1000)
    u_approx = compute_solution(net, explicit_basis, w, x_eval, epsilon)
    
    # 精确解
    u_exact = np.sin(np.pi * x_eval) + np.exp((x_eval - 1) / epsilon) * (1 - np.exp(-x_eval / epsilon)) / (1 - np.exp(-1 / epsilon))
    
    # 输出结果
    error = np.mean((u_approx - u_exact)**2)
    print(f"Mean Squared Error: {error:.2e}")
    
    # 可视化
    plt.plot(x_eval, u_approx, label="Approximate Solution")
    plt.plot(x_eval, u_exact, label="Exact Solution", linestyle="--")
    plt.legend()
    plt.show()
    # plt.savefig("solution_comparison.png")