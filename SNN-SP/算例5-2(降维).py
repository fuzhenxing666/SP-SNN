import torch
import torch.nn as nn
import numpy as np
from scipy.linalg import lstsq
import matplotlib.pyplot as plt

# 设置随机种子
torch.manual_seed(42)

# 定义神经网络
class Net(nn.Module):
    def __init__(self, hidden_size=100, subspace_dim=200):
        super(Net, self).__init__()
        self.hidden = nn.Sequential(
            nn.Linear(2, hidden_size),  # 输入为 (x, y)
            nn.Tanh(),
            nn.Linear(hidden_size, subspace_dim),
            nn.Tanh()
        )
        self.output = nn.Linear(subspace_dim, 1, bias=False)
        with torch.no_grad():
            self.output.weight.fill_(1.0)
        self.output.weight.requires_grad = True

    def forward(self, x, y):
        xy = torch.cat([x, y], dim=1)
        hidden_out = self.hidden(xy)
        output = self.output(hidden_out)
        return output

    def get_hidden_layer_output(self, x, y):
        xy = torch.cat([x, y], dim=1)
        return self.hidden(xy)

# 训练神经网络（第一阶段）
def train_network(net, epsilon, num_epochs=500, early_stop_threshold=1e-3):
    optimizer = torch.optim.Adam(net.hidden.parameters(), lr=0.001)
    
    # 内部点
    x_interior = torch.linspace(0, 1, 50).reshape(-1, 1).requires_grad_(True)
    y_interior = torch.linspace(0, 1, 50).reshape(-1, 1).requires_grad_(True)
    X, Y = torch.meshgrid(x_interior.squeeze(), y_interior.squeeze(), indexing='ij')
    X = X.reshape(-1, 1)
    Y = Y.reshape(-1, 1)
    
    # 边界点
    x_b = torch.linspace(0, 1, 50).reshape(-1, 1)
    y_b0 = torch.zeros_like(x_b)
    y_b1 = torch.ones_like(x_b)
    y_b = torch.linspace(0, 1, 50).reshape(-1, 1)
    x_b0 = torch.zeros_like(y_b)
    x_b1 = torch.ones_like(y_b)
    
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        
        # PDE 残差
        u = net(X, Y)
        u_x = torch.autograd.grad(u, X, grad_outputs=torch.ones_like(u), create_graph=True)[0]
        u_y = torch.autograd.grad(u, Y, grad_outputs=torch.ones_like(u), create_graph=True)[0]
        u_xx = torch.autograd.grad(u_x, X, grad_outputs=torch.ones_like(u_x), create_graph=True)[0]
        u_yy = torch.autograd.grad(u_y, Y, grad_outputs=torch.ones_like(u_y), create_graph=True)[0]
        pde_residual = -epsilon * (u_xx + u_yy) + u_x
        
        # 边界条件残差
        u_b_y0 = net(x_b, y_b0)
        u_b_y1 = net(x_b, y_b1)
        u_b_x0 = net(x_b0, y_b)
        u_b_x1 = net(x_b1, y_b)
        bc_residual_y0 = u_b_y0 - 0
        bc_residual_y1 = u_b_y1 - 0
        bc_residual_x0 = u_b_x0 - torch.sin(np.pi * y_b)
        bc_residual_x1 = u_b_x1 - 2 * torch.sin(np.pi * y_b)
        
        # 损失函数
        loss_pde = torch.mean(pde_residual**2)
        loss_bc = torch.mean(bc_residual_y0**2) + torch.mean(bc_residual_y1**2) + \
                  torch.mean(bc_residual_x0**2) + torch.mean(bc_residual_x1**2)
        loss = loss_pde + loss_bc
        loss.backward()
        optimizer.step()
        
        if epoch % 100 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item():.6f}")

# 定义显式基函数
def explicit_bases(x, y, epsilon, num_explicit_bases):
    if num_explicit_bases == 0:
        return torch.empty((x.shape[0], 0), device=x.device)
    bases = []
    for k in range(1, num_explicit_bases + 1):
        psi_k = torch.exp((x - 1) / epsilon).reshape(-1, 1)
        bases.append(psi_k)
    return torch.cat(bases, dim=1)

# 计算PDE算子
def compute_pde_operator(net, x, y, epsilon, explicit_bases_func, num_explicit_bases):
    x = x.requires_grad_(True)
    y = y.requires_grad_(True)
    phi = net.get_hidden_layer_output(x, y)
    psi = explicit_bases_func(x, y, epsilon, num_explicit_bases)
    basis = torch.cat([phi, psi], dim=1) if num_explicit_bases > 0 else phi
    
    L_basis_list = []
    for j in range(basis.shape[1]):
        basis_j = basis[:, j].unsqueeze(1)
        basis_j_x = torch.autograd.grad(basis_j, x, grad_outputs=torch.ones_like(basis_j), create_graph=True)[0]
        basis_j_y = torch.autograd.grad(basis_j, y, grad_outputs=torch.ones_like(basis_j), create_graph=True)[0]
        basis_j_xx = torch.autograd.grad(basis_j_x, x, grad_outputs=torch.ones_like(basis_j_x), create_graph=True)[0]
        basis_j_yy = torch.autograd.grad(basis_j_y, y, grad_outputs=torch.ones_like(basis_j_y), create_graph=True)[0]
        L_basis_j = -epsilon * (basis_j_xx + basis_j_yy) + basis_j_x
        L_basis_list.append(L_basis_j)
    return torch.cat(L_basis_list, dim=1)

# 组装矩阵
def assemble_matrix(net, explicit_bases_func, x_interior, y_interior, x_b, y_b, epsilon, num_explicit_bases):
    x_in = torch.tensor(x_interior, dtype=torch.float32).reshape(-1, 1)
    y_in = torch.tensor(y_interior, dtype=torch.float32).reshape(-1, 1)
    x_b = torch.tensor(x_b, dtype=torch.float32).reshape(-1, 1)
    y_b = torch.tensor(y_b, dtype=torch.float32).reshape(-1, 1)
    
    L_basis = compute_pde_operator(net, x_in, y_in, epsilon, explicit_bases_func, num_explicit_bases)
    A_interior = L_basis.detach().numpy()
    f_interior = np.zeros_like(x_interior).flatten()
    
    phi_b = net.get_hidden_layer_output(x_b, y_b)
    psi_b = explicit_bases_func(x_b, y_b, epsilon, num_explicit_bases)
    basis_b = torch.cat([phi_b, psi_b], dim=1) if num_explicit_bases > 0 else phi_b
    A_boundary = basis_b.detach().numpy()
    f_boundary = (2 * torch.sin(np.pi * y_b)).detach().numpy().flatten()  # 示例：x=1边界
    
    A = np.vstack([A_interior, A_boundary])
    f = np.hstack([f_interior, f_boundary])
    return A, f

# 求解线性系统
def solve_linear_system(A, f):
    w, _, _, _ = lstsq(A, f)
    return w

# 计算近似解
def compute_solution(net, explicit_bases_func, w, x, y, epsilon, num_explicit_bases):
    x_tensor = torch.tensor(x, dtype=torch.float32).reshape(-1, 1)
    y_tensor = torch.tensor(y, dtype=torch.float32).reshape(-1, 1)
    phi = net.get_hidden_layer_output(x_tensor, y_tensor).detach().numpy()
    psi = explicit_bases_func(x_tensor, y_tensor, epsilon, num_explicit_bases).detach().numpy()
    basis = np.hstack([phi, psi]) if num_explicit_bases > 0 else phi
    u_approx = basis @ w
    return u_approx

# 解析解
def analytical_solution(x, y, epsilon):
    r1 = (1 + np.sqrt(1 + 4 * epsilon**2 * np.pi**2)) / (2 * epsilon)
    r2 = (1 - np.sqrt(1 + 4 * epsilon**2 * np.pi**2)) / (2 * epsilon)
    numerator = (2 - np.exp(r2)) * np.exp(r1 * (x - 1)) + (1 - 2 * np.exp(-r1)) * np.exp(r2 * x)
    denominator = 1 - np.exp(r2 - r1)
    return (numerator / denominator) * np.sin(np.pi * y)

# 主程序
if __name__ == "__main__":
    epsilon = 1
    num_explicit_bases = 1
    net = Net(hidden_size=100, subspace_dim=100)
    
    # 训练网络
    train_network(net, epsilon)
    
    # 采样点
    x_interior = np.linspace(0, 1, 20)[1:-1]
    y_interior = np.linspace(0, 1, 20)[1:-1]
    X, Y = np.meshgrid(x_interior, y_interior, indexing='ij')
    x_in = X.flatten()
    y_in = Y.flatten()
    x_b = np.ones(50)
    y_b = np.linspace(0, 1, 50)
    
    # 组装矩阵并求解
    A, f = assemble_matrix(net, explicit_bases, x_in, y_in, x_b, y_b, epsilon, num_explicit_bases)
    w = solve_linear_system(A, f)
    
    # 计算近似解
    x_eval = np.linspace(0, 1, 100)
    y_eval = np.linspace(0, 1, 100)
    X_eval, Y_eval = np.meshgrid(x_eval, y_eval, indexing='ij')
    u_approx = compute_solution(net, explicit_bases, w, X_eval.flatten(), Y_eval.flatten(), epsilon, num_explicit_bases)
    u_exact = analytical_solution(X_eval, Y_eval, epsilon)
    
    # 可视化
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.contourf(X_eval, Y_eval, u_approx.reshape(X_eval.shape), levels=20, cmap="viridis")
    plt.colorbar(label="u_approx")
    plt.title("Approximate Solution")
    plt.xlabel("x")
    plt.ylabel("y")
    
    plt.subplot(1, 2, 2)
    plt.contourf(X_eval, Y_eval, u_exact, levels=20, cmap="viridis")
    plt.colorbar(label="u_exact")
    plt.title("Exact Solution")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.tight_layout()
    plt.show()