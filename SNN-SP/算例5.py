import torch
import torch.nn as nn
import numpy as np
from scipy.linalg import lstsq
import matplotlib.pyplot as plt

# Check for GPU availability
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Set default tensor type to double precision
torch.set_default_tensor_type(torch.DoubleTensor)
torch.manual_seed(42)   # 设置随机种子

class NormalizeLayer(nn.Module):
    """将输入数据线性变换到[-1, 1]范围"""
    def __init__(self, a, b):
        super(NormalizeLayer, self).__init__()
        self.register_buffer('a', torch.tensor(a, dtype=torch.float64))
        self.register_buffer('b', torch.tensor(b, dtype=torch.float64))
        
    def forward(self, x):
        return self.a * x + self.b

class Net(nn.Module):
    def __init__(self, hidden_size=100, subspace_dim=100):
        super(Net, self).__init__()
        # 输入范围 [0,1] -> 变换到 [-1,1]
        # self.normalize = NormalizeLayer(a=[2.0, 2.0], b=[-1.0, -1.0])
        
        self.hidden = nn.Sequential(
            nn.Linear(2, hidden_size),
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

    def forward(self, xy):
        # 先进行归一化处理
        # xy_normalized = self.normalize(xy)
        x = self.hidden(xy)
        x = self.output(x)
        return x

    def get_hidden_layer_output(self, xy):
        # xy_normalized = self.normalize(xy)
        return self.hidden(xy)

def exact_solution(xy, epsilon):
    x = xy[:, 0:1]
    y = xy[:, 1:2]
    
    sqrt_term = torch.sqrt(torch.tensor(1.0, device=xy.device) + 4*(epsilon**2)*(torch.pi**2))
    r1 = (1.0 + sqrt_term)/(2.0*epsilon)
    r2 = (1.0 - sqrt_term)/(2.0*epsilon)

    exp_r2 = torch.exp(r2)
    exp_neg_r1 = torch.exp(-r1)
    # 分子
    term1 = (2.0 - exp_r2) * torch.exp(r1*(x-1.0))
    term2 = (1.0 - 2.0*exp_neg_r1) * torch.exp(r2*x)

    # 分母
    denominator = 1.0 - torch.exp(r2 - r1)

    return (term1 + term2)/denominator * torch.sin(torch.pi*y)

# PDE source term f(x, t)
def f(xy, epsilon):
    y = xy[:, 1:2]
    return torch.zeros_like(y)

def train_network(net, epsilon, f, device, num_epochs=500, early_stop_threshold=1e-3):
    # 定义优化器
    optimizer = torch.optim.Adam(net.hidden.parameters(), lr=0.001)
    
    # 定义采样点数量
    N_i, N_b = 30, 30
    
    # 内部点
    x_i = torch.linspace(0.0, 1.0, N_i, device=device).reshape(-1, 1)[1:-1]  # 去掉边界点
    y_i = torch.linspace(0.0, 1.0, N_i, device=device).reshape(-1, 1)[1:-1]  # 去掉边界点
    X, Y = torch.meshgrid(x_i.squeeze(), y_i.squeeze(), indexing='ij')
    xy_i_base = torch.stack([X.flatten(), Y.flatten()], dim=1).requires_grad_(True)

    # 边界点
    xy_b = torch.linspace(0.0, 1.0, N_b, device=device).reshape(-1, 1)

    xy_b0_base = torch.cat([torch.zeros(N_b,1, dtype=torch.float64, device=device), xy_b], dim=1)  # x=0
    xy_b1_base = torch.cat([torch.ones(N_b,1, dtype=torch.float64, device=device), xy_b], dim=1)   # x=1
    xy_b2_base = torch.cat([xy_b, torch.zeros(N_b,1, dtype=torch.float64, device=device)], dim=1)  # y=0
    xy_b3_base = torch.cat([xy_b, torch.ones(N_b,1, dtype=torch.float64, device=device)], dim=1)   # y=1
    
    # 训练循环
    for epoch in range(num_epochs):
        # 在每次迭代中创建新的张量并启用梯度计算，确保计算图独立
        xy_i = xy_i_base.clone().requires_grad_(True)   # 内部点
        xy_b0 = xy_b0_base.clone().requires_grad_(True)  # 边界 x=0
        xy_b1 = xy_b1_base.clone().requires_grad_(True)  # 边界 x=1
        xy_b2 = xy_b2_base.clone().requires_grad_(True)  # 边界 y=0
        xy_b3 = xy_b3_base.clone().requires_grad_(True)  # 边界 y=1
        
        # 前向传播计算网络输出
        u_i = net(xy_i)
        
        # 计算偏导数
        u_x = torch.autograd.grad(u_i, xy_i, grad_outputs=torch.ones_like(u_i), create_graph=True)[0][:, 0:1]  # 空间x一阶偏导
        u_xx = torch.autograd.grad(u_x, xy_i, grad_outputs=torch.ones_like(u_x), create_graph=True)[0][:, 0:1]  # 空间x二阶偏导
        u_y = torch.autograd.grad(u_i, xy_i, grad_outputs=torch.ones_like(u_i), create_graph=True)[0][:, 1:2]  # 空间y一阶偏导
        u_yy = torch.autograd.grad(u_y, xy_i, grad_outputs=torch.ones_like(u_y), create_graph=True)[0][:, 1:2]  # 空间y二阶偏导

        
        # 计算 PDE 残差
        pde_res = -epsilon * (u_xx + u_yy) + u_x - f(xy_i, epsilon)  
        
        # 计算边界条件残差，分离 exact_solution 的计算图
        bc_res0 = net(xy_b0) - exact_solution(xy_b0, epsilon).detach()  # 边界 x=0 残差
        bc_res1 = net(xy_b1) - exact_solution(xy_b1, epsilon).detach()  # 边界 x=1 残差
        bc_res2 = net(xy_b2) - exact_solution(xy_b2, epsilon).detach()  # 边界 y=0 残差
        bc_res3 = net(xy_b3) - exact_solution(xy_b3, epsilon).detach()  # 边界 y=1 残差
        bc_res = torch.cat([bc_res0, bc_res1, bc_res2, bc_res3], dim=0)  # 拼接所有边界残差
        
        # 计算总损失
        loss = torch.mean(pde_res**2)
        # loss = torch.mean(pde_res**2) + torch.mean(bc_res**2)

        if epoch == 0:
            initial_loss = loss.item()
        
        # 梯度清零，反向传播，更新参数
        optimizer.zero_grad()
        loss.backward()  # 默认释放计算图，无需 retain_graph=True
        optimizer.step()
        if loss.item() <= early_stop_threshold * initial_loss:
            print(f"提前停止于第 {epoch} 轮, 损失: {loss.item():.6f}")
            break

        # 每 100 次迭代输出损失
        if epoch % 100 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item():.6f}")
    
    # 返回基础网格点
    return xy_i_base, xy_b0_base, xy_b1_base, xy_b2_base, xy_b3_base

# 显式基函数
def explicit_bases(xy, epsilon, num_explicit_bases):
    x = xy[:, 0:1]
    y = xy[:, 1:2]

    if num_explicit_bases == 0:
        return torch.empty(xy.shape[0], 0, dtype=torch.float64, device=xy.device)
    
    bases = []

    if num_explicit_bases >= 1:
        bases.append(torch.exp((x-1.0) / epsilon))
    if num_explicit_bases >= 2:
        bases.append(torch.exp((x-1.0) / epsilon) * torch.sin(torch.pi * y))
    
    # bases = [torch.exp((x-1.0) / epsilon)*torch.sin(torch.pi * y)]  
    return torch.cat(bases[:num_explicit_bases], dim=1)


def compute_pde_operator(net, xy, epsilon, explicit_bases_func, num_explicit_bases):
    # 确保 xy 可以求导
    xy = xy.requires_grad_(True)
    
    # 获取网络输出和显式基函数
    phi = net.get_hidden_layer_output(xy)  # [num_points, subspace_dim]
    psi = explicit_bases_func(xy, epsilon, num_explicit_bases)  # [num_points, num_explicit_bases]
    
    # 拼接基函数
    if num_explicit_bases > 0:
        basis = torch.cat([phi, psi], dim=1)  # [num_points, subspace_dim + num_explicit_bases]
    else:
        basis = phi  # [num_points, subspace_dim]
    
    L_basis_list = []
    for j in range(basis.shape[1]):
        basis_j = basis[:, j].unsqueeze(1)  # [num_points, 1]
        
        # 计算一阶梯度
        grad_basis_j = torch.autograd.grad(basis_j, xy, grad_outputs=torch.ones_like(basis_j), create_graph=True)[0]  # [num_points, 2]
        
        # 提取对 x 和 y 的一阶偏导
        basis_j_x = grad_basis_j[:, 0:1]  # 对 x 的偏导数
        basis_j_y = grad_basis_j[:, 1:2]  # 对 y 的偏导数
        
        # 计算对 x 和 y 的二阶偏导
        basis_j_xx = torch.autograd.grad(basis_j_x, xy, grad_outputs=torch.ones_like(basis_j_x), create_graph=True)[0][:, 0:1]  # [num_points, 1]
        basis_j_yy = torch.autograd.grad(basis_j_y, xy, grad_outputs=torch.ones_like(basis_j_y), create_graph=True)[0][:, 1:2]  # [num_points, 1]
        
        # 计算 PDE 算子
        L_basis_j = -epsilon * (basis_j_xx + basis_j_yy) + basis_j_x
        L_basis_list.append(L_basis_j)
    
    # 拼接所有基函数的 PDE 算子结果
    L_basis = torch.cat(L_basis_list, dim=1)  # [num_points, subspace_dim + num_explicit_bases]
    return L_basis

# 组装矩阵
def assemble_matrix(net, explicit_bases_func, xy_interior, xy_b0, xy_b1, xy_b2, xy_b3, f, epsilon, num_explicit_bases):
    # PDE部分
    L_basis = compute_pde_operator(net, xy_interior, epsilon, explicit_bases_func, num_explicit_bases)
    A_i = L_basis.detach().cpu().numpy()
    f_i = f(xy_interior, epsilon).detach().cpu().numpy().flatten()
    
    # 合并所有边界点（x=0, x=1, y=0, y=1）
    xy_b_all = torch.cat([xy_b0, xy_b1, xy_b2, xy_b3], dim=0)
    
    # 统一计算边界点的基函数
    phi_b_all = net.get_hidden_layer_output(xy_b_all)
    psi_b_all = explicit_bases_func(xy_b_all, epsilon, num_explicit_bases)
    basis_b_all = torch.cat([phi_b_all, psi_b_all], dim=1) if num_explicit_bases > 0 else phi_b_all
    
    # 统一计算边界点的精确解
    u_b_all = exact_solution(xy_b_all, epsilon).detach().cpu().numpy().flatten()
    
    # 组装全局矩阵和向量
    A = np.vstack([A_i, basis_b_all.detach().cpu().numpy()])  # 垂直堆叠内部和边界
    f_vec = np.hstack([f_i, u_b_all])                   # 水平拼接内部残差和边界条件
    return A, f_vec

# 最小二乘法求解线性系统
def solve_linear_system(A, f):
    w, _, _, _ = lstsq(A, f)
    return w

# 计算近似解
def compute_solution(net, explicit_bases_func, w, xy, epsilon, num_explicit_bases, device):
    xy_tensor = torch.tensor(xy, dtype=torch.float64, device=device)
    phi = net.get_hidden_layer_output(xy_tensor).detach().cpu().numpy()
    psi = explicit_bases_func(xy_tensor, epsilon, num_explicit_bases).detach().cpu().numpy() if num_explicit_bases > 0 else np.array([])
    basis = np.hstack([phi, psi]) if num_explicit_bases > 0 else phi
    return basis @ w

def evaluate_and_plot(net, w, epsilon, num_explicit_bases, device, explicit_bases_func=explicit_bases, n_points=100):
    # 生成评估点
    x_eval = np.linspace(0.0, 1.0, n_points)
    y_eval = np.linspace(0.0, 1.0, n_points)
    X, Y = np.meshgrid(x_eval, y_eval)
    xy_eval = np.column_stack((X.ravel(), Y.ravel()))
    
    # 计算近似解和精确解
    u_approx = compute_solution(net, explicit_bases_func, w, xy_eval, epsilon, num_explicit_bases, device)
    xy_tensor = torch.tensor(xy_eval, dtype=torch.float64, device=device)
    u_exact = exact_solution(xy_tensor, epsilon).detach().cpu().numpy()
    error = np.abs(u_approx.reshape(-1,1) - u_exact)
    
    # 正确重塑数组
    u_approx_2d = u_approx.reshape(n_points, n_points)
    u_exact_2d = u_exact.reshape(n_points, n_points)
    error_2d = error.reshape(n_points, n_points)

    # 计算误差指标
    max_error = np.max(error_2d)
    l2_error = np.sqrt(np.mean(error_2d**2))
    print(f"二范数误差: {l2_error:.2e}")
    print(f"最大误差: {max_error:.2e}")
    
    # 绘制结果
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    
    im0 = axs[0].contourf(X, Y, u_exact_2d, 100, cmap='jet')
    plt.colorbar(im0, ax=axs[0])
    axs[0].set_title("Exact Solution")
    axs[0].set_xlabel("x")
    axs[0].set_ylabel("y")
    
    im1 = axs[1].contourf(X, Y, u_approx_2d, 100, cmap='jet')
    plt.colorbar(im1, ax=axs[1])
    axs[1].set_title("Predicted Solution")
    axs[1].set_xlabel("x")
    axs[1].set_ylabel("y")
    
    im2 = axs[2].contourf(X, Y, error_2d, 100, cmap='jet')
    plt.colorbar(im2, ax=axs[2])
    axs[2].set_title("Pointwise Error")
    axs[2].set_xlabel("x")
    axs[2].set_ylabel("y")
    
    plt.tight_layout()
    plt.show()
    
    return {
        'u_exact': u_exact_2d,
        'u_approx': u_approx_2d,
        'error': error_2d,
        'max_error': max_error,
        'l2_error': l2_error
    }

# 主函数
if __name__ == "__main__":
    epsilon = 1e-3  
    num_explicit_bases = 2
    net = Net(hidden_size=100, subspace_dim=100).to(device)
    xy_interior, xy_b0, xy_b1, xy_b2, xy_b3 = train_network(net, epsilon, f, device, num_epochs=5000)
    A, f_vec = assemble_matrix(net, explicit_bases, xy_interior, xy_b0, xy_b1, xy_b2, xy_b3, f, epsilon, num_explicit_bases)
    w = solve_linear_system(A, f_vec)

    # 调用评估和绘图函数
    results = evaluate_and_plot(net, w, epsilon, num_explicit_bases, device)