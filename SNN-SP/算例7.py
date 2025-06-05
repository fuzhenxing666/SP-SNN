import torch
import torch.nn as nn
import numpy as np
from scipy.linalg import lstsq
import matplotlib.pyplot as plt

# Define device (use GPU if available, otherwise CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

torch.manual_seed(42)
torch.set_default_dtype(torch.float64)

# Define the neural network
class Net(nn.Module):
    def __init__(self, hidden_size=100, subspace_dim=100):
        super(Net, self).__init__()
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
        
        # Move model to device
        self.to(device)

    def forward(self, xt):
        x = self.hidden(xt)
        x = self.output(x)
        return x

    def get_hidden_layer_output(self, xt):
        return self.hidden(xt)

# Exact solution
def exact_solution(xt, epsilon):
    x = xt[:, 0:1]  
    t = xt[:, 1:2]  
    v = (torch.exp(torch.tensor(-1.0)/epsilon) + 
         x*(1 - torch.exp(torch.tensor(-1.0)/epsilon)) - 
         torch.exp(-(1.0 - x)/epsilon))
    return torch.exp(-t) * v

# PDE source term f(x, t)
def f(xt, epsilon):
    """
    计算 PDE 的源项 f(x,t) = u_t - epsilon * u_xx + (1-x^2)*u_x
    """
    x = xt[:, 0:1]
    t = xt[:, 1:2]

    u = exact_solution(xt, epsilon) 
    u_t = -u

    u_x = torch.exp(-t) * (1 - torch.exp(torch.tensor(-1.0)/epsilon) - (1.0/epsilon)*torch.exp(-(1.0-x)/epsilon))

    u_xx = torch.exp(-t) * (1.0/(epsilon**2)) * torch.exp(-(1.0-x)/epsilon)

    f_val = u_t - epsilon * u_xx + (1.0 - x**2) * u_x
    
    return f_val

def train_network(net, epsilon, f, num_epochs=500, early_stop_threshold=1e-3, T=1.0):
    # 定义优化器
    optimizer = torch.optim.Adam(net.hidden.parameters(), lr=0.001)
    
    # 定义采样点数量
    N_i, N_b, N_0 = 50, 30, 50
    
    # 内部点
    x_i = torch.linspace(0, 1, N_i, dtype=torch.float64, device=device)[1:-1] 
    t_i = torch.linspace(0, T, N_i, dtype=torch.float64, device=device)[1:-1]
    X, T = torch.meshgrid(x_i.squeeze(), t_i.squeeze(), indexing='ij')
    xt_i_base = torch.stack([X.flatten(), T.flatten()], dim=1)

    # 初始点
    x_0 = torch.linspace(0, 1, N_0, dtype=torch.float64, device=device).reshape(-1, 1)
    t_0 = torch.zeros_like(x_0)
    xt_0_base = torch.cat([x_0, t_0], dim=1)

    # 边界点
    t_b = torch.linspace(0, 1, N_b, dtype=torch.float64, device=device).reshape(-1, 1)
    x_b0, x_b1 = torch.zeros_like(t_b), torch.ones_like(t_b)
    xt_b0_base = torch.cat([x_b0, t_b], dim=1)
    xt_b1_base = torch.cat([x_b1, t_b], dim=1)
    
    # 训练循环
    for epoch in range(num_epochs):
        # 在每次迭代中创建新的张量并启用梯度计算，确保计算图独立
        xt_i = xt_i_base.clone().requires_grad_(True)  # 内部点
        xt_0 = xt_0_base.clone().requires_grad_(True)  # 初始条件点
        xt_b0 = xt_b0_base.clone().requires_grad_(True)  # 边界 x=0
        xt_b1 = xt_b1_base.clone().requires_grad_(True)  # 边界 x=1
        
        # 前向传播计算网络输出
        u_i = net(xt_i)
        
        # 计算偏导数
        u_t = torch.autograd.grad(u_i, xt_i, grad_outputs=torch.ones_like(u_i), create_graph=True)[0][:, 1:2]  # 时间偏导
        u_x = torch.autograd.grad(u_i, xt_i, grad_outputs=torch.ones_like(u_i), create_graph=True)[0][:, 0:1]  # 空间一阶偏导
        u_xx = torch.autograd.grad(u_x, xt_i, grad_outputs=torch.ones_like(u_x), create_graph=True)[0][:, 0:1]  # 空间二阶偏导
        
        # 计算 PDE 残差
        term = 1.0 - torch.pow(xt_i[:, 0:1], 2)
        pde_res = u_t - epsilon * u_xx + term * u_x - f(xt_i, epsilon)
        
        # 计算初始条件和边界条件残差，分离 exact_solution 的计算图
        ic_res = net(xt_0) - exact_solution(xt_0, epsilon).detach()  # 初始条件残差
        bc_res0 = net(xt_b0) - exact_solution(xt_b0, epsilon).detach()  # 边界 x=0 残差
        bc_res1 = net(xt_b1) - exact_solution(xt_b1, epsilon).detach()  # 边界 x=1 残差
        bc_res = torch.cat([bc_res0, bc_res1], dim=0)  # 拼接边界条件残差
        
        # 计算总损失
        loss = torch.mean(pde_res**2)
        # loss = torch.mean(pde_res**2) + torch.mean(ic_res**2) + torch.mean(bc_res0**2) + torch.mean(bc_res1**2)

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
    return xt_i_base, xt_0_base, xt_b0_base, xt_b1_base

# Explicit basis functions
def explicit_bases(xt, epsilon, num_explicit_bases):
    x = xt[:, 0:1]
    t = xt[:, 1:2]
    if num_explicit_bases == 0:
        return torch.empty(xt.shape[0], 0, dtype=torch.float64, device=device)
    bases = [torch.exp(-x / epsilon)*torch.exp(-t), torch.exp(-(1.0 - x) / epsilon)*torch.exp(-t)]
    return torch.cat(bases[:num_explicit_bases], dim=1)

def compute_pde_operator(net, xt, epsilon, explicit_bases_func, num_explicit_bases):
    # 确保 xt 可以求导
    xt = xt.requires_grad_(True)
    
    # 获取网络输出和显式基函数
    phi = net.get_hidden_layer_output(xt)  # [num_points, subspace_dim]
    psi = explicit_bases_func(xt, epsilon, num_explicit_bases)  # [num_points, num_explicit_bases]
    
    # 拼接基函数
    basis = torch.cat([phi, psi], dim=1) if num_explicit_bases > 0 else phi
    
    L_basis_list = []
    for j in range(basis.shape[1]):
        basis_j = basis[:, j].unsqueeze(1)  # [num_points, 1]
        
        # 计算一阶梯度
        grad_basis_j = torch.autograd.grad(basis_j, xt, grad_outputs=torch.ones_like(basis_j), create_graph=True)[0]  # [num_points, 2]
        
        # 提取对 t 和 x 的一阶偏导
        basis_j_t = grad_basis_j[:, 1:2]  # 对 t 的偏导数
        basis_j_x = grad_basis_j[:, 0:1]  # 对 x 的偏导数
        
        # 计算对 x 的二阶偏导
        basis_j_xx = torch.autograd.grad(basis_j_x, xt, grad_outputs=torch.ones_like(basis_j_x), create_graph=True)[0][:, 0:1]  # [num_points, 1]
        
        # 计算 PDE 算子
        term = 1.0 - torch.pow(xt[:, 0:1], 2)
        L_basis_j = basis_j_t - epsilon * basis_j_xx + term * basis_j_x
        L_basis_list.append(L_basis_j)
    
    # 拼接所有基函数的 PDE 算子结果
    L_basis = torch.cat(L_basis_list, dim=1)  # [num_points, subspace_dim + num_explicit_bases]
    return L_basis

# Assemble the matrix
def assemble_matrix(net, explicit_bases_func, xt_interior, xt_initial, xt_b0, xt_b1, f, epsilon, num_explicit_bases):
    # PDE部分
    L_basis = compute_pde_operator(net, xt_interior, epsilon, explicit_bases_func, num_explicit_bases)
    A_i = L_basis.cpu().detach().numpy()
    f_i = f(xt_interior, epsilon).cpu().detach().numpy().flatten()

    # 合并所有边界点（x=0, x=1）
    xt_b_all = torch.cat([xt_b0, xt_b1], dim=0)

    # 统一计算边界点的基函数
    phi_b_all = net.get_hidden_layer_output(xt_b_all)
    psi_b_all = explicit_bases_func(xt_b_all, epsilon, num_explicit_bases)
    basis_b_all = torch.cat([phi_b_all, psi_b_all], dim=1) if num_explicit_bases > 0 else phi_b_all

    # 统一计算边界点的精确解
    u_b_all = exact_solution(xt_b_all, epsilon).cpu().detach().numpy().flatten()

    # 初始条件部分
    phi_0 = net.get_hidden_layer_output(xt_initial)
    psi_0 = explicit_bases_func(xt_initial, epsilon, num_explicit_bases)
    basis_0 = torch.cat([phi_0, psi_0], dim=1) if num_explicit_bases > 0 else phi_0
    A_0 = basis_0.cpu().detach().numpy()
    u0 = exact_solution(xt_initial, epsilon).cpu().detach().numpy().flatten()

    
    A = np.vstack([A_i, A_0, basis_b_all.detach().numpy()])
    f_vec = np.hstack([f_i, u0, u_b_all])
    return A, f_vec

# Solve the linear system
def solve_linear_system(A, f):
    w, _, _, _ = lstsq(A, f)
    return w

# Compute the approximate solution
def compute_solution(net, explicit_bases_func, w, xt, epsilon, num_explicit_bases):
    xt_tensor = torch.tensor(xt, dtype=torch.float64, device=device)
    phi = net.get_hidden_layer_output(xt_tensor).cpu().detach().numpy()
    psi = explicit_bases_func(xt_tensor, epsilon, num_explicit_bases).cpu().detach().numpy() if num_explicit_bases > 0 else np.array([])
    basis = np.hstack([phi, psi]) if num_explicit_bases > 0 else phi
    return basis @ w

def evaluate_and_plot(net, w, epsilon, num_explicit_bases, device, explicit_bases_func=explicit_bases, n_points=100, T=1.0):
    # 生成评估点
    x_eval = np.linspace(0, 1, n_points)
    t_eval = np.linspace(0, T, n_points)
    X, T = np.meshgrid(x_eval, t_eval)
    xt_eval = np.column_stack((X.ravel(), T.ravel()))

    # 计算近似解和精确解
    u_approx = compute_solution(net, explicit_bases, w, xt_eval, epsilon, num_explicit_bases)
    xt_tensor = torch.tensor(xt_eval, dtype=torch.float64, device=device)
    u_exact = exact_solution(xt_tensor, epsilon).detach().cpu().numpy()
    error = np.abs(u_approx.reshape(-1, 1) - u_exact)

    # 正确重塑数组
    u_approx_2d = u_approx.reshape(n_points, n_points)
    u_exact_2d = u_exact.reshape(n_points, n_points)
    error_2d = error.reshape(n_points, n_points)

    # 计算误差
    max_error = np.max(error_2d)
    l2_error = np.sqrt(np.mean(error_2d**2))
    print(f"二范数误差: {l2_error:.2e}")
    print(f"最大误差: {max_error:.2e}")

    # 绘制结果
    fig1, axs1 = plt.subplots(1, 3, figsize=(15, 5))
    
    im0 = axs1[0].contourf(X, T, u_exact_2d, 100, cmap='jet')
    plt.colorbar(im0, ax=axs1[0])
    axs1[0].set_title("Exact Solution")
    axs1[0].set_xlabel("x")
    axs1[0].set_ylabel("t")
    
    im1 = axs1[1].contourf(X, T, u_approx_2d, 100, cmap='jet')
    plt.colorbar(im1, ax=axs1[1])
    axs1[1].set_title("Predicted Solution")
    axs1[1].set_xlabel("x")
    axs1[1].set_ylabel("t")
    
    im2 = axs1[2].contourf(X, T, error_2d, 100, cmap='jet')
    plt.colorbar(im2, ax=axs1[2])
    axs1[2].set_title("Pointwise Error")
    axs1[2].set_xlabel("x")
    axs1[2].set_ylabel("t")
    
    plt.tight_layout()
    plt.show()

    # 特定时间点的比较
    fig2, axs2 = plt.subplots(1, 3, figsize=(15, 5))
    for i, t in enumerate([0.0, 0.2, 0.4]):
        idx = np.argmin(np.abs(t_eval - t))
        u_approx_t = u_approx_2d[idx, :]
        u_exact_t = u_exact_2d[idx, :]
        axs2[i].plot(x_eval, u_exact_t, 'b-', label="Exact")
        axs2[i].plot(x_eval, u_approx_t, 'r--', label="Predicted")
        axs2[i].set_title(f"t = {t}")
        axs2[i].set_xlabel("x")
        axs2[i].set_ylabel("u")
        axs2[i].legend()
    
    plt.tight_layout()
    plt.show()

    return{
        "l2_error": l2_error,
        "max_error": max_error,
        "u_approx": u_approx_2d,
        "u_exact": u_exact_2d,
        "error": error_2d
    }


# Main execution
if __name__ == "__main__":
    epsilon = 1e-5
    T = 1.0
    num_explicit_bases = 2
    net = Net(hidden_size=100, subspace_dim=100)
    
    # Convert model parameters to double precision
    for param in net.parameters():
        param.data = param.data.to(dtype=torch.float64)
    
    xt_interior, xt_initial, xt_b0, xt_b1 = train_network(net, epsilon, f, num_epochs=5000, T=T)
    A, f_vec = assemble_matrix(net, explicit_bases, xt_interior, xt_initial, xt_b0, xt_b1, f, epsilon, num_explicit_bases)
    w = solve_linear_system(A, f_vec)
    
    # 调用评估和绘图函数
    results = evaluate_and_plot(net, w, epsilon, num_explicit_bases, device, explicit_bases_func=explicit_bases, n_points=100, T=T)
    
