import torch
import torch.nn as nn
import numpy as np
from scipy.linalg import lstsq
import matplotlib.pyplot as plt
import math

# 检查GPU可用性
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
torch.set_default_tensor_type(torch.DoubleTensor)
torch.manual_seed(42) 

# 参数设置
b = 0.2
alpha = 0.5
epsilon = 0.005
print(f"Using parameters: b={b}, alpha={alpha}, epsilon={epsilon}")

class Net(nn.Module):
    def __init__(self, hidden_size=100, subspace_dim=100):
        super(Net, self).__init__()
        self.hidden = nn.Sequential(
            nn.Linear(2, hidden_size),  # 输入为(r, θ)
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, subspace_dim),
            nn.Tanh()
        )
        self.output = nn.Linear(subspace_dim, 1, bias=False)  # 输出W
        with torch.no_grad():
            self.output.weight.fill_(1.0)
        self.output.weight.requires_grad = True

    def forward(self, xy):
        return self.output(self.hidden(xy))
    
    def get_hidden_layer_output(self, xy):
        return self.hidden(xy)

# 极坐标下的双调和算子
def biharmonic(u, r, theta):
    # 一阶导数
    u_r = torch.autograd.grad(u, r, grad_outputs=torch.ones_like(u), 
                              create_graph=True, retain_graph=True, allow_unused=True)[0]
    u_theta = torch.autograd.grad(u, theta, grad_outputs=torch.ones_like(u), 
                                 create_graph=True, retain_graph=True, allow_unused=True)[0]
    
    # 处理可能为None的情况
    if u_r is None:
        u_r = torch.zeros_like(u)
    if u_theta is None:
        u_theta = torch.zeros_like(u)
    
    # 二阶导数
    u_rr = torch.autograd.grad(u_r, r, grad_outputs=torch.ones_like(u_r), 
                              create_graph=True, retain_graph=True, allow_unused=True)[0]
    u_tt = torch.autograd.grad(u_theta, theta, grad_outputs=torch.ones_like(u_theta), 
                              create_graph=True, retain_graph=True, allow_unused=True)[0]
    
    # 处理可能为None的情况
    if u_rr is None:
        u_rr = torch.zeros_like(u)
    if u_tt is None:
        u_tt = torch.zeros_like(u)
    
    # 混合导数
    u_rt = torch.autograd.grad(u_r, theta, grad_outputs=torch.ones_like(u_r), 
                              create_graph=True, retain_graph=True, allow_unused=True)[0]
    
    # 处理可能为None的情况
    if u_rt is None:
        u_rt = torch.zeros_like(u)
    
    # 拉普拉斯算子
    laplace_u = u_rr + (1/r)*u_r + (1/r**2)*u_tt
    
    # 拉普拉斯算子的导数（用于双调和）
    laplace_u_r = torch.autograd.grad(laplace_u, r, grad_outputs=torch.ones_like(laplace_u), 
                                     create_graph=True, retain_graph=True, allow_unused=True)[0]
    laplace_u_t = torch.autograd.grad(laplace_u, theta, grad_outputs=torch.ones_like(laplace_u), 
                                     create_graph=True, retain_graph=True, allow_unused=True)[0]
    
    # 处理可能为None的情况
    if laplace_u_r is None:
        laplace_u_r = torch.zeros_like(laplace_u)
    if laplace_u_t is None:
        laplace_u_t = torch.zeros_like(laplace_u)
    
    # 双调和算子
    biharmonic_term1 = torch.autograd.grad(laplace_u_r, r, grad_outputs=torch.ones_like(laplace_u_r), 
                                          create_graph=True, retain_graph=True, allow_unused=True)[0]
    if biharmonic_term1 is None:
        biharmonic_term1 = torch.zeros_like(laplace_u)
    
    biharmonic_term2 = (1/r)*laplace_u_r
    
    biharmonic_term3_grad = torch.autograd.grad(laplace_u_t, theta, grad_outputs=torch.ones_like(laplace_u_t), 
                                              create_graph=True, retain_graph=True, allow_unused=True)[0]
    if biharmonic_term3_grad is None:
        biharmonic_term3_grad = torch.zeros_like(laplace_u)
    
    biharmonic_term3 = (1/r**2)*biharmonic_term3_grad
    
    biharmonic_u = biharmonic_term1 + biharmonic_term2 + biharmonic_term3
    
    return biharmonic_u, laplace_u

# 边界条件处理
def apply_boundary_conditions(r, theta, W, b_val, alpha_val):
    # 内边界 r = b
    mask_b = (r == b_val).squeeze()
    
    # 检查是否存在内边界点
    if not torch.any(mask_b):
        print("警告: 没有找到内边界点 r =", b_val)
        bc_b1 = torch.tensor(0.0, device=r.device, requires_grad=True)
        bc_b2 = torch.tensor(0.0, device=r.device, requires_grad=True)
    else:
        W_b = W[mask_b]
        theta_b = theta[mask_b]
        bc_b1 = W_b - b_val * alpha_val * torch.cos(theta_b)
        W_r_b = torch.autograd.grad(W_b, r[mask_b], grad_outputs=torch.ones_like(W_b), 
                                  create_graph=True, allow_unused=True, retain_graph=True)[0]
        bc_b2 = W_r_b - alpha_val * torch.cos(theta_b) if W_r_b is not None else torch.zeros_like(W_b)
    
    # 外边界 r = 1
    mask_1 = (r == 1.0).squeeze()
    
    # 检查是否存在外边界点
    if not torch.any(mask_1):
        print("警告: 没有找到外边界点 r = 1.0")
        bc_1 = torch.tensor(0.0, device=r.device, requires_grad=True)
        bc_2 = torch.tensor(0.0, device=r.device, requires_grad=True)
    else:
        W_1 = W[mask_1]
        theta_1 = theta[mask_1]
        bc_1 = W_1
        W_r_1 = torch.autograd.grad(W_1, r[mask_1], grad_outputs=torch.ones_like(W_1), 
                                  create_graph=True, allow_unused=True, retain_graph=True)[0]
        bc_2 = W_r_1 if W_r_1 is not None else torch.zeros_like(W_1)
    
    return bc_b1, bc_b2, bc_1, bc_2

# 渐近解
def asymptotic_solution(r, theta, epsilon, b, alpha):
    term1 = (b**2 * alpha / (1 - b**2)) * (1/r - r)
    term2 = epsilon * (2 * alpha * b / (1 - b**2)**2) * ((1 + b**3)/r - (1 + b)*r)
    term3 = epsilon * (-2 * alpha / (1 - b**2)) * (math.sqrt(b) / math.sqrt(r)) * torch.exp(-(r - b)/epsilon)
    term4 = epsilon * (2 * b**2 * alpha / (1 - b**2)) * (1 / math.sqrt(r)) * torch.exp(-(1 - r)/epsilon)
    
    W = (term1 + term2 + term3 + term4) * torch.cos(theta)
    return W

def train_network(net, epsilon, b, alpha, device, num_epochs=5000, n_points=1000):
    # 定义优化器
    optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
    
    # 生成训练点: (r, θ) in [b, 1] x [0, 2π]
    r_vals = torch.linspace(b, 1.0, int(np.sqrt(n_points)), device=device)
    theta_vals = torch.linspace(0, 2*np.pi, int(np.sqrt(n_points)), device=device)
    R, Theta = torch.meshgrid(r_vals, theta_vals, indexing='ij')
    r = R.reshape(-1, 1).requires_grad_(True)
    theta = Theta.reshape(-1, 1).requires_grad_(True)
    
    # 绘制训练点分布（可选）
    plt.figure(figsize=(10, 5))
    plt.subplot(121)
    plt.scatter(r.detach().cpu().numpy(), theta.detach().cpu().numpy(), s=1)
    plt.xlabel('r')
    plt.ylabel('θ')
    plt.title('Training Points in Polar Coordinates')
    
    # 转换为直角坐标进行绘图
    x = r * torch.cos(theta)
    y = r * torch.sin(theta)
    plt.subplot(122)
    plt.scatter(x.detach().cpu().numpy(), y.detach().cpu().numpy(), s=1)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Training Points in Cartesian Coordinates')
    plt.tight_layout()
    plt.show()
    
    print(f"Starting training with {len(r)} points...")
    
    # 训练循环
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        
        # 前向传播计算网络输出
        inputs = torch.cat([r, theta], dim=1)
        W = net(inputs)
        
        # 计算双调和算子
        biharmonic_W, laplace_W = biharmonic(W, r, theta)
        
        # 计算PDE残差: ε²ΔΔW - ΔW = 0
        pde_res = epsilon**2 * biharmonic_W - laplace_W
        
        # 应用边界条件
        bc_b1, bc_b2, bc_1, bc_2 = apply_boundary_conditions(r, theta, W, b, alpha)
        
        # 计算总损失
        loss_pde = torch.mean(pde_res**2)
        loss_bc = torch.mean(bc_b1**2) + torch.mean(bc_b2**2) + torch.mean(bc_1**2) + torch.mean(bc_2**2)
        loss = loss_pde
        
        if epoch == 0:
            initial_loss = loss.item()
            print(f"Initial loss: {initial_loss:.4e}")
        
        # 反向传播
        loss.backward()
        optimizer.step()
        
        # 打印训练进度
        if epoch % 100 == 0:
            print(f"Epoch {epoch}/{num_epochs}, Loss: {loss.item():.4e}, "
                  f"PDE: {loss_pde.item():.4e}, BC: {loss_bc.item():.4e}")
        
        # 提前停止
        if loss.item() < 1e-3 * initial_loss:
            print(f"Early stopping at epoch {epoch}, Loss: {loss.item():.4e}")
            break
    
    return r, theta

# 显式基函数（极坐标下）
def explicit_bases_polar(r_theta, epsilon, b, alpha, num_explicit_bases=4):
    r = r_theta[:, 0:1]
    theta = r_theta[:, 1:2]
    pi = torch.tensor(math.pi)
    
    # 渐近解的主要部分
    base1 = torch.exp(-(r-b)/epsilon) * torch.cos(theta)
    base2 = torch.exp(-(1-r)/epsilon) * torch.cos(theta)
    
    # 边界层修正项
    base3 = epsilon * (-2 * alpha / (1 - b**2)) * (torch.sqrt(torch.tensor(b)) / torch.sqrt(r)) * torch.exp(-(r - b)/epsilon) * torch.cos(theta)
    base4 = epsilon * (2 * b**2 * alpha / (1 - b**2)) * (1 / torch.sqrt(r)) * torch.exp(-(1 - r)/epsilon) * torch.cos(theta)
    
    bases = [base1, base2, base3, base4]
    
    if num_explicit_bases > len(bases):
        print(f"Warning: Requested {num_explicit_bases} bases but only {len(bases)} available.")
        num_explicit_bases = len(bases)
    
    return torch.cat(bases[:num_explicit_bases], dim=1)

# 修改计算PDE算子函数以适应极坐标
def compute_pde_operator(net, r_theta, epsilon, explicit_bases_func, num_explicit_bases, b, alpha):
    # 确保可以求导
    r_theta = r_theta.requires_grad_(True)
    r = r_theta[:, 0:1]
    theta = r_theta[:, 1:2]
    
    # 获取网络输出和显式基函数
    phi = net.get_hidden_layer_output(r_theta)  # [num_points, subspace_dim]
    psi = explicit_bases_func(r_theta, epsilon, b, alpha, num_explicit_bases)  # [num_points, num_explicit_bases]
    
    # 拼接基函数
    if num_explicit_bases > 0:
        basis = torch.cat([phi, psi], dim=1)  # [num_points, subspace_dim + num_explicit_bases]
    else:
        basis = phi  # [num_points, subspace_dim]
    
    L_basis_list = []
    for j in range(basis.shape[1]):
        basis_j = basis[:, j].unsqueeze(1)  # [num_points, 1]
        
        try:
            # 计算双调和算子
            biharmonic_j, laplace_j = biharmonic(basis_j, r, theta)
            
            # 计算 PDE 算子: ε²ΔΔ - Δ
            L_basis_j = epsilon**2 * biharmonic_j - laplace_j
            L_basis_list.append(L_basis_j)
        except Exception as e:
            print(f"处理基函数 {j} 时出错: {e}")
            # 出错时使用零向量
            L_basis_list.append(torch.zeros_like(basis_j))
    
    # 拼接所有基函数的 PDE 算子结果
    L_basis = torch.cat(L_basis_list, dim=1)  # [num_points, subspace_dim + num_explicit_bases]
    return L_basis

def assemble_matrix(net, explicit_bases_func, r_theta_interior, epsilon, 
                    b, alpha, num_explicit_bases):
    # 使用容差来选择边界点
    tol = 1e-6
    
    # PDE部分
    L_basis = compute_pde_operator(net, r_theta_interior, epsilon, 
                                  explicit_bases_func, num_explicit_bases, b, alpha)
    A_i = L_basis.detach().cpu().numpy()
    f_i = np.zeros((r_theta_interior.shape[0], 1))  # 齐次方程，源项为0
    
    # 边界条件处理
    # 内边界 r = b
    mask_b = (torch.abs(r_theta_interior[:, 0] - b) < tol).squeeze()
    if torch.sum(mask_b) > 0:
        r_theta_b = r_theta_interior[mask_b].detach().clone().requires_grad_(True)
    else:
        print("警告: 没有找到内边界点 r =", b)
        # 创建一些人工边界点
        theta_vals = torch.linspace(0, 2*np.pi, 20, device=r_theta_interior.device)
        r_vals = torch.ones_like(theta_vals) * b
        r_theta_b = torch.stack([r_vals, theta_vals], dim=1).requires_grad_(True)
    
    # 外边界 r = 1
    mask_1 = (torch.abs(r_theta_interior[:, 0] - 1.0) < tol).squeeze()
    if torch.sum(mask_1) > 0:
        r_theta_1 = r_theta_interior[mask_1].detach().clone().requires_grad_(True)
    else:
        print("警告: 没有找到外边界点 r = 1.0")
        # 创建一些人工边界点
        theta_vals = torch.linspace(0, 2*np.pi, 20, device=r_theta_interior.device)
        r_vals = torch.ones_like(theta_vals)
        r_theta_1 = torch.stack([r_vals, theta_vals], dim=1).requires_grad_(True)
    
    # 统一计算边界点的基函数
    phi_b = net.get_hidden_layer_output(r_theta_b)
    psi_b = explicit_bases_func(r_theta_b, epsilon, b, alpha, num_explicit_bases)
    basis_b = torch.cat([phi_b, psi_b], dim=1) if num_explicit_bases > 0 else phi_b
    
    phi_1 = net.get_hidden_layer_output(r_theta_1)
    psi_1 = explicit_bases_func(r_theta_1, epsilon, b, alpha, num_explicit_bases)
    basis_1 = torch.cat([phi_1, psi_1], dim=1) if num_explicit_bases > 0 else phi_1
    
    # 组装全局矩阵和向量
    A = np.vstack([A_i, 
                  basis_b.detach().cpu().numpy(), 
                  basis_1.detach().cpu().numpy()])
    
    # 边界条件值
    f_b1 = b * alpha * torch.cos(r_theta_b[:, 1:2])  # W = bα cosθ
    f_1 = torch.zeros_like(r_theta_1[:, 0:1])        # W = 0
    
    # 组装边界条件
    f_vec = np.vstack([f_i, 
                      f_b1.detach().cpu().numpy(),
                      f_1.detach().cpu().numpy()])
    
    return A, f_vec

# 修改评估和绘图函数
def evaluate_and_plot(net, w, epsilon, b, alpha, num_explicit_bases, device, 
                     n_points=100, plot=True):
    # 生成评估点 (极坐标)
    r_eval = torch.linspace(b, 1.0, n_points, device=device)
    theta_eval = torch.linspace(0, 2*np.pi, n_points, device=device)
    R, Theta = torch.meshgrid(r_eval, theta_eval, indexing='ij')
    r_theta_eval = torch.stack([R.flatten(), Theta.flatten()], dim=1)
    
    # 计算近似解
    approx_solution = compute_solution(net, explicit_bases_polar, w, 
                                      r_theta_eval, epsilon, 
                                      num_explicit_bases, device, b, alpha)
    
    # 计算渐近解
    asymptotic = asymptotic_solution(R.flatten()[:, None], 
                                    Theta.flatten()[:, None], 
                                    epsilon, b, alpha)
    
    # 转换为直角坐标进行绘图
    X = R * torch.cos(Theta)
    Y = R * torch.sin(Theta)
    
    # 调整形状
    approx_2d = approx_solution.reshape(n_points, n_points).detach().cpu().numpy()
    asymp_2d = asymptotic.reshape(n_points, n_points).detach().cpu().numpy()
    
    # 计算误差
    error = approx_2d - asymp_2d
    max_error = np.max(np.abs(error))
    l2_error = np.sqrt(np.mean(error**2))
    
    print(f"Relative L2 error: {l2_error:.2e}")
    print(f"Maximum error: {max_error:.2e}")
    
    if plot:
        # 绘图
        fig, axs = plt.subplots(1, 3, figsize=(15, 5))
        
        im0 = axs[0].contourf(X.cpu().numpy(), Y.cpu().numpy(), asymp_2d, 50, cmap='viridis')
        plt.colorbar(im0, ax=axs[0])
        axs[0].set_title("Asymptotic Solution")
        axs[0].set_xlabel("x")
        axs[0].set_ylabel("y")
        
        im1 = axs[1].contourf(X.cpu().numpy(), Y.cpu().numpy(), approx_2d, 50, cmap='viridis')
        plt.colorbar(im1, ax=axs[1])
        axs[1].set_title("Predicted Solution")
        axs[1].set_xlabel("x")
        axs[1].set_ylabel("y")
        
        im2 = axs[2].contourf(X.cpu().numpy(), Y.cpu().numpy(), error, 50, cmap='RdBu_r')
        plt.colorbar(im2, ax=axs[2])
        axs[2].set_title("Error")
        axs[2].set_xlabel("x")
        axs[2].set_ylabel("y")
        
        plt.tight_layout()
        plt.suptitle(f"ε = {epsilon}, b = {b}, α = {alpha}", fontsize=14)
        plt.show()
    
    return {
        'asymptotic': asymp_2d,
        'approx': approx_2d,
        'error': error,
        'max_error': max_error,
        'l2_error': l2_error
    }

def compute_solution(net, explicit_bases_func, w, r_theta, epsilon, 
                     num_explicit_bases, device, b, alpha):
    r_theta_tensor = torch.tensor(r_theta, dtype=torch.float64, device=device)
    phi = net.get_hidden_layer_output(r_theta_tensor)
    psi = explicit_bases_func(r_theta_tensor, epsilon, b, alpha, num_explicit_bases)
    basis = torch.cat([phi, psi], dim=1) if num_explicit_bases > 0 else phi
    w_tensor = torch.tensor(w, dtype=torch.float64, device=device)
    return basis @ w_tensor

# 主函数
if __name__ == "__main__":
    # 参数设置
    b_val = 0.2
    alpha_val = 0.5
    epsilon_val = 0.005
    num_explicit_bases = 4  # 使用渐近解中的4个项作为显式基
    
    # 创建网络
    net = Net(hidden_size=100, subspace_dim=100).to(device)
    
    # 训练网络
    r_train, theta_train = train_network(net, epsilon_val, b_val, alpha_val, device, 
                                       num_epochs=5000, n_points=2000)
    r_theta_train = torch.cat([r_train, theta_train], dim=1)
    
    # 组装矩阵和向量
    A, f_vec = assemble_matrix(net, explicit_bases_polar, r_theta_train, 
                              epsilon_val, b_val, alpha_val, num_explicit_bases)
    
    # 求解线性系统
    w, _, _, _ = lstsq(A, f_vec)
    
    # 评估和绘图
    results = evaluate_and_plot(net, w, epsilon_val, b_val, alpha_val, 
                               num_explicit_bases, device, n_points=100)