import torch
import torch.nn as nn
import numpy as np
from scipy.linalg import lstsq
import matplotlib.pyplot as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
torch.set_default_tensor_type(torch.DoubleTensor)
torch.manual_seed(42) 

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

    def forward(self, r_theta):
        phi = self.hidden(r_theta)
        w = self.output(phi)
        return w
    
    def get_hidden_layer_output(self, r_theta):
        return self.hidden(r_theta)
    
# 极坐标下的双调和算子
def biharmonic(w, r_theta):
    r = r_theta[:, 0:1]
    theta = r_theta[:, 1:2]

    # 计算一阶导数
    w_r = torch.autograd.grad(w, r, grad_outputs=torch.ones_like(w), create_graph=True)[0]
    w_theta = torch.autograd.grad(w, theta, grad_outputs=torch.ones_like(w), create_graph=True)[0]
    w_rr = torch.autograd.grad(w_r, r, grad_outputs=torch.ones_like(w_r), create_graph=True)[0]
    w_tt = torch.autograd.grad(w_theta, theta, grad_outputs=torch.ones_like(w_theta), create_graph=True)[0]

    # 拉普拉斯算子 ΔW
    laplacian = w_rr + (1/r) * w_r + (1/(r**2)) * w_tt

    # 计算ΔΔW
    laplacian_r = torch.autograd.grad(laplacian, r, grad_outputs=torch.ones_like(laplacian), create_graph=True)[0]
    laplacian_theta = torch.autograd.grad(laplacian, theta, grad_outputs=torch.ones_like(laplacian), create_graph=True)[0]
    laplacian_rr = torch.autograd.grad(laplacian_r, r, grad_outputs=torch.ones_like(laplacian_r), create_graph=True)[0]
    laplacian_tt = torch.autograd.grad(laplacian_theta, theta, grad_outputs=torch.ones_like(laplacian_theta), create_graph=True)[0]

    biharmonic_W = laplacian_rr + (1/r) * laplacian_r + (1/(r**2)) * laplacian_tt

    return biharmonic_W, laplacian

# 渐近解
def asymptotic_solution(r_theta):
    r = r_theta[:, 0:1]
    theta = r_theta[:, 1:2]

    term1 = (b**2 * alpha / (1 - b**2)) * (1/r - r)
    term2 = epsilon * (2 * alpha * b / (1 - b**2)**2) * ((1 + b**3)/r - (1 + b)*r)
    term3 = epsilon * (-2 * alpha / (1 - b**2)) * (torch.sqrt(b) / torch.sqrt(r)) * torch.exp(-(r - b)/epsilon)
    term4 = epsilon * (2 * b**2 * alpha / (1 - b**2)) * (1 / torch.sqrt(r)) * torch.exp(-(1 - r)/epsilon)

    W = (term1 + term2 + term3 + term4) * torch.cos(theta) 

    return W


def train_network(net, num_epochs=5000, lr=0.001, early_stop_threshold=1e-3):
    # 定义优化器
    optimizer = torch.optim.Adam(net.hidden.parameters(), lr=lr)

    # 定义采样点数量
    N_i, N_b = 50, 50

    # 训练点: (r, θ) in [b, 1] x [0, 2π]
    r_i = torch.linspace(b, 1, N_i, dtype=torch.float64, device=device)[1:-1]
    theta_i = torch.linspace(0, 2 * np.pi, N_i, dtype=torch.float64, device=device)[1:-1]
    R, Theta = torch.meshgrid(r_i, theta_i, indexing='ij')
    r_theta_base = torch.stack([R.flatten(), Theta.flatten()], dim=1)

    # 边界点: (r, θ) in {b/1} x [0, 2π]
    theta_b = torch.linspace(0, 2 * np.pi, N_b, dtype=torch.float64, device=device).reshape(-1, 1)
    r_b0, r_b1 = b * torch.ones_like(theta_b), torch.ones_like(theta_b)
    r_theta_b0 = torch.cat([r_b0, theta_b], dim=1)
    r_theta_b1 = torch.cat([r_b1, theta_b], dim=1)

    for epoch in range(num_epochs):
        rt_i = r_theta_base.clone().requires_grad_(True)
        rt_b0 = r_theta_b0.clone().requires_grad_(True)
        rt_b1 = r_theta_b1.clone().requires_grad_(True)

        w_i = net(rt_i)
        biharmonic_W_i, laplacian_i = biharmonic(w_i, rt_i)

        # 计算 PDE 残差
        pde_res = epsilon**2 * biharmonic_W_i - laplacian_i

        # 计算边界条件残差
        w_b0 = net(rt_b0)
        w_r_b0 = torch.autograd.grad(w_b0, rt_b0, grad_outputs=torch.ones_like(w_b0), create_graph=True)[0][:, 0:1]
        bc_res00 = net(rt_b0) - (b * alpha * torch.cos(rt_b0[:, 1:2]))
        bc_res01 = w_r_b0 - (alpha * torch.cos(rt_b0[:, 1:2]))
        bc_res0 = bc_res00 + bc_res01

        w_b1 = net(rt_b1)
        w_r_b1 = torch.autograd.grad(w_b1, rt_b1, grad_outputs=torch.ones_like(w_b1), create_graph=True)[0][:, 0:1]
        bc_res1 = w_b1 + w_r_b1
        bc_res = torch.cat([bc_res0, bc_res1], dim=0)

        # 计算总损失
        loss = torch.mean(pde_res**2) 

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
    return r_theta_base, r_theta_b0, r_theta_b1

def explicit_bases_polar(r_theta, num_explicit_bases):
    r = r_theta[:, 0:1]
    theta = r_theta[:, 1:2]

    if num_explicit_bases == 0:
        return torch.empty(r_theta.shape[0], 0, dtype=torch.float64, device=device)
    bases = [torch.exp(-(r-b)/epsilon) * torch.cos(theta), torch.exp(-(1-r)/epsilon) * torch.cos(theta)]
    return torch.cat(bases[:num_explicit_bases], dim=1)

def compute_pde_operator(net, r_theta, explicit_bases_func, num_explicit_bases):
    r_theta.requires_grad_(True)

    phi = net.get_hidden_layer_output(r_theta)
    psi = explicit_bases_func(r_theta, num_explicit_bases)

    # 拼接基函数
    if num_explicit_bases > 0:
        basis = torch.cat([phi, psi], dim=1)  # [num_points, subspace_dim + num_explicit_bases]
    else:
        basis = phi  # [num_points, subspace_dim]

    L_basis_list = []
    for j in range(basis.shape[1]):
        basis_j = basis[:, j].unsqueeze(1)  # [num_points, 1]
        biharmonic_j, laplace_j = biharmonic(basis_j, r_theta)

        # 计算 PDE 算子: ε²ΔΔ - Δ
        L_basis_j = epsilon**2 * biharmonic_j - laplace_j
        L_basis_list.append(L_basis_j)

    L_basis = torch.cat(L_basis_list, dim=1)  # [num_points, subspace_dim + num_explicit_bases]
    return L_basis

def compute_boundary_operator(net, r_theta_b, explicit_bases_func, num_explicit_bases):
    # 边界点（r=b）
    phi_b = net.get_hidden_layer_output(r_theta_b)
    psi_b = explicit_bases_func(r_theta_b, num_explicit_bases)
    basis_b = torch.cat([phi_b, psi_b], dim=1) if num_explicit_bases > 0 else phi_b

    B_basis_list = []
    for j in range(basis_b.shape[1]):
        basis_j = basis_b[:, j].unsqueeze(1)  # [num_points, 1]
        basis_j_r = torch.autograd.grad(basis_j, r_theta_b, grad_outputs=torch.ones_like(basis_j), create_graph=True)[0][:, 0:1]  # 只取 r 的梯度
        B_basis_list.append(basis_j_r)

    B_basis = torch.cat(B_basis_list, dim=1)  # [num_points, subspace_dim + num_explicit_bases]
    return basis_b, B_basis


def assemble_matrix(net, explicit_bases_func, r_theta_base, r_theta_b0, r_theta_b1, num_explicit_bases):
    # PDE部分
    L_basis = compute_pde_operator(net, r_theta_base, explicit_bases_func, num_explicit_bases)
    A_i = L_basis.cpu().detach().numpy()
    f_i = torch.zeros_like(L_basis[:, 0:1]).cpu().detach().numpy().flatten()

    # 边界点（r=b）
    basis_b0, B_basis_b0 = compute_boundary_operator(net, r_theta_b0, explicit_bases_func, num_explicit_bases)
    basis_b0, B_basis_b0 = basis_b0.cpu().detach().numpy(), B_basis_b0.cpu().detach().numpy()
    # 边界条件 u(b, θ) = b * α * cos(θ)
    u_b0 = b * alpha * torch.cos(r_theta_b0[:, 1:2]).cpu().detach().numpy().flatten()
    # 边界条件 u_r(b, θ) = 0
    u_B0 = torch.zeros_like(r_theta_b0[:, 0:1]).cpu().detach().numpy().flatten()


    # 边界点（r=1）
    basis_b1, B_basis_b1 = compute_boundary_operator(net, r_theta_b1, explicit_bases_func, num_explicit_bases)
    basis_b1, B_basis_b1 = basis_b1.cpu().detach().numpy(), B_basis_b1.cpu().detach().numpy()
    u_b1 = torch.zeros_like(r_theta_b1[:, 0:1]).cpu().detach().numpy().flatten()
    u_B1 = torch.zeros_like(r_theta_b1[:, 0:1]).cpu().detach().numpy().flatten()

    # 组装矩阵
    A = np.vstack([A_i, basis_b0, basis_b1, B_basis_b0, B_basis_b1])
    f = np.hstack([f_i, u_b0, u_b1, u_B0, u_B1])
    return A, f

def solve_linear_system(A, f):
    w, _, _, _ = lstsq(A, f)
    return w

def compute_solution(net, explicit_bases_func, w, rt,  num_explicit_bases):
    r_theta_tensor = torch.tensor(rt, dtype=torch.float64, device=device)
    phi = net.get_hidden_layer_output(r_theta_tensor).cpu().detach().numpy()
    psi = explicit_bases_polar(r_theta_tensor, num_explicit_bases).cpu().detach().numpy()
    basis = np.hstack([phi, psi]) if num_explicit_bases > 0 else phi
    return basis @ w

def evaluate_and_plot(net, w, explicit_bases_func, num_explicit_bases, n_points=100, plot=True):
    # 生成评估点
    r_eval = torch.linspace(b, 1, n_points, dtype=torch.float64, device=device)
    theta_eval = torch.linspace(0, 2 * np.pi, n_points, dtype=torch.float64, device=device)
    R_eval, Theta_eval = torch.meshgrid(r_eval, theta_eval, indexing='ij')
    r_theta_eval = torch.stack([R_eval.flatten(), Theta_eval.flatten()], dim=1)

    # 计算近似解
    w_approx = compute_solution(net, explicit_bases_func, w, r_theta_eval.cpu().numpy(), num_explicit_bases)
    w_exact = asymptotic_solution(r_theta_eval).cpu().detach().numpy().flatten()

    approx_2d = w_approx.reshape(n_points, n_points).detach().numpy()
    exact_2d = w_exact.reshape(n_points, n_points).detach().numpy()

    # 计算误差
    error = np.abs(approx_2d - exact_2d)
    max_error = np.max(error)
    l2_error = np.sqrt(np.mean(error**2))

    print(f"最大误差: {max_error:.6f}, L2误差: {l2_error:.6f}")

    if plot:
        # 绘制近似解和精确解
        fig, axs = plt.subplots(1, 3, figsize=(15, 5))

        im0 = axs[0].imshow(approx_2d, extent=(0, 2 * np.pi, b, 1), origin='lower', aspect='auto')
        axs[0].set_title('Predicted Solution')
        axs[0].set_xlabel('Theta')
        axs[0].set_ylabel('r')
        fig.colorbar(im0, ax=axs[0])

        im1 = axs[1].imshow(exact_2d, extent=(0, 2 * np.pi, b, 1), origin='lower', aspect='auto')
        axs[1].set_title('Exact Solution')
        axs[1].set_xlabel('Theta')
        axs[1].set_ylabel('r')
        fig.colorbar(im1, ax=axs[1])

        im2 = axs[2].imshow(error, extent=(0, 2 * np.pi, b, 1), origin='lower', aspect='auto')
        axs[2].set_title('Error')
        axs[2].set_xlabel('Theta')
        axs[2].set_ylabel('r')
        fig.colorbar(im2, ax=axs[2])

        plt.tight_layout()
        plt.show()

    return {
        'asymptotic': approx_2d,
        'exact': exact_2d,
        'error': error,
        'max_error': max_error,
        'l2_error': l2_error
    }

if __name__ == "__main__":
    # 参数设置
    b = 0.2
    alpha = 0.5
    epsilon = 0.005
    num_explicit_bases = 2

    # 初始化网络
    net = Net(hidden_size=100, subspace_dim=100).to(device)

    # 训练网络
    r_theta_base, r_theta_b0, r_theta_b1 = train_network(net, num_epochs=5000, lr=0.001, early_stop_threshold=1e-3)

    # 组装矩阵
    A, f = assemble_matrix(net, explicit_bases_polar, r_theta_base, r_theta_b0, r_theta_b1, num_explicit_bases)

    # 求解线性系统
    w = solve_linear_system(A, f)

    # 评估和绘图
    results = evaluate_and_plot(net, w, explicit_bases_polar, num_explicit_bases)