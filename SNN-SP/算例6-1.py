import torch
import torch.nn as nn
import numpy as np
from scipy.linalg import lstsq
import matplotlib.pyplot as plt

# 设置设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.set_default_tensor_type(torch.DoubleTensor)
torch.manual_seed(42)

b = 0.2
alpha = 0.5
epsilon = 0.005

class Net(nn.Module):
    def __init__(self, hidden_size=50, subspace_dim=50):
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
        nn.init.constant_(self.output.weight, 1.0)

    def forward(self, r_theta):
        phi = self.hidden(r_theta)
        return self.output(phi)
    
    def get_hidden_layer_output(self, r_theta):
        return self.hidden(r_theta)

# 极坐标下的双调和算子
def biharmonic(u, r_theta):
    r = r_theta[:, 0:1]
    theta = r_theta[:, 1:2]
    
    # 计算一阶导数
    grad_u = torch.autograd.grad(u, r_theta, grad_outputs=torch.ones_like(u),
                                 create_graph=True, retain_graph=True)[0]
    u_r = grad_u[:, 0:1]
    u_theta = grad_u[:, 1:2]
    
    # 计算二阶导数
    grad_u_r = torch.autograd.grad(u_r, r_theta, grad_outputs=torch.ones_like(u_r),
                                   create_graph=True, retain_graph=True)[0]
    u_rr = grad_u_r[:, 0:1]
    u_rtheta = grad_u_r[:, 1:2]
    
    grad_u_theta = torch.autograd.grad(u_theta, r_theta, grad_outputs=torch.ones_like(u_theta),
                                       create_graph=True, retain_graph=True)[0]
    u_thetar = grad_u_theta[:, 0:1]
    u_thetatheta = grad_u_theta[:, 1:2]
    
    # 拉普拉斯算子 ΔW
    laplacian_u = u_rr + (1/r)*u_r + (1/(r**2))*u_thetatheta
    
    # 计算ΔΔW
    grad_laplacian = torch.autograd.grad(laplacian_u, r_theta, 
                                         grad_outputs=torch.ones_like(laplacian_u),
                                         create_graph=True, retain_graph=True)[0]
    laplacian_r = grad_laplacian[:, 0:1]
    laplacian_theta = grad_laplacian[:, 1:2]
    
    grad_laplacian_r = torch.autograd.grad(laplacian_r, r_theta, 
                                          grad_outputs=torch.ones_like(laplacian_r),
                                          create_graph=True)[0]
    laplacian_rr = grad_laplacian_r[:, 0:1]

    grad_laplacian_theta = torch.autograd.grad(laplacian_theta, r_theta,
                                                  grad_outputs=torch.ones_like(laplacian_theta),
                                                    create_graph=True)[0]
    laplacian_thetatheta = grad_laplacian_theta[:, 1:2]

    
    biharmonic_u = laplacian_rr + (1/r)*laplacian_r + (1/(r**2))*laplacian_thetatheta
    
    return biharmonic_u

# 方程(26)的右侧项
def equation_rhs(u, r_theta):
    r = r_theta[:, 0:1]
    theta = r_theta[:, 1:2]
    
    # 计算一阶导数
    grad_u = torch.autograd.grad(u, r_theta, grad_outputs=torch.ones_like(u),
                                 create_graph=True, retain_graph=True)[0]
    u_r = grad_u[:, 0:1]
    u_theta = grad_u[:, 1:2]
    
    # 计算二阶导数
    grad_u_r = torch.autograd.grad(u_r, r_theta, grad_outputs=torch.ones_like(u_r),
                                   create_graph=True, retain_graph=True)[0]
    u_rr = grad_u_r[:, 0:1]
    
    grad_u_theta = torch.autograd.grad(u_theta, r_theta, grad_outputs=torch.ones_like(u_theta),
                                       create_graph=True, retain_graph=True)[0]
    u_thetatheta = grad_u_theta[:, 1:2]
    
    # 方程(25)右侧项的近似计算
    term1 = n_r * u_rr
    term2 = n_theta * (u_r/r + u_thetatheta/(r**2))
    term3 = 2 * n_rt * torch.autograd.grad((1/r)*u_theta, r_theta, 
                                         grad_outputs=torch.ones_like(u_theta),
                                         create_graph=True)[0][:, 0:1]
    
    rhs = (R**2 / D) * (term1 + term2 + term3)
    return rhs

def train_network(net, device, num_epochs=10000, lr=0.001, early_stop=1e-6):
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    
    # 生成训练点
    N_inner = 2000
    r_inner = torch.linspace(r0, R, N_inner//2, device=device).reshape(-1, 1)
    theta_inner = torch.linspace(0, 2*np.pi, 2, device=device).reshape(-1, 1)
    R_inner, Theta_inner = torch.meshgrid(r_inner.squeeze(), theta_inner.squeeze(), indexing='ij')
    r_theta_inner = torch.stack([R_inner.flatten(), Theta_inner.flatten()], dim=1).requires_grad_(True)
    
    # 边界点
    N_boundary = 500
    
    # 外边界 (r = R)
    r_b1 = R * torch.ones(N_boundary, 1, device=device)
    theta_b1 = torch.linspace(0, 2*np.pi, N_boundary, device=device).reshape(-1, 1)
    r_theta_b1 = torch.cat([r_b1, theta_b1], dim=1).requires_grad_(True)
    
    # 内边界 (r = r0)
    r_b2 = r0 * torch.ones(N_boundary, 1, device=device)
    theta_b2 = torch.linspace(0, 2*np.pi, N_boundary, device=device).reshape(-1, 1)
    r_theta_b2 = torch.cat([r_b2, theta_b2], dim=1).requires_grad_(True)
    
    # 周期性边界 (θ=0 和 θ=2π)
    N_periodic = 200
    r_periodic = torch.linspace(r0, R, N_periodic, device=device).reshape(-1, 1)
    theta_p1 = torch.zeros(N_periodic, 1, device=device)
    theta_p2 = 2*np.pi * torch.ones(N_periodic, 1, device=device)
    r_theta_p1 = torch.cat([r_periodic, theta_p1], dim=1).requires_grad_(True)
    r_theta_p2 = torch.cat([r_periodic, theta_p2], dim=1).requires_grad_(True)
    
    best_loss = float('inf')
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        
        # 内部点残差 (方程26)
        u_inner = net(r_theta_inner)
        biharmonic_u = biharmonic(u_inner, r_theta_inner)
        rhs = equation_rhs(u_inner, r_theta_inner)
        pde_loss = torch.mean((biharmonic_u - rhs)**2)
        
        # 外边界条件
        u_b1 = net(r_theta_b1)
        u_b1_r = torch.autograd.grad(u_b1, r_theta_b1, grad_outputs=torch.ones_like(u_b1),
                                    create_graph=True)[0][:, 0:1]
        bc1_loss = torch.mean(u_b1**2) + torch.mean(u_b1_r**2)  # W=0, dW/dr=0
        
        # 内边界条件
        u_b2 = net(r_theta_b2)
        u_b2_r = torch.autograd.grad(u_b2, r_theta_b2, grad_outputs=torch.ones_like(u_b2),
                                    create_graph=True)[0][:, 0:1]
        # W = -α*r₀*cosθ, dW/dr = -α*cosθ
        target_w = -alpha * r0 * torch.cos(theta_b2)
        target_w_r = -alpha * torch.cos(theta_b2)
        bc2_loss = torch.mean((u_b2 - target_w)**2) + torch.mean((u_b2_r - target_w_r)**2)
        
        # 周期性边界条件
        u_p1 = net(r_theta_p1)
        u_p2 = net(r_theta_p2)
        periodic_loss = torch.mean((u_p1 - u_p2)**2)
        
        # 总损失
        loss = pde_loss + bc1_loss + bc2_loss + periodic_loss
        
        if epoch == 0:
            initial_loss = loss.item()
        
        loss.backward()
        optimizer.step()
        
        # 早停检查
        if loss.item() < best_loss:
            best_loss = loss.item()
            torch.save(net.state_dict(), 'best_model.pth')
            
        if loss.item() <= early_stop * initial_loss:
            print(f"Early stopping at epoch {epoch}, loss: {loss.item():.6f}")
            break
            
        if epoch % 1000 == 0:
            print(f"Epoch {epoch}: Total Loss = {loss.item():.6f}, PDE Loss = {pde_loss.item():.6f}, "
                  f"BC Loss = {(bc1_loss + bc2_loss).item():.6f}")
    
    print("Training complete")
    return r_theta_inner, r_theta_b1, r_theta_b2, r_theta_p1, r_theta_p2

# 显式基函数 (根据物理特征定制)
def explicit_bases(r_theta, num_explicit_bases=2):
    r = r_theta[:, 0:1]
    theta = r_theta[:, 1:2]
    bases = []
    
    # 基本项
    if num_explicit_bases >= 1:
        bases.append((r - r0) * (R - r) * torch.cos(theta))  # 满足边界条件的基函数
    
    if num_explicit_bases >= 2:
        bases.append(torch.exp(-(R - r)) * torch.cos(theta))  # 边界层项
    
    # 添加更多基函数以捕捉预应力效应
    if num_explicit_bases >= 3:
        bases.append(torch.cos(theta) * torch.log(r))
    
    return torch.cat(bases[:num_explicit_bases], dim=1) if bases else torch.tensor([])

# PDE残差计算
def compute_pde_operator(net, r_theta, explicit_bases_func, num_explicit_bases):
    r_theta = r_theta.requires_grad_(True)
    
    # 获取神经网络的隐藏层输出和显式基函数
    phi = net.get_hidden_layer_output(r_theta)
    psi = explicit_bases_func(r_theta, num_explicit_bases)
    basis = torch.cat([phi, psi], dim=1) if num_explicit_bases > 0 else phi
    
    L_basis_list = []
    for j in range(basis.shape[1]):
        u = basis[:, j].unsqueeze(1)
        
        # 计算双调和算子
        biharmonic_u = biharmonic(u, r_theta)
        
        # 计算方程右侧项
        rhs = equation_rhs(u, r_theta)
        
        # PDE残差
        L_basis_j = biharmonic_u - rhs
        L_basis_list.append(L_basis_j)
    
    L_basis = torch.cat(L_basis_list, dim=1)
    return L_basis

# 组装线性系统
def assemble_system(net, explicit_bases_func, r_theta_i, r_theta_b1, r_theta_b2, r_theta_p1, r_theta_p2, num_explicit_bases):
    # PDE部分
    L_basis_i = compute_pde_operator(net, r_theta_i, explicit_bases_func, num_explicit_bases)
    A_i = L_basis_i.detach().cpu().numpy()
    b_i = np.zeros(A_i.shape[0])
    
    # 边界条件部分
    # 外边界
    phi_b1 = net.get_hidden_layer_output(r_theta_b1).detach().cpu().numpy()
    psi_b1 = explicit_bases_func(r_theta_b1, num_explicit_bases).detach().cpu().numpy()
    A_b1 = np.concatenate([phi_b1, psi_b1], axis=1) if num_explicit_bases > 0 else phi_b1
    b_b1 = np.zeros(A_b1.shape[0])
    
    # 内边界
    phi_b2 = net.get_hidden_layer_output(r_theta_b2).detach().cpu().numpy()
    psi_b2 = explicit_bases_func(r_theta_b2, num_explicit_bases).detach().cpu().numpy()
    A_b2 = np.concatenate([phi_b2, psi_b2], axis=1) if num_explicit_bases > 0 else phi_b2
    target_w = (-alpha * r0 * torch.cos(r_theta_b2[:, 1:2])).detach().cpu().numpy()
    b_b2 = target_w.flatten()
    
    # 周期性边界
    phi_p1 = net.get_hidden_layer_output(r_theta_p1).detach().cpu().numpy()
    psi_p1 = explicit_bases_func(r_theta_p1, num_explicit_bases).detach().cpu().numpy()
    phi_p2 = net.get_hidden_layer_output(r_theta_p2).detach().cpu().numpy()
    psi_p2 = explicit_bases_func(r_theta_p2, num_explicit_bases).detach().cpu().numpy()
    basis_p1 = np.concatenate([phi_p1, psi_p1], axis=1) if num_explicit_bases > 0 else phi_p1
    basis_p2 = np.concatenate([phi_p2, psi_p2], axis=1) if num_explicit_bases > 0 else phi_p2
    A_periodic = basis_p1 - basis_p2
    b_periodic = np.zeros(A_periodic.shape[0])
    
    # 组装全局系统
    A = np.vstack([A_i, A_b1, A_b2, A_periodic])
    b = np.hstack([b_i, b_b1, b_b2, b_periodic])
    
    return A, b

# 计算位移场
def compute_displacement(net, w, r_theta, num_explicit_bases, device):
    r_theta_tensor = torch.tensor(r_theta, dtype=torch.float64, device=device)
    phi = net.get_hidden_layer_output(r_theta_tensor).detach().cpu().numpy()
    psi = explicit_bases(r_theta_tensor, num_explicit_bases).detach().cpu().numpy() if num_explicit_bases > 0 else np.array([])
    basis = np.hstack([phi, psi]) if num_explicit_bases > 0 else phi
    return basis @ w

# 可视化结果
def visualize_results(net, w, num_explicit_bases, device, n_points=100):
    # 极坐标网格
    r_eval = np.linspace(r0, R, n_points)
    theta_eval = np.linspace(0, 2*np.pi, n_points)
    R_eval, Theta_eval = np.meshgrid(r_eval, theta_eval)
    r_theta_eval = np.column_stack([R_eval.ravel(), Theta_eval.ravel()])
    
    # 计算位移场
    W_approx = compute_displacement(net, w, r_theta_eval, num_explicit_bases, device)
    W_approx = W_approx.reshape(n_points, n_points)
    
    # 转换为笛卡尔坐标以可视化
    X = R_eval * np.cos(Theta_eval)
    Y = R_eval * np.sin(Theta_eval)
    
    # 绘图
    plt.figure(figsize=(12, 10))
    ax = plt.subplot(projection='3d')
    surf = ax.plot_surface(X, Y, W_approx, cmap='viridis', rstride=1, cstride=1,
                          linewidth=0, antialiased=False)
    ax.set_title("Lateral Displacement W(r, θ)")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Displacement W")
    plt.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
    
    # 径向截面图
    plt.figure(figsize=(10, 6))
    r_theta_fixed_theta = np.column_stack([r_eval, np.zeros_like(r_eval)])
    W_radial = compute_displacement(net, w, r_theta_fixed_theta, num_explicit_bases, device)
    plt.plot(r_eval, W_radial)
    plt.title("Radial Displacement Profile (θ=0)")
    plt.xlabel("r")
    plt.ylabel("W(r)")
    plt.grid(True)
    
    plt.show()
    return W_approx

if __name__ == "__main__":
    # 主参数设置
    num_explicit_bases = 3  # 显式基函数数量
    hidden_size = 100        # 隐藏层大小
    subspace_dim = 100       # 子空间维度
    
    # 创建网络
    net = Net(hidden_size, subspace_dim).to(device)
    
    # 训练网络
    r_theta_i, r_theta_b1, r_theta_b2, r_theta_p1, r_theta_p2 = train_network(
        net, device, num_epochs=5000, lr=0.001, early_stop=1e-3)
    
    # 定义显式基函数
    def explicit_bases_func(r_theta, num_explicit_bases):
        r = r_theta[:, 0:1]
        theta = r_theta[:, 1:2]
        if num_explicit_bases == 0:
            return torch.empty((r_theta.shape[0], 0), device=device)
        bases = [torch.exp(-(r-b)/epsilon) * torch.cos(theta), torch.exp(-(1-r)/epsilon) * torch.cos(theta)]
        return torch.cat(bases[:num_explicit_bases], dim=1)
    
    # 组装并求解线性系统
    A, b = assemble_system(net, explicit_bases_func, r_theta_i, 
                           r_theta_b1, r_theta_b2, r_theta_p1, r_theta_p2, num_explicit_bases)
    
    # 最小二乘求解
    w, _, _, _ = lstsq(A, b)
    
    # 计算并可视化结果
    displacement_field = visualize_results(net, w, num_explicit_bases, device)
    
    print("Simulation completed successfully")