import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

# Check for GPU availability
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Set default tensor type to double precision
torch.set_default_tensor_type(torch.DoubleTensor)
torch.manual_seed(42)

# Problem parameters
b = 0.2         # Dimensionless inner radius (r0/R)
alpha = 0.5     # Rotation angle of rigid inclusion
epsilon = 0.005 # Bending stiffness to prestress ratio (D/(R^2 n))

class Net(nn.Module):
    def __init__(self, hidden_size=100, subspace_dim=50):
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

    def forward(self, r_theta):
        r = r_theta[:, 0:1]
        theta = r_theta[:, 1:2]
        
        # Normalize r to [b, 1] range
        r_normalized = (r - b) / (1 - b)
        
        # Use both r and theta as inputs
        xy = torch.cat([r_normalized, theta], dim=1)
        features = self.hidden(xy)
        w = self.output(features)
        return w * torch.cos(theta)  # Enforce cosθ dependence

    def get_hidden_layer_output(self, r_theta):
        r = r_theta[:, 0:1]
        theta = r_theta[:, 1:2]
        r_normalized = (r - b) / (1 - b)
        xy = torch.cat([r_normalized, theta], dim=1)
        return self.hidden(xy)

# Exact asymptotic solution from Eq (29)
def exact_solution(r_theta, epsilon):
    r = r_theta[:, 0:1]
    theta = r_theta[:, 1:2]
    
    # Main solution
    main_solution = (b**2 * alpha) / (1 - b**2) * (1/r - r)
    
    # Regular correction
    reg_correction = epsilon * (2 * alpha * b) / ((1 - b**2)**2) * ((1 + b**3)/r - (1 + b) * r)
    
    # Inner boundary layer correction
    inner_boundary = -epsilon * (2 * alpha) / (1 - b**2) * (torch.sqrt(b) / torch.sqrt(r)) * torch.exp(-(r - b) / epsilon)
    
    # Outer boundary layer correction
    outer_boundary = epsilon * (2 * alpha * b**2) / (1 - b**2) * (1 / torch.sqrt(r)) * torch.exp(-(1 - r) / epsilon)
    
    # Combine all terms
    w_r = main_solution + reg_correction + inner_boundary + outer_boundary
    
    return w_r * torch.cos(theta)

# Calculate Laplacian in polar coordinates
def polar_laplacian(u, r, r_theta, create_graph=True):
    # First derivatives
    du_dr = torch.autograd.grad(u, r_theta, grad_outputs=torch.ones_like(u), 
                               create_graph=create_graph, retain_graph=True)[0][:, 0:1]
    
    # Second derivatives
    d2u_dr2 = torch.autograd.grad(du_dr, r_theta, grad_outputs=torch.ones_like(du_dr),
                                 create_graph=create_graph, retain_graph=True)[0][:, 0:1]
    
    d2u_dtheta2 = torch.autograd.grad(u, r_theta, grad_outputs=torch.ones_like(u),
                                     create_graph=create_graph, retain_graph=True)[0]
    d2u_dtheta2 = torch.autograd.grad(d2u_dtheta2[:, 1:2], r_theta, 
                                     grad_outputs=torch.ones_like(d2u_dtheta2[:, 1:2]),
                                     create_graph=create_graph, retain_graph=True)[0][:, 1:2]
    
    # Laplacian in polar coordinates
    laplacian = d2u_dr2 + (1/r)*du_dr + (1/r**2)*d2u_dtheta2
    return laplacian

# Calculate biharmonic operator
def biharmonic(w, r_theta):
    # Compute first Laplacian
    lap_w = polar_laplacian(w, r_theta[:, 0:1], r_theta)
    
    # Compute Laplacian of Laplacian
    biharmonic_term = polar_laplacian(lap_w, r_theta[:, 0:1], r_theta, create_graph=True)
    return biharmonic_term

# PDE residual function
def pde_residual(r_theta):
    r_theta.requires_grad_(True)
    w = net(r_theta)
    
    # Compute biharmonic operator
    lap_lap_w = biharmonic(w, r_theta)
    
    # Compute Laplacian
    lap_w = polar_laplacian(w, r_theta[:, 0:1], r_theta)
    
    # PDE: ε²ΔΔW - ΔW = 0
    residual = epsilon**2 * lap_lap_w - lap_w
    return residual

# Train the network
def train_network(net, device, num_epochs=2000, lr=0.001):
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=100, factor=0.5)
    
    # Domain sampling - fixed number of points
    N_r = 100  # Points along radial direction
    N_theta = 50  # Points along angular direction
    
    # Ensure consistent number of points for boundary sampling
    r_interior = torch.linspace(b, 1, N_r, device=device)[1:-1]
    theta_interior = torch.linspace(0, 2*torch.pi, N_theta, device=device)
    R, Theta = torch.meshgrid(r_interior, theta_interior, indexing='ij')
    r_theta_interior = torch.stack([R.flatten(), Theta.flatten()], dim=1).requires_grad_(True)
    
    # Boundary points
    r_inner = b * torch.ones(N_theta, device=device)
    r_outer = torch.ones(N_theta, device=device)
    theta_boundary = torch.linspace(0, 2*torch.pi, N_theta, device=device)
    
    r_theta_inner = torch.stack([r_inner, theta_boundary], dim=1).requires_grad_(True)
    r_theta_outer = torch.stack([r_outer, theta_boundary], dim=1).requires_grad_(True)
    
    # Training loop
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        
        # PDE loss
        pde = pde_residual(r_theta_interior)
        loss_pde = torch.mean(pde**2)
        
        # Boundary conditions
        # Inner boundary: W = bα cosθ, dW/dr = α cosθ
        w_inner = net(r_theta_inner)
        dw_dr_inner = torch.autograd.grad(w_inner, r_theta_inner, 
                                         grad_outputs=torch.ones_like(w_inner),
                                         create_graph=True, retain_graph=True)[0][:, 0:1]
        loss_inner_w = torch.mean((w_inner - b*alpha*torch.cos(theta_boundary).unsqueeze(1))**2)
        loss_inner_dwdr = torch.mean((dw_dr_inner - alpha*torch.cos(theta_boundary).unsqueeze(1))**2)
        
        # Outer boundary: W = 0, dW/dr = 0
        w_outer = net(r_theta_outer)
        dw_dr_outer = torch.autograd.grad(w_outer, r_theta_outer,
                                         grad_outputs=torch.ones_like(w_outer),
                                         create_graph=True, retain_graph=True)[0][:, 0:1]
        loss_outer_w = torch.mean(w_outer**2)
        loss_outer_dwdr = torch.mean(dw_dr_outer**2)
        
        # Total boundary loss
        loss_boundary = loss_inner_w + loss_inner_dwdr + loss_outer_w + loss_outer_dwdr
        
        # Total loss
        loss = loss_pde + loss_boundary
        
        # Backpropagation with retain_graph=True for the first backward pass
        loss.backward(retain_graph=True)
        optimizer.step()
        scheduler.step(loss)
        
        if epoch % 100 == 0:
            print(f'Epoch {epoch}: Total loss = {loss.item():.4e}, '
                  f'PDE loss = {loss_pde.item():.4e}, '
                  f'Boundary loss = {loss_boundary.item():.4e}')
        
        if loss.item() < 1e-6:
            print(f'Converged at epoch {epoch}')
            break
    
    print(f'Final loss: {loss.item():.4e}')
    return net

# Evaluate the solution
def evaluate_solution(net, n_points=100):
    # Create grid
    r_vals = np.linspace(b, 1, n_points)
    theta_vals = np.linspace(0, 2*np.pi, n_points)
    R, Theta = np.meshgrid(r_vals, theta_vals)
    
    # Convert to Cartesian for plotting
    X = R * np.cos(Theta)
    Y = R * np.sin(Theta)
    
    # Prepare input tensor
    r_theta = np.stack([R.flatten(), Theta.flatten()], axis=1)
    r_theta_tensor = torch.tensor(r_theta, dtype=torch.float64, device=device)
    
    # Predict solution
    with torch.no_grad():
        w_pred = net(r_theta_tensor).cpu().numpy().reshape(n_points, n_points)
        w_exact = exact_solution(r_theta_tensor, epsilon).cpu().numpy().reshape(n_points, n_points)
    
    # Calculate error
    error = np.abs(w_pred - w_exact)
    
    # 2D plot of the solution
    plt.figure(figsize=(15, 5))
    
    plt.subplot(131)
    plt.contourf(X, Y, w_exact, 50, cmap='jet')
    plt.colorbar()
    plt.title('Exact Solution')
    plt.axis('equal')
    
    plt.subplot(132)
    plt.contourf(X, Y, w_pred, 50, cmap='jet')
    plt.colorbar()
    plt.title('Neural Network Solution')
    plt.axis('equal')
    
    plt.subplot(133)
    plt.contourf(X, Y, error, 50, cmap='viridis')
    plt.colorbar()
    plt.title('Absolute Error')
    plt.axis('equal')
    
    plt.tight_layout()
    plt.savefig('ring_plate_solution_2D.png', dpi=300)
    plt.show()
    
    # 3D plot comparison
    fig = plt.figure(figsize=(18, 6))
    
    ax1 = fig.add_subplot(131, projection='3d')
    surf = ax1.plot_surface(X, Y, w_exact, cmap='jet', rstride=1, cstride=1, linewidth=0, antialiased=True)
    ax1.set_title('Exact Solution')
    fig.colorbar(surf, ax=ax1, shrink=0.5)
    
    ax2 = fig.add_subplot(132, projection='3d')
    surf = ax2.plot_surface(X, Y, w_pred, cmap='jet', rstride=1, cstride=1, linewidth=0, antialiased=True)
    ax2.set_title('Neural Network Solution')
    fig.colorbar(surf, ax=ax2, shrink=0.5)
    
    ax3 = fig.add_subplot(133, projection='3d')
    surf = ax3.plot_surface(X, Y, error, cmap='viridis', rstride=1, cstride=1, linewidth=0, antialiased=True)
    ax3.set_title('Absolute Error')
    fig.colorbar(surf, ax=ax3, shrink=0.5)
    
    plt.tight_layout()
    plt.savefig('ring_plate_solution_3D.png', dpi=300)
    plt.show()
    
    # Radial profile at θ=0
    plt.figure(figsize=(10, 6))
    r_vals = np.linspace(b, 1, 100)
    theta_zero = np.zeros_like(r_vals)
    r_theta = np.stack([r_vals, theta_zero], axis=1)
    r_theta_tensor = torch.tensor(r_theta, dtype=torch.float64, device=device)
    
    with torch.no_grad():
        w_pred_radial = net(r_theta_tensor).cpu().numpy().flatten()
        w_exact_radial = exact_solution(r_theta_tensor, epsilon).cpu().numpy().flatten()
    
    plt.plot(r_vals, w_exact_radial, 'b-', linewidth=2, label='Exact')
    plt.plot(r_vals, w_pred_radial, 'r--', linewidth=2, label='NN Prediction')
    plt.title(f'Radial Profile at θ=0 (ε={epsilon})')
    plt.xlabel('r')
    plt.ylabel('W')
    plt.legend()
    plt.grid(True)
    
    # Mark the boundaries
    plt.axvline(x=b, color='g', linestyle=':', label=f'Inner boundary r={b}')
    plt.axvline(x=1, color='m', linestyle=':', label=f'Outer boundary r=1')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('ring_plate_radial_profile.png', dpi=300)
    plt.show()
    
    # Calculate error metrics
    l2_error = np.sqrt(np.mean(error**2))
    max_error = np.max(error)
    
    print(f'L2 Error: {l2_error:.4e}')
    print(f'Maximum Error: {max_error:.4e}')
    
    return {
        'X': X, 'Y': Y, 
        'w_exact': w_exact, 'w_pred': w_pred, 'error': error,
        'l2_error': l2_error, 'max_error': max_error
    }

# Main execution
if __name__ == "__main__":
    # Create the network
    net = Net(hidden_size=100, subspace_dim=50).to(device)
    
    # Train the network
    print("Starting training...")
    net = train_network(net, device, num_epochs=2000, lr=0.001)
    
    # Evaluate and plot
    print("Evaluating solution...")
    results = evaluate_solution(net, n_points=100)