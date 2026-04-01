#!/usr/bin/env python3
"""
一维热传导方程 PINN 求解器 (带热源项)
1D Heat Conduction PINN Solver with Heat Source

方程：∂T/∂t = α ∂²T/∂x² + f(x,t)
初始：T(x,0) = 0
边界：T(0,t) = 0, T(1,t) = 0
热源：f(x,t) = 5·exp(-20(x-0.5)²)
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

torch.manual_seed(42)
np.random.seed(42)

# ==================== 参数 ====================
alpha = 0.01
L = 1.0
T_total = 1.0
dt = 0.01

nx = 50
dx = L / (nx - 1)
nt = int(T_total / dt) + 1
dt = T_total / (nt - 1)
x = np.linspace(0, L, nx)
t = np.linspace(0, T_total, nt)

print("="*50)
print("PINN Solver: Heat Equation with Source Term")
print("="*50)
print(f"Equation: dT/dt = α·d²T/dx² + f(x,t)")
print(f"α = {alpha}, L = {L}, T = [0, {T_total}]")
print(f"dx = {dx:.6f}, dt = {dt}, nt = {nt}")
print()

# ==================== 热源函数 ====================
def heat_source(x, t):
    """f(x,t) = 5 * exp(-20*(x-0.5)²)"""
    return 5 * np.exp(-20 * (x - 0.5)**2)

# ==================== 神经网络 ====================
class PINN(nn.Module):
    def __init__(self, hidden=[64, 64, 64, 64]):
        super().__init__()
        layers = []
        prev = 2
        for h in hidden:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.Tanh())
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)
    
    def forward(self, x, t):
        return self.net(torch.cat([x, t], dim=1))

model = PINN()
print(f"Network: 2 → 64 → 64 → 64 → 64 → 1 (Tanh)")
print()

# ==================== 训练数据 ====================
print("Generating training data...")

N_f = 5000
x_f = np.random.uniform(0.01, L-0.01, N_f)
t_f = np.random.uniform(0.01, T_total, N_f)
x_f_t = torch.FloatTensor(x_f).reshape(-1, 1) / L
t_f_t = torch.FloatTensor(t_f).reshape(-1, 1) / T_total

# 初始条件 T(x,0) = 0
N_0 = 500
x_0 = np.random.uniform(0, L, N_0)
t_0 = np.zeros(N_0)
u_0 = np.zeros(N_0)  # 初始温度为 0
x_0_t = torch.FloatTensor(x_0).reshape(-1, 1) / L
t_0_t = torch.FloatTensor(t_0).reshape(-1, 1) / T_total
u_0_t = torch.FloatTensor(u_0).reshape(-1, 1)

# 边界条件 T(0,t) = 0, T(1,t) = 0
N_b = 500
t_b = np.random.uniform(0, T_total, N_b)
x_b_l = np.zeros(N_b)
x_b_r = np.ones(N_b) * L
u_b = np.zeros(N_b)  # 边界温度为 0
x_b_l_t = torch.FloatTensor(x_b_l).reshape(-1, 1) / L
x_b_r_t = torch.FloatTensor(x_b_r).reshape(-1, 1) / L
t_b_t = torch.FloatTensor(t_b).reshape(-1, 1) / T_total
u_b_t = torch.FloatTensor(u_b).reshape(-1, 1)

print(f"  Collocation: {N_f}, IC: {N_0}, BC: {N_b}")
print()

# ==================== 损失函数 ====================
optimizer = torch.optim.Adam(model.parameters(), lr=0.002)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3000, gamma=0.6)

def compute_loss():
    model.train()
    
    x_f_r = x_f_t.requires_grad_(True)
    t_f_r = t_f_t.requires_grad_(True)
    
    u_f = model(x_f_r, t_f_r)
    
    # dT/dt
    u_t = torch.autograd.grad(u_f, t_f_r, torch.ones_like(u_f),
                               create_graph=True, retain_graph=True)[0]
    
    # d²T/dx²
    u_x = torch.autograd.grad(u_f, x_f_r, torch.ones_like(u_f),
                               create_graph=True, retain_graph=True)[0]
    u_xx = torch.autograd.grad(u_x, x_f_r, torch.ones_like(u_x),
                                create_graph=True, retain_graph=True)[0]
    
    # PDE: dT/dt - α·d²T/dx² - f(x,t) = 0
    f_source = heat_source(x_f_r.detach().numpy() * L, t_f_r.detach().numpy() * T_total)
    f_source_t = torch.FloatTensor(f_source).reshape(-1, 1)
    
    pde_res = u_t - alpha * u_xx - f_source_t
    loss_pde = torch.mean(pde_res ** 2)
    
    # IC loss
    u_0_pred = model(x_0_t, t_0_t)
    loss_0 = torch.mean((u_0_pred - u_0_t) ** 2)
    
    # BC loss
    u_bl = model(x_b_l_t, t_b_t)
    u_br = model(x_b_r_t, t_b_t)
    loss_bc = torch.mean((u_bl - u_b_t) ** 2) + torch.mean((u_br - u_b_t) ** 2)
    
    total = loss_pde + 10.0 * loss_0 + 5.0 * loss_bc
    return total, loss_pde, loss_0, loss_bc

# ==================== 训练 ====================
print("Training...")
print("-" * 50)

n_epochs = 15000
for epoch in range(n_epochs):
    optimizer.zero_grad()
    loss, lp, l0, lbc = compute_loss()
    loss.backward()
    optimizer.step()
    scheduler.step()
    
    if (epoch + 1) % 1500 == 0:
        print(f"Epoch [{epoch+1:5d}] | Loss: {loss.item():.6f} | PDE: {lp.item():.6f} | IC: {l0.item():.6f} | BC: {lbc.item():.6f}")

print("-" * 50)
print("Training done!")
print()

# ==================== 预测 ====================
print("Predicting...")
model.eval()

X_g, T_g = np.meshgrid(x, t)
X_t = torch.FloatTensor(X_g.flatten()).reshape(-1, 1) / L
T_t = torch.FloatTensor(T_g.flatten()).reshape(-1, 1) / T_total

with torch.no_grad():
    u_pinn = model(X_t, T_t).numpy().reshape(nt, nx)

print(f"Prediction shape: {u_pinn.shape}")
print()

# ==================== 有限差分参考解 ====================
print("Computing finite difference reference...")
u_fd = np.zeros((nt, nx))
r = alpha * dt / dx**2

for n in range(nt - 1):
    for i in range(1, nx - 1):
        u_fd[n+1, i] = u_fd[n, i] + r * (u_fd[n, i+1] - 2*u_fd[n, i] + u_fd[n, i-1]) + dt * heat_source(x[i], t[n])
    u_fd[n+1, 0] = 0
    u_fd[n+1, -1] = 0

# ==================== 误差计算 ====================
abs_err = np.abs(u_pinn - u_fd)
rel_err = np.zeros_like(abs_err)
mask = np.abs(u_fd) > 1e-6
rel_err[mask] = (abs_err[mask] / np.abs(u_fd[mask])) * 100
rel_err[~mask] = abs_err[~mask] * 100

print("\nError Statistics:")
print(f"  Max abs error: {np.max(abs_err):.6f}")
print(f"  Mean abs error: {np.mean(abs_err):.6f}")
print(f"  Max rel error: {np.max(rel_err):.4f}%")
print(f"  Mean rel error: {np.mean(rel_err):.4f}%")
print()

# ==================== 绘图 ====================
print("Generating plots...")
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

# PINN 解图
fig1 = plt.figure(figsize=(14, 10))

ax1 = fig1.add_subplot(2, 2, 1, projection='3d')
surf = ax1.plot_surface(T_g, X_g, u_pinn, cmap='hot', alpha=0.9)
ax1.set_xlabel('Time (s)')
ax1.set_ylabel('Position (m)')
ax1.set_zlabel('Temperature')
ax1.set_title('PINN Solution 3D')
fig1.colorbar(surf, ax=ax1, shrink=0.5)

ax2 = fig1.add_subplot(2, 2, 2)
cont = ax2.contourf(T_g, X_g, u_pinn, levels=20, cmap='hot')
ax2.set_xlabel('Time (s)')
ax2.set_ylabel('Position (m)')
ax2.set_title('PINN Contours')
fig1.colorbar(cont, ax=ax2)

ax3 = fig1.add_subplot(2, 2, 3)
for idx, c in zip([0, nt//4, nt//2, 3*nt//4, nt-1], ['b','g','orange','r','purple']):
    ax3.plot(x, u_pinn[idx, :], c=c, label=f't={t[idx]:.2f}s')
ax3.plot(x, heat_source(x, 0), 'k--', label='Source f(x)')
ax3.set_xlabel('Position (m)')
ax3.set_ylabel('Temperature')
ax3.set_title('Temperature Profiles')
ax3.legend()
ax3.grid(True, alpha=0.3)

ax4 = fig1.add_subplot(2, 2, 4)
for idx, lbl in zip([0, nx//4, nx//2, 3*nx//4, nx-1], ['x=0','x=0.25','x=0.5','x=0.75','x=1']):
    ax4.plot(t, u_pinn[:, idx], label=lbl)
ax4.set_xlabel('Time (s)')
ax4.set_ylabel('Temperature')
ax4.set_title('Temperature Evolution')
ax4.legend()
ax4.grid(True, alpha=0.3)

plt.tight_layout()
pinn_plot = f'/home/admin/Desktop/神经网络一维热传导/pinn_{timestamp}.png'
fig1.savefig(pinn_plot, dpi=150, bbox_inches='tight')
plt.close()

# 对比图
fig2 = plt.figure(figsize=(16, 10))

ax1 = fig2.add_subplot(2, 3, 1)
ax1.contourf(T_g, X_g, u_fd, levels=20, cmap='hot')
ax1.set_xlabel('Time (s)')
ax1.set_ylabel('Position (m)')
ax1.set_title('Finite Difference')

ax2 = fig2.add_subplot(2, 3, 2)
ax2.contourf(T_g, X_g, u_pinn, levels=20, cmap='hot')
ax2.set_xlabel('Time (s)')
ax2.set_ylabel('Position (m)')
ax2.set_title('PINN')

ax3 = fig2.add_subplot(2, 3, 3)
ax3.contourf(T_g, X_g, abs_err, levels=20, cmap='viridis')
ax3.set_xlabel('Time (s)')
ax3.set_ylabel('Position (m)')
ax3.set_title('Absolute Error')

ax4 = fig2.add_subplot(2, 3, 4)
ax4.contourf(T_g, X_g, rel_err, levels=20, cmap='viridis')
ax4.set_xlabel('Time (s)')
ax4.set_ylabel('Position (m)')
ax4.set_title('Relative Error (%)')

ax5 = fig2.add_subplot(2, 3, 5)
ax5.hist(rel_err.flatten(), bins=50, color='steelblue', edgecolor='black', alpha=0.7)
ax5.set_xlabel('Relative Error (%)')
ax5.set_ylabel('Frequency')
ax5.set_title('Error Distribution')
ax5.axvline(np.mean(rel_err), color='red', linestyle='--', label=f'Mean: {np.mean(rel_err):.3f}%')
ax5.legend()

ax6 = fig2.add_subplot(2, 3, 6)
center = nx // 2
ax6.plot(t, u_fd[:, center], 'b-', lw=2, label='FD')
ax6.plot(t, u_pinn[:, center], 'r--', lw=2, label='PINN')
ax6.set_xlabel('Time (s)')
ax6.set_ylabel('Temperature')
ax6.set_title('Center Point Comparison')
ax6.legend()
ax6.grid(True, alpha=0.3)

plt.tight_layout()
comp_plot = f'/home/admin/Desktop/神经网络一维热传导/comparison_{timestamp}.png'
fig2.savefig(comp_plot, dpi=150, bbox_inches='tight')
plt.close()

print(f"  PINN plot: {pinn_plot}")
print(f"  Comparison: {comp_plot}")

# ==================== Markdown 数据 ====================
print("Generating data file...")

md_file = f'/home/admin/Desktop/神经网络一维热传导/pinn_data_{timestamp}.md'

with open(md_file, 'w', encoding='utf-8') as f:
    f.write("# PINN Solution: Heat Equation with Source\n\n")
    f.write("## Equation\n\n")
    f.write("$$\\frac{\\partial T}{\\partial t} = \\alpha \\frac{\\partial^2 T}{\\partial x^2} + f(x,t)$$\n\n")
    f.write("## Parameters\n\n")
    f.write(f"- α = {alpha} m²/s\n")
    f.write(f"- Domain: [0, {L}] m × [0, {T_total}] s\n")
    f.write(f"- Grid: nx={nx}, nt={nt}\n")
    f.write(f"- Source: f(x,t) = 5·exp(-20(x-0.5)²)\n\n")
    
    f.write("## Conditions\n\n")
    f.write("- IC: T(x,0) = 0\n")
    f.write("- BC: T(0,t) = 0, T(1,t) = 0\n\n")
    
    f.write("## Network\n\n")
    f.write("- Architecture: 2 → 64 → 64 → 64 → 64 → 1\n")
    f.write("- Activation: Tanh\n")
    f.write("- Optimizer: Adam (lr=0.002)\n")
    f.write("- Epochs: 15000\n\n")
    
    f.write("## Error vs Finite Difference\n\n")
    f.write(f"- Max abs: {np.max(abs_err):.6f}\n")
    f.write(f"- Mean abs: {np.mean(abs_err):.6f}\n")
    f.write(f"- Max rel: {np.max(rel_err):.4f}%\n")
    f.write(f"- Mean rel: {np.mean(rel_err):.4f}%\n\n")
    
    f.write("## Temperature Data (PINN)\n\n")
    f.write("| t\\x |")
    for i in range(0, nx, 5):
        f.write(f" {x[i]:.2f} |")
    f.write("\n|")
    f.write("-|" * (len(range(0, nx, 5)) + 1))
    f.write("\n")
    
    for n in range(0, nt, max(1, nt//15)):
        f.write(f"| {t[n]:.2f} |")
        for i in range(0, nx, 5):
            f.write(f" {u_pinn[n,i]:.4f} |")
        f.write("\n")
    
    f.write("\n## Relative Error (%)\n\n")
    f.write("| t\\x |")
    for i in range(0, nx, 5):
        f.write(f" {x[i]:.2f} |")
    f.write("\n|")
    f.write("-|" * (len(range(0, nx, 5)) + 1))
    f.write("\n")
    
    for n in range(0, nt, max(1, nt//15)):
        f.write(f"| {t[n]:.2f} |")
        for i in range(0, nx, 5):
            f.write(f" {rel_err[n,i]:.4f} |")
        f.write("\n")
    
    f.write("\n## Center Point Comparison\n\n")
    f.write("| Time | FD | PINN | Diff | Rel Err (%) |\n")
    f.write("|------|----|------|------|-------------|\n")
    for n in range(0, nt, max(1, nt//20)):
        fd_v = u_fd[n, center]
        pinn_v = u_pinn[n, center]
        diff = abs(fd_v - pinn_v)
        rel = (diff / abs(fd_v)) * 100 if abs(fd_v) > 1e-6 else diff * 100
        f.write(f"| {t[n]:.2f} | {fd_v:.6f} | {pinn_v:.6f} | {diff:.6f} | {rel:.4f} |\n")
    
    f.write(f"\n---\n*Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n")

print(f"  Data: {md_file}")
print("\n" + "="*50)
print("COMPLETE")
print("="*50)
