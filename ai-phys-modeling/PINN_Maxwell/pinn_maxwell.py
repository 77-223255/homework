"""
PINN for 1D Maxwell Equations
一维电磁波的物理信息神经网络求解

麦克斯韦方程组:
∂E/∂x = μ₀ ∂H/∂t
∂H/∂x = ε₀ ∂E/∂t

初始条件:
E(x,0) = sin(kx), H(x,0) = 0

边界条件:
E(0,t) = E(1,t) = 0
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
import time

# 设置中文字体
rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial Unicode MS', 'SimHei']
rcParams['axes.unicode_minus'] = False

# 物理常数
mu0 = 1.0  # 磁导率 (归一化)
epsilon0 = 1.0  # 介电常数 (归一化)
c = 1.0 / np.sqrt(mu0 * epsilon0)  # 光速

# 波数参数
k = np.pi  # 使得 sin(kx) 在 x=0,1 处为 0
omega = c * k  # 角频率

# 计算域
x_min, x_max = 0.0, 1.0
t_min, t_max = 0.0, 1.0

# 解析解 (用于验证)
def analytical_solution(x, t):
    """解析解: E = sin(kx)cos(ωt), H = -sqrt(ε₀/μ₀)cos(kx)sin(ωt)"""
    E = np.sin(k * x) * np.cos(omega * t)
    H = np.sqrt(epsilon0 / mu0) * np.cos(k * x) * np.sin(omega * t)
    return E, H


class PINN(nn.Module):
    """物理信息神经网络"""
    
    def __init__(self, layers=[2, 50, 50, 50, 2]):
        super(PINN, self).__init__()
        self.layers = layers
        self.network = self.build_network()
        
    def build_network(self):
        """构建全连接网络"""
        modules = []
        for i in range(len(self.layers) - 1):
            modules.append(nn.Linear(self.layers[i], self.layers[i+1]))
            if i < len(self.layers) - 2:
                modules.append(nn.Tanh())
        return nn.Sequential(*modules)
    
    def forward(self, x, t):
        """前向传播"""
        inputs = torch.cat([x, t], dim=1)
        outputs = self.network(inputs)
        E = outputs[:, 0:1]
        H = outputs[:, 1:2]
        return E, H
    
    def compute_derivatives(self, x, t):
        """使用自动微分计算偏导数"""
        x.requires_grad_(True)
        t.requires_grad_(True)
        
        E, H = self.forward(x, t)
        
        # E/∂x
        dE_dx = torch.autograd.grad(E, x, 
                                     grad_outputs=torch.ones_like(E),
                                     create_graph=True)[0]
        
        # ∂E/∂t
        dE_dt = torch.autograd.grad(E, t,
                                     grad_outputs=torch.ones_like(E),
                                     create_graph=True)[0]
        
        # ∂H/∂x
        dH_dx = torch.autograd.grad(H, x,
                                     grad_outputs=torch.ones_like(H),
                                     create_graph=True)[0]
        
        # ∂H/∂t
        dH_dt = torch.autograd.grad(H, t,
                                     grad_outputs=torch.ones_like(H),
                                     create_graph=True)[0]
        
        return dE_dx, dE_dt, dH_dx, dH_dt, E, H


def sample_domain_points(N_f):
    """采样配置点 (内部点)"""
    x = np.random.uniform(x_min, x_max, (N_f, 1))
    t = np.random.uniform(t_min, t_max, (N_f, 1))
    return torch.FloatTensor(x), torch.FloatTensor(t)


def sample_initial_points(N_ic):
    """采样初始条件点 (t=0)"""
    x = np.random.uniform(x_min, x_max, (N_ic, 1))
    t = np.zeros((N_ic, 1))
    return torch.FloatTensor(x), torch.FloatTensor(t)


def sample_boundary_points(N_bc):
    """采样边界条件点 (x=0 和 x=1)"""
    t = np.random.uniform(t_min, t_max, (N_bc, 1))
    
    # x = 0 边界
    x_left = np.zeros((N_bc // 2, 1))
    t_left = t[:N_bc // 2]
    
    # x = 1 边界
    x_right = np.ones((N_bc - N_bc // 2, 1))
    t_right = t[N_bc // 2:]
    
    x = np.vstack([x_left, x_right])
    t = np.vstack([t_left, t_right])
    
    return torch.FloatTensor(x), torch.FloatTensor(t)


def compute_loss(model, x_f, t_f, x_ic, t_ic, x_bc, t_bc):
    """计算总损失"""
    
    # PDE 残差损失
    dE_dx, dE_dt, dH_dx, dH_dt, E, H = model.compute_derivatives(x_f, t_f)
    
    # 麦克斯韦方程残差
    residual_1 = dE_dx - mu0 * dH_dt  # ∂E/x - μ₀∂H/∂t = 0
    residual_2 = dH_dx - epsilon0 * dE_dt  # H/∂x - ε₀∂E/∂t = 0
    
    loss_pde = torch.mean(residual_1**2) + torch.mean(residual_2**2)
    
    # 初始条件损失
    E_ic, H_ic = model(x_ic, t_ic)
    E_exact_ic, H_exact_ic = analytical_solution(x_ic.detach().numpy(), 
                                                   t_ic.detach().numpy())
    E_exact_ic = torch.FloatTensor(E_exact_ic)
    H_exact_ic = torch.FloatTensor(H_exact_ic)
    
    loss_ic = torch.mean((E_ic - E_exact_ic)**2) + torch.mean((H_ic - H_exact_ic)**2)
    
    # 边界条件损失
    E_bc, _ = model(x_bc, t_bc)
    loss_bc = torch.mean(E_bc**2)  # E(0,t) = E(1,t) = 0
    
    # 总损失
    loss = loss_pde + loss_ic + loss_bc
    
    return loss, loss_pde, loss_ic, loss_bc


def train():
    """训练 PINN"""
    print("=" * 60)
    print("PINN for 1D Maxwell Equations")
    print("一维麦克斯韦方程组的物理信息神经网络求解")
    print("=" * 60)
    
    # 设置随机种子
    torch.manual_seed(42)
    np.random.seed(42)
    
    # 采样点数
    N_f = 5000   # 配置点
    N_ic = 500   # 初始条件点
    N_bc = 500   # 边界条件点
    
    # 采样
    x_f, t_f = sample_domain_points(N_f)
    x_ic, t_ic = sample_initial_points(N_ic)
    x_bc, t_bc = sample_boundary_points(N_bc)
    
    # 创建模型
    model = PINN(layers=[2, 50, 50, 50, 2])
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # 训练参数
    n_epochs = 10000
    loss_history = []
    loss_pde_history = []
    loss_ic_history = []
    loss_bc_history = []
    
    print(f"\n开始训练...")
    print(f"配置点数: {N_f}")
    print(f"初始条件点数: {N_ic}")
    print(f"边界条件点数: {N_bc}")
    print(f"训练轮数: {n_epochs}")
    print("-" * 60)
    
    start_time = time.time()
    
    for epoch in range(n_epochs):
        optimizer.zero_grad()
        loss, loss_pde, loss_ic, loss_bc = compute_loss(
            model, x_f, t_f, x_ic, t_ic, x_bc, t_bc
        )
        loss.backward()
        optimizer.step()
        
        loss_history.append(loss.item())
        loss_pde_history.append(loss_pde.item())
        loss_ic_history.append(loss_ic.item())
        loss_bc_history.append(loss_bc.item())
        
        if (epoch + 1) % 1000 == 0:
            elapsed = time.time() - start_time
            print(f"Epoch {epoch+1:5d}/{n_epochs:5d} | "
                  f"Loss: {loss.item():.6e} | "
                  f"PDE: {loss_pde.item():.6e} | "
                  f"IC: {loss_ic.item():.6e} | "
                  f"BC: {loss_bc.item():.6e} | "
                  f"Time: {elapsed:.1f}s")
    
    total_time = time.time() - start_time
    print("-" * 60)
    print(f"训练完成！总时间：{total_time:.1f}秒")
    print(f"最终损失：{loss_history[-1]:.6e}")
    
    return model, loss_history, loss_pde_history, loss_ic_history, loss_bc_history


def plot_results(model, loss_history, loss_pde_history, loss_ic_history, loss_bc_history):
    """绘制结果"""
    print("\n生成结果图像...")
    
    # 创建网格用于可视化
    nx, nt = 100, 100
    x = np.linspace(x_min, x_max, nx).reshape(-1, 1)
    t = np.linspace(t_min, t_max, nt).reshape(-1, 1)
    
    X, T = np.meshgrid(x, t)
    x_flat = torch.FloatTensor(X.flatten().reshape(-1, 1))
    t_flat = torch.FloatTensor(T.flatten().reshape(-1, 1))
    
    # PINN 预测
    with torch.no_grad():
        E_pred, H_pred = model(x_flat, t_flat)
    E_pred = E_pred.numpy().reshape(nt, nx)
    H_pred = H_pred.numpy().reshape(nt, nx)
    
    # 解析解
    E_exact, H_exact = analytical_solution(X, T)
    
    # 误差
    E_error = np.abs(E_pred - E_exact)
    H_error = np.abs(H_pred - H_exact)
    
    # 创建图像
    fig = plt.figure(figsize=(16, 12))
    
    # 1. 训练损失曲线
    ax1 = fig.add_subplot(3, 3, 1)
    ax1.semilogy(loss_history, 'b-', linewidth=2, label='Total Loss')
    ax1.semilogy(loss_pde_history, 'r--', linewidth=2, label='PDE Loss')
    ax1.semilogy(loss_ic_history, 'g--', linewidth=2, label='IC Loss')
    ax1.semilogy(loss_bc_history, 'm--', linewidth=2, label='BC Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss (log scale)')
    ax1.set_title('Training Loss History')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. E 场 PINN 解
    ax2 = fig.add_subplot(3, 3, 2)
    im2 = ax2.imshow(E_pred, extent=[x_min, x_max, t_min, t_max], 
                     origin='lower', aspect='auto', cmap='RdBu_r')
    ax2.set_xlabel('x')
    ax2.set_ylabel('t')
    ax2.set_title('E(x,t) - PINN Prediction')
    plt.colorbar(im2, ax=ax2)
    
    # 3. E 场解析解
    ax3 = fig.add_subplot(3, 3, 3)
    im3 = ax3.imshow(E_exact, extent=[x_min, x_max, t_min, t_max],
                     origin='lower', aspect='auto', cmap='RdBu_r')
    ax3.set_xlabel('x')
    ax3.set_ylabel('t')
    ax3.set_title('E(x,t) - Analytical Solution')
    plt.colorbar(im3, ax=ax3)
    
    # 4. E 场误差
    ax4 = fig.add_subplot(3, 3, 4)
    im4 = ax4.imshow(E_error, extent=[x_min, x_max, t_min, t_max],
                     origin='lower', aspect='auto', cmap='hot')
    ax4.set_xlabel('x')
    ax4.set_ylabel('t')
    ax4.set_title('E(x,t) - Absolute Error')
    plt.colorbar(im4, ax=ax4)
    
    # 5. H 场 PINN 解
    ax5 = fig.add_subplot(3, 3, 5)
    im5 = ax5.imshow(H_pred, extent=[x_min, x_max, t_min, t_max],
                     origin='lower', aspect='auto', cmap='RdBu_r')
    ax5.set_xlabel('x')
    ax5.set_ylabel('t')
    ax5.set_title('H(x,t) - PINN Prediction')
    plt.colorbar(im5, ax=ax5)
    
    # 6. H 场解析解
    ax6 = fig.add_subplot(3, 3, 6)
    im6 = ax6.imshow(H_exact, extent=[x_min, x_max, t_min, t_max],
                     origin='lower', aspect='auto', cmap='RdBu_r')
    ax6.set_xlabel('x')
    ax6.set_ylabel('t')
    ax6.set_title('H(x,t) - Analytical Solution')
    plt.colorbar(im6, ax=ax6)
    
    # 7. H 场误差
    ax7 = fig.add_subplot(3, 3, 7)
    im7 = ax7.imshow(H_error, extent=[x_min, x_max, t_min, t_max],
                     origin='lower', aspect='auto', cmap='hot')
    ax7.set_xlabel('x')
    ax7.set_ylabel('t')
    ax7.set_title('H(x,t) - Absolute Error')
    plt.colorbar(im7, ax=ax7)
    
    # 8. t=0.5 时刻的 E 场对比
    ax8 = fig.add_subplot(3, 3, 8)
    t_idx = nt // 2
    ax8.plot(x.flatten(), E_pred[t_idx, :], 'b-', linewidth=2, label='PINN')
    ax8.plot(x.flatten(), E_exact[t_idx, :], 'r--', linewidth=2, label='Analytical')
    ax8.set_xlabel('x')
    ax8.set_ylabel('E')
    ax8.set_title(f'E(x, t={t[t_idx][0]:.2f})')
    ax8.legend()
    ax8.grid(True, alpha=0.3)
    
    # 9. t=0.5 时刻的 H 场对比
    ax9 = fig.add_subplot(3, 3, 9)
    ax9.plot(x.flatten(), H_pred[t_idx, :], 'b-', linewidth=2, label='PINN')
    ax9.plot(x.flatten(), H_exact[t_idx, :], 'r--', linewidth=2, label='Analytical')
    ax9.set_xlabel('x')
    ax9.set_ylabel('H')
    ax9.set_title(f'H(x, t={t[t_idx][0]:.2f})')
    ax9.legend()
    ax9.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('results.png', dpi=150, bbox_inches='tight')
    print("已保存：results.png")
    plt.close()
    
    # 单独保存训练损失图
    fig2, ax = plt.subplots(figsize=(10, 6))
    ax.semilogy(loss_history, 'b-', linewidth=2, label='Total Loss')
    ax.semilogy(loss_pde_history, 'r--', linewidth=2, label='PDE Loss')
    ax.semilogy(loss_ic_history, 'g--', linewidth=2, label='IC Loss')
    ax.semilogy(loss_bc_history, 'm--', linewidth=2, label='BC Loss')
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Loss (log scale)', fontsize=12)
    ax.set_title('Training Loss History - PINN for Maxwell Equations', fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('training_loss.png', dpi=150, bbox_inches='tight')
    print("已保存：training_loss.png")
    plt.close()
    
    # 保存 E 场和 H 场的时空演化图
    fig3, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    
    im1 = ax1.imshow(E_pred, extent=[x_min, x_max, t_min, t_max],
                     origin='lower', aspect='auto', cmap='RdBu_r')
    ax1.set_xlabel('x')
    ax1.set_ylabel('t')
    ax1.set_title('Electric Field E(x,t) - PINN Solution')
    plt.colorbar(im1, ax=ax1)
    
    im2 = ax2.imshow(H_pred, extent=[x_min, x_max, t_min, t_max],
                     origin='lower', aspect='auto', cmap='RdBu_r')
    ax2.set_xlabel('x')
    ax2.set_ylabel('t')
    ax2.set_title('Magnetic Field H(x,t) - PINN Solution')
    plt.colorbar(im2, ax=ax2)
    
    plt.tight_layout()
    plt.savefig('fields_evolution.png', dpi=150, bbox_inches='tight')
    print("已保存：fields_evolution.png")
    plt.close()
    
    # 计算并打印误差统计
    print("\n" + "=" * 60)
    print("误差统计:")
    print(f"E 场最大误差：{np.max(E_error):.6e}")
    print(f"E 场平均误差：{np.mean(E_error):.6e}")
    print(f"H 场最大误差：{np.max(H_error):.6e}")
    print(f"H 场平均误差：{np.mean(H_error):.6e}")
    print("=" * 60)


def main():
    """主函数"""
    # 训练模型
    model, loss_history, loss_pde_history, loss_ic_history, loss_bc_history = train()
    
    # 保存模型
    torch.save(model.state_dict(), 'pinn_maxwell_model.pth')
    print("\n已保存模型：pinn_maxwell_model.pth")
    
    # 绘制结果
    plot_results(model, loss_history, loss_pde_history, loss_ic_history, loss_bc_history)
    
    print("\n所有结果已保存到当前目录!")
    print("文件列表:")
    print("  - pinn_maxwell.py (源代码)")
    print("  - pinn_maxwell_model.pth (训练好的模型)")
    print("  - results.png (完整结果对比)")
    print("  - training_loss.png (训练损失曲线)")
    print("  - fields_evolution.png (电磁场时空演化)")


if __name__ == "__main__":
    main()
