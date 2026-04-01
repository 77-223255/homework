#!/usr/bin/env python3
"""
一维热传导方程有限差分求解器 (带热源项)
1D Heat Conduction Equation Solver with Heat Source

方程：∂T/∂t = α ∂²T/∂x² + f(x,t)
方法：显式 FTCS 格式
"""

import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

# ==================== 参数设置 ====================
alpha = 0.01      # 热扩散系数 (m²/s)
L = 1.0           # 杆长度 (m)
T_total = 1.0     # 总时间 0~1s
dt = 0.01         # 时间间隔 0.01s

# 网格划分
nx = 50           # 空间网格点数
dx = L / (nx - 1) # 空间步长
nt = int(T_total / dt) + 1  # 时间步数
dt = T_total / (nt - 1)     # 重新调整 dt

# 稳定性检查
r = alpha * dt / dx**2
print(f"空间步长 dx = {dx:.6f} m")
print(f"时间步长 dt = {dt:.6f} s")
print(f"时间步数 nt = {nt}")
print(f"稳定性参数 r = {r:.4f} (需 <= 0.5) - {'OK' if r <= 0.5 else 'UNSTABLE!'}")

# ==================== 热源函数 ====================
def heat_source(x, t):
    """热源项 f(x,t) = 5 * exp(-20*(x-0.5)²)"""
    return 5 * np.exp(-20 * (x - 0.5)**2)

# ==================== 初始和边界条件 ====================
# 初始条件：T(x,0) = 0
# 边界条件：T(0,t) = 0, T(1,t) = 0
T_left = 0
T_right = 0

# ==================== 初始化 ====================
x = np.linspace(0, L, nx)
t = np.linspace(0, T_total, nt)
u = np.zeros((nt, nx))  # 初始温度为 0

# ==================== 有限差分求解 ====================
print("\n求解带热源的一维热传导方程...")

for n in range(nt - 1):
    for i in range(1, nx - 1):
        # FTCS 格式 + 热源项
        u[n+1, i] = u[n, i] + r * (u[n, i+1] - 2*u[n, i] + u[n, i-1]) + dt * heat_source(x[i], t[n])
    
    # 边界条件
    u[n+1, 0] = T_left
    u[n+1, -1] = T_right

print("求解完成!")

# ==================== 生成图表 ====================
print("\n生成图表...")

fig = plt.figure(figsize=(14, 10))

# 3D 表面图
ax1 = fig.add_subplot(2, 2, 1, projection='3d')
X, T_mesh = np.meshgrid(x, t)
surf = ax1.plot_surface(T_mesh, X, u, cmap='hot', alpha=0.9, edgecolor='none')
ax1.set_xlabel('Time (s)')
ax1.set_ylabel('Position (m)')
ax1.set_zlabel('Temperature T')
ax1.set_title('3D Temperature Surface')
fig.colorbar(surf, ax=ax1, shrink=0.5, label='Temperature')

# 等高线图
ax2 = fig.add_subplot(2, 2, 2)
contour = ax2.contourf(T_mesh, X, u, levels=20, cmap='hot')
ax2.set_xlabel('Time (s)')
ax2.set_ylabel('Position (m)')
ax2.set_title('Temperature Contours')
fig.colorbar(contour, ax=ax2, label='Temperature')

# 不同时刻的温度剖面
ax3 = fig.add_subplot(2, 2, 3)
time_indices = [0, nt//4, nt//2, 3*nt//4, nt-1]
colors = ['blue', 'green', 'orange', 'red', 'purple']
for idx, color in zip(time_indices, colors):
    ax3.plot(x, u[idx, :], color=color, label=f't = {t[idx]:.2f}s')
ax3.plot(x, heat_source(x, 0), 'k--', linewidth=2, label='Heat source f(x)')
ax3.set_xlabel('Position (m)')
ax3.set_ylabel('Temperature / Source')
ax3.set_title('Temperature Profiles')
ax3.legend()
ax3.grid(True, alpha=0.3)

# 不同位置的温度演化
ax4 = fig.add_subplot(2, 2, 4)
space_indices = [0, nx//4, nx//2, 3*nx//4, nx-1]
labels = ['x=0', 'x=0.25', 'x=0.5 (center)', 'x=0.75', 'x=1']
for idx, label in zip(space_indices, labels):
    ax4.plot(t, u[:, idx], label=label)
ax4.set_xlabel('Time (s)')
ax4.set_ylabel('Temperature')
ax4.set_title('Temperature Evolution')
ax4.legend()
ax4.grid(True, alpha=0.3)

plt.tight_layout()

timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
plot_file = f'/home/admin/Desktop/差分一维热传导/heat_result_{timestamp}.png'
plt.savefig(plot_file, dpi=150, bbox_inches='tight')
plt.close()
print(f"图片：{plot_file}")

# ==================== 生成 Markdown 数据 ====================
print("生成数据文件...")

md_file = f'/home/admin/Desktop/差分一维热传导/heat_data_{timestamp}.md'

with open(md_file, 'w', encoding='utf-8') as f:
    f.write("# 一维热传导方程数值解 (带热源)\n\n")
    f.write("## 控制方程\n\n")
    f.write("$$\\frac{\\partial T}{\\partial t} = \\alpha \\frac{\\partial^2 T}{\\partial x^2} + f(x,t)$$\n\n")
    f.write("## 参数\n\n")
    f.write(f"- 热扩散系数 α = {alpha} m²/s\n")
    f.write(f"- 空间域 [0, {L}] m\n")
    f.write(f"- 时间域 [0, {T_total}] s\n")
    f.write(f"- 空间步长 dx = {dx:.6f} m (nx={nx})\n")
    f.write(f"- 时间步长 dt = {dt} s (nt={nt})\n")
    f.write(f"- 热源 f(x,t) = 5·exp(-20(x-0.5)²)\n\n")
    
    f.write("## 条件\n\n")
    f.write("- 初始条件：T(x,0) = 0\n")
    f.write("- 边界条件：T(0,t) = 0, T(1,t) = 0\n\n")
    
    f.write("## 温度数据矩阵\n\n")
    f.write("| t\\x |")
    for i in range(0, nx, 5):
        f.write(f" {x[i]:.2f} |")
    f.write("\n|")
    f.write("-|" * (len(range(0, nx, 5)) + 1))
    f.write("\n")
    
    for n in range(0, nt, max(1, nt//15)):
        f.write(f"| {t[n]:.2f} |")
        for i in range(0, nx, 5):
            f.write(f" {u[n,i]:.4f} |")
        f.write("\n")
    
    f.write("\n## 中心点温度演化\n\n")
    f.write("| Time (s) | T(0.5, t) |\n")
    f.write("|----------|----------|\n")
    center = nx // 2
    for n in range(0, nt, max(1, nt//20)):
        f.write(f"| {t[n]:.2f} | {u[n, center]:.6f} |\n")
    
    f.write(f"\n---\n*Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n")

print(f"数据：{md_file}")
print("\n=== 完成 ===")
