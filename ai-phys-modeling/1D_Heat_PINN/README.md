# 一维热传导方程数值求解

## 问题描述

求解带热源的一维热传导方程：

$$\frac{\partial T}{\partial t} = \alpha \frac{\partial^2 T}{\partial x^2} + f(x,t)$$

**参数**:
- 热扩散系数 α = 0.01 m²/s
- 空间域 x ∈ [0, 1] m
- 时间域 t ∈ [0, 1] s
- 热源项 f(x,t) = 5·exp(-20(x-0.5)²)

**定解条件**:
- 初始条件：T(x,0) = 0
- 边界条件：T(0,t) = 0, T(1,t) = 0

## 求解方法

| 方法 | 子文件夹 | 说明 |
|------|----------|------|
| 有限差分法 | `差分一维热传导/` | 显式 FTCS 格式，dt=0.01s, nx=50 |
| 物理信息神经网络 | `神经网络一维热传导/` | 4 层×64 神经元，15000 轮训练 |

## 文件结构

```
Homework/
├── README.md                    # 本文件
├── pyproject.toml               # Python 环境配置
├── comparison_summary_*.png     # 9 子图汇总对比图
├── 差分一维热传导/
│   ├── README.md
│   ├── heat_conduction_1d.py
│   ├── heat_result_*.png
│   └── heat_data_*.md
└── 神经网络一维热传导/
    ├── README.md
    ├── heat_nn_solver.py
    ├── pinn_*.png
    ├── comparison_*.png
    └── pinn_data_*.md
```

## 汇总图说明

`comparison_summary_*.png` 包含 9 个子图：

| 子图 | 内容 |
|------|------|
| (A) | 有限差分解 - 3D 表面 |
| (B) | PINN 解 - 3D 表面 |
| (C) | 绝对误差 - 3D 光滑曲面 (样条插值) |
| (D) | 有限差分 - 等高线 |
| (E) | PINN - 等高线 |
| (F) | 相对误差热力图 (%) |
| (G) | 中心点温度演化 T(x=0.5, t) |
| (H) | t=1.0s 时刻温度剖面 |
| (I) | 误差统计柱状图 |

## 主要结果

**中心点温度 T(0.5, 1.0)**:
- 有限差分：4.2696
- PINN: 4.2632
- 相对误差：0.15%

**误差统计**:
- 最大绝对误差：0.0079
- 平均绝对误差：0.0016
- 平均相对误差：1.94%

## 环境配置

详见 `pyproject.toml`，或手动安装：

```bash
# 有限差分方法
pip install numpy matplotlib

# PINN 方法
pip install numpy matplotlib torch

# 生成汇总图 (需要 scipy 插值)
pip install numpy matplotlib torch scipy
```

---
*生成时间：2026-03-26*
