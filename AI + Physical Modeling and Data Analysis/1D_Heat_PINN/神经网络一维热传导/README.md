# 物理信息神经网络 (PINN) 求解一维热传导方程

## 控制方程

$$\frac{\partial T}{\partial t} = \alpha \frac{\partial^2 T}{\partial x^2} + f(x,t)$$

## 方法原理

PINN 将 PDE 残差嵌入损失函数，通过自动微分计算导数：

**损失函数**:
$$\mathcal{L} = \mathcal{L}_{PDE} + \lambda_{IC}\mathcal{L}_{IC} + \lambda_{BC}\mathcal{L}_{BC}$$

其中：
- $\mathcal{L}_{PDE}$: PDE 残差均方误差
- $\mathcal{L}_{IC}$: 初始条件误差
- $\mathcal{L}_{BC}$: 边界条件误差

## 网络架构

| 层 | 结构 |
|----|------|
| 输入 | (x/L, t/T) 归一化坐标 |
| 隐藏层 | 4 层 × 64 神经元 (Tanh 激活) |
| 输出 | 温度 T |

## 训练配置

| 参数 | 值 |
|------|-----|
| 优化器 | Adam (lr=0.002) |
| 学习率衰减 | StepLR (每 3000 轮 ×0.6) |
| 训练轮数 | 15000 |
| 配点数量 | 5000 (PDE) + 500 (IC) + 500 (BC) |

## 输出文件

| 文件 | 说明 |
|------|------|
| `heat_nn_solver.py` | 求解代码 |
| `pinn_*.png` | PINN 解图 |
| `comparison_*.png` | 与有限差分对比图 |
| `pinn_data_*.md` | 温度数据与误差表 |

## 运行

```bash
python3 heat_nn_solver.py
```

## 与有限差分对比

| 误差类型 | 值 |
|----------|-----|
| 最大绝对误差 | 0.0093 |
| 平均绝对误差 | 0.0016 |
| 平均相对误差 | 1.02% |

---
*方法：Physics-Informed Neural Network，无网格，自动微分*
