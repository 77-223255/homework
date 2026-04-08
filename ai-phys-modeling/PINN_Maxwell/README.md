# PINN for 1D Maxwell Equations

一维电磁波的物理信息神经网络（PINN）求解器。

## 功能

求解一维麦克斯韦方程组：
- ∂E/∂x = μ₀ ∂H/∂t
- ∂H/∂x = ε₀ ∂E/∂t

初始条件：E(x,0) = sin(kx), H(x,0) = 0  
边界条件：E(0,t) = E(1,t) = 0

## 输出文件

| 文件 | 说明 |
|------|------|
| `pinn_maxwell.py` | 源代码 |
| `pinn_maxwell_model.pth` | 训练好的模型权重 |
| `results.png` | PINN 预测 vs 解析解对比 |
| `training_loss.png` | 训练损失曲线 |
| `fields_evolution.png` | 电磁场时空演化图 |

## 环境配置

```bash
# 创建 conda 环境
conda create -n pinn_maxwell python=3.10
conda activate pinn_maxwell

# 安装依赖
pip install torch numpy matplotlib
```

## 运行

```bash
cd PINN_Maxwell_Fixed
python pinn_maxwell.py
```

训练约需 4 分钟（10000 轮），完成后自动生成结果图像。
