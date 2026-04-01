# 有限差分法求解一维热传导方程

## 控制方程

$$\frac{\partial T}{\partial t} = \alpha \frac{\partial^2 T}{\partial x^2} + f(x,t)$$

## 数值方法

**格式**: 显式 FTCS (Forward Time Central Space)

**离散格式**:
$$T_i^{n+1} = T_i^n + r(T_{i+1}^n - 2T_i^n + T_{i-1}^n) + \Delta t \cdot f(x_i, t_n)$$

其中 $r = \alpha \Delta t / \Delta x^2$ 为稳定性参数。

## 参数设置

| 参数 | 值 |
|------|-----|
| 空间步长 Δx | 0.0204 m (nx=50) |
| 时间步长 Δt | 0.01 s (nt=101) |
| 稳定性参数 r | 0.2401 (< 0.5, 稳定) |

## 稳定性条件

CFL 条件：$r = \frac{\alpha \Delta t}{\Delta x^2} \leq 0.5$

本例中 r = 0.2401，满足稳定性要求。

## 输出文件

| 文件 | 说明 |
|------|------|
| `heat_conduction_1d.py` | 求解代码 |
| `heat_result_*.png` | 温度分布图 (3D 表面、等高线、剖面、演化曲线) |
| `heat_data_*.md` | 温度数据表 |

## 运行

```bash
python3 heat_conduction_1d.py
```

---
*方法：经典显式有限差分，一阶时间精度，二阶空间精度*
