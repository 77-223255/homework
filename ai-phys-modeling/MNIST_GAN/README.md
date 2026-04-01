# MNIST GAN - 极简手写数字生成

## 快速开始

```bash
# 1. 创建 conda 环境
cd ~/Desktop/MNIST_GAN
conda env create -f environment.yml

# 2. 激活环境
conda activate mnist-gan

# 3. 运行训练
python train.py
```

## 输出

- `samples/epoch_XXX.png` - 每 5 个 epoch 保存的生成样本
- `samples/final_display.png` - 最终 16 张生成图像

## 自定义

| 参数 | 位置 | 默认值 | 说明 |
|------|------|--------|------|
| 噪声维度 | `NOISE_DIM` | 100 | 输入随机向量长度 |
| 训练轮数 | `EPOCHS` | 20 | 增加可提高质量 |
| 学习率 | `LR` | 0.0002 | Adam 优化器学习率 |
| 批次大小 | `BATCH_SIZE` | 64 | 每批样本数 |

## 生成新图像

训练完成后，在 Python 中：

```python
conda activate mnist-gan
python -c "
import torch
from train import G_net, save_samples
G_net.eval()
noise = torch.randn(16, 100)  # 新随机噪声
with torch.no_grad():
    fake = G_net(noise).numpy()
# fake 形状：(16, 1, 28, 28)，值域 [-1, 1]
"
```

## 预期效果

- **Epoch 1-5**: 随机噪声，无结构
- **Epoch 10-15**: 出现模糊的数字轮廓
- **Epoch 20+**: 可辨认的手写数字（0-9）

## 清理

```bash
conda deactivate
conda env remove -n mnist-gan
rm -rf data samples
```
