#!/usr/bin/env python3
"""极简 MNIST GAN - 生成手写数字"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
import os
import shutil

# 配置
NOISE_DIM = 100
IMG_SIZE = 28
BATCH_SIZE = 64
EPOCHS = 100
LR = 0.0002
SAMPLE_DIR = "samples"
EQUILIBRIUM_WINDOW = 5
EQUILIBRIUM_THRESHOLD = 0.05

# 清理旧样本
if os.path.exists(SAMPLE_DIR):
    shutil.rmtree(SAMPLE_DIR)
os.makedirs(SAMPLE_DIR, exist_ok=True)

# 数据加载
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])  # [0,1] -> [-1,1]
])
train_dataset = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

# Generator
class G(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(NOISE_DIM, 128), nn.LeakyReLU(0.2),
            nn.Linear(128, 256), nn.LeakyReLU(0.2),
            nn.Linear(256, 512), nn.LeakyReLU(0.2),
            nn.Linear(512, IMG_SIZE * IMG_SIZE), nn.Tanh()
        )
    def forward(self, x):
        return self.net(x).view(-1, 1, IMG_SIZE, IMG_SIZE)

# Discriminator
class D(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(IMG_SIZE * IMG_SIZE, 256), nn.LeakyReLU(0.2),
            nn.Linear(256, 128), nn.LeakyReLU(0.2),
            nn.Linear(128, 1), nn.Sigmoid()
        )
    def forward(self, x):
        return self.net(x)

# 初始化
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
G_net = G().to(device)
D_net = D().to(device)
criterion = nn.BCELoss()
opt_g = optim.Adam(G_net.parameters(), lr=LR)
opt_d = optim.Adam(D_net.parameters(), lr=LR)

def save_samples(epoch, noise):
    G_net.eval()
    with torch.no_grad():
        fake = G_net(noise).cpu().numpy()
    fig, axes = plt.subplots(4, 4, figsize=(4, 4))
    for i, ax in enumerate(axes.flat):
        ax.imshow(fake[i, 0], cmap='gray')
        ax.axis('off')
    plt.tight_layout()
    if isinstance(epoch, int):
        plt.savefig(f"{SAMPLE_DIR}/epoch_{epoch:03d}.png")
    else:
        plt.savefig(f"{SAMPLE_DIR}/{epoch}.png")
    plt.close()
    G_net.train()

# 训练
print(f"设备：{device}\n开始训练 {EPOCHS} epochs...")
print(f"提前结束：{EQUILIBRIUM_WINDOW} 轮内损失变化 < {EQUILIBRIUM_THRESHOLD} 视为纳什均衡\n")
fixed_noise = torch.randn(16, NOISE_DIM, device=device)

loss_history = []
best_epoch = 0
early_stop = False
d_std, g_std = 0.0, 0.0

for epoch in range(EPOCHS):
    for i, (real, _) in enumerate(train_loader):
        real = real.to(device)
        bs = real.size(0)
        
        # 训练 D
        real_label = torch.ones(bs, 1, device=device)
        fake_label = torch.zeros(bs, 1, device=device)
        
        opt_d.zero_grad()
        d_real = D_net(real)
        loss_d_real = criterion(d_real, real_label)
        
        noise = torch.randn(bs, NOISE_DIM, device=device)
        fake = G_net(noise).detach()
        d_fake = D_net(fake)
        loss_d_fake = criterion(d_fake, fake_label)
        
        loss_d = loss_d_real + loss_d_fake
        loss_d.backward()
        opt_d.step()
        
        # 训练 G
        opt_g.zero_grad()
        noise = torch.randn(bs, NOISE_DIM, device=device)
        fake = G_net(noise)
        d_fake = D_net(fake)
        loss_g = criterion(d_fake, real_label)
        loss_g.backward()
        opt_g.step()
    
    d_val, g_val = loss_d.item(), loss_g.item()
    loss_history.append((d_val, g_val))
    
    # 纳什均衡检测
    if len(loss_history) >= EQUILIBRIUM_WINDOW:
        recent = loss_history[-EQUILIBRIUM_WINDOW:]
        d_std = np.std([x[0] for x in recent])
        g_std = np.std([x[1] for x in recent])
        if d_std < EQUILIBRIUM_THRESHOLD and g_std < EQUILIBRIUM_THRESHOLD:
            print(f"\n⚡ 纳什均衡达成！D/G 损失稳定 {EQUILIBRIUM_WINDOW} 轮")
            early_stop = True
    
    status = "⚡ EQ" if early_stop else ""
    print(f"Epoch {epoch+1:3d} | D: {d_val:.4f} | G: {g_val:.4f} | σD: {d_std:.4f} | σG: {g_std:.4f} {status}")
    
    if (epoch + 1) % 5 == 0 or epoch == 0:
        save_samples(epoch + 1, fixed_noise)
        best_epoch = epoch + 1
    
    if early_stop:
        print(f"\n提前结束于 epoch {epoch+1}")
        break
    
# 保存模型
torch.save(G_net.state_dict(), "generator.pth")
torch.save(D_net.state_dict(), "discriminator.pth")
print("模型已保存：generator.pth, discriminator.pth")

# 最终生成 16 张图
print("\n训练完成！生成最终样本...")
save_samples("final", torch.randn(16, NOISE_DIM, device=device))
print(f"样本已保存到 {SAMPLE_DIR}/")

# 显示最终结果
G_net.eval()
with torch.no_grad():
    noise = torch.randn(16, NOISE_DIM, device=device)
    fake = G_net(noise).cpu().numpy()

fig, axes = plt.subplots(4, 4, figsize=(6, 6))
for i, ax in enumerate(axes.flat):
    ax.imshow(fake[i, 0], cmap='gray')
    ax.axis('off')
plt.suptitle("GAN 生成的手写数字", fontsize=14)
plt.tight_layout()
plt.savefig(f"{SAMPLE_DIR}/final_display.png", dpi=150)
plt.show()
