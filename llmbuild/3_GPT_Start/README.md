# 3_GPT_Start

## 用途
GPT 模型架构实现：Transformer Block、GPTModel 及文本生成。

## 文件说明
| 文件 | 说明 |
|------|------|
| `GPT_main_structure.py` | GPT 模型主结构：LayerNorm、GELU、FeedForward、TransformerBlock、GPTModel 及文本生成函数 |
| `self_attention_pack.py` | 注意力机制模块：SelfAttention、CausalAttention、MultiHeadAttention 实现 |

## 运行
```bash
python GPT_main_structure.py
```

## 核心概念
- **GPT 架构**：Token Embedding + Position Embedding + Transformer Blocks
- **LayerNorm**：层归一化，稳定训练
- **GELU**：高斯误差线性单元激活函数
- **Transformer Block**：Multi-Head Attention + FeedForward + Residual Connection
- **文本生成**：基于模型输出的自回归 Token 预测