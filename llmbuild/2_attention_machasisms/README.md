# 2_attention_machasisms

## 用途
LLM 注意力机制模块：Self-Attention 与 Multi-Head Attention 实现。

## 文件说明
| 文件 | 说明 |
|------|------|
| `self_attention.py` | Self-Attention、Causal Attention、Multi-Head Attention 的完整实现 |

## 运行
```bash
python self_attention.py
```

## 核心概念
- **Self-Attention**：序列内部各位置的自注意力计算
- **Causal Attention**：因果掩码，防止未来信息泄露
- **Multi-Head Attention**：多头并行注意力，增强表达能力
- **Query-Key-Value**：注意力机制的三大核心组件