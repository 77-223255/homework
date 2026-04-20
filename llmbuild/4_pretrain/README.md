# 4_pretrain

## 用途
GPT 模型预训练模块：模型训练、权重加载与文本生成。

## 文件说明
| 文件 | 说明 |
|------|------|
| `pretrain.py` | GPT 模型预训练脚本：训练循环、损失计算、评估与生成 |
| `gpt_download.py` | 下载 OpenAI GPT-2 预训练权重（124M/355M/774M/1558M） |
| `load.py` | 加载已保存的模型权重并生成文本 |
| `loadgpt2.py` | 将 OpenAI GPT-2 权重加载到自定义 GPTModel |
| `tiktokenizer_pack.py` | 数据加载器：GPTDatasetV1 与 create_dataloader_v1 |
| `GPT_main_structure_pack.py` | GPT 模型主结构（来自 3_GPT_Start） |
| `self_attention_pack.py` | 注意力机制模块（来自 3_GPT_Start） |
| `novel_cn.txt` | 中文小说训练数据 |
| `the-verdict.txt` | 英文测试文本（Edith Wharton 短篇小说） |

## 运行
```bash
# 预训练模型
python pretrain.py

# 加载本地模型权重生成文本
python load.py

# 加载 OpenAI GPT-2 权重生成文本
python loadgpt2.py
```

## 核心概念
- **预训练（Pretraining）**：在大规模文本上训练语言模型
- **损失函数**：Cross-Entropy Loss，衡量预测与真实 Token 的差距
- **权重加载**：加载预训练权重到模型（本地/OpenAI GPT-2）
- **文本生成**：Temperature、Top-K Sampling 控制生成多样性
- **Checkpoint**：保存 model_state_dict + optimizer_state_dict