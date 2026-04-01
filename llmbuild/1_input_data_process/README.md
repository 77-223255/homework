# 1_input_data_process

## 用途
LLM 数据预处理模块：文本分词与 Token ID 转换。

## 文件说明
| 文件 | 说明 |
|------|------|
| `simple_text_tokenizer.py` | 从零实现的简单分词器（正则分割 + 词表查找） |
| `tiktokenizer.py` | 基于 OpenAI tiktoken 的 GPT-2 分词器示例 |
| `the-verdict.txt` | 测试文本（Edith Wharton 短篇小说） |

## 运行
```bash
python simple_text_tokenizer.py
python tiktokenizer.py
```

## 核心概念
- **分词（Tokenization）**：文本 → 离散 Token ID
- **词表（Vocabulary）**：Token 到整数的映射表
- **Unknown Token**：处理词表外词汇的兜底策略
