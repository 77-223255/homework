# 1_input_data_process

## 用途
LLM 数据预处理模块：文本分词与 Token ID 转换。

## 文件说明
| 文件 | 说明 |
|------|------|
| <span style="color: green">`simple_text_tokenizer.py`</span> | 从零实现的简单分词器（正则分割 + 词表查找） |
| <span style="color: green">`tiktokenizer.py`</span> | 基于 OpenAI tiktoken 的 GPT-2 分词器示例 |
| <span style="color: blue">`the-verdict.txt`</span> | 测试文本（Edith Wharton 短篇小说） |