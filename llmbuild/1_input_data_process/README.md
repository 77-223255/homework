# 1_input_data_process

## 用途
LLM 数据预处理模块：文本分词与 Token ID 转换。

## 文件说明
🟢 用户运行文件  🟡 库文件  🔵 数据文件

| 文件 | 说明 |
|------|------|
| 🟢 `simple_text_tokenizer.py` | 从零实现的简单分词器（正则分割 + 词表查找） |
| 🟢 `tiktokenizer.py` | 基于 OpenAI tiktoken 的 GPT-2 分词器示例 |
| 🔵 `the-verdict.txt` | 测试文本（Edith Wharton 短篇小说） |