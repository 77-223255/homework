# LLM Build - 分词器学习项目

本项目是学习《Build a Large Language Model (From Scratch)》一书中分词器相关内容的代码实现。

## 项目结构

```
llmbuild/
├── README.md                  # 项目说明文档
├── simple_text_tokenizer.py   # 简单文本分词器实现
├── tiktokenizer.py            # 基于 tiktoken 库的分词器示例
└── the-verdict.txt            # 测试用文本文件
```

## 文件说明

### simple_text_tokenizer.py

自定义实现的简单文本分词器 `SimpleTokenizerV2`，主要功能：

- 使用正则表达式进行文本分割
- 支持标点符号分离
- 处理未知词（`<|unk|>`）
- 包含 `encode()` 和 `decode()` 方法

### tiktokenizer.py

使用 OpenAI 的 tiktoken 库进行分词的示例：

- 使用 GPT-2 编码器
- 演示 `encode()` 和 `decode()` 方法的基本用法

### the-verdict.txt

Edith Wharton 的短篇小说《The Verdict》，用作分词器的测试文本数据。

## 环境配置

本项目使用 conda 环境，依赖包括：

- Python 3.10
- tiktoken >= 0.5.1
- regex
- requests

创建环境：

```bash
conda env create -f environment.yml
conda activate llmbuild
```

## 运行示例

```bash
# 运行简单分词器
python simple_text_tokenizer.py

# 运行 tiktoken 分词器
python tiktokenizer.py
```

## 参考资料

- [LLMs-from-scratch](https://github.com/rasbt/LLMs-from-scratch) - Sebastian Raschka 的开源项目
- [tiktoken](https://github.com/openai/tiktoken) - OpenAI 的 BPE 分词器库