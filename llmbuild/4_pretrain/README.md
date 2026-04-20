# 4_pretrain

## 用途
GPT 模型预训练模块：模型训练、权重加载与文本生成。

## 文件说明
| 文件 | 说明 |
|------|------|
| <span style="color: green">`pretrain.py`</span> | GPT 模型预训练脚本：训练循环、损失计算、评估与生成 |
| <span style="color: green">`load.py`</span> | 加载已保存的模型权重并生成文本 |
| <span style="color: green">`loadgpt2.py`</span> | 将 OpenAI GPT-2 权重加载到自定义 GPTModel |
| <span style="color: orange">`gpt_download.py`</span> | 下载 OpenAI GPT-2 预训练权重（124M/355M/774M/1558M） |
| <span style="color: orange">`tiktokenizer_pack.py`</span> | 数据加载器：GPTDatasetV1 与 create_dataloader_v1 |
| <span style="color: orange">`GPT_main_structure_pack.py`</span> | GPT 模型主结构（来自 3_GPT_Start） |
| <span style="color: orange">`self_attention_pack.py`</span> | 注意力机制模块（来自 3_GPT_Start） |
| <span style="color: blue">`novel_cn.txt`</span> | 中文小说训练数据 |
| <span style="color: blue">`the-verdict.txt`</span> | 英文测试文本（Edith Wharton 短篇小说） |