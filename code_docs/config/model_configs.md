# 模型配置 Model Configs

> 🤖 **各类机器学习和深度学习模型的参数配置**

## 📋 功能说明

`model_configs.yaml` 定义了项目支持的所有模型的超参数配置，从传统机器学习到最新的大语言模型。

## 🎯 主要配置块

### traditional_models - 传统机器学习
- **svm**: 支持向量机参数 (C, kernel, gamma)
- **random_forest**: 随机森林参数 (n_estimators, max_depth)
- **naive_bayes**: 朴素贝叶斯参数
- **logistic_regression**: 逻辑回归参数

### neural_networks - 基础神经网络
- **textcnn**: 文本CNN配置 (filter_sizes, num_filters)
- **bilstm**: 双向LSTM配置 (hidden_dim, num_layers)
- **transformer_base**: 基础Transformer配置

### pretrained_models - 预训练模型
- **BERT系列**: bert, roberta, albert, electra, deberta
- **中文模型**: chinese-bert-wwm, chinese-roberta-wwm, macbert
- **多语言模型**: multilingual-bert, xlm-roberta

### multimodal_models - 多模态模型
- **CLIP系列**: clip-vit-b32, chinese-clip
- **BLIP系列**: blip-base 配置
- **融合策略**: 模态融合方法配置

### graph_neural_networks - 图神经网络
- **基础GNN**: GCN, GAT, GraphSAGE, GIN
- **高级模型**: Graph Transformer, GraphBERT
- **图池化**: 各种池化层配置

### large_language_models - 大语言模型
- **ChatGLM**: chatglm2-6b 配置
- **Qwen**: qwen-7b-chat 配置
- **生成参数**: temperature, top_p, max_tokens

### parameter_efficient_finetuning - 参数高效微调
- **LoRA**: rank, alpha, dropout 配置
- **AdaLoRA**: 自适应rank配置
- **P-Tuning**: prefix length 配置

## 💡 使用场景

- 选择和配置模型架构
- 设置模型超参数
- 配置预训练模型路径
- 调整微调策略参数
- 多模态融合配置

---

**[⬅️ 数据配置](data_configs.md) | [训练配置 ➡️](training_configs.md)**
