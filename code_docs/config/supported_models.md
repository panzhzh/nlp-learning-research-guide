# 支持模型列表 Supported Models

> 📋 **项目支持的所有模型及其兼容性信息**

## 📋 功能说明

`supported_models.yaml` 列出了项目中所有可用的模型，包括实现路径、支持任务、输入类型等详细信息。

## 🎯 主要配置块

### traditional_models - 传统机器学习模型
- **分类器列表**: SVM, RandomForest, NaiveBayes, LogisticRegression, XGBoost
- **实现信息**: sklearn/xgboost实现，类路径
- **支持任务**: 二分类、多分类
- **特征要求**: text_features

### neural_network_models - 神经网络模型
- **文本模型**: TextCNN, BiLSTM, TextRCNN, HierarchicalAttention
- **图像模型**: ImageCNN, ResNet系列
- **实现框架**: PyTorch
- **输入类型**: text, image

### pretrained_models - 预训练模型
- **文本编码器**:
  - English: BERT, RoBERTa, ALBERT, ELECTRA, DeBERTa
  - Chinese: Chinese-BERT-wwm, Chinese-RoBERTa-wwm, MacBERT, ERNIE
  - Multilingual: mBERT, XLM-RoBERTa
- **生成模型**: GPT-2, T5, BART
- **模型规格**: max_length, hidden_size等

### multimodal_models - 多模态模型
- **视觉-语言模型**: CLIP, Chinese-CLIP, BLIP, ALBEF, FLAVA
- **输入类型**: text + image
- **图像尺寸**: 224x224, 384x384等
- **文本长度**: 77, 512等限制

### graph_models - 图神经网络模型
- **基础GNN**: GCN, GAT, GraphSAGE, GIN
- **高级GNN**: GraphTransformer, GraphBERT
- **实现框架**: torch_geometric
- **支持任务**: node_classification, graph_classification

### large_language_models - 大语言模型
- **开源LLM**: ChatGLM2-6B, Qwen-7B-Chat, Baichuan2-7B-Chat
- **多模态LLM**: LLaVA, BLIP-2
- **模型规格**: 参数量, max_length
- **支持任务**: chat, classification, VQA

### specialized_models - 专用模型
- **谣言检测**: RumorDetectionTransformer
- **社交媒体**: SocialMediaBERT
- **领域专用**: 针对特定任务优化

### ensemble_models - 集成模型
- **投票集成**: VotingClassifier
- **堆叠集成**: StackingClassifier  
- **多模态集成**: MultiModalEnsemble
- **支持基模型**: 传统、神经网络、预训练模型

## 🔍 兼容性信息

### compatibility_matrix - 兼容性矩阵
- **任务兼容**: classification, regression, generation, retrieval
- **输入类型**: text_only, image_only, multimodal, graph
- **语言支持**: english, chinese, mixed

### recommended_models - 推荐模型
- **初学者**: LogisticRegression, BERT, CLIP
- **中级用户**: RoBERTa, Chinese-BERT-wwm, BLIP, GAT
- **高级用户**: DeBERTa, ChatGLM2-6B, LLaVA, GraphTransformer

### performance_benchmarks - 性能基准
在MR2数据集上的预期性能:
- **传统方法**: SVM (65%), RandomForest (68%)
- **预训练模型**: BERT (78%), RoBERTa (79%), Chinese-BERT-wwm (80%)
- **多模态**: CLIP (82%), BLIP (84%), Chinese-CLIP (83%)

## 💡 使用场景

- 查看支持的模型列表
- 选择适合任务的模型
- 了解模型兼容性和限制
- 查看性能基准参考
- 获取模型实现路径

---

**[⬅️ 训练配置](training_configs.md) | [RAG配置 ➡️](rag_configs.md)**
