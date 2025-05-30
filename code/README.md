# NLP技术实现代码库

> 🚀 **模块化的NLP技术实现，支持文本、图像、图结构的多模态分析**

## 📖 项目概述

本代码库提供了NLP研究中常用技术的统一实现，采用模块化设计，支持多种模型变体。主要特点：

- 🔄 **统一接口**：相似架构的模型使用统一代码，通过配置切换
- 🖼️ **多模态支持**：文本+图像+图结构的联合建模
- 🌏 **多语言**：中英文双语支持
- 📊 **图神经网络**：社交网络分析和图结构学习
- 🤗 **预训练模型**：集成主流预训练模型
- ⚡ **高效训练**：支持分布式训练和参数高效微调

## 🗂️ 目录结构

```
code/
├── README.md                           # 代码总体说明
├── requirements.txt                    # 依赖包列表
├── environment.yml                     # conda环境配置
├── setup.py                           # 包安装配置
├── config/                            # 🔧 配置文件
│   ├── model_configs.yaml             # 模型配置(支持多种变体)
│   ├── training_configs.yaml
│   ├── data_configs.yaml
│   └── supported_models.yaml          # 支持的模型列表
├── data/                              # 📚 数据集目录
│   ├── raw/                           # 原始数据
│   │   ├── train/
│   │   ├── val/
│   │   └── test/
│   ├── processed/                     # 处理后的数据
│   │   ├── train_features.pkl
│   │   ├── val_features.pkl
│   │   └── test_features.pkl
│   ├── dataset_items_train.json      # 训练集数据项
│   ├── dataset_items_val.json        # 验证集数据项
│   ├── dataset_items_test.json       # 测试集数据项
│   └── README.md                     # 数据集说明文档
├── preprocessing/                      # 📝 预处理
│   ├── __init__.py
│   ├── text_processing.py             # 文本处理(分词、清洗等)
│   ├── image_processing.py            # 🖼️ 图像处理
│   ├── graph_construction.py          # 📊 图构建
│   └── data_augmentation.py           # 数据增强
├── models/                            # 🤖 模型实现
│   ├── __init__.py
│   ├── traditional/                   # 传统方法
│   │   ├── ml_classifiers.py          # SVM/NB/RF等统一接口
│   │   └── feature_engineering.py
│   ├── neural_networks/               # 基础神经网络
│   │   ├── cnn_models.py              # CNN架构
│   │   ├── rnn_models.py              # RNN/LSTM/GRU
│   │   ├── attention_models.py        # 注意力机制
│   │   └── transformer_base.py        # 基础Transformer
│   ├── pretrained/                    # 🤗 预训练模型
│   │   ├── encoder_models.py          # BERT类(BERT/RoBERTa/ALBERT/ELECTRA/DeBERTa)
│   │   ├── decoder_models.py          # GPT类(GPT/GPT-2/GPT-Neo)
│   │   ├── encoder_decoder_models.py  # T5类(T5/mT5/UmT5)
│   │   ├── chinese_models.py          # 中文模型统一接口
│   │   └── multilingual_models.py     # 多语言模型统一接口
│   ├── multimodal/                    # 🖼️🔤 多模态模型
│   │   ├── vision_language_models.py  # CLIP/BLIP/ALBEF等
│   │   ├── fusion_strategies.py       # 各种融合方法
│   │   ├── chinese_multimodal.py      # 中文多模态
│   │   └── social_media_models.py     # 社交媒体特化模型
│   ├── graph_neural_networks/         # 📊 图神经网络
│   │   ├── basic_gnn_layers.py        # GCN/GAT/GraphSAGE/GIN层
│   │   ├── advanced_gnn_models.py     # Graph Transformer/GraphBERT
│   │   ├── heterogeneous_gnn.py       # 异构图神经网络
│   │   ├── temporal_gnn.py            # 时序图神经网络
│   │   └── multimodal_gnn.py          # 多模态GNN
│   └── llms/                          # 🚀 大语言模型
│       ├── open_source_llms.py        # 开源LLM(LLaMA/ChatGLM/Baichuan/Qwen)
│       ├── multimodal_llms.py         # 多模态LLM(LLaVA/BLIP-2)
│       └── prompt_engineering.py      # 提示工程
├── embeddings/                        # 📐 嵌入方法
│   ├── __init__.py
│   ├── word_embeddings.py             # Word2Vec/GloVe/FastText
│   ├── sentence_embeddings.py         # SentenceBERT/SimCSE等
│   ├── image_embeddings.py            # 图像特征提取
│   └── multimodal_embeddings.py       # 多模态嵌入
├── rag/                               # 🔍 RAG系统
│   ├── __init__.py
│   ├── retrievers.py                  # 各种检索器
│   ├── generators.py                  # 各种生成器
│   ├── vector_stores.py               # 向量数据库
│   └── multimodal_rag.py              # 多模态RAG
├── training/                          # 🏋️ 训练框架
│   ├── __init__.py
│   ├── base_trainer.py                # 基础训练器
│   ├── task_trainers.py               # 任务特定训练器
│   ├── distributed_training.py        # 分布式训练
│   ├── fine_tuning_methods.py         # 微调方法(LoRA/P-tuning等)
│   └── loss_functions.py              # 各种损失函数
├── evaluation/                        # 📊 评估
│   ├── __init__.py
│   ├── metrics.py                     # 各种评估指标
│   ├── statistical_tests.py           # 统计检验
│   └── visualization.py               # 结果可视化
├── utils/                             # 🛠️ 工具
│   ├── __init__.py
│   ├── data_utils.py                  # 数据处理工具
│   ├── model_utils.py                 # 模型工具
│   ├── file_utils.py                  # 文件操作
│   ├── logging_utils.py               # 日志工具
│   └── experiment_tracking.py         # 实验跟踪
├── datasets/                          # 📚 数据集处理类
│   ├── __init__.py
│   ├── base_dataset.py                # 基础数据集类
│   ├── multimodal_dataset.py          # 多模态数据集处理
│   ├── graph_dataset.py               # 图数据集处理
│   └── data_loaders.py                # 数据加载器
├── examples/                          # 📝 使用示例
│   ├── quick_start.py                 # 快速开始
│   ├── text_classification_demo.py    # 文本分类示例
│   ├── multimodal_analysis_demo.py    # 多模态分析示例
│   ├── graph_analysis_demo.py         # 图分析示例
│   └── tutorials/                     # Jupyter教程
│       ├── 01_getting_started.ipynb
│       ├── 02_text_models.ipynb
│       ├── 03_multimodal_models.ipynb
│       ├── 04_graph_models.ipynb
│       └── 05_advanced_techniques.ipynb
├── tests/                             # 🧪 测试
│   └── test_*.py
└── scripts/                           # 📜 脚本
    ├── setup_environment.py           # 环境设置
    ├── download_models.py             # 下载模型
    ├── prepare_dataset.py             # 数据集预处理
    └── run_experiments.py             # 运行实验
```

### 📝 数据预处理 (`preprocessing/`)
- `text_processing.py` - 文本处理（分词、清洗、增强）
- `image_processing.py` - 图像预处理和特征提取
- `graph_construction.py` - 社交图构建和图特征工程

### 🤖 模型实现 (`models/`)

#### 传统方法 (`traditional/`)
- `ml_classifiers.py` - 机器学习分类器统一接口
  ```python
  # 支持: SVM, RandomForest, NaiveBayes, LogisticRegression
  classifier = MLClassifier(method='svm', **params)
  ```

#### 预训练模型 (`pretrained/`)
- `encoder_models.py` - 编码器模型统一接口
  ```python
  # 支持: bert-base, roberta-base, albert-base, electra-base, deberta-base
  # 中文: chinese-bert-wwm, chinese-roberta-wwm, macbert
  model = EncoderModel(model_name='bert-base-uncased', num_classes=3)
  ```

- `decoder_models.py` - 解码器模型
  ```python
  # 支持: gpt2, gpt-neo, gpt-j
  model = DecoderModel(model_name='gpt2', task='generation')
  ```

- `encoder_decoder_models.py` - 编码器-解码器模型
  ```python
  # 支持: t5-base, mt5-base, umt5-base
  model = EncoderDecoderModel(model_name='t5-base')
  ```

#### 多模态模型 (`multimodal/`)
- `vision_language_models.py` - 视觉-语言模型
  ```python
  # 支持: clip, blip, albef, flava
  # 中文: chinese-clip, wenlan
  model = VisionLanguageModel(model_name='clip', fusion_method='attention')
  ```

#### 图神经网络 (`graph_neural_networks/`)
- `basic_gnn_layers.py` - 基础GNN层
  ```python
  # 支持: GCN, GAT, GraphSAGE, GIN
  gnn = BasicGNN(layer_type='gcn', hidden_dim=128, num_layers=2)
  ```

- `multimodal_gnn.py` - 多模态图神经网络
  ```python
  # 结合文本、图像、图结构
  model = MultimodalGNN(text_encoder='bert', image_encoder='resnet', gnn_type='gat')
  ```

#### 大语言模型 (`llms/`)
- `open_source_llms.py` - 开源大语言模型
  ```python
  # 支持: llama, llama2, chatglm, chatglm2, baichuan, qwen
  llm = OpenSourceLLM(model_name='chatglm2-6b', task='chat')
  ```

### 🏋️ 训练框架 (`training/`)
- `base_trainer.py` - 统一训练接口
- `fine_tuning_methods.py` - 参数高效微调
  ```python
  # 支持: LoRA, AdaLoRA, P-Tuning, Prefix-Tuning
  trainer = Trainer(model=model, fine_tuning_method='lora', lora_rank=16)
  ```

## 🚀 快速开始

### 环境设置

```bash
# 克隆仓库
git clone <repository-url>
cd code

# 创建环境
conda env create -f environment.yml
conda activate nlp-toolkit

# 或使用pip
pip install -r requirements.txt
```

### 基础使用

#### 1. 文本分类
```python
from models.pretrained.encoder_models import EncoderModel
from datasets.mr2_dataset import MR2Dataset
from training.base_trainer import Trainer

# 加载数据
dataset = MR2Dataset(data_dir='../datasets/MR2')

# 创建模型 (支持多种BERT变体)
model = EncoderModel(
    model_name='bert-base-uncased',  # 可选: roberta, albert, electra, deberta
    num_classes=3
)

# 训练
trainer = Trainer(model=model, dataset=dataset)
trainer.train()
```

#### 2. 多模态分析
```python
from models.multimodal.vision_language_models import VisionLanguageModel

# 多模态模型
model = VisionLanguageModel(
    model_name='clip',  # 可选: blip, albef, chinese-clip
    fusion_method='attention',
    num_classes=3
)

# 处理文本+图像数据
results = model(text_inputs, image_inputs)
```

#### 3. 图神经网络
```python
from models.graph_neural_networks.basic_gnn_layers import BasicGNN
from preprocessing.graph_construction import SocialGraphBuilder

# 构建社交图
graph_builder = SocialGraphBuilder()
graph = graph_builder.build_user_post_graph(dataset)

# GNN模型
gnn = BasicGNN(
    layer_type='gat',  # 可选: gcn, graphsage, gin
    input_dim=768,
    hidden_dim=128,
    num_classes=3
)
```

#### 4. 大语言模型
```python
from models.llms.open_source_llms import OpenSourceLLM

# LLM推理
llm = OpenSourceLLM(
    model_name='chatglm2-6b',  # 可选: llama2, baichuan, qwen
    task='classification'
)

results = llm.predict(texts, prompt_template="判断以下文本的情感: {text}")
```

## 📊 支持的模型

### 文本编码器
| 模型系列 | 具体模型 | 配置名称 |
|---------|---------|---------|
| **BERT** | BERT, RoBERTa, ALBERT, ELECTRA, DeBERTa | `bert-base-uncased`, `roberta-base`, `albert-base-v2`, `electra-base`, `deberta-base` |
| **中文BERT** | Chinese-BERT-wwm, MacBERT, ERNIE | `chinese-bert-wwm`, `hfl/chinese-macbert-base`, `ernie-1.0` |

### 多模态模型
| 模型系列 | 具体模型 | 配置名称 |
|---------|---------|---------|
| **CLIP** | CLIP, Chinese-CLIP | `clip-vit-base-patch32`, `chinese-clip-vit-base-patch16` |
| **BLIP** | BLIP, BLIP-2 | `blip-base`, `blip2-opt-2.7b` |

### 图神经网络
| GNN类型 | 实现 | 特点 |
|---------|------|------|
| **GCN** | 图卷积网络 | 基础图卷积 |
| **GAT** | 图注意力网络 | 注意力机制 |
| **GraphSAGE** | 图采样聚合 | 大图扩展性 |
| **GIN** | 图同构网络 | 理论保证 |

### 大语言模型
| 模型系列 | 具体模型 | 参数量 |
|---------|---------|--------|
| **LLaMA** | LLaMA, LLaMA-2 | 7B-70B |
| **ChatGLM** | ChatGLM, ChatGLM2 | 6B-130B |
| **百川** | Baichuan, Baichuan2 | 7B-13B |
| **通义千问** | Qwen, Qwen-Chat | 7B-72B |

## 🔧 配置管理

所有模型配置都在 `config/model_configs.yaml` 中管理：

```yaml
# BERT系列配置
bert_models:
  bert-base-uncased:
    model_type: "encoder"
    hidden_size: 768
    num_attention_heads: 12
    num_hidden_layers: 12
  
  roberta-base:
    model_type: "encoder" 
    hidden_size: 768
    num_attention_heads: 12
    num_hidden_layers: 12

# 多模态模型配置
multimodal_models:
  clip:
    text_encoder: "clip-text"
    image_encoder: "clip-vision"
    projection_dim: 512
```

## 📚 数据集

当前支持的数据集：
- **MR2数据集**: 多模态情感分析数据集
  - 训练集: 500条
  - 验证集: 300条  
  - 测试集: 100条
  - 包含: 文本 + 图像 + 用户社交图

## 🧪 实验示例

查看 `examples/` 目录下的完整示例：

- `quick_start.py` - 5分钟快速上手
- `text_classification_demo.py` - 文本分类完整流程
- `multimodal_analysis_demo.py` - 多模态分析示例
- `graph_analysis_demo.py` - 社交网络图分析
- `tutorials/` - Jupyter教程合集

## 🤝 贡献指南

1. Fork 项目
2. 创建特性分支: `git checkout -b feature/new-model`
3. 提交更改: `git commit -m 'Add new model support'`
4. 推送分支: `git push origin feature/new-model`
5. 创建 Pull Request

## 📄 许可证

本项目采用 MIT 许可证 - 查看 [LICENSE](LICENSE) 文件了解详情。

## 📧 联系方式

- 项目维护者: [你的名字]
- 邮箱: [你的邮箱]
- 问题反馈: [GitHub Issues](issues链接)

---

**⭐ 如果这个项目对你有帮助，请给我们一个星标！**
```

这样的设计既减少了代码冗余，又保持了灵活性。通过统一接口和配置文件，用户可以轻松切换不同的模型变体，而不需要修改核心代码。