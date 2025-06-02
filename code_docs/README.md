# NLP技术实现代码库文档

> 🚀 **模块化的NLP技术实现，支持文本、图像、图结构的多模态分析**

## 📚 文档导航

### 🔧 配置模块
- [配置模块概览](code_docs/config/) - 配置文件管理和参数设置
- [数据配置](code_docs/config/data_configs.md) - 数据集路径、预处理参数
- [模型配置](code_docs/config/model_configs.md) - 各类模型的配置参数
- [训练配置](code_docs/config/training_configs.md) - 训练超参数和策略配置
- [支持模型列表](code_docs/config/supported_models.md) - 2024-2025年度推荐技术栈
- [RAG配置](code_docs/config/rag_configs.md) - 检索增强生成系统配置

### 📚 数据处理模块
- [数据工具概览](code_docs/data_utils/) - 数据加载、处理和分析工具
- [MR2数据集](code_docs/data_utils/mr2_dataset.md) - 多模态谣言检测数据集类
- [数据加载器](code_docs/data_utils/data_loaders.md) - PyTorch数据加载器配置
- [数据分析工具](code_docs/data_utils/mr2_analysis.md) - 数据集深度分析和可视化
- [演示脚本](code_docs/data_utils/demo.md) - 快速上手与功能测试

### 🔧 预处理模块
- [预处理工具概览](code_docs/preprocessing/) - 文本和图像预处理工具
- [文本处理](code_docs/preprocessing/text_processing.md) - 中英文混合文本处理
- [图像处理](code_docs/preprocessing/image_processing.md) - 图像预处理和特征提取
- [演示脚本](code_docs/preprocessing/demo.md) - 预处理功能验证与测试

### 🤖 模型模块
- [模型库概览](code_docs/models/) - 各类机器学习和深度学习模型
- [传统机器学习](code_docs/models/traditional.md) - SVM、随机森林等传统方法
- [神经网络](code_docs/models/neural_networks.md) - CNN、RNN、Transformer等
- [预训练模型](code_docs/models/pretrained.md) - BERT、RoBERTa等预训练模型
- [多模态模型](code_docs/models/multimodal.md) - CLIP、BLIP等多模态模型
- [图神经网络](code_docs/models/graph_neural_networks.md) - GCN、GAT、GraphSAGE等
- [大语言模型](code_docs/models/llms.md) - Transformer、RAG、微调技术
- [可解释性AI](code_docs/models/explainable_ai.md) - LIME、SHAP、Anchors解释方法

### 🛠️ 工具模块
- [工具库概览](code_docs/utils/) - 通用工具和辅助函数
- [配置管理器](code_docs/utils/config_manager.md) - 配置文件加载和管理
- [文件工具](code_docs/utils/file_utils.md) - 文件读写和格式转换

## 🎯 技术特色与亮点

### 🌟 核心技术栈
- **多模态融合**: 文本 + 图像 + 图结构的统一建模
- **大语言模型**: 支持BERT、RoBERTa、ChatGLM等主流模型
- **参数高效微调**: LoRA、AdaLoRA等轻量化训练技术
- **检索增强生成**: 基于Qwen3-0.6B的RAG系统
- **可解释性AI**: LIME、SHAP等模型解释技术

### 🔧 工程化特色
- **严格数据验证**: 零容忍错误策略，确保数据质量
- **配置驱动设计**: YAML配置文件，支持灵活参数调整
- **路径自动检测**: 智能项目结构识别，简化部署
- **优雅降级处理**: 依赖缺失时的备选方案
- **多语言支持**: 中英文混合文本处理能力

### 📊 数据处理能力
- **MR2数据集专用**: 针对多模态谣言检测优化
- **图神经网络**: 支持社交网络关系建模
- **特征工程**: 统计特征 + 深度特征的融合
- **数据分析**: 完整的EDA和可视化工具链

## 🚀 快速开始

### 环境要求
- Python 3.8+
- PyTorch 1.12+
- CUDA 11.6+ (可选，用于GPU加速)

### 安装依赖
```bash
pip install -r requirements.txt
```

### 基础使用
```python
from utils.config_manager import get_config_manager
from data_utils.mr2_dataset import MR2Dataset
from models.traditional import MLClassifierTrainer

# 加载配置
config_mgr = get_config_manager()

# 加载数据集
dataset = MR2Dataset(split='train')

# 训练模型
trainer = MLClassifierTrainer()
results = trainer.train_all_models(dataset)
```

## 📖 学习路径建议

### 🌱 入门级 (1-3个月)
1. 从**配置模块**开始，理解项目配置架构
2. 学习**数据处理模块**，掌握数据加载和预处理
3. 实践**传统机器学习**方法，建立基线模型

### 🌿 进阶级 (3-12个月)
1. 深入**神经网络模型**，理解深度学习架构
2. 探索**预训练模型**，掌握迁移学习技术
3. 学习**多模态模型**，理解跨模态融合

### 🌳 高级级 (1-2年)
1. 研究**图神经网络**，掌握图结构建模
2. 实践**大语言模型**，体验最前沿技术
3. 应用**可解释性AI**，理解模型决策机制

### 🚀 专家级 (2年+)
1. 优化**训练框架**，提升训练效率
2. 设计**工程架构**，构建生产级系统
3. 贡献**开源项目**，推动技术发展

## 📊 项目统计

### 代码结构
```
代码库组成:
├── 📂 配置文件: 6个模块配置
├── 🔧 预处理: 文本+图像双模态处理
├── 📚 数据工具: 专用数据集+通用工具
├── 🤖 模型库: 传统ML → 大语言模型全覆盖
├── 📝 训练框架: 参数高效微调支持
└── 🛠️ 工具集: 配置管理+文件操作
```

### 技术覆盖
- **传统ML**: SVM、随机森林、朴素贝叶斯、逻辑回归
- **深度学习**: CNN、RNN、LSTM、Transformer
- **预训练模型**: BERT、RoBERTa、ALBERT、Chinese-BERT
- **多模态**: CLIP、BLIP、SimpleCLIP
- **图神经网络**: GCN、GAT、GraphSAGE、GIN
- **大语言模型**: GPT、ChatGLM、Qwen、RAG系统

## 🔗 相关资源

### 学习资源
- [Deep Learning Book](https://www.deeplearningbook.org/) - 深度学习理论基础
- [Transformers Course](https://huggingface.co/course/) - Transformer架构详解
- [Graph Neural Networks](https://distill.pub/2021/gnn-intro/) - 图神经网络入门

### 数据集资源
- [MR2数据集](https://github.com/multimodal-rumor/MR2) - 多模态谣言检测
- [CLIP数据集](https://github.com/openai/CLIP) - 图文对齐数据
- [社交网络数据](https://snap.stanford.edu/) - 图结构数据

### 工具生态
- [Hugging Face](https://huggingface.co/) - 预训练模型库
- [PyTorch Geometric](https://pytorch-geometric.readthedocs.io/) - 图神经网络
- [Weights & Biases](https://wandb.ai/) - 实验跟踪工具

## 📖 文档约定

- 📁 每个模块都有对应的文档目录
- 📝 README.md 提供模块概览和导航
- 🔍 具体功能文档按模块命名
- 💡 包含使用示例和最佳实践
- 🔗 支持文档间交叉引用和导航
- 🎯 学习重点标注，突出核心技术要点
- ⚡ 性能优化建议，提升工程效率
- 🛡️ 错误处理策略，确保系统稳定性

## 🤝 贡献指南

### 贡献方式
1. **Bug报告**: 提交Issue描述问题
2. **功能建议**: 提出新功能需求
3. **代码贡献**: 提交Pull Request
4. **文档改进**: 完善文档内容

### 开发规范
- 遵循PEP 8代码风格
- 编写单元测试覆盖新功能
- 更新相关文档
- 保持向后兼容性

---

*本项目致力于为NLP研究者和工程师提供完整的技术实现参考，从传统机器学习到最新的大语言模型，覆盖多模态AI的完整技术栈。*