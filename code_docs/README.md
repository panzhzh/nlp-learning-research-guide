# NLP技术实现代码库文档

> 🚀 **模块化的NLP技术实现，支持文本、图像、图结构的多模态分析**

## 📚 文档导航

### 🔧 配置模块
- [配置模块概览](config/README.md) - 配置文件管理和参数设置
- [数据配置](config/data_configs.md) - 数据集路径、预处理参数
- [模型配置](config/model_configs.md) - 各类模型的配置参数
- [训练配置](config/training_configs.md) - 训练超参数和策略配置

### 📚 数据处理模块
- [数据工具概览](data_utils/README.md) - 数据加载、处理和分析工具
- [MR2数据集](data_utils/mr2_dataset.md) - 多模态谣言检测数据集类
- [数据加载器](data_utils/data_loaders.md) - PyTorch数据加载器配置
- [数据分析工具](data_utils/mr2_analysis.md) - 数据集深度分析和可视化

### 🔧 预处理模块
- [预处理工具概览](preprocessing/README.md) - 文本和图像预处理工具
- [文本处理](preprocessing/text_processing.md) - 中英文混合文本处理
- [图像处理](preprocessing/image_processing.md) - 图像预处理和特征提取

### 🤖 模型模块
- [模型库概览](models/README.md) - 各类机器学习和深度学习模型
- [传统机器学习](models/traditional.md) - SVM、随机森林等传统方法
- [神经网络](models/neural_networks.md) - CNN、RNN、Transformer等
- [预训练模型](models/pretrained.md) - BERT、RoBERTa等预训练模型
- [多模态模型](models/multimodal.md) - CLIP、BLIP等多模态模型

### ��️ 训练模块
- [训练框架概览](training/README.md) - 模型训练和优化框架
- [基础训练器](training/base_trainer.md) - 通用训练器基类
- [微调方法](training/fine_tuning.md) - LoRA、AdaLoRA等参数高效微调

### 📊 评估模块
- [评估工具概览](evaluation/README.md) - 模型评估指标和分析工具
- [评估指标](evaluation/metrics.md) - 准确率、F1、AUC等指标计算
- [统计检验](evaluation/statistical_tests.md) - 显著性检验方法

### 🛠️ 工具模块
- [工具库概览](utils/README.md) - 通用工具和辅助函数
- [配置管理器](utils/config_manager.md) - 配置文件加载和管理
- [文件工具](utils/file_utils.md) - 文件读写和格式转换

### 📝 示例和脚本
- [使用示例](examples/README.md) - 快速上手和完整示例
- [执行脚本](scripts/README.md) - 批量处理和自动化脚本

## 🎯 快速开始

1. **环境设置**: 参考 [配置模块](config/README.md) 进行环境配置
2. **数据准备**: 使用 [数据工具](data_utils/README.md) 加载和处理数据
3. **模型训练**: 通过 [训练框架](training/README.md) 训练模型
4. **结果评估**: 利用 [评估工具](evaluation/README.md) 分析模型性能

## 📖 文档约定

- 📁 每个模块都有对应的文档目录
- 📝 README.md 提供模块概览和导航
- 🔍 具体功能文档按模块命名 (如 `file_utils.md`)
- 💡 包含使用示例和最佳实践
- 🔗 支持文档间交叉引用和导航

---

