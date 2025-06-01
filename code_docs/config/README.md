# 配置模块 Config Module

> 🔧 **项目配置文件管理，统一的YAML配置系统**

## 📋 模块概览

配置模块管理项目的所有配置文件，采用YAML格式提供结构化的参数管理。包含数据配置、模型配置、训练配置等。

## 📁 配置文件

| 配置文件 | 功能说明 |
|---------|---------|
| [data_configs.yaml](data_configs.md) | 数据集路径、预处理参数、数据加载器配置 |
| [model_configs.yaml](model_configs.md) | 各类模型的超参数和架构配置 |
| [training_configs.yaml](training_configs.md) | 训练流程、优化器、学习率等训练配置 |
| [supported_models.yaml](supported_models.md) | 支持的模型列表和兼容性信息 |
| [rag_configs.yaml](rag_configs.md) | RAG检索增强生成系统配置 |

## 🚀 快速使用

```python
from utils.config_manager import get_config_manager

# 获取配置管理器
config_mgr = get_config_manager()

# 获取各种配置
data_config = config_mgr.get_data_config()
model_config = config_mgr.get_model_config()
training_config = config_mgr.get_training_config()
```

## ✨ 主要特性

- **统一管理**: 所有配置集中管理
- **层次结构**: 支持嵌套配置组织
- **自动验证**: 内置配置验证机制
- **路径检测**: 自动检测项目和数据路径
- **类型安全**: YAML格式提供类型检查

---

**[⬅️ 返回主页](../README.md) | [数据配置 ➡️](data_configs.md)**
