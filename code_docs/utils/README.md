# 工具模块 Utils Module

> 🛠️ **通用工具库和辅助函数，提供项目基础设施功能**

## 📋 模块概览

工具模块提供项目中常用的通用工具函数，包括配置管理、文件操作、路径处理等基础设施功能，为其他模块提供支持。

## 📁 模块文件

| 文件名 | 功能说明 |
|-------|---------|
| [config_manager.py](config_manager.md) | 统一的配置文件加载和管理系统 |
| [file_utils.py](file_utils.md) | 文件读写、格式转换和路径操作工具 |

## 🚀 快速使用

```python
# 配置管理
from utils.config_manager import get_config_manager
config_mgr = get_config_manager()

# 文件操作
from utils.file_utils import FileUtils
data = FileUtils.read_json('config.json')
```

## ✨ 主要特性

### 配置管理特性
- **自动路径检测**: 智能识别项目根目录
- **配置验证**: 数据集完整性检查
- **统一接口**: 简洁的配置访问方法
- **输出管理**: 自动创建输出目录结构

### 文件工具特性
- **格式支持**: JSON、YAML、CSV、Pickle等
- **路径操作**: 目录创建、文件复制移动
- **批量处理**: 文件批处理和转换
- **错误处理**: 完善的异常处理机制

## 🔧 核心功能

- 项目配置统一管理
- 多格式文件读写操作
- 路径和目录操作
- 数据序列化和反序列化
- 文件格式转换
- 批量文件处理

---

**[⬅️ 预处理模块](../preprocessing/README.md) | [配置管理器 ➡️](config_manager.md)**
