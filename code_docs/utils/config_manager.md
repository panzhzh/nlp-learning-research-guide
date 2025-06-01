# 配置管理器 Config Manager

> 🔧 **统一的项目配置管理系统，自动检测路径并加载YAML配置**

## 📋 功能说明

`ConfigManager` 是项目的核心配置管理类，负责加载和管理所有YAML配置文件，提供统一的配置访问接口，并处理路径检测和数据验证。

## 🎯 主要功能

### 配置管理功能
- **自动路径检测**: 智能识别项目根目录和配置文件位置
- **配置文件加载**: 统一加载所有YAML格式配置文件
- **配置访问**: 提供简洁的配置获取方法
- **缓存机制**: 避免重复文件读取，提高性能

### 数据验证功能
- **数据目录验证**: 确保MR2数据集正确安装
- **文件完整性检查**: 验证必需的JSON数据文件
- **样本数量验证**: 检查最小样本数要求
- **格式一致性检查**: 验证数据格式正确性

### 目录管理功能
- **输出目录创建**: 自动创建项目输出目录结构
- **路径解析**: 处理相对路径和绝对路径转换
- **目录权限检查**: 确保输出目录可写

## 🚀 核心类和方法

### ConfigManager 类

#### 初始化方法
```python
ConfigManager(config_dir=None)
```

**参数说明:**
- `config_dir`: 配置文件目录，默认为项目根目录下的config文件夹

#### 配置获取方法
- `get_data_config()`: 获取数据配置
- `get_model_config()`: 获取模型配置  
- `get_training_config()`: 获取训练配置
- `get_config(config_type)`: 获取指定类型配置

#### 路径管理方法
- `get_data_dir()`: 获取数据目录路径
- `get_output_path(module, subdir)`: 获取输出路径
- `create_output_directories()`: 创建输出目录结构
- `get_absolute_path(relative_path)`: 转换为绝对路径

#### 特殊功能方法
- `get_label_mapping()`: 获取标签映射字典
- `get_analysis_config()`: 获取分析配置
- `check_data_requirements()`: 检查数据要求
- `_validate_data_directory()`: 验证数据目录

## 📦 便捷函数

### 全局实例函数
```python
# 获取全局配置管理器实例
get_config_manager()

# 直接获取各种配置
get_data_config()
get_training_config() 
get_model_config()

# 获取特殊信息
get_label_mapping()
get_data_dir()
get_output_path(module, subdir)
```

## 📊 配置文件结构

### 支持的配置文件
```
config/
├── data_configs.yaml          # 数据配置
├── model_configs.yaml         # 模型配置
├── training_configs.yaml      # 训练配置
├── supported_models.yaml      # 支持的模型列表
└── rag_configs.yaml          # RAG系统配置
```

### 自动创建的目录结构
```
project_root/
├── config/                    # 配置文件目录
├── data/                      # 数据目录
└── outputs/                   # 输出目录
    ├── data_utils/           # 数据工具输出
    │   ├── charts/
    │   ├── reports/
    │   └── analysis/
    ├── models/               # 模型输出
    └── logs/                 # 日志文件
```

## 💡 使用示例

### 基础配置获取
```python
from utils.config_manager import get_config_manager

# 获取配置管理器
config_mgr = get_config_manager()

# 获取数据配置
data_config = config_mgr.get_data_config()
print(f"数据目录: {config_mgr.get_data_dir()}")

# 获取标签映射
labels = config_mgr.get_label_mapping()
print(f"标签映射: {labels}")

# 获取训练配置
training_config = config_mgr.get_training_config()
general = training_config.get('general', {})
```

### 便捷函数使用
```python
from utils.config_manager import (
    get_data_config,
    get_label_mapping,
    get_data_dir,
    get_output_path
)

# 直接获取配置
data_config = get_data_config()
labels = get_label_mapping()
data_dir = get_data_dir()

# 获取输出路径
charts_dir = get_output_path('data_utils', 'charts')
models_dir = get_output_path('models', 'checkpoints')
```

### 数据验证使用
```python
from utils.config_manager import check_data_requirements

try:
    # 检查数据要求
    check_data_requirements()
    print("✅ 数据验证通过")
    
except FileNotFoundError as e:
    print(f"❌ 数据文件缺失: {e}")
    
except ValueError as e:
    print(f"❌ 数据验证失败: {e}")
```

### 路径管理使用
```python
# 创建输出目录
config_mgr.create_output_directories()

# 获取各种路径
data_dir = config_mgr.get_data_dir()
charts_dir = config_mgr.get_output_path('data_utils', 'charts')
reports_dir = config_mgr.get_output_path('data_utils', 'reports')

# 路径转换
abs_path = config_mgr.get_absolute_path('relative/path')
```

## 🔧 配置访问模式

### 层次化配置访问
```python
# 获取数据预处理配置
data_config = get_data_config()
processing = data_config.get('processing', {})
text_config = processing.get('text', {})

max_length = text_config.get('max_length', 512)
remove_urls = text_config.get('remove_urls', True)

# 获取训练参数
training_config = get_training_config()
neural_config = training_config.get('neural_networks', {})
basic_nn = neural_config.get('basic_nn', {})

epochs = basic_nn.get('epochs', 50)
learning_rate = basic_nn.get('learning_rate', 0.001)
```

### 默认值处理
```python
# 安全的配置获取，带默认值
data_config = get_data_config()
dataset_config = data_config.get('dataset', {})

# 批次大小配置
dataloader_config = data_config.get('dataloader', {})
train_config = dataloader_config.get('train', {})
batch_size = train_config.get('batch_size', 32)  # 默认32

# 数据路径配置
paths_config = dataset_config.get('paths', {})
base_dir = paths_config.get('base_dir', 'auto_detect')
```

## ⚠️ 重要说明

### 数据集要求
- **强制验证**: 配置管理器会严格验证数据集完整性
- **必需文件**: 必须包含train/val/test三个JSON文件
- **最小样本**: 每个数据分割必须包含最小数量样本(默认10个)
- **路径检测**: 自动检测项目根目录下的data文件夹

### 错误处理
- **文件不存在**: 抛出FileNotFoundError并提供解决建议
- **格式错误**: 抛出ValueError并给出详细错误信息
- **路径问题**: 自动尝试路径修正，失败时报告错误

### 性能考虑
- **单例模式**: 使用全局实例避免重复初始化
- **配置缓存**: 配置文件内容会被缓存，避免重复读取
- **惰性加载**: 按需加载配置文件，提高启动速度

## 🔍 故障排除

### 常见错误及解决方案

**1. 数据目录找不到**
```
❌ 数据目录不存在: /path/to/project/data
```
**解决**: 确保MR2数据集已下载并解压到项目根目录的data文件夹

**2. 配置文件缺失**
```
⚠️ 配置文件不存在: config/data_configs.yaml
```
**解决**: 检查config目录是否存在，YAML文件是否正确命名

**3. 数据文件验证失败**
```
❌ 缺少必要的数据文件: ['dataset_items_train.json']
```
**解决**: 确保所有必需的JSON数据文件都已正确解压

### 调试方法
```python
# 启用详细日志
import logging
logging.basicConfig(level=logging.DEBUG)

# 手动验证配置
config_mgr = ConfigManager()
print(f"项目根目录: {config_mgr.project_root}")
print(f"配置目录: {config_mgr.config_dir}")
print(f"数据目录: {config_mgr.get_data_dir()}")
```

---

**[⬅️ 工具模块概览](README.md) | [文件工具 ➡️](file_utils.md)**
