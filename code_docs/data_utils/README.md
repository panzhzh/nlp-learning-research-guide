# 数据工具模块文档

> 📚 **MR2数据集加载、处理和分析工具集**

## 📋 模块概览

数据工具模块专门为MR2多模态谣言检测数据集设计，提供数据加载、预处理、分析和可视化的完整工具链。

## 📁 模块结构

```
data_utils/
├── __init__.py                # 模块初始化和导出
├── mr2_dataset.py            # MR2数据集PyTorch类
├── data_loaders.py           # 数据加载器配置
├── mr2_analysis.py           # 数据集分析和可视化
└── demo.py                   # 使用演示脚本
```

## 📚 详细文档

### 核心组件
- [**mr2_dataset.py**](mr2_dataset.md) - MR2数据集的PyTorch Dataset实现
- [**data_loaders.py**](data_loaders.md) - 批量数据加载和预处理配置
- [**mr2_analysis.py**](mr2_analysis.md) - 数据集深度分析和统计可视化

### 辅助工具
- [**demo.py**](demo.md) - 数据工具使用演示和快速测试

## 🎯 主要功能

### 数据加载
- **严格验证**: 确保数据集完整性和格式正确性
- **多模态支持**: 同时处理文本、图像和元数据
- **批量处理**: 高效的DataLoader配置和批处理
- **错误处理**: 完善的异常处理和错误恢复

### 数据分析
- **统计分析**: 数据分布、标签统计、完整性分析
- **可视化**: 自动生成图表和分析报告
- **质量评估**: 数据质量检查和问题识别
- **深度洞察**: 文本长度、图像特征、检索信息分析

## 🚀 快速使用

### 基础数据加载
```python
from data_utils import MR2Dataset, create_all_dataloaders

# 创建数据集
dataset = MR2Dataset(
    data_dir='data',
    split='train',
    load_images=True
)

# 创建数据加载器
dataloaders = create_all_dataloaders()
train_loader = dataloaders['train']
```

### 数据分析
```python
from data_utils.mr2_analysis import MR2DatasetAnalyzer

# 创建分析器
analyzer = MR2DatasetAnalyzer(data_dir='data')

# 运行完整分析
results = analyzer.run_complete_analysis()
```

## 📊 数据集要求

### 必需文件
- `dataset_items_train.json` - 训练集数据项
- `dataset_items_val.json` - 验证集数据项  
- `dataset_items_test.json` - 测试集数据项

### 目录结构
```
data/
├── dataset_items_train.json
├── dataset_items_val.json
├── dataset_items_test.json
├── train/img/               # 训练集图像
├── val/img/                 # 验证集图像
└── test/img/                # 测试集图像
```

## ⚠️ 重要说明

- **真实数据要求**: 模块要求使用真实的MR2数据集，不支持演示数据
- **路径配置**: 自动检测数据目录，也可通过配置文件指定
- **资源管理**: 支持图像预加载和内存优化
- **并发处理**: 支持多进程数据加载和批处理

---

<div style="text-align: center; margin-top: 20px;">
  <a href="../README.md" style="background: #2196F3; color: white; text-decoration: none; padding: 8px 16px; border-radius: 4px;">
    🏠 返回主页
  </a>
  <button onclick="window.scrollTo(0,0)" style="background: #4CAF50; color: white; border: none; padding: 8px 16px; border-radius: 4px; cursor: pointer; margin-left: 10px;">
    📜 返回顶部
  </button>
</div>
