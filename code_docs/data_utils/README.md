# 数据工具模块 Data Utils Module

> 📚 **严格的MR2数据集处理工具，强制真实数据验证**

## 📋 模块概览

数据工具模块专为MR2多模态谣言检测数据集设计，提供严格的数据验证、加载和分析功能。**不支持演示数据，必须使用真实数据集。**

## 📁 核心组件

### 主要模块文件
| 文件名 | 功能说明 | 状态 |
|-------|----------|------|
| [**mr2_dataset.py**](code_docs/data_loaders/mr2_dataset.md) | 严格的MR2数据集PyTorch类实现 | ✅ 已实现 |
| [**data_loaders.py**](code_docs/data_loaders/data_loaders.md) | 强制验证的数据加载器配置 | ✅ 已实现 |
| [**mr2_analysis.py**](code_docs/data_loaders/mr2_analysis.md) | 完整的数据集分析和可视化工具 | ✅ 已实现 |
| [**demo.py**](code_docs/data_loaders/demo.md) | 简化的演示脚本 | ✅ 已实现 |

### 模块初始化 (__init__.py)
```python
# 智能导入机制，优先简化版本
try:
    from .mr2_dataset import SimpleMR2Dataset as MR2Dataset
    from .data_loaders import create_all_dataloaders, StrictDataLoaderConfig
except ImportError:
    # 自动处理导入错误，提供清晰的错误信息
```

## 🚀 核心特性

### 严格验证机制
- **强制真实数据**: 必须使用完整的MR2数据集
- **文件完整性检查**: 验证所有必需的JSON文件
- **最小样本要求**: 每个split至少10个样本
- **格式验证**: 严格的JSON格式和字段验证

### 多模态支持
- **文本处理**: 中英文混合文本的完整处理
- **图像处理**: 标准化的图像预处理管道
- **元数据处理**: 标签、语言、时间戳等信息
- **检索信息**: 直接检索和反向检索数据

### 高性能数据加载
- **批量处理**: 高效的DataLoader配置
- **错误恢复**: 完善的异常处理机制
- **内存优化**: 可选的图像预加载
- **并发支持**: 多进程数据加载

## 🎯 快速使用

### 基础数据集使用
```python
from data_utils import MR2Dataset

# 创建严格验证的数据集
try:
    dataset = MR2Dataset(
        data_dir='data',           # 数据目录
        split='train',             # 数据划分
        transform_type='train',    # 变换类型
        load_images=True           # 是否加载图像
    )
    print(f"✅ 数据集创建成功: {len(dataset)} 样本")
    
except FileNotFoundError as e:
    print(f"❌ 数据文件不存在: {e}")
except ValueError as e:
    print(f"❌ 数据验证失败: {e}")
```

### 批量数据加载器
```python
from data_utils import create_all_dataloaders

# 创建所有数据加载器（严格模式）
try:
    dataloaders = create_all_dataloaders(
        batch_sizes={'train': 32, 'val': 64, 'test': 64}
    )
    
    train_loader = dataloaders['train']
    print(f"✅ 训练集: {len(train_loader.dataset)} 样本")
    
except RuntimeError as e:
    print(f"❌ 数据加载器创建失败: {e}")
```

### 数据集分析
```python
from data_utils.mr2_analysis import MR2DatasetAnalyzer

# 创建分析器并运行完整分析
analyzer = MR2DatasetAnalyzer(data_dir='data')
results = analyzer.run_complete_analysis()

# 生成可视化图表和报告
# 输出到 outputs/data_utils/ 目录
```

## 📊 数据集要求

### 必需文件结构
```
data/
├── dataset_items_train.json      # 训练集数据项 (必需)
├── dataset_items_val.json        # 验证集数据项 (必需)  
├── dataset_items_test.json       # 测试集数据项 (必需)
├── train/img/                    # 训练集图像目录
├── val/img/                      # 验证集图像目录
├── test/img/                     # 测试集图像目录
├── train/img_html_news/          # 直接检索结果
│   └── direct_annotation.json
└── train/inverse_search/         # 反向检索结果
    └── inverse_annotation.json
```

### 数据验证规则
```python
# 每个数据项必须包含的字段
required_fields = {
    'caption': str,     # 文本内容 (必需)
    'label': int,       # 标签 (0, 1, 2)
    'image_path': str,  # 图像路径 (可选)
    'language': str     # 语言类型 (可选)
}

# 标签映射
labels = {
    0: "Non-rumor",     # 非谣言
    1: "Rumor",         # 谣言
    2: "Unverified"     # 未验证
}
```

## 🔧 高级功能

### 严格数据加载器配置
```python
from data_utils.data_loaders import StrictDataLoaderConfig

# 严格配置类特点
config = StrictDataLoaderConfig()
# - 自动检查数据要求
# - 从配置文件加载参数
# - 提供默认的安全配置
```

### 批处理函数
```python
from data_utils.data_loaders import StrictCollateFunction

# 严格的批处理函数特点
collate_fn = StrictCollateFunction()
# - 验证批次数据完整性
# - 处理缺失图像的情况
# - 自动创建标准张量格式
```

### 数据分析工具
```python
from data_utils.mr2_analysis import MR2DatasetAnalyzer

analyzer = MR2DatasetAnalyzer()

# 支持的分析功能
analyzer.basic_statistics()      # 基础统计分析
analyzer.text_analysis()         # 文本内容分析  
analyzer.image_analysis()        # 图像数据分析
analyzer.annotation_analysis()   # 检索标注分析
analyzer.create_visualizations() # 生成可视化图表
analyzer.generate_report()       # 生成分析报告
```

## 📈 分析和可视化

### 自动生成内容
分析工具会自动创建以下内容：

**图表文件 (outputs/data_utils/charts/)**:
- `basic_distribution.png` - 基础数据分布
- `text_distribution.png` - 文本内容分析
- `image_distribution.png` - 图像数据分布
- `annotation_analysis.png` - 检索标注分析
- `comprehensive_dashboard.png` - 综合仪表板

**分析报告 (outputs/data_utils/reports/)**:
- `mr2_dataset_analysis_report.md` - 完整分析报告

### 统计分析内容
- **数据分布**: 各split的样本数量和标签分布
- **文本分析**: 长度分布、语言检测、常用词统计
- **图像分析**: 尺寸分布、格式统计、完整性检查
- **质量评估**: 数据完整性和质量指标

## ⚠️ 重要说明

### 严格模式特点
- **不支持演示数据**: 必须使用真实的MR2数据集文件
- **强制验证**: 所有数据文件必须存在且格式正确
- **最小样本要求**: 每个数据分割需要满足最小样本数
- **错误即停**: 任何验证失败都会立即抛出异常

### 数据获取
```bash
# MR2数据集下载链接
链接: https://pan.baidu.com/s/1sfUwsaeV2nfl54OkrfrKVw?pwd=jxhc 
提取码: jxhc

# 解压后放置到项目根目录的data文件夹
```

### 常见错误处理
```python
# 常见错误类型及解决方案
try:
    dataset = MR2Dataset(data_dir='data', split='train')
except FileNotFoundError:
    # 解决: 下载并解压MR2数据集到正确位置
    print("请下载MR2数据集并解压到data目录")
except ValueError as e:
    # 解决: 检查数据文件完整性和格式
    print(f"数据验证失败: {e}")
except RuntimeError:
    # 解决: 检查配置文件和依赖库
    print("检查配置管理器和必要依赖库")
```

## 💡 使用建议

### 开发工作流
1. **确保数据集**: 首先下载完整的MR2数据集
2. **验证安装**: 运行`demo.py`检查模块功能
3. **数据分析**: 使用分析工具了解数据特征
4. **模型训练**: 基于分析结果选择合适的模型

### 性能优化
- **减少workers**: 如果遇到多进程问题，减少`num_workers`
- **批次调整**: 根据GPU内存调整`batch_size`
- **图像预加载**: 根据内存情况选择是否预加载图像

### 调试技巧
- **打印样本**: 使用`dataset.print_sample_info(0)`查看样本
- **检查统计**: 使用`dataset.get_statistics()`查看数据集统计
- **逐步验证**: 先验证单个样本，再验证整个数据集

---

**[⬅️ RAG配置](code_docs/config/) | [MR2数据集 ➡️](code_docs/data_loaders/mr2_dataset.md)**