# 数据集分析工具 MR2 Analysis

> 📊 **完整的MR2数据集分析和可视化工具，自动生成图表和报告**

## 📋 工具概览

`MR2DatasetAnalyzer`是专为MR2数据集设计的深度分析工具，提供统计分析、可视化图表生成和完整报告输出功能。

## 🚀 核心类

### MR2DatasetAnalyzer
主要分析器类，集成配置管理和输出管理：

```python
from data_utils.mr2_analysis import MR2DatasetAnalyzer

# 自动配置模式（推荐）
analyzer = MR2DatasetAnalyzer()

# 手动配置模式
analyzer = MR2DatasetAnalyzer(
    data_dir='data',
    output_dir='outputs'
)
```

### 初始化特点
- **智能配置**: 自动集成ConfigManager，使用项目配置
- **目录管理**: 自动创建分层输出目录结构
- **错误处理**: 完善的异常处理和数据为空时的处理
- **演示数据支持**: 当真实数据不可用时可创建演示数据（用于测试）

## 📊 分析功能

### 1. 基础统计分析
```python
def basic_statistics(self) -> Dict[str, Any]:
    """
    基础统计分析
    
    Returns:
        {
            'train': {
                'total_samples': int,
                'label_distribution': Counter,
                'has_image': int,
                'has_direct_annotation': int,
                'has_inverse_annotation': int
            },
            'val': {...},
            'test': {...}
        }
    """
```

**分析内容**：
- 各split的样本数量统计
- 标签分布统计（0: Non-rumor, 1: Rumor, 2: Unverified）
- 图像文件完整性检查
- 检索标注文件可用性统计

### 2. 文本内容分析  
```python
def text_analysis(self) -> Dict[str, Any]:
    """
    文本内容深度分析
    
    Returns:
        {
            'total_texts': int,
            'length_distribution': List[int],
            'language_distribution': Counter,
            'word_count_distribution': List[int],
            'common_words': Counter,
            'samples_by_length': {
                'short': List[str],
                'medium': List[str], 
                'long': List[str]
            }
        }
    """
```

**分析内容**：
- 文本长度分布（字符和词数）
- 语言检测（中文/英文/混合）
- 常用词统计和词频分析
- 字符频率分布
- 按长度分类的文本样例

### 3. 图像数据分析
```python
def image_analysis(self) -> Dict[str, Any]:
    """
    图像数据分析
    
    Returns:
        {
            'total_images': int,
            'valid_images': int,
            'image_sizes': List[Tuple[int, int]],
            'image_formats': Counter,
            'size_distribution': {
                'width': List[int],
                'height': List[int]
            },
            'file_sizes': List[int]
        }
    """
```

**分析内容**：
- 图像文件统计（总数/有效数）
- 图像尺寸分布分析
- 图像格式统计（JPEG/PNG等）
- 文件大小分布
- 图像质量检查

### 4. 检索标注分析
```python
def annotation_analysis(self) -> Dict[str, Any]:
    """
    检索标注数据分析
    
    Returns:
        {
            'direct_annotations': int,
            'inverse_annotations': int,
            'direct_stats': {
                'total_retrieved_images': List[int],
                'domains': Counter
            },
            'inverse_stats': {
                'entities_count': List[int],
                'common_entities': Counter,
                'fully_matched': List[int],
                'partially_matched': List[int]
            }
        }
    """
```

**分析内容**：
- 直接检索结果统计（基于文本检索的图像和网页）
- 反向检索结果统计（基于图像的反向搜索）
- 检索域名分布分析
- 实体识别结果统计
- 匹配类型分布（完全匹配/部分匹配）

## 📈 可视化图表

### 自动生成的图表
分析工具会在`outputs/data_utils/charts/`目录下生成以下图表：

#### 1. basic_distribution.png - 基础分布分析
```python
def _plot_basic_distribution(self):
    """
    生成4个子图：
    - 数据集大小分布（柱状图）
    - 标签分布（饼图）
    - 数据完整性分析（分组柱状图）
    - 按split的标签分布（堆叠柱状图）
    """
```

#### 2. text_distribution.png - 文本内容分析
```python
def _plot_text_distribution(self):
    """
    生成6个子图：
    - 文本长度分布（直方图）
    - 词数分布（直方图）
    - 语言分布（饼图）
    - 常用词云（水平柱状图）
    - 字符频率分布（柱状图）
    - 文本统计摘要（文本框）
    """
```

#### 3. image_distribution.png - 图像数据分析
```python
def _plot_image_distribution(self):
    """
    生成4个子图：
    - 图像尺寸散点图
    - 宽度分布直方图
    - 高度分布直方图
    - 图像格式饼图
    """
```

#### 4. annotation_analysis.png - 检索标注分析
```python
def _plot_annotation_analysis(self):
    """
    生成6个子图：
    - 检索数量对比
    - 标注文件可用性
    - 热门域名分布
    - 实体数量分布
    - 常见实体排行
    - 匹配类型分布
    """
```

#### 5. comprehensive_dashboard.png - 综合仪表板
```python
def _create_dashboard(self):
    """
    综合仪表板包含：
    - 数据集概览
    - 标签分布
    - 文本长度分布
    - 语言分析
    - 质量指标
    - 汇总统计
    """
```

## 📝 报告生成

### generate_report 方法
```python
def generate_report(self) -> str:
    """
    生成完整的Markdown分析报告
    
    Returns:
        报告内容字符串
        
    输出文件:
        outputs/data_utils/reports/mr2_dataset_analysis_report.md
    """
```

### 报告内容结构
```markdown
# MR2多模态谣言检测数据集分析报告

## 执行摘要
- 数据集总体概况
- 关键统计数据

## 数据集统计
- 数据量分布
- 标签分布详情

## 文本内容分析  
- 文本统计信息
- 语言分布分析

## 数据质量评估
- 各split完整性评估
- 质量指标分析

## 建议和结论
- 数据集特点总结
- 使用建议
```

## 🔧 完整分析流程

### run_complete_analysis 方法
```python
def run_complete_analysis(self) -> Dict[str, Any]:
    """
    运行完整的数据集分析流程
    
    步骤:
    1. 加载数据
    2. 基础统计分析
    3. 文本分析
    4. 图像分析
    5. 检索标注分析
    6. 创建可视化图表
    7. 生成分析报告
    
    Returns:
        完整的分析结果字典
    """
```

## 🎯 使用示例

### 基础使用
```python
from data_utils.mr2_analysis import MR2DatasetAnalyzer

# 创建分析器
analyzer = MR2DatasetAnalyzer()

# 运行完整分析
results = analyzer.run_complete_analysis()

print("✅ 分析完成!")
print(f"📁 输出目录: {analyzer.charts_dir.parent}")
print(f"📊 图表目录: {analyzer.charts_dir}")
print(f"📄 报告目录: {analyzer.reports_dir}")
```

### 分步分析
```python
# 逐步进行分析
analyzer = MR2DatasetAnalyzer()

# 1. 加载数据
data = analyzer.load_data()
print(f"数据加载完成，包含 {len(data)} 个splits")

# 2. 基础统计
stats = analyzer.basic_statistics()
print(f"基础统计完成")

# 3. 文本分析
text_stats = analyzer.text_analysis()
print(f"文本分析完成，分析了 {text_stats.get('total_texts', 0)} 个文本")

# 4. 图像分析
image_stats = analyzer.image_analysis()
print(f"图像分析完成，发现 {image_stats.get('valid_images', 0)} 个有效图像")

# 5. 生成可视化
analyzer.create_visualizations()
print("可视化图表生成完成")

# 6. 生成报告
report = analyzer.generate_report()
print("分析报告生成完成")
```

### 演示数据模式
```python
# 当真实数据不可用时，可以创建演示数据进行测试
analyzer = MR2DatasetAnalyzer(data_dir='data')

# 检查数据可用性
availability = analyzer.check_data_availability()
if not any(availability.values()):
    print("❓ 没有找到真实数据，是否创建演示数据？")
    # analyzer.create_demo_data()  # 创建演示数据
```

## ⚙️ 配置集成

### 配置管理器集成
```python
# 自动使用配置管理器的设置
if USE_CONFIG_MANAGER:
    self.config_manager = get_config_manager()
    self.data_dir = get_data_dir()
    self.charts_dir = get_output_path('data_utils', 'charts')
    self.reports_dir = get_output_path('data_utils', 'reports')
    
    # 获取分析配置
    analysis_config = get_analysis_config()
    self.colors = analysis_config.get('visualization', {}).get('colors', {})
    self.label_mapping = get_label_mapping()
```

### 颜色主题配置
```python
# 从配置文件获取的颜色主题
default_colors = {
    'primary': '#FF6B6B',     # 主色调
    'secondary': '#4ECDC4',   # 次要色调
    'tertiary': '#45B7D1',    # 第三色调
    'accent': '#96CEB4',      # 强调色
    'warning': '#FFEAA7',     # 警告色
    'info': '#DDA0DD',        # 信息色
    'success': '#98FB98'      # 成功色
}
```

## 🔍 错误处理和容错

### 数据缺失处理
```python
# 当数据文件不存在时
if not any(availability.values()):
    print("❓ 没有找到真实数据，是否创建演示数据？")
    try:
        self.create_demo_data()
        availability = self.check_data_availability()
    except Exception as e:
        print(f"❌ 创建演示数据失败: {e}")

# 当分析结果为空时
if not self.analysis_results.get('basic_stats'):
    print("⚠️  没有分析结果，跳过图表生成")
    return
```

### 图表生成容错
```python
# 每个图表生成都有独立的错误处理
try:
    self._plot_basic_distribution()
except Exception as e:
    print(f"❌ 生成基础分布图失败: {e}")

try:
    self._plot_text_distribution()
except Exception as e:
    print(f"❌ 生成文本分布图失败: {e}")
```

## 📁 输出目录结构

```
outputs/data_utils/
├── charts/                          # 图表输出目录
│   ├── basic_distribution.png       # 基础分布分析
│   ├── text_distribution.png        # 文本内容分析
│   ├── image_distribution.png       # 图像数据分析
│   ├── annotation_analysis.png      # 检索标注分析
│   └── comprehensive_dashboard.png  # 综合仪表板
├── reports/                         # 报告输出目录
│   └── mr2_dataset_analysis_report.md
└── analysis/                        # 原始分析数据
    └── analysis_results.json       # 分析结果数据
```

## 💡 分析结果解读

### 关键指标
- **数据完整性**: 图像、标注文件的可用性比例
- **语言分布**: 中英文混合比例，反映数据集的多语言特性
- **标签平衡性**: 三类标签的分布是否均衡
- **文本质量**: 长度分布、词汇丰富度等指标

### 质量评估标准
- **优秀**: 数据完整性>90%，标签分布相对均衡
- **良好**: 数据完整性>70%，有轻微的标签不平衡
- **需要改进**: 数据完整性<70%，存在明显质量问题

---

**[⬅️ 数据加载器](data_loaders.md) | [演示脚本 ➡️](demo.md)**