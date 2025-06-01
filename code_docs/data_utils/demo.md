# 演示脚本 Demo Script

> 📝 **数据工具模块的快速演示和测试脚本**

## 📋 脚本概览

`demo.py`提供了数据工具模块的简化演示脚本，专门用于快速测试和功能验证。

## 🎯 脚本功能

### 主要功能
- **快速演示**: 展示数据集分析的核心功能
- **功能验证**: 验证模块是否正常工作
- **错误诊断**: 帮助识别常见的配置和数据问题
- **简化接口**: 提供最简单的使用方式

### 脚本结构
```python
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author: ipanzhzh
# data_utils/demo.py

"""
数据集分析演示 - 简化版
直接运行即可分析MR2数据集
"""

from mr2_analysis import MR2DatasetAnalyzer

def main():
    """简单演示数据集分析"""
    print("📊 MR2数据集分析演示")
    print("="*50)
    
    # 创建分析器
    analyzer = MR2DatasetAnalyzer(data_dir='../data')
    
    # 运行完整分析
    analyzer.run_complete_analysis()
    
    print("\n✅ 数据分析演示完成!")

if __name__ == "__main__":
    main()
```

## 🚀 快速使用

### 直接运行
```bash
# 进入data_utils目录
cd data_utils

# 运行演示脚本
python demo.py
```

### 期望输出
```
📊 MR2数据集分析演示
==================================================

🔧 使用配置管理器
🔧 数据目录: /path/to/project/data
🔧 输出目录: /path/to/project/outputs/data_utils

🔄 开始MR2数据集完整分析
🔍 检查数据文件可用性...
✅ 找到 train 数据文件
✅ 找到 val 数据文件  
✅ 找到 test 数据文件

📚 === 基础统计分析 ===
TRAIN 数据集:
  样本总数: 500
  标签分布: {0: 200, 1: 180, 2: 120}
  有图像: 450
  有直接检索: 400
  有反向检索: 380

VAL 数据集:
  样本总数: 300
  标签分布: {0: 120, 1: 110, 2: 70}
  有图像: 270
  有直接检索: 240
  有反向检索: 230

TEST 数据集:
  样本总数: 100
  标签分布: {0: 40, 1: 35, 2: 25}
  有图像: 90
  有直接检索: 80
  有反向检索: 75

总样本数: 900

📝 === 文本内容分析 ===
文本总数: 900
平均长度: 45.2 字符
平均词数: 12.8 词
语言分布: {'mixed': 450, 'chinese': 300, 'english': 150}
最常见词汇: [('the', 120), ('of', 95), ('and', 85), ('to', 78), ('这是', 65)]

🖼️ === 图像数据分析 ===
图像总数: 900
有效图像: 810
图像格式: {'JPEG': 720, 'PNG': 90}
平均尺寸: 640 x 480
平均文件大小: 125.6 KB

🔍 === 检索标注分析 ===
直接检索标注数: 720
反向检索标注数: 685
平均检索图像数: 8.2
热门域名: [('news.com', 45), ('example.org', 38), ('media.net', 32)]
平均实体数: 3.4
常见实体: [('person', 156), ('location', 134), ('organization', 98)]

📊 === 生成可视化图表 ===
✅ 基础分布图已保存: outputs/data_utils/charts/basic_distribution.png
✅ 文本分布图已保存: outputs/data_utils/charts/text_distribution.png
✅ 图像分布图已保存: outputs/data_utils/charts/image_distribution.png
✅ 标注分析图已保存: outputs/data_utils/charts/annotation_analysis.png
✅ 综合仪表板已保存: outputs/data_utils/charts/comprehensive_dashboard.png
✅ 所有图表已生成完成

📄 === 生成分析报告 ===
✅ 分析报告已保存到: outputs/data_utils/reports/mr2_dataset_analysis_report.md

🎉 === 分析完成! ===
📁 输出目录: outputs/data_utils
📊 图表目录: outputs/data_utils/charts
📄 报告目录: outputs/data_utils/reports

✅ 数据分析演示完成!
```

## 🔧 脚本特点

### 极简设计
- **最少代码**: 只需几行代码就能运行完整分析
- **自动配置**: 自动检测项目配置和数据路径
- **一键运行**: 无需复杂的参数设置

### 错误处理
```python
# 实际的错误处理逻辑（虽然脚本简化，但分析器内部有完整错误处理）
try:
    analyzer = MR2DatasetAnalyzer(data_dir='../data')
    analyzer.run_complete_analysis()
except Exception as e:
    print(f"❌ 演示运行失败: {e}")
    print("请检查:")
    print("1. MR2数据集是否已下载并解压")
    print("2. 数据文件路径是否正确")
    print("3. 依赖库是否安装完整")
```

## 🎯 使用场景

### 1. 功能验证
用于验证数据工具模块是否正常工作：
```bash
# 新环境测试
python demo.py

# 检查输出是否正常
ls ../outputs/data_utils/charts/
ls ../outputs/data_utils/reports/
```

### 2. 快速分析
当需要快速了解数据集特征时：
```python
# 修改数据路径进行分析
analyzer = MR2DatasetAnalyzer(data_dir='/path/to/your/data')
analyzer.run_complete_analysis()
```

### 3. 调试工具
帮助识别数据或配置问题：
- 数据文件是否存在
- 数据格式是否正确
- 配置是否正确加载
- 输出目录是否可写

## 🔄 扩展使用

### 自定义数据路径
```python
# 修改demo.py使用不同的数据路径
def main():
    print("📊 MR2数据集分析演示")
    print("="*50)
    
    # 自定义数据路径
    custom_data_dir = '/path/to/custom/data'
    analyzer = MR2DatasetAnalyzer(data_dir=custom_data_dir)
    
    analyzer.run_complete_analysis()
    print("\n✅ 数据分析演示完成!")
```

### 分步演示
```python
def detailed_demo():
    """详细的分步演示"""
    print("📊 详细MR2数据集分析演示")
    
    analyzer = MR2DatasetAnalyzer(data_dir='../data')
    
    # 1. 检查数据可用性
    print("\n🔍 === 步骤1: 检查数据可用性 ===")
    availability = analyzer.check_data_availability()
    for split, available in availability.items():
        status = "✅ 可用" if available else "❌ 不可用"
        print(f"  {split}: {status}")
    
    # 2. 加载数据
    print("\n📂 === 步骤2: 加载数据 ===")
    data = analyzer.load_data()
    print(f"加载完成，包含 {len(data)} 个数据分割")
    
    # 3. 基础统计
    print("\n📊 === 步骤3: 基础统计分析 ===")
    stats = analyzer.basic_statistics()
    
    # 4. 文本分析
    print("\n📝 === 步骤4: 文本内容分析 ===")
    text_stats = analyzer.text_analysis()
    
    # 5. 图像分析
    print("\n🖼️ === 步骤5: 图像数据分析 ===")
    image_stats = analyzer.image_analysis()
    
    # 6. 生成图表
    print("\n📈 === 步骤6: 生成可视化图表 ===")
    analyzer.create_visualizations()
    
    # 7. 生成报告
    print("\n📄 === 步骤7: 生成分析报告 ===")
    report = analyzer.generate_report()
    
    print("\n✅ 详细演示完成!")

if __name__ == "__main__":
    # 可以选择运行不同的演示
    # main()           # 简单演示
    detailed_demo()    # 详细演示
```

## 🛠️ 故障排除

### 常见问题及解决方案

#### 1. 数据文件不存在
```
❌ 数据目录不存在: /path/to/data
```
**解决方案**：
1. 下载MR2数据集
2. 解压到项目根目录的`data`文件夹
3. 确保包含`dataset_items_train.json`等文件

#### 2. 依赖库缺失
```
❌ 无法导入模块: No module named 'matplotlib'
```
**解决方案**：
```bash
pip install matplotlib seaborn pandas numpy PIL
```

#### 3. 输出目录权限问题
```
❌ 无法创建输出目录: Permission denied
```
**解决方案**：
1. 检查输出目录写入权限
2. 使用`sudo`运行（不推荐）
3. 更改输出目录到有权限的位置

#### 4. 内存不足
```
❌ 内存不足: Unable to allocate array
```
**解决方案**：
1. 关闭其他程序释放内存
2. 减少图表的DPI设置
3. 分批处理大数据集

### 调试模式
```python
def debug_demo():
    """调试模式演示"""
    import logging
    logging.basicConfig(level=logging.DEBUG)
    
    print("🔧 调试模式演示")
    
    try:
        analyzer = MR2DatasetAnalyzer(data_dir='../data')
        
        # 检查配置
        print(f"数据目录: {analyzer.data_dir}")
        print(f"图表目录: {analyzer.charts_dir}")
        print(f"报告目录: {analyzer.reports_dir}")
        
        # 检查数据文件
        availability = analyzer.check_data_availability()
        print(f"数据可用性: {availability}")
        
        # 运行分析
        results = analyzer.run_complete_analysis()
        print("✅ 调试完成")
        
    except Exception as e:
        print(f"❌ 调试发现问题: {e}")
        import traceback
        traceback.print_exc()
```

## 💡 开发建议

### 修改演示脚本
```python
# 1. 添加自定义分析
def custom_demo():
    analyzer = MR2DatasetAnalyzer(data_dir='../data')
    
    # 只运行特定分析
    analyzer.load_data()
    analyzer.basic_statistics()
    analyzer.text_analysis()
    
    # 自定义输出
    stats = analyzer.analysis_results
    print(f"自定义统计: {stats['basic_stats']}")

# 2. 集成到其他脚本
from data_utils.demo import main as run_demo

def my_analysis_pipeline():
    # 运行演示分析
    run_demo()
    
    # 继续其他处理
    process_results()
```

### 性能优化
```python
# 对于大数据集，可以采样分析
def performance_demo():
    analyzer = MR2DatasetAnalyzer(data_dir='../data')
    
    # 只分析部分数据
    analyzer.max_samples = 1000  # 限制样本数
    analyzer.run_complete_analysis()
```

## 📝 输出说明

### 成功运行的标志
- ✅ 显示绿色的成功消息
- 📊 生成完整的图表文件
- 📄 生成分析报告文件
- 🎉 显示"分析完成"消息

### 文件输出检查
```bash
# 检查生成的文件
ls -la outputs/data_utils/charts/
# 应该看到5个PNG图表文件

ls -la outputs/data_utils/reports/
# 应该看到Markdown报告文件

# 检查文件大小（确保不是空文件）
du -h outputs/data_utils/charts/*
du -h outputs/data_utils/reports/*
```

---

**[⬅️ 数据分析](mr2_analysis.md) | [预处理模块 ➡️](../preprocessing/README.md)**