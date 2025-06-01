# 预处理模块 Preprocessing Module

> 🔧 **文本和图像预处理工具，支持中英文混合和多模态数据处理**

## 📋 模块概览

预处理模块提供专门针对MR2数据集的文本和图像预处理功能，支持中英文混合文本处理、图像标准化、特征提取等功能。

## 📁 模块文件

| 文件名 | 功能说明 |
|-------|---------|
| [text_processing.py](text_processing.md) | 中英文混合文本预处理器 |
| [image_processing.py](image_processing.md) | 图像预处理和特征提取器 |
| [demo.py](demo.md) | 预处理功能演示和测试脚本 |

## 🚀 快速使用

```python
from preprocessing import TextProcessor, ImageProcessor

# 文本处理
text_processor = TextProcessor(language='mixed')
cleaned_text = text_processor.clean_text(text)

# 图像处理  
image_processor = ImageProcessor(target_size=(224, 224))
tensor = image_processor.process_single_image(image_path)
```

## ✨ 主要特性

### 文本处理特性
- **多语言支持**: 中文、英文、混合语言处理
- **智能分词**: jieba + NLTK分词引擎
- **深度清洗**: URL、提及、emoji等噪声清理
- **特征提取**: 长度、词频、语言检测等

### 图像处理特性
- **标准化处理**: 尺寸调整、格式转换、归一化
- **数据增强**: 翻转、旋转、颜色调整等
- **特征提取**: 颜色、纹理、边缘等底层特征
- **批量处理**: 高效的图像批处理管道

## 🔧 依赖库

- **文本处理**: jieba, nltk, emoji
- **图像处理**: PIL, opencv-python, torchvision
- **深度学习**: torch, numpy

---

**[⬅️ 数据工具模块](../data_utils/README.md) | [文本处理 ➡️](text_processing.md)**
