# 预处理模块 Preprocessing Module

> 🔧 **专业的中英文混合文本和图像预处理工具，支持MR2多模态数据**

## 📋 模块概览

预处理模块提供专门针对MR2数据集的文本和图像预处理功能，支持中英文混合文本处理、标准化图像预处理和多模态特征提取。

## 📁 核心组件

### 主要模块文件
| 文件名 | 功能说明 | 特点 |
|-------|----------|------|
| [**text_processing.py**](text_processing.md) | 中英文混合文本预处理器 | 智能分词、多语言检测、特征提取 |
| [**image_processing.py**](image_processing.md) | 图像预处理和特征提取器 | 标准化处理、数据增强、批量处理 |
| [**demo.py**](demo.md) | 预处理功能演示脚本 | 快速测试、功能验证 |

### 模块初始化 (__init__.py)
```python
from .text_processing import TextProcessor
from .image_processing import ImageProcessor

__all__ = [
    'TextProcessor',
    'ImageProcessor'
]
```

## 🚀 核心特性

### 多语言文本处理
- **中英文混合**: 智能识别和处理中英文混合文本
- **语言检测**: 自动检测文本语言类型
- **智能分词**: jieba + NLTK联合分词引擎
- **深度清洗**: URL、提及、emoji等噪声清理
- **特征提取**: 长度、词频、语言特征等多维特征

### 专业图像处理
- **标准化处理**: ImageNet标准归一化和尺寸调整
- **数据增强**: 多级别数据增强策略
- **特征提取**: 颜色、纹理、几何等底层特征
- **批量处理**: 高效的MR2数据集批量处理
- **质量控制**: 完善的图像质量检查和错误恢复

### 配置驱动设计
- **配置集成**: 与项目配置管理器无缝集成
- **参数化**: 所有处理参数都可通过配置文件调整
- **默认兼容**: 即使没有配置文件也能正常工作

## 🎯 快速使用

### 文本处理
```python
from preprocessing import TextProcessor

# 创建中英文混合文本处理器
processor = TextProcessor(language='mixed')

# 文本清洗
text = "这是测试文本 This is test! @user https://example.com 😊"
cleaned = processor.clean_text(text)
print(f"清洗后: {cleaned}")
# 输出: "这是测试文本 This is test"

# 语言检测
language = processor.detect_language(text)
print(f"语言类型: {language}")  # 输出: "mixed"

# 智能分词
tokens = processor.tokenize(text)
print(f"分词结果: {tokens}")
# 输出: ['这是', '测试', '文本', 'this', 'test']

# 特征提取
features = processor.extract_features(text)
print(f"文本长度: {features['text_length']}")
print(f"词数: {features['token_count']}")
print(f"语言: {features['language']}")
```

### 图像处理
```python
from preprocessing import ImageProcessor

# 创建图像处理器
processor = ImageProcessor(target_size=(224, 224))

# 处理单张图像
tensor = processor.process_single_image(
    image_path='data/train/img/example.jpg',
    transform_type='train'  # 包含数据增强
)
print(f"图像张量形状: {tensor.shape}")  # torch.Size([3, 224, 224])

# 获取图像信息
info = processor.get_image_info('data/train/img/example.jpg')
print(f"图像尺寸: {info['width']} x {info['height']}")
print(f"文件大小: {info['file_size_mb']:.2f} MB")

# 提取图像特征
image = processor.load_image('data/train/img/example.jpg')
features = processor.extract_image_features(image)
print(f"亮度: {features['brightness']:.2f}")
print(f"对比度: {features['contrast']:.2f}")

# 批量处理MR2数据集
results = processor.process_mr2_dataset(
    splits=['train'],
    save_features=True
)
print(f"处理完成: {results['train']['processed_images']} 张图像")
```

## 🔧 高级功能

### 文本处理高级特性

#### 配置驱动的参数
```python
# 从配置文件自动加载参数
processor = TextProcessor(language='mixed')
# 自动获取：
# - max_length: 512
# - remove_urls: True
# - remove_mentions: True
# - normalize_whitespace: True
```

#### 批量文本处理
```python
texts = [
    "这是第一个测试文本",
    "This is the second test text",
    "混合语言文本 mixed language"
]

# 批量预处理
results = processor.preprocess_batch(texts)
for i, result in enumerate(results):
    print(f"文本{i+1}: {len(result['tokens'])} tokens")
    print(f"特征: {result['features']['language']}")
```

#### 自定义处理配置
```python
# 自定义清洗参数
processor = TextProcessor()
processor.remove_hashtags = True  # 移除话题标签
processor.remove_urls = False     # 保留URL

cleaned = processor.clean_text(text)
```

### 图像处理高级特性

#### 多级数据增强
```python
# 轻度增强（50%翻转 + 5度旋转）
light_augmented = processor.apply_augmentation(image, 'light')

# 中度增强（翻转 + 旋转 + 亮度调整）
medium_augmented = processor.apply_augmentation(image, 'medium')

# 重度增强（全套增强 + 高斯模糊）
heavy_augmented = processor.apply_augmentation(image, 'heavy')
```

#### 自定义变换配置
```python
import torchvision.transforms as transforms

# 自定义训练变换
processor.train_transforms = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomCrop((224, 224)),
    transforms.RandomHorizontalFlip(p=0.7),
    transforms.ColorJitter(brightness=0.3, contrast=0.3),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                        std=[0.229, 0.224, 0.225])
])
```

#### MR2数据集专用处理
```python
# 处理完整的MR2数据集
results = processor.process_mr2_dataset(
    splits=['train', 'val', 'test'],
    save_features=True
)

# 创建统计信息
statistics = processor.create_image_statistics(results)
print(f"总图像数: {statistics['total_images']}")
print(f"平均尺寸: {statistics['avg_width']:.1f} x {statistics['avg_height']:.1f}")
print(f"格式分布: {statistics['format_distribution']}")
```

## 🎨 特征提取

### 文本特征
```python
features = processor.extract_features(text)
"""
返回完整特征字典:
{
    'text_length': int,           # 文本长度
    'word_count': int,            # 词数
    'char_count': int,            # 字符数
    'language': str,              # 语言类型
    'tokens': List[str],          # 分词结果
    'token_count': int,           # token数量
    'exclamation_count': int,     # 感叹号数量
    'question_count': int,        # 问号数量
    'uppercase_ratio': float,     # 大写字母比例
    'digit_count': int,           # 数字字符数
    'url_count': int,             # URL数量
    'mention_count': int,         # 提及数量
    'hashtag_count': int,         # 话题标签数量
    'emoji_count': int            # emoji数量
}
"""
```

### 图像特征
```python
features = processor.extract_image_features(image)
"""
返回图像特征字典:
{
    'width': float,               # 图像宽度
    'height': float,              # 图像高度
    'aspect_ratio': float,        # 宽高比
    'total_pixels': float,        # 总像素数
    'mean_r': float,              # 红色通道均值
    'mean_g': float,              # 绿色通道均值
    'mean_b': float,              # 蓝色通道均值
    'std_r': float,               # 红色通道标准差
    'std_g': float,               # 绿色通道标准差
    'std_b': float,               # 蓝色通道标准差
    'brightness': float,          # 整体亮度
    'contrast': float,            # 对比度
    'edge_density': float         # 边缘密度
}
"""
```

## ⚙️ 配置集成

### 配置文件支持
```python
# 自动从配置文件加载参数
try:
    from utils.config_manager import get_data_config
    config = get_data_config()
    processing_config = config.get('processing', {})
    
    # 文本处理配置
    text_config = processing_config.get('text', {})
    self.max_length = text_config.get('max_length', 512)
    self.remove_urls = text_config.get('remove_urls', True)
    
    # 图像处理配置
    image_config = processing_config.get('image', {})
    self.target_size = image_config.get('target_size', [224, 224])
    self.normalize_mean = image_config.get('normalize_mean', [0.485, 0.456, 0.406])
    
except ImportError:
    # 没有配置管理器时使用默认配置
    pass
```

### 配置参数对应关系
| 配置文件参数 | 功能说明 | 默认值 |
|-------------|----------|--------|
| `processing.text.max_length` | 最大文本长度 | 512 |
| `processing.text.remove_urls` | 是否移除URL | True |
| `processing.text.tokenization` | 分词类型 | "mixed" |
| `processing.image.target_size` | 目标图像尺寸 | [224, 224] |
| `processing.image.normalize_mean` | 归一化均值 | ImageNet标准 |
| `processing.image.quality_threshold` | 质量阈值 | 0.3 |

## 🔍 依赖库管理

### 文本处理依赖
```python
# 必需库
import re, string, emoji
from typing import List, Dict, Optional

# 可选库（自动检测）
try:
    import jieba              # 中文分词
    HAS_JIEBA = True
except ImportError:
    HAS_JIEBA = False
    print("⚠️ jieba未安装，中文分词功能不可用")

try:
    import nltk               # 英文处理
    from nltk.corpus import stopwords
    HAS_NLTK = True
except ImportError:
    HAS_NLTK = False
    print("⚠️ nltk未安装，英文高级处理功能不可用")
```

### 图像处理依赖
```python
# 必需库
import cv2, numpy as np
from PIL import Image, ImageEnhance, ImageFilter
import torch, torchvision.transforms as transforms

# 自动降级处理
try:
    import torchvision.transforms.functional as F
except ImportError:
    print("⚠️ torchvision版本较老，部分功能可能不可用")
```

## 🚨 错误处理和质量控制

### 文本处理错误处理
```python
# 安全的文本处理
def clean_text(self, text: str) -> str:
    if not text or not isinstance(text, str):
        return ""
    
    try:
        # 各种清洗步骤
        cleaned = self._apply_cleaning_rules(text)
        return cleaned
    except Exception as e:
        logger.warning(f"文本清洗失败: {e}")
        return text  # 返回原文本
```

### 图像处理质量控制
```python
# 安全的图像加载
def load_image_safe(self, image_path: str) -> Dict[str, Any]:
    try:
        # 检查文件存在性
        if not full_image_path.exists():
            return self.create_empty_image_result(str(full_image_path))
        
        # 尝试加载图像
        with Image.open(full_image_path) as image:
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            image_tensor = self.image_transforms(image)
            return {
                'image': image_tensor,
                'has_image': True,
                'image_path': str(full_image_path),
                'image_size': image.size
            }
            
    except Exception as e:
        logger.error(f"处理图像失败 {full_image_path}: {e}")
        return self.create_empty_image_result(str(full_image_path))
```

## 💡 性能优化建议

### 文本处理优化
- **批量处理**: 使用`preprocess_batch`方法提高效率
- **缓存分词**: 对于重复文本，缓存分词结果
- **并行处理**: 大数据量时使用多进程处理

### 图像处理优化
- **内存管理**: 及时释放不需要的图像对象
- **批次控制**: 根据内存调整批处理大小
- **格式优化**: 选择合适的图像格式和质量设置

### 配置优化
```python
# 性能优化配置示例
optimized_config = {
    'text': {
        'batch_size': 1000,      # 批处理大小
        'use_cache': True,       # 启用缓存
        'max_length': 256        # 适当减少最大长度
    },
    'image': {
        'target_size': [224, 224],  # 标准尺寸
        'quality_threshold': 0.5,   # 提高质量阈值
        'batch_process': True       # 启用批处理
    }
}
```

---

**[⬅️ 演示脚本](../data_utils/demo.md) | [文本处理 ➡️](text_processing.md)**