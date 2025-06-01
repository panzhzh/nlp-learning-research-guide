# 预处理演示脚本 Demo Script

> 📝 **预处理模块的快速演示和功能验证脚本**

## 📋 脚本概览

`demo.py`提供了预处理模块的简化演示脚本，展示文本和图像处理功能的完整使用流程。

## 🚀 脚本功能

### 主要特性
- **功能演示**: 展示文本和图像预处理的核心功能
- **快速验证**: 验证依赖库和模块是否正常工作
- **错误诊断**: 帮助识别配置和环境问题
- **简化接口**: 提供最直观的使用方式

### 脚本结构
```python
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author: ipanzhzh
# preprocessing/demo.py

"""
预处理模块演示 - 简化版
直接运行即可体验文本和图像预处理
"""

from text_processing import TextProcessor
from image_processing import ImageProcessor

def main():
    """简单演示预处理功能"""
    print("🔧 预处理模块演示")
    print("="*50)
    
    # 文本处理演示
    print("\n📝 文本处理演示:")
    processor = TextProcessor(language='mixed')
    
    test_texts = [
        "这是一个测试文本 This is a test!",
        "包含URL的文本 https://example.com 和@username",
        "混合语言文本 with English words 中文字符"
    ]
    
    for text in test_texts:
        print(f"\n原文: {text}")
        cleaned = processor.clean_text(text)
        tokens = processor.tokenize(text)
        print(f"清洗: {cleaned}")
        print(f"分词: {tokens[:5]}...")  # 只显示前5个
    
    # 图像处理演示
    print("\n🖼️  图像处理演示:")
    img_processor = ImageProcessor(target_size=(224, 224))
    
    # 处理数据集（只处理train，演示用）
    try:
        results = img_processor.process_mr2_dataset(splits=['train'])
        if results:
            print("图像处理完成!")
    except Exception as e:
        print(f"图像处理演示跳过: {e}")
    
    print("\n✅ 预处理演示完成!")

if __name__ == "__main__":
    main()
```

## 🎯 演示内容

### 文本处理演示
演示脚本会展示以下文本处理功能：

#### 1. 多语言文本处理
```python
test_texts = [
    "这是一个测试文本 This is a test!",        # 中英混合
    "包含URL的文本 https://example.com 和@username",  # 包含噪声
    "混合语言文本 with English words 中文字符"   # 复杂混合
]
```

#### 2. 处理步骤展示
```python
for text in test_texts:
    print(f"\n原文: {text}")
    
    # 文本清洗
    cleaned = processor.clean_text(text)
    print(f"清洗: {cleaned}")
    
    # 智能分词
    tokens = processor.tokenize(text)
    print(f"分词: {tokens[:5]}...")  # 只显示前5个token
```

#### 期望的文本处理输出
```
📝 文本处理演示:

原文: 这是一个测试文本 This is a test!
清洗: 这是一个测试文本 This is a test
分词: ['这是', '一个', '测试', '文本', 'this']...

原文: 包含URL的文本 https://example.com 和@username
清洗: 包含URL的文本 和
分词: ['包含', 'url', '文本']...

原文: 混合语言文本 with English words 中文字符
清洗: 混合语言文本 with English words 中文字符
分词: ['混合', '语言', '文本', 'with', 'english']...
```

### 图像处理演示
演示脚本会尝试处理MR2数据集的图像：

#### 1. 图像处理器创建
```python
img_processor = ImageProcessor(target_size=(224, 224))
```

#### 2. 批量处理演示
```python
try:
    # 只处理train split进行演示
    results = img_processor.process_mr2_dataset(splits=['train'])
    if results:
        print("图像处理完成!")
        
        # 显示处理统计
        stats = results['train']
        print(f"处理统计: 成功{stats['processed_images']}, 失败{stats['failed_images']}")
        
except Exception as e:
    print(f"图像处理演示跳过: {e}")
    print("这是正常的，如果没有图像数据的话")
```

#### 期望的图像处理输出
```
🖼️ 图像处理演示:

📂 处理 train 数据集
  已处理 50 张图像
  已处理 100 张图像
  ...

✅ train 数据集处理完成:
  总数: 500
  成功: 450
  失败: 50

💾 图像特征已保存到: data/processed

图像处理完成!
```

## 🚀 快速运行

### 直接运行
```bash
# 进入preprocessing目录
cd preprocessing

# 运行演示脚本
python demo.py
```

### 从项目根目录运行
```bash
# 从项目根目录运行
python -m preprocessing.demo
```

## 🔧 自定义演示

### 扩展文本处理演示
```python
def detailed_text_demo():
    """详细的文本处理演示"""
    print("📝 详细文本处理演示")
    processor = TextProcessor(language='mixed')
    
    test_texts = [
        "这是第一个测试文本",
        "This is the second test text!",
        "混合语言文本 mixed language text 😊",
        "包含@用户提及 #话题标签 https://example.com",
        "特殊标点!!! 问号??? 省略号..."
    ]
    
    for i, text in enumerate(test_texts, 1):
        print(f"\n=== 测试 {i} ===")
        print(f"原文: {text}")
        
        # 语言检测
        language = processor.detect_language(text)
        print(f"语言: {language}")
        
        # 文本清洗
        cleaned = processor.clean_text(text)
        print(f"清洗: {cleaned}")
        
        # 分词处理
        tokens = processor.tokenize(text)
        print(f"分词: {tokens}")
        
        # 特征提取
        features = processor.extract_features(text)
        print(f"特征:")
        print(f"  长度: {features['text_length']}")
        print(f"  词数: {features['token_count']}")
        print(f"  感叹号: {features['exclamation_count']}")
        print(f"  问号: {features['question_count']}")
        print(f"  URL数: {features['url_count']}")
        print(f"  提及数: {features['mention_count']}")
```

### 扩展图像处理演示
```python
def detailed_image_demo():
    """详细的图像处理演示"""
    print("🖼️ 详细图像处理演示")
    processor = ImageProcessor(target_size=(224, 224))
    
    # 查找测试图像
    test_image_dir = Path("../data/train/img")
    if test_image_dir.exists():
        image_files = list(test_image_dir.glob("*.jpg"))[:3]  # 只取3张
        
        for i, image_path in enumerate(image_files, 1):
            print(f"\n=== 图像 {i} ===")
            print(f"路径: {image_path}")
            
            # 获取图像信息
            info = processor.get_image_info(image_path)
            if info:
                print(f"信息:")
                print(f"  尺寸: {info['width']} x {info['height']}")
                print(f"  格式: {info['format']}")
                print(f"  大小: {info['file_size_mb']} MB")
            
            # 处理图像
            tensor = processor.process_single_image(image_path, 'val')
            if tensor is not None:
                print(f"张量形状: {tensor.shape}")
            
            # 提取特征
            image = processor.load_image(image_path)
            if image is not None:
                features = processor.extract_image_features(image)
                print(f"特征:")
                print(f"  亮度: {features.get('brightness', 0):.2f}")
                print(f"  对比度: {features.get('contrast', 0):.2f}")
                print(f"  边缘密度: {features.get('edge_density', 0):.3f}")
    else:
        print("没有找到测试图像目录")
```

### 数据增强演示
```python
def augmentation_demo():
    """数据增强演示"""
    print("🎨 数据增强演示")
    processor = ImageProcessor(target_size=(224, 224))
    
    # 查找一张测试图像
    test_image_dir = Path("../data/train/img")
    if test_image_dir.exists():
        image_files = list(test_image_dir.glob("*.jpg"))
        if image_files:
            test_image_path = image_files[0]
            print(f"测试图像: {test_image_path}")
            
            # 加载原始图像
            original_image = processor.load_image(test_image_path)
            if original_image is not None:
                print("原始图像加载成功")
                
                # 测试不同级别的增强
                augment_levels = ['light', 'medium', 'heavy']
                
                for level in augment_levels:
                    print(f"\n{level.upper()} 增强:")
                    
                    # 应用增强
                    augmented = processor.apply_augmentation(original_image, level)
                    
                    # 转换为张量
                    tensor = processor.val_transforms(augmented)
                    print(f"  增强后张量形状: {tensor.shape}")
                    
                    # 可选：保存增强后的图像
                    output_dir = Path("../outputs/augmented")
                    output_dir.mkdir(parents=True, exist_ok=True)
                    
                    output_path = output_dir / f"{level}_augmented.jpg"
                    augmented.save(output_path)
                    print(f"  保存到: {output_path}")
```

## 🧪 功能验证

### 依赖库检查
```python
def check_dependencies():
    """检查预处理模块的依赖库"""
    print("🔍 检查依赖库...")
    
    dependencies = {
        'jieba': '中文分词',
        'nltk': '英文处理',
        'PIL': '图像处理',
        'cv2': '计算机视觉',
        'torch': 'PyTorch张量',
        'torchvision': '图像变换',
        'numpy': '数值计算'
    }
    
    missing_deps = []
    
    for module, description in dependencies.items():
        try:
            __import__(module)
            print(f"✅ {module}: {description}")
        except ImportError:
            print(f"❌ {module}: {description} - 未安装")
            missing_deps.append(module)
    
    if missing_deps:
        print(f"\n缺失的依赖库: {', '.join(missing_deps)}")
        print("请安装缺失的库以获得完整功能")
    else:
        print("\n✅ 所有依赖库检查通过")
    
    return len(missing_deps) == 0
```

### 配置检查
```python
def check_configuration():
    """检查配置系统"""
    print("⚙️ 检查配置系统...")
    
    try:
        from utils.config_manager import get_data_config
        config = get_data_config()
        print("✅ 配置管理器工作正常")
        
        # 检查关键配置
        processing = config.get('processing', {})
        if processing:
            print("✅ 预处理配置已加载")
        else:
            print("⚠️ 预处理配置为空")
            
    except ImportError:
        print("⚠️ 配置管理器不可用，使用默认配置")
    except Exception as e:
        print(f"❌ 配置系统错误: {e}")
```

## 🚨 错误处理和故障排除

### 常见问题诊断
```python
def diagnose_issues():