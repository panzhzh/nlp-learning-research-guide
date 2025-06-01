# 演示脚本 Demo Script

> 📝 **预处理功能的快速演示和测试脚本**

## 📋 功能说明

`demo.py` 提供预处理模块的快速演示脚本，展示文本和图像预处理功能的完整使用流程，适合快速测试和功能验证。

## 🎯 主要功能

### 演示功能
- **文本处理演示**: 展示文本清洗、分词、特征提取
- **图像处理演示**: 展示图像加载、预处理、批量处理
- **功能对比**: 展示不同参数配置的效果
- **性能测试**: 简单的处理速度和质量评估

### 测试功能
- **依赖库检查**: 验证所需库是否正确安装
- **数据文件检查**: 检查测试数据是否存在
- **功能完整性测试**: 验证各项功能是否正常工作
- **错误处理测试**: 测试异常情况的处理能力

## 🚀 脚本结构

### main() 函数
主要演示流程:
1. 文本处理器功能演示
2. 图像处理器功能演示
3. 性能和质量展示
4. 错误处理示例

### 演示内容
- **文本处理**: 多语言文本处理示例
- **图像处理**: 单张和批量图像处理
- **特征提取**: 文本和图像特征展示
- **配置对比**: 不同参数设置的效果

## 💡 使用方法

### 直接运行
```bash
# 进入preprocessing目录
cd preprocessing

# 运行演示脚本
python demo.py
```

### 分模块演示
```python
from preprocessing.demo import main

# 运行完整演示
main()
```

### 自定义演示
```python
from preprocessing import TextProcessor, ImageProcessor

# 创建处理器
text_processor = TextProcessor(language='mixed')
image_processor = ImageProcessor(target_size=(224, 224))

# 自定义测试
test_texts = ["Your test texts here"]
test_images = ["path/to/test/image.jpg"]
```

## 📊 演示输出

### 文本处理演示输出
```
📝 文本处理演示:

测试 1: 这是一个测试文本 This is a test text!
  语言: mixed
  清洗后: 这是一个测试文本 This is a test text
  分词结果: ['这是', '一个', '测试', '文本', 'this', 'test', 'text']
  特征: 长度=42, 词数=7, 语言=mixed

测试 2: 今天天气不错，适合出门游玩。
  语言: chinese
  清洗后: 今天天气不错，适合出门游玩。
  分词结果: ['今天', '天气', '不错', '适合', '出门', '游玩']
  特征: 长度=14, 词数=6, 语言=chinese
```

### 图像处理演示输出
```
🖼️ 图像处理演示:

测试图像: data/train/img/example.jpg
图像信息: {'width': 640, 'height': 480, 'format': 'JPEG', 'file_size_mb': 0.15}
处理结果tensor形状: torch.Size([3, 224, 224])
图像特征: {'brightness': 128.5, 'contrast': 45.2, 'edge_density': 0.12}

批量处理: 已处理 50 张图像
```

## 🔧 演示配置

### 测试文本样例
```python
test_texts = [
    "这是一个测试文本 This is a test text!",
    "今天天气不错，适合出门游玩。",
    "Breaking news: AI technology advances rapidly!",
    "混合语言文本 with English words and 中文字符",
    "包含URL的文本 https://example.com 和@username提及",
    "带有emoji的文本 😊 和 #hashtag 标签"
]
```

### 处理器配置
```python
# 文本处理器配置
text_processor = TextProcessor(language='mixed')
text_processor.remove_urls = True
text_processor.remove_mentions = True

# 图像处理器配置
image_processor = ImageProcessor(target_size=(224, 224))
```

## 📝 示例代码

### 文本处理演示
```python
def demo_text_processing():
    """演示文本处理功能"""
    print("📝 文本处理演示:")
    
    processor = TextProcessor(language='mixed')
    
    test_texts = [
        "这是一个测试文本 This is a test!",
        "包含URL的文本 https://example.com 和@username",
        "混合语言文本 with English words 中文字符"
    ]
    
    for i, text in enumerate(test_texts, 1):
        print(f"\n测试 {i}: {text}")
        
        # 语言检测
        language = processor.detect_language(text)
        print(f"  语言: {language}")
        
        # 文本清洗
        cleaned = processor.clean_text(text)
        print(f"  清洗后: {cleaned}")
        
        # 分词
        tokens = processor.tokenize(text)
        print(f"  分词结果: {tokens[:5]}...")  # 只显示前5个
        
        # 特征提取
        features = processor.extract_features(text)
        print(f"  特征: 长度={features['text_length']}, "
              f"词数={features['token_count']}, "
              f"语言={features['language']}")
```

### 图像处理演示
```python
def demo_image_processing():
    """演示图像处理功能"""
    print("🖼️ 图像处理演示:")
    
    processor = ImageProcessor(target_size=(224, 224))
    
    # 处理单张图像（如果存在）
    test_image_dir = Path("../data/train/img")
    if test_image_dir.exists():
        image_files = list(test_image_dir.glob("*.jpg"))
        if image_files:
            test_image = image_files[0]
            print(f"测试图像: {test_image}")
            
            # 获取图像信息
            img_info = processor.get_image_info(test_image)
            print(f"图像信息: {img_info}")
            
            # 处理图像
            tensor = processor.process_single_image(test_image)
            if tensor is not None:
                print(f"处理结果tensor形状: {tensor.shape}")
```

## 🧪 测试功能

### 依赖库检查
```python
def check_dependencies():
    """检查依赖库是否安装"""
    try:
        import jieba
        import nltk
        import PIL
        import cv2
        import torch
        import torchvision
        print("✅ 所有依赖库检查通过")
        return True
    except ImportError as e:
        print(f"❌ 依赖库缺失: {e}")
        return False
```

### 功能完整性测试
```python
def test_functionality():
    """测试功能完整性"""
    # 文本处理测试
    text_processor = TextProcessor()
    test_text = "测试文本 test text"
    
    try:
        cleaned = text_processor.clean_text(test_text)
        tokens = text_processor.tokenize(test_text)
        features = text_processor.extract_features(test_text)
        print("✅ 文本处理功能正常")
    except Exception as e:
        print(f"❌ 文本处理错误: {e}")
    
    # 图像处理测试
    image_processor = ImageProcessor()
    try:
        # 创建测试图像张量
        test_tensor = torch.zeros(3, 224, 224)
        print("✅ 图像处理功能正常")
    except Exception as e:
        print(f"❌ 图像处理错误: {e}")
```

## 📈 性能测试

### 处理速度测试
```python
def benchmark_performance():
    """简单的性能基准测试"""
    import time
    
    # 文本处理速度
    processor = TextProcessor()
    test_texts = ["测试文本"] * 100
    
    start_time = time.time()
    for text in test_texts:
        processor.tokenize(text)
    text_time = time.time() - start_time
    
    print(f"文本处理速度: {len(test_texts)/text_time:.1f} 文本/秒")
```

## ⚠️ 使用说明

### 运行环境要求
- Python 3.7+
- 已安装项目依赖库
- 足够的内存进行图像处理

### 测试数据要求
- 文本测试: 内置测试文本，无需额外数据
- 图像测试: 需要MR2数据集或测试图像文件
- 批量测试: 需要完整的数据目录结构

### 常见问题
1. **依赖库缺失**: 运行前检查并安装所需库
2. **数据文件缺失**: 图像演示需要测试图像文件
3. **内存不足**: 大批量处理时注意内存使用
4. **编码问题**: 确保文本文件使用UTF-8编码

### 故障排除
- 检查Python版本和依赖库版本
- 确认数据文件路径正确
- 查看控制台错误信息
- 调整批处理大小减少内存占用

---

**[⬅️ 图像处理](image_processing.md) | [模型库模块 ➡️](../models/README.md)**
