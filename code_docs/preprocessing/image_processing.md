# 图像处理器 Image Processing

> 🖼️ **图像预处理、特征提取和批量处理工具**

## 📋 功能说明

`ImageProcessor` 专门处理MR2数据集中的图像数据，提供图像加载、预处理、特征提取和批量处理功能。

## 🎯 主要功能

### 图像加载功能
- **格式支持**: JPEG、PNG、BMP等主流图像格式
- **质量验证**: 自动检测和过滤损坏图像
- **模式转换**: 自动转换为RGB模式
- **错误处理**: 完善的异常处理和恢复机制

### 预处理功能
- **尺寸调整**: 统一调整到目标尺寸
- **归一化**: ImageNet标准归一化
- **数据增强**: 训练时的随机变换
- **张量转换**: PIL到PyTorch张量转换

### 特征提取功能
- **颜色特征**: RGB均值、标准差、亮度、对比度
- **几何特征**: 宽度、高度、纵横比、像素总数
- **纹理特征**: 边缘密度等底层特征
- **质量特征**: 文件大小、格式信息

### 批量处理功能
- **数据集处理**: 批量处理MR2数据集图像
- **特征保存**: 自动保存提取的特征
- **进度跟踪**: 处理进度显示和统计
- **结果汇总**: 生成处理统计报告

## 🚀 核心类和方法

### ImageProcessor 类

#### 初始化方法
```python
ImageProcessor(target_size=(224, 224))
```

**参数说明:**
- `target_size`: 目标图像尺寸 (height, width)

#### 图像加载方法
- `load_image(image_path, mode='RGB')`: 安全加载图像
- `validate_image(image)`: 验证图像质量
- `get_image_info(image_path)`: 获取图像基础信息

#### 预处理方法
- `resize_image(image, size, method)`: 调整图像尺寸
- `apply_augmentation(image, augment_type)`: 应用数据增强
- `process_single_image(image_path, transform_type)`: 处理单张图像

#### 特征提取方法
- `extract_image_features(image)`: 提取图像特征
- `create_image_statistics(results)`: 创建统计信息

#### 批量处理方法
- `process_mr2_dataset(splits, save_features)`: 处理MR2数据集
- `batch_process_images(image_paths, transform_type)`: 批量处理图像
- `save_image_features(features_data, split)`: 保存特征数据

## 🔧 图像变换配置

### 训练时变换 (transform_type='train')
```python
transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, 
                           saturation=0.2, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                        std=[0.229, 0.224, 0.225])
])
```

### 验证时变换 (transform_type='val')
```python
transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                        std=[0.229, 0.224, 0.225])
])
```

### 可视化变换 (transform_type='visual')
```python
transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
    # 不进行归一化，便于可视化
])
```

## 📊 返回数据格式

### 图像信息
```python
info = processor.get_image_info(image_path)
# 返回字典:
{
    'path': str,              # 图像路径
    'filename': str,          # 文件名
    'format': str,            # 图像格式 (JPEG, PNG等)
    'mode': str,              # 颜色模式 (RGB, RGBA等)
    'size': tuple,            # 图像尺寸 (width, height)
    'width': int,             # 图像宽度
    'height': int,            # 图像高度
    'file_size': int,         # 文件大小 (字节)
    'file_size_mb': float,    # 文件大小 (MB)
    'aspect_ratio': float     # 宽高比
}
```

### 图像特征
```python
features = processor.extract_image_features(image)
# 返回字典:
{
    'width': float,           # 图像宽度
    'height': float,          # 图像高度
    'aspect_ratio': float,    # 宽高比
    'total_pixels': float,    # 总像素数
    'mean_r': float,          # 红色通道均值
    'mean_g': float,          # 绿色通道均值
    'mean_b': float,          # 蓝色通道均值
    'std_r': float,           # 红色通道标准差
    'std_g': float,           # 绿色通道标准差
    'std_b': float,           # 蓝色通道标准差
    'brightness': float,      # 整体亮度
    'contrast': float,        # 对比度
    'edge_density': float     # 边缘密度
}
```

### 批量处理结果
```python
results = processor.process_mr2_dataset(splits=['train'])
# 返回字典:
{
    'train': {
        'total_items': int,           # 总数据项数
        'processed_images': int,      # 成功处理的图像数
        'failed_images': int,         # 处理失败的图像数
        'image_info': Dict,           # 图像信息字典
        'image_features': Dict        # 图像特征字典
    }
}
```

## 💡 使用示例

### 单张图像处理
```python
from preprocessing import ImageProcessor

# 创建处理器
processor = ImageProcessor(target_size=(224, 224))

# 获取图像信息
info = processor.get_image_info('path/to/image.jpg')
print(f"图像尺寸: {info['width']} x {info['height']}")
print(f"文件大小: {info['file_size_mb']:.2f} MB")

# 处理图像
tensor = processor.process_single_image(
    image_path='path/to/image.jpg',
    transform_type='train'
)
print(f"处理后张量形状: {tensor.shape}")
```

### 特征提取
```python
# 加载图像
image = processor.load_image('path/to/image.jpg')

# 提取特征
features = processor.extract_image_features(image)
print(f"图像亮度: {features['brightness']:.2f}")
print(f"对比度: {features['contrast']:.2f}")
print(f"边缘密度: {features['edge_density']:.3f}")
```

### 批量处理数据集
```python
# 处理整个数据集
results = processor.process_mr2_dataset(
    splits=['train', 'val', 'test'],
    save_features=True
)

# 查看处理结果
for split, stats in results.items():
    print(f"{split}: {stats['processed_images']}/{stats['total_items']} 成功")

# 创建统计信息
statistics = processor.create_image_statistics(results)
print(f"总图像数: {statistics['total_images']}")
print(f"平均尺寸: {statistics['avg_width']:.1f} x {statistics['avg_height']:.1f}")
```

### 数据增强配置
```python
# 轻度增强
image = processor.load_image('path/to/image.jpg')
augmented = processor.apply_augmentation(image, augment_type='light')

# 重度增强
heavy_augmented = processor.apply_augmentation(image, augment_type='heavy')

# 自定义变换
processor.train_transforms = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomCrop((224, 224)),
    transforms.RandomHorizontalFlip(p=0.7),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                        std=[0.229, 0.224, 0.225])
])
```

## 🔧 性能优化

### 批量处理优化
- **批次大小**: 根据内存调整批处理大小
- **多进程**: 使用多进程并行处理大数据集
- **缓存机制**: 缓存常用的变换和特征

### 内存优化
- **惰性加载**: 按需加载图像数据
- **及时释放**: 处理完成后及时释放内存
- **格式优化**: 选择合适的图像格式和质量

### 质量控制
- **尺寸检查**: 过滤过小或过大的图像
- **格式验证**: 检查图像格式完整性
- **内容检查**: 检测空白或损坏图像

## 🔄 数据增强策略

### 轻度增强 (light)
- 50%概率水平翻转
- ±5度随机旋转

### 中度增强 (medium)  
- 50%概率水平翻转
- ±10度随机旋转
- 亮度调整 (0.8-1.2倍)

### 重度增强 (heavy)
- 50%概率水平翻转
- ±15度随机旋转
- 亮度调整 (0.7-1.3倍)
- 对比度调整 (0.8-1.2倍)
- 轻微高斯模糊

## ⚠️ 重要说明

### 图像格式支持
- **推荐格式**: JPEG, PNG
- **支持格式**: BMP, TIFF等PIL支持的格式
- **不支持**: 动图(GIF)、视频格式

### 内存使用
- 大图像处理时注意内存占用
- 批量处理时控制并发数量
- 及时释放不需要的图像对象

### 质量要求
- 最小尺寸: 50x50像素
- 最大尺寸: 2000x2000像素
- 文件大小: 建议小于10MB

---

**[⬅️ 文本处理](text_processing.md) | [演示脚本 ➡️](demo.md)**
