# 图像处理器 Image Processing

> 🖼️ **专业的图像预处理、特征提取和批量处理工具，专为MR2数据集优化**

## 📋 功能概览

`ImageProcessor`是专门为MR2数据集设计的图像处理器，提供图像加载、预处理、特征提取和批量处理功能，支持多种数据增强策略。

## 🚀 核心类

### ImageProcessor
主要图像处理类，集成配置管理和批量处理：

```python
from preprocessing import ImageProcessor

# 创建图像处理器
processor = ImageProcessor(target_size=(224, 224))
```

#### 初始化参数
```python
def __init__(self, target_size: Tuple[int, int] = (224, 224)):
    """
    初始化图像处理器
    
    Args:
        target_size: 目标图像尺寸 (height, width)
                    默认(224, 224)适配大多数预训练模型
    """
```

### 自动配置集成
```python
# 自动从配置管理器加载参数
if USE_CONFIG:
    config = get_data_config()
    self.processing_config = config.get('processing', {}).get('image', {})
    self.data_dir = get_data_dir()
else:
    self.processing_config = {}
    self.data_dir = Path('data')

# 设置处理参数
self.normalize_mean = self.processing_config.get('normalize_mean', [0.485, 0.456, 0.406])
self.normalize_std = self.processing_config.get('normalize_std', [0.229, 0.224, 0.225])
self.quality_threshold = self.processing_config.get('quality_threshold', 0.3)
```

## 🔧 图像变换配置

### 自动变换设置
```python
def setup_transforms(self):
    """设置三种类型的图像变换"""
    
    # 1. 训练变换（包含数据增强）
    self.train_transforms = transforms.Compose([
        transforms.Resize(self.target_size),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, 
                              saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=self.normalize_mean, 
                           std=self.normalize_std)
    ])
    
    # 2. 验证变换（无增强）
    self.val_transforms = transforms.Compose([
        transforms.Resize(self.target_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=self.normalize_mean, 
                           std=self.normalize_std)
    ])
    
    # 3. 可视化变换（无归一化）
    self.visual_transforms = transforms.Compose([
        transforms.Resize(self.target_size),
        transforms.ToTensor()
    ])
```

### 变换类型说明
| 变换类型 | 用途 | 特点 |
|----------|------|------|
| `train` | 训练阶段 | 包含随机增强，提高泛化能力 |
| `val` | 验证/测试 | 无随机性，保证结果一致性 |
| `visual` | 可视化 | 无归一化，便于图像显示 |

## 🔄 核心处理方法

### 安全图像加载
```python
def load_image(self, image_path: Union[str, Path], mode: str = 'RGB') -> Optional[Image.Image]:
    """
    安全加载图像文件
    
    Args:
        image_path: 图像文件路径
        mode: 图像模式 ('RGB', 'RGBA', 'L')
        
    Returns:
        PIL Image对象，失败返回None
        
    安全特性:
    - 文件存在性检查
    - 格式自动转换
    - 质量验证
    - 异常处理
    """
```

### 图像质量验证
```python
def validate_image(self, image: Image.Image) -> bool:
    """
    验证图像质量
    
    验证项目:
    - 最小尺寸检查 (50x50)
    - 图像数据完整性
    - 格式有效性
    
    Returns:
        是否通过验证
    """
```

### 单张图像处理
```python
def process_single_image(self, 
                        image_path: Union[str, Path], 
                        transform_type: str = 'val',
                        apply_augment: bool = False,
                        augment_type: str = 'light') -> Optional[torch.Tensor]:
    """
    处理单张图像的完整流程
    
    Args:
        image_path: 图像路径
        transform_type: 变换类型 ('train', 'val', 'visual')
        apply_augment: 是否应用额外增强
        augment_type: 增强级别 ('light', 'medium', 'heavy')
        
    Returns:
        处理后的tensor (3, H, W)，失败返回None
    """
```

## 🎨 数据增强策略

### apply_augmentation 方法
```python
def apply_augmentation(self, image: Image.Image, augment_type: str = 'light') -> Image.Image:
    """应用不同级别的数据增强"""
```

#### 轻度增强 (light)
```python
if augment_type == 'light':
    if np.random.random() > 0.5:
        image = F.hflip(image)  # 50%概率水平翻转
    if np.random.random() > 0.7:
        angle = np.random.uniform(-5, 5)  # ±5度随机旋转
        image = F.rotate(image, angle)
```

#### 中度增强 (medium)  
```python
elif augment_type == 'medium':
    if np.random.random() > 0.5:
        image = F.hflip(image)  # 水平翻转
    if np.random.random() > 0.6:
        angle = np.random.uniform(-10, 10)  # ±10度旋转
        image = F.rotate(image, angle)
    if np.random.random() > 0.6:
        # 亮度调整 (0.8-1.2倍)
        enhancer = ImageEnhance.Brightness(image)
        factor = np.random.uniform(0.8, 1.2)
        image = enhancer.enhance(factor)
```

#### 重度增强 (heavy)
```python
elif augment_type == 'heavy':
    # 水平翻转 (50%概率)
    if np.random.random() > 0.5:
        image = F.hflip(image)
        
    # 旋转 (±15度)
    if np.random.random() > 0.5:
        angle = np.random.uniform(-15, 15)
        image = F.rotate(image, angle)
        
    # 亮度调整 (0.7-1.3倍)
    if np.random.random() > 0.5:
        enhancer = ImageEnhance.Brightness(image)
        factor = np.random.uniform(0.7, 1.3)
        image = enhancer.enhance(factor)
        
    # 对比度调整 (0.8-1.2倍)
    if np.random.random() > 0.5:
        enhancer = ImageEnhance.Contrast(image)
        factor = np.random.uniform(0.8, 1.2)
        image = enhancer.enhance(factor)
        
    # 轻微高斯模糊
    if np.random.random() > 0.3:
        image = image.filter(ImageFilter.GaussianBlur(radius=0.5))
```

## 📊 特征提取

### extract_image_features 方法
```python
def extract_image_features(self, image: Image.Image) -> Dict[str, float]:
    """
    提取全面的图像特征
    
    特征类别:
    - 几何特征: 宽度、高度、纵横比、像素总数
    - 颜色特征: RGB均值和标准差
    - 质量特征: 亮度、对比度
    - 纹理特征: 边缘密度
    
    Returns:
        图像特征字典
    """
```

### 完整特征说明
```python
features = {
    # === 几何特征 ===
    'width': 640.0,                    # 图像宽度
    'height': 480.0,                   # 图像高度  
    'aspect_ratio': 1.33,              # 宽高比
    'total_pixels': 307200.0,          # 总像素数
    
    # === RGB颜色特征 ===
    'mean_r': 128.5,                   # 红色通道均值
    'mean_g': 132.1,                   # 绿色通道均值
    'mean_b': 125.8,                   # 蓝色通道均值
    'std_r': 45.2,                     # 红色通道标准差
    'std_g': 48.1,                     # 绿色通道标准差
    'std_b': 42.9,                     # 蓝色通道标准差
    
    # === 图像质量特征 ===
    'brightness': 128.8,               # 整体亮度(0-255)
    'contrast': 45.4,                  # 对比度(标准差)
    'edge_density': 0.12               # 边缘密度(0-1)
}
```

### 边缘密度计算
```python
# 使用OpenCV Canny边缘检测
try:
    edges = cv2.Canny(gray.astype(np.uint8), 50, 150)
    features['edge_density'] = float(np.sum(edges > 0) / edges.size)
except:
    features['edge_density'] = 0.0
```

## 📦 批量处理功能

### MR2数据集专用批量处理
```python
def process_mr2_dataset(self, 
                       splits: List[str] = ['train', 'val', 'test'], 
                       save_features: bool = True) -> Dict[str, Dict]:
    """
    处理完整的MR2数据集
    
    Args:
        splits: 要处理的数据划分
        save_features: 是否保存特征到文件
        
    Returns:
        处理结果字典:
        {
            'train': {
                'total_items': int,
                'processed_images': int,
                'failed_images': int,
                'image_info': Dict,      # 图像基本信息
                'image_features': Dict   # 提取的特征
            }
        }
    """
```

### 批量处理流程
```python
# 完整的批量处理流程
for split in splits:
    print(f"\n📂 处理 {split} 数据集")
    
    # 1. 加载数据集信息
    dataset_file = self.data_dir / f'dataset_items_{split}.json'
    with open(dataset_file, 'r') as f:
        dataset_items = json.load(f)
    
    # 2. 处理每个数据项
    for item_id, item_data in dataset_items.items():
        if 'image_path' not in item_data:
            continue
            
        image_path = self.data_dir / item_data['image_path']
        
        try:
            # 获取图像信息
            img_info = self.get_image_info(image_path)
            split_results['image_info'][item_id] = img_info
            
            # 提取图像特征
            image = self.load_image(image_path)
            if image is not None:
                features = self.extract_image_features(image)
                split_results['image_features'][item_id] = features
                split_results['processed_images'] += 1
            else:
                split_results['failed_images'] += 1
                
        except Exception as e:
            logger.error(f"处理图像失败 {image_path}: {e}")
            split_results['failed_images'] += 1
    
    # 3. 保存特征(可选)
    if save_features:
        self.save_image_features(split_results, split)
```

### 特征保存
```python
def save_image_features(self, features_data: Dict, split: str):
    """
    保存图像特征到文件
    
    保存位置:
    - data/processed/{split}_image_info.json     # 图像基本信息
    - data/processed/{split}_image_features.json # 提取的特征
    """
    processed_dir = self.data_dir / 'processed'
    processed_dir.mkdir(exist_ok=True)
    
    # 保存图像信息
    info_file = processed_dir / f'{split}_image_info.json'
    with open(info_file, 'w', encoding='utf-8') as f:
        json.dump(features_data['image_info'], f, indent=2, ensure_ascii=False)
    
    # 保存图像特征
    features_file = processed_dir / f'{split}_image_features.json'
    with open(features_file, 'w', encoding='utf-8') as f:
        json.dump(features_data['image_features'], f, indent=2, ensure_ascii=False)
```

## 📈 统计分析

### create_image_statistics 方法
```python
def create_image_statistics(self, results: Dict) -> Dict[str, Any]:
    """
    创建图像处理统计信息
    
    统计内容:
    - 图像数量统计
    - 尺寸分布分析
    - 格式分布统计
    - 文件大小分析
    - 处理成功率
    """
```

### 统计信息示例
```python
statistics = {
    'total_images': 900,               # 总图像数
    'successful_images': 810,          # 成功处理数
    'failed_images': 90,               # 处理失败数
    'success_rate': 0.9,               # 成功率
    
    'size_distribution': {
        'avg_width': 640.5,            # 平均宽度
        'avg_height': 480.2,           # 平均高度
        'max_width': 1920,             # 最大宽度
        'min_width': 128,              # 最小宽度
    },
    
    'format_distribution': {
        'JPEG': 720,                   # JPEG格式数量
        'PNG': 90                      # PNG格式数量
    },
    
    'avg_file_size': 125.6,            # 平均文件大小(KB)
    'total_file_size': 112.8           # 总文件大小(MB)
}
```

## 🎯 使用示例

### 基础使用
```python
from preprocessing import ImageProcessor

# 创建处理器
processor = ImageProcessor(target_size=(224, 224))

# 处理单张图像
image_path = 'data/train/img/example.jpg'

# 1. 获取图像信息
info = processor.get_image_info(image_path)
print(f"图像信息:")
print(f"  尺寸: {info['width']} x {info['height']}")
print(f"  格式: {info['format']}")
print(f"  文件大小: {info['file_size_mb']:.2f} MB")

# 2. 处理图像
tensor = processor.process_single_image(
    image_path, 
    transform_type='train'  # 包含数据增强
)
print(f"处理后张量形状: {tensor.shape}")  # torch.Size([3, 224, 224])

# 3. 提取特征
image = processor.load_image(image_path)
features = processor.extract_image_features(image)
print(f"图像特征:")
print(f"  亮度: {features['brightness']:.2f}")
print(f"  对比度: {features['contrast']:.2f}")
print(f"  边缘密度: {features['edge_density']:.3f}")
```

### 数据增强测试
```python
# 测试不同级别的数据增强
image = processor.load_image('data/train/img/example.jpg')

# 轻度增强
light_aug = processor.apply_augmentation(image, 'light')
processor.save_image_with_info(light_aug, 'outputs/light_augmented.jpg')

# 中度增强
medium_aug = processor.apply_augmentation(image, 'medium')
processor.save_image_with_info(medium_aug, 'outputs/medium_augmented.jpg')

# 重度增强
heavy_aug = processor.apply_augmentation(image, 'heavy')
processor.save_image_with_info(heavy_aug, 'outputs/heavy_augmented.jpg')
```

### 批量处理MR2数据集
```python
# 处理完整数据集
print("🔄 开始批量处理MR2数据集...")

results = processor.process_mr2_dataset(
    splits=['train', 'val', 'test'],
    save_features=True
)

# 查看处理结果
for split, stats in results.items():
    success_rate = stats['processed_images'] / stats['total_items'] * 100
    print(f"{split.upper()}:")
    print(f"  总数: {stats['total_items']}")
    print(f"  成功: {stats['processed_images']}")
    print(f"  失败: {stats['failed_images']}")
    print(f"  成功率: {success_rate:.1f}%")

# 创建统计信息
statistics = processor.create_image_statistics(results)
print(f"\n📊 整体统计:")
print(f"总图像数: {statistics['total_images']}")
print(f"成功处理: {statistics['successful_images']}")
print(f"平均尺寸: {statistics.get('avg_width', 0):.0f} x {statistics.get('avg_height', 0):.0f}")
print(f"格式分布: {statistics.get('format_distribution', {})}")
```

### 自定义变换配置
```python
import torchvision.transforms as transforms

# 创建自定义变换
custom_transforms = transforms.Compose([
    transforms.Resize((256, 256)),           # 先放大
    transforms.RandomCrop((224, 224)),       # 随机裁剪
    transforms.RandomHorizontalFlip(p=0.7),  # 更高翻转概率
    transforms.ColorJitter(
        brightness=0.3, 
        contrast=0.3, 
        saturation=0.2, 
        hue=0.1
    ),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406], 
        std=[0.229, 0.224, 0.225]
    )
])

# 应用自定义变换
processor.train_transforms = custom_transforms

# 处理图像
tensor = processor.process_single_image(
    'data/train/img/example.jpg', 
    transform_type='train'
)
```

## ⚡ 性能优化

### 批量处理优化
```python
def batch_process_images(self, 
                        image_paths: List[Union[str, Path]], 
                        transform_type: str = 'val',
                        batch_size: int = 32) -> List[torch.Tensor]:
    """
    高效的批量图像处理
    
    优化策略:
    - 分批处理减少内存占用
    - 并行加载和预处理
    - 自动错误恢复
    """
    processed_tensors = []
    
    for i in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[i:i+batch_size]
        batch_tensors = []
        
        for path in batch_paths:
            tensor = self.process_single_image(path, transform_type)
            if tensor is not None:
                batch_tensors.append(tensor)
        
        if batch_tensors:
            # 堆叠tensor以节省内存
            batch_tensor = torch.stack(batch_tensors)
            processed_tensors.append(batch_tensor)
            
        # 可选：显示进度
        if (i // batch_size) % 10 == 0:
            print(f"已处理: {i + len(batch_paths)}/{len(image_paths)}")
    
    return processed_tensors
```

### 内存优化策略
```python
# 1. 及时释放图像对象
def process_with_memory_optimization(self, image_path):
    try:
        image = self.load_image(image_path)
        
        # 提取特征
        features = self.extract_image_features(image)
        
        # 处理图像
        tensor = self.image_transforms(image)
        
        # 及时删除image对象
        del image
        
        return tensor, features
    except Exception as e:
        return None, {}

# 2. 分块处理大数据集
def process_large_dataset_chunked(self, dataset_size, chunk_size=1000):
    """分块处理大数据集以避免内存溢出"""
    for chunk_start in range(0, dataset_size, chunk_size):
        chunk_end = min(chunk_start + chunk_size, dataset_size)
        
        # 处理当前块
        self._process_chunk(chunk_start, chunk_end)
        
        # 强制垃圾回收
        import gc
        gc.collect()
```

### 质量控制优化
```python
def enhanced_quality_control(self, image_path):
    """增强的图像质量控制"""
    
    # 1. 文件级检查
    if not os.path.exists(image_path):
        return False, "文件不存在"
    
    file_size = os.path.getsize(image_path)
    if file_size < 1024:  # 小于1KB
        return False, "文件过小"
    
    # 2. 图像级检查
    try:
        with Image.open(image_path) as image:
            # 尺寸检查
            if image.width < 50 or image.height < 50:
                return False, "图像尺寸过小"
            
            # 宽高比检查
            aspect_ratio = image.width / image.height
            if aspect_ratio > 10 or aspect_ratio < 0.1:
                return False, "宽高比异常"
            
            # 颜色通道检查
            if len(image.getbands()) not in [1, 3, 4]:
                return False, "颜色通道异常"
                
        return True, "质量检查通过"
        
    except Exception as e:
        return False, f"图像损坏: {e}"
```

## 🔧 高级功能

### 图像尺寸智能调整
```python
def resize_image(self, 
                image: Image.Image, 
                size: Optional[Tuple[int, int]] = None, 
                method: str = 'lanczos') -> Image.Image:
    """
    智能图像尺寸调整
    
    Args:
        image: PIL Image对象
        size: 目标尺寸，None时使用self.target_size
        method: 重采样方法
                'lanczos' - 高质量(默认)
                'bilinear' - 平衡质量和速度
                'bicubic' - 高质量慢速
    """
    if size is None:
        size = self.target_size
    
    # 选择重采样方法
    resample_methods = {
        'lanczos': Image.Resampling.LANCZOS,
        'bilinear': Image.Resampling.BILINEAR,
        'bicubic': Image.Resampling.BICUBIC
    }
    resample = resample_methods.get(method, Image.Resampling.LANCZOS)
    
    return image.resize(size, resample)
```

### 图像信息获取增强
```python
def get_image_info(self, image_path: Union[str, Path]) -> Dict[str, Any]:
    """
    获取详细的图像信息
    
    Returns:
        {
            'path': str,              # 图像路径
            'filename': str,          # 文件名
            'format': str,            # 图像格式
            'mode': str,              # 颜色模式
            'size': tuple,            # 尺寸 (width, height)
            'width': int,             # 宽度
            'height': int,            # 高度
            'file_size': int,         # 文件大小(字节)
            'file_size_mb': float,    # 文件大小(MB)
            'aspect_ratio': float,    # 宽高比
            'color_channels': int,    # 颜色通道数
            'has_transparency': bool, # 是否有透明通道
            'creation_time': str      # 创建时间(如果有)
        }
    """
    try:
        with Image.open(image_path) as image:
            file_size = os.path.getsize(image_path)
            
            # 基础信息
            info = {
                'path': str(image_path),
                'filename': os.path.basename(image_path),
                'format': image.format,
                'mode': image.mode,
                'size': image.size,
                'width': image.width,
                'height': image.height,
                'file_size': file_size,
                'file_size_mb': round(file_size / (1024 * 1024), 2),
                'aspect_ratio': round(image.width / image.height, 2)
            }
            
            # 高级信息
            info['color_channels'] = len(image.getbands())
            info['has_transparency'] = 'transparency' in image.info or image.mode in ('RGBA', 'LA')
            
            # 尝试获取EXIF信息
            try:
                exif = image._getexif()
                if exif and 306 in exif:  # DateTime标签
                    info['creation_time'] = exif[306]
            except:
                pass
                
            return info
            
    except Exception as e:
        logger.error(f"获取图像信息失败 {image_path}: {e}")
        return {}
```

## 🚨 错误处理和日志

### 完善的错误处理
```python
def load_image_safe(self, image_path: str) -> Dict[str, Any]:
    """
    安全的图像加载，包含完整错误处理
    """
    full_image_path = self.data_dir / image_path
    
    try:
        # 1. 文件存在性检查
        if not full_image_path.exists():
            logger.warning(f"图像文件不存在: {full_image_path}")
            return self.create_empty_image_result(str(full_image_path))
        
        # 2. 文件大小检查
        file_size = full_image_path.stat().st_size
        if file_size == 0:
            logger.warning(f"图像文件为空: {full_image_path}")
            return self.create_empty_image_result(str(full_image_path))
        
        # 3. 尝试加载图像
        with Image.open(full_image_path) as image:
            # 转换为RGB模式
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # 质量检查
            if not self.validate_image(image):
                logger.warning(f"图像质量检查未通过: {full_image_path}")
                return self.create_empty_image_result(str(full_image_path))
            
            # 应用变换
            image_tensor = self.image_transforms(image)
            
            return {
                'image': image_tensor,
                'has_image': True,
                'image_path': str(full_image_path),
                'image_size': image.size,
                'load_status': 'success'
            }
            
    except OSError as e:
        logger.error(f"文件系统错误 {full_image_path}: {e}")
        return self.create_empty_image_result(str(full_image_path))
    except Image.UnidentifiedImageError as e:
        logger.error(f"无法识别的图像格式 {full_image_path}: {e}")
        return self.create_empty_image_result(str(full_image_path))
    except Exception as e:
        logger.error(f"处理图像失败 {full_image_path}: {e}")
        return self.create_empty_image_result(str(full_image_path))

def create_empty_image_result(self, image_path: str) -> Dict[str, Any]:
    """创建空图像结果，用于错误情况"""
    return {
        'image': torch.zeros(3, *self.target_size),
        'has_image': False,
        'image_path': image_path,
        'image_size': None,
        'load_status': 'failed'
    }
```

## 🧪 测试和验证

### 内置测试功能
```python
def test_image_processor():
    """完整的图像处理器测试"""
    print("🖼️ 测试图像处理模块")
    
    processor = ImageProcessor(target_size=(224, 224))
    
    # 1. 测试图像信息获取
    test_image_dir = Path("data/train/img")
    if test_image_dir.exists():
        image_files = list(test_image_dir.glob("*.jpg"))
        if image_files:
            test_image = image_files[0]
            print(f"测试图像: {test_image}")
            
            # 获取图像信息
            img_info = processor.get_image_info(test_image)
            if img_info:
                print(f"图像信息: {img_info}")
            
            # 处理图像
            tensor = processor.process_single_image(test_image, transform_type='val')
            if tensor is not None:
                print(f"处理结果tensor形状: {tensor.shape}")
                
            # 提取特征
            image = processor.load_image(test_image)
            if image is not None:
                features = processor.extract_image_features(image)
                print(f"图像特征数量: {len(features)}")
                
    # 2. 测试批量处理
    print("\n🔄 测试批量处理...")
    try:
        results = processor.process_mr2_dataset(splits=['train'], save_features=False)
        if results:
            stats = results['train']
            print(f"批量处理结果: 成功{stats['processed_images']}, 失败{stats['failed_images']}")
    except Exception as e:
        print(f"批量处理测试失败: {e}")
    
    print("✅ 图像处理模块测试完成")

if __name__ == "__main__":
    test_image_processor()
```

## 💡 最佳实践

### 1. 目标尺寸选择
```python
# 根据模型选择合适的尺寸
model_target_sizes = {
    'resnet': (224, 224),      # ResNet标准尺寸
    'efficientnet': (224, 224), # EfficientNet-B0
    'vit': (224, 224),         # Vision Transformer
    'clip': (224, 224),        # CLIP标准尺寸
    'swin': (224, 224),        # Swin Transformer
    'custom_small': (128, 128), # 自定义小尺寸
    'custom_large': (384, 384)  # 自定义大尺寸
}

target_size = model_target_sizes.get('resnet', (224, 224))
processor = ImageProcessor(target_size=target_size)
```

### 2. 数据增强策略选择
```python
# 根据数据集大小选择增强策略
def choose_augmentation_strategy(dataset_size):
    if dataset_size < 1000:
        return 'heavy'    # 小数据集用重度增强
    elif dataset_size < 5000:
        return 'medium'   # 中等数据集用中度增强
    else:
        return 'light'    # 大数据集用轻度增强

# 应用策略
augment_type = choose_augmentation_strategy(len(dataset))
```

### 3. 质量控制配置
```python
# 严格的质量控制
strict_processor = ImageProcessor(target_size=(224, 224))
strict_processor.quality_threshold = 0.8  # 提高质量阈值

# 宽松的质量控制（适用于数据稀缺情况）
lenient_processor = ImageProcessor(target_size=(224, 224))
lenient_processor.quality_threshold = 0.2  # 降低质量阈值
```

## ⚠️ 注意事项

### 依赖库要求
- **PIL (Pillow)**: 9.0+，图像基础处理
- **OpenCV**: 4.5+，边缘检测和高级处理
- **PyTorch**: 1.10+，张量处理和变换
- **torchvision**: 0.11+，预定义变换
- **numpy**: 1.20+，数值计算

### 内存管理
- 大图像处理时注意内存占用
- 批量处理时控制batch_size
- 及时释放不需要的图像对象
- 使用适当的图像压缩质量

### 性能考虑
- JPEG格式通常比PNG处理更快
- 较小的target_size能提升处理速度
- 多进程处理大数据集时注意系统资源

---

**[⬅️ 文本处理](text_processing.md) | [演示脚本 ➡️](demo.md)**