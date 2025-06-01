# MR2数据集类 MR2Dataset

> 📚 **严格验证的PyTorch数据集类，专为MR2多模态谣言检测设计**

## 📋 类概览

`MR2Dataset`是严格验证的PyTorch Dataset实现，只支持真实数据集，提供完整的多模态数据处理功能。

```python
from data_utils import MR2Dataset

# 严格模式：必须使用真实数据集
dataset = MR2Dataset(
    data_dir='data',
    split='train',
    transform_type='train',
    load_images=True
)
```

## 🚀 核心功能

### 初始化参数
```python
def __init__(self, 
             data_dir: Union[str, Path],
             split: str = 'train',
             transform_type: str = 'train', 
             target_size: Tuple[int, int] = (224, 224),
             load_images: bool = True):
```

| 参数 | 类型 | 说明 | 默认值 |
|------|------|------|--------|
| `data_dir` | str/Path | 数据目录路径 | 必需 |
| `split` | str | 数据划分 ('train', 'val', 'test') | 'train' |
| `transform_type` | str | 图像变换类型 ('train', 'val') | 'train' |
| `target_size` | tuple | 目标图像尺寸 (H, W) | (224, 224) |
| `load_images` | bool | 是否加载图像 | True |

### 严格验证机制

#### 数据要求验证
```python
# 自动调用的验证流程
def __init__(self):
    # 1. 检查数据要求
    check_data_requirements()
    
    # 2. 设置配置
    self.setup_config()
    
    # 3. 设置图像变换
    self.setup_transforms()
    
    # 4. 加载数据集
    self.load_dataset()
    
    # 5. 验证数据集
    self.validate_dataset()
```

#### 文件验证
```python
# 必需文件检查
required_files = [
    f'dataset_items_{split}.json'  # 对应split的JSON文件
]

# 数据格式验证
required_fields = {
    'caption': str,    # 文本内容（必需）
    'label': int,      # 标签（必需）
    'image_path': str  # 图像路径（可选）
}
```

## 🖼️ 图像处理

### 图像变换配置
```python
# 训练时变换（数据增强）
if self.transform_type == 'train':
    self.image_transforms = transforms.Compose([
        transforms.Resize(self.target_size),
        transforms.RandomHorizontalFlip(p=0.3),  # 降低随机性
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])  # ImageNet标准
    ])

# 验证/测试时变换（无增强）
else:
    self.image_transforms = transforms.Compose([
        transforms.Resize(self.target_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
```

### 安全图像加载
```python
def load_image_safe(self, image_path: str) -> Dict[str, Any]:
    """
    安全加载图像，处理各种异常情况
    
    Returns:
        {
            'image': torch.Tensor,      # 图像张量 (3, H, W)
            'has_image': bool,          # 是否成功加载
            'image_path': str,          # 图像路径
            'image_size': tuple         # 原始尺寸 (可选)
        }
    """
```

## 📊 数据项格式

### 返回数据结构
```python
def __getitem__(self, idx: int) -> Dict[str, Any]:
    """
    返回单个数据样本
    
    Returns:
        {
            'item_id': str,           # 数据项ID
            'text': str,              # 主要文本字段
            'caption': str,           # 兼容性字段（同text）
            'label': int,             # 标签 (0, 1, 2)
            'language': str,          # 语言类型
            'text_length': int,       # 文本长度
            'token_count': int,       # 词数统计
            'image': torch.Tensor,    # 图像张量 (3, H, W)
            'has_image': bool,        # 是否有有效图像
            'image_path': str,        # 图像路径
            'image_size': tuple       # 图像尺寸（如果有）
        }
    """
```

### 标签映射
```python
# 从配置管理器获取的标签映射
self.label_mapping = {
    0: "Non-rumor",     # 非谣言
    1: "Rumor",         # 谣言  
    2: "Unverified"     # 未验证
}
```

## 🔧 实用方法

### 统计信息获取
```python
# 获取数据集统计信息
stats = dataset.get_statistics()
"""
返回:
{
    'total_samples': int,                    # 总样本数
    'label_distribution': Dict[str, int],    # 标签分布
    'has_image_count': int,                  # 有效图像数量
    'text_length_stats': {                   # 文本长度统计
        'min': int,
        'max': int, 
        'mean': float,
        'std': float
    }
}
"""

# 获取标签分布
label_dist = dataset.get_label_distribution()
# 返回: {'Non-rumor': 150, 'Rumor': 120, 'Unverified': 30}
```

### 样本查询
```python
# 根据ID获取样本
sample = dataset.get_sample_by_id('item_123')

# 打印样本信息（调试用）
dataset.print_sample_info(idx=0)
"""
输出示例:
🔍 样本 0 信息:
   ID: item_001
   文本: 这是一个测试文本 This is a test...
   标签: 1 (Rumor)
   文本长度: 45
   有图像: True
   图像路径: data/train/img/item_001.jpg
   图像张量形状: torch.Size([3, 224, 224])
"""
```

## 🎯 使用示例

### 基础使用
```python
from data_utils import MR2Dataset

# 创建训练集
train_dataset = MR2Dataset(
    data_dir='data',
    split='train',
    transform_type='train',
    load_images=True
)

print(f"训练集大小: {len(train_dataset)}")
print(f"标签分布: {train_dataset.get_label_distribution()}")

# 获取单个样本
sample = train_dataset[0]
print(f"样本ID: {sample['item_id']}")
print(f"文本: {sample['text'][:50]}...")
print(f"标签: {sample['label']} ({train_dataset.label_mapping[sample['label']]})")
print(f"图像形状: {sample['image'].shape}")
```

### 验证集创建
```python
# 创建验证集（无数据增强）
val_dataset = MR2Dataset(
    data_dir='data',
    split='val',
    transform_type='val',  # 无增强变换
    load_images=True
)

# 对比训练集和验证集的变换
print("训练集变换:", train_dataset.image_transforms)
print("验证集变换:", val_dataset.image_transforms)
```

### 错误处理示例
```python
try:
    # 尝试创建数据集
    dataset = MR2Dataset(data_dir='data', split='train')
    
    # 验证数据集基本信息
    if len(dataset) == 0:
        raise ValueError("数据集为空")
        
    # 测试样本访问
    sample = dataset[0]
    print("✅ 数据集创建成功")
    
except FileNotFoundError as e:
    print(f"❌ 数据文件不存在: {e}")
    print("解决方案:")
    print("1. 下载MR2数据集")
    print("2. 解压到项目根目录的data文件夹")
    print("3. 确保包含所有必需的JSON文件")
    
except ValueError as e:
    print(f"❌ 数据验证失败: {e}")
    print("解决方案:")
    print("1. 检查数据文件格式")
    print("2. 确保最小样本数要求")
    print("3. 验证标签和字段完整性")
    
except Exception as e:
    print(f"❌ 未知错误: {e}")
```

### 与DataLoader结合使用
```python
from torch.utils.data import DataLoader

# 创建数据集
dataset = MR2Dataset(data_dir='data', split='train')

# 创建DataLoader
dataloader = DataLoader(
    dataset,
    batch_size=32,
    shuffle=True,
    num_workers=4,
    collate_fn=None  # 使用默认collate_fn或自定义
)

# 迭代数据
for batch_idx, batch in enumerate(dataloader):
    print(f"批次 {batch_idx}:")
    print(f"  文本数量: {len(batch['text'])}")
    print(f"  标签形状: {batch['labels'].shape if 'labels' in batch else 'N/A'}")
    print(f"  图像形状: {batch['images'].shape if 'images' in batch else 'N/A'}")
    
    if batch_idx >= 2:  # 只显示前3个批次
        break
```

## ⚙️ 配置集成

### 配置管理器集成
```python
# 数据集自动使用配置管理器
from utils.config_manager import get_data_config, get_label_mapping

# 配置自动加载
self.label_mapping = get_label_mapping()
data_config = get_data_config()
self.dataset_config = data_config.get('dataset', {})
```

### 预处理配置
```python
# 从配置文件获取图像处理参数
processing_config = data_config.get('processing', {}).get('image', {})
self.normalize_mean = processing_config.get('normalize_mean', [0.485, 0.456, 0.406])
self.normalize_std = processing_config.get('normalize_std', [0.229, 0.224, 0.225])
```

## 🔍 调试和性能

### 调试功能
```python
# 打印详细的样本信息
dataset.print_sample_info(0)

# 获取统计信息进行调试
stats = dataset.get_statistics()
print(f"数据集统计: {stats}")

# 检查图像加载情况
sample = dataset[0]
if sample['has_image']:
    print(f"✅ 图像加载成功: {sample['image'].shape}")
else:
    print("❌ 图像加载失败")
```

### 性能优化
- **图像预加载**: 设置`load_images=False`可以跳过图像加载
- **变换优化**: 验证时使用`transform_type='val'`避免数据增强
- **内存管理**: 及时释放不需要的样本引用

## ⚠️ 重要注意事项

### 数据要求
- **真实数据集**: 必须使用完整的MR2数据集，不支持演示数据
- **文件完整性**: 所有JSON文件必须存在且格式正确
- **最小样本数**: 每个split需要至少10个有效样本

### 兼容性
- **向后兼容**: 提供`SimpleMR2Dataset`别名
- **字段兼容**: 同时提供`text`和`caption`字段
- **标签兼容**: 支持整数标签和字符串标签映射

### 错误恢复
- **图像错误**: 图像加载失败时自动创建零张量
- **格式错误**: 提供详细的错误信息和解决建议
- **路径问题**: 自动处理相对路径和绝对路径

---

**[⬅️ 数据工具概览](README.md) | [数据加载器 ➡️](data_loaders.md)**